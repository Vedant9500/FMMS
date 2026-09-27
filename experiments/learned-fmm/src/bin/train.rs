use burn::{
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Tensor, TensorData},
};
use learned_fmm::{
    build_graph, generate, normalize_to_unit_box, random_rotate, split_near_far, ConfigTag,
    InteractionGraph, SnapshotData,
};
use learned_fmm::model::{TreeConfig, TreeModel};
use fmm::models::Particle;
use fmm::tree::Octree;

type BT = Autodiff<NdArray>;

struct Sample {
    parts: Vec<Particle>,
    target: Vec<f32>,
    graph: InteractionGraph,
}

fn arg_val(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn col(hdr: &[String], name: &str) -> Option<usize> {
    hdr.iter().position(|h| h == name)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = arg_val(&args, "--data", "/tmp/opencode/learned-fmm");
    let epochs: usize = arg_val(&args, "--epochs", "5").parse().unwrap();
    let lr: f64 = arg_val(&args, "--lr", "1e-3").parse().unwrap();
    let emb: usize = arg_val(&args, "--emb", "32").parse().unwrap();
    let hidden: usize = arg_val(&args, "--hidden", "64").parse().unwrap();
    let max_rows: usize = arg_val(&args, "--max-rows", "0").parse().unwrap();
    let probe: usize = arg_val(&args, "--probe", "0").parse().unwrap();
    let test_frac: f64 = arg_val(&args, "--test-frac", "0.1").parse().unwrap();
    let device = Default::default();
    let text = std::fs::read_to_string(format!("{data}/manifest.csv")).unwrap();
    let mut lines = text.lines();
    let hdr: Vec<String> = lines.next().unwrap().split(',').map(str::to_string).collect();
    let c = |name: &str, default: usize| col(&hdr, name).unwrap_or(default);
    let ci = |name: &str| col(&hdr, name);
    let rows: Vec<String> = lines.map(str::to_string).collect();
    let nrows = if max_rows > 0 {
        rows.len().min(max_rows)
    } else {
        rows.len()
    };
    let mut samples = Vec::with_capacity(nrows);
    for (ri, line) in rows.iter().take(nrows).enumerate() {
        let f: Vec<String> = line.split(',').map(str::to_string).collect();
        let tag = ConfigTag::parse(&f[c("config", 1)]).unwrap();
        let n: usize = f[c("n", 2)].parse().unwrap();
        let seed: u64 = f[c("seed", 3)].parse().unwrap();
        let p: usize = ci("p").map(|i| f[i].parse().unwrap()).unwrap_or(8);
        let eps: f64 = ci("eps").map(|i| f[i].parse().unwrap()).unwrap_or(1e-5);
        let n_leaf: usize = ci("n_leaf").map(|i| f[i].parse().unwrap()).unwrap_or(16);
        let rotate = ci("rotate").map(|i| f[i] == "1").unwrap_or(true);
        let mut parts = generate(tag, n, seed);
        normalize_to_unit_box(&mut parts);
        if rotate {
            random_rotate(&mut parts, seed ^ 0x51ED270B);
        }
        let tree = Octree::build(&parts, n_leaf);
        let (_, far, stats) = split_near_far(&parts, &tree);
        let scale = stats.rms_far.max(1e-12);
        let mut target = Vec::with_capacity(n * 3);
        for i in 0..n {
            target.extend_from_slice(&[
                (far[i][0] / scale) as f32,
                (far[i][1] / scale) as f32,
                (far[i][2] / scale) as f32,
            ]);
        }
        if ri == 0 {
            let fi = ci("file").unwrap_or(f.len() - 1);
            let file = &f[fi];
            match SnapshotData::read_from(&std::path::Path::new(&data).join(file)) {
                Ok(snap) => {
                    let mut num = 0.0f64;
                    let mut den = 0.0f64;
                    for i in 0..n {
                        for d in 0..3 {
                            let a = snap.a_far[i][d] as f64;
                            let b = far[i][d];
                            num += (a - b) * (a - b);
                            den += b * b;
                        }
                    }
                    let rel = (num / den).sqrt();
                    assert!(rel < 1e-5, "dump mismatch rel={rel}");
                    println!("row-0 dump check: rel={rel:.2e} ok");
                }
                Err(e) => println!("row-0 dump check skipped: {e}"),
            }
        }
        let graph = build_graph(&parts, &tree, p, eps);
        samples.push(Sample {
            parts,
            target,
            graph,
        });
    }
    println!("loaded {nrows} samples");
    let cfg = TreeConfig { emb, hidden };
    let mut model = TreeModel::<BT>::new(&cfg, &device);
    if probe > 0 {
        for (pi, s) in samples.iter().take(probe).enumerate() {
            let t0 = std::time::Instant::now();
            let pred = model.forward(emb, &s.parts, &s.graph, &device);
            let dt = t0.elapsed().as_secs_f32();
            let v: Vec<f32> = pred.into_data().into_vec().unwrap();
            let mut rms = 0.0f64;
            let mut max = 0.0f32;
            for &x in &v {
                rms += (x as f64) * (x as f64);
                max = max.max(x.abs());
            }
            rms = (rms / v.len() as f64).sqrt();
            let edges: usize = s.graph.incoming.iter().map(|e| e.len()).sum();
            println!(
                "probe {pi}: n={} cells={} edges={} pred_rms={rms:.4} pred_max={max:.4} {dt:.3}s",
                s.parts.len(),
                s.graph.nodes.len(),
                edges
            );
        }
        return;
    }
    let mut optim = AdamConfig::new().init();
    let mut rng = fmm::models::Rng::new(999);
    let ntest = ((nrows as f64 * test_frac) as usize).min(nrows.saturating_sub(1));
    let ntrain = nrows - ntest;
    println!("train={ntrain} test={ntest}");
    for e in 0..epochs {
        let t0 = std::time::Instant::now();
        let mut order: Vec<usize> = (0..ntrain).collect();
        for i in (1..ntrain).rev() {
            let j = (rng.unit() * (i + 1) as f64) as usize;
            order.swap(i, j);
        }
        let mut sum = 0.0;
        let mut rsum = 0.0;
        for &si in &order {
            let s = &samples[si];
            let n = s.parts.len();
            let pred = model.forward(emb, &s.parts, &s.graph, &device);
            let tgt: Tensor<BT, 2> =
                Tensor::from_data(TensorData::new(s.target.clone(), [n, 3]), &device);
            let diff = pred - tgt.clone();
            let rel = diff.clone().powf_scalar(2.0).mean()
                / tgt.powf_scalar(2.0).mean().clamp_min(1e-12);
            let robust = diff.cosh().log().mean();
            let loss = rel.clone() + robust.clone();
            sum += loss.clone().into_scalar();
            rsum += rel.into_scalar();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);
        }
        let mut tsum = 0.0;
        let mut trsum = 0.0;
        for si in ntrain..nrows {
            let s = &samples[si];
            let n = s.parts.len();
            let pred = model.forward(emb, &s.parts, &s.graph, &device);
            let tgt: Tensor<BT, 2> =
                Tensor::from_data(TensorData::new(s.target.clone(), [n, 3]), &device);
            let diff = pred - tgt.clone();
            let rel = diff.clone().powf_scalar(2.0).mean()
                / tgt.powf_scalar(2.0).mean().clamp_min(1e-12);
            let robust = diff.cosh().log().mean();
            let tot = rel.clone() + robust;
            tsum += tot.into_scalar();
            trsum += rel.into_scalar();
        }
        println!(
            "epoch {e}: train={:.6} (rel={:.6}) test={:.6} (rel={:.6}) ({:.1}s)",
            sum / ntrain as f32,
            rsum / ntrain as f32,
            tsum / ntest.max(1) as f32,
            trsum / ntest.max(1) as f32,
            t0.elapsed().as_secs_f32()
        );
    }
}
