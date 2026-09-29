use burn::{
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsAccumulator, GradientsParams, Optimizer},
    tensor::{Tensor, TensorData},
};
use learned_fmm::{
    build_graph_x, generate, normalize_to_unit_box, random_rotate, split_near_far, ConfigTag,
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
    let rounds: usize = arg_val(&args, "--rounds", "1").parse().unwrap();
    let pool: u8 = arg_val(&args, "--pool", "0").parse().unwrap();
    let h_trunk: usize = arg_val(&args, "--h-trunk", "0").parse().unwrap();
    let h_head: usize = arg_val(&args, "--h-head", "0").parse().unwrap();
    let mono: u8 = arg_val(&args, "--mono", "0").parse().unwrap();
    let geom: u8 = arg_val(&args, "--geom", "0").parse().unwrap();
    let equiv: u8 = arg_val(&args, "--equiv", "0").parse().unwrap();
    let up_mode: u8 = arg_val(&args, "--up-mode", "0").parse().unwrap();
    let xlevel: u8 = arg_val(&args, "--xlevel", "0").parse().unwrap();
    let batch: usize = arg_val(&args, "--batch", "16").parse().unwrap();
    // threads>1 deadlocks: burn Autodiff hangs on concurrent backward.
    let threads: usize = arg_val(&args, "--threads", "1").parse().unwrap();
    let probe: usize = arg_val(&args, "--probe", "0").parse().unwrap();
    let probe_rot: usize = arg_val(&args, "--probe-rot", "0").parse().unwrap();
    let eval_every: usize = arg_val(&args, "--eval-every", "1").parse().unwrap();
    let eval_every = eval_every.max(1);
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
    let mut row0: Option<(String, usize, u64, usize, f64, usize)> = None;
    for (ri, line) in rows.iter().take(nrows).enumerate() {
        let f: Vec<String> = line.split(',').map(str::to_string).collect();
        let tag = ConfigTag::parse(&f[c("config", 1)]).unwrap();
        let n: usize = f[c("n", 2)].parse().unwrap();
        let seed: u64 = f[c("seed", 3)].parse().unwrap();
        let p: usize = ci("p").map(|i| f[i].parse().unwrap()).unwrap_or(8);
        let eps: f64 = ci("eps").map(|i| f[i].parse().unwrap()).unwrap_or(1e-5);
        let n_leaf: usize = ci("n_leaf").map(|i| f[i].parse().unwrap()).unwrap_or(16);
        let rotate = ci("rotate").map(|i| f[i] == "1").unwrap_or(true);
        if ri == 0 {
            row0 = Some((f[c("config", 1)].clone(), n, seed, p, eps, n_leaf));
        }
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
        let graph = build_graph_x(&parts, &tree, p, eps, xlevel);
        samples.push(Sample {
            parts,
            target,
            graph,
        });
    }
    println!("loaded {nrows} samples");
    let cfg = TreeConfig { emb, hidden, rounds, pool, h_trunk, h_head, mono, geom, equiv, up_mode };
    let mut model = TreeModel::<BT>::new(&cfg, &device);
    if probe > 0 {
        for (pi, s) in samples.iter().take(probe).enumerate() {
            let t0 = std::time::Instant::now();
            let pred = model.forward(&cfg, &s.parts, &s.graph, &device);
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
    if probe_rot > 0 {
        let (cfgstr, rn, rseed, rp, reps, rn_leaf) = row0.clone().unwrap();
        let rtag = ConfigTag::parse(&cfgstr).unwrap();
        let mut rels = Vec::new();
        for k in 0..probe_rot.max(8) {
            let mut parts = generate(rtag, rn, rseed);
            normalize_to_unit_box(&mut parts);
            random_rotate(&mut parts, rseed ^ (0x51ED270B + k as u64 * 0x9E3779B9));
            let tree = Octree::build(&parts, rn_leaf);
            let (_, far, stats) = split_near_far(&parts, &tree);
            let scale = stats.rms_far.max(1e-12);
            let mut target = Vec::with_capacity(rn * 3);
            for i in 0..rn {
                target.extend_from_slice(&[
                    (far[i][0] / scale) as f32,
                    (far[i][1] / scale) as f32,
                    (far[i][2] / scale) as f32,
                ]);
            }
            let graph = build_graph_x(&parts, &tree, rp, reps, xlevel);
            let pred = model.forward(&cfg, &parts, &graph, &device);
            let tgt: Tensor<BT, 2> =
                Tensor::from_data(TensorData::new(target, [rn, 3]), &device);
            let diff = pred - tgt.clone();
            let rel = diff.clone().powf_scalar(2.0).mean()
                / tgt.powf_scalar(2.0).mean().clamp_min(1e-12);
            let v: f32 = rel.into_scalar();
            rels.push(v);
            println!("rot-probe {k}: rel={v:.6}");
        }
        let m = rels.iter().sum::<f32>() / rels.len() as f32;
        let sd = (rels.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / rels.len() as f32).sqrt();
        println!("rot-probe: mean={m:.6} std={sd:.6} (low std = rotation-robust)");
        return;
    }
    let mut optim = AdamConfig::new().init();
    let mut rng = fmm::models::Rng::new(999);
    let ntest = ((nrows as f64 * test_frac) as usize).min(nrows.saturating_sub(1));
    let ntrain = nrows - ntest;
    println!("train={ntrain} test={ntest}");
    let mut last_tsum = 0.0f32;
    let mut last_trsum = 0.0f32;
    for e in 0..epochs {        let t0 = std::time::Instant::now();
        let mut order: Vec<usize> = (0..ntrain).collect();
        for i in (1..ntrain).rev() {
            let j = (rng.unit() * (i + 1) as f64) as usize;
            order.swap(i, j);
        }
        let mut sum = 0.0;
        let mut rsum = 0.0;
        for chunk in order.chunks(batch.max(1)) {
            let scale = 1.0 / chunk.len() as f32;
            let t = threads.min(chunk.len()).max(1);
            let mut master = GradientsAccumulator::<TreeModel<BT>>::new();
            let (sr, mr, dr) = (&samples, &model, &device);
            std::thread::scope(|s| {
                let mut hs = Vec::new();
                for sub in chunk.chunks(chunk.len().div_ceil(t)) {
                    let owned: Vec<usize> = sub.to_vec();
                    hs.push(s.spawn(move || {
                        let mut acc = GradientsAccumulator::<TreeModel<BT>>::new();
                        let mut ls = 0.0f32;
                        let mut rs = 0.0f32;
                        for si in owned {
                            let smp = &sr[si];
                            let n = smp.parts.len();
                            let pred = mr.forward(&cfg, &smp.parts, &smp.graph, dr);
                            let tgt: Tensor<BT, 2> = Tensor::from_data(
                                TensorData::new(smp.target.clone(), [n, 3]),
                                dr,
                            );
                            let diff = pred - tgt.clone();
                            let rel = diff.clone().powf_scalar(2.0).mean()
                                / tgt.powf_scalar(2.0).mean().clamp_min(1e-12);
                            let robust = diff.cosh().log().mean();
                            ls += (rel.clone() + robust.clone()).into_scalar();
                            rs += rel.clone().into_scalar();
                            let loss = (rel + robust).mul_scalar(scale);
                            let grads = loss.backward();
                            acc.accumulate(mr, GradientsParams::from_grads(grads, mr));
                        }
                        (acc.grads(), ls, rs)
                    }));
                }
                for h in hs {
                    let (g, ls, rs) = h.join().unwrap();
                    master.accumulate(&model, g);
                    sum += ls;
                    rsum += rs;
                }
            });
            model = optim.step(lr, model, master.grads());
        }
        let mut tsum = last_tsum;
        let mut trsum = last_trsum;
        if e % eval_every == 0 || e + 1 == epochs {
            tsum = 0.0;
            trsum = 0.0;
            for si in ntrain..nrows {
                let s = &samples[si];
                let n = s.parts.len();
                let pred = model.forward(&cfg, &s.parts, &s.graph, &device);
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
            last_tsum = tsum;
            last_trsum = trsum;
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
