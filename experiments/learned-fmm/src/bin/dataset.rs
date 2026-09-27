use learned_fmm::{generate, normalize_to_unit_box, random_rotate, split_near_far, ConfigTag, SnapshotData};
use fmm::tree::Octree;

fn arg_val(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if has_flag(&args, "--help") {
        println!("usage: dataset --out DIR [--snapshots K] [--n-min A] [--n-max B] [--config all|plummer|cube|clumps|mixed] [--seed S] [--n-leaf L] [--no-rotate]");
        return;
    }
    let out = arg_val(&args, "--out", "/tmp/opencode/learned-fmm");
    let snapshots: usize = arg_val(&args, "--snapshots", "16").parse().unwrap();
    let n_min: usize = arg_val(&args, "--n-min", "1024").parse().unwrap();
    let n_max: usize = arg_val(&args, "--n-max", "4096").parse().unwrap();
    let config = arg_val(&args, "--config", "all");
    let seed_base: u64 = arg_val(&args, "--seed", "1000").parse().unwrap();
    let n_leaf: usize = arg_val(&args, "--n-leaf", "16").parse().unwrap();
    let p: usize = arg_val(&args, "--p", "8").parse().unwrap();
    let eps: f64 = arg_val(&args, "--eps", "1e-5").parse().unwrap();
    let rotate = !has_flag(&args, "--no-rotate");
    assert!(n_min > 0 && n_max >= n_min && snapshots > 0);

    let outdir = std::path::Path::new(&out);
    std::fs::create_dir_all(outdir).unwrap();
    let mut manifest = String::from("id,config,n,seed,p,eps,n_leaf,rotate,rms_total,rms_near,rms_far,far_frac,min_r,file\n");
    let mut sum_far = 0.0;

    for s in 0..snapshots {
        let tag = if config == "all" {
            ConfigTag::all()[s % 4]
        } else {
            ConfigTag::parse(&config).expect("bad --config")
        };
        let n = if snapshots == 1 {
            n_min
        } else {
            n_min + (n_max - n_min) * s / (snapshots - 1)
        };
        let seed = seed_base + s as u64;
        let mut parts = generate(tag, n, seed);
        normalize_to_unit_box(&mut parts);
        if rotate {
            random_rotate(&mut parts, seed ^ 0x51ED270B);
        }
        let tree = Octree::build(&parts, n_leaf);
        let (near, far, stats) = split_near_far(&parts, &tree);
        assert!(stats.min_r > 0.0, "snapshot {s}: colliding particles");
        let snap = SnapshotData::from_parts(tag, &parts, &near, &far);
        let fname = format!("snap_{s:06}_{}_{n}.bin", tag.name());
        snap.write_to(&outdir.join(&fname)).unwrap();
        sum_far += stats.far_frac;
        manifest.push_str(&format!(
            "{s},{},{n},{seed},{p},{eps},{n_leaf},{},{:.6},{:.6},{:.6},{:.4},{:.6},{fname}\n",
            tag.name(),
            if rotate { 1 } else { 0 },
            stats.rms_total,
            stats.rms_near,
            stats.rms_far,
            stats.far_frac,
            stats.min_r
        ));
        println!(
            "snapshot {s}: {} n={n} cells={} rms_total={:.4} far_frac={:.3} min_r={:.4}",
            tag.name(),
            tree.cells.len(),
            stats.rms_total,
            stats.far_frac,
            stats.min_r
        );
    }
    std::fs::write(outdir.join("manifest.csv"), &manifest).unwrap();
    println!(
        "done: {snapshots} snapshots -> {out} (mean far_frac={:.3})",
        sum_far / snapshots as f64
    );
}
