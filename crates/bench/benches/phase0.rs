//! Phase 0 benches: tree build + FMM run at increasing N, plus roofline
//! labels for each kernel. Run: cargo bench -p fmm-bench --bench phase0.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use fmm::direct::direct;
use fmm::fmm::Fmm;
use fmm::models::plummer;

fn run_fmm(n: usize, p: usize, eps: f64, n_leaf: usize) {
    let parts = plummer(n, [0.0; 3], 42);
    let fmm = Fmm::new(&parts, p, eps, n_leaf);
    let r = fmm.run(&parts);
    black_box(r.accel);
}

fn bench_fmm(c: &mut Criterion) {
    for n in [10_000, 100_000] {
        c.bench_function(&format!("fmm/p5_eps1e-5/n{n}"), |b| {
            b.iter(|| run_fmm(n, 5, 1e-5, 32))
        });
    }
    c.bench_function("direct/n10k", |b| {
        b.iter(|| {
            let parts = plummer(10_000, [0.0; 3], 42);
            black_box(direct(&parts));
        })
    });
}

fn bench_tree(c: &mut Criterion) {
    for n in [10_000, 100_000] {
        c.bench_function(&format!("tree/n{n}"), |b| {
            b.iter(|| {
                let parts = plummer(n, [0.0; 3], 42);
                black_box(fmm::tree::Octree::build(&parts, 32));
            })
        });
    }
}

/// Roofline labels: FLOPs + arithmetic intensity per kernel, matching
/// docs/02-lowlevel.md §4. Compute/memory-bound verdict vs a ~10-20
/// FLOP/byte CPU machine balance.
fn roofline_report() {
    let bal = 10.0; // conservative CPU machine balance, flops/byte
    let mut out = String::from("\nRoofline kernel labels (Phase 0)\n");
    for p in [2usize, 5, 8] {
        let fl = fmm::roofline::m2l_flops(p);
        let n = (p + 1) * (p + 1);
        let bytes = ((n + n) * 16) as f64; // moments in + local out, complex f64
        let i = fmm::roofline::intensity(fl, bytes);
        let verdict = if i > bal { "compute-bound" } else { "memory-bound" };
        out.push_str(&format!("M2L p={p}: {fl:.0} flops, {i:.1} flops/byte -> {verdict}\n"));
    }
    let fl = fmm::roofline::p2p_flops_per_pair();
    let bytes = 32.0; // 2 positions (6 f64) per pair
    let i = fmm::roofline::intensity(fl, bytes);
    let verdict = if i > bal { "compute-bound" } else { "memory-bound" };
    out.push_str(&format!("P2P/pair: {fl:.0} flops, {i:.2} flops/byte -> {verdict}\n"));
    println!("{out}");
}

fn bench_roofline(_c: &mut Criterion) {
    roofline_report();
}

criterion_group!(benches, bench_fmm, bench_tree, bench_roofline);
criterion_main!(benches);
