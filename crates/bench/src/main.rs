//! Phase-1 O(N) scaling sweep: FMM vs direct O(N^2), accuracy, and
//! momentum conservation across N = 10^3 .. 10^6.
//!
//! Usage: fmm-bench [max_n] [p] [eps]

use fmm::bodies::Bodies;
use fmm::direct::{direct, scaled_error};
use fmm::fmm::Fmm;
use fmm::models::plummer;
use std::time::Instant;

fn time_fmm(parts: &[fmm::models::Particle], p: usize, eps: f64, nthreads: usize) -> f64 {
    // Warmup.
    let f = Fmm::new(parts, p, eps, 16);
    f.run_threaded(parts, nthreads);
    // Best of 2 timed runs.
    let mut best = f64::INFINITY;
    for _ in 0..2 {
        let t0 = Instant::now();
        f.run_threaded(parts, nthreads);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn time_direct(parts: &[fmm::models::Particle]) -> f64 {
    let t0 = Instant::now();
    direct(parts);
    t0.elapsed().as_secs_f64()
}

/// Sample accuracy: exact force + f = Σ m/r² for `samp` random particles,
/// O(samp·N). Returns (max_s, rms_s).
fn sampled_error(
    parts: &[fmm::models::Particle],
    accel: &[[f64; 3]],
    samp: usize,
    seed: u64,
) -> (f64, f64) {
    let n = parts.len();
    let mut rng = seed;
    let mut next = move || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as usize % n
    };
    let mut worst = 0.0f64;
    let mut sumsq = 0.0f64;
    for _ in 0..samp {
        let i = next();
        let (mut ax, mut ay, mut az, mut f) = (0.0f64, 0.0, 0.0, 0.0);
        let pi = parts[i];
        for j in 0..n {
            if j == i {
                continue;
            }
            let pj = parts[j];
            let dx = pj.pos[0] - pi.pos[0];
            let dy = pj.pos[1] - pi.pos[1];
            let dz = pj.pos[2] - pi.pos[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv = 1.0 / r2.sqrt();
            let inv2 = inv * inv;
            let inv3 = inv2 * inv;
            ax += pj.mass * inv3 * dx;
            ay += pj.mass * inv3 * dy;
            az += pj.mass * inv3 * dz;
            f += pj.mass * inv2;
        }
        let da = (accel[i][0] - ax).powi(2) + (accel[i][1] - ay).powi(2) + (accel[i][2] - az).powi(2);
        let s = da.sqrt() / f;
        worst = worst.max(s);
        sumsq += s * s;
    }
    (worst, (sumsq / samp as f64).sqrt())
}

/// P2P throughput microbench: run p2p_self over all leaf cells (N bodies,
/// n_leaf=16), threaded across `nthreads` via disjoint slot ranges.
/// Reports GFLOPS at 16 flops/pair (roofline.rs accounting).
fn p2p_bench(n: usize, nthreads: usize, n_leaf: usize) {
    use fmm::bodies::Bodies;

    let parts = plummer(n, [0.0; 3], 42);
    let tree = fmm::tree::Octree::build(&parts, n_leaf);
    let mut bodies = Bodies::from_parts(&parts, &tree.order);

    let mut leaves: Vec<(usize, usize)> = Vec::new();
    for c in 0..tree.cells.len() {
        if tree.is_leaf(c) {
            leaves.push((tree.cells[c].first, tree.cells[c].count));
        }
    }
    leaves.sort_by_key(|&(f, _)| f);
    let pairs: usize = leaves.iter().map(|&(_, cnt)| cnt * (cnt - 1) / 2).sum();
    let flops = pairs as f64 * fmm::roofline::p2p_flops_per_pair();
    let avg = leaves.iter().map(|&(_, c)| c).sum::<usize>() as f64 / leaves.len() as f64;
    println!("p2p bench: N={n} n_leaf={n_leaf} leaves={} avg={avg:.2}/leaf pairs={pairs} flops/pass={:.2e}", leaves.len(), flops);

    // Partition leaves into `nthreads` contiguous slot ranges.
    let mut bounds = vec![0usize; nthreads + 1];
    let target = n / nthreads;
    let mut t = 1usize;
    for (idx, &(first, count)) in leaves.iter().enumerate() {
        if t < nthreads && idx > 0 && first + count >= target * t {
            bounds[t] = idx;
            t += 1;
        }
    }
    bounds[nthreads] = leaves.len();
    let mut leaf_ranges = Vec::new();
    for t in 0..nthreads {
        let slice = &leaves[bounds[t]..bounds[t + 1]];
        let lo = slice.first().map(|&(f, _)| f).unwrap_or(0);
        let hi = slice.last().map(|&(f, c)| f + c).unwrap_or(lo);
        leaf_ranges.push((slice.to_vec(), lo, hi));
    }

    // Synthetic large-block measurement: one kernel call on `big` bodies,
    // isolates pure kernel throughput from per-cell call overhead.
    let big = 2048usize;
    let big_pairs = big * (big - 1) / 2;
    let big_flops = big_pairs as f64 * fmm::roofline::p2p_flops_per_pair();
    let mut best_b = f64::INFINITY;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            fmm::p2p::p2p_self(&mut bodies, 0, big);
        }
        best_b = best_b.min(t0.elapsed().as_secs_f64() / 20.0);
    }
    println!(
        "  synthetic block=2048: {:.4}s/call = {:>7.1} GFLOPS",
        best_b,
        big_flops / best_b / 1e9
    );

    // Threaded synthetic: each thread runs its own 2048-body block, disjoint views.
    let mut best_bt = f64::INFINITY;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        std::thread::scope(|s| {
            let raw: *mut Bodies = &mut bodies;
            let nthreads = nthreads;
            for t in 0..nthreads {
                let mut v = unsafe { body_view(&mut *raw, (t * big) % (bodies.x.len() - big), (t * big) % (bodies.x.len() - big) + big) };
                s.spawn(move || {
                    for _ in 0..20 {
                        fmm::p2p::p2p_self(&mut v, 0, big);
                    }
                    std::mem::forget(v);
                });
            }
        });
        best_bt = best_bt.min(t0.elapsed().as_secs_f64() / 20.0);
    }
    println!(
        "  synthetic threaded x{}: {:.4}s/call = {:>7.1} GFLOPS",
        nthreads,
        best_bt,
        big_flops * nthreads as f64 / best_bt / 1e9
    );

    // Single-thread baseline.
    let passes = 50;
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        for _ in 0..passes {
            for &(f, cnt) in &leaves {
                fmm::p2p::p2p_self(&mut bodies, f, cnt);
            }
        }
        best = best.min(t0.elapsed().as_secs_f64() / passes as f64);
    }
    println!("  single-thread: {:.4}s/pass = {:>7.1} GFLOPS", best, flops / best / 1e9);

    // Threaded across disjoint ranges (views share the same backing arrays).
    let mut best_t = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        std::thread::scope(|s| {
            let raw: *mut Bodies = &mut bodies;
            for (ls, lo, hi) in &leaf_ranges {
                let ls = ls.clone();
                let mut v = unsafe { body_view(&mut *raw, *lo, *hi) };
                s.spawn(move || {
                    for _ in 0..passes {
                        for &(f, cnt) in &ls {
                            fmm::p2p::p2p_self(&mut v, f - *lo, cnt);
                        }
                    }
                    std::mem::forget(v);
                });
            }
        });
        best_t = best_t.min(t0.elapsed().as_secs_f64() / passes as f64);
    }
    println!(
        "  threaded x{}: {:.4}s/pass = {:>7.1} GFLOPS ({:.2}x)",
        nthreads,
        best_t,
        flops / best_t / 1e9,
        best / best_t
    );
}

/// Mutable non-owning view over a contiguous slot range of `b`.
/// Caller must `std::mem::forget` the result (it aliases `b`).
unsafe fn body_view(b: &mut Bodies, lo: usize, hi: usize) -> Bodies {
    fn seg(v: &mut Vec<f64>, lo: usize, hi: usize) -> Vec<f64> {
        unsafe { Vec::from_raw_parts(v.as_mut_ptr().add(lo), hi - lo, hi - lo) }
    }
    Bodies {
        x: seg(&mut b.x, lo, hi),
        y: seg(&mut b.y, lo, hi),
        z: seg(&mut b.z, lo, hi),
        m: seg(&mut b.m, lo, hi),
        ax: seg(&mut b.ax, lo, hi),
        ay: seg(&mut b.ay, lo, hi),
        az: seg(&mut b.az, lo, hi),
        pot: seg(&mut b.pot, lo, hi),
    }
}

/// TEMP Unit-2: M2L kernel cost split. Cache-resident pairs (i, j from a
/// small pool), so we measure the kernel + table build, not DRAM.
fn m2l_bench(n: usize, p: usize) {
    use fmm::harmonic::{dim, C};
    let parts = plummer(n, [0.0; 3], 42);
    let tree = fmm::tree::Octree::build(&parts, 16);
    let d = dim(p);
    let ncells = tree.cells.len();
    let mut moments = vec![C::ZERO; ncells * d];
    let bodies = fmm::bodies::Bodies::from_parts(&parts, &tree.order);
    fmm::moments::compute_moments(&bodies, &tree, p, &mut moments);
    // Collect well-separated leaf pairs via the MAC (same accept as the walk).
    let pows = fmm::moments::multipole_power(&moments, ncells, p);
    let binom = fmm::moments::binomials(p);
    let (_, fmin) = fmm::mac::f_prepass(&bodies, &tree);
    let mut pairs = Vec::new();
    'outer: for a in 0..ncells {
        if !tree.is_leaf(a) {
            continue;
        }
        for b in 0..ncells {
            if b <= a || !tree.is_leaf(b) {
                continue;
            }
            if fmm::mac::accept(a, b, &tree.cells, &pows, &binom, p, 1e-5, &fmin)
                && fmm::mac::accept(b, a, &tree.cells, &pows, &binom, p, 1e-5, &fmin)
            {
                pairs.push((a, b));
                if pairs.len() >= 4096 {
                    break 'outer;
                }
            }
        }
    }
    println!("m2l bench: N={n} p={p} pairs={} cells={}", pairs.len(), ncells);

    // Full kernel (table build + conj + parity copy + convolution).
    let mut locals = vec![C::ZERO; ncells * d];
    let mut scratch = fmm::translate::M2lScratch::new(p);
    let reps = 20usize;
    let mut best_full = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            for &(a, b) in &pairs {
                fmm::translate::m2l_mutual_scratch(&tree, a, b, &moments, &mut locals, p, &mut scratch);
            }
        }
        best_full = best_full.min(t0.elapsed().as_secs_f64() / (reps * pairs.len()) as f64);
    }

    // Theta-table build only.
    let mut th = vec![C::ZERO; d];
    let mut best_th = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        for _ in 0..(reps * pairs.len()) {
            let ca = tree.cells[pairs[0].0];
            let cb = tree.cells[pairs[0].1];
            let r = [cb.cen[0] - ca.cen[0], cb.cen[1] - ca.cen[1], cb.cen[2] - ca.cen[2]];
            fmm::harmonic::theta_table_into(r, p, &mut th);
        }
        best_th = best_th.min(t0.elapsed().as_secs_f64() / (reps * pairs.len()) as f64);
    }

    // Convolution only (reuse one prebuilt scratch state by re-running the
    // same pair; subtract nothing — report raw). Approximation: full - theta.
    println!(
        "  full kernel: {:.0}ns/pair   theta build: {:.0}ns/pair   approx conv+setup: {:.0}ns/pair",
        best_full * 1e9,
        best_th * 1e9,
        (best_full - best_th) * 1e9
    );
}

/// TEMP Unit-3: rotation-vs-naive M2L across p (crossover re-measure).
/// Uses many accepted leaf pairs (like the original 4000-pair microbench)
/// so the threshold reflects real pair geometry, not one pair.
fn rot_sweep(n: usize) {
    use fmm::harmonic::{dim, C};
    let parts = plummer(n, [0.0; 3], 42);
    let tree = fmm::tree::Octree::build(&parts, 16);
    let bodies = fmm::bodies::Bodies::from_parts(&parts, &tree.order);
    println!("rot sweep: N={n} (multi-pair microbench, cache-resident)");
    println!(
        "{:>4} {:>10} {:>10} {:>10} {:>8}",
        "p", "naive_ns", "rot_ns", "ratio", "err_max"
    );
    for p in (16..=34).step_by(2) {
        let d = dim(p);
        let ncells = tree.cells.len();
        let mut moments = vec![C::ZERO; ncells * d];
        fmm::moments::compute_moments(&bodies, &tree, p, &mut moments);
        // Accepted leaf pairs via the MAC.
        let pows = fmm::moments::multipole_power(&moments, ncells, p);
        let binom = fmm::moments::binomials(p);
        let (_, fmin) = fmm::mac::f_prepass(&bodies, &tree);
        let mut pairs = Vec::new();
        'outer: for a in 0..ncells {
            if !tree.is_leaf(a) {
                continue;
            }
            for b in a + 1..ncells {
                if !tree.is_leaf(b) {
                    continue;
                }
                if fmm::mac::accept(a, b, &tree.cells, &pows, &binom, p, 1e-5, &fmin)
                    && fmm::mac::accept(b, a, &tree.cells, &pows, &binom, p, 1e-5, &fmin)
                {
                    pairs.push((a, b));
                    if pairs.len() >= 4096 {
                        break 'outer;
                    }
                }
            }
        }
        if pairs.is_empty() {
            continue;
        }
        let reps = if p <= 24 { 8 } else { 12 };
        // Naive.
        let mut loc = vec![C::ZERO; ncells * d];
        let mut scratch = fmm::translate::M2lScratch::new(p);
        let mut best_n = f64::INFINITY;
        for _ in 0..2 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for &(a, b) in &pairs {
                    fmm::translate::m2l_mutual_scratch(&tree, a, b, &moments, &mut loc, p, &mut scratch);
                }
            }
            best_n = best_n.min(t0.elapsed().as_secs_f64() / (reps * pairs.len()) as f64);
        }
        // Rotation.
        let mut ctx = fmm::rotation::RotCtx::new(p);
        let mut best_r = f64::INFINITY;
        for _ in 0..2 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for &(a, b) in &pairs {
                    ctx.m2l_pair(&tree, a, b, &moments, &mut loc);
                }
            }
            best_r = best_r.min(t0.elapsed().as_secs_f64() / (reps * pairs.len()) as f64);
        }
        // Error vs naive on the first pair, both kernels on FRESH buffers.
        let (a, b) = pairs[0];
        let mut l1 = vec![C::ZERO; ncells * d];
        let mut l2 = vec![C::ZERO; ncells * d];
        fmm::translate::m2l_mutual_scratch(&tree, a, b, &moments, &mut l1, p, &mut scratch);
        ctx.m2l_pair(&tree, a, b, &moments, &mut l2);
        let mut err = 0.0f64;
        for c in [a, b] {
            for i in 0..d {
                let nv = l1[c * d + i];
                let r = l2[c * d + i];
                let mag = nv.re.abs() + nv.im.abs() + 1e-30;
                err = err.max(((r.re - nv.re).abs() + (r.im - nv.im).abs()) / mag);
            }
        }
        println!(
            "{:>4} {:>10.0} {:>10.0} {:>10.3} {:>8.1e}  ({} pairs)",
            p,
            best_n * 1e9,
            best_r * 1e9,
            best_r / best_n,
            err,
            pairs.len()
        );
    }
}

fn net_force(parts: &[fmm::models::Particle], accel: &[[f64; 3]]) -> f64 {
    let mut fsum = [0.0; 3];
    let mut ftot = 0.0;
    for i in 0..parts.len() {
        let m = parts[i].mass;
        for d in 0..3 {
            fsum[d] += m * accel[i][d];
        }
        ftot += m * (accel[i][0].powi(2) + accel[i][1].powi(2) + accel[i][2].powi(2)).sqrt();
    }
    let norm = (fsum[0].powi(2) + fsum[1].powi(2) + fsum[2].powi(2)).sqrt();
    norm / ftot
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let p: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let eps: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e-5);
    let avx = std::is_x86_feature_detected!("avx512f");
    let default_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    // P2P throughput mode: fmm-bench p2p [N] [nthreads]
    if args.get(4).map(|s| s.as_str()) == Some("p2p") {
        let n: usize = args.get(5).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1_000_000);
        let nt: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(8);
        let nleaf: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(16);
        p2p_bench(n, nt, nleaf);
        return;
    }

    // TEMP Unit-2 mode: fmm-bench m2l [N] [p] — split kernel cost.
    if args.get(4).map(|s| s.as_str()) == Some("m2l") {
        let n: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100_000);
        let p: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(8);
        m2l_bench(n, p);
        return;
    }

    // TEMP Unit-3 mode: fmm-bench rot <N> — rotation vs naive across p.
    if args.get(4).map(|s| s.as_str()) == Some("rot") {
        let n: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100_000);
        rot_sweep(n);
        return;
    }

    // Profile mode: time each phase for a single N.
    // fmm-bench [max_n] [p] [eps] <N> [nthreads]  (N >= 1000)
    // A 5th argument < 1000 is instead the sweep's nthreads.
    if let Some(np) = args.get(4) {
        if let Ok(n) = np.parse::<usize>() {
            if n < 1000 {
                sweep(max_n, p, eps, n, avx);
                return;
            }
            profile(n, p, eps, args.get(5).and_then(|s| s.parse().ok()).unwrap_or(default_threads));
            return;
        }
    }
    sweep(max_n, p, eps, default_threads, avx);
    return;
}

fn profile(n: usize, p: usize, eps: f64, nthreads: usize) {
        let parts = plummer(n, [0.0; 3], 42);
        let f = Fmm::new(&parts, p, eps, 16);
        for pass in 0..2 {
            let t0 = Instant::now();
            let tree = &f.tree;
            let t1 = Instant::now();
            let mut bodies = fmm::bodies::Bodies::from_parts(&parts, &tree.order);
            let ncells = tree.cells.len();
            let d = fmm::harmonic::dim(p);
            let mut moments = vec![fmm::harmonic::C::ZERO; ncells * d];
            if nthreads > 1 {
                fmm::moments::compute_moments_par(&bodies, &tree, p, &mut moments, nthreads);
            } else {
                fmm::moments::compute_moments(&bodies, &tree, p, &mut moments);
            }
            let t2 = Instant::now();
            let pows = fmm::moments::multipole_power(&moments, ncells, p);
            let binom = fmm::moments::binomials(p);
            let (_, fmin) = if nthreads > 1 {
                fmm::mac::f_prepass_par(&bodies, &tree, nthreads)
            } else {
                fmm::mac::f_prepass(&bodies, &tree)
            };
            let t3 = Instant::now();
            let mut locals = vec![fmm::harmonic::C::ZERO; ncells * d];
            // Reset per-pass so printed stats are per-evaluation, not 2x.
            let _ = fmm::walk::walk_counts();
            if nthreads <= 1 {
                let mut rot = if p >= fmm::rotation::ROT_P { Some(fmm::rotation::RotCtx::new(p)) } else { None };
                fmm::walk::dual_walk(
                    &mut bodies, &tree, p, eps, &pows, &binom, &fmin, &moments, &mut locals, rot.as_mut(),
                );
            } else {
                fmm::walk::dual_walk_par(
                    &mut bodies, &tree, p, eps, &pows, &binom, &fmin, &moments, &mut locals, nthreads,
                );
            }
            let t4 = Instant::now();
            if nthreads > 1 {
                fmm::translate::l2l_down_par(&tree, &mut locals, p, nthreads);
            } else {
                fmm::translate::l2l_down(&tree, &mut locals, p);
            }
            let t5 = Instant::now();
            let mut pot = vec![0.0; n];
            let mut accel = vec![[0.0; 3]; n];
            if nthreads > 1 {
                fmm::translate::l2p_scatter_par(
                    &tree, &bodies, &locals, p, &mut pot, &mut accel, nthreads,
                );
            } else {
                fmm::translate::l2p_scatter(&tree, &bodies, &locals, p, &mut pot, &mut accel);
            }
            let t6 = Instant::now();
            if pass == 1 {
                println!("profile N={n} p={p} eps={eps} nthreads={nthreads}: tree-build+clone {:.4}s, moments {:.4}s, prepass/mac {:.4}s, walk {:.4}s, l2l {:.4}s, l2p+scatter {:.4}s, total {:.4}s",
                    (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64(),
                    (t4 - t3).as_secs_f64(), (t5 - t4).as_secs_f64(), (t6 - t5).as_secs_f64(), (t6 - t0).as_secs_f64());
                let (pop, acc, m2l, s2, p2, m2l_ops) = fmm::walk::walk_counts();
                println!("  walk stats: pops={pop}, accept/M2L={m2l}, m2l-pair-ops={m2l_ops}, p2p_self={s2}, p2p_pair={p2} (accept-calls={acc})");
                let (rec, merge, part, alloc, replay, reduce) = fmm::walk::walk_phase_ms();
                println!("  walk phases (ms): record={rec} merge={merge} partition={part} alloc={alloc} replay={replay} reduce={reduce}");
            }
        }
}

// Sweep mode: fmm-bench [max_n] [p] [eps] [nthreads]
fn sweep(max_n: usize, p: usize, eps: f64, nthreads: usize, avx: bool) {

    println!("FMM O(N) scaling sweep  (p={p}, eps={eps}, n_leaf=16, avx512={avx}, nthreads={nthreads}, best-of-2)");

    let ns: Vec<usize> = vec![1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
        .into_iter()
        .filter(|&n| n <= max_n)
        .collect();

    println!("\n{:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}", "N", "t_fmm(s)", "t_dir(s)", "speedup", "max_s", "rms_s", "netF/|F|");
    for &n in &ns {
        let parts = plummer(n, [0.0; 3], 42);
        let t = time_fmm(&parts, p, eps, nthreads);
        let f = Fmm::new(&parts, p, eps, 16);
        let res = f.run(&parts);
        let (max_s, rms_s) = if n <= 10_000 {
            let (a_ref, _) = direct(&parts);
            let (m, _, r, _) = scaled_error(&a_ref, &res.accel, &parts);
            (m, r)
        } else {
            sampled_error(&parts, &res.accel, 300, 7)
        };
        let net = net_force(&parts, &res.accel);
        let td = if n <= 10_000 {
            let td = time_direct(&parts);
            print!("{:>10} {:>12.4} {:>12.4} {:>12.1} ", n, t, td, td / t);
            td
        } else {
            print!("{:>10} {:>12.4} {:>12} {:>12} ", n, t, "-", "-");
            0.0
        };
        let _ = td;
        println!("{:>12.2e} {:>12.2e} {:>12.1e}", max_s, rms_s, net);
    }

    // Crossover check: direct vs FMM at the low end.
    if max_n >= 10_000 {
        println!("\nCrossover (direct vs FMM), p={p}, eps={eps}:");
        for &n in &[1_000usize, 3_000, 10_000] {
            let parts = plummer(n, [0.0; 3], 42);
            let td = time_direct(&parts);
            let tf = time_fmm(&parts, p, eps, nthreads);
            println!("  N={n:>7}: direct {td:8.4}s  FMM {tf:8.4}s  speedup {:.1}x", td / tf);
        }
    }
}