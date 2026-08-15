//! Phase-1 O(N) scaling sweep: FMM vs direct O(N^2), accuracy, and
//! momentum conservation across N = 10^3 .. 10^6.
//!
//! Usage: fmm-bench [max_n] [p] [eps]

use fmm::direct::{direct, scaled_error};
use fmm::fmm::Fmm;
use fmm::models::plummer;
use std::time::Instant;

fn time_fmm(parts: &[fmm::models::Particle], p: usize, eps: f64) -> f64 {
    // Warmup.
    let f = Fmm::new(parts, p, eps, 16);
    f.run(parts);
    // Best of 2 timed runs.
    let mut best = f64::INFINITY;
    for _ in 0..2 {
        let t0 = Instant::now();
        f.run(parts);
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

    // Profile mode: time each phase for a single N.
    if let Some(np) = args.get(4) {
        let n: usize = np.parse().unwrap();
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
            fmm::moments::compute_moments(&bodies, &tree, p, &mut moments);
            let t2 = Instant::now();
            let pows = fmm::moments::multipole_power(&moments, ncells, p);
            let binom = fmm::moments::binomials(p);
            let (_, fmin) = fmm::mac::f_prepass(&bodies, &tree);
            let t3 = Instant::now();
            let mut locals = vec![fmm::harmonic::C::ZERO; ncells * d];
            let mut rot = if p >= fmm::rotation::ROT_P { Some(fmm::rotation::RotCtx::new(p)) } else { None };
            fmm::walk::dual_walk(
                &mut bodies, &tree, p, eps, &pows, &binom, &fmin, &moments, &mut locals, rot.as_mut(),
            );
            let t4 = Instant::now();
            for c in 0..ncells {
                let cell = tree.cells[c];
                for &ch in &cell.children {
                    if ch >= 0 {
                        fmm::translate::l2l(&tree, c, ch as usize, &mut locals, p);
                    }
                }
            }
            let t5 = Instant::now();
            let mut pot = vec![0.0; n];
            let mut accel = vec![[0.0; 3]; n];
            for c in 0..ncells {
                if tree.is_leaf(c) {
                    let cell = tree.cells[c];
                    for k in cell.first..cell.first + cell.count {
                        let id = tree.order[k] as usize;
                        let mut p0 = 0.0;
                        let mut a = [0.0; 3];
                        fmm::translate::l2p(&tree, c, [bodies.x[k], bodies.y[k], bodies.z[k]], &locals, p, &mut p0, &mut a);
                        pot[id] += bodies.pot[k] + p0;
                        accel[id][0] += bodies.ax[k] + a[0];
                        accel[id][1] += bodies.ay[k] + a[1];
                        accel[id][2] += bodies.az[k] + a[2];
                    }
                }
            }
            let t6 = Instant::now();
            if pass == 1 {
                println!("profile N={n} p={p} eps={eps}: tree-build+clone {:.4}s, moments {:.4}s, prepass/mac {:.4}s, walk {:.4}s, l2l {:.4}s, l2p+scatter {:.4}s, total {:.4}s",
                    (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64(),
                    (t4 - t3).as_secs_f64(), (t5 - t4).as_secs_f64(), (t6 - t5).as_secs_f64(), (t6 - t0).as_secs_f64());
                let (pop, acc, m2l, s2, p2, m2l_ops) = fmm::walk::walk_counts();
                println!("  walk stats: pops={pop}, accept/M2L={m2l}, m2l-pair-ops={m2l_ops}, p2p_self={s2}, p2p_pair={p2} (accept-calls={acc})");
            }
        }
        return;
    }

    println!("FMM O(N) scaling sweep  (p={p}, eps={eps}, n_leaf=16, avx512={avx}, best-of-2)");

    let ns: Vec<usize> = vec![1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
        .into_iter()
        .filter(|&n| n <= max_n)
        .collect();

    println!("\n{:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}", "N", "t_fmm(s)", "t_dir(s)", "speedup", "max_s", "rms_s", "netF/|F|");
    for &n in &ns {
        let parts = plummer(n, [0.0; 3], 42);
        let t = time_fmm(&parts, p, eps);
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
            let tf = time_fmm(&parts, p, eps);
            println!("  N={n:>7}: direct {td:8.4}s  FMM {tf:8.4}s  speedup {:.1}x", td / tf);
        }
    }
}