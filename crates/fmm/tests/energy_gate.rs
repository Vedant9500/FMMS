//! ROADMAP Phase-1 exit gate (line 48): energy + momentum conservation for a
//! Plummer cluster integrated with KDK leapfrog + mutual FMM.
//!
//! One FMM force evaluation per step (drift-kick-drift). The mutual kernel
//! conserves momentum to machine precision; energy drift is driven by the FMM
//! force error (stochastic heating, ~eps·sqrt(steps)).
//!
//! Initial condition is virialized (local Maxwellian, COM removed, rescaled to
//! exact virial balance). A cold start is NOT used: the cold-start collapse
//! forms a hard binary in the core at t~0.03 for any N, which a fixed-dt
//! leapfrog cannot resolve (verified: direct and FMM agree to 1e-8 through the
//! resulting energy jump — it is integrator physics, not FMM error).
//!
//! Full gate is ignored (N=10⁶ × 10³ steps ≈ hours). Run explicitly in
//! release:
//!
//!   cargo test --release -p fmm --test energy_gate -- --ignored --nocapture
//!
//! Scale overrides (env): FMM_ENERGY_N, FMM_ENERGY_STEPS, FMM_ENERGY_DT,
//! FMM_ENERGY_EPS, FMM_ENERGY_THREADS.

use fmm::fmm::Fmm;
use fmm::models::{plummer, Particle, Rng};

struct State {
    x: Vec<[f64; 3]>,
    v: Vec<[f64; 3]>,
    m: Vec<f64>,
}

fn gaussian(rng: &mut Rng) -> f64 {
    let u1 = rng.unit().max(1e-15);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Virialized Plummer: isotropic Maxwellian with σ²(r) = 1/(6√(1+r²)) (a=1),
/// COM drift removed, velocities rescaled to exact virial balance.
fn warm_start(parts: &[Particle], pot: &[f64]) -> State {
    let n = parts.len();
    let mut s = State {
        x: parts.iter().map(|p| p.pos).collect(),
        v: vec![[0.0; 3]; n],
        m: parts.iter().map(|p| p.mass).collect(),
    };
    let mut rng = Rng::new(43);
    for i in 0..n {
        let r2 = parts[i].pos[0] * parts[i].pos[0]
            + parts[i].pos[1] * parts[i].pos[1]
            + parts[i].pos[2] * parts[i].pos[2];
        let sigma = (1.0 / (6.0 * (1.0 + r2).sqrt())).sqrt();
        for d in 0..3 {
            s.v[i][d] = sigma * gaussian(&mut rng);
        }
    }
    let mtot: f64 = s.m.iter().sum();
    let mut pcom = [0.0; 3];
    for i in 0..n {
        for d in 0..3 {
            pcom[d] += s.m[i] * s.v[i][d];
        }
    }
    for i in 0..n {
        for d in 0..3 {
            s.v[i][d] -= pcom[d] / mtot;
        }
    }
    let t0 = kinetic(&s);
    let w0 = potential(&s, pot);
    if t0 > 0.0 {
        let c = (-0.5 * w0 / t0).sqrt();
        for i in 0..n {
            for d in 0..3 {
                s.v[i][d] *= c;
            }
        }
    }
    s
}

fn to_parts(s: &State) -> Vec<Particle> {
    (0..s.m.len())
        .map(|i| Particle {
            mass: s.m[i],
            pos: s.x[i],
        })
        .collect()
}

fn kinetic(s: &State) -> f64 {
    let mut t = 0.0;
    for i in 0..s.m.len() {
        let v = s.v[i];
        t += 0.5 * s.m[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    t
}

/// Potential energy W = -1/2 Σ m_i pot_i (pot convention: Σ m_j / r_ij > 0).
fn potential(s: &State, pot: &[f64]) -> f64 {
    let mut w = 0.0;
    for i in 0..s.m.len() {
        w += s.m[i] * pot[i];
    }
    -0.5 * w
}

/// |Σ m v| / Σ m |v| — near zero for a COM-cleaned mutual-FMM run.
fn momentum_rel(s: &State) -> f64 {
    let mut p = [0.0; 3];
    let mut mag = 0.0;
    for i in 0..s.m.len() {
        let v = s.v[i];
        let vm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        p[0] += s.m[i] * v[0];
        p[1] += s.m[i] * v[1];
        p[2] += s.m[i] * v[2];
        mag += s.m[i] * vm;
    }
    let norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    norm / mag.max(f64::MIN_POSITIVE)
}

struct EnergyStats {
    e0: f64,
    e_last: f64,
    drift: f64,
    max_dev: f64,
    slope: f64,
    momentum: f64,
}

fn integrate(
    n: usize,
    steps: usize,
    dt: f64,
    p: usize,
    eps: f64,
    threads: usize,
    verbose: bool,
) -> EnergyStats {
    let parts = plummer(n, [0.0; 3], 42);
    let mut fmm = Fmm::new(&parts, p, eps, 16);
    let res0 = fmm.run_threaded(&parts, threads);
    let s0 = warm_start(&parts, &res0.pot);
    let mut s = s0;
    let e0 = kinetic(&s) + potential(&s, &res0.pot);
    let mut e_min = e0;
    let mut e_max = e0;
    let mut e_last = e0;
    let mut sum_t = 0.0f64;
    let mut sum_e = 0.0f64;
    let mut sum_tt = 0.0f64;
    let mut sum_te = 0.0f64;
    let t0 = std::time::Instant::now();
    for step in 1..=steps {
        for i in 0..n {
            let v = s.v[i];
            s.x[i][0] += 0.5 * dt * v[0];
            s.x[i][1] += 0.5 * dt * v[1];
            s.x[i][2] += 0.5 * dt * v[2];
        }
        let parts = to_parts(&s);
        fmm.refresh(&parts);
        let res = fmm.run_threaded(&parts, threads);
        for i in 0..n {
            let a = res.accel[i];
            s.v[i][0] += dt * a[0];
            s.v[i][1] += dt * a[1];
            s.v[i][2] += dt * a[2];
        }
        for i in 0..n {
            let v = s.v[i];
            s.x[i][0] += 0.5 * dt * v[0];
            s.x[i][1] += 0.5 * dt * v[1];
            s.x[i][2] += 0.5 * dt * v[2];
        }
        let e = kinetic(&s) + potential(&s, &res.pot);
        e_min = e_min.min(e);
        e_max = e_max.max(e);
        e_last = e;
        let st = step as f64;
        sum_t += st;
        sum_e += e;
        sum_tt += st * st;
        sum_te += st * e;
        if verbose && (step % 50 == 0 || step == steps) {
            let el = t0.elapsed().as_secs_f64();
            let mut vmax = 0.0f64;
            for i in 0..n {
                let v = s.v[i];
                let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                vmax = vmax.max(vv);
            }
            println!(
                "  step {step:>6}/{steps}  dE/E0={:+.3e}  t={:.3}  vmax={:.3e}  ({:.2}s/step, {}s elapsed)",
                (e - e0) / e0.abs(),
                st * dt,
                vmax.sqrt(),
                el / st,
                el
            );
        }
    }
    let denom = steps as f64 * sum_tt - sum_t * sum_t;
    let slope = if denom.abs() > 0.0 {
        (steps as f64 * sum_te - sum_t * sum_e) / denom
    } else {
        0.0
    };
    EnergyStats {
        e0,
        e_last,
        drift: (e_last - e0) / e0.abs(),
        max_dev: (e_min - e0).abs().max((e_max - e0).abs()) / e0.abs(),
        slope: slope / e0.abs(),
        momentum: momentum_rel(&s),
    }
}

fn env_u64(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
fn energy_conservation_smoke() {
    let st = integrate(1024, 100, 1e-4, 4, 1e-3, 1, false);
    println!(
        "smoke: E0={:.6e} drift={:.3e} max_dev={:.3e} slope={:.3e}/step momentum={:.3e}",
        st.e0, st.drift, st.max_dev, st.slope, st.momentum
    );
    // Virialized Plummer: E0 = T + W = W/2 = -3π/64 ≈ -0.147.
    assert!(
        (st.e0 + 0.147).abs() < 0.03,
        "E0={} not ~-0.147 (Plummer)",
        st.e0
    );
    assert!(st.momentum < 1e-12, "momentum rel {}", st.momentum);
    assert!(st.drift.abs() < 1e-2, "energy drift {} too large", st.drift);
    assert!(st.max_dev < 1e-2, "energy max dev {} too large", st.max_dev);
}

#[test]
#[ignore = "N=10^6 x 10^3 steps; run explicitly: cargo test --release -p fmm --test energy_gate -- --ignored --nocapture"]
fn energy_conservation_gate() {
    let n = env_u64("FMM_ENERGY_N", 1_000_000);
    let steps = env_u64("FMM_ENERGY_STEPS", 1_000);
    let dt = env_f64("FMM_ENERGY_DT", 1e-3);
    let eps = env_f64("FMM_ENERGY_EPS", 1e-5);
    let threads = env_u64("FMM_ENERGY_THREADS", 8);
    println!("energy gate: N={n} steps={steps} dt={dt} p=8 eps={eps} threads={threads}");
    let t0 = std::time::Instant::now();
    let st = integrate(n, steps, dt, 8, eps, threads, true);
    println!("energy gate result:");
    println!("  E0={:.8e}  E_last={:.8e}", st.e0, st.e_last);
    println!(
        "  drift={:.3e}  max_dev={:.3e}  slope={:.3e}/step  momentum={:.3e}  [{}s]",
        st.drift,
        st.max_dev,
        st.slope,
        st.momentum,
        t0.elapsed().as_secs_f64()
    );
    assert!(st.momentum < 1e-12, "momentum rel {}", st.momentum);
    assert!(
        st.drift.abs() < 1e-3,
        "energy drift {} exceeds gate",
        st.drift
    );
    assert!(
        st.max_dev < 1e-3,
        "energy max dev {} exceeds gate",
        st.max_dev
    );
    assert!(
        st.slope.abs() < 1e-6,
        "secular slope {} too large",
        st.slope
    );
}
