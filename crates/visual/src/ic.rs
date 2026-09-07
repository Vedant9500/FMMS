//! Initial-condition presets: structured galaxies, not a bare Plummer blob.
//! Units G = M_tot = 1. Every generator returns zero-COM, zero-momentum state.

use fmm::models::Rng;
use std::sync::LazyLock;

/// Log-space stellar scales, computed once: per-class (ln teff_lo, ln teff_hi,
/// ln lum_lo, ln lum_hi) plus the temp-map anchor ln(2300) and 1/ln(50000/2300).
/// `stellar` runs per star, so the ~6 ln() calls it used to repeat per star
/// (~120ms/M) happen exactly once here.
struct StarScales {
    log: [(f64, f64, f64, f64); 8],
    ln_lo: f64,
    inv_span: f64,
}

static SCALES: LazyLock<StarScales> = LazyLock::new(|| {
    let one = |c: SpecClass| {
        let (tlo, thi) = c.teff_range();
        let (llo, lhi) = c.lum_range();
        (tlo.ln(), thi.ln(), llo.ln(), lhi.ln())
    };
    // NB: table order must match SpecClass discriminants (O=0..Giant=7);
    // the census tests already pin that order via `as usize` indexing.
    let order = [
        SpecClass::O,
        SpecClass::B,
        SpecClass::A,
        SpecClass::F,
        SpecClass::G,
        SpecClass::K,
        SpecClass::M,
        SpecClass::Giant,
    ];
    let log = [
        one(order[0]),
        one(order[1]),
        one(order[2]),
        one(order[3]),
        one(order[4]),
        one(order[5]),
        one(order[6]),
        one(order[7]),
    ];
    let ln_lo = 2300.0f64.ln();
    let inv_span = 1.0 / (50000.0f64.ln() - ln_lo);
    StarScales {
        log,
        ln_lo,
        inv_span,
    }
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Cluster,
    Spiral,
    Merger,
    BlackHole,
}

impl Preset {
    pub fn from_str(s: &str) -> Option<Preset> {
        if s.eq_ignore_ascii_case("cluster") {
            Some(Preset::Cluster)
        } else if s.eq_ignore_ascii_case("spiral") {
            Some(Preset::Spiral)
        } else if s.eq_ignore_ascii_case("merger") {
            Some(Preset::Merger)
        } else if s.eq_ignore_ascii_case("bh")
            || s.eq_ignore_ascii_case("blackhole")
            || s.eq_ignore_ascii_case("black-hole")
        {
            Some(Preset::BlackHole)
        } else {
            None
        }
    }
}

pub struct Star {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub mass: f64,
    pub temp: f32,
    pub mag: f32,
    pub teff: f32,
    pub lum: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpecClass {
    O,
    B,
    A,
    F,
    G,
    K,
    M,
    Giant,
}

impl SpecClass {
    fn teff_range(&self) -> (f64, f64) {
        match self {
            SpecClass::O => (33000.0, 50000.0),
            SpecClass::B => (10000.0, 33000.0),
            SpecClass::A => (7300.0, 10000.0),
            SpecClass::F => (6000.0, 7300.0),
            SpecClass::G => (5300.0, 6000.0),
            SpecClass::K => (3900.0, 5300.0),
            SpecClass::M => (2300.0, 3900.0),
            SpecClass::Giant => (3500.0, 5000.0),
        }
    }
    fn lum_range(&self) -> (f64, f64) {
        match self {
            SpecClass::O => (30000.0, 800000.0),
            SpecClass::B => (25.0, 30000.0),
            SpecClass::A => (5.0, 25.0),
            SpecClass::F => (1.5, 5.0),
            SpecClass::G => (0.6, 1.5),
            SpecClass::K => (0.08, 0.6),
            SpecClass::M => (0.001, 0.08),
            SpecClass::Giant => (10.0, 1000.0),
        }
    }
}

fn sample_class(rng: &mut Rng, young: bool) -> SpecClass {
    let u = rng.unit();
    // Common-first branch order (M dominates): same thresholds as the
    // cosmic fractions below, ~6x fewer compares at 1M. Boundaries are
    // identical to the rare-first form, just tested descending.
    if young {
        if u >= 0.2333003 {
            SpecClass::M
        } else if u >= 0.1133003 {
            SpecClass::K
        } else if u >= 0.0373003 {
            SpecClass::G
        } else if u >= 0.0073003 {
            SpecClass::F
        } else if u >= 0.0012003 {
            SpecClass::A
        } else if u >= 0.0000003 {
            SpecClass::B
        } else {
            SpecClass::O
        }
    } else if u >= 0.2712 {
        SpecClass::M
    } else if u >= 0.2412 {
        SpecClass::Giant
    } else if u >= 0.0912 {
        SpecClass::K
    } else if u >= 0.0112 {
        SpecClass::G
    } else if u >= 0.0012 {
        SpecClass::F
    } else if u >= 0.0002 {
        SpecClass::A
    } else {
        SpecClass::B
    }
}

fn stellar(rng: &mut Rng, class: SpecClass) -> (f32, f32, f32, f32) {
    let (tlo, thi, llo, lhi) = SCALES.log[class as usize];
    let u = rng.unit();
    // Single-u 1-parameter model: teff and lum rank together within a class
    // (hotter <=> brighter), gauss scatter decorrelates. Matches the
    // mass-lum rank match downstream; do not "fix" with a second draw
    // (shifts the whole RNG stream and all golden counts).
    let lte = thi + (tlo - thi) * u + 0.02 * gauss(rng);
    let teff = lte.exp();
    // Clamp in log space: identical bounds to linear [llo*0.5, lhi*2]
    // (exp is monotonic), no extra exp() calls.
    let llum = (lhi + (llo - lhi) * u + 0.1 * gauss(rng))
        .clamp(llo - std::f64::consts::LN_2, lhi + std::f64::consts::LN_2);
    let lum = llum.exp();
    let t = 1.0 - (lte - SCALES.ln_lo) * SCALES.inv_span;
    let temp = t.clamp(0.02, 1.0) as f32;
    let mag = ((lum.log10() + 3.0) / 6.0).clamp(0.03, 1.0) as f32;
    (temp, mag, teff as f32, lum as f32)
}

pub fn class_index(teff: f32, lum: f32) -> Option<usize> {
    if teff <= 0.0 {
        return None;
    }
    if lum > 12.0 && teff < 7300.0 {
        return Some(7);
    }
    // Common-first: M dwarfs dominate any field.
    Some(if teff < 3900.0 {
        6
    } else if teff < 5300.0 {
        5
    } else if teff < 6000.0 {
        4
    } else if teff < 7300.0 {
        3
    } else if teff < 10000.0 {
        2
    } else if teff < 33000.0 {
        1
    } else {
        0
    })
}

#[allow(dead_code)]
pub fn census(teffs: &[f32], lums: &[f32]) -> [usize; 8] {
    let mut c = [0usize; 8];
    for (&t, &l) in teffs.iter().zip(lums.iter()) {
        if let Some(k) = class_index(t, l) {
            c[k] += 1;
        }
    }
    c
}

fn assign_looks(rng: &mut Rng, masses: &[f64], young: bool) -> Vec<(f32, f32, f32, f32)> {
    let n = masses.len();
    let mut looks: Vec<(f32, f32, f32, f32)> = (0..n)
        .map(|_| {
            let c = sample_class(rng, young);
            stellar(rng, c)
        })
        .collect();
    // Sort the 16B tuples in place by lum: kills the `li` index vec (8MB/M)
    // and the gather indirection. Same rank permutation as before.
    // `total_cmp` (no NaN possible, never panics) + unstable (2-3x faster;
    // ties are measure-zero on continuous lum).
    looks.sort_unstable_by(|a, b| a.3.total_cmp(&b.3));
    let mut mi: Vec<usize> = (0..n).collect();
    mi.sort_unstable_by(|&a, &b| masses[a].total_cmp(&masses[b]));
    let mut out = vec![(0.0f32, 0.0f32, 0.0f32, 0.0f32); n];
    for (k, m) in mi.iter().enumerate() {
        out[*m] = looks[k];
    }
    out
}

pub(crate) fn gauss(rng: &mut Rng) -> f64 {
    // Single-variate Box-Muller (spare discarded on purpose): caching the
    // second variate would halve RNG draws but shift the entire init stream
    // and every golden count. Transcendentals here are one-time init cost;
    // the sorts were the real bottleneck (fixed in assign_looks).
    let u1 = rng.unit().max(1e-12);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Bottom-heavy mass spectrum (Salpeter flavor), normalized to sum 1.
/// Ratio ~10:1 — wider spreads segregate and eject lights within minutes.
fn spectrum(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut w: Vec<f64> = (0..n).map(|_| 0.3 + 2.7 * rng.unit().powi(4)).collect();
    let s: f64 = w.iter().sum::<f64>().max(1e-18);
    for x in &mut w {
        *x /= s;
    }
    w
}

#[derive(Clone, Copy, Debug)]
pub struct Halo {
    pub m: f64,
    pub a: f64,
}

/// Halo acceleration: a = -M r_vec / (r (r + a)^2).
pub fn halo_accel(h: &Halo, pos: &[f64; 3]) -> [f64; 3] {
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
        .sqrt()
        .max(1e-9);
    let ra = r + h.a;
    let s = -h.m / (r * ra * ra);
    [s * pos[0], s * pos[1], s * pos[2]]
}

/// Halo contribution to v_circ^2: M r / (r + a)^2.
pub fn halo_v2(h: &Halo, r: f64) -> f64 {
    h.m * r / ((r + h.a) * (r + h.a))
}

/// Renormalize masses to exactly 1 and subtract COM position/momentum in one
/// pass pair. Fused: the old code ran a renorm loop in `generate` plus two
/// passes here (3 streaming passes over 72B/star); this is 2.
fn renorm_zero_com(s: &mut [Star]) {
    let mut mp = [0.0; 3];
    let mut mv = [0.0; 3];
    let mut mtot = 0.0;
    for st in s.iter() {
        mtot += st.mass;
        for d in 0..3 {
            mp[d] += st.mass * st.pos[d];
            mv[d] += st.mass * st.vel[d];
        }
    }
    if mtot <= 0.0 {
        return;
    }
    let inv = 1.0 / mtot;
    let cp = [mp[0] * inv, mp[1] * inv, mp[2] * inv];
    let cv = [mv[0] * inv, mv[1] * inv, mv[2] * inv];
    for st in s.iter_mut() {
        st.mass *= inv;
        for d in 0..3 {
            st.pos[d] -= cp[d];
            st.vel[d] -= cv[d];
        }
    }
}

fn disk_pos(rng: &mut Rng, rd: f64, r_min: f64, r_max: f64, arms: bool) -> ([f64; 3], f64, f64) {
    let mut r = r_min;
    let mut phi = 0.0;
    let mut ok = false;
    for _ in 0..24 {
        r = r_min - rd * (1.0 - rng.unit()).ln();
        if r > r_max {
            continue;
        }
        phi = 2.0 * std::f64::consts::PI * rng.unit();
        if !arms {
            ok = true;
            break;
        }
        let arm = (2.0 * phi - 3.0 * (r + 0.15).ln()).cos().powi(2);
        if rng.unit() < 0.40 + 0.60 * arm {
            ok = true;
            break;
        }
    }
    if !ok {
        // 24 straight rejects (~never: P ~ 0.3^24): randomize azimuth instead
        // of piling at phi=0 with r clamped to r_max.
        phi = 2.0 * std::f64::consts::PI * rng.unit();
    }
    r = r.clamp(r_min, r_max);
    let sig = 0.05 + 0.02 * r;
    let x = r * phi.cos();
    let y = r * phi.sin();
    ([x, y, sig * gauss(rng)], r, phi)
}

struct DiskSpec {
    rd: f64,
    m_disk: f64,
    m_bulge: f64,
    m_bh: f64,
    bulge_a: f64,
    r_max: f64,
    halo: Option<Halo>,
}

fn menc_live(spec: &DiskSpec, r: f64) -> f64 {
    let disk = spec.m_disk * (1.0 - (1.0 + r / spec.rd) * (-r / spec.rd).exp());
    let bulge = spec.m_bulge * (r / (r + spec.bulge_a)).powi(2);
    (spec.m_bh + disk + bulge).max(1e-9)
}

/// Full circular speed squared: live enclosed mass + halo (flat outer curve).
fn vc2(spec: &DiskSpec, r: f64) -> f64 {
    let rr = r.max(1e-6);
    let mut v2 = menc_live(spec, rr) / rr;
    if let Some(h) = &spec.halo {
        v2 += halo_v2(h, rr);
    }
    v2
}

fn spiral_with(
    n: usize,
    rng: &mut Rng,
    spec: &DiskSpec,
    omp: f32,
    nuc: (f64, f64, f64),
) -> (Vec<Star>, Vec<Gas>) {
    let bh_n = if n > 8 { 1 } else { 0 };
    let disk_n = (n - bh_n) * 70 / 100;
    let bulge_n = n - bh_n - disk_n;
    let w = spectrum(rng, disk_n + bulge_n);
    let total = spec.m_disk + spec.m_bulge;
    let disk_looks = assign_looks(rng, &w[..disk_n], true);
    let bulge_looks = assign_looks(rng, &w[disk_n..], false);
    let mut out = Vec::with_capacity(n);
    for i in 0..disk_n {
        let (pos, r, _) = disk_pos(rng, spec.rd, 0.05, spec.r_max, true);
        let vc = vc2(spec, r).sqrt().min(3.0);
        let sv = 0.07 * vc;
        // Tangent from position, not phi trig: (-y/r, x/r) == (-sin, cos).
        let rr = r.max(1e-9);
        let tx = -pos[1] / rr;
        let ty = pos[0] / rr;
        let vel = [
            tx * vc + sv * gauss(rng),
            ty * vc + sv * gauss(rng),
            sv * gauss(rng),
        ];
        let (temp, mag, teff, lum) = disk_looks[i];
        out.push(Star {
            pos,
            vel,
            mass: w[i] * total,
            temp,
            mag,
            teff,
            lum,
        });
    }
    for i in 0..bulge_n {
        // Hernquist inverse CDF with bounded resample: the old `.min(3.0)`
        // stacked ~15% (a=0.25) of the bulge on an exact r=3.0 shell.
        let mut r = 3.0;
        for _ in 0..8 {
            let s = rng.unit().sqrt().max(1e-9);
            r = spec.bulge_a * s / (1.0 - s);
            if r <= 3.0 {
                break;
            }
        }
        r = r.clamp(0.02, 3.0);
        let z = 2.0 * rng.unit() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * rng.unit();
        let (sphi, cphi) = phi.sin_cos();
        let sp = (1.0 - z * z).sqrt().max(1e-9);
        let pos = [r * sp * cphi, r * sp * sphi, r * z];
        let v2 = vc2(spec, r);
        let vc = v2.sqrt().min(3.0);
        // sb samples the curve at max(r, 0.05) as before; reuse v2 on the
        // common path, one extra eval only for the ~3% with r < 0.05.
        let v2b = if r >= 0.05 { v2 } else { vc2(spec, 0.05) };
        let sb = (0.35 * v2b.sqrt()).min(1.5);
        let vel = [
            -sphi * 0.3 * vc + sb * gauss(rng),
            cphi * 0.3 * vc + sb * gauss(rng),
            sb * gauss(rng),
        ];
        let (temp, mag, teff, lum) = bulge_looks[i];
        out.push(Star {
            pos,
            vel,
            mass: w[disk_n + i] * total,
            temp,
            mag,
            teff,
            lum,
        });
    }
    if bh_n == 1 {
        out.push(Star {
            pos: [0.0; 3],
            vel: [0.0; 3],
            mass: spec.m_bh,
            temp: 0.5,
            mag: 0.02,
            teff: 0.0,
            lum: 0.0,
        });
    }
    let disk = &out[..disk_n];
    let gas = gas_for_disk(
        rng,
        disk,
        spec.rd,
        spec.r_max,
        0.1,
        true,
        &|r| vc2(spec, r).sqrt(),
        omp as f64,
        nuc.0,
        nuc.1,
        nuc.2,
    );
    (out, gas)
}

/// BH-disk circular v^2: point-mass BH + exponential disk + halo.
/// Single source of truth (was copy-pasted 3x: BH arm, BH gas closure,
/// `gas_disk_table` BH arm).
fn bh_disk_v2(m_bh: f64, m_disk: f64, h: &Halo, r: f64) -> f64 {
    let rr = r.max(1e-6);
    (m_bh + m_disk * (1.0 - (1.0 + rr / 0.6) * (-rr / 0.6).exp())) / rr + halo_v2(h, rr)
}

/// Precomputed X-then-Z rotation: the merger loop applied per-star trig
/// (~4M sin_cos pairs at 1M). Hoist one matrix per galaxy.
struct Rot {
    sx: f64,
    cx: f64,
    sz: f64,
    cz: f64,
}

impl Rot {
    fn new(cx: f64, cz: f64) -> Self {
        let (sx, cx_) = cx.sin_cos();
        let (sz, cz_) = cz.sin_cos();
        Self {
            sx,
            cx: cx_,
            sz,
            cz: cz_,
        }
    }
}

#[inline]
fn rot_apply(m: &Rot, p: [f64; 3]) -> [f64; 3] {
    let y1 = m.cx * p[1] - m.sx * p[2];
    let z1 = m.sx * p[1] + m.cx * p[2];
    [m.cz * p[0] - m.sz * y1, m.sz * p[0] + m.cz * y1, z1]
}

fn spiral_spec() -> DiskSpec {
    DiskSpec {
        rd: 1.0,
        m_disk: 0.68,
        m_bulge: 0.30,
        m_bh: 0.02,
        bulge_a: 0.25,
        r_max: 4.0,
        halo: Some(Halo { m: 3.0, a: 3.0 }),
    }
}

pub fn generate(preset: Preset, n: usize, seed: u64) -> (Vec<Star>, Vec<Gas>, Option<Halo>) {
    let mut rng = Rng::new(seed);
    let halo = match preset {
        Preset::Cluster => None,
        Preset::Spiral => Some(Halo { m: 3.0, a: 3.0 }),
        Preset::BlackHole => Some(Halo { m: 1.5, a: 2.0 }),
        Preset::Merger => Some(Halo { m: 6.0, a: 5.0 }),
    };
    let (mut s, gas): (Vec<Star>, Vec<Gas>) = match preset {
        Preset::Cluster => {
            // Plummer positions sampled inline with a dedicated stream
            // (same seed/sequence as `fmm::models::plummer`, bit-identical):
            // the old code built a 32MB Vec<Particle> at 1M just to copy
            // `pos` out and throw the masses away.
            let w = spectrum(&mut rng, n);
            let looks = assign_looks(&mut rng, &w, false);
            let mut prng = Rng::new(seed);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let u = prng.unit();
                let ww = u.powf(2.0 / 3.0);
                let r = (ww / (1.0 - ww)).sqrt();
                let z = 2.0 * prng.unit() - 1.0;
                let phi = 2.0 * std::f64::consts::PI * prng.unit();
                let sp = (1.0 - z * z).sqrt();
                let pos = [r * sp * phi.cos(), r * sp * phi.sin(), r * z];
                let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
                let sig = (1.0 / (6.0 * (1.0 + r2).sqrt())).sqrt();
                let (temp, mag, teff, lum) = looks[i];
                out.push(Star {
                    pos,
                    vel: [
                        sig * gauss(&mut rng),
                        sig * gauss(&mut rng),
                        sig * gauss(&mut rng),
                    ],
                    mass: w[i],
                    temp,
                    mag,
                    teff,
                    lum,
                });
            }
            (out, Vec::new())
        }
        Preset::Spiral => spiral_with(n, &mut rng, &spiral_spec(), 0.35, (0.07, 0.30, 0.08)),
        Preset::BlackHole => {
            let bh_halo = Halo { m: 1.5, a: 2.0 };
            let bh_n = if n > 8 { 1 } else { 0 };
            let disk_n = (n - bh_n) * 85 / 100;
            let halo_n = n - bh_n - disk_n;
            let (m_bh, m_disk, m_halo) = (0.12, 0.78, 0.10);
            let w = spectrum(&mut rng, disk_n + halo_n);
            let disk_looks = assign_looks(&mut rng, &w[..disk_n], true);
            let halo_looks = assign_looks(&mut rng, &w[disk_n..], false);
            let mut out = Vec::with_capacity(n);
            for i in 0..disk_n {
                let (pos, r, _) = disk_pos(&mut rng, 0.6, 0.15, 2.5, true);
                let vc = bh_disk_v2(m_bh, m_disk, &bh_halo, r).sqrt().min(3.0);
                let sv = 0.05 * vc;
                let rr = r.max(1e-9);
                let vel = [
                    -pos[1] / rr * vc + sv * gauss(&mut rng),
                    pos[0] / rr * vc + sv * gauss(&mut rng),
                    sv * gauss(&mut rng),
                ];
                let (mut temp, mut mag, mut teff, mut lum) = disk_looks[i];
                if r < 0.35 {
                    temp = (0.02 + 0.08 * rng.unit()) as f32;
                    mag = (mag * 1.5).min(0.9);
                    teff = 12000.0 + 8000.0 * rng.unit() as f32;
                    lum = (lum * 2.0).min(500.0);
                }
                out.push(Star {
                    pos,
                    vel,
                    mass: w[i] * (m_disk + m_halo),
                    temp,
                    mag,
                    teff,
                    lum,
                });
            }
            for i in 0..halo_n {
                // Bounded resample, not `.min(4.0)`: the old cap stacked ~30%
                // (a=0.8) of halo stars on an exact r=4.0 shell.
                let mut r = 4.0;
                for _ in 0..8 {
                    let s = rng.unit().sqrt().max(1e-9);
                    r = 0.8 * s / (1.0 - s);
                    if r <= 4.0 {
                        break;
                    }
                }
                r = r.clamp(0.05, 4.0);
                let z = 2.0 * rng.unit() - 1.0;
                let phi = 2.0 * std::f64::consts::PI * rng.unit();
                let (sphi, cphi) = phi.sin_cos();
                let sp = (1.0 - z * z).sqrt().max(1e-9);
                let sb = (0.4 * (m_bh / r.max(0.1)).sqrt()).min(1.5);
                let (temp, mag, teff, lum) = halo_looks[i];
                out.push(Star {
                    pos: [r * sp * cphi, r * sp * sphi, r * z],
                    vel: [
                        sb * gauss(&mut rng),
                        sb * gauss(&mut rng),
                        sb * gauss(&mut rng),
                    ],
                    mass: w[disk_n + i] * (m_disk + m_halo),
                    temp,
                    mag,
                    teff,
                    lum,
                });
            }
            if bh_n == 1 {
                out.push(Star {
                    pos: [0.0; 3],
                    vel: [0.0; 3],
                    mass: m_bh,
                    temp: 0.5,
                    mag: 0.02,
                    teff: 0.0,
                    lum: 0.0,
                });
            }
            let gas = gas_for_disk(
                &mut rng,
                &out,
                0.6,
                2.5,
                0.2,
                true,
                &|r| bh_disk_v2(m_bh, m_disk, &bh_halo, r).sqrt(),
                0.5,
                0.15,
                0.40,
                0.12,
            );
            (out, gas)
        }
        Preset::Merger => {
            let spec = DiskSpec {
                rd: 0.8,
                m_disk: 0.34,
                m_bulge: 0.15,
                m_bh: 0.01,
                bulge_a: 0.2,
                r_max: 3.0,
                halo: None,
            };
            let (mut a, mut ga) = spiral_with(n / 2, &mut rng, &spec, 0.2, (0.0, 0.0, 0.0));
            let (mut b, mut gb) = spiral_with(n - n / 2, &mut rng, &spec, 0.2, (0.0, 0.0, 0.0));
            // One rotation matrix per galaxy (was 2 sin_cos per star+gas).
            let ra = Rot::new(0.3, 0.0);
            let rb = Rot::new(-1.1, 0.5);
            for st in a.iter_mut() {
                st.pos = rot_apply(&ra, st.pos);
                st.vel = rot_apply(&ra, st.vel);
                st.pos[0] -= 2.6;
                st.pos[1] += 0.2;
                st.vel[0] += 0.16;
                st.vel[1] += 0.10;
                st.vel[2] += 0.03;
            }
            for g in ga.iter_mut() {
                g.pos = rot_apply(&ra, g.pos);
                g.pos[0] -= 2.6;
                g.pos[1] += 0.2;
                g.vel = rot_apply(&ra, g.vel);
                g.vel[0] += 0.16;
                g.vel[1] += 0.10;
                g.vel[2] += 0.03;
            }
            // NOTE: no `mass *= 0.5` here. Each half already sums to ~0.5 and
            // `renorm_zero_com` below scales the total to exactly 1, so the
            // halve was a 2M-mult no-op. Do not re-add it.
            for st in b.iter_mut() {
                st.pos = rot_apply(&rb, st.pos);
                st.vel = rot_apply(&rb, st.vel);
                st.pos[0] += 2.6;
                st.pos[1] -= 0.2;
                st.vel[0] -= 0.16;
                st.vel[1] -= 0.10;
                st.vel[2] -= 0.03;
            }
            for g in gb.iter_mut() {
                g.pos = rot_apply(&rb, g.pos);
                g.pos[0] += 2.6;
                g.pos[1] -= 0.2;
                g.vel = rot_apply(&rb, g.vel);
                g.vel[0] -= 0.16;
                g.vel[1] -= 0.10;
                g.vel[2] -= 0.03;
            }
            a.reserve(b.len());
            a.extend(b);
            ga.reserve(gb.len());
            ga.extend(gb);
            (a, ga)
        }
    };
    // Spec masses already sum to ~1 (merger joins two ~0.5s); exact 1.0 plus
    // zero COM/momentum in one fused pass pair (see renorm_zero_com).
    renorm_zero_com(&mut s);
    (s, gas, halo)
}
#[derive(Clone)]
pub struct Gas {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub size: f32,
    /// Emission (0/1/2) vs dust (3) as f32, not u8: the render upload packs
    /// it into `GpuItem.misc` (f32x4), so u8 would just add casts. Gas is
    /// capped at ~1800 knots anyway; the struct is never the 1M bottleneck.
    pub kind: f32,
    pub seed: f32,
    /// Supernova / tidal heat glow 0..1. Merge transfers pump it, tidal
    /// heating near the BH floors it, step_gas decays it (tau ~0.6 sim).
    /// Streams to the emission shader (misc.w) for white-blue flash.
    pub flash: f32,
}

/// M3 nuclear structure: cleared cavity (feedback blows out the center)
/// + circumnuclear ring outside it. Returns repositioned pos. Ring is
/// feathered (wide scatter + m=2 spiral modulation, not a perfect circle)
/// and the cavity leaks: ~15% of inner knots stay inside as a tenuous hot
/// component so lanes spiral inward instead of a void+moat. Leak decision
/// is a deterministic pos hash (no extra rng: exact-count tests stay put).
fn nuclear_pos(
    rng: &mut Rng,
    pos: [f64; 3],
    r_clear: f64,
    r_ring: f64,
    ring_frac: f64,
    ring: bool,
) -> [f64; 3] {
    if r_clear <= 0.0 {
        return pos;
    }
    let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
    let mut p = pos;
    if ring && r < r_ring * 1.5 && rng.unit() < ring_frac {
        let phi = pos[1].atan2(pos[0]);
        let spiral = 1.0 + 0.25 * (2.0 * phi + 1.0).sin();
        // Clamp the wide (0.50) scatter: ~2% of draws go negative and would
        // mirror the knot through the origin.
        let nr = r_ring * (1.0 + 0.50 * gauss(rng)).max(0.2) * spiral;
        let s = nr / r.max(1e-9);
        p[0] *= s;
        p[1] *= s;
        p[2] *= 0.3;
    } else if r < r_clear {
        // Deterministic leak: hash in [0,1), ~15% stay inside (flattened).
        let h = ((pos[0] * 74.7 + pos[1] * 19.3 + pos[2] * 11.1).sin() * 43758.5).fract();
        let h = if h < 0.0 { h + 1.0 } else { h };
        if h < 0.15 {
            p[2] *= 0.3;
            return p;
        }
        let nr = r_clear * (1.1 + 0.6 * rng.unit());
        let s = nr / r.max(1e-9);
        p[0] *= s;
        p[1] *= s;
    }
    p
}
fn gas_vel(rng: &mut Rng, pos: [f64; 3], vcirc: &impl Fn(f64) -> f64, disp: f64) -> [f64; 3] {
    // Circular velocity for a fresh knot: prograde around +Z plus a small
    // collisional dispersion (cold gas rides near-circular, ~11% faster
    // than lagging stars). Tight for lanes, looser for knots.
    let r0 = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
    let r = r0.max(0.05);
    let v = vcirc(r);
    let (s, c) = (pos[1] / r.max(1e-9), pos[0] / r.max(1e-9));
    [
        -s * v + disp * v * gauss(rng),
        c * v + disp * v * gauss(rng),
        0.5 * disp * v * gauss(rng),
    ]
}

fn arm_lane_phi(rng: &mut Rng, rd: f64, r_min: f64, r_max: f64) -> ([f64; 3], f64, f64) {
    let mut r = r_min;
    for _ in 0..24 {
        r = r_min - rd * (1.0 - rng.unit()).ln();
        if r <= r_max {
            break;
        }
    }
    r = r.clamp(r_min, r_max);
    let k = if rng.unit() < 0.5 {
        0.0
    } else {
        std::f64::consts::PI
    };
    let phi = (k - 0.55 + 3.0 * (r + 0.15).ln()) / 2.0 + 0.08 * gauss(rng);
    let sig = 0.04 + 0.015 * r;
    ([r * phi.cos(), r * phi.sin(), sig * gauss(rng)], r, phi)
}

fn gas_for_disk(
    mut rng: &mut Rng,
    stars: &[Star],
    rd: f64,
    r_max: f64,
    lane_r_min: f64,
    lanes: bool,
    vcirc: &impl Fn(f64) -> f64,
    omp: f64,
    r_clear: f64,
    r_ring: f64,
    ring_frac: f64,
) -> Vec<Gas> {
    let n = stars.len();
    // Gas is O(1) by design: knot/lane counts cap at 900+900 no matter how
    // large N grows. That cap is what keeps 1M stars affordable.
    let nk = (300 + n / 100).min(900);
    let nl = if lanes { (200 + n / 150).min(900) } else { 0 };
    let mut out = Vec::with_capacity(nk + nl);
    let anchors: Vec<&Star> = stars.iter().filter(|st| st.teff >= 7300.0).collect();
    for _ in 0..nk {
        let (pos, hot) = if !anchors.is_empty() && rng.unit() < 0.6 {
            // Cast is in range: unit() < 1 so product < len. min() is belt
            // and braces against a future RNG that can return exactly 1.
            let st = anchors[((rng.unit() * anchors.len() as f64) as usize).min(anchors.len() - 1)];
            let j = 0.03 * gauss(&mut rng);
            let j2 = 0.03 * gauss(&mut rng);
            ([st.pos[0] + j, st.pos[1] + j2, st.pos[2] * 0.7], st.teff)
        } else {
            let (p, _, _) = disk_pos(&mut rng, rd, 0.1, r_max, true);
            (p, 0.0)
        };
        let pos = nuclear_pos(&mut rng, pos, r_clear, r_ring, ring_frac, true);
        // M2 downstream offset: HII ignites after the shock, so anchored
        // knots sit ahead of the lane inside CR, behind it outside.
        let mut pos = pos;
        if hot >= 7300.0 {
            let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt().max(0.05);
            let dphi = 0.1 * (vcirc(r) / r - omp).signum();
            if dphi != 0.0 {
                let (s, c) = dphi.sin_cos();
                let (x, y) = (pos[0], pos[1]);
                pos[0] = x * c - y * s;
                pos[1] = x * s + y * c;
            }
        }
        let kind = if hot >= 33000.0 {
            1.0
        } else if rng.unit() < 0.1 {
            2.0
        } else {
            0.0
        };
        out.push(Gas {
            pos,
            vel: gas_vel(&mut rng, pos, vcirc, 0.04),
            size: (0.015 + 0.035 * rng.unit().powi(2)) as f32,
            kind,
            seed: rng.unit() as f32,
            flash: 0.0,
        });
    }
    for _ in 0..nl {
        let (pos, _, _) = arm_lane_phi(rng, rd, lane_r_min, r_max);
        let pos = nuclear_pos(rng, pos, r_clear, r_ring, ring_frac, false);
        out.push(Gas {
            pos,
            vel: gas_vel(rng, pos, vcirc, 0.02),
            size: (0.03 + 0.07 * rng.unit()) as f32,
            kind: 3.0,
            seed: rng.unit() as f32,
            flash: 0.0,
        });
    }
    out
}

pub fn pattern_speed(preset: Preset) -> f32 {
    match preset {
        Preset::Spiral => 0.35,
        Preset::BlackHole => 0.5,
        Preset::Merger => 0.2,
        Preset::Cluster => 0.0,
    }
}

/// Frozen disk+bulge curve for M1 gas advection: total circular v² minus
/// the live components (analytic halo + BH point masses), log-sampled.
/// Gas feels live halo/BHs plus this table; zero back-reaction on stars.
pub struct GasTable {
    pub tab: Vec<f32>,
    pub rmin: f64,
    pub rmax: f64,
    /// 1/ln(rmax/rmin), precomputed: `v2` is per-gas per-step hot.
    pub inv_span: f64,
}

impl GasTable {
    pub fn empty() -> Self {
        Self {
            tab: Vec::new(),
            rmin: 1.0,
            rmax: 1.0,
            inv_span: 0.0,
        }
    }

    pub fn v2(&self, r: f64) -> f64 {
        if self.tab.is_empty() {
            return 0.0;
        }
        let n = self.tab.len();
        let f = (r.clamp(self.rmin, self.rmax) / self.rmin).ln() * self.inv_span * (n - 1) as f64;
        let f = f.clamp(0.0, (n - 1) as f64);
        let i = (f as usize).min(n - 2);
        let t = f - i as f64;
        self.tab[i] as f64 * (1.0 - t) + self.tab[i + 1] as f64 * t
    }
}

pub fn gas_disk_table(preset: Preset) -> GasTable {
    const N: usize = 96;
    const RMIN: f64 = 0.02;
    let (total, halo, mbh, rmax): (Box<dyn Fn(f64) -> f64>, Option<Halo>, f64, f64) = match preset {
        Preset::Spiral => {
            let spec = spiral_spec();
            (
                Box::new(move |r| vc2(&spec, r)),
                Some(Halo { m: 3.0, a: 3.0 }),
                0.02,
                6.0,
            )
        }
        Preset::BlackHole => {
            let h = Halo { m: 1.5, a: 2.0 };
            (
                Box::new(move |r| bh_disk_v2(0.12, 0.78, &h, r)),
                Some(Halo { m: 1.5, a: 2.0 }),
                0.12,
                4.5,
            )
        }
        Preset::Merger => {
            // Table total must include the halo: the old halo=None spec made
            // total-live = disk+bulge-halo, negative at large r, clamped to 0
            // (gas lost all disk gravity outward). Stars keep the halo-free
            // spec for their velocities (golden init); only the table changes.
            let spec = DiskSpec {
                rd: 0.8,
                m_disk: 0.34,
                m_bulge: 0.15,
                m_bh: 0.01,
                bulge_a: 0.2,
                r_max: 3.0,
                halo: Some(Halo { m: 6.0, a: 5.0 }),
            };
            (
                Box::new(move |r| vc2(&spec, r)),
                Some(Halo { m: 6.0, a: 5.0 }),
                0.01,
                7.0,
            )
        }
        Preset::Cluster => return GasTable::empty(),
    };
    let mut tab = Vec::with_capacity(N);
    for i in 0..N {
        let r = RMIN * (rmax / RMIN).powf(i as f64 / (N - 1) as f64);
        let live = halo.map(|h| halo_v2(&h, r)).unwrap_or(0.0) + mbh / r.max(1e-6);
        tab.push((total(r) - live).max(0.0) as f32);
    }
    GasTable {
        tab,
        rmin: RMIN,
        rmax,
        inv_span: 1.0 / (rmax / RMIN).ln(),
    }
}

// NOTE: no standalone `generate_gas` helper. Gas is built inside `generate`
// from the same RNG stream (single source of truth); tests use the gas that
// `generate` returns. A separate helper with its own seed/params diverged
// silently in the past (different lane widths), so it was deleted.

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(s: &[Star]) -> (f64, [f64; 3], [f64; 3], f64) {
        let mut m = 0.0;
        let mut p = [0.0; 3];
        let mut v = [0.0; 3];
        let mut lz = 0.0;
        for st in s {
            m += st.mass;
            for d in 0..3 {
                p[d] += st.mass * st.pos[d];
                v[d] += st.mass * st.vel[d];
            }
            lz += st.mass * (st.pos[0] * st.vel[1] - st.pos[1] * st.vel[0]);
            for d in 0..3 {
                assert!(st.pos[d].is_finite() && st.vel[d].is_finite());
            }
            assert!((0.0..=1.0).contains(&st.temp));
            if st.teff == 0.0 {
                assert_eq!(st.lum, 0.0);
            } else {
                assert!((0.03..=1.0).contains(&st.mag));
                assert!((2000.0..=56000.0).contains(&st.teff), "teff={}", st.teff);
                assert!((0.0004..=2.0e6).contains(&st.lum), "lum={}", st.lum);
            }
        }
        (m, p, v, lz)
    }

    #[test]
    fn presets_conserve_mass_and_com() {
        for preset in [
            Preset::Cluster,
            Preset::Spiral,
            Preset::Merger,
            Preset::BlackHole,
        ] {
            let (s, _, _) = generate(preset, 512, 11);
            assert_eq!(s.len(), 512);
            let (m, p, v, _) = stats(&s);
            assert!((m - 1.0).abs() < 1e-9, "{preset:?} mass {m}");
            for d in 0..3 {
                assert!(p[d].abs() < 1e-9, "{preset:?} com {p:?}");
                assert!(v[d].abs() < 1e-9, "{preset:?} mom {v:?}");
            }
        }
    }

    #[test]
    fn spiral_20k_fmm_stays_finite() {
        let mut sim = crate::sim::Sim::new(20_000, 0.01, true, None, Preset::Spiral);
        for step in 0..6 {
            sim.step();
            for i in 0..sim.n() {
                for d in 0..3 {
                    assert!(sim.pos[i][d].is_finite(), "step {step} body {i}");
                }
            }
        }
    }

    #[test]
    fn spiral_100k_tree_has_no_overfull_leaves() {
        let (s, _, _) = generate(Preset::Spiral, 100_000, 42);
        let parts: Vec<fmm::models::Particle> = s
            .iter()
            .map(|st| fmm::models::Particle {
                mass: st.mass,
                pos: st.pos,
            })
            .collect();
        let tree = fmm::tree::Octree::build(&parts, 64);
        let mut maxc = 0;
        let mut over = 0;
        for (c, cell) in tree.cells.iter().enumerate() {
            if tree.is_leaf(c) {
                maxc = maxc.max(cell.count);
                if cell.count > 64 {
                    over += 1;
                }
            }
        }
        assert_eq!(over, 0, "max leaf count={maxc}");
    }

    #[test]
    fn halo_accel_matches_analytic() {
        let h = Halo { m: 3.0, a: 3.0 };
        let a = halo_accel(&h, &[3.0, 0.0, 0.0]);
        assert!((a[0] + 1.0 / 12.0).abs() < 1e-12, "{a:?}");
        assert_eq!(&a[1..], &[0.0, 0.0]);
        assert!((halo_v2(&h, 3.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn spiral_rotation_curve_flat() {
        let spec = spiral_spec();
        let v1 = vc2(&spec, 1.0).sqrt();
        let v4 = vc2(&spec, 4.0).sqrt();
        assert!(v4 / v1 > 0.85, "v1={v1} v4={v4}");
        // halo carries ~half the outer curve (no-halo decline would be ~0.7)
        let h = spec.halo.unwrap();
        assert!(halo_v2(&h, 4.0) > 0.2);
    }

    #[test]
    fn presets_deterministic() {
        for preset in [
            Preset::Cluster,
            Preset::Spiral,
            Preset::Merger,
            Preset::BlackHole,
        ] {
            let (a, _, _) = generate(preset, 128, 7);
            let (b, _, _) = generate(preset, 128, 7);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.pos, y.pos);
                assert_eq!(x.vel, y.vel);
                assert_eq!(x.mass, y.mass);
                assert_eq!(x.temp, y.temp);
                assert_eq!(x.mag, y.mag);
                assert_eq!(x.teff, y.teff);
                assert_eq!(x.lum, y.lum);
            }
        }
    }

    #[test]
    fn spiral_rotates_and_merger_has_two_holes() {
        let (s, _, _) = generate(Preset::Spiral, 1024, 3);
        let (_, _, _, lz) = stats(&s);
        assert!(lz > 0.2, "lz={lz}");
        let (m, _, _) = generate(Preset::Merger, 1024, 3);
        let mean = 1.0 / 1024.0;
        let heavies = m.iter().filter(|st| st.mass > 5.0 * mean).count();
        assert_eq!(heavies, 2, "two BHs");
        let (b, _, _) = generate(Preset::BlackHole, 512, 3);
        let maxm = b.iter().map(|st| st.mass).fold(0.0, f64::max);
        assert!(maxm > 0.1, "maxm={maxm}");
    }

    #[test]
    fn young_abundance_matches_cosmic() {
        let mut rng = Rng::new(11);
        let n = 20000;
        let mut c = [0usize; 8];
        for _ in 0..n {
            let k = sample_class(&mut rng, true) as usize;
            c[k] += 1;
        }
        let f = |i: usize| c[i] as f64 / n as f64;
        assert!((f(6) - 0.76).abs() < 0.025, "M={}", f(6));
        assert!((f(5) - 0.12).abs() < 0.02, "K={}", f(5));
        assert!((f(4) - 0.076).abs() < 0.015, "G={}", f(4));
        assert!((f(3) - 0.03).abs() < 0.01, "F={}", f(3));
        assert!((f(2) - 0.0061).abs() < 0.005, "A={}", f(2));
        assert!(f(1) < 0.004, "B={}", f(1));
        assert!(c[0] <= 3, "O={}", c[0]);
        assert_eq!(c[7], 0, "no giants in young field");
    }

    #[test]
    fn old_population_has_giants_no_o() {
        let mut rng = Rng::new(12);
        let n = 20000;
        let mut c = [0usize; 8];
        for _ in 0..n {
            let k = sample_class(&mut rng, false) as usize;
            c[k] += 1;
        }
        let f = |i: usize| c[i] as f64 / n as f64;
        assert_eq!(c[0], 0, "no O in old population");
        assert!((f(7) - 0.03).abs() < 0.01, "giants={}", f(7));
        assert!(f(6) > 0.68 && f(6) < 0.78, "M={}", f(6));
    }

    #[test]
    fn looks_rank_matched_to_mass() {
        let mut rng = Rng::new(13);
        let w = spectrum(&mut rng, 400);
        let looks = assign_looks(&mut rng, &w, true);
        let mut idx: Vec<usize> = (0..400).collect();
        idx.sort_unstable_by(|&a, &b| w[a].total_cmp(&w[b]));
        let lo: f64 = idx[..200].iter().map(|&i| looks[i].3 as f64).sum::<f64>() / 200.0;
        let hi: f64 = idx[200..].iter().map(|&i| looks[i].3 as f64).sum::<f64>() / 200.0;
        assert!(hi > lo * 2.0, "lo={lo} hi={hi}");
    }

    #[test]
    fn spiral_heroes_rare_and_teff_mapped() {
        let (s, _, _) = generate(Preset::Spiral, 20000, 42);
        let stars: Vec<&Star> = s.iter().filter(|st| st.teff > 0.0).collect();
        let heroes = stars.iter().filter(|st| st.mag > 0.85).count() as f64 / stars.len() as f64;
        assert!(heroes < 0.015, "heroes={heroes}");
        let blue = stars.iter().filter(|st| st.temp < 0.2).count() as f64 / stars.len() as f64;
        assert!(blue < 0.02, "blue={blue}");
        let mfrac = stars.iter().filter(|st| st.teff < 3900.0).count() as f64 / stars.len() as f64;
        assert!(mfrac > 0.55 && mfrac < 0.85, "mfrac={mfrac}");
        for st in stars {
            let t =
                1.0 - ((st.teff as f64).ln() - 2300.0f64.ln()) / (50000.0f64.ln() - 2300.0f64.ln());
            assert!(
                (st.temp as f64 - t.clamp(0.02, 1.0)).abs() < 1e-5,
                "t={}",
                st.temp
            );
        }
    }

    #[test]
    fn gas_counts_and_kinds() {
        // Knot/lane totals are structural: nk=300+disk_n/100, nl=200+disk_n/150
        // with disk_n=(2000-1)*70/100=1399 here (gas anchors on disk stars).
        let (_, g, _) = generate(Preset::Spiral, 2000, 42);
        assert_eq!(g.iter().filter(|x| x.kind < 2.5).count(), 313);
        assert_eq!(g.iter().filter(|x| x.kind > 2.5).count(), 209);
        let (_, gc, _) = generate(Preset::Cluster, 512, 42);
        assert!(gc.is_empty());
        // Merger halves build lanes too (kind 3 dust); just require sane kinds.
        let (_, gm, _) = generate(Preset::Merger, 512, 42);
        assert!(!gm.is_empty());
        assert!(
            gm.iter().all(|x| [0.0, 1.0, 2.0, 3.0].contains(&x.kind)),
            "bad kind"
        );
        for x in &g {
            assert!(x.pos.iter().all(|v| v.is_finite()));
            assert!((0.014..=0.11).contains(&x.size), "size={}", x.size);
            assert!((0.0..=1.0).contains(&x.seed));
        }
    }

    #[test]
    fn gas_lanes_hug_arm_phase() {
        let (_, g, _) = generate(Preset::Spiral, 512, 42);
        let mut worst = 0.0f64;
        let mut n = 0;
        for x in g.iter().filter(|x| x.kind > 2.5) {
            let r = (x.pos[0] * x.pos[0] + x.pos[1] * x.pos[1]).sqrt();
            let phi = x.pos[1].atan2(x.pos[0]);
            let psi = 2.0 * phi - 3.0 * (r + 0.15).ln();
            let mut d = (psi + 0.55) % std::f64::consts::PI;
            if d < 0.0 {
                d += std::f64::consts::PI;
            }
            d = d.min(std::f64::consts::PI - d);
            worst = worst.max(d);
            n += 1;
        }
        assert!(n > 100);
        assert!(worst < 0.85, "worst={worst}");
    }

    #[test]
    fn cavity_empty_and_ring_present() {
        let (_, g, _) = generate(Preset::Spiral, 2000, 42);
        let r = |x: &Gas| (x.pos[0] * x.pos[0] + x.pos[1] * x.pos[1]).sqrt();
        // Leaky cavity: ~15% of inner knots stay as a tenuous hot component
        // (lanes spiral inward, no void) — but never a crowd.
        let inside = g.iter().filter(|x| r(x) < 0.07).count();
        assert!(
            (inside as f64) < 0.02 * g.len() as f64,
            "inside={inside} n={}",
            g.len()
        );
        let ring = g.iter().filter(|x| (0.25..=0.35).contains(&r(x))).count() as f64
            / (std::f64::consts::PI * (0.35 * 0.35 - 0.25 * 0.25));
        let ctrl = g.iter().filter(|x| (0.50..=0.70).contains(&r(x))).count() as f64
            / (std::f64::consts::PI * (0.70 * 0.70 - 0.50 * 0.50));
        // Feathered (wide + spiral-modulated) so the bar is softer than the
        // old perfect-circle pile-up, still a clear overdensity.
        assert!(ring > 1.5 * ctrl, "ring={ring} ctrl={ctrl}");
        // Fed, not moated: the inner feed band holds a fair share of the
        // ring density (detached annulus would read ~0 here).
        let fed = g.iter().filter(|x| (0.07..0.20).contains(&r(x))).count() as f64
            / (std::f64::consts::PI * (0.20 * 0.20 - 0.07 * 0.07));
        assert!(fed > 0.3 * ring, "fed={fed} ring={ring}");
    }

    #[test]
    fn hii_sits_downstream_of_lane() {
        let (_, g, _) = generate(Preset::Spiral, 2000, 42);
        let mut sum = 0.0;
        let mut n = 0;
        for x in g.iter().filter(|x| x.kind < 1.5) {
            let r = (x.pos[0] * x.pos[0] + x.pos[1] * x.pos[1]).sqrt();
            if !(0.8..=2.6).contains(&r) {
                continue;
            }
            let psi = 2.0 * x.pos[1].atan2(x.pos[0]) - 3.0 * (r + 0.15).ln();
            // Fold to [-π/2, π/2] centered on the lane: downstream reads +.
            let mut d = (psi + 0.55 + std::f64::consts::PI / 2.0) % std::f64::consts::PI;
            if d < 0.0 {
                d += std::f64::consts::PI;
            }
            d -= std::f64::consts::PI / 2.0;
            sum += d;
            n += 1;
        }
        assert!(n > 50, "n={n}");
        assert!(sum / n as f64 > 0.03, "mean={}", sum / n as f64);
    }

    #[test]
    fn gas_init_near_circular() {
        let spec = spiral_spec();
        let (_, g, _) = generate(Preset::Spiral, 512, 42);
        assert!(!g.is_empty());
        for x in &g {
            let r = (x.pos[0] * x.pos[0] + x.pos[1] * x.pos[1]).sqrt().max(0.05);
            let sp = (x.vel[0] * x.vel[0] + x.vel[1] * x.vel[1] + x.vel[2] * x.vel[2]).sqrt();
            let vc = vc2(&spec, r).sqrt();
            assert!((0.7..=1.35).contains(&(sp / vc)), "r={r} sp={sp} vc={vc}");
            let lz = x.pos[0] * x.vel[1] - x.pos[1] * x.vel[0];
            assert!(lz > 0.0, "prograde lz={lz}");
        }
    }

    #[test]
    fn gas_table_matches_curve() {
        let t = gas_disk_table(Preset::Spiral);
        let spec = spiral_spec();
        for r in [0.05, 0.3, 1.0, 2.5, 4.0] {
            let want = (vc2(&spec, r) - halo_v2(&Halo { m: 3.0, a: 3.0 }, r) - 0.02 / r).max(0.0);
            assert!(
                (t.v2(r) - want).abs() < 1e-4,
                "r={r} got={} want={want}",
                t.v2(r)
            );
        }
        assert_eq!(gas_disk_table(Preset::Cluster).v2(1.0), 0.0);
    }

    #[test]
    fn gas_deterministic() {
        let (_, ga, _) = generate(Preset::Spiral, 256, 7);
        let (_, gb, _) = generate(Preset::Spiral, 256, 7);
        assert_eq!(ga.len(), gb.len());
        for (x, y) in ga.iter().zip(gb.iter()) {
            assert_eq!(x.pos, y.pos);
            assert_eq!(x.vel, y.vel);
            assert_eq!(x.kind, y.kind);
            assert_eq!(x.size, y.size);
            assert_eq!(x.seed, y.seed);
        }
    }

    #[test]
    fn merger_gas_rides_along() {
        let (s, g, _) = generate(Preset::Merger, 1024, 3);
        assert!(!g.is_empty());
        let mean_ax = g.iter().map(|x| x.pos[0].abs()).sum::<f64>() / g.len() as f64;
        assert!(mean_ax > 1.0, "mean|x|={mean_ax}");
        assert_eq!(s.len(), 1024);
    }

    #[test]
    fn spiral_generate_returns_gas() {
        let (_, g, _) = generate(Preset::Spiral, 2000, 42);
        assert_eq!(g.iter().filter(|x| x.kind < 2.5).count(), 313);
        assert_eq!(g.iter().filter(|x| x.kind > 2.5).count(), 209);
        let (_, gc, _) = generate(Preset::Cluster, 256, 42);
        assert!(gc.is_empty());
        assert!((pattern_speed(Preset::Spiral) - 0.35).abs() < 1e-6);
        assert_eq!(pattern_speed(Preset::Cluster), 0.0);
    }

    /// 1M init gate (run explicitly in release):
    /// `cargo test --release -p fmms-visual -- --ignored spiral_1m`.
    /// Guards the end goal: generate must stay O(n log n)-small, mass exact,
    /// gas still O(1)-capped.
    #[test]
    #[ignore]
    fn spiral_1m_generate_sane() {
        let t = std::time::Instant::now();
        let (s, g, _) = generate(Preset::Spiral, 1_000_000, 42);
        let dt = t.elapsed();
        assert_eq!(s.len(), 1_000_000);
        let m: f64 = s.iter().map(|st| st.mass).sum();
        assert!((m - 1.0).abs() < 1e-6, "mass {m}");
        assert!(!g.is_empty() && g.len() <= 1800, "gas {}", g.len());
        eprintln!("1M spiral init: {dt:?}");
    }
}
