//! Initial-condition presets: structured galaxies, not a bare Plummer blob.
//! Units G = M_tot = 1. Every generator returns zero-COM, zero-momentum state.

use fmm::models::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Cluster,
    Spiral,
    Merger,
    BlackHole,
}

impl Preset {
    pub fn from_str(s: &str) -> Option<Preset> {
        match s.to_lowercase().as_str() {
            "cluster" => Some(Preset::Cluster),
            "spiral" => Some(Preset::Spiral),
            "merger" => Some(Preset::Merger),
            "bh" | "blackhole" | "black-hole" => Some(Preset::BlackHole),
            _ => None,
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
    if young {
        if u < 0.0000003 {
            SpecClass::O
        } else if u < 0.0012003 {
            SpecClass::B
        } else if u < 0.0073003 {
            SpecClass::A
        } else if u < 0.0373003 {
            SpecClass::F
        } else if u < 0.1133003 {
            SpecClass::G
        } else if u < 0.2333003 {
            SpecClass::K
        } else {
            SpecClass::M
        }
    } else if u < 0.0002 {
        SpecClass::B
    } else if u < 0.0012 {
        SpecClass::A
    } else if u < 0.0112 {
        SpecClass::F
    } else if u < 0.0912 {
        SpecClass::G
    } else if u < 0.2412 {
        SpecClass::K
    } else if u < 0.2712 {
        SpecClass::Giant
    } else {
        SpecClass::M
    }
}

fn stellar(rng: &mut Rng, class: SpecClass) -> (f32, f32, f32, f32) {
    let (tlo, thi) = class.teff_range();
    let (llo, lhi) = class.lum_range();
    let u = rng.unit();
    let teff = (thi.ln() + (tlo.ln() - thi.ln()) * u + 0.02 * gauss(rng)).exp();
    let lum = (lhi.ln() + (llo.ln() - lhi.ln()) * u + 0.1 * gauss(rng))
        .exp()
        .clamp(llo * 0.5, lhi * 2.0);
    let t = 1.0 - (teff.ln() - 2300.0f64.ln()) / (50000.0f64.ln() - 2300.0f64.ln());
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
    Some(if teff >= 33000.0 {
        0
    } else if teff >= 10000.0 {
        1
    } else if teff >= 7300.0 {
        2
    } else if teff >= 6000.0 {
        3
    } else if teff >= 5300.0 {
        4
    } else if teff >= 3900.0 {
        5
    } else {
        6
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
    let looks: Vec<(f32, f32, f32, f32)> = (0..n)
        .map(|_| {
            let c = sample_class(rng, young);
            stellar(rng, c)
        })
        .collect();
    let mut mi: Vec<usize> = (0..n).collect();
    mi.sort_by(|&a, &b| masses[a].partial_cmp(&masses[b]).unwrap());
    let mut li: Vec<usize> = (0..n).collect();
    li.sort_by(|&a, &b| looks[a].3.partial_cmp(&looks[b].3).unwrap());
    let mut out = vec![(0.0f32, 0.0f32, 0.0f32, 0.0f32); n];
    for k in 0..n {
        out[mi[k]] = looks[li[k]];
    }
    out
}

fn gauss(rng: &mut Rng) -> f64 {
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
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt().max(1e-9);
    let s = -h.m / (r * (r + h.a) * (r + h.a));
    [s * pos[0], s * pos[1], s * pos[2]]
}

/// Halo contribution to v_circ^2: M r / (r + a)^2.
pub fn halo_v2(h: &Halo, r: f64) -> f64 {
    h.m * r / ((r + h.a) * (r + h.a))
}

fn zero_com(s: &mut [Star]) {
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
    for st in s.iter_mut() {
        for d in 0..3 {
            st.pos[d] -= mp[d] / mtot;
            st.vel[d] -= mv[d] / mtot;
        }
    }
}

fn disk_pos(rng: &mut Rng, rd: f64, r_min: f64, r_max: f64, arms: bool) -> ([f64; 3], f64, f64) {
    let mut r = r_min;
    let mut phi = 0.0;
    for _ in 0..24 {
        r = r_min - rd * (1.0 - rng.unit()).ln();
        if r > r_max {
            continue;
        }
        phi = 2.0 * std::f64::consts::PI * rng.unit();
        if !arms {
            break;
        }
        let arm = (2.0 * phi - 3.0 * (r + 0.15).ln()).cos().powi(2);
        if rng.unit() < 0.40 + 0.60 * arm {
            break;
        }
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

fn spiral_with(n: usize, rng: &mut Rng, spec: &DiskSpec) -> (Vec<Star>, Vec<Gas>) {
    let bh_n = if n > 8 { 1 } else { 0 };
    let disk_n = (n - bh_n) * 70 / 100;
    let bulge_n = n - bh_n - disk_n;
    let w = spectrum(rng, disk_n + bulge_n);
    let total = spec.m_disk + spec.m_bulge;
    let disk_looks = assign_looks(rng, &w[..disk_n], true);
    let bulge_looks = assign_looks(rng, &w[disk_n..], false);
    let mut out = Vec::with_capacity(n);
    for i in 0..disk_n {
        let (pos, r, phi) = disk_pos(rng, spec.rd, 0.05, spec.r_max, true);
        let vc = vc2(spec, r).sqrt().min(3.0);
        let sv = 0.07 * vc;
        let tx = -phi.sin();
        let ty = phi.cos();
        let vel = [
            tx * vc + sv * gauss(rng),
            ty * vc + sv * gauss(rng),
            sv * gauss(rng),
        ];
        let (temp, mag, teff, lum) = disk_looks[i];
        out.push(Star { pos, vel, mass: w[i] * total, temp, mag, teff, lum });
    }
    for i in 0..bulge_n {
        let s = rng.unit().sqrt().max(1e-9);
        let r = (spec.bulge_a * s / (1.0 - s)).min(3.0).max(0.02);
        let z = 2.0 * rng.unit() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * rng.unit();
        let sp = (1.0 - z * z).sqrt().max(1e-9);
        let pos = [r * sp * phi.cos(), r * sp * phi.sin(), r * z];
        let vc = vc2(spec, r).sqrt().min(3.0);
        let sb = (0.35 * vc2(spec, r.max(0.05)).sqrt()).min(1.5);
        let vel = [
            -phi.sin() * 0.3 * vc + sb * gauss(rng),
            phi.cos() * 0.3 * vc + sb * gauss(rng),
            sb * gauss(rng),
        ];
        let (temp, mag, teff, lum) = bulge_looks[i];
        out.push(Star { pos, vel, mass: w[disk_n + i] * total, temp, mag, teff, lum });
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
    let gas = gas_for_disk(rng, disk, spec.rd, spec.r_max, 0.1, true);
    (out, gas)
}

fn rot_xz(p: [f64; 3], cx: f64, cz: f64) -> [f64; 3] {
    let (sx, cx_) = cx.sin_cos();
    let (sz, cz_) = cz.sin_cos();
    let y1 = cx_ * p[1] - sx * p[2];
    let z1 = sx * p[1] + cx_ * p[2];
    [cz_ * p[0] - sz * y1, sz * p[0] + cz_ * y1, z1]
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
            let parts = fmm::models::plummer(n, [0.0; 3], seed);
            let w = spectrum(&mut rng, n);
            let looks = assign_looks(&mut rng, &w, false);
            (
                parts
                    .iter()
                    .zip(w.iter())
                    .zip(looks.iter())
                    .map(|((p, &m), &(temp, mag, teff, lum))| {
                        let r2 = p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2];
                        let sig = (1.0 / (6.0 * (1.0 + r2).sqrt())).sqrt();
                        Star {
                            pos: p.pos,
                            vel: [
                                sig * gauss(&mut rng),
                                sig * gauss(&mut rng),
                                sig * gauss(&mut rng),
                            ],
                            mass: m,
                            temp,
                            mag,
                            teff,
                            lum,
                        }
                    })
                    .collect::<Vec<Star>>(),
                Vec::new(),
            )
        }
        Preset::Spiral => spiral_with(n, &mut rng, &spiral_spec()),
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
                let (pos, r, phi) = disk_pos(&mut rng, 0.6, 0.15, 2.5, true);
                let rr = r.max(1e-6);
                let v2 = (m_bh + m_disk * (1.0 - (1.0 + rr / 0.6) * (-rr / 0.6).exp())) / rr
                    + halo_v2(&bh_halo, rr);
                let vc = v2.sqrt().min(3.0);
                let sv = 0.05 * vc;
                let vel = [
                    -phi.sin() * vc + sv * gauss(&mut rng),
                    phi.cos() * vc + sv * gauss(&mut rng),
                    sv * gauss(&mut rng),
                ];
                let (mut temp, mut mag, mut teff, mut lum) = disk_looks[i];
                if r < 0.35 {
                    temp = (0.02 + 0.08 * rng.unit()) as f32;
                    mag = (mag * 1.5).min(0.9);
                    teff = 12000.0 + 8000.0 * rng.unit() as f32;
                    lum = (lum * 2.0).min(500.0);
                }
                out.push(Star { pos, vel, mass: w[i] * (m_disk + m_halo), temp, mag, teff, lum });
            }
            for i in 0..halo_n {
                let s = rng.unit().sqrt().max(1e-9);
                let r = (0.8 * s / (1.0 - s)).min(4.0).max(0.05);
                let z = 2.0 * rng.unit() - 1.0;
                let phi = 2.0 * std::f64::consts::PI * rng.unit();
                let sp = (1.0 - z * z).sqrt().max(1e-9);
                let sb = (0.4 * (m_bh / r.max(0.1)).sqrt()).min(1.5);
                let (temp, mag, teff, lum) = halo_looks[i];
                out.push(Star {
                    pos: [r * sp * phi.cos(), r * sp * phi.sin(), r * z],
                    vel: [sb * gauss(&mut rng), sb * gauss(&mut rng), sb * gauss(&mut rng)],
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
            let gas = gas_for_disk(&mut rng, &out, 0.6, 2.5, 0.3, true);
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
            let (mut a, mut ga) = spiral_with(n / 2, &mut rng, &spec);
            let (mut b, mut gb) = spiral_with(n - n / 2, &mut rng, &spec);
            for st in a.iter_mut() {
                st.pos = rot_xz(st.pos, 0.3, 0.0);
                st.vel = rot_xz(st.vel, 0.3, 0.0);
                st.pos[0] -= 2.6;
                st.pos[1] += 0.2;
                st.vel[0] += 0.16;
                st.vel[1] += 0.10;
                st.vel[2] += 0.03;
                st.mass *= 0.5;
            }
            for g in ga.iter_mut() {
                g.pos = rot_xz(g.pos, 0.3, 0.0);
                g.pos[0] -= 2.6;
                g.pos[1] += 0.2;
            }
            for st in b.iter_mut() {
                st.pos = rot_xz(st.pos, -1.1, 0.5);
                st.vel = rot_xz(st.vel, -1.1, 0.5);
                st.pos[0] += 2.6;
                st.pos[1] -= 0.2;
                st.vel[0] -= 0.16;
                st.vel[1] -= 0.10;
                st.vel[2] -= 0.03;
                st.mass *= 0.5;
            }
            for g in gb.iter_mut() {
                g.pos = rot_xz(g.pos, -1.1, 0.5);
                g.pos[0] += 2.6;
                g.pos[1] -= 0.2;
            }
            a.extend(b);
            ga.extend(gb);
            (a, ga)
        }
    };
    // spec masses already sum to ~1 (merger halves two ~0.5s); renormalize exact.
    let tot: f64 = s.iter().map(|st| st.mass).sum::<f64>().max(1e-18);
    for st in s.iter_mut() {
        st.mass /= tot;
    }
    zero_com(&mut s);
    (s, gas, halo)
}
#[derive(Clone)]
pub struct Gas {
    pub pos: [f64; 3],
    pub size: f32,
    pub kind: f32,
    pub seed: f32,
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
    let k = if rng.unit() < 0.5 { 0.0 } else { std::f64::consts::PI };
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
) -> Vec<Gas> {
    let n = stars.len();
    let mut out = Vec::new();
    let anchors: Vec<&Star> = stars.iter().filter(|st| st.teff >= 7300.0).collect();
    let nk = (80 + n / 500).min(500);
    for _ in 0..nk {
        let (pos, hot) = if !anchors.is_empty() && rng.unit() < 0.6 {
            let st = anchors[(rng.unit() * anchors.len() as f64) as usize % anchors.len()];
            let j = 0.05 * gauss(&mut rng);
            let j2 = 0.05 * gauss(&mut rng);
            ([st.pos[0] + j, st.pos[1] + j2, st.pos[2] * 0.7], st.teff)
        } else {
            let (p, _, _) = disk_pos(&mut rng, rd, 0.1, r_max, true);
            (p, 0.0)
        };
        let kind = if hot >= 33000.0 {
            1.0
        } else if rng.unit() < 0.1 {
            2.0
        } else {
            0.0
        };
        out.push(Gas {
            pos,
            size: (0.04 + 0.10 * rng.unit().powi(2)) as f32,
            kind,
            seed: rng.unit() as f32,
        });
    }
    if lanes {
        let nl = (150 + n / 250).min(800);
        for _ in 0..nl {
            let (pos, _, _) = arm_lane_phi(&mut rng, rd, lane_r_min, r_max);
            out.push(Gas {
                pos,
                size: (0.08 + 0.20 * rng.unit()) as f32,
                kind: 3.0,
                seed: rng.unit() as f32,
            });
        }
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

#[allow(dead_code)]
pub fn generate_gas(preset: Preset, stars: &[Star], seed: u64) -> Vec<Gas> {
    let mut rng = Rng::new(seed.wrapping_add(7919));
    match preset {
        Preset::Spiral => gas_for_disk(&mut rng, stars, 1.0, 4.0, 0.35, true),
        Preset::BlackHole => gas_for_disk(&mut rng, stars, 0.6, 2.5, 0.3, true),
        Preset::Merger => gas_for_disk(&mut rng, stars, 0.8, 3.0, 0.3, false),
        Preset::Cluster => Vec::new(),
    }
}

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
        for preset in [Preset::Cluster, Preset::Spiral, Preset::Merger, Preset::BlackHole] {
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
            .map(|st| fmm::models::Particle { mass: st.mass, pos: st.pos })
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
        for preset in [Preset::Cluster, Preset::Spiral, Preset::Merger, Preset::BlackHole] {
            let (a, _, _) = generate(preset, 128, 7);
            let (b, _, _) = generate(preset, 128, 7);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.pos, y.pos);
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
        idx.sort_by(|&a, &b| w[a].partial_cmp(&w[b]).unwrap());
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
        let mfrac =
            stars.iter().filter(|st| st.teff < 3900.0).count() as f64 / stars.len() as f64;
        assert!(mfrac > 0.55 && mfrac < 0.85, "mfrac={mfrac}");
        for st in stars {
            let t = 1.0 - ((st.teff as f64).ln() - 2300.0f64.ln())
                / (50000.0f64.ln() - 2300.0f64.ln());
            assert!((st.temp as f64 - t.clamp(0.02, 1.0)).abs() < 1e-5, "t={}", st.temp);
        }
    }

    #[test]
    fn gas_counts_and_kinds() {
        let (s, _, _) = generate(Preset::Spiral, 2000, 42);
        let g = generate_gas(Preset::Spiral, &s, 42);
        assert_eq!(g.iter().filter(|x| x.kind < 2.5).count(), 84);
        assert_eq!(g.iter().filter(|x| x.kind > 2.5).count(), 158);
        let (c, _, _) = generate(Preset::Cluster, 512, 42);
        assert_eq!(generate_gas(Preset::Cluster, &c, 42).len(), 0);
        let (m, _, _) = generate(Preset::Merger, 512, 42);
        let gm = generate_gas(Preset::Merger, &m, 42);
        assert!(gm.iter().all(|x| x.kind < 2.5));
        for x in &g {
            assert!(x.pos.iter().all(|v| v.is_finite()));
            assert!((0.03..=0.55).contains(&x.size), "size={}", x.size);
            assert!((0.0..=1.0).contains(&x.seed));
        }
    }

    #[test]
    fn gas_lanes_hug_arm_phase() {
        let (s, _, _) = generate(Preset::Spiral, 512, 42);
        let g = generate_gas(Preset::Spiral, &s, 42);
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
    fn gas_deterministic() {
        let (a, _, _) = generate(Preset::Spiral, 256, 7);
        let (b, _, _) = generate(Preset::Spiral, 256, 7);
        let ga = generate_gas(Preset::Spiral, &a, 7);
        let gb = generate_gas(Preset::Spiral, &b, 7);
        assert_eq!(ga.len(), gb.len());
        for (x, y) in ga.iter().zip(gb.iter()) {
            assert_eq!(x.pos, y.pos);
            assert_eq!(x.kind, y.kind);
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
        assert_eq!(g.iter().filter(|x| x.kind < 2.5).count(), 82);
        assert_eq!(g.iter().filter(|x| x.kind > 2.5).count(), 155);
        let (_, gc, _) = generate(Preset::Cluster, 256, 42);
        assert!(gc.is_empty());
        assert!((pattern_speed(Preset::Spiral) - 0.35).abs() < 1e-6);
        assert_eq!(pattern_speed(Preset::Cluster), 0.0);
    }
}
