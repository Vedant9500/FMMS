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

/// Mass-mapped stellar look: heavy = blue-white bright, light = red dim.
/// Heroes (diffraction spikes) kept to ~0.4% so cores don't white out.
fn look(rng: &mut Rng, w: f64, wmin: f64, wmax: f64, temp_shift: f64, mag_scale: f64) -> (f32, f32) {
    let q = ((w - wmin) / (wmax - wmin).max(1e-18)).clamp(0.0, 1.0);
    let temp = (0.92 - 0.75 * q + temp_shift + 0.06 * gauss(rng)).clamp(0.02, 1.0);
    let mut mag = 0.06 + 0.50 * q.powf(1.6);
    if rng.unit() > 0.996 {
        mag = 0.85 + 0.15 * rng.unit();
    }
    (temp as f32, (mag * mag_scale).min(1.0) as f32)
}

fn minmax(xs: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in xs {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (lo, hi)
}

/// Static analytic dark halo (Hernquist profile): the standard stand-in
/// for live halo particles (cf. Sellwood's static-background disk studies).
/// Real spirals sit in ~5-10x halos — flat Rubin curves need Menc growing
/// ~linearly, which the visible disk alone cannot supply.
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
    temp_disk: f64,
    temp_bulge: f64,
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

fn spiral_with(n: usize, rng: &mut Rng, spec: &DiskSpec) -> Vec<Star> {
    let bh_n = if n > 8 { 1 } else { 0 };
    let disk_n = (n - bh_n) * 70 / 100;
    let bulge_n = n - bh_n - disk_n;
    let w = spectrum(rng, disk_n + bulge_n);
    let (wmin, wmax) = minmax(&w);
    let total = spec.m_disk + spec.m_bulge;
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
        let mut q = (w[i] - wmin) / (wmax - wmin).max(1e-18);
        q = q.clamp(0.0, 1.0);
        let (mut temp, mut mag) = look(rng, w[i], wmin, wmax, spec.temp_disk, 1.15);
        if q < 0.35 && rng.unit() < 0.15 {
            temp = (0.03 + 0.07 * rng.unit()) as f32;
            mag = (mag * 1.6).min(0.8);
        }
        out.push(Star { pos, vel, mass: w[i] * total, temp, mag });
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
        let (temp, mag) = look(rng, w[disk_n + i], wmin, wmax, spec.temp_bulge, 0.9);
        out.push(Star { pos, vel, mass: w[disk_n + i] * total, temp, mag });
    }
    if bh_n == 1 {
        out.push(Star { pos: [0.0; 3], vel: [0.0; 3], mass: spec.m_bh, temp: 0.5, mag: 0.02 });
    }
    out
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
        temp_disk: -0.12,
        temp_bulge: 0.12,
        halo: Some(Halo { m: 3.0, a: 3.0 }),
    }
}

pub fn generate(preset: Preset, n: usize, seed: u64) -> (Vec<Star>, Option<Halo>) {
    let mut rng = Rng::new(seed);
    let halo = match preset {
        Preset::Cluster => None,
        Preset::Spiral => Some(Halo { m: 3.0, a: 3.0 }),
        Preset::BlackHole => Some(Halo { m: 1.5, a: 2.0 }),
        Preset::Merger => Some(Halo { m: 6.0, a: 5.0 }),
    };
    let mut s: Vec<Star> = match preset {
        Preset::Cluster => {
            let parts = fmm::models::plummer(n, [0.0; 3], seed);
            let w = spectrum(&mut rng, n);
            let (wmin, wmax) = minmax(&w);
            parts
                .iter()
                .zip(w.iter())
                .map(|(p, &m)| {
                    let r2 = p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2];
                    let sig = (1.0 / (6.0 * (1.0 + r2).sqrt())).sqrt();
                    let (temp, mag) = look(&mut rng, m, wmin, wmax, 0.0, 1.0);
                    Star {
                        pos: p.pos,
                        vel: [sig * gauss(&mut rng), sig * gauss(&mut rng), sig * gauss(&mut rng)],
                        mass: m,
                        temp,
                        mag,
                    }
                })
                .collect()
        }
        Preset::Spiral => spiral_with(n, &mut rng, &spiral_spec()),
        Preset::BlackHole => {
            let bh_halo = Halo { m: 1.5, a: 2.0 };
            let bh_n = if n > 8 { 1 } else { 0 };
            let disk_n = (n - bh_n) * 85 / 100;
            let halo_n = n - bh_n - disk_n;
            let (m_bh, m_disk, m_halo) = (0.12, 0.78, 0.10);
            let w = spectrum(&mut rng, disk_n + halo_n);
            let (wmin, wmax) = minmax(&w);
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
                let (mut temp, mut mag) = look(&mut rng, w[i], wmin, wmax, -0.20, 1.15);
                if r < 0.35 {
                    temp = (0.02 + 0.08 * rng.unit()) as f32;
                    mag = (mag * 1.5).min(0.9);
                }
                out.push(Star { pos, vel, mass: w[i] * (m_disk + m_halo), temp, mag });
            }
            for i in 0..halo_n {
                let s = rng.unit().sqrt().max(1e-9);
                let r = (0.8 * s / (1.0 - s)).min(4.0).max(0.05);
                let z = 2.0 * rng.unit() - 1.0;
                let phi = 2.0 * std::f64::consts::PI * rng.unit();
                let sp = (1.0 - z * z).sqrt().max(1e-9);
                let sb = (0.4 * (m_bh / r.max(0.1)).sqrt()).min(1.5);
                out.push(Star {
                    pos: [r * sp * phi.cos(), r * sp * phi.sin(), r * z],
                    vel: [sb * gauss(&mut rng), sb * gauss(&mut rng), sb * gauss(&mut rng)],
                    mass: w[disk_n + i] * (m_disk + m_halo),
                    temp: look(&mut rng, w[disk_n + i], wmin, wmax, 0.1, 0.8).0,
                    mag: look(&mut rng, w[disk_n + i], wmin, wmax, 0.1, 0.8).1,
                });
            }
            if bh_n == 1 {
                out.push(Star { pos: [0.0; 3], vel: [0.0; 3], mass: m_bh, temp: 0.5, mag: 0.02 });
            }
            out
        }
        Preset::Merger => {
            let spec = DiskSpec {
                rd: 0.8,
                m_disk: 0.34,
                m_bulge: 0.15,
                m_bh: 0.01,
                bulge_a: 0.2,
                r_max: 3.0,
                temp_disk: -0.12,
                temp_bulge: 0.12,
                halo: None,
            };
            let mut a = spiral_with(n / 2, &mut rng, &spec);
            let mut b = spiral_with(n - n / 2, &mut rng, &spec);
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
            a.extend(b);
            a
        }
    };
    // spec masses already sum to ~1 (merger halves two ~0.5s); renormalize exact.
    let tot: f64 = s.iter().map(|st| st.mass).sum::<f64>().max(1e-18);
    for st in s.iter_mut() {
        st.mass /= tot;
    }
    zero_com(&mut s);
    (s, halo)
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
            assert!((0.0..=1.0).contains(&st.mag));
        }
        (m, p, v, lz)
    }

    #[test]
    fn presets_conserve_mass_and_com() {
        for preset in [Preset::Cluster, Preset::Spiral, Preset::Merger, Preset::BlackHole] {
            let (s, _) = generate(preset, 512, 11);
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
        let (s, _) = generate(Preset::Spiral, 100_000, 42);
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
            let (a, _) = generate(preset, 128, 7);
            let (b, _) = generate(preset, 128, 7);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.pos, y.pos);
                assert_eq!(x.mass, y.mass);
            }
        }
    }

    #[test]
    fn spiral_rotates_and_merger_has_two_holes() {
        let (s, _) = generate(Preset::Spiral, 1024, 3);
        let (_, _, _, lz) = stats(&s);
        assert!(lz > 0.2, "lz={lz}");
        let (m, _) = generate(Preset::Merger, 1024, 3);
        let mean = 1.0 / 1024.0;
        let heavies = m.iter().filter(|st| st.mass > 5.0 * mean).count();
        assert_eq!(heavies, 2, "two BHs");
        let (b, _) = generate(Preset::BlackHole, 512, 3);
        let maxm = b.iter().map(|st| st.mass).fold(0.0, f64::max);
        assert!(maxm > 0.1, "maxm={maxm}");
    }
}
