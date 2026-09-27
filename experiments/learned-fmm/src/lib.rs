use fmm::models::{self, Particle, Rng};
use fmm::tree::Octree;
use std::io::{Read, Write};

pub const EMB_DIM: usize = 32;

pub struct LeafFeatures {
    pub count: usize,
    pub com: [f64; 3],
    pub total_mass: f64,
    pub spread: f64,
}

pub struct UpwardInput {
    pub child_embs: Vec<[f64; EMB_DIM]>,
    pub child_centers: Vec<[f64; 3]>,
    pub parent_center: [f64; 3],
}

pub struct InteractionMsg {
    pub emb_u: [f64; EMB_DIM],
    pub emb_v: [f64; EMB_DIM],
    pub rel_pos: [f64; 3],
    pub size_ratio: f64,
}

pub struct DownwardInput {
    pub parent_local: [f64; EMB_DIM],
    pub own_upward: [f64; EMB_DIM],
    pub interaction_sum: [f64; EMB_DIM],
}

pub trait LearnedOperators {
    fn leaf_encode(&self, f: &LeafFeatures) -> [f64; EMB_DIM];
    fn upward(&self, input: &UpwardInput) -> [f64; EMB_DIM];
    fn interaction(&self, m: &InteractionMsg) -> [f64; EMB_DIM];
    fn downward(&self, input: &DownwardInput) -> [f64; EMB_DIM];
    fn decode(&self, p: &Particle, leaf_local: &[f64; EMB_DIM]) -> [f64; 3];
}

pub fn leaf_features(parts: &[Particle], ids: &[usize], center: [f64; 3]) -> LeafFeatures {
    let mut com = [0.0; 3];
    let mut total = 0.0;
    let mut spread = 0.0;
    for &i in ids {
        total += parts[i].mass;
        for d in 0..3 {
            com[d] += parts[i].mass * parts[i].pos[d];
        }
    }
    if total > 0.0 {
        for d in 0..3 {
            com[d] /= total;
        }
    }
    for &i in ids {
        let mut r2 = 0.0;
        for d in 0..3 {
            let dx = parts[i].pos[d] - center[d];
            r2 += dx * dx;
        }
        spread += parts[i].mass * r2;
    }
    if total > 0.0 {
        spread = (spread / total).sqrt();
    }
    LeafFeatures {
        count: ids.len(),
        com,
        total_mass: total,
        spread,
    }
}

pub fn relative_l2(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.len() {
        for d in 0..3 {
            let diff = a[i][d] - b[i][d];
            num += diff * diff;
            den += a[i][d] * a[i][d];
        }
    }
    if den == 0.0 {
        return num.sqrt();
    }
    (num / den).sqrt()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConfigTag {
    Plummer = 0,
    Cube = 1,
    Clumps = 2,
    Mixed = 3,
}

impl ConfigTag {
    pub fn all() -> [ConfigTag; 4] {
        [
            ConfigTag::Plummer,
            ConfigTag::Cube,
            ConfigTag::Clumps,
            ConfigTag::Mixed,
        ]
    }
    pub fn name(self) -> &'static str {
        match self {
            ConfigTag::Plummer => "plummer",
            ConfigTag::Cube => "cube",
            ConfigTag::Clumps => "clumps",
            ConfigTag::Mixed => "mixed",
        }
    }
    pub fn parse(s: &str) -> Option<ConfigTag> {
        match s {
            "plummer" => Some(ConfigTag::Plummer),
            "cube" => Some(ConfigTag::Cube),
            "clumps" => Some(ConfigTag::Clumps),
            "mixed" => Some(ConfigTag::Mixed),
            _ => None,
        }
    }
    fn byte(self) -> u8 {
        self as u8
    }
    fn from_byte(b: u8) -> Option<ConfigTag> {
        match b {
            0 => Some(ConfigTag::Plummer),
            1 => Some(ConfigTag::Cube),
            2 => Some(ConfigTag::Clumps),
            3 => Some(ConfigTag::Mixed),
            _ => None,
        }
    }
}

fn gaussian(rng: &mut Rng) -> f64 {
    let u1 = rng.unit().max(1e-15);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn cube_parts(n: usize, seed: u64) -> Vec<Particle> {
    let mut rng = Rng::new(seed);
    let mass = 1.0 / n as f64;
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        parts.push(Particle {
            mass,
            pos: [
                rng.unit() * 2.0 - 1.0,
                rng.unit() * 2.0 - 1.0,
                rng.unit() * 2.0 - 1.0,
            ],
        });
    }
    parts
}

fn clumps_parts(n: usize, seed: u64) -> Vec<Particle> {
    let mut rng = Rng::new(seed);
    let mass = 1.0 / n as f64;
    let k = 4;
    let mut centers = Vec::with_capacity(k);
    for _ in 0..k {
        centers.push([
            rng.unit() * 1.2 - 0.6,
            rng.unit() * 1.2 - 0.6,
            rng.unit() * 1.2 - 0.6,
        ]);
    }
    let sigma = 0.10;
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[(i * 2654435761) % k];
        parts.push(Particle {
            mass,
            pos: [
                c[0] + sigma * gaussian(&mut rng),
                c[1] + sigma * gaussian(&mut rng),
                c[2] + sigma * gaussian(&mut rng),
            ],
        });
    }
    parts
}

pub fn generate(tag: ConfigTag, n: usize, seed: u64) -> Vec<Particle> {
    assert!(n > 0);
    match tag {
        ConfigTag::Plummer => models::plummer(n, [0.0; 3], seed),
        ConfigTag::Cube => cube_parts(n, seed),
        ConfigTag::Clumps => clumps_parts(n, seed),
        ConfigTag::Mixed => {
            let n1 = n / 2;
            let n2 = n - n1;
            let mut a = models::plummer(n1.max(1), [0.0; 3], seed);
            let mut b = cube_parts(n2.max(1), seed ^ 0x9E3779B97F4A7C15);
            let m = 1.0 / n as f64;
            for p in a.iter_mut().chain(b.iter_mut()) {
                p.mass = m;
            }
            a.extend(b);
            a.truncate(n);
            a
        }
    }
}

pub fn normalize_to_unit_box(parts: &mut [Particle]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in parts.iter() {
        for d in 0..3 {
            lo[d] = lo[d].min(p.pos[d]);
            hi[d] = hi[d].max(p.pos[d]);
        }
    }
    let mut ctr = [0.0f64; 3];
    let mut half = 0.0f64;
    for d in 0..3 {
        ctr[d] = 0.5 * (lo[d] + hi[d]);
        half = half.max(0.5 * (hi[d] - lo[d]));
    }
    half = half.max(f64::MIN_POSITIVE);
    for p in parts.iter_mut() {
        for d in 0..3 {
            p.pos[d] = (p.pos[d] - ctr[d]) / half;
        }
    }
}

pub fn random_rotate(parts: &mut [Particle], seed: u64) {
    let mut rng = Rng::new(seed);
    let u1 = rng.unit();
    let u2 = rng.unit() * 2.0 * std::f64::consts::PI;
    let u3 = rng.unit() * 2.0 * std::f64::consts::PI;
    let (s1, c1) = (u1.sqrt(), (1.0 - u1).sqrt());
    let q = [
        s1 * u2.sin(),
        s1 * u2.cos(),
        c1 * u3.sin(),
        c1 * u3.cos(),
    ];
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    for p in parts.iter_mut() {
        let v = p.pos;
        for d in 0..3 {
            p.pos[d] = r[d][0] * v[0] + r[d][1] * v[1] + r[d][2] * v[2];
        }
    }
}

pub struct SplitStats {
    pub rms_total: f64,
    pub rms_near: f64,
    pub rms_far: f64,
    pub far_frac: f64,
    pub min_r: f64,
}

fn rms(v: &[[f64; 3]]) -> f64 {
    let mut s = 0.0;
    for a in v {
        s += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    }
    (s / v.len() as f64).sqrt()
}

pub fn split_near_far(
    parts: &[Particle],
    tree: &Octree,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, SplitStats) {
    let n = parts.len();
    let leaves: Vec<usize> = (0..tree.cells.len()).filter(|&c| tree.is_leaf(c)).collect();
    let nl = leaves.len();
    let mut leaf_ord = vec![0usize; tree.cells.len()];
    for (o, &c) in leaves.iter().enumerate() {
        leaf_ord[c] = o;
    }
    let mut of_pid = vec![0usize; n];
    for (o, &c) in leaves.iter().enumerate() {
        let cell = &tree.cells[c];
        for k in cell.first..cell.first + cell.count {
            of_pid[tree.order[k] as usize] = o;
        }
    }
    let mut adj = vec![false; nl * nl];
    for a in 0..nl {
        let ca = &tree.cells[leaves[a]];
        for b in 0..nl {
            let cb = &tree.cells[leaves[b]];
            let dx = cb.cen[0] - ca.cen[0];
            let dy = cb.cen[1] - ca.cen[1];
            let dz = cb.cen[2] - ca.cen[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            adj[a * nl + b] = a == b || r <= ca.rho + cb.rho;
        }
    }
    let mut near = vec![[0.0; 3]; n];
    let mut far = vec![[0.0; 3]; n];
    let mut min_r2 = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let pa = &parts[i];
            let pb = &parts[j];
            let dx = pb.pos[0] - pa.pos[0];
            let dy = pb.pos[1] - pa.pos[1];
            let dz = pb.pos[2] - pa.pos[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < min_r2 {
                min_r2 = r2;
            }
            let inv = 1.0 / r2.sqrt();
            let inv3 = inv * inv * inv;
            let fx = inv3 * dx;
            let fy = inv3 * dy;
            let fz = inv3 * dz;
            let is_near = adj[of_pid[i] * nl + of_pid[j]];
            if is_near {
                near[i][0] += pb.mass * fx;
                near[i][1] += pb.mass * fy;
                near[i][2] += pb.mass * fz;
                near[j][0] -= pa.mass * fx;
                near[j][1] -= pa.mass * fy;
                near[j][2] -= pa.mass * fz;
            } else {
                far[i][0] += pb.mass * fx;
                far[i][1] += pb.mass * fy;
                far[i][2] += pb.mass * fz;
                far[j][0] -= pa.mass * fx;
                far[j][1] -= pa.mass * fy;
                far[j][2] -= pa.mass * fz;
            }
        }
    }
    let rms_near = rms(&near);
    let rms_far = rms(&far);
    let mut total = vec![[0.0; 3]; n];
    for i in 0..n {
        for d in 0..3 {
            total[i][d] = near[i][d] + far[i][d];
        }
    }
    let rms_total = rms(&total);
    let stats = SplitStats {
        rms_total,
        rms_near,
        rms_far,
        far_frac: if rms_total > 0.0 {
            rms_far / rms_total
        } else {
            0.0
        },
        min_r: min_r2.sqrt(),
    };
    (near, far, stats)
}

pub struct SnapshotData {
    pub tag: ConfigTag,
    pub pos: Vec<[f32; 3]>,
    pub mass: Vec<f32>,
    pub a_total: Vec<[f32; 3]>,
    pub a_near: Vec<[f32; 3]>,
    pub a_far: Vec<[f32; 3]>,
}

const MAGIC: &[u8; 8] = b"LFMMD001";

fn write_f32s(w: &mut impl Write, xs: &[f32]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf)
}

fn read_f32s(r: &mut impl Read, xs: &mut [f32]) -> std::io::Result<()> {
    let mut buf = vec![0u8; xs.len() * 4];
    r.read_exact(&mut buf)?;
    for (i, x) in xs.iter_mut().enumerate() {
        *x = f32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]);
    }
    Ok(())
}

impl SnapshotData {
    pub fn from_parts(
        tag: ConfigTag,
        parts: &[Particle],
        near: &[[f64; 3]],
        far: &[[f64; 3]],
    ) -> Self {
        let n = parts.len();
        let mut pos = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut a_total = Vec::with_capacity(n);
        let mut a_near = Vec::with_capacity(n);
        let mut a_far = Vec::with_capacity(n);
        for i in 0..n {
            pos.push([parts[i].pos[0] as f32, parts[i].pos[1] as f32, parts[i].pos[2] as f32]);
            mass.push(parts[i].mass as f32);
            a_total.push([
                (near[i][0] + far[i][0]) as f32,
                (near[i][1] + far[i][1]) as f32,
                (near[i][2] + far[i][2]) as f32,
            ]);
            a_near.push([near[i][0] as f32, near[i][1] as f32, near[i][2] as f32]);
            a_far.push([far[i][0] as f32, far[i][1] as f32, far[i][2] as f32]);
        }
        SnapshotData {
            tag,
            pos,
            mass,
            a_total,
            a_near,
            a_far,
        }
    }

    pub fn write_to(&self, path: &std::path::Path) -> std::io::Result<()> {
        let n = self.pos.len() as u32;
        let mut f = std::fs::File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&n.to_le_bytes())?;
        f.write_all(&[self.tag.byte()])?;
        let flat = |v: &[[f32; 3]]| -> Vec<f32> {
            let mut o = Vec::with_capacity(v.len() * 3);
            for a in v {
                o.extend_from_slice(a);
            }
            o
        };
        write_f32s(&mut f, &flat(&self.pos))?;
        write_f32s(&mut f, &self.mass)?;
        write_f32s(&mut f, &flat(&self.a_total))?;
        write_f32s(&mut f, &flat(&self.a_near))?;
        write_f32s(&mut f, &flat(&self.a_far))?;
        Ok(())
    }

    pub fn read_from(path: &std::path::Path) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        assert_eq!(&magic, MAGIC);
        let mut nb = [0u8; 4];
        f.read_exact(&mut nb)?;
        let n = u32::from_le_bytes(nb) as usize;
        let mut tb = [0u8; 1];
        f.read_exact(&mut tb)?;
        let tag = ConfigTag::from_byte(tb[0]).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "bad config tag")
        })?;
        let mut raw = vec![0f32; n * 3];
        let mut get3 = |f: &mut std::fs::File| -> std::io::Result<Vec<[f32; 3]>> {
            read_f32s(f, &mut raw)?;
            Ok(raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
        };
        let pos = get3(&mut f)?;
        let mut mass = vec![0f32; n];
        read_f32s(&mut f, &mut mass)?;
        let a_total = get3(&mut f)?;
        let a_near = get3(&mut f)?;
        let a_far = get3(&mut f)?;
        Ok(SnapshotData {
            tag,
            pos,
            mass,
            a_total,
            a_near,
            a_far,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fmm::direct::direct;

    #[test]
    fn leaf_features_single_particle() {
        let parts = vec![Particle {
            mass: 2.0,
            pos: [1.0, 0.0, 0.0],
        }];
        let f = leaf_features(&parts, &[0], [0.0; 3]);
        assert_eq!(f.count, 1);
        assert!((f.total_mass - 2.0).abs() < 1e-12);
        assert!((f.com[0] - 1.0).abs() < 1e-12);
        assert!((f.spread - 1.0).abs() < 1e-12);
    }

    #[test]
    fn relative_l2_identity() {
        let a = vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        assert!(relative_l2(&a, &a) < 1e-15);
    }

    #[test]
    fn generate_deterministic() {
        for tag in ConfigTag::all() {
            let a = generate(tag, 256, 7);
            let b = generate(tag, 256, 7);
            assert_eq!(a.len(), 256);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.pos, y.pos);
                assert_eq!(x.mass, y.mass);
            }
            let total: f64 = a.iter().map(|p| p.mass).sum();
            assert!((total - 1.0).abs() < 1e-12, "{tag:?} total={total}");
        }
    }

    #[test]
    fn normalize_bounds_and_mass() {
        let mut parts = generate(ConfigTag::Mixed, 512, 3);
        normalize_to_unit_box(&mut parts);
        for p in &parts {
            for d in 0..3 {
                assert!(p.pos[d] >= -1.0 - 1e-9 && p.pos[d] <= 1.0 + 1e-9, "{:?}", p.pos);
            }
        }
        let total: f64 = parts.iter().map(|p| p.mass).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rotation_preserves_radii() {
        let mut parts = generate(ConfigTag::Cube, 256, 11);
        let r0: Vec<f64> = parts
            .iter()
            .map(|p| (p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2]).sqrt())
            .collect();
        random_rotate(&mut parts, 99);
        for (p, &r) in parts.iter().zip(r0.iter()) {
            let r1 = (p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2]).sqrt();
            assert!((r1 - r).abs() < 1e-12, "{r1} vs {r}");
        }
    }

    #[test]
    fn split_sums_to_total() {
        let mut parts = generate(ConfigTag::Plummer, 256, 5);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let (near, far, stats) = split_near_far(&parts, &tree);
        let (a_ref, _) = direct(&parts);
        let total: Vec<[f64; 3]> = near
            .iter()
            .zip(far.iter())
            .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
            .collect();
        assert!(relative_l2(&a_ref, &total) < 1e-12, "rel={}", relative_l2(&a_ref, &total));
        assert!(stats.min_r > 0.0);
        assert!(stats.far_frac >= 0.0 && stats.far_frac <= 1.0);
        for v in near.iter().chain(far.iter()) {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        }
    }

    #[test]
    fn snapshot_roundtrip() {
        let mut parts = generate(ConfigTag::Clumps, 128, 21);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let (near, far, _) = split_near_far(&parts, &tree);
        let snap = SnapshotData::from_parts(ConfigTag::Clumps, &parts, &near, &far);
        let dir = std::env::temp_dir().join("learned-fmm-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("snap_test.bin");
        snap.write_to(&path).unwrap();
        let back = SnapshotData::read_from(&path).unwrap();
        assert_eq!(back.tag, ConfigTag::Clumps);
        assert_eq!(back.pos, snap.pos);
        assert_eq!(back.mass, snap.mass);
        assert_eq!(back.a_total, snap.a_total);
        assert_eq!(back.a_near, snap.a_near);
        assert_eq!(back.a_far, snap.a_far);
        std::fs::remove_file(&path).ok();
    }
}
