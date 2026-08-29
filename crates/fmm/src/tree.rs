//! Morton-sorted octree.
//!
//! Particles are quantized into a cubic root and sorted by 64-bit Morton key.
//! The tree is built top-down by recursive binary splits on the key prefix;
//! leaf cells hold ≤ `n_leaf` particles. Each cell stores its expansion
//! centre `cen` (z = s), the spherical radius `rho` = max distance of its
//! contents from `cen`, and total mass.
//!
//! Two expansion-centre strategies (Dehnen 2014, §5.1):
//!   - `Com`:   centre of mass. Dipole vanishes, best for low p.
//!   - `Sphere`: enclosing sphere (Ritter grow/enclose). Tighter `rho` for
//!     extended cells, best for high p where the multipole order dominates.

use crate::models::Particle;

const BITS: u32 = 21; // bits per axis -> 63-bit keys, sufficient for N <= 2^21 per axis

/// Expansion-centre strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CenterMode {
    /// Centre of mass (default; dipole-free multipoles).
    Com,
    /// Minimum-enclosing-sphere centre (Ritter grow/enclose).
    Sphere,
}

/// Interleave a 21-bit coordinate into every third bit of a u64.
fn dilate(mut x: u64) -> u64 {
    x &= (1u64 << BITS) - 1;
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    (x | (x << 2)) & 0x1249_2492_4924_9249
}

fn morton_key(x: f64, y: f64, z: f64, lo: &[f64; 3], inv_side: f64) -> u64 {
    let q = |c: f64| {
        let v = ((c - lo[0]) * inv_side).clamp(0.0, 1.0);
        (v * ((1u64 << BITS) - 1) as f64) as u64
    };
    let mut key = dilate(q(z));
    key |= dilate(q(y)) << 1;
    key |= dilate(q(x)) << 2;
    key
}

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub first: usize, // index into `order`
    pub count: usize,
    pub children: [i32; 8], // -1 if absent
    pub leaf: bool,
    pub level: u32,
    pub cen: [f64; 3], // expansion centre (CoM or sphere centre)
    pub mass: f64,
    pub rho: f64, // max |x_a - cen| over cell contents
}

pub struct Octree {
    pub cells: Vec<Cell>,
    pub order: Vec<u32>, // sorted index -> particle id
    pub n_leaf: usize,
    pub mode: CenterMode,
    /// Scratch retained across `refresh_*` calls (allocation reuse only;
    /// contents are recomputed from scratch every build/refresh).
    scratch_keys: Vec<u64>,
    scratch_keys_sorted: Vec<u64>,
    scratch_idx: Vec<usize>,
}

impl Octree {
    pub fn build(parts: &[Particle], n_leaf: usize) -> Self {
        Self::build_with_mode(parts, n_leaf, CenterMode::Com)
    }

    pub fn build_with_mode(parts: &[Particle], n_leaf: usize, mode: CenterMode) -> Self {
        let mut tree = Octree {
            cells: Vec::new(),
            order: Vec::new(),
            n_leaf,
            mode,
            scratch_keys: Vec::new(),
            scratch_keys_sorted: Vec::new(),
            scratch_idx: Vec::new(),
        };
        tree.refresh_with_mode(parts, mode);
        tree
    }

    /// Rebuild the tree for new positions, reusing every allocation (cells,
    /// order, key/index scratch). Output is identical to a fresh
    /// `build_with_mode` on the same input; asserted by `refresh_matches_build`.
    pub fn refresh_with_mode(&mut self, parts: &[Particle], mode: CenterMode) {
        assert!(!parts.is_empty());
        self.mode = mode;
        let n = parts.len();
        // Bounding box.
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in parts {
            for d in 0..3 {
                lo[d] = lo[d].min(p.pos[d]);
                hi[d] = hi[d].max(p.pos[d]);
            }
        }
        let side = (hi[0] - lo[0])
            .max(hi[1] - lo[1])
            .max(hi[2] - lo[2])
            .max(f64::MIN_POSITIVE);
        let inv_side = 1.0 / side;

        self.scratch_keys.clear();
        self.scratch_keys.resize(n, 0);
        let keys = &mut self.scratch_keys;
        for (i, p) in parts.iter().enumerate() {
            keys[i] = morton_key(p.pos[0], p.pos[1], p.pos[2], &lo, inv_side);
        }

        self.order.clear();
        self.order.resize(n, 0);
        self.scratch_keys_sorted.clear();
        self.scratch_keys_sorted.resize(n, 0);
        self.scratch_idx.clear();
        self.scratch_idx.resize(n, 0);
        let idx = &mut self.scratch_idx;
        for (i, slot) in idx.iter_mut().enumerate() {
            *slot = i;
        }
        idx.sort_unstable_by_key(|&i| keys[i]);
        for (k, &i) in idx.iter().enumerate() {
            self.order[k] = i as u32;
            self.scratch_keys_sorted[k] = keys[i];
        }

        self.cells.clear();
        build_rec(
            &self.scratch_keys_sorted,
            &mut self.cells,
            0,
            0,
            parts.len(),
            self.n_leaf,
            BITS,
        );

        self.geometry(parts);
    }

    /// Bottom-up pass computing mass, expansion centre, and ρ for every cell.
    /// Cells are stored depth-first, parents before children.
    fn geometry(&mut self, parts: &[Particle]) {
        let n = self.cells.len();
        for i in (0..n).rev() {
            let cell = self.cells[i];
            let is_leaf = cell.children.iter().all(|&x| x < 0);
            let first = cell.first;
            let count = cell.count;
            let children = cell.children;
            let mut mass = 0.0;
            if is_leaf {
                // Leaf: aggregate directly from particles.
                for k in first..first + count {
                    mass += parts[self.order[k] as usize].mass;
                }
                let cen = match self.mode {
                    CenterMode::Com => com_of_slots(parts, self, first, count, mass),
                    CenterMode::Sphere => sphere_of_slots(parts, self, first, count),
                };
                let mut rho: f64 = 0.0;
                for k in first..first + count {
                    let p = &parts[self.order[k] as usize];
                    rho = rho.max(dist(p.pos, cen));
                }
                self.cells[i].mass = mass;
                self.cells[i].cen = cen;
                self.cells[i].rho = rho;
            } else {
                // Internal: combine children (they appear later in the array).
                for &c in &children {
                    if c >= 0 {
                        mass += self.cells[c as usize].mass;
                    }
                }
                let cen = match self.mode {
                    CenterMode::Com => {
                        let mut com = [0.0; 3];
                        for &c in &children {
                            if c < 0 {
                                continue;
                            }
                            let ch = &self.cells[c as usize];
                            for d in 0..3 {
                                com[d] += ch.mass * ch.cen[d];
                            }
                        }
                        for d in 0..3 {
                            com[d] /= mass;
                        }
                        com
                    }
                    CenterMode::Sphere => {
                        let mut items: Vec<([f64; 3], f64)> = Vec::with_capacity(8);
                        for &c in &children {
                            if c >= 0 {
                                let ch = &self.cells[c as usize];
                                items.push((ch.cen, ch.rho));
                            }
                        }
                        ritter(&items).0
                    }
                };
                // ρ_parent = max over children of (ρ_child + |cen_child - cen|).
                let mut rho: f64 = 0.0;
                for &c in &children {
                    if c < 0 {
                        continue;
                    }
                    let ch = &self.cells[c as usize];
                    rho = rho.max(ch.rho + dist(ch.cen, cen));
                }
                self.cells[i].mass = mass;
                self.cells[i].cen = cen;
                self.cells[i].rho = rho;
            }
        }
    }

    pub fn is_leaf(&self, c: usize) -> bool {
        self.cells[c].leaf
    }

    /// Cell indices grouped by level (CSR: offsets into `idx`). Level-sweep
    /// passes (moments bottom-up, L2L top-down) can then run level by level;
    /// cells within a level are mutually independent.
    pub fn levels_csr(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.cells.len();
        let maxl = self.cells.iter().map(|c| c.level).max().unwrap_or(0) as usize;
        let mut off = vec![0usize; maxl + 2];
        for c in &self.cells {
            off[c.level as usize + 1] += 1;
        }
        for i in 1..off.len() {
            off[i] += off[i - 1];
        }
        let mut idx = vec![0usize; n];
        let mut pos = off[..maxl + 1].to_vec();
        for (i, c) in self.cells.iter().enumerate() {
            let l = c.level as usize;
            idx[pos[l]] = i;
            pos[l] += 1;
        }
        (off, idx)
    }
}

fn com_of_slots(
    parts: &[Particle],
    tree: &Octree,
    first: usize,
    count: usize,
    mass: f64,
) -> [f64; 3] {
    let mut com = [0.0; 3];
    for k in first..first + count {
        let p = &parts[tree.order[k] as usize];
        for d in 0..3 {
            com[d] += p.mass * p.pos[d];
        }
    }
    for d in 0..3 {
        com[d] /= mass;
    }
    com
}

/// Centre/radius for the particles of a leaf (point items, radius 0).
fn sphere_of_slots(parts: &[Particle], tree: &Octree, first: usize, count: usize) -> [f64; 3] {
    let items: Vec<([f64; 3], f64)> = (first..first + count)
        .map(|k| {
            let p = &parts[tree.order[k] as usize];
            (p.pos, 0.0)
        })
        .collect();
    ritter(&items).0
}

/// Ritter (1990) grow/enclose bounding sphere for a set of (centre, radius)
/// items. Two passes: extremes seed the sphere, then an expansion pass.
fn ritter(items: &[([f64; 3], f64)]) -> ([f64; 3], f64) {
    assert!(!items.is_empty());
    // Seed from a diameter approximation: any item, then the two successive
    // farthest items (two-farthest-points heuristic). Axis extremes alone can
    // miss the true diameter pair and inflate the sphere.
    let mut a = 0usize;
    let mut b = 0usize;
    let mut best = -1.0f64;
    for (i, (c, _)) in items.iter().enumerate() {
        let d2 = dist2(*c, items[a].0);
        if d2 > best {
            best = d2;
            b = i;
        }
    }
    a = b;
    best = -1.0;
    for (i, (c, _)) in items.iter().enumerate() {
        let d2 = dist2(*c, items[a].0);
        if d2 > best {
            best = d2;
            b = i;
        }
    }
    // Seed sphere from the two far points' centres.
    let pa = items[a].0;
    let pb = items[b].0;
    let dir = {
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt().max(f64::MIN_POSITIVE);
        [dx / d, dy / d, dz / d]
    };
    let far_a = [
        pa[0] + items[a].1 * dir[0],
        pa[1] + items[a].1 * dir[1],
        pa[2] + items[a].1 * dir[2],
    ];
    let far_b = [
        pb[0] - items[b].1 * dir[0],
        pb[1] - items[b].1 * dir[1],
        pb[2] - items[b].1 * dir[2],
    ];
    let mut c = [
        (far_a[0] + far_b[0]) * 0.5,
        (far_a[1] + far_b[1]) * 0.5,
        (far_a[2] + far_b[2]) * 0.5,
    ];
    let mut r = dist(far_a, far_b) * 0.5;

    // Two expansion passes: each point/item outside the sphere grows it
    // toward that item's far surface point.
    for _ in 0..2 {
        for (q, rad) in items.iter() {
            let d = dist(*q, c);
            if d + rad > r {
                let dd = d.max(f64::MIN_POSITIVE);
                let dir = [(q[0] - c[0]) / dd, (q[1] - c[1]) / dd, (q[2] - c[2]) / dd];
                let p = [
                    q[0] + rad * dir[0],
                    q[1] + rad * dir[1],
                    q[2] + rad * dir[2],
                ];
                let d_far = d + rad;
                let t = (d_far - r) / (2.0 * d_far);
                for k in 0..3 {
                    c[k] += t * (p[k] - c[k]);
                }
                r = (d_far + r) * 0.5;
            }
        }
    }

    // Grow-only refinement is order-dependent and can overshoot (Ritter's
    // known weakness). A bounding-box-centred sphere is tight for symmetric
    // sets (e.g. a uniform grid), so keep whichever candidate is smaller.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (q, rad) in items.iter() {
        for d in 0..3 {
            lo[d] = lo[d].min(q[d] - rad);
            hi[d] = hi[d].max(q[d] + rad);
        }
    }
    let cb = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let mut rb: f64 = 0.0;
    for (q, rad) in items.iter() {
        rb = rb.max(dist(*q, cb) + rad);
    }
    if rb < r {
        (cb, rb)
    } else {
        (c, r)
    }
}

#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[inline]
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    dist2(a, b).sqrt()
}

fn build_rec(
    keys: &[u64],
    cells: &mut Vec<Cell>,
    level: u32,
    start: usize,
    end: usize,
    n_leaf: usize,
    max_level: u32,
) -> usize {
    let count = end - start;
    let leaf = count <= n_leaf || level >= max_level;
    cells.push(Cell {
        first: start,
        count,
        children: [-1; 8],
        leaf,
        level,
        cen: [0.0; 3],
        mass: 0.0,
        rho: 0.0,
    });
    let id = cells.len() - 1;
    if leaf {
        return id;
    }
    // Child octant = the 3 bits at position `shift`. Keys are sorted, so each
    // octant occupies a contiguous range.
    let shift = 3 * (max_level - (level + 1));
    let mut lo = start;
    for oct in 0..8u64 {
        let mut hi = lo;
        while hi < end && ((keys[hi] >> shift) & 0b111) == oct {
            hi += 1;
        }
        if hi > lo {
            let cid = build_rec(keys, cells, level + 1, lo, hi, n_leaf, max_level);
            cells[id].children[oct as usize] = cid as i32;
        }
        lo = hi;
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::plummer;

    fn points() -> Vec<Particle> {
        let mut v = Vec::new();
        for i in 0..64usize {
            let x = (i % 4) as f64;
            let y = ((i / 4) % 4) as f64;
            let z = (i / 16) as f64;
            v.push(Particle {
                mass: 1.0,
                pos: [x, y, z],
            });
        }
        v
    }

    #[test]
    fn build_and_geometry() {
        let parts = points();
        let tree = Octree::build(&parts, 1);
        assert_eq!(tree.cells.len(), 1 + 8 + 64);
        // Root covers all mass; every particle in exactly one leaf.
        let root = &tree.cells[0];
        assert!((root.mass - 64.0).abs() < 1e-12);
        assert!((root.cen[0] - 1.5).abs() < 1e-9);
        assert!((root.cen[1] - 1.5).abs() < 1e-9);
        assert!((root.cen[2] - 1.5).abs() < 1e-9);
        // rho = distance from cen to a corner.
        let expect = ((1.5f64 * 1.5) * 3.0).sqrt();
        assert!((root.rho - expect).abs() < 1e-9);
    }

    #[test]
    fn degenerate_points() {
        // All points identical -> can't be split further; single leaf holds all.
        let parts = vec![
            Particle {
                mass: 2.0,
                pos: [1.0, 1.0, 1.0]
            };
            5
        ];
        let tree = Octree::build(&parts, 5);
        assert_eq!(tree.cells.len(), 1);
        assert!(tree.is_leaf(0));
        assert!((tree.cells[0].mass - 10.0).abs() < 1e-12);
    }

    #[test]
    fn refresh_matches_build() {
        let parts = plummer(4096, [0.0; 3], 99);
        for mode in [CenterMode::Com, CenterMode::Sphere] {
            let fresh = Octree::build_with_mode(&parts, 16, mode);
            // A tree built on shifted positions, then refreshed back.
            let shifted: Vec<Particle> = parts
                .iter()
                .map(|p| Particle {
                    mass: p.mass,
                    pos: [p.pos[0] + 5.0, p.pos[1] - 3.0, p.pos[2] + 1.0],
                })
                .collect();
            let mut rebuilt = Octree::build_with_mode(&shifted, 16, mode);
            rebuilt.refresh_with_mode(&parts, mode);
            assert_eq!(fresh.cells.len(), rebuilt.cells.len());
            for (a, b) in fresh.cells.iter().zip(rebuilt.cells.iter()) {
                assert_eq!(a.first, b.first);
                assert_eq!(a.count, b.count);
                assert_eq!(a.children, b.children);
                assert_eq!(a.leaf, b.leaf);
                assert_eq!(a.level, b.level);
                assert_eq!(a.cen, b.cen);
                assert_eq!(a.mass.to_bits(), b.mass.to_bits());
                assert_eq!(a.rho.to_bits(), b.rho.to_bits());
            }
            assert_eq!(fresh.order, rebuilt.order);
            // Refresh twice more (allocation-reuse path) — still identical.
            rebuilt.refresh_with_mode(&parts, mode);
            assert_eq!(fresh.order, rebuilt.order);
        }
    }

    #[test]
    fn refresh_degenerate_points() {
        // Identical positions: single leaf; refresh must reproduce it.
        let parts = vec![
            Particle {
                mass: 2.0,
                pos: [1.0, 1.0, 1.0]
            };
            5
        ];
        let mut tree = Octree::build_with_mode(&parts, 16, CenterMode::Com);
        tree.refresh_with_mode(&parts, CenterMode::Com);
        assert_eq!(tree.cells.len(), 1);
        assert!(tree.is_leaf(0));
        assert!((tree.cells[0].mass - 10.0).abs() < 1e-12);
    }

    #[test]
    fn dilate_spreads_bits() {
        assert_eq!(dilate(0b11), 0b1001); // bits 0 and 1 -> positions 0 and 3
        assert_eq!(dilate(1), 1);
    }

    #[test]
    fn sphere_mode_rho_no_larger_than_com() {
        // A 3x3x3 grid of unit mass points: Com rho = dist to corner;
        // sphere centre rho is at most that (sphere centre = cube centre here).
        let parts = points();
        let com = Octree::build(&parts, 1);
        let sph = Octree::build_with_mode(&parts, 1, CenterMode::Sphere);
        assert!(sph.cells[0].rho <= com.cells[0].rho + 1e-9);
        // Sphere encloses every particle.
        for i in (0..com.cells.len()).rev() {
            let cell = sph.cells[i];
            if sph.is_leaf(i) {
                for k in cell.first..cell.first + cell.count {
                    let p = &parts[sph.order[k] as usize];
                    let d = dist(p.pos, cell.cen);
                    assert!(d <= cell.rho + 1e-9, "leaf {i} encloses particle");
                }
            }
        }
        // Internal cells enclose their children's spheres.
        for i in (0..com.cells.len()).rev() {
            let cell = sph.cells[i];
            if !sph.is_leaf(i) {
                for &c in &cell.children {
                    if c >= 0 {
                        let ch = sph.cells[c as usize];
                        let d = dist(ch.cen, cell.cen);
                        assert!(d + ch.rho <= cell.rho + 1e-9, "cell {i} encloses child");
                    }
                }
            }
        }
    }
}
