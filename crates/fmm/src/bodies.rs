//! Structure-of-Arrays particle storage (SoA).
//!
//! Bodies are indexed by *slot* = position in `Octree::order`, so the
//! particles of a leaf cell occupy the contiguous range `[first, first+count)`.
//! This gives the P2P kernel contiguous, vectorizable loads (docs/02-lowlevel.md
//! §3: SoA ≈ 2.3× over AoS) and keeps the Morton-sorted cache locality.

use crate::models::Particle;

/// SoA particle state. Accelerations/potentials accumulate here during the
/// walk (near field) and L2P (far field); caller scatters to particle ids.
#[derive(Debug, Clone)]
pub struct Bodies {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub m: Vec<f64>,
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
    pub az: Vec<f64>,
    pub pot: Vec<f64>,
}

impl Bodies {
    /// Build from `parts` in the Morton-sorted order given by `tree.order`.
    /// Slot `k` corresponds to particle `order[k]`.
    pub fn from_parts(parts: &[Particle], order: &[u32]) -> Self {
        let n = parts.len();
        let mut b = Bodies {
            x: vec![0.0; n],
            y: vec![0.0; n],
            z: vec![0.0; n],
            m: vec![0.0; n],
            ax: vec![0.0; n],
            ay: vec![0.0; n],
            az: vec![0.0; n],
            pot: vec![0.0; n],
        };
        for (k, &id) in order.iter().enumerate() {
            let p = &parts[id as usize];
            b.x[k] = p.pos[0];
            b.y[k] = p.pos[1];
            b.z[k] = p.pos[2];
            b.m[k] = p.mass;
        }
        b
    }

    /// Re-gather positions/masses into existing allocations (no realloc when
    /// `n` is unchanged). GPU path helper: the device accumulates forces, so
    /// ax/ay/az/pot are left untouched (skips a 4-array memset per step).
    pub fn gather_into(&mut self, parts: &[Particle], order: &[u32]) {
        let n = parts.len();
        self.x.resize(n, 0.0);
        self.y.resize(n, 0.0);
        self.z.resize(n, 0.0);
        self.m.resize(n, 0.0);
        for (k, &id) in order.iter().enumerate() {
            let p = &parts[id as usize];
            self.x[k] = p.pos[0];
            self.y[k] = p.pos[1];
            self.z[k] = p.pos[2];
            self.m[k] = p.mass;
        }
    }

    /// Scatter slot-indexed accelerations/potentials back to particle id order.
    pub fn scatter(&self, order: &[u32], accel: &mut [[f64; 3]], pot: &mut [f64]) {
        for (k, &id) in order.iter().enumerate() {
            accel[id as usize] = [self.ax[k], self.ay[k], self.az[k]];
            pot[id as usize] = self.pot[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::plummer;

    #[test]
    fn gather_into_matches_from_parts() {
        let parts = plummer(64, [0.0; 3], 7);
        let tree = crate::tree::Octree::build(&parts, 4);
        let b = Bodies::from_parts(&parts, &tree.order);
        let mut re = b.clone();
        // clobber then re-gather: positions must round-trip exactly
        for v in re.x.iter_mut().chain(re.y.iter_mut()).chain(re.z.iter_mut()).chain(re.m.iter_mut()) { *v = 123.0; }
        re.gather_into(&parts, &tree.order);
        assert_eq!(re.x, b.x);
        assert_eq!(re.y, b.y);
        assert_eq!(re.z, b.z);
        assert_eq!(re.m, b.m);
    }

    #[test]
    fn slot_order_matches_order() {
        let parts = plummer(64, [0.0; 3], 7);
        let tree = crate::tree::Octree::build(&parts, 4);
        let b = Bodies::from_parts(&parts, &tree.order);
        for (k, &id) in tree.order.iter().enumerate() {
            assert_eq!(b.x[k], parts[id as usize].pos[0]);
            assert_eq!(b.y[k], parts[id as usize].pos[1]);
            assert_eq!(b.z[k], parts[id as usize].pos[2]);
            assert_eq!(b.m[k], parts[id as usize].mass);
        }
    }
}
