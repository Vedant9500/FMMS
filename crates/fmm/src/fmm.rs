//! FMM orchestration: build tree, bodies, moments, power, f pre-pass, walk,
//! L2L, L2P, scatter to particle-id order.

use crate::bodies::Bodies;
use crate::harmonic::{dim, C};
use crate::models::Particle;
use crate::rotation::RotCtx;
use crate::tree::{CenterMode, Octree};

pub struct Fmm {
    pub tree: Octree,
    pub p: usize,
    pub eps: f64,
    pub n_leaf: usize,
}

pub struct Result {
    pub accel: Vec<[f64; 3]>,
    pub pot: Vec<f64>,
    pub n_cells: usize,
}

impl Fmm {
    /// Build with expansion centers at the cell center of mass (Phase-0
    /// default; momentum-conserving).
    pub fn new(parts: &[Particle], p: usize, eps: f64, n_leaf: usize) -> Self {
        Fmm { tree: Octree::build(parts, n_leaf), p, eps, n_leaf }
    }

    /// Build with an explicit center mode (com or Ritter bounding sphere).
    pub fn with_center(
        parts: &[Particle],
        p: usize,
        eps: f64,
        n_leaf: usize,
        mode: CenterMode,
    ) -> Self {
        Fmm { tree: Octree::build_with_mode(parts, n_leaf, mode), p, eps, n_leaf }
    }

    /// Run the full FMM (serial walk): returns accelerations and potentials
    /// (particle-id order).
    pub fn run(&self, parts: &[Particle]) -> Result {
        self.run_impl(parts, 1)
    }

    /// Run the full FMM with `nthreads` threads in the dual tree walk.
    pub fn run_threaded(&self, parts: &[Particle], nthreads: usize) -> Result {
        self.run_impl(parts, nthreads.max(1))
    }

    fn run_impl(&self, parts: &[Particle], nthreads: usize) -> Result {
        let p = self.p;
        let d = dim(p);
        let ncells = self.tree.cells.len();
        let n = parts.len();
        let mut bodies = Bodies::from_parts(parts, &self.tree.order);
        let mut moments = vec![C::ZERO; ncells * d];
        crate::moments::compute_moments(&bodies, &self.tree, p, &mut moments);
        let pows = crate::moments::multipole_power(&moments, ncells, p);
        let binom = crate::moments::binomials(p);
        let (_, fmin) = if nthreads > 1 {
            crate::mac::f_prepass_par(&bodies, &self.tree, nthreads)
        } else {
            crate::mac::f_prepass(&bodies, &self.tree)
        };
        let mut locals = vec![C::ZERO; ncells * d];
        if nthreads <= 1 {
            let mut rot = if p >= crate::rotation::ROT_P { Some(RotCtx::new(p)) } else { None };
            crate::walk::dual_walk(
                &mut bodies, &self.tree, p, self.eps, &pows, &binom, &fmin,
                &moments, &mut locals, rot.as_mut(),
            );
        } else {
            crate::walk::dual_walk_par(
                &mut bodies, &self.tree, p, self.eps, &pows, &binom, &fmin,
                &moments, &mut locals, nthreads,
            );
        }
        // L2L pass down: each cell adds its expansion to its children.
        for c in 0..ncells {
            let cell = self.tree.cells[c];
            for &ch in &cell.children {
                if ch >= 0 {
                    crate::translate::l2l(&self.tree, c, ch as usize, &mut locals, p);
                }
            }
        }
        // L2P at particles + scatter slot -> particle id.
        let mut pot = vec![0.0; n];
        let mut accel = vec![[0.0; 3]; n];
        for c in 0..ncells {
            if self.tree.is_leaf(c) {
                let cell = self.tree.cells[c];
                for k in cell.first..cell.first + cell.count {
                    let id = self.tree.order[k] as usize;
                    let mut p0 = 0.0;
                    let mut a = [0.0; 3];
                    crate::translate::l2p(
                        &self.tree, c, [bodies.x[k], bodies.y[k], bodies.z[k]],
                        &locals, p, &mut p0, &mut a,
                    );
                    pot[id] += bodies.pot[k] + p0;
                    accel[id][0] += bodies.ax[k] + a[0];
                    accel[id][1] += bodies.ay[k] + a[1];
                    accel[id][2] += bodies.az[k] + a[2];
                }
            }
        }
        Result { accel, pot, n_cells: ncells }
    }
}