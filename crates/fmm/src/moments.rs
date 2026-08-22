//! Multipole construction: P2M and M2M kernels, and multipole power.
//!
//! P2M  (eq 3c):  M_n^m(z) = Σ_a μ_a Υ_n^m(x_a - z)
//! M2M  (eq 3d):  M_n^m(z') = Σ_{k=0}^n Σ_{l=-k}^k Υ_k^l(z-z') M_{n-k}^{m-l}(z)
//! Multipole power (eq 12): P_n² = Σ_{m=-n}^n (n-m)!(n+m)! |M_n^m|².

use crate::harmonic::{idx, upsilon_table_into, C};
use crate::tree::Octree;

/// Factorials 0!..=2p.
pub fn factorials(p: usize) -> Vec<f64> {
    let mut f = vec![1.0; 2 * p + 1];
    for i in 1..f.len() {
        f[i] = f[i - 1] * i as f64;
    }
    f
}

/// Binomial C(p, k) for k = 0..=p.
pub fn binomials(p: usize) -> Vec<f64> {
    let fact = factorials(p);
    let mut b = vec![1.0; p + 1];
    for k in 0..=p {
        b[k] = fact[p] / (fact[k] * fact[p - k]);
    }
    b
}

/// P2M for one leaf cell into `moments` (scratch Υ table, no allocation).
fn p2m_cell(
    bodies: &crate::bodies::Bodies,
    tree: &Octree,
    c: usize,
    p: usize,
    moments: &mut [C],
    u: &mut [C],
) {
    let dim = crate::harmonic::dim(p);
    let cell = tree.cells[c];
    let off = c * dim;
    let z = cell.cen;
    for k in cell.first..cell.first + cell.count {
        upsilon_table_into(
            [bodies.x[k] - z[0], bodies.y[k] - z[1], bodies.z[k] - z[2]],
            p,
            u,
        );
        let m = bodies.m[k];
        for i in 0..dim {
            moments[off + i] = moments[off + i].add(u[i].scale(m));
        }
    }
}

/// M2M for one internal cell from its children (scratch Υ table).
fn m2m_cell(tree: &Octree, c: usize, p: usize, moments: &mut [C], u: &mut [C]) {
    let dim = crate::harmonic::dim(p);
    let cell = tree.cells[c];
    let off = c * dim;
    let z = cell.cen;
    for &ch in &cell.children {
        if ch < 0 {
            continue;
        }
        let child = tree.cells[ch as usize];
        let d = [child.cen[0] - z[0], child.cen[1] - z[1], child.cen[2] - z[2]];
        upsilon_table_into(d, p, u);
        let coff = ch as usize * dim;
        for n in 0..=p {
            for m in -(n as i32)..=n as i32 {
                // M_n^m(z') += Σ_k Σ_l Υ_k^l(z-z') M_{n-k}^{m-l}(z)
                let mut s = C::ZERO;
                for k in 0..=n {
                    let nk = n - k;
                    // |m - l| <= nk and |l| <= k, solved for l.
                    let lo = (-(k as i32)).max(m - nk as i32);
                    let hi = (k as i32).min(m + nk as i32);
                    for l in lo..=hi {
                        s = s.add(u[idx(k, l)].mul(moments[coff + idx(nk, m - l)]));
                    }
                }
                moments[off + idx(n, m)] = moments[off + idx(n, m)].add(s);
            }
        }
    }
}

/// Compute multipole moments for every cell.
///
/// `moments` is a flat array of `cells.len() * dim` complex coefficients.
/// Upward pass: leaves are built by P2M, internal cells by M2M from children.
pub fn compute_moments(bodies: &crate::bodies::Bodies, tree: &Octree, p: usize, moments: &mut [C]) {
    let dim = crate::harmonic::dim(p);
    let ncells = tree.cells.len();
    assert_eq!(moments.len(), ncells * dim);
    let mut u = vec![C::ZERO; dim];
    for c in (0..ncells).rev() {
        if tree.is_leaf(c) {
            p2m_cell(bodies, tree, c, p, moments, &mut u);
        } else {
            m2m_cell(tree, c, p, moments, &mut u);
        }
    }
}

/// Level-sweep parallel `compute_moments` (deepest level first; cells within a
/// level touch disjoint `moments` ranges and read only deeper-level children).
/// Bit-identical to the serial pass — per-cell arithmetic and child order are
/// unchanged; parallelism only reorders cells within a level.
pub fn compute_moments_par(
    bodies: &crate::bodies::Bodies,
    tree: &Octree,
    p: usize,
    moments: &mut [C],
    nthreads: usize,
) {
    let dim = crate::harmonic::dim(p);
    let total = moments.len();
    let (off, idx) = tree.levels_csr();
    for l in (0..off.len() - 1).rev() {
        let cells = &idx[off[l]..off[l + 1]];
        let chunk = cells.len().div_ceil(nthreads.max(1) * 4).max(1);
        let base = crate::SendPtr(moments.as_mut_ptr());
        std::thread::scope(|s| {
            for piece in cells.chunks(chunk.max(1)) {
                let base = crate::SendPtr(base.ptr());
                let mut u = vec![C::ZERO; dim];
                s.spawn(move || {
                    let moments: &mut [C] =
                        unsafe { std::slice::from_raw_parts_mut(base.ptr(), total) };
                    for &c in piece {
                        if tree.is_leaf(c) {
                            p2m_cell(bodies, tree, c, p, moments, &mut u);
                        } else {
                            m2m_cell(tree, c, p, moments, &mut u);
                        }
                    }
                });
            }
        });
    }
}

/// Multipole power P_n for every cell (eq 12). Returns flat array of
/// `cells.len() * (p+1)` values P_n (already square-rooted).
pub fn multipole_power(moments: &[C], ncells: usize, p: usize) -> Vec<f64> {
    let dim = crate::harmonic::dim(p);
    let fact = factorials(p);
    let mut pp = vec![0.0; ncells * (p + 1)];
    for c in 0..ncells {
        let off = c * dim;
        for n in 0..=p {
            let mut s = 0.0;
            for m in -(n as i32)..=n as i32 {
                let am = m.unsigned_abs() as usize;
                let mc = moments[off + idx(n, m)];
                let w = fact[n - am] * fact[n + am];
                s += w * (mc.re * mc.re + mc.im * mc.im);
            }
            pp[c * (p + 1) + n] = s.sqrt();
        }
    }
    pp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{plummer, Particle};

    #[test]
    fn compute_moments_par_matches_serial() {
        let parts = plummer(2048, [0.0; 3], 13);
        let tree = crate::tree::Octree::build(&parts, 16);
        let p = 6;
        let dim = crate::harmonic::dim(p);
        let bodies = crate::bodies::Bodies::from_parts(&parts, &tree.order);
        let mut m1 = vec![C::ZERO; tree.cells.len() * dim];
        compute_moments(&bodies, &tree, p, &mut m1);
        let mut m2 = vec![C::ZERO; tree.cells.len() * dim];
        compute_moments_par(&bodies, &tree, p, &mut m2, 8);
        assert_eq!(m1, m2);
    }

    #[test]
    fn monopole_is_mass() {
        let parts = vec![
            Particle { mass: 2.0, pos: [1.0, 0.0, 0.0] },
            Particle { mass: 3.0, pos: [0.0, 2.0, 0.0] },
        ];
        let tree = Octree::build(&parts, 1);
        let dim = crate::harmonic::dim(2);
        let mut moments = vec![C::ZERO; tree.cells.len() * dim];
        let bodies = crate::bodies::Bodies::from_parts(&parts, &tree.order);
        compute_moments(&bodies, &tree, 2, &mut moments);
        // Root cell is index 0 with all mass.
        let m00 = moments[idx(0, 0)];
        assert!((m00.re - 5.0).abs() < 1e-12);
        assert!((m00.im - 0.0).abs() < 1e-12);
        // Dipole about the CoM vanishes.
        let m10 = moments[idx(1, 0)];
        let m11 = moments[idx(1, 1)];
        assert!(m10.re.abs() < 1e-12 && m10.im.abs() < 1e-12);
        assert!(m11.re.abs() < 1e-12 && m11.im.abs() < 1e-12);
    }

    #[test]
    fn binomial_and_fact() {
        let f = factorials(4);
        assert_eq!(&f[..6], &[1.0, 1.0, 2.0, 6.0, 24.0, 120.0]);
        let b = binomials(4);
        assert_eq!(b, vec![1.0, 4.0, 6.0, 4.0, 1.0]);
    }
}