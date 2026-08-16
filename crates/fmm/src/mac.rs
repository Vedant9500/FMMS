//! Multipole acceptance criterion and the f pre-pass.
//!
//! Error estimate (Dehnen 2014):
//!   E_{A→B} = (1/M_A) Σ_{k=0}^p C(p,k) P_{k,A} ρ_{s,B}^{p-k} / r^p   (eq 13)
//!   E~_{A→B} = 8 max{ρ_z,A, ρ_s,B}/(ρ_z,A+ρ_s,B) E_{A→B}            (eq 15)
//! with z = s = com, so ρ_z = ρ_s = ρ.
//!
//! Acceptance (eq 16b):
//!   θ < 1  ∧  E~_{A→B} M_A/r² < ε · min_{b∈B}{f_b}
//! where f_b = Σ_{a≠b} m_a/|x_a-x_b|² (eq 4) is estimated by a p=0 treecode
//! monopole pre-pass with θ<1 (Dehnen §5.4.2).

use crate::tree::Octree;

/// Opening angle θ = (ρ_A + ρ_B) / r.
#[inline]
pub fn theta(ca: &crate::tree::Cell, cb: &crate::tree::Cell, r: f64) -> f64 {
    (ca.rho + cb.rho) / r
}

/// E~_{A→B} (eq 13 + 15). `pows` is the flat per-cell multipole-power array
/// (cells.len() * (p+1)); `binom` is the C(p,k) table. Returns None if r == 0.
pub fn error_estimate(
    ca_idx: usize,
    cb: &crate::tree::Cell,
    ca_mass: f64,
    ca_rho: f64,
    pows: &[f64],
    binom: &[f64],
    p: usize,
    r: f64,
) -> Option<f64> {
    if r == 0.0 {
        return None;
    }
    let r2 = r * r;
    let mut rp = 1.0f64;
    for _ in 0..p / 2 {
        rp *= r2;
    }
    if p % 2 == 1 {
        rp *= r;
    }
    // rho_B^j for j = 0..=p (stack array; p is bounded by ROT_P usage <= ~30).
    let mut rpows = [1.0f64; 64];
    for j in 1..=p {
        rpows[j] = rpows[j - 1] * cb.rho;
    }
    let base = ca_idx * (p + 1);
    let mut s = 0.0;
    for k in 0..=p {
        s += binom[k] * pows[base + k] * rpows[p - k];
    }
    let e = s / (ca_mass * rp);
    let denom = ca_rho + cb.rho;
    if denom == 0.0 {
        // Both cells are points: the monopole estimate is exact.
        return Some(0.0);
    }
    Some(8.0 * ca_rho.max(cb.rho) / denom * e)
}

/// Whether cell a may act on cell b via a multipole expansion (eq 16b).
pub fn accept(
    ca_idx: usize,
    cb_idx: usize,
    cells: &[crate::tree::Cell],
    pows: &[f64],
    binom: &[f64],
    p: usize,
    eps: f64,
    fmin: &[f64],
) -> bool {
    let ca = &cells[ca_idx];
    let cb = &cells[cb_idx];
    let dx = cb.cen[0] - ca.cen[0];
    let dy = cb.cen[1] - ca.cen[1];
    let dz = cb.cen[2] - ca.cen[2];
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if theta(ca, cb, r) >= 1.0 {
        return false;
    }
    match error_estimate(ca_idx, cb, ca.mass, ca.rho, pows, binom, p, r) {
        None => false,
        Some(est) => {
            if est == 0.0 {
                return true;
            }
            let r2 = r * r;
            est * ca.mass / r2 < eps * fmin[cb_idx]
        }
    }
}

/// Monopole (p=0, θ<1) tree-walk estimate of f_b for one particle.
/// `stack` is caller-owned scratch, cleared per call.
fn f_walk(
    bodies: &crate::bodies::Bodies,
    tree: &Octree,
    i: usize,
    stack: &mut Vec<usize>,
) -> f64 {
    let pb = [bodies.x[i], bodies.y[i], bodies.z[i]];
    stack.clear();
    stack.push(0);
    let mut fi = 0.0;
    while let Some(c) = stack.pop() {
        let cell = tree.cells[c];
        if tree.is_leaf(c) {
            for k in cell.first..cell.first + cell.count {
                if k == i {
                    continue;
                }
                let dx = pb[0] - bodies.x[k];
                let dy = pb[1] - bodies.y[k];
                let dz = pb[2] - bodies.z[k];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 > 0.0 {
                    fi += bodies.m[k] / r2;
                }
            }
        } else {
            let dx = pb[0] - cell.cen[0];
            let dy = pb[1] - cell.cen[1];
            let dz = pb[2] - cell.cen[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > cell.rho * cell.rho {
                fi += cell.mass / d2;
            } else {
                for &ch in &cell.children {
                    if ch >= 0 {
                        stack.push(ch as usize);
                    }
                }
            }
        }
    }
    fi
}

/// Per-cell fmin from per-particle f (children appear after parents in the
/// cell array, so the reverse sweep sees leaves first).
fn fmin_of(f: &[f64], tree: &Octree) -> Vec<f64> {
    let ncells = tree.cells.len();
    let mut fmin = vec![f64::INFINITY; ncells];
    for c in (0..ncells).rev() {
        let cell = tree.cells[c];
        if tree.is_leaf(c) {
            let mut m = f64::INFINITY;
            for k in cell.first..cell.first + cell.count {
                m = m.min(f[k]);
            }
            fmin[c] = m;
        } else {
            let mut m = f64::INFINITY;
            for &ch in &cell.children {
                if ch >= 0 {
                    m = m.min(fmin[ch as usize]);
                }
            }
            fmin[c] = m;
        }
    }
    fmin
}

/// Estimate f_b = Σ_{a≠b} m_a/|x_a-x_b|² for every particle with a p=0
/// treecode (monopole, θ<1). Returns (per-particle f_b, per-cell min).
pub fn f_prepass(bodies: &crate::bodies::Bodies, tree: &Octree) -> (Vec<f64>, Vec<f64>) {
    let n = bodies.x.len();
    let mut f = vec![0.0; n];
    let mut stack: Vec<usize> = Vec::with_capacity(64);
    for i in 0..n {
        f[i] = f_walk(bodies, tree, i, &mut stack);
    }
    let fmin = fmin_of(&f, tree);
    (f, fmin)
}

/// Parallel `f_prepass`: disjoint contiguous particle ranges per thread.
/// Bit-identical to `f_prepass` — each f_b is an independent serial sum, and
/// the fmin sweep is unchanged. Bodies are in Morton order, so contiguous
/// chunks each span many octants, which balances the walk cost.
pub fn f_prepass_par(
    bodies: &crate::bodies::Bodies,
    tree: &Octree,
    nthreads: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = bodies.x.len();
    let mut f = vec![0.0; n];
    let chunk = n.div_ceil(nthreads.max(1));
    std::thread::scope(|s| {
        for (t, piece) in f.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            s.spawn(move || {
                let mut stack: Vec<usize> = Vec::with_capacity(64);
                for (off, v) in piece.iter_mut().enumerate() {
                    *v = f_walk(bodies, tree, base + off, &mut stack);
                }
            });
        }
    });
    let fmin = fmin_of(&f, tree);
    (f, fmin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{plummer, Particle};

    #[test]
    fn f_prepass_small() {
        // Two particles: f = m/d² each.
        let parts = vec![
            Particle { mass: 1.0, pos: [0.0, 0.0, 0.0] },
            Particle { mass: 2.0, pos: [3.0, 0.0, 0.0] },
        ];
        let tree = Octree::build(&parts, 1);
        let bodies = crate::bodies::Bodies::from_parts(&parts, &tree.order);
        let (f, _) = f_prepass(&bodies, &tree);
        assert!((f[0] - 2.0 / 9.0).abs() < 1e-12);
        assert!((f[1] - 1.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn f_prepass_par_matches_serial() {
        let parts = plummer(2048, [0.0; 3], 11);
        let tree = Octree::build(&parts, 16);
        let bodies = crate::bodies::Bodies::from_parts(&parts, &tree.order);
        let (f1, m1) = f_prepass(&bodies, &tree);
        let (f2, m2) = f_prepass_par(&bodies, &tree, 8);
        assert_eq!(f1, f2);
        assert_eq!(m1, m2);
    }
}