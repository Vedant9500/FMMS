//! Translation and evaluation kernels.
//!
//! M2L  (eq 3b): F_n^m(s_B) += Σ_{k=0}^{p-n} Σ_{l=-k}^k M_k^{l*}(z_A) Θ_{n+k}^{m+l}(s_B - z_A)
//! L2L  (eq 3e): F_n^m(s') += Σ_{k=0}^{p-n} Σ_l Υ_k^{l*}(s-s') F_{n+k}^{m+l}(s)
//! L2P  (eq 3f): Ψ_n^m(x_b) = Σ_{k=0}^{p-n} Σ_l Υ_k^{l*}(s_B-x_b) F_{n+k}^{m+l}(s_B)
//!
//! Acceleration (Dehnen 2014, §2.1): a = ∇Ψ_0^0 = -(Re{Ψ_1^1}, Im{Ψ_1^1}, Ψ_1^0).

use crate::harmonic::{idx, theta_table, theta_table_into, upsilon_table_into, C, dim};
use crate::tree::Octree;

/// Reusable per-worker scratch for `m2l_mutual_scratch` (avoids the 4 heap
/// allocations per call the wrapper makes). Arithmetic is identical to the
/// `m2l_mutual` baseline.
pub struct M2lScratch {
    pub th: Vec<C>,
    pub ths: Vec<C>,
    pub cma: Vec<C>,
    pub cmb: Vec<C>,
}

impl M2lScratch {
    pub fn new(p: usize) -> Self {
        let d = dim(p);
        M2lScratch {
            th: vec![C::ZERO; d],
            ths: vec![C::ZERO; d],
            cma: vec![C::ZERO; d],
            cmb: vec![C::ZERO; d],
        }
    }
}

/// Compute the mutual M2L for a pair of well-separated cells a (source of b)
/// and b (source of a). The Θ table at r = s_B - z_A is reused for the
/// reverse direction via parity Θ_n^m(-r) = (-1)^n Θ_n^m(r).
///
/// * `moments` — per-cell multipoles, flat ncells*dim.
/// * `locals`  — per-cell local expansions, flat ncells*dim.
pub fn m2l_mutual(
    tree: &Octree,
    a: usize,
    b: usize,
    moments: &[C],
    locals: &mut [C],
    p: usize,
) {
    let mut scratch = M2lScratch::new(p);
    m2l_mutual_scratch(tree, a, b, moments, locals, p, &mut scratch);
}

/// Allocation-free variant of `m2l_mutual`; `scratch` must be sized for `p`.
/// Bit-compatible with `m2l_mutual` (identical operation order).
pub fn m2l_mutual_scratch(
    tree: &Octree,
    a: usize,
    b: usize,
    moments: &[C],
    locals: &mut [C],
    p: usize,
    scratch: &mut M2lScratch,
) {
    let d = dim(p);
    let ca = tree.cells[a];
    let cb = tree.cells[b];
    // r = s_B - z_A, with z = s = expansion centre.
    let r = [
        cb.cen[0] - ca.cen[0],
        cb.cen[1] - ca.cen[1],
        cb.cen[2] - ca.cen[2],
    ];
    theta_table_into(r, p, &mut scratch.th);
    let offa = a * d;
    let offb = b * d;
    // Precompute conjugated moments once (they are constant across (n, m)),
    // and a copy of the Θ table pre-scaled by the parity (-1)^{order} so the
    // reverse direction needs no per-iteration sign.
    let cma = &mut scratch.cma;
    let cmb = &mut scratch.cmb;
    for i in 0..d {
        cma[i] = moments[offa + i].conj();
        cmb[i] = moments[offb + i].conj();
    }
    let ths = &mut scratch.ths;
    let th = &scratch.th;
    {
        let mut o = 0usize;
        for n in 0..=p {
            let sgn = if n % 2 == 0 { 1.0 } else { -1.0 };
            for _ in -(n as isize)..=n as isize {
                ths[o] = th[o].scale(sgn);
                o += 1;
            }
        }
    }
    // A -> B: F_B[n][m] += Σ_k Σ_l M_A[k][l]* Θ[n+k][m+l](r)
    // B -> A: F_A[n][m] += Σ_k Σ_l M_B[k][l]* Θ[n+k][m+l](-r)
    // with Θ(-r) = (-1)^{n+k} Θ(r), folded into `ths`.
    unsafe {
        let thp = th.as_ptr();
        let thsp = ths.as_ptr();
        let cmap = cma.as_ptr();
        let cmbp = cmb.as_ptr();
        let lp = locals.as_mut_ptr();
        for n in 0..=p {
            let n2n = (n * (n + 1)) as isize; // idx(n, 0)
            for m in -(n as isize)..=n as isize {
                let nidx = (n2n + m) as usize;
                let mut fa_r = 0.0f64;
                let mut fa_i = 0.0f64; // into locals[a]
                let mut fb_r = 0.0f64;
                let mut fb_i = 0.0f64; // into locals[b]
                for k in 0..=(p - n) {
                    let s = n + k;
                    let sk = (s * (s + 1)) as isize; // idx(s, 0)
                    let kb = (k * (k + 1)) as isize; // idx(k, 0)
                    let ki = k as isize;
                    let thx0 = (sk + m - ki) as usize; // idx(s, m - k), l = -k
                    let mut thx = thx0;
                    let mut i = (kb - ki) as usize; // idx(k, -k)
                    for _ in -(k as isize)..=k as isize {
                        let t = *thp.add(thx);
                        let ts = *thsp.add(thx);
                        let ma = *cmap.add(i);
                        let mb = *cmbp.add(i);
                        // fb += ma * t
                        fb_r += ma.re * t.re - ma.im * t.im;
                        fb_i += ma.re * t.im + ma.im * t.re;
                        // fa += mb * ts
                        fa_r += mb.re * ts.re - mb.im * ts.im;
                        fa_i += mb.re * ts.im + mb.im * ts.re;
                        thx += 1;
                        i += 1;
                    }
                }
                let lb = lp.add(offb + nidx);
                *lb = (*lb).add(C::new(fb_r, fb_i));
                let la = lp.add(offa + nidx);
                *la = (*la).add(C::new(fa_r, fa_i));
            }
        }
    }
}

/// M2L for a single source cell a onto sink cell b (one-sided, used in tests).
pub fn m2l_one_sided(tree: &Octree, a: usize, b: usize, moments: &[C], locals: &mut [C], p: usize) {
    let d = dim(p);
    let ca = tree.cells[a];
    let cb = tree.cells[b];
    let r = [
        cb.cen[0] - ca.cen[0],
        cb.cen[1] - ca.cen[1],
        cb.cen[2] - ca.cen[2],
    ];
    let th = theta_table(r, p);
    let offa = a * d;
    let offb = b * d;
    for n in 0..=p {
        for m in -(n as i32)..=n as i32 {
            let nidx = idx(n, m);
            let mut fb = C::ZERO;
            for k in 0..=(p - n) {
                for l in -(k as i32)..=k as i32 {
                    let t = th[idx(n + k, m + l)];
                    fb = fb.add(moments[offa + idx(k, l)].conj().mul(t));
                }
            }
            locals[offb + nidx] = locals[offb + nidx].add(fb);
        }
    }
}

/// Pass a local expansion down to children (eq 3e).
/// `s` = parent centre, `s'` = child centre; r = s - s'.
pub fn l2l(tree: &Octree, parent: usize, child: usize, locals: &mut [C], p: usize) {
    let d = dim(p);
    let mut u = vec![C::ZERO; d];
    l2l_into(tree, parent, child, locals, p, &mut u);
}

/// Allocation-free `l2l` (`u` sized dim(p), overwritten per call).
pub fn l2l_into(
    tree: &Octree,
    parent: usize,
    child: usize,
    locals: &mut [C],
    p: usize,
    u: &mut [C],
) {
    let cp = tree.cells[parent];
    let cc = tree.cells[child];
    let r = [
        cp.cen[0] - cc.cen[0],
        cp.cen[1] - cc.cen[1],
        cp.cen[2] - cc.cen[2],
    ];
    upsilon_table_into(r, p, u);
    let offp = parent * dim(p);
    let offc = child * dim(p);
    for n in 0..=p {
        for m in -(n as i32)..=n as i32 {
            let nidx = idx(n, m);
            let mut s = C::ZERO;
            for k in 0..=(p - n) {
                for l in -(k as i32)..=k as i32 {
                    // Υ_k^{l*}(s-s') F_{n+k}^{m+l}(s)
                    s = s.add(u[idx(k, l)].conj().mul(locals[offp + idx(n + k, m + l)]));
                }
            }
            locals[offc + nidx] = locals[offc + nidx].add(s);
        }
    }
}

/// Full L2L downward sweep (parents before children in cell order).
pub fn l2l_down(tree: &Octree, locals: &mut [C], p: usize) {
    for c in 0..tree.cells.len() {
        let cell = tree.cells[c];
        for &ch in &cell.children {
            if ch >= 0 {
                l2l(tree, c, ch as usize, locals, p);
            }
        }
    }
}

/// Level-sweep parallel `l2l_down` (ascending level; parents at level l read
/// locals finalized at level l-1, children writes disjoint). Bit-identical to
/// the serial sweep.
pub fn l2l_down_par(tree: &Octree, locals: &mut [C], p: usize, nthreads: usize) {
    let (off, idx) = tree.levels_csr();
    let total = locals.len();
    for l in 0..off.len() - 1 {
        let cells = &idx[off[l]..off[l + 1]];
        let chunk = cells.len().div_ceil(nthreads.max(1) * 4).max(1);
        let base = crate::SendPtr(locals.as_mut_ptr());
        std::thread::scope(|s| {
            for piece in cells.chunks(chunk.max(1)) {
                let base = crate::SendPtr(base.ptr());
                let mut u = vec![C::ZERO; dim(p)];
                s.spawn(move || {
                    let locals: &mut [C] =
                        unsafe { std::slice::from_raw_parts_mut(base.ptr(), total) };
                    for &c in piece {
                        if tree.is_leaf(c) {
                            continue;
                        }
                        let cell = tree.cells[c];
                        for &ch in &cell.children {
                            if ch >= 0 {
                                l2l_into(tree, c, ch as usize, locals, p, &mut u);
                            }
                        }
                    }
                });
            }
        });
    }
}

/// Potential (Ψ_0^0) and acceleration at a position from a cell's local
/// expansion (eq 3f + §2.1). Adds into `pot` and `accel`.
pub fn l2p(tree: &Octree, cell: usize, pos: [f64; 3], locals: &[C], p: usize, pot: &mut f64, accel: &mut [f64; 3]) {
    let d = dim(p);
    let mut u = vec![C::ZERO; d];
    l2p_into(tree, cell, pos, locals, p, pot, accel, &mut u);
}

/// Allocation-free `l2p` (`u` sized dim(p), overwritten per call).
pub fn l2p_into(
    tree: &Octree,
    cell: usize,
    pos: [f64; 3],
    locals: &[C],
    p: usize,
    pot: &mut f64,
    accel: &mut [f64; 3],
    u: &mut [C],
) {
    let c = tree.cells[cell];
    let r = [
        c.cen[0] - pos[0],
        c.cen[1] - pos[1],
        c.cen[2] - pos[2],
    ];
    upsilon_table_into(r, p, u);
    let off = cell * dim(p);
    // Ψ_0^0
    let mut psi00 = C::ZERO;
    for k in 0..=p {
        for l in -(k as i32)..=k as i32 {
            psi00 = psi00.add(u[idx(k, l)].conj().mul(locals[off + idx(k, l)]));
        }
    }
    // Ψ_1^1, Ψ_1^0 (n = 1, so k runs 0..=p-1)
    let mut psi11 = C::ZERO;
    let mut psi10 = C::ZERO;
    if p >= 1 {
        for k in 0..=(p - 1) {
            for l in -(k as i32)..=k as i32 {
                let uc = u[idx(k, l)].conj();
                psi11 = psi11.add(uc.mul(locals[off + idx(1 + k, 1 + l)]));
                psi10 = psi10.add(uc.mul(locals[off + idx(1 + k, l)]));
            }
        }
    }
    *pot += psi00.re;
    // a = -(Re{Ψ_1^1}, Im{Ψ_1^1}, Ψ_1^0)
    accel[0] -= psi11.re;
    accel[1] -= psi11.im;
    accel[2] -= psi10.re;
}

/// L2P over all leaf particles + scatter slot results to particle-id order.
pub fn l2p_scatter(
    tree: &Octree,
    bodies: &crate::bodies::Bodies,
    locals: &[C],
    p: usize,
    pot: &mut [f64],
    accel: &mut [[f64; 3]],
) {
    let d = dim(p);
    let mut u = vec![C::ZERO; d];
    for c in 0..tree.cells.len() {
        if tree.is_leaf(c) {
            let cell = tree.cells[c];
            for k in cell.first..cell.first + cell.count {
                let id = tree.order[k] as usize;
                let mut p0 = 0.0;
                let mut a = [0.0; 3];
                l2p_into(
                    tree, c, [bodies.x[k], bodies.y[k], bodies.z[k]], locals, p, &mut p0,
                    &mut a, &mut u,
                );
                pot[id] += bodies.pot[k] + p0;
                accel[id][0] += bodies.ax[k] + a[0];
                accel[id][1] += bodies.ay[k] + a[1];
                accel[id][2] += bodies.az[k] + a[2];
            }
        }
    }
}

/// Parallel `l2p_scatter` over disjoint cell ranges. Each particle id lives
/// in exactly one leaf and each leaf in exactly one chunk, so the id-order
/// writes never collide across threads (raw pointers only carry that
/// invariant). Bit-identical to the serial pass.
pub fn l2p_scatter_par(
    tree: &Octree,
    bodies: &crate::bodies::Bodies,
    locals: &[C],
    p: usize,
    pot: &mut [f64],
    accel: &mut [[f64; 3]],
    nthreads: usize,
) {
    let ncells = tree.cells.len();
    let chunk = ncells.div_ceil(nthreads.max(1));
    let pot_ptr = crate::SendPtr(pot.as_mut_ptr());
    let accel_ptr = crate::SendPtr(accel.as_mut_ptr());
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(ncells);
            if hi <= lo {
                continue;
            }
            let pot_ptr = crate::SendPtr(pot_ptr.ptr());
            let accel_ptr = crate::SendPtr(accel_ptr.ptr());
            let mut u = vec![C::ZERO; dim(p)];
            s.spawn(move || {
                for c in lo..hi {
                    if !tree.is_leaf(c) {
                        continue;
                    }
                    let cell = tree.cells[c];
                    for k in cell.first..cell.first + cell.count {
                        let id = tree.order[k] as usize;
                        let mut p0 = 0.0;
                        let mut a = [0.0; 3];
                        l2p_into(
                            tree, c, [bodies.x[k], bodies.y[k], bodies.z[k]], locals, p,
                            &mut p0, &mut a, &mut u,
                        );
                        unsafe {
                            *pot_ptr.ptr().add(id) += bodies.pot[k] + p0;
                            let ac = accel_ptr.ptr().add(id);
                            (*ac)[0] += bodies.ax[k] + a[0];
                            (*ac)[1] += bodies.ay[k] + a[1];
                            (*ac)[2] += bodies.az[k] + a[2];
                        }
                    }
                }
            });
        }
    });
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::Bodies;
    use crate::models::plummer;

    #[test]
    fn l2l_l2p_par_match_serial() {
        let parts = plummer(2048, [0.0; 3], 17);
        let p = 8;
        let eps = 1e-5;
        let tree = crate::tree::Octree::build(&parts, 16);
        let d = dim(p);
        let ncells = tree.cells.len();
        let n = parts.len();
        let bodies = Bodies::from_parts(&parts, &tree.order);
        let mut moments = vec![C::ZERO; ncells * d];
        crate::moments::compute_moments(&bodies, &tree, p, &mut moments);
        let pows = crate::moments::multipole_power(&moments, ncells, p);
        let binom = crate::moments::binomials(p);
        let (_, fmin) = crate::mac::f_prepass(&bodies, &tree);
        let mut bw = bodies.clone();
        let mut l0 = vec![C::ZERO; ncells * d];
        crate::walk::dual_walk(
            &mut bw, &tree, p, eps, &pows, &binom, &fmin, &moments, &mut l0, None,
        );

        let mut l1 = l0.clone();
        l2l_down(&tree, &mut l1, p);
        let mut l2 = l0.clone();
        l2l_down_par(&tree, &mut l2, p, 8);
        assert_eq!(l1, l2);

        let (mut pot1, mut acc1) = (vec![0.0; n], vec![[0.0; 3]; n]);
        let (mut pot2, mut acc2) = (vec![0.0; n], vec![[0.0; 3]; n]);
        l2p_scatter(&tree, &bw, &l1, p, &mut pot1, &mut acc1);
        l2p_scatter_par(&tree, &bw, &l1, p, &mut pot2, &mut acc2, 8);
        assert_eq!(pot1, pot2);
        assert_eq!(acc1, acc2);
    }
}
