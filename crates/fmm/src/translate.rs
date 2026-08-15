//! Translation and evaluation kernels.
//!
//! M2L  (eq 3b): F_n^m(s_B) += Σ_{k=0}^{p-n} Σ_{l=-k}^k M_k^{l*}(z_A) Θ_{n+k}^{m+l}(s_B - z_A)
//! L2L  (eq 3e): F_n^m(s') += Σ_{k=0}^{p-n} Σ_l Υ_k^{l*}(s-s') F_{n+k}^{m+l}(s)
//! L2P  (eq 3f): Ψ_n^m(x_b) = Σ_{k=0}^{p-n} Σ_l Υ_k^{l*}(s_B-x_b) F_{n+k}^{m+l}(s_B)
//!
//! Acceleration (Dehnen 2014, §2.1): a = ∇Ψ_0^0 = -(Re{Ψ_1^1}, Im{Ψ_1^1}, Ψ_1^0).

use crate::harmonic::{idx, theta_table, upsilon_table, C, dim};
use crate::tree::Octree;

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
    let d = dim(p);
    let ca = tree.cells[a];
    let cb = tree.cells[b];
    // r = s_B - z_A, with z = s = expansion centre.
    let r = [
        cb.cen[0] - ca.cen[0],
        cb.cen[1] - ca.cen[1],
        cb.cen[2] - ca.cen[2],
    ];
    let th = theta_table(r, p);
    let offa = a * d;
    let offb = b * d;
    // Precompute conjugated moments once (they are constant across (n, m)),
    // and a copy of the Θ table pre-scaled by the parity (-1)^{order} so the
    // reverse direction needs no per-iteration sign.
    let mut cma = Vec::with_capacity(d);
    let mut cmb = Vec::with_capacity(d);
    for i in 0..d {
        cma.push(moments[offa + i].conj());
        cmb.push(moments[offb + i].conj());
    }
    let mut ths = th.clone();
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
    let cp = tree.cells[parent];
    let cc = tree.cells[child];
    let r = [
        cp.cen[0] - cc.cen[0],
        cp.cen[1] - cc.cen[1],
        cp.cen[2] - cc.cen[2],
    ];
    let u = upsilon_table(r, p);
    let offp = parent * d;
    let offc = child * d;
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

/// Potential (Ψ_0^0) and acceleration at a position from a cell's local
/// expansion (eq 3f + §2.1). Adds into `pot` and `accel`.
pub fn l2p(tree: &Octree, cell: usize, pos: [f64; 3], locals: &[C], p: usize, pot: &mut f64, accel: &mut [f64; 3]) {
    let d = dim(p);
    let c = tree.cells[cell];
    let r = [
        c.cen[0] - pos[0],
        c.cen[1] - pos[1],
        c.cen[2] - pos[2],
    ];
    let u = upsilon_table(r, p);
    let off = cell * d;
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