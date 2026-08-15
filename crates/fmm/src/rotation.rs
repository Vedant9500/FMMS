//! Rotation-accelerated M2L (O(p³) per pair, Dehnen axis-swap route).
//!
//! The M2L convolution is diagonal in the azimuthal index once the
//! interaction vector is rotated onto the z-axis:
//!
//!   F_n^m(B) = Σ_k M̃_k^{-m} · (n+k)!/R^{n+k+1}
//!
//! So per pair we: back-rotate the source multipoles, apply the axial M2L,
//! then forward-rotate the local expansion. Rotations act per degree as
//! block-diagonal matrices.
//!
//! Instead of building Wigner-d planes per pair (the previous Cheng/Dehnen
//! route), we use Dehnen (2014) Appendix A.6.1: any rotation decomposes into
//! a z-rotation, an x↔z axis swap, a second z-rotation, and another axis
//! swap. A z-rotation is a diagonal azimuth phase e^{imθ} (O(p²) work); the
//! axis swap is a fixed per-degree matrix B_n (O(p³) matvec work, built once
//! in `RotCtx::new`). The full alignment for g = R_z(α) R_y(β) is
//!
//!   D(g) = Z(α+π/2)·B·Z(-β)·B·Z(-π/2)   (verified to ~1e-13 in Python)
//!
//! acting on Θ coefficients, which is equivalent to the Wigner-D rotation
//! property Θ_n^m(gx) = Σ_{m'} (c^m/c^{m'}) e^{imα} d^n_{m m'}(β) Θ_n^{m'}(x)
//! (verified empirically). This removes the per-pair Wigner-d plane build
//! entirely: the hot path is two real swap matvecs plus diagonal phases per
//! degree, per direction.
//!
//! Multipoles are Dehnen Υ coefficients (P2M); the back-rotation therefore
//! applies the (n-|m|)!(n+|m|)! factors to enter Θ space, does the swap
//! matvecs, and divides back out. The ±π/2 phases and factorial factors are
//! folded into precomputed per-(n,m) diagonals; only the pair-dependent
//! phases e^{imα}, e^{imβ} are computed per pair.
//!
//! The swap matrices B_n (eq. 63) are real, satisfy B·B = I, and are
//! validated against the code's exact Θ basis (LSQ to 1e-13) and against the
//! naive M2L kernel (m2l_rot_matches_naive).

use crate::harmonic::{idx, C, dim};
use crate::tree::Octree;

/// Minimum p at which the rotation M2L beats the naive O(p⁴) kernel.
/// Naive ~ p⁴/6 complex MACs vs rotation ~ 4·Σ(2k+1)² real MACs ≈ 16p³/3.
/// Measured crossover after the 2026-08-15 axis-swap rewrite (no per-pair
/// Wigner-d plane build): p=24 1.03×, p=26 0.98×, p=30 0.89× (4000-pair
/// microbench, release). Relative error stays ≤ ~1e-8 up to p=32.
pub const ROT_P: usize = 26;

/// Total size of the block-layout plane for degrees 0..=p:
/// Σ_{j=0}^p (2j+1)² = (p+1)(4p²+8p+3)/3.
pub const fn blocksum(p: usize) -> usize {
    (p + 1) * (4 * p * p + 8 * p + 3) / 3
}

/// Offset of degree `j`'s block within the plane.
#[inline]
pub fn block_off(j: usize) -> usize {
    if j == 0 {
        0
    } else {
        j * (4 * j * j - 1) / 3
    }
}

/// Rotation context: per-p swap matrices + folded diagonals + scratch.
pub struct RotCtx {
    p: usize,
    fact: Vec<f64>,  // j! for j = 0..=p
    sw: Vec<f64>,    // swap matrices B_n, block layout (real)
    swt: Vec<f64>,   // B_n^T, block layout (real)
    in_: Vec<C>,     // back input diagonal: e^{-imπ/2}·(n-|m|)!(n+|m|)!
    out_: Vec<C>,    // back output diagonal: e^{imπ/2}/((n-|m|)!(n+|m|)!)
    fin: Vec<C>,     // forward input diagonal: e^{-imπ/2}/((n-|m|)!(n+|m|)!)
    fout: Vec<C>,    // forward output diagonal base: e^{imπ/2}·(n-|m|)!(n+|m|)!
    b: Vec<C>,       // back-rotated source multipoles, dim(p)
    f: Vec<C>,       // axial output, dim(p)
    tmp: Vec<C>,     // per-degree scratch (max size 2p+1)
    tmp2: Vec<C>,    // per-degree scratch 2 (max size 2p+1)
    pha: Vec<C>,     // pair phase e^{imα} (or (-1)^m e^{imα}), size 2p+1
    phb: Vec<C>,     // pair phase e^{imβ} (or (-1)^m e^{-imβ}), size 2p+1
    facr: Vec<f64>,  // j!/R^{j+1}, size p+1
}

impl RotCtx {
    pub fn new(p: usize) -> Self {
        assert!(p <= 60, "precomputed rotation tables only for p <= 60");
        let d = dim(p);
        let mut fact = vec![1.0; p + 1];
        for i in 1..=p {
            fact[i] = fact[i - 1] * i as f64;
        }
        let mut f2 = vec![1.0; 2 * p + 1];
        for i in 1..f2.len() {
            f2[i] = f2[i - 1] * i as f64;
        }
        let bs = blocksum(p);
        // Build the real swap matrices B_n (Dehnen eq. 63) in block layout,
        // plus their transposes. B_0 = [1]; rows M ∈ [-n, n], cols L ∈ [-n, n].
        let mut sw = vec![0.0; bs];
        let mut swt = vec![0.0; bs];
        sw[0] = 1.0;
        swt[0] = 1.0;
        for n in 1..=p {
            let nb = 2 * n + 1;
            let off = block_off(n);
            let nb_prev = 2 * n - 1;
            let off_prev = block_off(n - 1);
            for mr in 0..nb {
                let m = mr as i32 - n as i32;
                let row = off + mr * nb;
                let row_prev = off_prev + mr.saturating_sub(1) * nb_prev;
                for lc in 0..nb {
                    let l = lc as i32 - n as i32;
                    let v = if m.unsigned_abs() as usize <= n - 1 {
                        // eq 63a: interior rows use source row M.
                        let a = if (l - 1).unsigned_abs() as usize <= n - 1 {
                            sw[row_prev + ((l - 1) + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        let b = if (l + 1).unsigned_abs() as usize <= n - 1 {
                            sw[row_prev + ((l + 1) + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        (a - b) * 0.5
                    } else if m == n as i32 {
                        // eq 63b (+): boundary row M = n uses source row n-1,
                        // which sits at prev-block row index 2n-2.
                        let src = off_prev + (2 * n - 2) * nb_prev;
                        let a = if (l - 1).unsigned_abs() as usize <= n - 1 {
                            sw[src + ((l - 1) + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        let b = if (l + 1).unsigned_abs() as usize <= n - 1 {
                            sw[src + ((l + 1) + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        let c = if l.unsigned_abs() as usize <= n - 1 {
                            sw[src + (l + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        (a + b + 2.0 * c) * 0.5
                    } else {
                        // eq 63b (−): boundary row M = -n uses source row -n+1.
                        let src = off_prev + 0 * nb_prev;
                        let a = if (l - 1).unsigned_abs() as usize <= n - 1 {
                            sw[src + ((l - 1) + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        let b = if (l + 1).unsigned_abs() as usize <= n - 1 {
                            sw[src + ((l + 1) + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        let c = if l.unsigned_abs() as usize <= n - 1 {
                            sw[src + (l + (n - 1) as i32) as usize]
                        } else {
                            0.0
                        };
                        (a + b - 2.0 * c) * 0.5
                    };
                    sw[row + lc] = v;
                    swt[off + lc * nb + mr] = v; // transpose
                }
            }
        }
        // Folded diagonals. idx(n,m) = n² + (m+n). For azimuth m:
        //   e^{imπ/2} = i^m,  e^{-imπ/2} = (-i)^m.
        let mut in_ = vec![C::ZERO; d];
        let mut out_ = vec![C::ZERO; d];
        let mut fin = vec![C::ZERO; d];
        let mut fout = vec![C::ZERO; d];
        for n in 0..=p {
            for m in -(n as i32)..=n as i32 {
                let i = idx(n, m);
                let am = m.unsigned_abs() as usize;
                let fm = f2[n - am] * f2[n + am];
                // e^{-imπ/2}
                let ei = match m.rem_euclid(4) {
                    0 => C::new(1.0, 0.0),
                    1 => C::new(0.0, -1.0),
                    2 => C::new(-1.0, 0.0),
                    _ => C::new(0.0, 1.0),
                };
                in_[i] = ei.scale(fm);      // back input
                out_[i] = ei.conj().scale(1.0 / fm); // back output
                fin[i] = ei.scale(1.0 / fm); // forward input
                fout[i] = ei.conj().scale(fm); // forward output base
            }
        }
        RotCtx {
            p,
            fact,
            sw,
            swt,
            in_,
            out_,
            fin,
            fout,
            b: vec![C::ZERO; d],
            f: vec![C::ZERO; d],
            tmp: vec![C::ZERO; 2 * p + 1],
            tmp2: vec![C::ZERO; 2 * p + 1],
            pha: vec![C::ZERO; 2 * p + 1],
            phb: vec![C::ZERO; 2 * p + 1],
            facr: vec![0.0; p + 1],
        }
    }

    /// Mutual rotation M2L for a well-separated pair (a, b).
    ///
    /// Per pair: compute the pair phases e^{imα}, e^{imβ}, then per direction
    /// back-rotate + axial M2L + forward-rotate. The back-rotation is two real
    /// swap matvecs with a diagonal β phase between; the forward-rotation is
    /// the same with B^T and conjugated β phase. No allocations on the hot
    /// path.
    pub fn m2l_pair(
        &mut self,
        tree: &Octree,
        a: usize,
        b: usize,
        moments: &[C],
        locals: &mut [C],
    ) {
        let p = self.p;
        let d = dim(p);
        let ca = tree.cells[a];
        let cb = tree.cells[b];
        let r = [cb.cen[0] - ca.cen[0], cb.cen[1] - ca.cen[1], cb.cen[2] - ca.cen[2]];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let rinv = 1.0 / r2.sqrt();
        let alpha = r[1].atan2(r[0]);
        let s_beta = (r[0] * r[0] + r[1] * r[1]).sqrt();
        let beta = s_beta.atan2(r[2]);

        // (n+k)!/R^{n+k+1}
        let mut irp = rinv;
        for j in 0..=p {
            self.facr[j] = self.fact[j] * irp;
            irp *= rinv;
        }
        // Direction A→B: phases e^{imα}, e^{imβ}.
        for m in -(p as i32)..=p as i32 {
            let a = m as f64 * alpha;
            self.pha[(m + p as i32) as usize] = C::new(a.cos(), a.sin());
            let bb = m as f64 * beta;
            self.phb[(m + p as i32) as usize] = C::new(bb.cos(), bb.sin());
        }
        self.apply(a * d, b * d, moments, locals);
        // Direction B→A sees r → -r: azimuth α+π and β → π-β, so
        // pha → (-1)^m e^{imα}, phb → (-1)^m e^{-imβ}.
        for i in 0..self.pha.len() {
            let m = i as i32 - p as i32;
            let sgn = if m.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            self.pha[i] = self.pha[i].scale(sgn);
            self.phb[i] = self.phb[i].conj().scale(sgn);
        }
        self.apply(b * d, a * d, moments, locals);
    }

    /// One direction: far-field of source block `so` -> local block `do`.
    ///
    ///   1) back-rotate: for each degree k,
    ///        u[m] = in_[k,m]·pha[m]·conj(M[k,m])
    ///        s    = B_k · u            (real swap matvec)
    ///        q[m] = phb[m]·s[m]
    ///        r    = B_k · q
    ///        b[m] = out_[k,m]·r[m]
    ///   2) axial M2L: F̃[n][m] = Σ_k b[idx(k,-m)] · facr[n+k]
    ///   3) forward: for each degree n,
    ///        g[m] = fin[n,m]·F̃[n,m]
    ///        s    = B_n^T · g
    ///        q[m] = conj(phb[m])·s[m]
    ///        r    = B_n^T · q
    ///        out[l] += fout[n,l]·pha[l]·r[l]
    #[inline]
    fn apply(
        &mut self,
        so: usize,
        do_: usize,
        moments: &[C],
        locals: &mut [C],
    ) {
        let p = self.p;
        let b = &mut self.b;
        let f = &mut self.f;
        let facr = &self.facr;
        unsafe {
            // 1) back-rotate.
            for k in 0..=p {
                let nb = 2 * k + 1;
                let k2 = k * k;
                let mptr = moments.as_ptr().add(so + k2);
                let in_ = &self.in_;
                let out_ = &self.out_;
                let sw = &self.sw;
                let pha = &self.pha;
                let phb = &self.phb;
                let off = block_off(k);
                let po = p - k; // offset so pha[po + c] is azimuth (c-k)+p
                // u[m] = in_[k,m]·pha[m]·conj(M[k,m])  -> tmp
                for c in 0..nb {
                    let m = *mptr.add(c);
                    let ph = *pha.get_unchecked(po + c);
                    let iv = *in_.get_unchecked(k2 + c);
                    // conj(M)·ph·in : (m.re, -m.im) · (ph.re,ph.im) · iv
                    let vr = m.re * ph.re + m.im * ph.im;
                    let vi = m.re * ph.im - m.im * ph.re;
                    *self.tmp.get_unchecked_mut(c) =
                        C::new(vr * iv.re - vi * iv.im, vr * iv.im + vi * iv.re);
                }
                // s = B_k · u  (real matrix × complex vector) -> tmp2
                for r in 0..nb {
                    let prow = sw.as_ptr().add(off + r * nb);
                    let mut sr = 0.0f64;
                    let mut si = 0.0f64;
                    for c in 0..nb {
                        let w = *prow.add(c);
                        let u = *self.tmp.get_unchecked(c);
                        sr += w * u.re;
                        si += w * u.im;
                    }
                    *self.tmp2.get_unchecked_mut(r) = C::new(sr, si);
                }
                // q[m] = phb[m]·s[m]  (in place on tmp2)
                for r in 0..nb {
                    let pbb = *phb.get_unchecked(po + r);
                    let s = *self.tmp2.get_unchecked(r);
                    *self.tmp2.get_unchecked_mut(r) = C::new(
                        s.re * pbb.re - s.im * pbb.im,
                        s.re * pbb.im + s.im * pbb.re,
                    );
                }
                // r = B_k · q  -> tmp;  b[m] = out_[k,m]·r[m]
                let bptr = b.as_mut_ptr().add(k2);
                for r in 0..nb {
                    let prow = sw.as_ptr().add(off + r * nb);
                    let mut sr = 0.0f64;
                    let mut si = 0.0f64;
                    for c in 0..nb {
                        let w = *prow.add(c);
                        let q = *self.tmp2.get_unchecked(c);
                        sr += w * q.re;
                        si += w * q.im;
                    }
                    let ov = *out_.get_unchecked(k2 + r);
                    *bptr.add(r) = C::new(sr * ov.re - si * ov.im, sr * ov.im + si * ov.re);
                }
            }
            // 2) axial M2L: F̃[n][m] = Σ_{k=|m|}^{p-n} b[idx(k,-m)] · facr[n+k]
            for n in 0..=p {
                let nb = 2 * n + 1;
                let n2 = n * n;
                let fptr = f.as_mut_ptr().add(n2);
                for r in 0..nb {
                    let m = r as i32 - n as i32;
                    let am = m.unsigned_abs() as usize;
                    let mut fr = 0.0f64;
                    let mut fi = 0.0f64;
                    for k in am..=(p - n) {
                        let bv = *b.get_unchecked(k * k + (k as i32 - m) as usize);
                        let fv = facr[n + k];
                        fr += bv.re * fv;
                        fi += bv.im * fv;
                    }
                    *fptr.add(r) = C::new(fr, fi);
                }
            }
            // 3) forward-rotate into locals.
            let swt = &self.swt;
            let fin = &self.fin;
            let fout = &self.fout;
            let pha = &self.pha;
            let phb = &self.phb;
            for n in 0..=p {
                let nb = 2 * n + 1;
                let off = block_off(n);
                let n2 = n * n;
                let po = p - n; // offset so pha[po + c] is azimuth (c-n)+p
                let lptr = locals.as_mut_ptr().add(do_ + n2);
                let fptr = f.as_ptr().add(n2);
                // g[m] = fin[n,m]·F̃[n,m]  -> tmp
                for c in 0..nb {
                    let fv = *fptr.add(c);
                    let iv = *fin.get_unchecked(n2 + c);
                    *self.tmp.get_unchecked_mut(c) = C::new(
                        fv.re * iv.re - fv.im * iv.im,
                        fv.re * iv.im + fv.im * iv.re,
                    );
                }
                // s = B_n^T · g  -> tmp2
                for r in 0..nb {
                    let prow = swt.as_ptr().add(off + r * nb);
                    let mut sr = 0.0f64;
                    let mut si = 0.0f64;
                    for c in 0..nb {
                        let w = *prow.add(c);
                        let g = *self.tmp.get_unchecked(c);
                        sr += w * g.re;
                        si += w * g.im;
                    }
                    *self.tmp2.get_unchecked_mut(r) = C::new(sr, si);
                }
                // q = conj(phb)·s  (in place on tmp2)
                for r in 0..nb {
                    let pb = *phb.get_unchecked(po + r);
                    let s = *self.tmp2.get_unchecked(r);
                    // conj(phb)·s : (pb.re, -pb.im)·(s.re, s.im)
                    *self.tmp2.get_unchecked_mut(r) = C::new(
                        s.re * pb.re + s.im * pb.im,
                        s.im * pb.re - s.re * pb.im,
                    );
                }
                // r = B_n^T · q  -> tmp;  out[l] += fout·pha·r
                for r in 0..nb {
                    let prow = swt.as_ptr().add(off + r * nb);
                    let mut sr = 0.0f64;
                    let mut si = 0.0f64;
                    for c in 0..nb {
                        let w = *prow.add(c);
                        let q = *self.tmp2.get_unchecked(c);
                        sr += w * q.re;
                        si += w * q.im;
                    }
                    let ov = *fout.get_unchecked(n2 + r);
                    let pv = *pha.get_unchecked(po + r);
                    let acc = C::new(sr * ov.re - si * ov.im, sr * ov.im + si * ov.re);
                    let cur = *lptr.add(r);
                    let add = C::new(
                        acc.re * pv.re - acc.im * pv.im,
                        acc.re * pv.im + acc.im * pv.re,
                    );
                    *lptr.add(r) = C::new(cur.re + add.re, cur.im + add.im);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmonic::theta_table;
    use crate::models::Particle;
    use crate::models::Rng;

    #[test]
    fn swap_involution() {
        // B·B = I for every degree (Dehnen eq. 63 property).
        for n in 0..=10 {
            let nb = 2 * n + 1;
            let off = block_off(n);
            let mut b = vec![0.0; blocksum(n)];
            b[0] = 1.0;
            // reuse the same construction as RotCtx::new for this degree set
            let ctx = RotCtx::new(n);
            for i in 0..nb * nb {
                b[off + i] = ctx.sw[off + i];
            }
            for r in 0..nb {
                for c in 0..nb {
                    let mut s = 0.0;
                    for k in 0..nb {
                        s += b[off + r * nb + k] * b[off + k * nb + c];
                    }
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (s - want).abs() < 1e-9,
                        "n={n} r={r} c={c}: (B·B)[rc]={s} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn swap_rotates_theta() {
        // Θ_n^m(P x) = Σ_l B_n^{ml} Θ_n^l(x) with P the x↔z swap, in the
        // code's exact Θ basis. Direct check against theta_table.
        let p = 6;
        let ctx = RotCtx::new(p);
        let x = [0.6, -0.9, 1.3];
        let px = [x[2], x[1], x[0]];
        let t = theta_table(x, p);
        let tp = theta_table(px, p);
        for n in 0..=p {
            let nb = 2 * n + 1;
            let off = block_off(n);
            for r in 0..nb {
                let m = r as i32 - n as i32;
                let mut s = C::ZERO;
                for c in 0..nb {
                    let l = c as i32 - n as i32;
                    let w = ctx.sw[off + r * nb + c];
                    s = s.add(t[idx(n, l)].scale(w));
                }
                let got = tp[idx(n, m)];
                assert!(
                    (got.re - s.re).abs() < 1e-9 && (got.im - s.im).abs() < 1e-9,
                    "swap n={n} m={m}: {got:?} vs {s:?}"
                );
            }
        }
    }

    #[test]
    fn m2l_rot_matches_naive() {
        let p = 8;
        let parts: Vec<Particle> = (0..128)
            .map(|i| {
                let f = i as f64 * 0.1;
                Particle { mass: 1.0, pos: [f, f, f] }
            })
            .collect();
        let tree = crate::tree::Octree::build(&parts, 4);
        let (mut a, mut b) = (0usize, 0usize);
        for c in 0..tree.cells.len() {
            if tree.is_leaf(c) {
                if a == 0 {
                    a = c;
                } else if b == 0 {
                    b = c;
                } else {
                    break;
                }
            }
        }
        assert!(a != b);
        let d = dim(p);
        let mut moments = vec![C::ZERO; tree.cells.len() * d];
        let mut rng = crate::models::Rng::new(11);
        for i in 0..d {
            moments[a * d + i] = C::new(rng.unit() - 0.5, rng.unit() - 0.5);
            moments[b * d + i] = C::new(rng.unit() - 0.5, rng.unit() - 0.5);
        }
        let mut loc_naive = vec![C::ZERO; tree.cells.len() * d];
        let mut loc_rot = vec![C::ZERO; tree.cells.len() * d];
        crate::translate::m2l_mutual(&tree, a, b, &moments, &mut loc_naive, p);
        let mut ctx = RotCtx::new(p);
        ctx.m2l_pair(&tree, a, b, &moments, &mut loc_rot);
        for c in [a, b] {
            for i in 0..d {
                let (r, n) = (loc_rot[c * d + i], loc_naive[c * d + i]);
                let tol = 1e-9 * (n.re.abs() + n.im.abs()).max(1e-30);
                assert!(
                    (r.re - n.re).abs() < tol && (r.im - n.im).abs() < tol,
                    "cell {c} idx {i}: rot ({},{}) vs naive ({},{})",
                    r.re,
                    r.im,
                    n.re,
                    n.im
                );
            }
        }
    }
}
