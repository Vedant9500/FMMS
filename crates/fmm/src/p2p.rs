//! P2P direct kernel: mutual Newton's-third-law-exact pair summation.
//!
//! SoA slots, FP64 accumulation, rsqrt+NR reciprocal sqrt. Two paths:
//!   - scalar (portable, exact reference)
//!   - AVX-512 (8 FP64 lanes/iter, runtime-dispatched)
//! Both compute the *same* mutual update: a_i += m_j·d/r³, a_j -= m_i·d/r³
//! with the identical dx/dy/dz and inv3 on both sides, so the total force
//! vanishes to machine precision in either path.

use crate::bodies::Bodies;

/// Scalar mutual kernel for one leaf cell (i < j pairs).
pub fn p2p_self_scalar(b: &mut Bodies, first: usize, count: usize) {
    for i in 0..count {
        let xi = b.x[first + i];
        let yi = b.y[first + i];
        let zi = b.z[first + i];
        let mi = b.m[first + i];
        let mut axi = 0.0;
        let mut ayi = 0.0;
        let mut azi = 0.0;
        let mut poti = 0.0;
        for j in i + 1..count {
            let dx = b.x[first + j] - xi;
            let dy = b.y[first + j] - yi;
            let dz = b.z[first + j] - zi;
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv = 1.0 / r2.sqrt();
            let inv3 = inv * inv * inv;
            let mj = b.m[first + j];
            axi += mj * inv3 * dx;
            ayi += mj * inv3 * dy;
            azi += mj * inv3 * dz;
            poti += mj * inv;
            let fm = mi * inv3;
            b.ax[first + j] -= fm * dx;
            b.ay[first + j] -= fm * dy;
            b.az[first + j] -= fm * dz;
            b.pot[first + j] += mi * inv;
        }
        b.ax[first + i] += axi;
        b.ay[first + i] += ayi;
        b.az[first + i] += azi;
        b.pot[first + i] += poti;
    }
}

/// Scalar mutual kernel between two leaf cells (all i × j pairs).
pub fn p2p_pair_scalar(
    b: &mut Bodies,
    a_first: usize,
    a_count: usize,
    b_first: usize,
    b_count: usize,
) {
    for i in 0..a_count {
        let xi = b.x[a_first + i];
        let yi = b.y[a_first + i];
        let zi = b.z[a_first + i];
        let mi = b.m[a_first + i];
        let mut axi = 0.0;
        let mut ayi = 0.0;
        let mut azi = 0.0;
        let mut poti = 0.0;
        for j in 0..b_count {
            let dx = b.x[b_first + j] - xi;
            let dy = b.y[b_first + j] - yi;
            let dz = b.z[b_first + j] - zi;
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv = 1.0 / r2.sqrt();
            let inv3 = inv * inv * inv;
            let mj = b.m[b_first + j];
            axi += mj * inv3 * dx;
            ayi += mj * inv3 * dy;
            azi += mj * inv3 * dz;
            poti += mj * inv;
            let fm = mi * inv3;
            b.ax[b_first + j] -= fm * dx;
            b.ay[b_first + j] -= fm * dy;
            b.az[b_first + j] -= fm * dz;
            b.pot[b_first + j] += mi * inv;
        }
        b.ax[a_first + i] += axi;
        b.ay[a_first + i] += ayi;
        b.az[a_first + i] += azi;
        b.pot[a_first + i] += poti;
    }
}

/// Scalar mutual update for a single pair; shared by the AVX-512 kernels' tails.
#[inline(always)]
fn scalar_mutual(b: &mut Bodies, first: usize, i: usize, j: usize) {
    let dx = b.x[first + j] - b.x[first + i];
    let dy = b.y[first + j] - b.y[first + i];
    let dz = b.z[first + j] - b.z[first + i];
    let r2 = dx * dx + dy * dy + dz * dz;
    let inv = 1.0 / r2.sqrt();
    let inv3 = inv * inv * inv;
    let mi = b.m[first + i];
    let mj = b.m[first + j];
    b.ax[first + i] += mj * inv3 * dx;
    b.ay[first + i] += mj * inv3 * dy;
    b.az[first + i] += mj * inv3 * dz;
    b.pot[first + i] += mj * inv;
    let fm = mi * inv3;
    b.ax[first + j] -= fm * dx;
    b.ay[first + j] -= fm * dy;
    b.az[first + j] -= fm * dz;
    b.pot[first + j] += mi * inv;
}

/// AVX-512 FP64 kernel for one leaf cell, i-unrolled 2. j-lanes start at i+1 so
/// every lane is a valid pair (no per-block mask needed); the intra-pair (i0,i1)
/// and the unaligned tail run scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn p2p_self_avx512(b: &mut Bodies, first: usize, count: usize) {
    use std::arch::x86_64::*;
    let three = _mm512_set1_pd(3.0);
    let half = _mm512_set1_pd(0.5);
    let zero = _mm512_setzero_pd();
    let mut i = 0usize;
    while i + 2 <= count {
        let i0 = i;
        let i1 = i + 1;
        let x0 = _mm512_set1_pd(b.x[first + i0]);
        let y0 = _mm512_set1_pd(b.y[first + i0]);
        let z0 = _mm512_set1_pd(b.z[first + i0]);
        let x1 = _mm512_set1_pd(b.x[first + i1]);
        let y1 = _mm512_set1_pd(b.y[first + i1]);
        let z1 = _mm512_set1_pd(b.z[first + i1]);
        let m0 = _mm512_set1_pd(b.m[first + i0]);
        let m1 = _mm512_set1_pd(b.m[first + i1]);
        let mut a0x = zero;
        let mut a0y = zero;
        let mut a0z = zero;
        let mut p0 = zero;
        let mut a1x = zero;
        let mut a1y = zero;
        let mut a1z = zero;
        let mut p1 = zero;
        scalar_mutual(b, first, i0, i1);
        let mut jj = i1 + 1;
        while jj + 8 <= count {
            let o = jj;
            let xb = _mm512_loadu_pd(b.x.as_ptr().add(first + o));
            let yb = _mm512_loadu_pd(b.y.as_ptr().add(first + o));
            let zb = _mm512_loadu_pd(b.z.as_ptr().add(first + o));
            let mb = _mm512_loadu_pd(b.m.as_ptr().add(first + o));
            let dx0 = _mm512_sub_pd(xb, x0);
            let dy0 = _mm512_sub_pd(yb, y0);
            let dz0 = _mm512_sub_pd(zb, z0);
            let r2 = _mm512_fmadd_pd(dz0, dz0, _mm512_fmadd_pd(dy0, dy0, _mm512_mul_pd(dx0, dx0)));
            let inv = _mm512_rsqrt14_pd(r2);
            let inv = _mm512_mul_pd(_mm512_mul_pd(inv, _mm512_fnmadd_pd(_mm512_mul_pd(inv, inv), r2, three)), half);
            let inv = _mm512_mul_pd(_mm512_mul_pd(inv, _mm512_fnmadd_pd(_mm512_mul_pd(inv, inv), r2, three)), half);
            let inv3 = _mm512_mul_pd(_mm512_mul_pd(inv, inv), inv);
            let f = _mm512_mul_pd(mb, inv3);
            a0x = _mm512_fmadd_pd(f, dx0, a0x);
            a0y = _mm512_fmadd_pd(f, dy0, a0y);
            a0z = _mm512_fmadd_pd(f, dz0, a0z);
            p0 = _mm512_fmadd_pd(mb, inv, p0);
            let fm0 = _mm512_mul_pd(m0, inv3);
            let dx1 = _mm512_sub_pd(xb, x1);
            let dy1 = _mm512_sub_pd(yb, y1);
            let dz1 = _mm512_sub_pd(zb, z1);
            let r2b = _mm512_fmadd_pd(dz1, dz1, _mm512_fmadd_pd(dy1, dy1, _mm512_mul_pd(dx1, dx1)));
            let invb = _mm512_rsqrt14_pd(r2b);
            let invb = _mm512_mul_pd(_mm512_mul_pd(invb, _mm512_fnmadd_pd(_mm512_mul_pd(invb, invb), r2b, three)), half);
            let invb = _mm512_mul_pd(_mm512_mul_pd(invb, _mm512_fnmadd_pd(_mm512_mul_pd(invb, invb), r2b, three)), half);
            let inv3b = _mm512_mul_pd(_mm512_mul_pd(invb, invb), invb);
            let fb = _mm512_mul_pd(mb, inv3b);
            a1x = _mm512_fmadd_pd(fb, dx1, a1x);
            a1y = _mm512_fmadd_pd(fb, dy1, a1y);
            a1z = _mm512_fmadd_pd(fb, dz1, a1z);
            p1 = _mm512_fmadd_pd(mb, invb, p1);
            let fm1 = _mm512_mul_pd(m1, inv3b);
            let axj = _mm512_loadu_pd(b.ax.as_ptr().add(first + o));
            _mm512_storeu_pd(b.ax.as_mut_ptr().add(first + o), _mm512_fnmadd_pd(fm0, dx0, _mm512_fnmadd_pd(fm1, dx1, axj)));
            let ayj = _mm512_loadu_pd(b.ay.as_ptr().add(first + o));
            _mm512_storeu_pd(b.ay.as_mut_ptr().add(first + o), _mm512_fnmadd_pd(fm0, dy0, _mm512_fnmadd_pd(fm1, dy1, ayj)));
            let azj = _mm512_loadu_pd(b.az.as_ptr().add(first + o));
            _mm512_storeu_pd(b.az.as_mut_ptr().add(first + o), _mm512_fnmadd_pd(fm0, dz0, _mm512_fnmadd_pd(fm1, dz1, azj)));
            let pj = _mm512_loadu_pd(b.pot.as_ptr().add(first + o));
            _mm512_storeu_pd(b.pot.as_mut_ptr().add(first + o), _mm512_fmadd_pd(m0, inv, _mm512_fmadd_pd(m1, invb, pj)));
            jj += 8;
        }
        for j in jj..count {
            scalar_mutual(b, first, i0, j);
            scalar_mutual(b, first, i1, j);
        }
        b.ax[first + i0] += hsum512(a0x);
        b.ay[first + i0] += hsum512(a0y);
        b.az[first + i0] += hsum512(a0z);
        b.pot[first + i0] += hsum512(p0);
        b.ax[first + i1] += hsum512(a1x);
        b.ay[first + i1] += hsum512(a1y);
        b.az[first + i1] += hsum512(a1z);
        b.pot[first + i1] += hsum512(p1);
        i += 2;
    }
    for i in i..count {
        for j in i + 1..count {
            scalar_mutual(b, first, i, j);
        }
    }
}

/// AVX-512 FP64 kernel between two leaf cells (all i × j pairs).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn p2p_pair_avx512(
    b: &mut Bodies,
    a_first: usize,
    a_count: usize,
    b_first: usize,
    b_count: usize,
) {
    use std::arch::x86_64::*;
    let three = _mm512_set1_pd(3.0);
    let half = _mm512_set1_pd(0.5);
    let zero = _mm512_setzero_pd();
    for i in 0..a_count {
        let xi = _mm512_set1_pd(b.x[a_first + i]);
        let yi = _mm512_set1_pd(b.y[a_first + i]);
        let zi = _mm512_set1_pd(b.z[a_first + i]);
        let mi = b.m[a_first + i];
        let mut axacc = zero;
        let mut ayacc = zero;
        let mut azacc = zero;
        let mut potacc = zero;
        let mut jj = 0usize;
        while jj + 8 <= b_count {
            let xb = _mm512_loadu_pd(b.x.as_ptr().add(b_first + jj));
            let yb = _mm512_loadu_pd(b.y.as_ptr().add(b_first + jj));
            let zb = _mm512_loadu_pd(b.z.as_ptr().add(b_first + jj));
            let mb = _mm512_loadu_pd(b.m.as_ptr().add(b_first + jj));
            let dx = _mm512_sub_pd(xb, xi);
            let dy = _mm512_sub_pd(yb, yi);
            let dz = _mm512_sub_pd(zb, zi);
            let r2 = _mm512_fmadd_pd(dz, dz, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dx, dx)));
            let inv = _mm512_rsqrt14_pd(r2);
            let inv = _mm512_mul_pd(_mm512_mul_pd(inv, _mm512_fnmadd_pd(_mm512_mul_pd(inv, inv), r2, three)), half);
            let inv = _mm512_mul_pd(_mm512_mul_pd(inv, _mm512_fnmadd_pd(_mm512_mul_pd(inv, inv), r2, three)), half);
            let inv3 = _mm512_mul_pd(_mm512_mul_pd(inv, inv), inv);
            let f = _mm512_mul_pd(mb, inv3);
            axacc = _mm512_fmadd_pd(f, dx, axacc);
            ayacc = _mm512_fmadd_pd(f, dy, ayacc);
            azacc = _mm512_fmadd_pd(f, dz, azacc);
            potacc = _mm512_fmadd_pd(mb, inv, potacc);
            let fm = _mm512_set1_pd(mi);
            let fm = _mm512_mul_pd(fm, inv3);
            let axj = _mm512_loadu_pd(b.ax.as_ptr().add(b_first + jj));
            let axj = _mm512_fnmadd_pd(fm, dx, axj);
            _mm512_storeu_pd(b.ax.as_mut_ptr().add(b_first + jj), axj);
            let ayj = _mm512_loadu_pd(b.ay.as_ptr().add(b_first + jj));
            let ayj = _mm512_fnmadd_pd(fm, dy, ayj);
            _mm512_storeu_pd(b.ay.as_mut_ptr().add(b_first + jj), ayj);
            let azj = _mm512_loadu_pd(b.az.as_ptr().add(b_first + jj));
            let azj = _mm512_fnmadd_pd(fm, dz, azj);
            _mm512_storeu_pd(b.az.as_mut_ptr().add(b_first + jj), azj);
            let pj = _mm512_loadu_pd(b.pot.as_ptr().add(b_first + jj));
            let pj = _mm512_fmadd_pd(_mm512_set1_pd(mi), inv, pj);
            _mm512_storeu_pd(b.pot.as_mut_ptr().add(b_first + jj), pj);
            jj += 8;
        }
        for j in jj..b_count {
            let dx = b.x[b_first + j] - b.x[a_first + i];
            let dy = b.y[b_first + j] - b.y[a_first + i];
            let dz = b.z[b_first + j] - b.z[a_first + i];
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv = 1.0 / r2.sqrt();
            let inv3 = inv * inv * inv;
            let mj = b.m[b_first + j];
            b.ax[a_first + i] += mj * inv3 * dx;
            b.ay[a_first + i] += mj * inv3 * dy;
            b.az[a_first + i] += mj * inv3 * dz;
            b.pot[a_first + i] += mj * inv;
            let fm = mi * inv3;
            b.ax[b_first + j] -= fm * dx;
            b.ay[b_first + j] -= fm * dy;
            b.az[b_first + j] -= fm * dz;
            b.pot[b_first + j] += mi * inv;
        }
        b.ax[a_first + i] += hsum512(axacc);
        b.ay[a_first + i] += hsum512(ayacc);
        b.az[a_first + i] += hsum512(azacc);
        b.pot[a_first + i] += hsum512(potacc);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum512(v: std::arch::x86_64::__m512d) -> f64 {
    use std::arch::x86_64::*;
    _mm512_reduce_add_pd(v)
}

/// Mutual P2P within a leaf cell. Runtime-dispatch to AVX-512 when available.
pub fn p2p_self(b: &mut Bodies, first: usize, count: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe {
                p2p_self_avx512(b, first, count);
            }
            return;
        }
    }
    p2p_self_scalar(b, first, count);
}

/// Mutual P2P between two leaf cells. Runtime-dispatch to AVX-512.
pub fn p2p_pair(b: &mut Bodies, a_first: usize, a_count: usize, b_first: usize, b_count: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe {
                p2p_pair_avx512(b, a_first, a_count, b_first, b_count);
            }
            return;
        }
    }
    p2p_pair_scalar(b, a_first, a_count, b_first, b_count);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::plummer;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let n = 24;
        let parts = plummer(n, [0.0; 3], 5);
        let tree = crate::tree::Octree::build(&parts, 16);
        let b = Bodies::from_parts(&parts, &tree.order);
        let mut ref_ = b.clone();
        let mut simd = b.clone();
        p2p_self_scalar(&mut ref_, 0, n);
        unsafe { p2p_self_avx512(&mut simd, 0, n) };
        for k in 0..n {
            let tol = 1e-12 * (1.0 + ref_.ax[k].abs() + simd.ax[k].abs());
            assert!((ref_.ax[k] - simd.ax[k]).abs() < tol, "ax[{k}]");
            assert!((ref_.ay[k] - simd.ay[k]).abs() < tol, "ay[{k}]");
            assert!((ref_.az[k] - simd.az[k]).abs() < tol, "az[{k}]");
            assert!((ref_.pot[k] - simd.pot[k]).abs() < 1e-12 * (1.0 + ref_.pot[k].abs() + simd.pot[k].abs()), "pot[{k}]");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_pair_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let n = 40;
        let parts = plummer(n, [0.0; 3], 9);
        let tree = crate::tree::Octree::build(&parts, 16);
        let b = Bodies::from_parts(&parts, &tree.order);
        let mut ref_ = b.clone();
        let mut simd = b.clone();
        let half = n / 2;
        p2p_pair_scalar(&mut ref_, 0, half, half, n - half);
        unsafe { p2p_pair_avx512(&mut simd, 0, half, half, n - half) };
        for k in 0..n {
            let tol = 1e-12 * (1.0 + ref_.ax[k].abs() + simd.ax[k].abs());
            assert!((ref_.ax[k] - simd.ax[k]).abs() < tol, "ax[{k}]");
            assert!((ref_.ay[k] - simd.ay[k]).abs() < tol, "ay[{k}]");
            assert!((ref_.az[k] - simd.az[k]).abs() < tol, "az[{k}]");
            assert!((ref_.pot[k] - simd.pot[k]).abs() < 1e-12 * (1.0 + ref_.pot[k].abs() + simd.pot[k].abs()), "pot[{k}]");
        }
    }

    #[test]
    fn scalar_momentum_exact() {
        let n = 200;
        let parts = plummer(n, [0.0; 3], 3);
        let tree = crate::tree::Octree::build(&parts, 8);
        let mut b = Bodies::from_parts(&parts, &tree.order);
        for c in 0..tree.cells.len() {
            let cell = tree.cells[c];
            if tree.is_leaf(c) {
                p2p_self(&mut b, cell.first, cell.count);
            }
        }
        let mut f = [0.0; 3];
        for k in 0..n {
            let m = b.m[k];
            f[0] += m * b.ax[k];
            f[1] += m * b.ay[k];
            f[2] += m * b.az[k];
        }
        let norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        assert!(norm < 1e-14, "net force {norm}");
    }
}