//! Solid spherical harmonics — the FMM expansion basis.
//!
//! Implements Dehnen (2014), arXiv:1405.2255, Appendix A:
//!   Θ_n^m(r) = Δ_n^m |r|^{-1}                       (eq 40)
//!   Υ_n^m(r) = r^{2n+1} Θ_n^m(r) / ((n-m)!(n+m)!)   (eq 41)
//!
//! Both are evaluated by stable three-term recursions:
//!   r² Θ_n^m = (2n-1) z Θ_{n-1}^m - ([n-1]²-m²) Θ_{n-2}^m        (eq 52)
//!   (n²-m²) Υ_n^m = (2n-1) z Υ_{n-1}^m - r² Υ_{n-2}^m             (eq 53)
//! with the diagonal seeds
//!   Θ_n^n = (2n-1)(x+iy)/r² Θ_{n-1}^{n-1},  Υ_n^n = (x+iy)/(2n) Υ_{n-1}^{n-1}
//! and their m = -n counterparts.                       (eq 54)
//!
//! Negative m follows from Δ_n^{-m} = (-1)^m Δ_n^{m*}:  (eq 39)
//!   Θ_n^{-k} = (-1)^k Θ_n^{k*},  Υ_n^{-k} = (-1)^k Υ_n^{k*}.
//!
//! Index convention: idx(n,m) = n(n+1) + m, -n ≤ m ≤ n, so the block of
//! order n occupies [n², n²+2n] and dim(p) = (p+1)² is contiguous.

/// Complex number, scalar f64 arithmetic. No external crates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    pub const ZERO: C = C { re: 0.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        C { re, im }
    }

    #[inline]
    pub const fn conj(self) -> Self {
        C { re: self.re, im: -self.im }
    }

    #[inline]
    pub fn add(self, o: Self) -> Self {
        C { re: self.re + o.re, im: self.im + o.im }
    }

    #[inline]
    pub fn sub(self, o: Self) -> Self {
        C { re: self.re - o.re, im: self.im - o.im }
    }

    #[inline]
    pub fn mul(self, o: Self) -> Self {
        C {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        C { re: self.re * s, im: self.im * s }
    }
}

/// Number of expansion coefficients for order p: (p+1)².
#[inline]
pub const fn dim(p: usize) -> usize {
    (p + 1) * (p + 1)
}

/// Flat index for (n, m), -n ≤ m ≤ n.
#[inline]
pub const fn idx(n: usize, m: i32) -> usize {
    ((n * (n + 1)) as i64 + m as i64) as usize
}

/// Evaluate Θ_n^m(r), all 0 ≤ n ≤ p, into a table of size dim(p).
///
/// Θ_0^0 = 1/|r|; Θ is singular at r = 0. Caller must ensure |r| > 0.
pub fn theta_table(r: [f64; 3], p: usize) -> Vec<C> {
    let d = dim(p);
    let mut t = vec![C::ZERO; d];
    let (x, y, z) = (r[0], r[1], r[2]);
    let r2 = x * x + y * y + z * z;
    let ir = 1.0 / r2.sqrt();
    t[idx(0, 0)] = C::new(ir, 0.0);

    // m = 0 column.
    for n in 1..=p {
        let a = (2 * n - 1) as f64 * z;
        let b = ((n - 1) * (n - 1)) as f64;
        let p1 = t[idx(n - 1, 0)];
        let p2 = if n >= 2 { t[idx(n - 2, 0)] } else { C::ZERO };
        t[idx(n, 0)] = p1.scale(a).sub(p2.scale(b)).scale(1.0 / r2);
    }

    // m > 0: diagonal seed then upward recursion in n.
    for m in 1..=p {
        let mi = m as i32;
        t[idx(m, mi)] = C::new(x, y).scale((2 * m - 1) as f64 / r2).mul(t[idx(m - 1, m as i32 - 1)]);
        for n in m + 1..=p {
            let a = (2 * n - 1) as f64 * z;
            let b = ((n - 1) * (n - 1) - m * m) as f64;
            t[idx(n, mi)] = t[idx(n - 1, mi)]
                .scale(a)
                .sub(t[idx(n - 2, mi)].scale(b))
                .scale(1.0 / r2);
        }
    }

    // Negative m by Θ_n^{-k} = (-1)^k Θ_n^{k*}.
    for m in 1..=p {
        let sgn = if m % 2 == 0 { 1.0 } else { -1.0 };
        for n in m..=p {
            t[idx(n, -(m as i32))] = t[idx(n, m as i32)].conj().scale(sgn);
        }
    }
    t
}

/// Evaluate Υ_n^m(r), all 0 ≤ n ≤ p, into a table of size dim(p).
///
/// Υ is a homogeneous polynomial of degree n, regular everywhere.
pub fn upsilon_table(r: [f64; 3], p: usize) -> Vec<C> {
    let d = dim(p);
    let mut u = vec![C::ZERO; d];
    let (x, y, z) = (r[0], r[1], r[2]);
    let r2 = x * x + y * y + z * z;
    u[idx(0, 0)] = C::new(1.0, 0.0);

    for m in 0..=p {
        let mi = m as i32;
        if m > 0 {
            u[idx(m, mi)] = C::new(x, y).scale(1.0 / (2 * m) as f64).mul(u[idx(m - 1, m as i32 - 1)]);
        }
        for n in m + 1..=p {
            let a = (2 * n - 1) as f64 * z;
            let b = r2;
            let c = (n * n - m * m) as f64;
            let p2 = if n >= m + 2 { u[idx(n - 2, mi)] } else { C::ZERO };
            u[idx(n, mi)] = u[idx(n - 1, mi)].scale(a).sub(p2.scale(b)).scale(1.0 / c);
        }
    }

    // Negative m by Υ_n^{-k} = (-1)^k Υ_n^{k*}.
    for m in 1..=p {
        let sgn = if m % 2 == 0 { 1.0 } else { -1.0 };
        for n in m..=p {
            u[idx(n, -(m as i32))] = u[idx(n, m as i32)].conj().scale(sgn);
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference: Dehnen (2014) Table 3, which lists the real-valued functions
    // (n-m)!(n+m)! U_n^m = r^{2n+1} T_n^m, with U = Re/Im of Υ and T = Re/Im of Θ.
    fn check_tables(x: f64, y: f64, z: f64, p: usize) {
        let r2 = x * x + y * y + z * z;
        let u = upsilon_table([x, y, z], p);
        let t = theta_table([x, y, z], p);
        let fact = |k: usize| (1..=k).fold(1.0, |a, i| a * i as f64);

        // U_n^m = Re(Υ_n^|m|) for m≥0, Im(Υ_n^|m|) for m<0 (eq 58a).
        let u_val = |n: usize, m: i32| -> f64 {
            let c = u[idx(n, m.unsigned_abs() as i32)];
            if m >= 0 { c.re } else { c.im }
        };
        let t_val = |n: usize, m: i32| -> f64 {
            let c = t[idx(n, m.unsigned_abs() as i32)];
            if m >= 0 { c.re } else { c.im }
        };
        // (n-m)!(n+m)! U must equal the polynomial; likewise r^{2n+1} T.
        let expect = |n: usize, m: i32, poly: f64| {
            let am = m.unsigned_abs() as usize;
            let c = fact(n - am) * fact(n + am);
            let lhs_u = c * u_val(n, m);
            let lhs_t = r2.powf(n as f64 + 0.5) * t_val(n, m);
            assert!((lhs_u - poly).abs() < 1e-12, "U n={} m={}: {lhs_u} vs {poly}", n, m);
            assert!((lhs_t - poly).abs() < 1e-12, "T n={} m={}: {lhs_t} vs {poly}", n, m);
        };

        // Table 3 entries.
        expect(0, 0, 1.0);
        expect(1, 0, z);
        expect(1, 1, x);
        expect(1, -1, y);
        expect(2, 0, 3.0 * z * z - r2);
        expect(2, 1, 3.0 * x * z);
        expect(2, -1, 3.0 * y * z);
        expect(2, 2, 3.0 * (x * x - y * y));
        expect(2, -2, 6.0 * x * y);
        expect(3, 0, 3.0 * z * (5.0 * z * z - 3.0 * r2));
        expect(3, 1, 3.0 * x * (5.0 * z * z - r2));
        // (4,0): 3(35z⁴-30z²r²+3r⁴); (4,1): 15xz(7z²-3r²).
    }

    #[test]
    fn table3_low_orders() {
        check_tables(0.7, -0.4, 1.2, 3);
    }

    #[test]
    fn theta_singular_parity() {
        // Θ_n^m(-r) = (-1)^n Θ_n^m(r).
        let r = [1.3, -2.1, 0.4];
        let p = 4;
        let t = theta_table(r, p);
        let tm = theta_table([-r[0], -r[1], -r[2]], p);
        for n in 0..=p {
            for m in -(n as i32)..=n as i32 {
                let expect = t[idx(n, m)].scale(if n % 2 == 0 { 1.0 } else { -1.0 });
                assert!(
                    (expect.re - tm[idx(n, m)].re).abs() < 1e-12
                        && (expect.im - tm[idx(n, m)].im).abs() < 1e-12,
                    "parity n={} m={}",
                    n,
                    m
                );
            }
        }
    }

    #[test]
    fn upsilon_homogeneous() {
        // Υ_n^m(αr) = α^n Υ_n^m(r).
        let r = [1.3, -2.1, 0.4];
        let a = 1.7;
        let p = 4;
        let u = upsilon_table(r, p);
        let us = upsilon_table([a * r[0], a * r[1], a * r[2]], p);
        for n in 0..=p {
            for m in -(n as i32)..=n as i32 {
                let expect = u[idx(n, m)].scale(a.powf(n as f64));
                assert!(
                    (expect.re - us[idx(n, m)].re).abs() < 1e-12
                        && (expect.im - us[idx(n, m)].im).abs() < 1e-12,
                    "homogeneity n={} m={}",
                    n,
                    m
                );
            }
        }
    }

    #[test]
    fn upsilon_theta_relation() {
        // Υ_n^m = r^{2n+1} Θ_n^m / ((n-m)!(n+m)!)  (eq 41).
        let r = [0.9, -1.7, 0.2];
        let p = 3;
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let u = upsilon_table(r, p);
        let t = theta_table(r, p);
        let fact = |k: usize| (1..=k).fold(1.0, |a, i| a * i as f64);
        for n in 0..=p {
            for m in -(n as i32)..=n as i32 {
                let am = m.unsigned_abs() as usize;
                let c = fact(n - am) * fact(n + am);
                let expect = t[idx(n, m)].scale(r2.powf(n as f64 + 0.5) / c);
                assert!(
                    (expect.re - u[idx(n, m)].re).abs() < 1e-12
                        && (expect.im - u[idx(n, m)].im).abs() < 1e-12,
                    "relation n={} m={}",
                    n,
                    m
                );
            }
        }
    }

    #[test]
    fn index_layout() {
        assert_eq!(dim(0), 1);
        assert_eq!(dim(1), 4);
        assert_eq!(dim(3), 16);
        // block of order n sits in [n², n²+2n].
        assert_eq!(idx(1, -1), 1);
        assert_eq!(idx(1, 0), 2);
        assert_eq!(idx(1, 1), 3);
        assert_eq!(idx(2, -2), 4);
        assert_eq!(idx(2, 0), 6);
        assert_eq!(idx(2, 2), 8);
        assert_eq!(idx(3, -3), 9);
        assert_eq!(idx(3, 3), 15);
    }
}