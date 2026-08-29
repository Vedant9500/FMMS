//! Particle model and analytic density samplers (Plummer, Hernquist).
//!
//! Natural units: G = M = a = 1. Equal-mass particles unless overridden.
//! Deterministic SplitMix64 PRNG for reproducible tests/benches.

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub mass: f64,
    pub pos: [f64; 3],
}

/// Deterministic 64-bit PRNG (SplitMix64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sample n particles from a Plummer sphere (M=G=a=1) at the given center,
/// with net-zero velocity (cold start). Equal masses.
pub fn plummer(n: usize, center: [f64; 3], seed: u64) -> Vec<Particle> {
    let mut rng = Rng::new(seed);
    let mass = 1.0 / n as f64;
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        // Inverse CDF: r = a (u^{2/3} / (1 - u^{2/3}))^{1/2}
        let u = rng.unit();
        let w = u.powf(2.0 / 3.0);
        let r = (w / (1.0 - w)).sqrt();
        // Random direction on unit sphere.
        let z = 2.0 * rng.unit() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * rng.unit();
        let s = (1.0 - z * z).sqrt();
        parts.push(Particle {
            mass,
            pos: [
                center[0] + r * s * phi.cos(),
                center[1] + r * s * phi.sin(),
                center[2] + r * z,
            ],
        });
    }
    parts
}

/// Sample n particles from a Hernquist sphere (M=G=a=1) at the given center.
pub fn hernquist(n: usize, center: [f64; 3], seed: u64) -> Vec<Particle> {
    let mut rng = Rng::new(seed);
    let mass = 1.0 / n as f64;
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        // Inverse CDF: r = a sqrt(u) / (1 - sqrt(u))
        let s = rng.unit().sqrt();
        let r = s / (1.0 - s);
        let z = 2.0 * rng.unit() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * rng.unit();
        let sp = (1.0 - z * z).sqrt();
        parts.push(Particle {
            mass,
            pos: [
                center[0] + r * sp * phi.cos(),
                center[1] + r * sp * phi.sin(),
                center[2] + r * z,
            ],
        });
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plummer_shell_mass() {
        // Integrated mass within r should match the analytic profile
        // M(<r)/M = u^{3/2} with u = r²/(r²+a²), a=1.
        let n = 100_000;
        let parts = plummer(n, [0.0; 3], 42);
        let mut within = 0.0;
        let mut total = 0.0;
        for p in &parts {
            let r2 = p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2];
            total += p.mass;
            if r2 <= 1.0 {
                within += p.mass;
            }
        }
        let frac = within / total;
        // Analytic: r=1, a=1 -> u=1/2 -> u^{3/2}=0.3536
        assert!((frac - 0.3536).abs() < 0.02, "frac={frac}");
    }

    #[test]
    fn hernquist_shell_mass() {
        let n = 100_000;
        let parts = hernquist(n, [0.0; 3], 7);
        let mut within = 0.0;
        let mut total = 0.0;
        for p in &parts {
            let r2 = p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2];
            total += p.mass;
            if r2 <= 1.0 {
                within += p.mass;
            }
        }
        let frac = within / total;
        // Analytic: r=1, a=1 -> s=sqrt(r)/... M(<r)/M = (r/(r+a))² = 1/4
        assert!((frac - 0.25).abs() < 0.02, "frac={frac}");
    }
}
