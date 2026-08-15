//! Roofline-style kernel classification and FLOP accounting for Phase 0.
//!
//! Each kernel is labelled with its arithmetic intensity and the per-particle
//! FLOPs at expansion order p, following docs/02-lowlevel.md. Used by the
//! bench harness to compare achieved GFLOP/s against the roofline.

/// FLOPs per M2L interaction at order p (one direction).
/// Naive quadruple loop: (p+1)² targets × (p+1)² sources ≈ (p+1)⁴ complex
/// multiply-adds; one complex MAC ≈ 6 FLOPs (2×2 real mul + add each).
pub fn m2l_flops(p: usize) -> f64 {
    let d = ((p + 1) * (p + 1)) as f64;
    d * d * 6.0
}

/// FLOPs per P2P particle pair (mutual kernel): 1 rsqrt + 6 mul + 6 add for
/// force + 1 mul + 1 add for potential.
pub fn p2p_flops_per_pair() -> f64 {
    16.0
}

/// Total FLOPs for a treecode monopole pre-pass of N particles.
pub fn prepass_flops(n: usize, tree_visits: usize) -> f64 {
    // Each accepted cell: 1 r2 + 3 mul + 2 add + 1 div = ~8 FLOPs.
    8.0 * tree_visits as f64 + n as f64
}

/// Arithmetic intensity (FLOP/byte) for a kernel given bytes moved per item.
pub fn intensity(flops: f64, bytes: f64) -> f64 {
    flops / bytes.max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m2l_flops_grows_quartic() {
        assert!((m2l_flops(2) / m2l_flops(1) - 5.0625).abs() < 1e-9);
    }
}