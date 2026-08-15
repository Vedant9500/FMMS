//! Brute-force O(N²) direct summation, used as reference for validation.
//! Mutual kernel: exact Newton's third law; potential positive convention.

use crate::models::Particle;

pub fn direct(parts: &[Particle]) -> (Vec<[f64; 3]>, Vec<f64>) {
    let n = parts.len();
    let mut a = vec![[0.0; 3]; n];
    let mut pot = vec![0.0; n];
    for i in 0..n {
        for j in i + 1..n {
            let pa = &parts[i];
            let pb = &parts[j];
            let dx = pb.pos[0] - pa.pos[0];
            let dy = pb.pos[1] - pa.pos[1];
            let dz = pb.pos[2] - pa.pos[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv = 1.0 / r2.sqrt();
            let inv3 = inv * inv * inv;
            a[i][0] += pb.mass * inv3 * dx;
            a[i][1] += pb.mass * inv3 * dy;
            a[i][2] += pb.mass * inv3 * dz;
            a[j][0] -= pa.mass * inv3 * dx;
            a[j][1] -= pa.mass * inv3 * dy;
            a[j][2] -= pa.mass * inv3 * dz;
            pot[i] += pb.mass * inv;
            pot[j] += pa.mass * inv;
        }
    }
    (a, pot)
}

/// Scaled acceleration error δa/f and relative error δa/a statistics.
/// `f` is the per-particle estimate Σ m/|Δx|² (eq 4), computed exactly here.
pub fn scaled_error(a_ref: &[[f64; 3]], a_fmm: &[[f64; 3]], parts: &[Particle]) -> (f64, f64, f64, f64) {
    let n = parts.len();
    // Exact f.
    let mut f = vec![0.0; n];
    for i in 0..n {
        for j in i + 1..n {
            let pa = &parts[i];
            let pb = &parts[j];
            let dx = pb.pos[0] - pa.pos[0];
            let dy = pb.pos[1] - pa.pos[1];
            let dz = pb.pos[2] - pa.pos[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let inv2 = 1.0 / r2;
            f[i] += pb.mass * inv2;
            f[j] += pa.mass * inv2;
        }
    }
    let mut max_scaled = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut sum_scaled2 = 0.0f64;
    let mut sum_rel2 = 0.0f64;
    for i in 0..n {
        let dx = a_fmm[i][0] - a_ref[i][0];
        let dy = a_fmm[i][1] - a_ref[i][1];
        let dz = a_fmm[i][2] - a_ref[i][2];
        let da = (dx * dx + dy * dy + dz * dz).sqrt();
        let aref = (a_ref[i][0] * a_ref[i][0]
            + a_ref[i][1] * a_ref[i][1]
            + a_ref[i][2] * a_ref[i][2])
            .sqrt();
        let s = da / f[i];
        let rel = if aref > 0.0 { da / aref } else { 0.0 };
        max_scaled = max_scaled.max(s);
        max_rel = max_rel.max(rel);
        sum_scaled2 += s * s;
        sum_rel2 += rel * rel;
    }
    let rms_scaled = (sum_scaled2 / n as f64).sqrt();
    let rms_rel = (sum_rel2 / n as f64).sqrt();
    (max_scaled, max_rel, rms_scaled, rms_rel)
}