//! Integration: FMM vs direct O(N²) for small random configurations.
//! The FMM should converge to the direct solution as p grows.
//! Error metric is the scaled error δa/f (eq 16b), not δa/|a| which
//! diverges for particles near the centre of mass.

use fmm::direct::{direct, scaled_error};
use fmm::fmm::Fmm;
use fmm::models::plummer;

#[test]
fn fmm_matches_direct_small() {
    let n = 256;
    let parts = plummer(n, [0.0; 3], 1234);
    let (a_ref, pot_ref) = direct(&parts);

    for (p, eps) in [(2, 1e-2), (3, 1e-3), (4, 1e-4), (5, 1e-5)] {
        let fmm = Fmm::new(&parts, p, eps, 16);
        let res = fmm.run(&parts);
        let (max_s, _, rms_s, rms_rel) = scaled_error(&a_ref, &res.accel, &parts);
        assert!(
            max_s < 10.0 * eps,
            "p={p} eps={eps}: max scaled error {max_s} too large"
        );
        assert!(
            rms_rel < 1e-2,
            "p={p} eps={eps}: rms relative error {rms_rel} too large"
        );
        assert!(
            rms_s < 10.0 * eps,
            "p={p} eps={eps}: rms scaled error {rms_s} too large"
        );
        // Potential: compare relative RMS.
        let mut rel = 0.0;
        for i in 0..n {
            let d = (res.pot[i] - pot_ref[i]).abs();
            let scale = pot_ref[i].abs().max(1e-12);
            rel += (d / scale) * (d / scale);
        }
        rel = (rel / n as f64).sqrt();
        assert!(rel < 1e-3, "p={p} eps={eps}: potential rel err {rel}");
    }
}

#[test]
fn momentum_conserved() {
    // Mutual kernel -> Newton's third law exact; total force must vanish to
    // machine precision.
    let n = 512;
    let parts = plummer(n, [0.0; 3], 99);
    let fmm = Fmm::new(&parts, 4, 1e-4, 16);
    let res = fmm.run(&parts);
    let mut fsum = [0.0; 3];
    let mut ftot = 0.0;
    for i in 0..n {
        let m = parts[i].mass;
        for d in 0..3 {
            fsum[d] += m * res.accel[i][d];
        }
        ftot += m
            * (res.accel[i][0].powi(2) + res.accel[i][1].powi(2) + res.accel[i][2].powi(2)).sqrt();
    }
    let norm = (fsum[0].powi(2) + fsum[1].powi(2) + fsum[2].powi(2)).sqrt();
    assert!(
        norm / ftot < 1e-14,
        "net force {norm} (rel {})",
        norm / ftot
    );
}
