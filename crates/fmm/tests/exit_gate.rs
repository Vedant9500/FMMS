//! Exit-gate test (ROADMAP Phase 0): force-compare at ε=10⁻⁵ for N=10⁴.
//!
//! Accuracy metric follows Dehnen 2014 Fig. 10: rms scaled error δa/f.
//! Max errors are expected to be ~100× larger (tails, Fig. 8); the paper
//! gates on rms / 99.99-percentile, not max.

use fmm::direct::{direct, scaled_error};
use fmm::fmm::Fmm;
use fmm::models::{hernquist, plummer};

#[test]
#[ignore = "N=10^4 direct O(N^2); run explicitly: cargo test --release -p fmm --test exit_gate -- --ignored"]
fn exit_gate_force_compare_n1e4() {
    let n = 10_000;
    let parts = plummer(n, [0.0; 3], 2024);
    let fmm = Fmm::new(&parts, 5, 1e-5, 32);
    let res = fmm.run(&parts);

    let (a_ref, _) = direct(&parts);
    let (max_s, _, rms_s, _) = scaled_error(&a_ref, &res.accel, &parts);
    println!("max_scaled={max_s:e} rms_scaled={rms_s:e}");
    assert!(rms_s < 1e-4, "rms scaled error {rms_s} exceeds gate");
    assert!(max_s < 1e-2, "max scaled error {max_s} exceeds gate");
}

#[test]
fn exit_gate_hernquist_smoke() {
    // Lighter: hernquist N=2^12, sanity only.
    let n = 4096;
    let parts = hernquist(n, [0.0; 3], 7);
    let fmm = Fmm::new(&parts, 4, 1e-3, 32);
    let res = fmm.run(&parts);
    let (a_ref, _) = direct(&parts);
    let (_, _, rms_s, _) = scaled_error(&a_ref, &res.accel, &parts);
    assert!(rms_s < 1e-2, "rms scaled error {rms_s}");
}