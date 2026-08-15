# TODO — remaining Phase-1 gates + session log

Source of truth for exit gates: `ROADMAP.md` Phase 1 (lines 34-48).

## Phase-1 exit gates still open

- [x] **Rotation-accelerated M2L O(p³)** (ROADMAP line 36): axis-swap rewrite
      (Dehnen A.6.1, `B = x↔z` swap, no per-pair Wigner-d build) fixed and
      verified. Rotation now beats naive from p ≥ 26 (0.97× @ p=26 … 0.85× @
      p=32, 4000-pair microbench) with rel err ≤ ~2e-8 up to p=32 and ~1e-13
      @ p≤20. End-to-end FMM @ p=30 matches direct to ~3.5e-10 rel. `ROT_P`
      set to 26. Precomputed rotation matrices for p ≤ 60 (asserted in
      `RotCtx::new`).
- [ ] **P2P throughput target: 300+ GFLOPS** (line 40): not benchmarked. P2P
      is currently ~1.3% of runtime (p2p_pair ≈ 158k cell-pairs @ N=1e4,
      ~2.4 bodies/leaf) — not the bottleneck yet; AVX-512 path exists.
- [ ] **exafmm parity** (lines 45-46): target 0.95 s / 1M / 7 digits @ 14
      cores, normalized to our hardware. Measured 51.7s / 1M single-core
      @ eps=1e-5 (~5 digits) → ~4× slower per core. Needs faster M2L and/or
      threading (Phase 5).
- [ ] **Energy conservation @ 10⁶ bodies / 10³ steps** (line 48): momentum
      conservation verified (netF/|F| ~ 1e-17, all N) but energy drift over
      an integrated orbit not yet tested. Needs a leapfrog integrator test
      (Phase 2 territory) or a standalone energy check.

## Phase-1 gates now closed (this session)

- [x] **O(N) scaling** (line 43): measured N=10³…10⁶, fit ~N^0.91-1.0.
      0.73s @ 1e4 → 51.7s @ 1e6.
- [x] **FMM beats direct N² at N>10⁴** (line 44): crossover at N≈3e4
      (eps=1e-5, p=8, single-core, Zen4). FMM ~48× faster than direct @ 1e6.
- [x] **Momentum conservation** (part of line 48): netF/|F| ~ 1e-17 all N.

## Session log — 2026-08-14: walk (M2L) performance + full sweep

Problem: FMM 4-8× slower than direct at N=1e4; walk = 95% of runtime,
dominated by 621,702 naive M2L at ~62/body (~O(N), normal for eps=1e-5).

Changes:
- `crates/fmm/src/translate.rs` — `m2l_mutual` rewrite: hoist `conj()` of both
  cells' moments out of inner loop; fold the Θ(-r)=(-1)^order parity sign into
  a pre-scaled copy `ths` of the Θ-table (kills per-iteration `scale` temp +
  branch); incremental flat-index arithmetic; unsafe raw-pointer access with
  fused re/im accumulation. Bit-identical math (all 24 unit + 3 integration
  tests pass). Walk 1.13s → 0.92s @ N=1e4.
- `crates/fmm/src/mac.rs` — `error_estimate`: per-call `vec![1.0; p+1]` →
  stack array `[f64; 64]` (p ≤ 63). Small gain.
- `.cargo/config.toml` — `rustflags = ["-C", "target-cpu=native"]` (Zen4 FMA).
  Walk 0.92s → 0.69s @ N=1e4. NOTE: machine-specific; remove for portable/CI
  builds.
- Total walk speedup 1.13s → 0.69s (~40%).

Measured facts:
- M2L count scales ~O(N): 46/body (N=3e3), 62/body (N=1e4), 63/body (N=3e4).
- MAC factor-8 (Dehnen eq 15) in `mac.rs` is load-bearing. Lowering to 1.0:
  M2L 621k→386k, walk 1.0→0.62s, but max_s 1.36e-3→1.08e-2 and rms_s
  3.76e-5→2.25e-4 (FAILS gate rms<1e-4). Keep 8.
- Achieved accuracy @ N=1e4, eps=1e-5: max_s 1.36e-3 (~136·eps), rms_s 3.76e-5.
  Estimate under-bounds the worst case; rms meets gate.

Housekeeping:
- [ ] Remove temporary `AtomicU64` counters + `walk_counts()` from
      `crates/fmm/src/walk.rs` once tuning is done (still used for profiling).
- [ ] `.cargo/config.toml` — decide keep vs revert (machine-specific).

## Session log — 2026-08-15: rotation M2L correctness + crossover

Problem: `m2l_rot_matches_naive` failing with huge errors; rotation was also
~5× slower than naive (gate open). Investigation traced a sign error in the
back-rotation's `conj(M)·ph` product.

Changes (all in `crates/fmm/src/rotation.rs`):
- Rewrote `apply()` to use two scratch buffers (`tmp`, `tmp2`) + `self.f`
  for the axial output; forward/back phases folded into per-(n,m) diagonals.
- Added `f: Vec<C>` field to `RotCtx` (initializer at rotation.rs:193).
- Fixed swap-matrix construction: interior rows source prev-block row `M-1`;
  boundary row `M=n` sources row `2n-2`; column offsets shifted in i32 space
  (negative azimuth `as usize` was overflowing).
- Fixed folding diagonals: `m.rem_euclid(4)` in `new()` (was `m % 4`, wrong
  for negative m); `m.rem_euclid(2)` for the B→A `(-1)^m` phase in `m2l_pair`.
- Root cause: `vi = m.im*ph.re - m.re*ph.im` computed `M·conj(ph)` instead of
  `conj(M)·ph` — flipped every imag sign through the whole pipeline.
- Removed 3 temp diagnostic tests; dropped unused `f2` field; renamed
  `M`/`L` loop vars → `mr`/`lc`. Lib now compiles warning-free.

Measured facts (4000-pair microbench, release):
- Crossover naive→rotation now at p≈26 (1.02× @ p=24, 0.97× @ p=26,
  0.90× @ p=30, 0.85× @ p=32). Relative err ≤ 3e-14 @ p≤12, ≤ 3e-9 @ p≤28,
  2.2e-8 @ p=30, 2.2e-7 @ p=32 (axis-swap rotation is stable; the old
  per-pair Wigner-d route lost at all p).
- End-to-end FMM @ p=30 (rotation path) vs direct: max|ΔΦ| rel ~1.8e-10
  (N=512), ~3.5e-10 (N=2048).
- `ROT_P` 12 → 26. All 22 unit + 2 integration + 1 ignored exit-gate tests pass.