# LIMITATIONS.md — known limitations & deferred improvements

Living list of things we know are suboptimal, incomplete, or machine-specific.
Companion to `todo.md` (session log) and `ROADMAP.md` (gates). Items here are
intentionally NOT silently "fixed" — each needs a re-measure / re-verify before
changing, per the conventions in `AGENTS.md`.

## Physics / integrator

- **Cold-start Plummer collapse is not collisionless** (energy gate):
  a hard binary forms in the core at t≈0.03 for any N (observed N=4096…10⁵),
  and a fixed-dt leapfrog cannot resolve it → energy jumps then runaways
  (ΔE/E → +10³). Verified NOT an FMM bug: direct O(N²) force integration
  reproduces the identical jump to 1e-8. The energy-conservation test must
  use a virialized IC.
- **Unresolved close encounters limit fixed-dt energy conservation**:
  even virialized, N=10⁵ × 10³ steps @ dt=1e-3 accumulates drift 1.03e-3
  (2 discrete heating events, vmax 1.7→2.1), 3% over the 1e-3 gate. No
  secular FMM heating (slope 6.5e-7/step < 1e-6). At N=10⁶ per-event jumps
  scale down 10× (m ∝ 1/N), so the full gate should hold at 1e-3 — not yet
  verified.
- **No softening**: all kernels are pure 1/r. Softened (Plummer/Yukawa) kernels
  are the bbFMM optional path (ROADMAP Phase 6) and would kill the
  unresolved-encounter contamination above.
- **Fixed timestep only**: no block-timestep / adaptive dt (Phase 2).
  Collisional regimes (core collapse, binaries) need variable dt + likely
  KS regularization (ROADMAP risk #4).
- **No velocity sampler in `models.rs`**: the virialized Plummer IC (local
  Maxwellian, exact-virial rescale) lives inside the energy-gate test, not the
  library. Phase-2 integrator work will want it promoted.

## FMM / performance

- **P2P 300+ GFLOPS fp64 not reachable on this hardware**: fp64 ceiling
  ~23 GFLOPS/core (~184 threaded on 8 cores). Best measured 20.2 single /
  153.6 threaded synthetic, 2.6/17.3 real-tree (scalar-tail bound). Closing
  the gate needs fp32 inner kernels (~1.8×) which break the 1e-12
  AVX-vs-scalar contract, or faster hardware.
- **exafmm parity (0.95 s / 1M) not reached**: ~15.7 s total @ 8 threads.
  Needs faster M2L and/or memory headroom. 15/16 threads OOM (per-thread
  `bodies.clone()` 64 MB + C accumulator 52 MB ≈ 116 MB/thread; machine at
  9/14 GB).
- **Walk is latency/bandwidth-bound**: 8→12 threads gives only ~10% (12-thread
  walk 7.8–8.5 s vs 8.5–10.5 s @ 1M). STREAM 8-thread ~18.5 GB/s, walk uses
  ~6 GB/s.
- **Tree is rebuilt every integrator step**: no incremental update or reuse
  of the previous step's tree. Fine for a gate, wasteful for production
  integration.
- **Tree build is single-threaded** (Morton sort + recursive build + geometry
  pass); at 1M it is ~1 s of the ~17 s total.

## Verification gaps

- **Full energy gate N=10⁶ × 10³ deferred**: machinery + warm-start IC
  validated (N=3×10⁴ × 200: drift 4.7e-5; N=10⁵ × 10³: drift 1.03e-3,
  encounter-dominated), but the exact ROADMAP line-48 number at 10⁶ bodies is
  unmeasured. Run after the walk is faster:
  `FMM_ENERGY_N=1000000 FMM_ENERGY_STEPS=1000 cargo test --release -p fmm
  --test energy_gate -- --ignored --nocapture`
- **Phase-1 gate lines 36/40/43-46 (rotation M2L, P2P, O(N) scaling, exafmm
  parity)**: some gates were closed "as-is" at the fp64 hardware ceiling
  (P2P) or with caveats — revisit if hardware / precision targets change.

## Housekeeping / tech debt

- `walk.rs` temporary `AtomicU64` profiling counters + `walk_counts()` — used
  by bench profile mode; remove after Phase-1 gates close.
- `.cargo/config.toml` has machine-specific `target-cpu=native` (Zen4 FMA);
  git-ignored, performance numbers assume it, portable/CI builds omit it.
- `bodies.clone()` per walk thread copies read-only x/y/z/m (32 MB @ 1M) —
  shareable to fit 16 threads.
- Clippy `erasing_op` deny in `rotation.rs:147` (`0 * nb_prev`) + pre-existing
  doc lints — rotation.rs is off-limits per AGENTS.md; needs explicit
  go-ahead.
- P2P kernel variant sweep lives in `/tmp/opencode/p2p_bench.rs` (not in
  repo) — keep for future kernel work.
- `ROT_P = 26` and MAC factor-8 are measured constants (re-measure before
  tuning); machine is an 8-core 35 W laptop with limited headroom.
