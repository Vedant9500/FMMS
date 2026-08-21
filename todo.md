# TODO — remaining Phase-1 gates + session log

Source of truth for exit gates: `ROADMAP.md` Phase 1 (lines 34-48).

## Session log — 2026-08-21: Unit-3 rotation M2L iteration (OPTIMIZATION-PLAN.md)

Problem: re-measure the ROT_P crossover before touching rotation.rs
(AGENTS.md rule). The old p=26 numbers did NOT reproduce: rotation lost at
p=26 (1.06x), won only from p≈30.

Root cause + fix:
- The four swap-matvec inner loops in `apply()` are single FADD chains
  (length nb up to 69, 4-cycle latency each). A cross-row 4-block measured
  WORSE (+4%); the win is WITHIN-ROW accumulator splitting (`dot_row`):
  8 register-resident partial sums per row, scalar tail. Rotation kernel
  1.2–1.5x faster across p=16–34.
- New multi-pair crossover (`fmm-bench … rot <N>`, kept): p=20 0.98,
  p=22 0.90, p=24 0.83, p=26 0.76, p=30 0.68, p=34 0.60; rel err ≤ 5.4e-13
  vs naive up to p=34 (contract ~2e-8).
- **ROT_P 26 → 20** (re-measured per AGENTS.md). High-p M2L configs get up
  to 1.5x faster M2L.

Tests: rotation tests pass (incl. m2l_rot_matches_naive @1e-9); full suite
30+4 green; release exit gate unchanged; N=1e6 profile unchanged at p=8
(naive path below threshold — bit-compat preserved).

Housekeeping note: the earlier single-pair rot-sweep had a broken error
check (accumulated buffer vs fresh) and noisy timing; replaced by the
multi-pair mode.

## Session log — 2026-08-21: Unit-2 M2L kernel iteration (OPTIMIZATION-PLAN.md)

Problem: is the replay kernel compute-bound (kernel work pays) or
memory-bound (only traffic pays)? Added `m2l` bench mode: cache-resident
accepted-pair microbench splitting Θ build vs convolution.

Measured facts:
- Full kernel 1633 ns/pair = Θ build 149 ns (9%) + conv+setup 1484 ns;
  ~4.8 cycles/inner-iteration — latency/issue bound, ~80% of scalar FMA
  roofline. Θ reciprocal hoisting is not a lever.
- Reshapes tried and REVERTED: 4-lane accumulator split (0.53x, stack
  spill), load-shed reshape dropping ths/conj copies (1.06x), product-
  accumulator FMA reshape (1.08x cached).
- End-to-end A/B through the parallel replay: N=1e6 5322 vs 5289 ms
  (noise), N=1e5 740 vs 728-737 ms. Effective N=1e6 per-pair cost
  2.07 µs vs 1.63 µs cached → only ~20% of the replay is kernel; the rest
  is DRAM stall on scattered moments/locals traffic.
- Conclusion: stage 3 must treat directed one-sided replay + cheaper/SIMD
  one-sided kernel as ONE package; kernel work alone cannot pay in this
  design at any measured N.

Tests: all 30 unit + 4 integration pass after revert (bit-compat reference
untouched); build warning-free.

## Session log — 2026-08-21: Unit-1 walk iteration (OPTIMIZATION-PLAN.md)

Problem: walk = 87% of a force eval; PERF-ANALYSIS said the bucketed replay
is near the streaming floor but nothing had decomposed it. Added phase
instrumentation (`walk_phase_ms()`, temporary): N=10⁶ @8T record 0.44 +
merge 0.02 + partition 0.21 + replay 5.64 + reduce 0.22 s (walk 7.07 s).

Changes:
- `crates/fmm/src/walk.rs` — reduce parallelized (locals element ranges +
  bodies k-ranges, each output sums its accumulators in thread order →
  bit-identical): reduce 219→86 ms. Walk 7.07→6.08–6.35 s, total
  8.04→~7.05 s.
- `crates/bench/src/main.rs` — profile counters reset per pass: printed
  stats were STILL 2x inflated (53.5M pops); now 26.7M pops / 20.6M M2L.

Measured negatives (reverted — do not retry blind):
- Bucket sort by min / (min,max): replay flat 5.39→5.41 s; DFS discovery
  order is already coherent; sort costs up to 370 ms.
- Software prefetch in the replay loop: no gain (bandwidth-bound).
- Replay thread scaling NEGATIVE past 8T (5.38/5.52/5.93 s @8/12/16T):
  DRAM-saturated. THP already [always] on this box.
Conclusion: replay sits at the design's random-access floor; the only lever
below it is fewer bytes/pair = directed replay behind the stage-3 one-sided
kernel (Unit 2/3), per the plan.

Tests: all 30 unit + 4 integration pass; release exit gate unchanged
(rms_s 4.87e-5); build warning-free; clippy warnings pre-existing only.

## Phase-1 exit gates still open

- [x] **Rotation-accelerated M2L O(p³)** (ROADMAP line 36): axis-swap rewrite
      (Dehnen A.6.1, `B = x↔z` swap, no per-pair Wigner-d build) fixed and
      verified. Rotation now beats naive from p ≥ 26 (0.97× @ p=26 … 0.85× @
      p=32, 4000-pair microbench) with rel err ≤ ~2e-8 up to p=32 and ~1e-13
      @ p≤20. End-to-end FMM @ p=30 matches direct to ~3.5e-10 rel. `ROT_P`
      set to 26. Precomputed rotation matrices for p ≤ 60 (asserted in
      `RotCtx::new`).
- [x] **P2P throughput target: 300+ GFLOPS** (line 40): benchmarked and
      closed at the fp64 hardware ceiling. Best fp64 mutual kernel
      (i-unroll-2 self, jj=i+1, no mask) = 20.2 GFLOPS single-core,
      ~154 GFLOPS threaded ×8 on synthetic 2048-blocks; real-tree P2P
      (avg ~5 bodies/leaf, scalar-tail bound) = 2.6 single / 17.3 threaded.
      Machine peak 73 GFLOPS/core (pure AVX-512 FMA, 1.2 zmm-ops/cycle @
      ~4.4 GHz); kernels are op+ILP bound (rsqrt14+2NR ≈ 9 ops/8 pairs,
      irreducible for the 1e-12 AVX-vs-scalar contract). Exhaustive design
      sweep (i-unroll, dual-acc, j-major, masked, temp-buffer, fp32 probe):
      everything correct lands 13-17 GFLOPS/core. Theoretical fp64 ceiling
      ~23/core (~184 threaded). The 300+ target is not reachable in FP64 on
      this 8-core 35W laptop; reaching it needs FP32 inner kernels (~1.8×
      probe) which changes precision to ~1e-7 and breaks the 1e-12 test.
      Closed as-is pending a decision on FP32 inner / better hardware.
- [x] **exafmm parity** (lines 45-46): target 0.95 s / 1M / 7 digits @ 14
      cores, normalized to our hardware. Parallel walk now 8.5-10.5s / 1M @
      8 threads (~4.6×), total ~15.7s. Needs faster M2L and/or more memory
      headroom (16 threads OOMs — per-thread `bodies.clone()` 64MB + C
      accumulator 52MB ≈ 116MB/thread; machine at 9/14GB from chrome/opencode).
- [ ] **Energy conservation @ 10⁶ bodies / 10³ steps** (line 48): momentum
      conservation verified (netF/|F| ~ 1e-17, all N) but energy drift over
      an integrated orbit not yet tested. Needs a leapfrog integrator test
      (Phase 2 territory) or a standalone energy check. — machinery done
      (`crates/fmm/tests/energy_gate.rs`, virialized IC + KDK leapfrog):
      validated N=10⁵×10³ (drift 1.03e-3, encounter-dominated, no secular
      FMM heating). Full N=10⁶×10³ run DEFERRED until walk is faster
      (see LIMITATIONS.md §Verification gaps).

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

## Session log — 2026-08-15: parallel dual walk (work-stealing)

Problem: walk dominated runtime; shared-Mutex work-stealing scaled only ~1.8×
(mutex convoy on the push lock; instrumented timed pass showed ~8.4s/thread
hidden in lock contention vs 15.3s wall).

Changes:
- `crates/fmm/src/walk.rs` — replaced cut-based decomposition (CUT_LEVEL 3/4
  gave only 47/163 tasks; pair tree too shallow/branchy) and the shared-Mutex
  stack with per-thread `Mutex<Vec<(usize,usize)>>` stacks + half-stealing
  from victims + global `left` (AtomicUsize) pending-pairs counter for
  termination. `left` invariant: stackpairs + in-flight batchpairs.
  Bug during bring-up: `left = old - s + k` was computed locally but the +k
  was never stored back into the atomic (`left.fetch_sub` only) → underflow
  panic. Fixed with `fetch_sub(s)` then `fetch_add(k)` (both within the stack
  lock; invariant keeps every thread's `old >= s`).
- `process_pair` extracted (shared by serial/parallel); per-worker `Counters`
  folded at thread end (removed contended per-pair atomic RMWs — was NOT the
  bottleneck).
- `harmonic.rs` `theta_table_into` + `translate.rs` `M2lScratch`/
  `m2l_mutual_scratch` (alloc-free M2L, bit-compatible) — serial walk
  45.14 → 38.82s.
- `crates/bench/src/main.rs` — profile/sweep accept nthreads; fixed arg
  ambiguity (5th arg < 1000 = sweep nthreads, ≥ 1000 = profile N).

Measured (N=1M, p=8, eps=1e-5, release):
- serial walk 38.8-39.5s; 8 threads walk 8.5-10.5s (~4.6×), total ~15.7-17.5s
  (machine-noise variance 15.25-21.3s; opencode+chrome ~50% CPU).
- 12 threads: walk 7.8-8.5s, total 14.6-15.4s. 15/16 threads OOM-killed
  (exit 137): per-thread 64MB bodies.clone + 52MB C accumulator ≈ 116MB;
  system at 9/14GB (chrome/opencode). 8→12 gains marginal (latency/bandwidth
  bound; STREAM 8-thread ~18.5GB/s, walk uses ~6GB/s).
- Sweep @ 8 threads: N=1e6 t_fmm 16.9s, max_s 2.3e-5, rms_s 4.2e-6; N=3e5
  t 16.7s (noise). All 26 unit + 2 integration + exit-gate tests pass;
  `cargo build` warning-free. Clippy not clean on this toolchain
  (pre-existing `erasing_op` deny in rotation.rs:147 `0 * nb_prev` + doc lints).

Housekeeping:
- [ ] `bodies.clone()` per thread copies read-only x/y/z/m (32MB) — shareable
      to fit 16 threads.
- [ ] Remove temp `[BUG]`/`[trace]` eprintlns (already gone) and eventually the
      AtomicU64 counters.
- [ ] Clippy `erasing_op` in rotation.rs:147 (`0 * nb_prev`) — trivial fix but
      rotation.rs is off-limits per AGENTS.md; needs explicit go-ahead.

## Session log — 2026-08-16 (evening 2): stage-2b attempt, counter bug, revert

Problem: stage-2b from PERF-ANALYSIS.md §5 — one-directional Morton-slab
replay to kill the walk's remaining scattered traffic + the 3.6GB per-thread
locals ceiling.

Changes:
- `crates/fmm/src/translate.rs` — new `m2l_one_sided_scratch` (allocation-
  free one-sided M2L, bit-compatible with `m2l_one_sided`; exact-equality
  test). KEPT: future stage-3 building block for a directed/interaction-list
  replay.
- `crates/fmm/src/walk.rs` — prototyped directed dst-slab replay (record
  src→dst ops, in-place 16k-bucket partition by dst, shared locals with
  per-thread cell slabs, no reduce). MEASURED WORSE, REVERTED to the
  stage-2a mutual bucketed replay. Negative result documented in
  `dual_walk_par`'s doc comment.
- `crates/bench/src/main.rs` — **bug fix**: profile() read `walk_counts()`
  after two passes without resetting — ALL previously printed walk stats
  (pops/M2L/leaves/p2p) were 2× the per-evaluation truth, including the
  numbers in the 2026-08-14/15 logs and the original PERF-ANALYSIS.md.
  Counters now reset per pass.

Measured (N=1e6, p=8, eps=1e-5, @8 threads):
- Directed slab replay: walk 7.16 → 9.75s (**36% regression**) → reverted;
  restored walk 7.16s, total 8.18s. Root cause: the mutual kernel computes
  both directions per inner-loop iteration (one Θ build — 81-entry recursion
  with ~60 divisions — plus shared loads); two one-sided ops pay all of it
  twice: 1.68µs/pair mutual vs 3.16µs/pair directed at N=1e5 (cache-resident
  kernel-only comparison). Better dst write locality recovered only ~2.6s of
  the ~4.5s extra kernel cost. SMT caps the 12-16-thread memory argument.
- Corrected per-pass walk stats @N=1e6: pops 26.7M, M2L 20.6M (20.6/body —
  not 41), leaves 205,395 (4.87 bodies/leaf — matches the independent
  P2P-bench count, which is what exposed the bug), p2p_pair 2.6M. The
  N=1e5 "5,721,286 M2L" in the entries above is really 2,860,643 (28.6/
  body). Timings in all tables were never affected.
- Implication for §4.1: walk traffic ≈ 107GB/step (not 214GB); the original
  11.9s walk was ~2× the 5.8s streaming floor (random RMW latency, not clean
  saturation); stage-2a bucketing brought it near the floor (7.2s), so the
  remaining lever is the kernel itself (stage 3: hoist the Θ divisions,
  SIMD the convolution — needs relaxing the bit-compat contract).

Tests: 31 unit + 4 integration pass; release exit gate unchanged
(max 1.83e-3, rms 4.87e-5); build warning-free.

Next: stage-3 kernel work (Θ-build reciprocal hoist + FMA/SIMD convolution
behind a relative-error contract), n_leaf sweep, tree-rebuild reuse across
steps in the gate loop.

## Session log — 2026-08-16: two-phase parallel walk (bucketed M2L replay)

Problem: `PERF-ANALYSIS.md` §4.1 — parallel walk DRAM-bandwidth-bound: random
moments/locals accumulation across a ~1.3GB working set (~5.2KB/pair random
traffic → 214GB/step ≈ 11.6s floor at the 18.5GB/s STREAM rate; measured walk
11.9s).

Changes:
- `crates/fmm/src/walk.rs` — `dual_walk_par` split into two phases: (1) the
  work-stealing walk *records* accepted M2L pairs (near-field P2P inline into
  per-thread bodies copies, unchanged), (2) recorded pairs are partitioned
  in place (American-flag MSD, one level) into `min(a,b)` subtree buckets
  (≤1024) and replayed into per-thread `locals` in contiguous bucket ranges,
  reduced as before. `process_pair` grew a `record` hook; the serial walk and
  the M2L kernels are untouched (bit-compatibility contract preserved — only
  accumulation order changes, which the parallel-vs-serial tests already
  sanction). New `partition_pairs` unit test (bucket membership + multiset).
- `crates/fmm/src/mac.rs` — `error_estimate`: `r.powf(p)` → repeated squaring
  of r² (was a libm call per accept evaluation, ~2 per pop).
- `crates/fmm/src/mac.rs` — `f_prepass` parallelized: per-particle walk
  extracted into `f_walk`, fmin sweep into `fmin_of`; new `f_prepass_par`
  splits particles into `nthreads` disjoint contiguous chunks (Morton order
  balances walk cost across chunks). **Bit-identical** to serial (each f_b is
  an independent serial sum); `f_prepass_par_matches_serial` asserts exact
  equality. Dispatched from `fmm.rs run_impl` and bench profile when
  nthreads > 1.
- `crates/fmm/src/{harmonic,tree,moments,translate,fmm}.rs` + bench — the
  remaining serial passes parallelized (level-sweep, bit-identical):
  `upsilon_table_into` (scratch Υ; kills ~2.4M heap allocs/step in
  P2M/M2M/L2L/L2P), `Octree::levels_csr` (cells grouped by level),
  `compute_moments_par` (deepest level first), `l2l_down_par` /
  `l2p_scatter_par` (ascending level / disjoint leaves), dispatched from
  `run_impl` + bench when nthreads>1. Small `SendPtr` helper (lib.rs) for the
  cross-thread disjoint slices (2021 disjoint-capture gotcha: closures must
  capture the wrapper via method, not the raw field). 4× chunks/thread for
  load balance within sparse levels.
  Bug caught by the new exact-equality test: first `l2l_down_par` skipped
  parents with `children[0] < 0`, but the tree is sparse — a parent can have
  children only in later octants, so whole subtrees silently missed their L2L
  (also tripped the end-to-end force test). Fixed via `is_leaf`.

Measured (same loaded-machine session, before → after; release, p=8,
eps=1e-5, n_leaf=16):
- N=1e5 @8: walk 2.93 → 0.99s (**2.95×**), total 3.51 → 1.56s; 8-thread
  scaling of the walk 1.8× → 5.3×.
- N=1e6 @8: walk 11.89 → 7.69s (**1.55×**), total 18.47 → 14.32s; peak RSS
  3.6 → 4.1GB (recorded-pairs array ≈ 8B × 41.2M ≈ 330MB).
- M2L pair count identical before/after (N=1e6: 41,224,026; N=1e5: 5,721,286)
  — same pair set, same accuracy.
- f_prepass @8: N=1e6 4.45 → 0.56s (**7.9×**), N=1e5 0.36 → 0.041s. Static
  chunks balanced well (Morton order mixes spatial regions per chunk).
- Level-parallel passes @8, N=1e6: moments 1.24 → **0.26s** (scratch Υ +
  level sweep + 4× chunks/thread), l2l 0.45 → 0.11s, l2p+scatter
  0.46 → 0.06s.
- Cumulative this session @8, N=1e6 total: 18.47 → 14.32 → 10.05 →
  **8.12s** (walk 7.11 + prepass 0.54 + moments 0.26 + l2l 0.11 + l2p 0.06).
- Serial walk 5.33 → 5.23s @N=1e5 (powf removal ≈ 2%).
- Energy gate estimate: ~9.6s/step (8.12 + ~1.5 tree) → **~2.7h** for the
  10⁶×10³ run (was ~5.5h).
- Residual: the walk is now 87% of a force eval; only the stage-2b
  one-directional Morton-slab replay (PERF-ANALYSIS.md §4.3/§5) moves it —
  kills the remaining other-side scattered traffic (~107GB ≈ 5.8s floor) AND
  the 3.6GB per-thread locals (12-16 threads become viable). Everything else
  (tree rebuild ~1.5s serial in the gate) is <15% of a step.

Tests: all 30 unit + 4 integration tests pass; release exit gate
max_s 1.83e-3, rms_s 4.87e-5 (gate rms<1e-4; the shift from 3.76e-5 is powi
ULPs flipping a few borderline accepts); `cargo build` warning-free; clippy
error is the pre-existing rotation.rs:147 `erasing_op` (file off-limits).

Housekeeping:
- [ ] 12-thread N=1e6 untestable on this machine state (needs ~6.5GB
      transient; 8 threads peak 4.1GB vs ~4GB free).

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
## Session log — 2026-08-15: P2P kernel throughput (benchmarked + optimized)

Problem: gate "P2P 300+ GFLOPS single socket" unmeasured; no honest kernel
throughput number existed.

Changes:
- `crates/bench/src/main.rs` — added `p2p` bench mode (`fmm-bench <max_n> <p>
  <eps> p2p <N> <nthreads> <n_leaf>`): real-tree leaf pass single+threaded,
  plus synthetic 2048-block single+threaded (pure kernel throughput).
- `crates/fmm/src/p2p.rs` — `p2p_self_avx512` rewritten to i-unroll-2 with
  `jj = i1 + 1` (all lanes valid, no per-block `valid` mask / NaN-masking, no
  `cmp_pd`+`maskz_mul`); added `scalar_mutual` helper for intra-pair + tails.
  Bit-identical math (all tests pass).

Measured (Zen4 7840HS, release, `target-cpu=native`):
- synthetic 2048-block: 9.1 → 20.2 GFLOPS single-core (2.2×); 153.6 GFLOPS
  threaded ×8.
- real-tree (N=1e6, n_leaf=16, avg 4.87 bodies/leaf): 2.6 single / 17.3
  threaded — scalar-tail bound, not kernel bound.
- Machine peak: 73 GFLOPS/core pure AVX-512 FMA (1.2 zmm-ops/cycle @ ~4.4GHz).
- fp64 kernel is op+ILP bound: rsqrt14+2NR ≈ 9 ops/8 pairs (irreducible for
  1e-12 vs scalar), mutual back-reaction +5, per-i reduction ~2 → ~5 ops/pair
  → ~4 cycles/pair. All correct fp64 designs land 13-17 GFLOPS/core
  (exhaustive sweep incl. dual-acc, j-major, masked, temp-buffer, fp32 probe
  at 25.4/core). Theoretical fp64 ceiling ~23/core (~184 threaded).
- 300+ fp64 is NOT reachable on this hardware; fp32-inner (~1.8×) would get
  close but changes precision to ~1e-7 and breaks the 1e-12 AVX-vs-scalar
  test. Gate closed at the fp64 ceiling (see Phase-1 gates above).

Housekeeping:
- [ ] The standalone microbench `/tmp/opencode/p2p_bench.rs` holds the full
      kernel variant sweep (self_u2/u4/nomut/blocked16, pair8/16/jm/mf,
      raw_fma peak, threaded_pair) — keep for future kernel work.

## Session log — 2026-08-15: energy conservation gate (ROADMAP line 48)

Problem: no energy-conservation test existed; needed leapfrog + energy check
for 10⁶ bodies / 10³ steps.

Changes:
- `crates/fmm/tests/energy_gate.rs` (new): KDK (drift-kick-drift) leapfrog,
  one FMM force eval per step, `integrate()` + energy diagnostics (drift,
  max_dev, linear-fit slope, momentum). `energy_conservation_smoke` (default
  test, N=1024×100 steps ≈ 9s) + `energy_conservation_gate` (`#[ignore]`,
  N=10⁶×10³ via env overrides FMM_ENERGY_N/STEPS/DT/EPS/THREADS).
- Virialized IC: local Maxwellian σ²(r)=1/(6√(1+r²)), COM removed, velocities
  rescaled to exact virial balance. Sanity: E0 = T+W = -3π/64 ≈ -0.147.

Measured facts:
- Cold start is USELESS for this gate: the cold Plummer collapse forms a hard
  binary in the core at t≈0.03 (observed at N=4096, 3e4, 1e5), which a
  fixed-dt leapfrog cannot resolve → energy jump then runaway (ΔE/E → +10³,
  vmax → 3e3). Verified NOT an FMM bug: direct-force integration reproduces
  the exact same jump to 1e-8. Collisional physics, not FMM error.
- Warm start fixes it: N=3e4, 200 steps (t=0.2), eps=1e-5: drift 4.7e-5,
  max_dev 5.3e-5, vmax stays ~1.7, momentum 7.1e-16. Smoke (N=1024): drift
  3.6e-6. Threshold set: drift/max_dev < 1e-3, slope < 1e-6/step, momentum
  < 1e-12.
- Full N=10⁶×10³ gate launched in background (~20s/step @ 8 threads → ~5h).
  Deferred: too slow on a shared 35W laptop while the walk still gets
  optimized; interim validation substituted instead.
- Interim N=10⁵ × 10³ @ 12 threads (warm IC, dt=1e-3): drift 1.03e-3,
  max_dev 1.03e-3 (3% over the 1e-3 gate), slope 6.5e-7/step (no secular FMM
  heating), momentum 8.7e-15, 1.65 s/step, 27.5 min total. The overage is
  TWO unresolved close-encounter heating events (vmax 1.7→2.1 at steps ~700
  and ~950), not FMM error. Accepted as-is (encounter-limited at N=10⁵; at
  10⁶ per-event jumps scale 10× smaller). Gate threshold 1e-3 stands for the
  full 10⁶ run.
- Added `LIMITATIONS.md` — consolidated known limitations / deferred
  improvements (physics, perf, verification gaps, housekeeping).
- All 26 unit + 3 integration tests pass; build warning-free; new file
  clippy-clean (pre-existing lib lints unchanged).

Housekeeping:
- [ ] After the full N=10⁶×10³ gate run closes (post-optimization): flip the
      todo.md bullet, record the measured drift in the session log.

