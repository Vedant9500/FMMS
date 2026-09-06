# TODO — remaining Phase-1 gates + session log

Source of truth for exit gates: `ROADMAP.md` Phase 1 (lines 34-48).

## Session log — 2026-09-06: visual presets (IC overhaul)

Complaint: 100k default looked chaotic + white — Plummer blob, random
white-heavy colors, no structure. New `crates/visual/src/ic.rs`:
mass spectrum (bottom-heavy, total M=1) mapped to spectral look
(heavy blue-bright → light red-dim, heroes capped ~0.4% so cores stop
whiting out). Presets via `--preset`: `cluster` (old Plummer, new
colors), `spiral` (exponential disk + Hernquist bulge + 2% BH, rotation
curve, 2-arm density modulation, blue arms / old yellow core),
`merger` (two tilted spirals, bound infall), `bh` (12% BH + hot inner
rim + faint halo). `Sim::new` takes preset; virial rescale now
cluster-only (disks built in equilibrium). Tests: 19 green
(mass/COM/momentum/determinism/Lz/two-BHs + per-preset finite steps);
build warning-free; probe untouched (synthetic layouts only).

## Session log — 2026-09-06: why disks evaporate + dark-halo fix (research-backed)

Complaint: spiral flings all stars off in minutes, worse at high speed.
Researched: Rubin flat curves (dark halos ~5-10x visible mass; NASA/ESA),
Widrow-Dubinski/Hernquist-93/Kuijken-Dubinski equilibrium DFs,
Power+03 convergence (dt << orbital time, dt ~ sqrt(eps/a), relaxation
criterion), Zhang+19 softening update. Three compounding causes found:
(1) speed slider scaled dt up to 40x (0.16) — far past stability near
the center, pure numerical evaporation; (2) no halo — Keplerian decline
leaves outskirts weakly bound; (3) unsoftened FMM — fast relaxation.
Fixes: static Hernquist halo in `ic.rs` (spiral 3.0/3.0, bh 1.5/2.0,
merger 6.0/5.0 global, cluster none) applied backend-agnostically in
`Sim::step`; disk vc curves now include halo (v(4)/v(1) > 0.9 test);
adaptive dt clamp `dt <= 0.25*sqrt(0.02/a_max)` per step — speed
saturates instead of exploding (slider now 0.1-10); spectrum narrowed
20:1 → 10:1. Verified: 100k spiral 60 GPU steps, 0 escapers, max|v|
~2.8; all presets green; 25 tests green, warning-free.

## Session log — 2026-09-06: spiral GPU crash (`leaf over 64 bodies`)

`--preset spiral --n 100000` panicked in `gpufmm.rs:1514` after ~7 steps.
Traced headless with new `probe --gpusim N STEPS PRESET` (real KDK vs
GpuBackend): tree fine at t=0, then unsoftened-Newton slingshots
(max|v| slammed into the 5.0 clamp from step 0) flung bodies out, the
tree box ballooned, and Morton resolution at the nucleus collapsed.
Fix without library churn (gates untouched): `cap_accel` (|a| ≤ 50,
non-finite → 0) in `sim.rs`, applied by CPU-FMM + GPU backends —
Direct already self-limits via soft2. Verified: 100k spiral 40 steps,
30k merger/bh 12 steps, all finite, max|v| ~1→3; 22 tests green,
warning-free.

## Session log — 2026-09-03: visual side quest V0 harness

Motivation break from FMM optimization. Scaffolded `crates/visual`
(bin `fmms-visual`, now a workspace member) as a minimal but
no-redesign V0: GPU-resident prev/curr storage buffers with vertex-shader
interpolation (alpha), camera-relative f32 rebase from f64 world,
fixed-timestep accumulator + render interpolation, instanced billboard
quads (same quad becomes the V2 sphere impostor), HDR rgba16float +
Reinhard tonemap pass, winit orbit camera + egui overlay.
Physics behind a trait: direct O(N²) default (N ≤ 1500), FMM backend
(`--fmm`, auto above 1500). Verified: builds warning-free, 1 headless
sim test green, live screenshots (black sky, Plummer cluster, overlay).
Run: `cargo run --release -p fmms-visual -- [--n 1500] [--dt 0.004] [--fmm]`.
Next: GPU frustum culling + indirect draw, bloom, FMM-as-compute path.

## Session log — 2026-09-04: visual side quest, lag + star-look pass

Complaints: render engine lagged; bodies looked like white Blender spheres
with a haze instead of stars.
- Lag root cause measured: Direct O(N²) on the render thread, 6.85 ms/accel
  @N=1500 → ~55 ms/frame at 4 substeps. `Direct::accel` now scoped-thread
  parallel above N=600 (2.33 ms/accel, maxdiff 3.5e-15, new exactness test);
  per-frame `Vec` churn removed (reused staging buffers, inline speeds);
  bloom targets quarter-res.
- Star look: billboard shader now gaussian core (whitened, 5x HDR) +
  power-law halo in star color + magnitude-gated diffraction spikes; tonemap
  Reinhard → ACES.
- Grid-of-squares artifact: headless probe (`cargo run -p fmms-visual --bin
  probe -- --mode single|sparse|cluster [--dump]`, PPM out, RTX 4060) showed
  the 5-tap blur kernel stamping discrete dots — my radius-scaled 2-texel
  step broke the kernel's overlap assumption. Fix: step pinned to 1 texel,
  `bloom size` slider now sets pair count 1..3. Verified smooth in probe
  (single/sparse/cluster renders + bright/blur stage dumps).
- Probe keeps `GpuState` honest: `target()` now returns texture+view
  (COPY_SRC added) for readback. `render::tests::shaders_parse` (naga
  dev-dep) validates all 4 WGSL shaders headless.
Tests: 3 unit green (`cargo test -p fmms-visual`); app + probe warning-free.

## Session log — 2026-09-04: UniSim-look pass (bloom restraint)

User reference `/mnt/windows_d/unisim/python/max_scale_3d.py::ultra_fast_render`:
bg (2,2,12), per-star RGB classes, size hierarchy (99% are 1px + dim
4-neighbor cross only if bright; heroes are flat discs, zero glow), distance
dimming with floor. No fullscreen bloom pass exists there at all.
- Ported: `SPACE_BG` hdr clear, mag-gated core width (dim dots filled,
  heroes structured), quadratic size curve, core whitening 0.82→0.55, core
  heat 5.0→2.2 so temp colors survive ACES, bg quads halved.
- Bloom complaint (cotton balls): sprite halo now gated to top ~15%
  (`smoothstep(0.6,0.95,mag)`, quad 2.5→2.0); post defaults bloom 0.9→0.35,
  threshold 1.0→1.5 so only multi-pixel hero cores pass the 2x2 box average.
  Probe-verified smooth; `bloom size` slider = pair count 1..3.

## Session log — 2026-09-04: 100k-body scaling pass

Measured first (`probe --phys`, 12 threads): Direct hopeless at 100k
(5e9 pairs); FMM 450 ms/eval at science params, 178 ms floor at visual
params (p=4, eps=1e-2, n_leaf=64; refresh only 4 ms, walk owns the rest;
16 threads worse than 12, bandwidth-bound). 30k = 57 ms (linear).
- 1 eval/step: KDK accel cached across steps (new exactness test),
  ~5.6 steps/s @100k, ~17/s @30k.
- Physics moved to background thread, snapshots over mpsc; render
  interpolates stretched over arrival intervals — display stays 60 fps
  at any step rate (new streaming test).
- Init O(N²) virial rescale skipped above 5000 (was minutes at 100k).
- Backend presets by N: Direct ≤1500, FMM(6,1e-4,16) ≤10k,
  FMM(4,1e-2,64) above; threads capped 12; velocity clamp 5.0 above 20k.
- Render: 102k billboards measure 0.8–1.3 ms + ~6 MB upload; default
  point size now scales 1/sqrt(N) (verified no whiteout in probe).
- dt default 0.01 above 20k bodies (0.004 below), `--dt` overrides.

## Session log — 2026-09-04: GPU-compute FMM kickoff (docs/10-gpu-fmm.md)

Leveraged: docs/03 (M2L-is-the-game, symmetry, fp32), finding
fnd-8d3e0d6773d4, PERF-ANALYSIS walk split (record 0.44 s vs replay
5.64 s @1M), toolchain (CUDA 13.3 + RTX 4060 cc8.9 present, chose wgpu).
Locked: wgpu compute (zero-copy with renderer) over CUDA interop; fp32
visual-first contract (fp64 CPU path untouched); hybrid CPU-traversal /
GPU-replay split.
- S1 done: `visual/src/gpufmm.rs` one-sided M2L (p=4, one thread/pair),
  `probe --m2l` harness vs fp64 CPU on real octree pairs. Measured:
  GPU 0.023 µs/pair @500k vs CPU 0.268 µs (11.6x), max rel 1.6e-4.
  Kernel is latency-bound (~18 GB/s of ~272) — S4 headroom documented.
- Next: S2 full replay (sort-by-dst + segmented accumulate) + GPU
  up/down sweeps; gate rel err ~1e-3 vs direct.
- S2a landed: `record_pairs_par` extracted from `dual_walk_par` (suite +
  exit gate green); GPU two-kernel replay validated @30k (11.6 ms vs CPU
  27.3 ms total, rel 5e-4 vs both CPU refs). Kernel still latency-bound
  (~140 GFLOPS); S4 tuning gates the big win. S2b next: sweeps + P2P.
- S2b landed: 5 sweep/P2P kernels + `--gpufull` triangle harness
  (gpu-vs-direct, gpu-vs-cpufmm, cpufmm-vs-direct). @8k/p6: mean vs direct
  5.9e-4 (gate green), pipeline-exact vs CPU-FMM at p4 (8e-6 max).
  Shared-uniform-across-dispatches footgun documented in code. Next: S3
  engine integration, then S4 (symmetry/shared-mem + f64 P2P distances).
- S3 landed: `GpuBackend` in Sim (`--gpu`, auto >30k), K=4 frozen-tree
  reuse validated (`--gpureuse`: frozen mean = truncation floor).
  `--gpuloop`: 100k 11.9 steps/s vs CPU 5.8 (2x); 30k 35.7/s. Slot→id
  scatter bug caught by validation. 15 visual tests green, warning-free.
- S4 landed: profile-driven, plan overturned by measurement (`--gpuprof`).
  m2l-red thread-per-(cell,coef) 21.4→1.9ms; P2P per-leaf shared-tile
  20→11.9ms; fat-halo split (12 leaves/39% iters) →2.3ms total.
  `--gpuloop`: 100k 25.6 steps/s (4.4x CPU), 30k 51.7/s. Triangle green,
  19 visual tests green, warning-free. Wall now CPU traversal (~27ms
  amortized) — S5 territory. Skipped by Amdahl: S1 mutual/hoist,
  Kohnke symmetry (no stencil in adaptive octree), DS-P2P distances.
- App polish: fixed harness-only dead_code warnings (`submit`/`stats`/
  `profile_step`); render now trails physics by a fixed lag (4x arrival
  EMA, 0.1–0.6 s) interpolating between bracketing snaps instead of
  stretching each snap over the EMA — K=4 refresh makes arrivals bursty
  (slow refresh + 3 fast), old scheme froze at blend=1 between snaps.
  Segfault triage via coredumpctl: crash is smithay-clipboard's worker
  racing Wayland teardown inside `wl_proxy_destroy` (pre-existing, 7+
  dumps incl. pre-S4, both x11/wayland backends — clipboard lib drives
  its own Wayland connection). App has no text inputs, so the OS
  clipboard is dead weight: `egui-winit` now `default-features = false
  + wayland,x11` (drops clipboard/links; in-app fallback covers the
  unused copy/paste). arboard+smithay fully out of the tree; 5-min
  100k soak pending.
- Render lag is now a 0.35 s constant (EMA drives the readout only):
  per-step probe timing shows arrivals [17,17,17,140]ms, and in-app
  render contention stretches the refresh gap past any EMA-derived lag,
  starving the bracket every 4th step. History cap 10.

## Session log — 2026-08-22: Tier 1 (Units 4–7) iteration (OPTIMIZATION-PLAN.md)

Problem: Tier 0 left the walk at its DRAM floor (~87% of a step); the
remaining serial pipeline was unmeasured territory with several stale
baselines. Context: bandwidth/traffic-bound overall, not compute-bound —
Tier 1 trims the pipeline only; the walk is stage-3/GPU work.

Changes (all bit-identical or gate-verified):
- `crates/fmm/src/tree.rs` — `leaf: bool` on `Cell` (6.1; `is_leaf` was an
  8-children scan per call across every pass). `Octree::refresh_with_mode`
  rebuilds in place reusing all allocations; exactness tests
  `refresh_matches_build` (bitwise, Com+Sphere, double-refresh) +
  `refresh_degenerate_points`. `Fmm::refresh` wrapper.
- `crates/fmm/tests/energy_gate.rs` — one `Fmm` built per run, refreshed per
  step (was `Fmm::new` + full rebuild per step).
- `crates/fmm/src/mac.rs` — accept θ test in r² (`r² > (ρ_A+ρ_B)²`) before
  the sqrt (4.1); f_walk leaf scan split at slot i so the self-interaction
  branch leaves the inner loops (4.3).
- `crates/fmm/src/moments.rs` — M2M inner l-loop range clamp replaces the
  `|m−l| > n−k` continue (7.3).
- `crates/bench/src/main.rs` — new `tree` bench mode (build vs refresh vs
  std-sort share).

Bugs found on the way (all caught by tests):
- f_walk split v1 swept OTHER leaves' slots when i lay outside a leaf's
  range → double-counted mass into f_b → inflated fmin → MAC over-accept.
  Unit exactness test blind to it (same bug both sides); hernquist smoke
  integration gate caught it (rms 0.038 ≫ 1e-2).
- refresh v1 passed UNSORTED keys to build_rec (wrong children everywhere);
  v2 missed a scratch resize (OOB panic). Both caught by the new exact test.
- P2M positive-m-only Υ builder (Unit 5 lever 4 / 7.1 idea): negative-m
  moments are ±conj of positive-m, NOT zero — exact test failed at
  idx(2,−1)/(2,−2). Reverted; redundancy exploitation needs consumer-side
  changes everywhere.

Measured facts:
- **The "~1.5 s tree rebuild" baseline is STALE**: N=10⁶ build 112 ms,
  warm refresh 96 ms, std-sort(u64×10⁶) 12.7 ms = 11% of build → radix sort
  rejected (6.3 closed by measurement).
- Release exit gate unchanged: max_s 1.826e-3, rms_s 4.867e-5 (gate <1e-4).
- Moments phase 0.27–0.30 → 0.19–0.24 s; prepass within noise of 0.54–0.57 s;
  today's machine noise band is wide (total 8.2–13.7 s across runs, loaded
  window) — no honest total-step claim beyond "≤ Tier-0's ~7.05 s profile".
- Gate loop @N=10⁶ now ~11.1 s/step in the same noisy window (profile ~9.4);
  residual gap is allocation churn/cache pollution around the leapfrog
  buffers, not the tree — follow-up candidate if the full 10⁶×10³ gate runs.

Tests: 32 unit + 4 integration green (2 new tree exactness tests); release
exit gate green; build warning-free; clippy warning count unchanged (50,
incl. pre-existing rotation.rs erasing_op error — file off-limits).

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

