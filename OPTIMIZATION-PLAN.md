# OPTIMIZATION-PLAN.md — per-file optimization units for `crates/fmm/src`

Companion to `PERF-ANALYSIS.md` (root causes + staged plan) and `todo.md`
(session log). This file divides the 14 source files into **independent
optimization units**: one file (or one part of a file), one owner at a time,
each with its own measured baseline, target, guard tests, and contract.

**Ground rules (from AGENTS.md, apply to every unit):**

- Measure before and after; release build (`target-cpu=native`), same machine
  state, N=10⁶ @ p=8, eps=1e-5 profile mode unless the unit says otherwise.
- `m2l_mutual`/`m2l_pair` stay **bit-compatible** with their naive baselines
  until a unit explicitly relaxes the contract (only Unit 3 does, stage-3).
- `rotation.rs` internals and `ROT_P = 26` are off-limits without re-measure;
  `MAC` factor 8 is load-bearing — never tune casually.
- Every unit ends with: full `cargo test` green, `cargo build` warning-free,
  numbers appended to the unit's log line in this file.

**Current baseline (2026-08-16, N=10⁶ @ 8 threads):** total 8.12 s/step =
walk 7.13 + prepass 0.54 + moments 0.26 + l2l 0.11 + l2p 0.06, plus ~1.5 s
serial tree rebuild in the gate loop.

---

## Tier 0 — hot path (walk + kernels): 87% of a step

### Unit 1 — `walk.rs` (583 lines): the bucketed replay

**Part:** `dual_walk_par` phase 1/2, `partition_pairs`, `Counters`,
`process_pair`. The single hottest file (7.13 s of 8.12 s).

**Contract:** pair set identical to serial walk (test:
`parallel_matches_serial_*`, `partition_pairs_buckets_and_preserves_multiset`).
Accumulation order may change only within already-sanctioned reassociation
tolerances.

**Levers, in measured-value order:**

1. **Bucket-count sweep** (cheap first): `nbuckets = 1024` is a guess.
   Sweep 256 / 1024 / 4096 / 16384 at N=10⁶; watch walk time + RSS
   (`ends` array is O(nbuckets), pairs array unchanged).
2. **Bucket key = `min(a,b)`** — try `max(a,b)` and `a` (source) variants;
   the replay locality depends on which side stays cache-resident.
3. **Per-thread locals → shared locals with Morton-slab ownership**
   (PERF-ANALYSIS §4.3): kills 8 × 450 MB zero + reduce pass (~0.4 s) and
   the 4.1 GB RSS ceiling. Requires the directed replay — only with Unit 3's
   cheaper one-sided kernel (the 2026-08-16 dst-slab attempt regressed 36%
   with the current kernel; do not repeat without it).
4. **P2P inline path**: near-field writes go to per-thread `Bodies` clones
   (64 MB/thread). Consider per-thread *accel-only* accumulators
   (4 × 8 B × N instead of 8 × 8 B × N) — `m`/`x`/`y`/`z` are read-only.
5. **`is_leaf` in the walk loop**: `tree.rs` Unit 6's cached flag helps here
   too (called per pop).

**Baseline:** walk 7.13 s (N=10⁶, 8 thr) · **Target:** ≤ 6 s (bucket/key
sweeps), stretch ≤ 4.5 s (slab replay + Unit 3).

**Session log — 2026-08-21 (Unit 1 iteration):**

Phase instrumentation (`walk_phase_ms()`, temporary like the counters) on the
2026-08-21 quiet machine, N=10⁶ @8T: baseline walk 7.07 s = record 0.44 +
merge 0.02 + partition 0.21 + **replay 5.64** + reduce 0.22.

Applied (all tests green, exit gate rms_s 4.87e-5 unchanged):
- **Parallel reduce** (locals element-range split + bodies k-range split;
  each element sums accumulators in thread order → bit-identical):
  reduce 219→86 ms.
- **Per-pass counter reset in bench profile** — printed stats were still 2×
  inflated (53.5M pops); now 26.7M pops / 20.6M M2L per evaluation.
- Result: walk 7.07→6.08–6.35 s, total 8.04→~7.05 s.

Measured negatives (reverted, do not retry blind):
- **Bucket sort by min / (min,max)**: replay flat (5.39→5.41 s) — the walk's
  DFS discovery order is already spatially coherent; sorting costs up to
  370 ms. Walk-order bucketing is sufficient for min-side locality.
- **Software prefetch (i+4) in the replay loop**: no gain — the loop is
  bandwidth-bound, not latency-bound (prefetch adds traffic).
- **Thread scaling of the replay is NEGATIVE** past 8T (5.38→5.52 @12T,
  5.93 @16T): DRAM-bandwidth-saturated. Confirms §4.1: only moving fewer
  bytes per pair helps now.
- THP is `[always]` on this machine — TLB is not the residual gap.

Remaining Unit-1 headroom is entirely behind the stage-3 one-sided kernel
(directed replay halves RMW traffic). Optional micro-item (not done, ~2%
ceiling): parallelize `partition_pairs` histogram/scatter (~170 ms serial).
Optional structural item: share read-only x/y/z/m across workers instead of
per-thread `bodies.clone()` (~256 MB copy traffic/step, ~0.5%; touches p2p
signatures).

---

### Unit 2 — `translate.rs` (439 lines): M2L naive kernel + L2L/L2P passes

**Part:** `m2l_mutual_scratch` (the O(p⁴) kernel — 20.6M calls/step),
`M2lScratch`, `l2l_into`, `l2p_into` + the `_par` wrappers. The kernel is
~1.68 µs/pair, dependency-chain limited (~1.3 flops/cycle serial).

**Contract:** `m2l_mutual_scratch` bit-compatible with `m2l_mutual` (unit
tests + `m2l_rot_matches_naive` protect this). L2L/L2P bit-identical to
serial (exact-equality tests exist).

**Levers:**

1. **Θ-build reciprocal hoist** (stage-3 opener, *no* contract change):
   `theta_table_into` does ~60 divisions per pair (Unit 5's `1/r²` scaling
   per entry). Hoist `1/r²` — already done — but the *pair* cost is the
   recursion itself; measure a fused `theta_table + m2l` entry point that
   skips the `ths` parity copy (fold `(-1)^n` into the accumulation sign).
2. **FMA reassociation of the inner loop** (relaxes bit-compat → relative-
   error contract): `fb_r += ma.re*t.re - ma.im*t.im` → `_mm512_fmadd`
   chains, SoA re/im layout for moments/locals (PERF-ANALYSIS §4.1).
   Gate: rms_s must stay < 1e-4 (currently 4.87e-5, ~2× margin).
3. **L2L inner-loop shape**: `l2l_into` is O(p⁴)-shaped per parent-child
   edge with `C::mul` temporaries; same FMA/SoA treatment as (2) if it ever
   matters (0.11 s today — low priority).
4. **Delete `m2l_one_sided`/`m2l_one_sided_scratch`?** No — kept as stage-3
   building block per todo.md. Mark `#[cfg(test)]`-gated if dead-code lints
   complain.

**Baseline:** ~11k flops/pair, 1.68 µs/pair (cache-resident) · **Target:**
stage-3 SIMD kernel 2–3× on the convolution; end-to-end walk ≤ 4 s with
Unit 1.

**Session log — 2026-08-21 (Unit 2 iteration):**

New `m2l` bench mode (`fmm-bench dummy 8 1e-5 m2l <N> <p>`): cache-resident
4096 accepted leaf pairs, splits kernel cost. Measured: full kernel
1633 ns/pair = Θ build 149 ns (**9%**) + convolution+setup 1484 ns.
Convolution ≈ 1365 inner iterations → ~4.8 cycles/iter — latency/issue bound,
~80% of scalar FMA roofline already. Θ-build reciprocal hoisting is NOT a
lever (9% share).

Kernel reshapes tried (all reverted; bit-compat reference untouched):
1. 4-lane modulo accumulator split: 3080 ns/pair (0.53×) — stack spill from
   the lane indexing. Dead end, do not retry in this form.
2. Load-shed reshape (drop `ths` parity copy + conj folding into arithmetic;
   8→6 loads/iter): 1525 ns/pair (1.06×) — load ports were NOT binding.
3. Product-accumulator FMA reshape (independent mul_add chains, parity as
   subtractor): 1516 ns/pair (1.08× cached) — near the scalar issue ceiling.

End-to-end A/B through the parallel replay: N=10⁶ replay 5322 vs 5289 ms
(noise); N=10⁵ replay 740 vs 728–737 ms — the reshaped kernel gains nothing
even where compute should matter, because even N=10⁵'s ~60 MB working set
thrashes L3. Effective per-pair cost at N=10⁶ (5.32 s / 20.6M) is 258 ns×8
threads ≈ 2.07 µs vs 1.63 µs cached: only ~20% of replay time is kernel,
the rest is DRAM stall on the scattered moments/locals traffic.

**Revised conclusion for stage 3:** SIMD/FMA kernel work alone cannot pay —
the memory layout must change first or simultaneously. The directed
one-sided replay (halves RMW traffic) and the cheaper/SIMD one-sided kernel
are one work package, not sequential stages.

Kept from this unit:
- The `m2l` bench mode (kernel cost split, reusable).
- Phase instrumentation from Unit 1.
- Negative results above (do not retry blind).

---

### Unit 3 — `rotation.rs` (559 lines): O(p³) M2L (p ≥ 26 path)

**Part:** `RotCtx::new` (swap matrices + folded diagonals), `m2l_pair`,
`apply`. Currently only exercised at p ≥ 26 (bench default p=8 doesn't hit
it; accuracy runs at p=28–30 do).

**Contract:** `m2l_rot_matches_naive` (rel err ≤ ~2e-8 up to p=32);
`swap_involution`; `swap_rotates_theta`. **`ROT_P = 26` is measured —
re-measure with the 4000-pair microbench after any change here.** Clippy
`erasing_op` at rotation.rs:147 is pre-existing; needs explicit go-ahead
before touching.

**Levers (only after Units 1–2 make p ≥ 26 the default path, or for
high-accuracy configs):**

1. **Swap matvec as 3 real dot-products per output** — the inner loop is
   `sr += w * u.re; si += w * u.im;` over a real matrix row: auto-vectorizes
   poorly due to the complex shuffle. Try interleaved re/im storage of
   `tmp`/`tmp2` so the matvec is one contiguous dot product.
2. **Precompute `pha`/`phb` for repeated α/β**: pairs sharing a parent often
   share directions to ~2⁻¹¹; quantize (α, β) to a 4k-entry cache.
3. **`apply()` is called twice per pair with conjugate phases** — fold the
   B→A parity into the diagonals (as the naive kernel does with `ths`) to
   skip the second phase pass.

**Baseline:** crossover p=26 (0.97×) → p=32 (0.85×), rel err ≤ 2e-8 @ p=32 ·
**Target:** crossover p ≤ 20 without accuracy loss; only then consider
lowering `ROT_P`.

**Session log — 2026-08-21 (Unit 3 iteration):**

Re-measured the crossover first (new `rot` bench mode, multi-pair accepted-
pair microbench — replaces the single-pair variant which had a broken error
check and noisy timing): **ROT_P=26 no longer reproduced** on this machine —
rotation lost at p=26 (1.06×) and only won from p≈30.

Root cause found: the four swap-matvec inner loops in `apply()` are single
FADD chains of length nb (up to 69) × 4-cycle latency. A cross-row 4-block
(`dot4`) measured WORSE (+4%, array-return overhead); the fix was **within-
row** accumulator splitting (`dot_row`: 8 register-resident partial sums,
tail rows scalar). Rotation kernel: 1.2–1.5× faster across p=16–34.

New crossover table (multi-pair, cache-resident):
p=16 1.22 · p=18 1.10 · **p=20 0.98** · p=22 0.90 · p=24 0.83 · p=26 0.76 ·
p=30 0.68 · p=34 0.60. Rel err ≤ 5.4e-13 up to p=34 (contract ≤ ~2e-8).

Changes kept:
- `rotation.rs` — `dot_row` reshape of all four matvec sites; `ROT_P` 26 → 20
  (re-measured per AGENTS.md; asserted nowhere, dispatch is `p >= ROT_P`).
- bench `rot` mode kept for future crossover re-measures.

Verification: rotation unit tests (swap involution/rotates-θ,
m2l_rot_matches_naive @1e-9) pass; full suite 30+4 green; release exit gate
unchanged; N=10⁶ profile unchanged at p=8 (below threshold — naive path,
bit-compat preserved).

Note: high-p configs (p ≥ 20: accuracy runs, dense-field studies) now get
up to 1.5× faster M2L. The p=26 doc numbers were stale hardware/state
measurements — re-measuring before trusting constants, as AGENTS.md says.

---

## Tier 1 — serial pipeline (13% of a step, all parallelized already)

### Unit 4 — `mac.rs` (235 lines): error estimate + f prepass

**Part:** `accept` (2 calls per pop, ~68M evaluations/step),
`error_estimate` (already `powf`-free), `f_walk`/`f_prepass_par`
(0.54 s @ N=10⁶).

**Contract:** `f_prepass_par` bit-identical to serial (exact-equality test);
MAC factor 8 untouched; `theta >= 1` short-circuit order preserved.

**Levers:**

1. **`accept` early-outs**: currently computes `r` (sqrt) then `theta` then
   `error_estimate`. Compare `r²` against `(ρ_A+ρ_B)²` first — kills the
   sqrt on every rejected-by-θ pair (~6.9M/step).
2. **`error_estimate` inner sum**: `binom[k] * pows[base+k] * rpows[p-k]` is
   2 mul + 1 add over ≤ 9 terms; fine. Skip if (1) covers the win.
3. **f_walk leaf scan**: `if k == i` branch inside the hot loop — split the
   loop at `k == i` (two range loops) so the branch leaves the inner loop.
4. **f_prepass reuse across steps** (gate-only win): f_b changes slowly
   (LIMITATIONS notes this); recompute every K steps, measure drift impact
   on the MAC. Gate-only, needs energy-gate re-validation.

**Baseline:** prepass 0.54 s @ N=10⁶ · **Target:** ≤ 0.4 s from (1)+(3);
(4) is a gate-loop win measured separately.

**Session log — 2026-08-22 (Unit 4 iteration):**

- **accept r² gate**: θ test now compares `r² > (ρ_A+ρ_B)²` before the sqrt
  (~6.9M θ-rejected evaluations/step stop paying sqrt). ULP-level borderline
  accept flips possible by construction; release exit gate unchanged in
  practice: max_s 1.826e-3, rms_s 4.867e-5 (was 4.87e-5, gate <1e-4).
- **f_walk leaf-scan split at slot i** — first version had a real bug the
  unit tests missed: with `i` outside the leaf's slot range the second half
  loop swept *other leaves'* slots (double-counted their mass into f_b →
  inflated fmin → MAC over-accepted; hernquist smoke rms 0.038 ≫ 1e-2,
  caught only by the integration gate). Fixed by clamping both halves to the
  leaf range (`below = min(i, end)`, `above = max(first, i+1)`); accumulation
  order unchanged. Lesson: exactness tests that share the changed code path
  on both sides cannot catch this class — the N-vs-direct gates can.
- Both kept. `f_prepass_par_matches_serial` still exact-equal.

Measured (loaded-machine window): prepass/mac 0.57/0.51/0.76 s across final
runs vs baseline 0.54–0.57 — within noise; no honest claim beyond the sqrt
removal (~2–3% of a step ceiling).

---

### Unit 5 — `harmonic.rs` (295 lines): Θ/Υ table recursions

**Part:** `theta_table_into` (per M2L pair: ~81 entries, ~60 divisions via
`.scale(1/r²)`), `upsilon_table_into` (per P2M/M2M/L2L/L2P edge).

**Contract:** Dehnen Table-3 checks (`table3_low_orders`), parity,
homogeneity, Υ–Θ relation — all at 1e-12. **The recursion *values* are
sacred; the arithmetic *shape* may change only if those stay bit-identical
or the M2L contract is explicitly relaxed.**

**Levers:**

1. **Division hoisting**: each recursion step does `.scale(1.0 / r2)` /
   `.scale(1.0 / c)` — multiply by a precomputed reciprocal instead of
   dividing per entry. Bit-compatibility risk: `x * (1/c)` ≠ `x / c` in
   general — measure accuracy impact on the full suite before keeping.
2. **SoA re/im tables** for `C`-arrays (`th`, `ths`, `u`): makes Unit 2's
   SIMD kernel natural; do together with Unit 2 lever 2.
3. **Negative-m fold**: the conj+sign pass is O(p²) per call — fold the
   sign into the positive-m recursion's seeds instead (saves a full pass;
   bit-compat risk again, same gate).
4. **`upsilon_table_into` for L2P**: called per particle (10⁶/step) with
   p=8 → 81-entry table each. Consider a p=1-only fast path (L2P needs only
   n ≤ 1 rows of Υ): ~4 entries instead of 81. **This is the biggest single
   win in the file** — L2P is 0.06 s but the Υ build dominates its cost;
   measure `l2p` with a dedicated `upsilon_row01_into`.

**Baseline:** ~60 div + 81 complex ops per Θ table (20.6M/step) ·
**Target:** Θ build ≤ 0.5 µs/pair; L2P Υ cost ~30× down from (4).

**Session log — 2026-08-22 (Unit 5 evaluated, skipped):**

- **Lever 1 (Θ reciprocal hoist): dead** — Unit 2's `m2l` bench already
  measured Θ build at 9% of kernel cost, and the kernel is ~20% of replay →
  ceiling ≈ 2% of a step, not worth the bit-compat surface.
- **Lever 4 (L2P p=1-only Υ path): false premise** — `l2p_into`
  consumes `u[idx(k,l)]` for ALL k ≤ p−1 in the ψ₁¹/ψ₁⁰ sums; no n≤1-only
  path exists. L2P is 0.06 s anyway.
- **Negative result (new, measured)**: skipping the negative-m mirror pass
  in P2M via a positive-m-only Υ builder breaks the moments — Σ_a μ_a Υ_n^{-k}
  is ±conj(Σ_a μ_a Υ_n^k), i.e. *determined by* positive-m, NOT zero; the
  `compute_moments_par_matches_serial` exact test failed on idx(2,−1)/(2,−2).
  Exploiting the mirror redundancy means deriving negative m in every
  consumer (M2M, power, M2L, L2L/L2P) — a global layout change, out of scope
  for a 0.26 s pass. Reverted.
- Unit 5 stays closed this tier; SoA re/im tables belong to the stage-3/GPU
  package.

---

### Unit 6 — `tree.rs` (465 lines): build + geometry + queries

**Part:** `build_with_mode` (Morton keys, `sort_unstable_by_key`, recursive
split), `geometry` (bottom-up mass/cen/ρ), `levels_csr`, `is_leaf`.

**Contract:** `build_and_geometry`, `degenerate_points`, `sphere_mode_rho_*`
tests. Cell *ordering* (DFS, parents-before-children) is load-bearing for
`geometry`'s reverse sweep and `levels_csr` — keep it.

**Levers:**

1. **`is_leaf` flag**: `is_leaf` scans 8 children per call; it's called per
   pop in the walk and per cell in every pass. Add a `leaf: bool` to `Cell`
   (or check `children[0] < 0` only — but sparse trees break that, see the
   2026-08-16 l2l bug; the flag is the safe form). Bit-identical.
2. **Tree reuse across gate steps** (~1.5 s/step in the energy gate):
   `Fmm::new` rebuilds everything; add `Octree::refresh_positions` that
   re-keys + re-sorts + re-runs `geometry` without reallocating
   cells/order/keys buffers. Same output, allocation-free.
3. **Sort**: `sort_unstable_by_key` on u64 keys is ~1 s at N=10⁶. Try
   radix sort (8×8-bit passes, keys are 63-bit Morton) — measured decision,
   keep the faster.
4. **`geometry` Sphere mode allocates a Vec per internal cell**
   (`ritter` items): stack-allocate `[Option<...>; 8]`-style fixed arrays —
   only matters for Sphere mode runs.

**Baseline:** tree rebuild ~1.5 s serial (gate loop) · **Target:** ≤ 0.8 s
from (2)+(3); (1) is a walk-side win (Unit 1 benefits).

**Session log — 2026-08-22 (Units 6.1/6.2/6.3 iteration):**

- **6.1 `leaf: bool` on Cell**, set in `build_rec`; `is_leaf` returns the
  flag. All passes benefit; 32 unit + 4 integration green.
- **6.2 tree reuse**: `Octree::refresh_with_mode` rebuilds for new positions
  reusing cells/order/keys/idx allocations (bbox → re-key → sort idx →
  gather sorted keys → build_rec on cleared cells → geometry). New exact
  tests `refresh_matches_build` (bitwise cells+order, Com AND Sphere modes,
  double-refresh) and `refresh_degenerate_points`. `Fmm::refresh(parts)`
  wrapper added; energy_gate now builds ONE Fmm and refreshes per step
  (was `Fmm::new` per step). Two bugs caught during bring-up by the exact
  test: unsorted keys passed to build_rec (wrong children), then missing
  scratch resize (index OOB panic).
- **6.3 measured, rejected**: new bench `tree` mode (`fmm-bench … tree <N>
  [n_leaf]`). N=10⁶ n_leaf=16: **build 112 ms, warm refresh 96 ms,
  std-sort(u64×10⁶) 12.7 ms = 11% of build**. The "~1.5 s tree rebuild"
  baseline is stale (2026-08-16 loaded-machine estimate); radix sort has
  nothing to win. The real gate-vs-profile gap (~1.3–1.7 s in a loaded
  window at N=10⁶) is allocation churn + cache pollution from the
  interleaved leapfrog buffers (`to_parts` Vec churn, moments/locals
  realloc), NOT the tree — candidate for a follow-up runner-scratch item if
  the gate ever runs again.
- Kept: leaf flag, refresh API, tree bench mode.

---

### Unit 7 — `moments.rs` (214 lines): P2M/M2M + power

**Part:** `p2m_cell` (per leaf, per particle Υ build — 205k leaves × 4.87
bodies), `m2m_cell` (per internal cell, per child O(p⁴)-shaped shift),
`multipole_power`.

**Contract:** `compute_moments_par` bit-identical to serial (exact-equality
test); `monopole_is_mass` (dipole vanishes at CoM).

**Levers:**

1. **P2M: skip the Υ table, accumulate directly**: `moments += m·Υ(x_a−z)`
   builds an 81-entry table per particle then does an 81-length AXPY.
   Fuse: recurse Υ and accumulate in the same loop (saves the table
   round-trip; bit-compat risk — the accumulation order is identical, only
   the storage is elided, so it *should* be bit-identical: verify with the
   exact test).
2. **M2M: exploit the CoM center** — at CoM centers the child→parent
   dipole terms vanish for the *first* level but not deeper; measure before
   assuming.
3. **M2M inner triple loop** has a `continue` on `|m−l| > n−k` — restructure
   the l-range to `max(−k, m−nk)..=min(k, m+nk)` so the branch leaves the
   loop. Bit-identical (same terms summed, same order within each (n,m)).
4. **`multipole_power`**: only `pows[·, 0..=p]` per cell is needed by MAC;
   it's already minimal. Skip.

**Baseline:** moments 0.26 s @ N=10⁶ · **Target:** ≤ 0.2 s from (1)+(3).

**Session log — 2026-08-22 (Unit 7 iteration):**

- **7.3 M2M range clamp kept**: the `|m−l| > n−k` continue replaced by
  l-range `[max(−k, m−(n−k)) ..= min(k, m+(n−k))]` — same terms, same order,
  branch out of the inner loop. Exact-equality test green.
- **7.1 P2M Υ-fuse REVERTED (negative result)**: the "storage elision is
  bit-identical" premise was wrong in an unexpected way — a positive-m-only
  table builder (to skip the mirror pass) breaks P2M because negative-m
  moments are NOT zero (see Unit 5 log); and fusing recursion+accumulation
  without it showed no measurable gain in the 0.26 s phase. Dropped.
- Moments phase: 0.27/0.30 → **0.19–0.24 s** across final runs (baseline
  window 0.26–0.30) — consistent with the M2M clamp, within today's noise.

---

## Tier 2 — support cast (measure-only; touch only with cause)

### Unit 8 — `p2p.rs` (400 lines): near-field kernel

**Part:** `p2p_self_avx512` / `p2p_pair_avx512` + scalar references.

**Status:** closed at the fp64 hardware ceiling (20.2/153.6 GFLOPS
synthetic, 17.3 threaded real-tree; ceiling ~23/core). Real-tree throughput
is scalar-tail bound (avg 4.87 bodies/leaf → the 8-wide kernel rarely fills).

**Levers (only if n_leaf sweep reopens this):**

1. **n_leaf sweep (32/64)** — the real lever, but it's a *tree/walk* config
   change with MAC interplay; run it under Unit 1, not here.
2. **Pair-kernel i-unroll** like the self-kernel (pair path still 1 i/iter).
3. **fp32 inner** breaks the 1e-12 AVX-vs-scalar contract — out of scope
   without an explicit precision-contract decision.

**Baseline:** 17.3 GFLOPS threaded real-tree · **Target:** none (closed
gate); revisit only with n_leaf ≥ 32.

---

### Unit 9 — `bodies.rs` (74 lines): SoA layout

**Part:** `Bodies` (8 separate `Vec<f64>`), `from_parts`, `scatter`.

**Levers:**

1. **Interleave (x,y,z,m) into one `Vec<f64>` of stride-4** (AoS-of-SoA):
   P2P loads x/y/z/m of 8 consecutive j's from one cache line region
   instead of 4 streams. Measured decision — the AVX-512 kernel currently
   does 4 gather-ish `loadu`s; stride-4 makes them one load + shuffles.
   Risky for the 1e-12 contract only via reassociation (loads don't change
   arithmetic order — actually bit-safe).
2. **`from_parts` gather**: it's a 4-stream scatter from `parts[id]` —
   already optimal shape. Skip.

**Baseline:** P2P real-tree 17.3 GFLOPS (shared with Unit 8) · **Target:**
measure-only; adopt stride-4 only if the P2P bench moves > 5%.

---

### Unit 10 — `direct.rs`, `models.rs`, `fmm.rs`, `lib.rs`, `roofline.rs`

**Part:** validation + orchestration + scaffolding. Not hot.

- `direct.rs`: O(N²) reference — correctness matters, speed doesn't (it's
  the *baseline*, not the product). **Do not optimize**; a faster reference
  is a weaker reference.
- `models.rs`: IC samplers, run once per test/bench. The velocity sampler
  promotion (LIMITATIONS §Physics) is Phase-2 work, not optimization.
- `fmm.rs`: orchestrator — its only lever is *tree reuse* (Unit 6 lever 2)
  and passing `nthreads` through; the dispatch itself is free.
- `lib.rs`: `SendPtr` helper only. No work.
- `roofline.rs`: accounting constants. No work.

**Rule:** nothing in Unit 10 gets optimized without a measured profile
showing it matters.

---

## Execution order (dependency-driven)

```
Unit 6.1 (is_leaf flag)  ──┐
Unit 4.1 (accept r² gate) ─┼─► Unit 1 (walk sweeps) ──► Unit 5+2 (Θ/SoA/FMA)
Unit 6.2 (tree reuse)   ───┘                                │
                                                            ▼
                                              Unit 3 (rotation, only if p≥26
                                              becomes the working point)
```

1. **Quick wins first** (all bit-identical, low risk): 6.1, 4.1, 4.3,
   6.2, 7.1, 7.3.
2. **Walk sweeps** (Unit 1.1–1.2, 1.4): measured, reversible, no contract
   changes.
3. **The big one** (Units 5+2 together: SoA re/im + reciprocal hoist +
   FMA kernel under a relaxed relative-error contract; re-verify rms gate).
4. **Rotation** (Unit 3) only if/when p ≥ 26 becomes the working point.
5. **Gate run** after stages 1–3: full N=10⁶ × 10³ energy gate
   (~2.7 h estimate) closes the last open ROADMAP line.

## Verification matrix (run per unit)

| Unit | Exactness test | Perf probe |
|---|---|---|
| 1 walk | `parallel_matches_serial_*` | `fmm-bench 1000000 8 1e-5 1000000 8` |
| 2 translate | bit-compat unit tests, `m2l_rot_matches_naive` | 4000-pair microbench |
| 3 rotation | `m2l_rot_matches_naive`, `swap_*` | 4000-pair microbench @ p=26–32 |
| 4 mac | `f_prepass_par_matches_serial` | profile `prepass/mac` line |
| 5 harmonic | Table-3 + parity + homogeneity @ 1e-12 | Θ-build µs/pair |
| 6 tree | `build_and_geometry`, `degenerate_points` | gate-loop tree time |
| 7 moments | `compute_moments_par_matches_serial` | profile `moments` line |
| 8/9 p2p/bodies | `avx512_matches_scalar` (1e-12) | `fmm-bench … p2p` mode |
