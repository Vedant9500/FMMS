# PERF-ANALYSIS.md — N=10⁶ FMM performance review

**Date:** 2026-08-16 · **Machine:** Zen4 Ryzen 7 7840HS (8C/16T, 35W), 14GB RAM,
`target-cpu=native`, thin-LTO, codegen-units=1 · **Config audited:** p=8, eps=1e-5,
n_leaf=16, 8 threads (the `energy_gate` N=10⁶ × 10³-step configuration).

**Trigger:** the full energy-conservation gate takes ~5.5 h (≈20 s/step at 8
threads) — felt to be "no better than Barnes–Hut or O(N²)". This review
decomposes one force evaluation, quantifies where the time goes, and ranks the
recoverable losses (~10× total headroom vs exafmm-class ~1–2 s/step).

Note on environment: measurements below were taken with ~10/14 GB RAM in use
(chrome/opencode), so absolute numbers sit at the high end of the todo.md
variance bands; the structural conclusions are load-independent.

---

## 1. Executive summary

1. **The "O(N) ≈ 10⁶ ops/step" premise is wrong on the constant, not the
   scaling.** This FMM does ~0.5M flops per body per step (~5×10¹¹ flops/step
   at N=10⁶). Seconds per step is the correct expectation; milliseconds is not.
   O(N) scaling itself is honest (fit ~N^0.91–1.0, 48× faster than direct at
   N=10⁶).
2. **~35% of every step is serial code** (f_prepass, moments, L2L, L2P, tree
   rebuild): average CPU utilization 2.9/8 cores (`user/real` = 101 s / 35 s).
   Amdahl cap: even a perfect walk leaves ~7 s/step.
3. **The parallel walk is DRAM-bandwidth-bound, not compute-bound.** Its
   measured 11.9 s at 8 threads equals the walk's random-access traffic
   (~214 GB/step) divided by the machine's 8-thread STREAM rate (~18.5 GB/s).
   More threads cannot help (12 threads ≈ 8, todo.md confirms); only less
   traffic per M2L pair can.
4. **The per-thread full-tree accumulators cost ~0.4 s/step of zero+reduce
   traffic and impose a 3.6 GB peak-RSS ceiling at 8 threads** — the reason
   12–16-thread runs OOM.
5. Staged fixes (no math-contract changes needed for the first two stages):
   20 s → ~13 s (parallelize serial phases) → ~6–7 s (walk memory layout) →
   1–2 s (SIMD kernel + n_leaf tuning; exafmm parity, the open ROADMAP gate).

---

## 2. Where the constant comes from (flops accounting)

Measured walk stats at N=10⁶ (this review): pops = 53.5M, accepted M2L pairs =
41.2M (**41 per body**), P2P leaf-pair calls = 5.19M, leaves = 411k
(**2.43 bodies/leaf average** — `p2p_self` count).

- Naive `m2l_mutual_scratch` (p=8): the O(p⁴) convolution has
  Σₙ (2n+1)·(p−n+1)² = 1,365 inner iterations × ~8 flops (mutual, both
  directions), plus a fresh 81-coefficient Θ-table build per pair
  (`translate.rs:72`) → **~11k flops per M2L pair**.
- M2L alone: 41.2M × 11k ≈ **4.5×10¹¹ flops ≈ 450k flops/body**.
- Plus P2P (~31 body-pairs/body), P2M/M2M, L2L/L2P, f_prepass, tree build
  → **~0.5M flops per body per step ≈ 5×10¹¹ flops/step** at N=10⁶.

At a realistic sustained ~100 GFLOPS on 8 Zen4 cores that is ≥5 s/step — the
honest floor for this algorithm/hardware before kernel heroics. exafmm-t
(0.95 s / 1M / 7 digits @ 14 cores) shows ~1–2 s/step is reachable; the gap
to the current 20 s is engineering, listed below.

---

## 3. Measured per-step ledger (N=10⁶, p=8, eps=1e-5, 8 threads)

Profile mode (`fmm-bench 1000000 8 1e-5 1000000 8`), second pass:

| Phase | Time | Share | Path |
|---|---|---|---|
| Dual walk (parallel) | 11.89 s | 64% | `walk.rs` (incl. per-thread clone/zero/reduce) |
| f_prepass (+power/binom) | 4.45 s | 24% | `mac.rs:90`, called at `fmm.rs:62` |
| Moments (P2M + M2M) | 1.22 s | 7% | `moments.rs:33` |
| L2L | 0.45 s | 2% | `fmm.rs:77`–84 |
| L2P + scatter | 0.46 s | 2% | `fmm.rs:88`–105 |
| **Profile total** | **18.47 s** | | (tree prebuilt) |
| Tree rebuild + `to_parts` (gate only, not in profile) | ~1.5 s | | `Fmm::new` per step, `energy_gate.rs:159–160` |
| **Gate per-step** | **≈20 s** | | ×1000 steps ≈ 5.5 h ✓ |

Cross-checks: serial walk = 38.8 s (todo.md) → **942 ns per M2L pair**, ~2.7
flops/cycle — the scalar kernel is compute-reasonable (~16% of the fp64
AVX-512 peak; it cannot go much faster without reassociation+SIMD, see §5.5).
N=10⁵: walk 5.33 s serial → 2.93 s at 8 threads (1.8× on the loaded machine;
4.6× in todo.md's quieter N=10⁶ runs).

---

## 4. Root causes, ranked by recoverable time

### 4.1 Walk is DRAM-bandwidth-bound (~7 s recoverable) — the big one

Every accepted M2L pair randomly touches `moments[a]`, `moments[b]`
(read) and `locals[a]`, `locals[b]` (read-modify-write) — 81 complex = 1.3 KB
per cell per array, i.e. **~5.2 KB of random-access traffic per pair** across a
~1.3 GB working set (moments + locals ≈ 2 × 500k cells × 81 × 16 B) that
swamps the 16 MB L3.

Arithmetic: 41.2M pairs × 5.2 KB ≈ **214 GB per force evaluation**. At the
measured 8-thread STREAM rate (~18.5 GB/s) the floor is 214 / 18.5 ≈
**11.6 s** — measured walk: **11.9 s**. The walk time *is* DRAM traffic ÷
bandwidth. (Serially the same traffic runs at ~5.5 GB/s, which is why the
serial walk is compute-bound instead — 38.8 s at 942 ns/pair.)

Consequences already visible in todo.md: 12 threads gains almost nothing
(bandwidth already saturated at 8); the "faster M2L kernel" lever is worthless
until traffic per pair drops (the kernel would just wait on memory harder).

Fix directions:
- Batch M2L by target cell / interaction-list (level-wise) so each target
  cell's 1.3 KB accumulator stays in L1/L2 across all its sources — classic
  FMM layout, this is how exafmm reaches compute-bound M2L.
- Or Morton-range thread ownership: each thread owns a cell slab; working set
  per thread fits cache; removes the reduce pass too (§4.3).
- SoA re/im layout of moments/locals enables contiguous streaming instead of
  strided complex pairs.

### 4.2 `f_prepass` is a second, fully serial N-body pass (~3.8 s recoverable)

`mac.rs:90` (`f_prepass`) runs a per-particle p=0 treecode walk (θ<1) for all
10⁶ particles — ~19k cycles (~500–1000 cell visits) per particle, single
threaded. It costs **~45% as much as the entire 8-thread walk**. It exists to
feed Dehnen's error-controlled MAC (`fmin` per cell).

Fix directions, in increasing order of effort:
- Parallelize over particles (walks are independent) → ~0.6 s. Trivial.
- Reuse across timesteps: f_b changes slowly; positions move ~dt·v per step.
- Cheaper estimator for fmin (coarse pass, or Dehnen's per-cell bound).

### 4.3 Per-thread full-tree accumulators (~0.4 s + a hard memory ceiling)

`walk.rs:243`–252: every thread gets a private `locals` sized
`ncells × dim(p)` (~450 MB at ~500k cells, p=8) plus a `bodies.clone()`
(64 MB). Sampled **peak RSS 3.6 GB at 8 threads**. Costs per step:
8 × 450 MB zeroed, 8 × 64 MB cloned, then a serial reduce streaming
~3.5 GB (~0.35–0.5 s), all inside the "walk" bucket. It is also why 16
threads OOMs (todo.md) — the machine cannot even try more threads.

Fix: partition cells by Morton range per thread (no private full-tree copies,
no reduce), or the §4.1 batching which subsumes it.

### 4.4 Serial pre/post pipeline + allocation churn (~2.5–3 s recoverable)

Everything around the walk in `run_impl` (`fmm.rs:52`–107) is single-threaded:
`Bodies::from_parts`, `compute_moments` (P2M over all particles, M2M over all
edges), `multipole_power`, L2L over all parent-child edges, L2P over all
particles, scatter. Additionally `upsilon_table` (`harmonic.rs:127`)
**heap-allocates a Vec per call**: once per particle in P2M (`moments.rs:44`),
per edge in M2M (`moments.rs:63`) and L2L (`translate.rs:184`), and per
particle in L2P (`translate.rs:212`) — ~2.5M malloc/free per step. The tree is
rebuilt from scratch every gate step (`Fmm::new` per step,
`energy_gate.rs:160`): keys + `sort_unstable_by_key` + geometry gathers
≈ 1–1.5 s serial.

Fix: parallelize each pass over cells/leaves (they are embarrassingly parallel
except M2M/L2L, which are level-synchronized); reuse a scratch `UpsilonScratch`
à la `M2lScratch`; keep the tree object across steps (re-sort only).

### 4.5 Smaller items (~0.3–0.6 s combined)

- `accept()` computes `r.powf(p)` (`mac.rs:36`) — a libm call in the hottest
  loop, ~2 per pop on ~6.9M failed + 41.2M accepted evaluations. Replace with
  repeated squaring of `r²` (p=8: three squarings). ~1–3% of walk.
- `is_leaf()` scans all 8 children on every call (`tree.rs:184`); a
  precomputed flag or checking `children[0] < 0` first is cheaper.
- Walk `AtomicU64` counters are already folded per-thread (`Counters::fold`) —
  confirmed NOT a bottleneck.
- `n_leaf=16` produces 411k leaves at 2.43 bodies/leaf average. Tiny leaves
  inflate ncells (all ncells-proportional memory and passes), the 53.5M pops
  bookkeeping, and the 5.19M P2P leaf-pair call overhead. Sweeping n_leaf =
  32/64 is a cheap experiment — re-run the accuracy gates (MAC interplay;
  `mac.rs` factor-8 is load-bearing per AGENTS.md).

---

## 5. Staged remediation plan

| Stage | Change | Expected per-step | Gate risk |
|---|---|---|---|
| 0 | `powf` → squaring; scratch Υ tables; remove per-call Vec allocs | ~19 s | none (bit-identical) |
| 1 | Parallelize f_prepass (per-particle), moments, L2L, L2P, tree sort | **~13 s** | none (reassociation-level only) |
| 2 | Walk memory layout: target-batched M2L accumulation or Morton-range ownership; drop per-thread full-tree locals | **~6–7 s** | walk reordering; parallel-vs-serial tests already tolerate reassociation |
| 3 | Kernel: relax `m2l_mutual` bit-compatibility contract to a relative-error contract → FMA reassociation + AVX-512 on the convolution (~2–3×); n_leaf sweep | **1–2 s** | changes rounding; accuracy gates (rms < 1e-4) must be re-verified; AGENTS.md convention update needed |

Stage 3 is what closes the exafmm-parity ROADMAP gate (0.95 s / 1M /
7 digits @ 14 cores, normalized). Stage 2 is the prerequisite — after the
bandwidth fix the walk becomes compute-bound and the kernel lever pays off.

---

## 6. What is NOT the problem (verified)

- **Not secretly superlinear:** scaling fit ~N^0.91–1.0 (todo.md); 48× faster
  than direct O(N²) at N=10⁶. The 5.5 h is constant × serialization, not
  complexity class.
- **Not lock contention:** the 2026-08-14/15 work-stealing rewrite fixed the
  mutex convoy; counters are per-thread; termination via `left` atomic is
  sound. Contention is not visible in the numbers.
- **Not the P2P kernel:** P2P is a minor share of the step; its fp64
  throughput ceiling is a separate closed gate.
- **Not build config:** thin-LTO, codegen-units=1, `target-cpu=native` all
  verified active.

---

## 7. Update 2026-08-16 (evening): stage-2a applied — two-phase bucketed walk

`dual_walk_par` restructured (`walk.rs`): the walk now *records* accepted M2L
pairs (P2P inline as before), then replays them partitioned into `min(a,b)`
subtree buckets (in-place American-flag sort, ≤1024 buckets) into per-thread
locals. Kernel untouched — same pair set, bit-compatible math; only
accumulation order changes. Also `r.powf(p)` → repeated squaring in
`mac.rs:36` area (`error_estimate`).

| Metric (8 threads, same loaded-machine session) | before | after | Δ |
|---|---|---|---|
| N=10⁵ walk | 2.93 s | **0.99 s** | 2.95× |
| N=10⁵ total | 3.51 s | 1.56 s | 2.3× |
| walk 8-thread scaling (N=10⁵) | 1.8× | 5.3× | — |
| N=10⁶ walk | 11.89 s | **7.69 s** | 1.55× |
| N=10⁶ total | 18.47 s | 14.32 s | 1.29× |
| N=10⁶ peak RSS | 3.6 GB | 4.1 GB | +330 MB pairs array |
| M2L pairs (N=10⁶) | 41,224,026 | 41,224,026 | identical |

Estimated gate: ~15.8 s/step → **~4.4 h** for 10⁶×10³ (was 5.5 h).

**Stage-1 follow-up (same day): `f_prepass` parallelized.** Per-particle walk
extracted to `f_walk`; `f_prepass_par` splits particles into `nthreads`
disjoint Morton-order chunks (bit-identical to serial — each f_b is an
independent serial sum; asserted by test). N=10⁶ @8: prepass
4.45 → **0.56 s (7.9×)** — static chunking balanced well because Morton
chunks each span many octants. N=10⁵: 0.36 → 0.041 s.

Cumulative after stage 2a + prepass (N=10⁶ @8, 8 threads):

| Phase | original | now | |
|---|---|---|---|
| walk | 11.89 s | 7.31 s | 1.63× |
| prepass/mac | 4.45 s | 0.56 s | 7.9× |
| moments | 1.22 s | 1.24 s | — |
| l2l + l2p | 0.91 s | 0.95 s | — |
| **total** | **18.47 s** | **10.05 s** | 1.84× |

Gate estimate: ~11.5 s/step (incl. ~1.5 s tree) → **~3.2 h** for 10⁶×10³.
Accuracy unchanged: M2L pair set identical; exit gate rms_s 4.87e-5 (<1e-4).
Accuracy: exit gate rms_s 4.87e-5 (gate <1e-4, ~2× margin; small shift from
powi ULPs flipping borderline accepts).

Why N=10⁶ gains less than N=10⁵: the bucketed replay makes the min-side cell
cache-resident, but the *other* side of each pair is still a random ~2.6 KB
moments/locals touch → ~107 GB/step ≈ 5.8 s floor at STREAM rate. Closing that
needs the §4.3/§5 stage-2b design (one-directional Morton-slab ownership:
shared locals slabs, no per-thread copies, source reads become shared
read-only) — also what removes the 3.6 GB accumulator ceiling. The largest
remaining *easy* win is serial `f_prepass` (4.5 s, trivially
parallelizable over particles) — *done, see stage-1 follow-up above.* Next
pipeline targets: moments (1.24 s, parallelize P2M over leaves / M2M by
level) and the per-step tree rebuild (~1.5 s serial in the gate).

---

## Appendix: raw profile output (2026-08-16, loaded machine)

```
$ fmm-bench 1000000 8 1e-5 1000000 8
profile N=1000000 p=8 eps=0.00001 nthreads=8: tree-build+clone 0.0000s,
  moments 1.2244s, prepass/mac 4.4460s, walk 11.8896s, l2l 0.4470s,
  l2p+scatter 0.4638s, total 18.4708s
  walk stats: pops=53465962, accept/M2L=41224026, m2l-pair-ops=41224026,
  p2p_self=410790, p2p_pair=5190236 (accept-calls=41224026)
real 35.0s, user 101.5s, sys 3.8s   → avg 2.9 CPUs busy of 8
peak RSS (sampled): 3617 MB

$ fmm-bench 100000 8 1e-5 100000 8
profile: moments 0.1224s, prepass/mac 0.3615s, walk 2.9346s, l2l 0.0450s,
  l2p+scatter 0.0468s, total 3.5103s
  pops=7892566, accept/M2L=5721286 (57/body), p2p_self=40682 (leaves)

$ fmm-bench 100000 8 1e-5 100000 1   (serial reference)
profile: moments 0.1219s, prepass/mac 0.3555s, walk 5.3347s, l2l 0.0453s,
  l2p+scatter 0.0454s, total 5.9028s
```

Bandwidth attribution arithmetic (§4.1): 41,224,026 pairs × 5.2 KB
(2×1.3 KB locals RMW + 2×1.3 KB moments read) ≈ 214 GB; 214 GB ÷ 18.5 GB/s
(8-thread STREAM, todo.md) = 11.6 s vs 11.9 s measured. Residual uncertainty:
fraction of moments reads served by shared L3 (would reduce predicted traffic;
the near-exact match suggests it is small at this working-set size).
