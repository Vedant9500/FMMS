# Learned FMM — tree-structured far-field surrogate (experiment)

Research spike, not part of the Phase-1 gates. FMMS stays pristine: this crate
depends on `fmm` as a library only, never patches it. The final head-to-head
(Phase 4) runs against untouched FMMS as reference.

## Core idea (adapted to this repo — no Python/PyTorch)

Real FMM has four moves: P2M, M2M, M2L, L2P. This experiment copies that shape
but replaces every analytic operator with a learned one:

- Each octree node gets a learned embedding instead of multipole coefficients.
- Upward: parent embedding = MLP over children embeddings + relative geometry.
- Sideways (learned M2L): node sums learned messages from its interaction
  list (the well-separated same-level cells, same MAC definition FMMS uses).
- Downward: child inherits MLP over [parent local, own upward embedding,
  interaction sum]. Leaves decode to per-particle far-field accelerations.

Design decision that saves pain: keep the near field analytic. Exact P2P
handles nearby particles; the network predicts only the far-field
contribution. Far field is smooth and low-frequency — exactly what multipoles
approximate well. Hybrid solver stays honest: exact where it matters, learned
where it is smooth.

## Why Rust + burn instead of Python + PyTorch

The original sketch proposed Python (config generator, numpy direct kernel,
Python octree) + PyTorch. For this repo that duplicates three things FMMS
already owns and validates:

| Need | Python sketch | Rust-native (this experiment) |
|---|---|---|
| Configs | Plummer/cube/clumps in numpy | `fmm::models` (Plummer, Hernquist, `Rng`) + new samplers here for cube/clumps/mixed |
| Labels | numpy direct kernel | `fmm::direct::direct` (validated O(N²) reference, mutual/antisymmetric) |
| Tree + near/far split | new Python octree + adjacency rule | `fmm::tree::Octree` + leaf-level θ<1 rule (Phase-0 approximation of `mac::accept`; see lessons) |
| Training | PyTorch | `burn` with `ndarray` backend (pure Rust, CPU, <1h for this size) |
| Metrics | custom | `fmm::direct::scaled_error` + `tests/energy_gate.rs` KDK pattern |

Consequences:

- Zero octree duplication, zero label mismatch with the reference solver.
- Deterministic data via the repo's `Rng` (SplitMix64), not numpy seeds.
- No second toolchain: inside `experiments/learned-fmm`,
  `cargo run --release --bin dataset` builds data (Phase 2 will add `train`).
- Phase 4 is a function call (`Fmm::new` / `run`), not a file-format bridge.

`burn` over `candle`: pure-Rust training loop + ndarray CPU backend matches
the "small, trains on CPU" compute profile; `candle` is kept as fallback if
we need safetensors interop with a Python run. Decision is reversible at
Phase 2 — the operator traits in `src/lib.rs` are backend-agnostic.

## Data (Phase 0 — built, see lessons for what changed)

- Configs: Plummer spheres, uniform cubes, clustered clumps, mixed.
  Reuse `fmm::models::{plummer, hernquist, Rng}`; cube/clump/mixed samplers
  live here (prime-index assignment — never fract/mod grids that collide
  particles and produce 1/r = inf).
- Labels: exact Newtonian `direct` per snapshot, no softening (deliberate:
  softening would bias labels against the FMMS reference; the price is
  outlier dynamic range, paid in the loss instead — see lessons).
- Split: leaf-level θ<1 — a pair is near iff same leaf or `r ≤ ρ_A+ρ_B`.
  This is the necessary half of `mac::accept` (the θ gate) without the
  error-estimate term; Phase 1 upgrades interaction lists to full
  `mac::accept` using `moments` + `f_prepass`.
- Scale: 5k–20k snapshots at N=1k–4k. Measured ~37ms/snapshot at N=4096
  (release) and ~52KB (N=1024) / ~213KB (N=4096) per file, so the full
  20k-snapshot corpus is ~12 min and ~4GB. Target confirmed feasible.
- Normalization (non-negotiable): G=1 units (repo default); positions
  centered and max-side-scaled to [-1,1]³ per snapshot (aspect preserved).
  Masses need no log-scaling in v1 — all configs are equal-mass with total
  M=1. Log-mass features enter only if variable-mass spectra are added later.
- Augmentation: random SO(3) rotation per snapshot (Shoemake quaternion,
  deterministic per seed).
- Format: versioned little-endian binaries (`LFMMD001`, f32 arrays) +
  `manifest.csv`. Parquet deferred — nothing in v1 needs columnar queries.

## Model (Phase 2, `burn`)

- Node embedding dim 32–64.
- Leaf: MLP over particle features relative to cell center
  (count, center of mass, total mass, spread).
- Upward (learned M2M): MLP over concatenated child embeddings + geometry.
- Interaction (learned M2L): message = MLP over
  [emb_u, emb_v, relative position, size ratio], summed over interaction list.
- Downward (learned L2L): child gets MLP over
  [parent local, own upward embedding, interaction sum].
- Decode (learned L2P): per-particle MLP over
  [particle features, leaf local embedding] → far-field acceleration.
- Loss: relative L2 (primary) + log-cosh (co-primary, not a fallback).
  Phase 0 showed this is mandatory, not optional — see lessons.

## Eval harness (Phase 3 — the part that makes it real)

One plot, error vs time per snapshot, three baselines + one ablation:

1. brute-force `direct`,
2. FMMS `Fmm` at p=3,5,8,
3. learned model,
4. flat MLP with no tree structure ("did the tree actually matter").

Metrics reuse `direct::scaled_error` (max/rms, scaled + relative).

Downstream test ("is it any good"): KDK rollout reusing the
`tests/energy_gate.rs` warm-start pattern (virialized Plummer, never cold
start — cold collapse forms an unresolvable hard binary). Static per-snapshot
error is a lab metric; energy drift over ~1000 steps is the truth. Compare
learned rollout vs FMM rollout drift.

## What will probably go wrong

- Network memorizes the near field despite the split → hybrid design is
  load-bearing; verify by ablating near/far (far-only target must win).
- Interaction lists explode at coarse levels → cap or sample per level.
- Single-regime overfit → train on mixed N and mixed config types from day one.
- Downward pass carries most expressive power (same as L2P in real FMM) →
  allocate capacity there first.

## Compute

Small. MLP(32–64) on N=1k–4k snapshots trains on CPU in well under an hour.
The expensive part is engineering time, not electricity.

## What Phase 0 taught us (updates to the original idea)

The sketch said "log-cosh if outliers misbehave" and "same MAC definition".
Measured data forced five revisions:

1. **Outliers are the norm, not an if.** Plummer/mixed snapshots span
   rms_total ~0.5–2000 from close pairs (min_r ~1e-4, exact 1/r²). A few
   close-pair particles dominate raw MSE across the whole corpus. So:
   per-snapshot target standardization is mandatory, relative L2 is the
   primary loss, log-cosh is co-primary from day one. This mirrors why
   FMMS's own metric is scaled (`direct::scaled_error` δa/f) rather than
   absolute — the physics demands scale-free errors.
2. **The split is θ<1, not full MAC.** `mac::accept` needs moments +
   fmin; Phase 0 has neither, so near = θ-failures at leaf level. It is the
   right v0 (it bounds exactly the pairs the walk must P2P), but the learned
   far target slightly over-covers what analytic M2L would carry. Phase 1
   closes this with full-`accept` interaction lists.
3. **No mass normalization problem exists in v1.** Equal-mass, M=1 configs
   have nothing to log-scale. The sketch's "masses log-scaled" applies only
   to future variable-mass spectra — recorded, not built.
4. **Mixed-regime training is confirmed necessary, with numbers.**
   far_frac varies 0.35–0.78 by config type and rms_total varies 3 orders of
   magnitude. A single-regime model would die on everything else, exactly as
   the sketch warned — now with a measured magnitude.
5. **No softening, by decision.** Exact labels keep the Phase-4 comparison
   against FMMS honest. Do not "fix" the dynamic range by softening the
   data; fix it in the loss (lesson 1).

## Gap verdicts (measured against the walk's own pair decomposition)

`graph --walk` runs FMMS's `record_pairs_par` — the walk's exact M2L/P2P
sets — against our lists (p=8, grid: 4 configs × N 256/1024/4096 × eps
1e-5/1e-3). `covered == same_level` in all 24 runs: our lists are an exact
superset of the walk's same-level pairs.

**G1 — same-level keeps the rule, loses ~half the walk's pairs.**
Cross-level share by count: cube ~0–14% (uniform/shallow trees) vs
plummer/clumps/mixed 40–82%; by source mass 0.36–0.87, worst in
concentrated configs. The "≥80% same-level" guess was FALSE. Mitigating
facts: our directed lists carry 2–3× the walk's pair volume (one-sided
accepts the walk would split are valid learned paths), and every listed
edge is a true accept. Decision: keep same-level for v1. Consequences:
(a) cross-level edges are the prime v2 extension; (b) cross-scale flow must
travel multi-hop (same-level + up/down passes), so v1 needs ≥2
message-passing steps or it accepts the gap; (c) concentrated configs
suffer most — watch them in eval.

**G2 — sparsity is honest, not a blocker.** Thin lists (mean_in 1–9)
coincide with small far_frac (0.07–0.32); dense lists (200–450) with large
far_frac (0.46–0.88). Where there is little far field to route, lists are
thin and the analytic near field dominates. Corollary: far_frac grows with
N (plummer 0.20 → 0.58 → 0.87 at 256 → 1024 → 4096) — train mixed-N or
underweight the far field. The eps knob works: 1e-5 → 1e-3 densifies our
lists 1.7–2.7× monotonically, so list-eps stays a free hyperparameter
(recorded per snapshot), defaulting to 1e-5 for solver fidelity.

**G3 — no format change.** Phase 2 rebuilds graphs on the fly from manifest
seeds: deterministic pipeline (`graph_deterministic` pins it), 26ms/snapshot
≪ a training step. Storing lists would cost ~4MB/snapshot at N=4k (~84GB
for 20k snapshots) — avoided entirely.

## Phases

- Phase 0: `dataset` bin — config generator + `direct` labels + near/far
  split + dumps. No ML dependency.
- Phase 1: tree/graph extractor reusing `Octree` — per-node features,
  children, interaction lists via full `mac::accept` (`graph` bin). No new
  octree. Lists are parameterized by (p, eps); edge volume matches analytic
  M2L scale (~100 edges/particle at N=4k, eps=1e-5).
- Phase 2: `burn` model implementing the operator traits in `src/lib.rs`.
- Phase 3: eval bin — three-way plot + flat-MLP ablation + KDK energy drift.
- Phase 4 (only if v1 works): head-to-head against FMMS (`Fmm::new`, p=3,5,8)
  on held-out configs. FMMS is called, never modified.

## Layout

```
experiments/learned-fmm/
  README.md        this file (adapted proposal + measured revisions)
  LIMITATIONS.md   living list of known limits (multi-hop risk, manifest gap, open calls)
  Cargo.toml       detached crate ([workspace]), path-dep on fmm only in v0
  src/lib.rs       configs + split + snapshot I/O + graph extractor (+ operator traits for Phase 2)
  src/bin/dataset.rs  generator (configs + labels + split stats + dumps)
  src/bin/graph.rs    graph stats (nodes, edges per level for one config/N)
```

## Phase-0 file format

Per-snapshot little-endian binary, magic `LFMMD001`, u32 n, u8 config tag,
then f32 arrays: pos(3n), mass(n), a_total(3n), a_near(3n), a_far(3n).
`manifest.csv` carries id, config, n, seed, p, eps, n_leaf,
rms_total/near/far, far_frac, min_r, filename — the (p, eps, n_leaf) triple
is what graph rebuilds need (LIMITATIONS gap, now closed). ~52KB/snapshot
at N=1024; ~37ms/snapshot at N=4096 (release), so 20k snapshots ≈ 12 min.

## Commands

```sh
cargo run --release --bin dataset -- --out /tmp/opencode/learned-fmm --snapshots 100 --n-min 1024 --n-max 4096 --config all
cargo run --release --bin graph -- --config mixed --n 4096 --p 8 --eps 1e-5
cargo run --release --bin graph -- --n 1024 --eps 1e-5 --walk   # topology vs walk + far_frac joint row
cargo test --release
```

## Status

Phase 0 done: 4-config generator (deterministic `Rng`, unit-box
normalization, SO(3) augmentation), leaf-adjacency near/far split,
versioned binary dumps + manifest. 7 lib tests green, incl.
`split_sums_to_total` (proven non-vacuous by mutation). Idea revised per
lessons above (loss, split scope, masses, no-softening decision).

Phase 1 done: `extract_nodes` (level, center, rho, mass, count, children,
leaf features) + `build_graph` (directed same-level lists via full
`mac::accept` with real moments/`f_prepass` pipeline) + `walk_coverage`
(cross-check vs the walk's recorded pairs) + `graph` bin.
12 lib tests green, incl. two-body positive control, non-vacuity guard, and
walk-superset pin (all proven against mutants). Gap verdicts above:
same-level kept for v1 with multi-hop consequence recorded, sparsity
cleared as honest, graphs rebuilt from seeds (no format change).

Backend decided: `burn` 0.21 (`ndarray` + `autodiff`, CPU). Spike (`flat`
bin, per-particle [pos, mass] → far field, one N=256 snapshot, standardized
targets): 64-wide plateaus at rel_mse ~0.02 (capacity floor, not breakage —
loss fell 50×), 256-wide memorizes to ~5e-4 in 5000 steps (~seconds on
CPU). Training loop, autodiff, and Adam all work; an Adam loss spike at
late training says lr 1e-3 runs hot — schedule or clip in real training.
