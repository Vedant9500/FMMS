# LIMITATIONS.md — learned-FMM experiment: known limits & deferred calls

Living list of things measured-but-unresolved, scoped-out, or assumed.
Companion to `README.md` (idea + gap verdicts). Items here are intentionally
NOT silently "fixed" — each needs a re-measure / ablation before changing.

## Topology / model expressivity

- **Multi-hop sufficiency is unproven (the one open structural risk):**
  same-level lists force cross-scale flow through ≥2 message-passing steps,
  yet cross-level pairs carry ~40–60% of walk M2L source mass at N=4k
  (up to 0.87 in concentrated configs at N=1k). Whether learned multi-hop
  routing recovers that flow is unknown until Phase 2 ablates 1 vs 2 vs 3
  hops on plummer/mixed @ N≥1k, eps=1e-5, measuring far-field relative L2.
  If the gap persists, the fix is cross-level edges (prime v2 extension),
  not more same-level capacity. Note the backward paths are as long as the
  forward ones: the small-config learning test (emb=8) crawled (30% drop in
  200 steps) while emb=32 learns steadily — depth attenuates gradients too,
  so hop ablations must also watch trainability, not just final error.
- **Same-level restriction is v1 scope, not fidelity:** our lists are an
  exact superset of the walk's same-level pairs (`covered == same_level` in
  all 24 grid runs) but structurally exclude cross-level pairs the Dehnen
  dual walk M2Ls. Kept deliberately (matches the sketch's interaction-list
  abstraction, 2–3× the walk's pair volume); revisit via the ablation above.
- **Sparse lists in concentrated configs at small N:** mixed @ N≤1k gives
  mean_in 1–9 (deep-narrow trees). Cleared as honest (far_frac there is
  0.07–0.08 — little to route), but small-N training under-exercises the
  learned M2L in exactly those regimes. Mitigation: train mixed-N up to 4k
  (dense there, mean_in 200–450) and watch concentrated configs in eval.

## Data / targets

- **Outlier dynamic range constrains the loss:** exact 1/r² labels give
  rms_total ~0.5–2000 across snapshots (min_r ~1e-4). Phase 2 must use
  per-snapshot standardization + relative L2 / log-cosh — raw MSE would let
  a few close-pair snapshots own the corpus. Do not "fix" by softening the
  data (deliberate no-softening call, README lessons §5).
- **f32 dump precision unmeasured against training:** labels stored f32
  (~1e-7 quantization) to keep the corpus small. Presumed below the model
  error floor, but if training plateaus near 1e-6 relative, re-check this
  before blaming the architecture.
- **Manifest gap CLOSED:** `manifest.csv` now carries (p, eps, n_leaf)
  alongside (seed, config, n) — graph rebuilds are fully specified.
- **Static equal-mass snapshots only:** no velocities, no mass spectra.
  Phase-3 energy-rollout eval needs a velocity sampler — the virialized
  Maxwellian currently lives inside `fmm`'s energy-gate test, not in a
  reusable library (same limitation as root `LIMITATIONS.md`). Promote or
  reimplement when Phase 3 starts. Log-mass features deferred with spectra.
- **N capped at 4k by O(N²) split cost:** 37ms/snapshot at N=4096; ~16× per
  4× N. Larger-N corpora are possible (~0.6s at N=16k) but corpus time and
  file sizes grow quadratically — revisit if small-N models saturate.

## Training / eval (open calls)

- **Backend DECIDED (`burn` 0.21, spike green):** `ndarray` + `autodiff`
  on CPU; 2000 full-batch steps on N=256 run in <1s. Memorization check:
  64-wide flat MLP plateaus at rel_mse ~0.02, 256-wide reaches ~5e-4 —
  plumbing validated, and the capacity floor hints the tree matters.
  Watch late-training Adam spikes (lr 1e-3 runs hot).
- **List-eps is a free hyperparameter:** 1e-5 → 1e-3 densifies lists
  1.7–2.7× monotonically. Default stays 1e-5 (solver fidelity); sweep it in
  Phase 2 and record per snapshot (see manifest gap above).
- **Full 5k–20k corpus not yet generated:** per-snapshot cost measured
  (~37ms split + ~26ms graph at N=4k ⇒ ~12 min for 20k), total ~4GB dumps.
  Pilot 1k snapshots before committing to the full run.
- **Weight init is unseeded:** the data pipeline is deterministic
  (rebuild-from-seed verified to 2.5e-8), but burn Linear init uses an
  unseeded global RNG — training runs are not bitwise reproducible.
  Pin a seed before publishing any comparison numbers.
- **No train/test split yet:** `train` fits every manifest row. Held-out
  evaluation (and the FMMS + flat-MLP comparison) belongs to the eval
  harness — do not quote pilot losses as generalization numbers.
