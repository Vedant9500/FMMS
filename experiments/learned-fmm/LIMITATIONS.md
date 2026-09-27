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
  not more same-level capacity.
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
- **Manifest cannot yet rebuild graphs:** `manifest.csv` carries
  (seed, config, n) but graph rebuilds also need (p, eps, n_leaf), which
  change tree structure and list density. Before the Phase-2 corpus run,
  add those columns (or pin defaults here). Rebuild-from-seed is the
  strategy (avoids ~84GB of stored lists); the manifest gap would silently
  break it.
- **Static equal-mass snapshots only:** no velocities, no mass spectra.
  Phase-3 energy-rollout eval needs a velocity sampler — the virialized
  Maxwellian currently lives inside `fmm`'s energy-gate test, not in a
  reusable library (same limitation as root `LIMITATIONS.md`). Promote or
  reimplement when Phase 3 starts. Log-mass features deferred with spectra.
- **N capped at 4k by O(N²) split cost:** 37ms/snapshot at N=4096; ~16× per
  4× N. Larger-N corpora are possible (~0.6s at N=16k) but corpus time and
  file sizes grow quadratically — revisit if small-N models saturate.

## Training / eval (open calls)

- **Backend undecided (`burn` vs `candle`):** `burn`+ndarray is the default
  (pure Rust, CPU, matches the small-model profile); operator traits in
  `src/lib.rs` are backend-agnostic precisely so this stays reversible.
  Decide at Phase 2 start; `candle` is fallback for safetensors interop.
- **List-eps is a free hyperparameter:** 1e-5 → 1e-3 densifies lists
  1.7–2.7× monotonically. Default stays 1e-5 (solver fidelity); sweep it in
  Phase 2 and record per snapshot (see manifest gap above).
- **Full 5k–20k corpus not yet generated:** per-snapshot cost measured
  (~37ms split + ~26ms graph at N=4k ⇒ ~12 min for 20k), total ~4GB dumps.
  Pilot 1k snapshots before committing to the full run.
