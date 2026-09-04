# PROJECT.md — FMMS implementation overview

Workspace layout and what each crate/module does. Companion to `README.md`
(motivation) and `ROADMAP.md` (gates). For build/test commands see `AGENTS.md`.

## Workspace

```
crates/
├── fmm/       the library: FMM implementation (published crate `fmm`)
├── bench/     O(N) scaling sweep + phase profile binary (crate `fmm-bench`)
└── visual/    live GPU visual system V0 harness (bin `fmms-visual`, workspace member)
```

`crates/fmm/src/`:

| Module | Responsibility |
|---|---|
| `lib.rs` | crate root, module list |
| `models.rs` | `Particle`, `Plummer` generator, deterministic `Rng` |
| `tree.rs` | `Octree` build, cell storage, `is_leaf`, centers/rhos |
| `moments.rs` | P2M multipole computation, `multipole_power`, `binomials` |
| `translate.rs` | M2L (naive O(p⁴) `m2l_mutual`, `m2l_one_sided`), L2L, L2P |
| `rotation.rs` | rotation-accelerated M2L O(p³), Dehnen axis-swap route, `ROT_P` crossover |
| `harmonic.rs` | spherical-harmonic basis: `idx`, `dim`, `C` complex, `theta_table` |
| `mac.rs` | multipole-acceptance criterion, error estimate, f-prepass |
| `walk.rs` | dual tree walk (far field M2L + near field P2P), profiling counters |
| `p2p.rs` | P2P direct kernel (SoA, cache-blocked, AVX-512 path) |
| `bodies.rs` | SoA `Bodies` layout |
| `direct.rs` | naive O(N²) reference: `direct`, `scaled_error` |
| `roofline.rs` | FLOP/intensity accounting per kernel |
| `fmm.rs` | `Fmm` orchestrator: tree→moments→mac→walk→L2L→L2P |

## Key design decisions

- **Θ (theta) basis** for multipoles/locals (Dehnen Υ coefficients), real
  `theta_table` per pair reused across the M2L convolution.
- **Two M2L paths**, switched by `rotation::ROT_P` (currently 20):
  - naive `m2l_mutual` — O(p⁴), used for p < ROT_P;
  - `RotCtx::m2l_pair` — O(p³) axis-swap rotation (Dehnen 2014 A.6.1),
    precomputed swap matrices, wins from p ≥ 20.
- **`m2l_mutual`** operates both directions per pair (shares the Θ table),
  folds the Θ(-r) parity into a pre-scaled table, uses raw-pointer fused
  re/im accumulation.
- **MAC factor 8** (Dehnen eq. 15) in `mac.rs` is load-bearing: lowering it
  to 1.0 fails the rms error gate (rms_s 2.25e-4 vs 1e-4 required).
- Momentum conservation holds to netF/|F| ~ 1e-17 at all N.

## Measured status (2026-08-15)

- O(N) scaling verified N=10³…10⁶, fit ~N^0.91-1.0 (0.73s @ 1e4 → 51.7s @ 1e6).
- FMM beats direct N² at N ≳ 3e4 (eps=1e-5, p=8, single-core).
- Walk = ~95% of runtime, dominated by M2L (~62 pairs/body at eps=1e-5).
- Rotation M2L crossover at p ≈ 20 (re-measured 2026-08-21, dot_row matvecs;
  was 26); rel err ≤ ~5e-13 up to p=34.

## Verification

- 26 unit tests + 3 integration tests (`tests/`), all passing; plus an
  energy-conservation smoke test (N=10³ steps, ~9s) in `tests/energy_gate.rs`.
- `exit_gate` integration test (ignored by default; N=10⁴ direct O(N²)) run
  explicitly in release mode.
- `energy_gate` integration test (ignored by default; N=10⁶ × 10³ steps, ~5h)
  run explicitly in release; validated at N=10⁵×10³ as interim.
- Python cross-checks of the rotation algebra in `/tmp/opencode` (not part of
  the repo).

## Open gates

See `todo.md` and `LIMITATIONS.md`. Remaining: P2P 300+ GFLOPS (closed at fp64
hardware ceiling), exafmm parity, full N=10⁶ energy-conservation gate (machinery
done, run deferred), walk-counter/housekeeping cleanup.
