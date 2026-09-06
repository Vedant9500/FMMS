# FMMS — Fast Multipole Method N-Body Simulator

Research documentation and implementation roadmap for a performance-obsessed
N-body simulator using the Fast Multipole Method (FMM).

**Status: Phase 1 implementation active** — serial Rust FMM with verified
O(N) scaling, rotation-accelerated M2L, and a production direct kernel. See
`ROADMAP.md` for gates and `todo.md` for the live session log.

## Goal

Build the fastest practical N-body simulator for the target hardware, chasing
every measurable improvement: algorithm, numerics, SIMD, cache, parallelism,
precision, and energy. Every optimization must be backed by measurement and,
where possible, by peer-reviewed literature.

## The core idea

| Method | Complexity | Verdict |
|---|---|---|
| Direct summation | O(N²) | baseline; wins only for small N |
| Barnes–Hut treecode | O(N log N) | good low-accuracy solver, simpler than FMM |
| **Fast Multipole Method** | **O(N)** | chosen algorithm; wins from N ≳ 10⁴ at high accuracy |

FMM beats the treecode decisively at high accuracy (relative force error
≲ 10⁻⁵) and large N. The two are complementary — a hybrid treecode/FMM with a
dynamic multipole-acceptance criterion (falcON-style, Dehnen 2000/2002/2014)
is the strongest known serial formulation.

## Documentation index

| Doc | Contents |
|---|---|
| `docs/01-algorithms.md` | FMM theory, variants, operator costs, which to choose |
| `docs/02-lowlevel.md` | SIMD, data layout, cache, fast math, precision |
| `docs/03-gpu.md` | CUDA design, M2L optimization, mixed precision |
| `docs/04-language.md` | Language/toolchain analysis (Rust vs C++ vs Fortran) |
| `docs/05-energy.md` | FLOP/watt, power capping, DVFS, RAPL |
| `docs/06-integrators.md` | Time integration choices |
| `docs/07-parallel.md` | Threading, MPI, task-based, load balancing |
| `docs/08-references.md` | Annotated bibliography with URLs |
| `docs/09-visual-system.md` | Live 3D visual system (Universe-Sandbox-style): GPU-driven rendering, precision, LOD, planets, trails, stack |
| `docs/10-gpu-fmm.md` | GPU-compute FMM stages (M2L kernels, replay, sweeps, engine integration) |
| `docs/11-visual-roadmap.md` | Visual track plan: stars done/future, gas & dust next, later phases |

## Roadmap

See `ROADMAP.md`. Phase-1 gates and session log: `todo.md`. For AI coding
agents, see `AGENTS.md`. High-level crate overview: `PROJECT.md`.

## Design principles (from research)

1. **Algorithm dominates.** Correct M2L strategy and tree/walk design matter
   more than any instruction-level trick. M2L is 90%+ of far-field cost.
2. **Measure before micro-optimizing.** Use the Roofline model to classify each
   kernel as compute- or memory-bound, then optimize the right thing.
3. **Precision is a budget, not a constant.** FP32 where accuracy allows,
   FP64 where it doesn't, double-single emulation in between. On GPUs this is
   a 3–4× lever.
4. **Energy follows throughput.** GPU offload and power capping give 15×–30%
   energy-efficiency gains; instruction tricks give fractions of a percent.
5. **Bit-shift fast-inverse-sqrt is a historical curiosity.** Modern hardware
   `rsqrt` + one Newton–Raphson step is faster and accurate to 24 bits.
6. **The visual system is live, not a movie.** Body state stays on the GPU;
   compute updates it, the renderer draws the same buffers with no per-frame
   CPU uploads. This is the whole game for millions of bodies at 60 fps.
7. **Don't break the GPU with universe scale.** CPU keeps double-precision
   state; subtract the camera every frame and hand the GPU small floats.
   f64 in shaders is the wrong answer.
