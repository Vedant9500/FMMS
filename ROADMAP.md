# ROADMAP — from research to implementation

Research phase complete. This is the implementation plan, ordered to maximize
measured performance per unit of engineering effort.

**Architecture decision (from research):**

- **Algorithm:** Dehnen-style dual-tree-walk FMM — solid-harmonic/Cartesian
  expansion, rotation-accelerated M2L (O(p³)), error-controlled MAC, mutual
  cell–cell interactions (momentum conserving). Fallback: direct N² mode for
  N < 10⁴, treecode mode for cheap low-accuracy steps.
- **Language:** Rust host (tree, walk, integrator, IO) + CUDA kernels in
  C/C++ via FFI. CPU-only path compiles the same kernels to AVX-512.
- **Precision:** FP64 state/integrator, FP32 inner kernels with FP64
  accumulation, rsqrt+NR reciprocal sqrt.
- **Parallel:** Rayon/OpenMP tasks (dual tree walk) → MPI for cluster.

---

## Phase 0 — Foundation (milestone: validated math core)

- [x] Set up repo: Rust workspace + CUDA sub-crate, benchmark harness
      (criterion), golden-test suite (Plummer/Hernquist models from NEMO,
      energy & momentum conservation, force-compare vs direct summation).
- [x] Implement octree build (Morton sort) + dual tree walk skeleton
      (interaction list generation only, no physics yet).
- [x] Unit-test expansion math: multipole moments, M2M shift, L2L shift,
      P2M/P2P against brute force for random small systems.
- [x] Implement error-controlled MAC (Dehnen 2014) with the two-parameter
      (p, ε) interface.
- [x] **Exit:** force-compare test passes at ε=10⁻⁵ for N=10⁴; Roofline model
      labels each kernel compute- vs memory-bound on target hardware.

## Phase 1 — Correct FMM (milestone: O(N) scaling proven)

- [x] Serial FMM with rotation-accelerated M2L O(p³) (Cheng/Dehnen route),
      precomputed rotation matrices for p ≤ 30 (our axis-swap Dehnen A.6.1
      variant supports p ≤ 60; beats naive M2L from p ≥ 26).
- [ ] P2P direct kernel, SoA layout, cache-blocked (BS=16), AVX-512
      intrinsics (8 bodies/iter), rsqrt+NR, FP64 accumulate. Target:
      300+ GFLOPS single socket.
- [ ] Expansion-center strategy: CoM for low-accuracy mode, sphere-center
      for high-accuracy (Dehnen 2014).
- [ ] Verify O(N) scaling (N=10⁴…10⁷); confirm FMM beats direct N² and
      BH-tree at the crossovers from research (N>10⁴ @ ε≲10⁻⁵).
- [ ] Benchmark vs reference: exafmm-t numbers (0.95 s / 1M / 7 digits
      @ 14 cores) as a fair target, normalized to our hardware.
- **Exit:** O(N) scaling curve measured; energy + momentum conservation
  validated for 10⁶ bodies, 10³ steps.

## Phase 2 — Integrator & production physics (milestone: real simulation)

- [ ] Block-timestep integrator: leapfrog/KDK (collisionless) + 4th-order
      Hermite with block steps (collisional); PRS23 timestep criterion
      (rounding-robust).
- [ ] Predictor/force/corrector flow where forces computed once per active
      block.
- [ ] Optional: close-encounter regularization (KS) for collisional mode.
- [ ] Energy benchmark: match Taichi-class behavior (energy error < 10⁻⁴ at
      core collapse; momentum to machine precision with mutual FMM).
- **Exit:** end-to-end Plummer-cluster evolution matches reference codes'
      Lagrangian radii / core collapse time within tolerance.

## Phase 3 — GPU (milestone: M2L compute-bound)

- [ ] CUDA port of P2P (shared-memory tiling, one thread per target).
- [ ] **M2L symmetric kernel** (Kohnke approach): precompute symmetry-class
      operators, coalesced shared-memory layout, aim compute-bound ~70% peak.
- [ ] Tree build on CPU, upload once; pinned memory + streams overlap.
- [ ] FP32 inner + FP64 accumulate; `-use_fast_math`; rsqrt.
- [ ] Direct-N² GPU fallback for N < 10⁴ (one block per target).
- [ ] Dynamic-parallelism M2L variant for high-p (p≳10) as an option.
- **Exit:** reproduce ~30–70× vs single CPU core; FMM beats GPU direct N²
  above N≈10⁴; mixed-precision accuracy audit passes (GENGA-style
  comparison).

## Phase 4 — Energy engineering (milestone: joules, not just GFLOPS)

- [ ] Add energy metrics (RAPL, nvidia-smi) to the benchmark harness;
      report GFLOPS + FLOP/watt.
- [ ] Per-GPU-model power-cap sweep at install time (55–70% TDP sweet spot);
      CPU capped whenever GPU path active.
- [ ] DVFS on memory-bound phases (VLI/up-down passes).
- [ ] Document per-config (p, ε, q, cap) energy Pareto frontier.
- **Exit:** measured ≥2× energy-efficiency improvement over uncapped default
  config, with <10% throughput cost, on target GPU.

## Phase 5 — Distributed (milestone: cluster scaling)

- [ ] Morton-range partition + MPI boundary exchange of cell multipoles;
      acceleration reduction only; per-rank position updates (no broadcast).
- [ ] Task-based dual tree walk inside a node (Rayon/OpenMP tasks).
- [ ] Zoltan-style load balancing for non-uniform distributions.
- [ ] Multi-GPU: NCCL boundary exchange; reproduce 44–66% parallel
      efficiency at 10⁶–10⁷ particles.
- **Exit:** weak scaling ≥60% at 16 nodes / 10⁶ particles; FMM scaling beats
  a PME-style reference at ≥8 nodes.

## Phase 6 — Polish & specialization (milestone: "every decimal")

- [ ] Loop unrolling, prefetching, alignment audit on the two hot kernels
      (P2P, M2L); measure each change in isolation.
- [ ] Explore plane-wave M2L (diagonal) for p>30 — only if profiling shows
      M2L still dominates.
- [ ] Optional bbFMM path for non-1/r kernels (Plummer softening, Yukawa) via
      the 16-operator symmetric variant.
- [ ] Auto-tuner: search (p, ε, q, thread/block counts, power cap) per
      hardware; store best config.
- [ ] Final: comprehensive benchmark report (GFLOPS, FLOP/watt, energy
      conservation, scaling) vs published references.

---

## Known risks / decisions to revisit

1. **Rust GPU ecosystem** is immature — if FFI friction dominates, consider
   C++ host. Decision gate at end of Phase 3.
2. **FMM vs direct crossover** hardware-dependent; keep both modes and an
   auto-dispatch.
3. **Plane-wave M2L** is complex; only pursue if rotation M2L becomes the
   bottleneck at target p.
4. **Close encounters** (collisional) need regularization; if out of scope,
   restrict to collisionless mode (KDK + FMM) which is already momentum-
   conserving.

---

# PART 2 — Live 3D Visual System

Design doc: `docs/09-visual-system.md`. End product is a live, interactive
3D system (Universe-Sandbox-like), NOT a precomputed replay. Pygame-style
playback is allowed only as a throwaway physics test harness.

**Architecture decision (from research-2):**
- GPU-driven pipeline: body state lives in GPU storage buffers; compute
  updates in place; renderer reads same buffers via indirect instancing.
  Zero per-frame CPU position uploads.
- Rust host (`winit`+`egui`) + **wgpu** renderer (Vulkan/DX12/Metal).
- FMM compute stays CUDA (research-1), bridged via
  `VK_EXT_external_memory_fd` + `cudaImportExternalMemory` + timeline
  semaphores (zero-copy). Fallback: wgpu compute shaders for the same kernels.
- Universe-scale precision: CPU dvec world state, per-frame camera rebase to
  f32 for GPU (camera-relative rendering). No f64 in shaders.
- Fixed-timestep physics + render-rate interpolation (smooth at any Hz).
- HDR offscreen target (`rgba16float`) + tone map + bloom; additive blending
  for star/dust fields; instanced billboards (or splats at high N).
- Planets: sphere impostors at distance → mesh/quadtree terrain near surface.
- Octree LOD reused from FMM physics for rendering huge populations.

## Phase V0 — Live render harness (milestone: see the sim move)

- [ ] `winit` + `wgpu` window, egui overlay, orbit camera.
- [ ] Upload body positions once; per-frame render instanced billboards,
      color by velocity/mass/temperature (HDR + tone map).
- [ ] Fixed-timestep accumulator + interpolation (smooth camera & bodies).
- [ ] Test harness ONLY: optional Pygame/Matplotlib playback path for physics
      validation (never the deliverable).
- [ ] Benchmark: reproduce the GPU-driven numbers (see doc §2) at 60 fps.
- **Exit:** 1M bodies @ 60 fps with camera rebase; verified no per-frame
  position upload.

## Phase V1 — Universe-scale precision + LOD (milestone: fly to any scale)

- [ ] CPU dvec world + camera-relative rebase; reversed-Z `D32_SFLOAT` depth.
- [ ] Octree LOD renderer (share FMM octree): frustum+solid-angle culling,
      per-octant instanced draws, stream visible nodes.
- [ ] GPU frustum/distance culling with indirect draw (flat frame time).
- [ ] Trails: GPU polyline quad-strips for bound orbits; procedural motion
      stretch (screen-velocity).
- **Exit:** zoom from kpc view of a galaxy down to a planet without jitter;
      >10M bodies rendered with culling at 60 fps.

## Phase V2 — Bodies & planets (milestone: the "sandbox" look)

- [ ] Sphere impostors (ray-traced on quad, `SV_Depth`, perspective scaling)
      + atmosphere glow + bump maps.
- [ ] Impostor→mesh transition; cube-sphere quadtree terrain with GPU
      procedural noise (SpaceEngine pattern), S3TC-compressed texture arrays.
- [ ] Star/dust fields: additive, post-pass glare (mip pyramid).
- [ ] Optional GPU surface sim (temperature/gas) via per-body grids + VRAM
      Atlas — the Universe-Sandbox signature feature.
- **Exit:** an interactive solar-system/galaxy scene visually comparable to
  Universe Sandbox on consumer hardware.

## Phase V3 — Zero-copy CUDA↔Vulkan (milestone: one pipeline, no copies)

- [ ] Export Vulkan buffers via `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`;
      import into CUDA; timeline-semaphore sync both ways.
- [ ] FMM compute (CUDA) reads/writes the same storage buffers the renderer
      draws → truly zero-copy live simulation.
- [ ] Cross-API correctness test (pixel-identical to CPU-readback reference).
- **Exit:** physics and rendering on one GPU with no host round-trips; full
  perf budget (doc §9) met.

## Phase V4 — Distant stars & polish (milestone: cinematic)

- [ ] Precomputed photometric star background rendered into a camera-centered
      sky field; proper-motion layer optional.
- [ ] Bloom tuning, exposure/tonemap curves (Lupton-asinh for astronomy),
      FXAA/TAA, optional DLSS/FSR upscaling.
- [ ] UI: selection, labels, time controls (variable sim speed), scenario
      spawning.
- [ ] Software-raster star path (compute splatting) for >10M-star scenes.
- **Exit:** the complete live, interactive, efficient visual system; energy +
  perf benchmark report (GFLOPS, FLOP/watt, frame time budget).

## Rough effort estimate (relative) — both parts

| Phase | Effort | Risk |
|---|---|---|
| 0 | 2 | low |
| 1 | 5 | medium (M2L math) |
| 2 | 3 | medium |
| 3 | 5 | high (M2L GPU) |
| 4 | 2 | low |
| 5 | 3 | medium |
| 6 | 2 | low |
| V0 | 2 | low |
| V1 | 3 | medium (precision) |
| V2 | 4 | medium (terrain) |
| V3 | 3 | high (CUDA↔Vulkan) |
| V4 | 2 | low |