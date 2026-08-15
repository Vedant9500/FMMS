# 09 — Live 3D Visual System (Universe-Sandbox-style)

Second research thread: a **live, interactive 3D visual system**, not a
precomputed-then-replayed movie. The end product is a real-time, GPU-driven
simulation you fly around in — like Universe Sandbox. Pygame-style playback is
acceptable only as a throwaway test harness for physics correctness, never the
deliverable.

This doc is the companion to `01–08` (physics/algorithm) and covers the
rendering + interactivity half.

## 1. The cardinal rule: never cross the wall per-frame

The #1 bottleneck in live particle sims is the CPU→GPU transfer ("the great
wall of data separation"). Passing transform matrices or full position arrays
from CPU to GPU every frame caps you at ~100k particles (measured ~15 fps at
1M in a naive Unity instancing setup). The fix, used by every system that
renders millions of particles live:

> **State lives on the GPU. Compute shaders update it in place. The renderer
> reads the same storage buffers via instanced/indirect draws.**

- Compute pass writes positions/velocities in storage buffers.
- Render pass draws instanced quads/triangles, reading per-instance data with
  `instance_index` from the same buffer.
- The CPU only uploads constants (uniforms) and structural changes, never the
  particle array, every frame.

Measured results of this pattern: 10M particles @ ~63 fps on a GTX 1060
(WebGPU thesis); 2M @ 60 fps on a laptop (raylib compute example); 1M+
particles smooth even on low-end PCs (indirect instancing). Our FMM targets
10⁴–10⁷ bodies, squarely in reach.

## 2. Rendering a lot of points, fast

| Technique | When | Note |
|---|---|---|
| Point sprites (`GL_POINTS` / `point-list`) | tiny or distant particles | fastest, but fixed 1px; no perspective distortion handling |
| **Instanced billboard quads** | primary choice | camera-facing quad, perspective-correct scale (see impostor math), depth write; one draw per LOD level |
| Vertex pulling | >4M particles | 41% faster than instancing at 10M (GTX 1060); index buffer pulls from storage buffer |
| Compute-shader single-pixel splatting (additive) | millions of sub-pixel particles | 31–350% faster than rasterization; atomic-add to a pixel buffer, then fullscreen pass (Turitzin) |
| Software rasterizer for stars | billions, cinematic glare | Uni Weimar: smaller-than-pixel stars + post-process glare scales best |

Draw-call strategy: **indirect instancing / GPU-driven rendering**. Frustum
and distance culling run on the GPU; a single `drawIndices`-style indirection
buffer lets the GPU decide what to draw, so frame time is flat regardless of
total body count. Additive blending (no depth sorting needed) for star/dust
fields.

**Quality floor for a Universe-Sandbox feel:** render to an HDR offscreen
target (`rgba16float`), accumulate additively, then tone-map (ACES / Reinhard
/ Lupton-asinh) in a final pass. Add bloom as a post-process. Stars drawn as
small as possible + a post-pass glare mipmap pyramid (Uni Weimar measured
0.34–1.38 ms at FHD/4K) beats big billboards by a wide margin.

## 3. Universe-scale precision without breaking the GPU

An N-body sim spans 10¹⁰+ in coordinate range (AU to kpc/galaxy). f32 GPU
math can't represent that. The standard solution (Gaia Sky, SpaceEngine,
Kerbal, HDRP, VulkanSceneGraph, Godot double builds):

1. **CPU keeps double-precision (or better) world state.**
2. Every frame, subtract the camera position (dvec math) from each body →
   `P_local = P_world - P_camera`.
3. Cast the result to f32 and send to the GPU. Near-camera objects get
   sub-mm precision; far objects are sub-pixel anyway.
4. GPU works entirely in f32 at full speed. No f64 shaders needed.

Two refinements:

- **Hierarchical reference frames** (VulkanSceneGraph `CoordinateFrame`,
  chrisjz/universe): attach a double-precision origin to each local system
  (solar system, galaxy); when the camera is local to a frame, the delta is
  exactly zero. Works to distances of 10³⁰⁴ (tested) with zero jitter.
- **Reversed-Z + `D32_SFLOAT` depth buffer** for correct depth across the
  huge near/far ratio (VSG uses this by default).

Shaders do not need f64. If you ever do want f64 shaders, note WGSL exposes it
only via the optional `SHADER_FLOAT64` native feature (no Metal, 8–16× slower
on NVIDIA) — avoid.

## 4. Planets and bodies (the "sandbox" look)

A cheap, high-quality path to Universe-Sandbox-style bodies:

- **Far away:** instanced point/quad — indistinguishable from a dot.
- **Mid:** **sphere impostor** — one camera-facing quad, fragment shader
  ray-traces a sphere (position + normal computed analytically), writes
  `SV_Depth` so it intersects the scene correctly (bgolus/paroj tutorials).
  Add perspective-correct quad scaling via the right-triangle trig
  (`quadScale = tan(asin(r/d)) * d`). Includes atmosphere glow shader
  (precomputed transmittance/inscatter, Sean O'Neil style) and bump map.
- **Near surface:** real mesh / cube-sphere quadtree terrain with GPU
  procedural generation (SpaceEngine: elevation+color noise in shaders,
  quadtree per cube face, displacement mapping in vertex shader, texture
  arrays with S3TC compression). This is the *later* phase; impostor-to-mesh
  transitions + mip progression handle everything between.

This mirrors Universe Sandbox: planets are "mostly spheres with fancy
shaders"; the expensive surface simulation (temperature, liquid, gas) is
optional and GPU-side via per-body grids + a shared Atlas budgeted by VRAM.

## 5. Trails / orbits / motion

- **Orbit trails:** GPU line renderers as polyline quad-strips (geometry
  shader on CPU → GPU lines; Gaia Sky measured big wins). Instanced Kepler
  orbital trails for bound bodies.
- **Motion trails / star streaks:** vertex-shader procedural stretch based on
  screen-space velocity (Gaia Sky) — no history buffers, compatible with
  additive blending, scales to hundreds of thousands of objects.

## 6. Interactivity / camera / loop

- **Fixed-timestep simulation** (Glenn Fiedler pattern): accumulator consumes
  real time in fixed `dt` steps; 0, 1, or N physics ticks per frame.
- **Interpolation** between previous/current physics state (`alpha =
  accumulator/dt`) for the *rendered* pose → smooth motion at any refresh
  rate; camera updates every frame (variable timestep). This is exactly the
  Bevy/Adobe/SnapNet pattern and keeps FMM block-timestep integrators
  deterministic.
- Input captured per-frame and routed to the tick loop; camera is a game
  object updated at render rate, then re-based through the precision pipeline
  in §3.

## 7. Level-of-detail for huge populations (≥10⁶ bodies)

Same octree the FMM physics already builds. Group bodies into octants with a
max per-node count; at render time:

- Cull octants by frustum + solid-angle/distance metric (`θ`).
- Render only octants whose projected size justifies the detail; subsample
  dense distant octants.
- Stream/load visible nodes only; LRU cache; progressive fill.

This is the Gaia Sky / Firefly / OpenSpace / Potree pattern (Firefly: ≳10
million particles live in-browser, billions via octree). The unification win:
**one spatial hierarchy serves physics and LOD rendering.**

## 8. Stack decision for our FMMS project

Given research-1 chose **Rust host + CUDA kernels via FFI**:

| Component | Choice | Why |
|---|---|---|
| Window/loop/UI | Rust (`winit` + `egui`) | fits Rust host |
| Graphics API | **wgpu** (Vulkan/DX12/Metal native backends) | mature GPU-driven compute+instancing in Rust; WGSL |
| FMM compute | CUDA (existing plan) | needs Vulkan↔CUDA zero-copy: `VK_EXT_external_memory_fd` + `cudaImportExternalMemory` + timeline semaphores (NVIDIA-proven pattern) |
| Fallback compute | wgpu compute shaders (pure-Rust GPU path) | same renderer, swap the compute backend |
| Test-only playback | Pygame/Matplotlib | acceptable only for physics correctness smoke tests |

**Alternative (fastest to a Universe-Sandbox clone):** Unity DOTS/URP — it is
literally what Universe Sandbox uses (N-body on CPU via ECS/Burst, surface sim
on GPU, URP shaders). Lower ceiling than the custom path but huge DX. Decision
gate: if we ever need Unity-scale UX velocity, port the renderer; the physics
engine stays.

## 9. Performance budget (derived)

Per frame @ 60 fps (16.7 ms), target GPU budget:

| Stage | Budget |
|---|---|
| Physics (FMM) compute | 3–6 ms (1M–10M bodies) |
| Particle render (billboards/splats) | 2–4 ms |
| HDR + tone map + bloom | 1–1.5 ms |
| Planets/bodies (impostors) | 1–2 ms |
| Trails | ≤1 ms |
| **Total** | **≤12 ms** (headroom for UI/camera) |

These are measured ranges from the sources above, not guesses; validate on
target hardware in Phase V0 benchmarks.

## 10. Related prior findings

See library findings: fnd-2a463ce1f68c (GPU-driven data flow), fnd-ddf74c8b40f9
(particle render hierarchy), fnd-7c676f50c535 (camera-relative precision),
fnd-85e7a6561967 (fixed-timestep/interpolation), fnd-39f8cd579a77 (octree
LOD), fnd-17a8aa325430 (Universe Sandbox architecture), fnd-459366d4c8d8
(stack options), fnd-86ae90529a22 (visual quality techniques).