# 10 — GPU-compute FMM (wgpu, fp32, visual-first)

## Decisions (locked)

- **wgpu compute, not CUDA.** Same device as the renderer → zero-copy
  buffers, one toolchain, portable. CUDA buys control but needs context
  interop for ~1.5x; revisit only if wgpu tops out.
- **fp32, visual-first contract.** Consumer Ada fp64 is 1/64 rate — fp64 on
  GPU is not a target. Accuracy gate: rel err vs CPU-direct ~1e-3 at p≤6.
  The fp64 CPU path stays untouched for science gates.
- **Hybrid split: CPU traversal, GPU everything else.** Measured at 1M:
  record (traversal) 0.44 s vs replay 5.64 s of 7.07 s walk. Traversal is
  branchy pointer-chasing (bad GPU fit, cheap on CPU); replay is a flat
  array of independent pairs (perfect GPU fit, 80% of cost).
- Per-step transfers at 100k/loose: pairs ~26 MB up, positions/masses
  ~3 MB up, accels ~2 MB down. Moments/locals never cross PCIe (P2M/M2M
  and L2L/L2P run on GPU in S2).

## Stages

- [x] **S1 — M2L kernel prototype** (`visual/src/gpufmm.rs`, `probe --m2l`).
  One-sided M2L port (Θ build + convolution, one thread/pair, p=4).
  Measured RTX 4060: 0.023 µs/pair @500k pairs vs CPU 0.268 µs
  (**11.6x**), max rel err 1.6e-4 vs fp64 CPU. Traffic ~18 GB/s of
  ~272 — latency/occupancy-bound, headroom left for S4.
- [x] **S2a — GPU replay with accumulation** (`walk::record_pairs_par`
  refactor + `M2L_RED_WGSL` segmented reduce). `dual_walk_par` now calls
  the extracted recorder (full suite + release exit gate green).
  `probe --m2lacc` @30k/loose: record 12.4 ms + group 1.0 ms (CPU) +
  replay 11.6 ms (GPU) vs CPU walk 27.3 ms; max rel 5.1e-4 vs serial
  one-sided AND vs `dual_walk_par` locals (pair-set validated — both refs
  agree below the fp32 floor). Detour survived: thread-per-cell starved
  the GPU (~500 threads, 33.7 ms); two-kernel S1-slots + segmented reduce
  fixed it. Parity, not victory: kernel runs ~140 GFLOPS (1% of peak,
  latency-bound Θ recursion) — S4 now gates the real win.
- [x] **S2b — GPU sweeps + P2P** (`WGSL_HELPERS` shared Υ-table prelude,
  `p2m/m2m/l2l/l2p/p2p_source(p)` generators, `probe --gpufull N ref p eps`).
  Full eval @8k/p6/eps=1e-4: GPU 40 ms; **gpu-vs-cpu-fmm max 8.2e-6 (p4)
  / 3.2e-2 max tails (p6)**; **mean vs direct 5.9e-4 — gate green**.
  @30k/loose: GPU 27.4 ms vs CPU 61.6 ms (2.2x end-to-end incl. 13 ms CPU
  record), pipeline agreement max 2.7e-5. Caught en route: a shared uniform
  rewritten between dispatches silently broadcasts the last value to every
  pass (all writes land pre-submit) — per-dispatch uniform buffers now.
  Known tail: fp32 close-encounter pairs (max 3e-2); S4 fix = f64/HiGPUs
  distances in P2P.
- [x] **S3 — visual-engine integration.** `GpuBackend` (visual preset
  p=4/eps=1e-2/nl=64) behind `sim::Physics`, `--gpu` flag + auto above
  30k. Temporal coherence: tree + interaction lists refresh every K=4
  steps, positions upload fresh, moments/locals recompute on GPU, accels
  download only (~2 MB/step). `probe --gpuloop`: **100k @ 83.9 ms/step
  (11.9 steps/s) vs CPU 172 ms (5.8/s) — 2x**; 30k @ 28 ms (35.7/s).
  `probe --gpureuse`: frozen-K4 mean 4.9e-3 = truncation floor (adds
  nothing); fresh-vs-CPU max 1.8e-5. Caught en route: backend returned
  slot order, Sim needs particle-id order — scatter added (was a dynamics
  bug). Remaining wall: CPU traversal ~55 ms (prepass+record) even
  amortized; S4/S5 territory.
- [x] **S4 — kernel tuning** (measured, not as planned — the profile
  overturned the plan). `probe --gpuprof` (isolated submit+poll per stage;
  no timestamp queries on this adapter) @100k showed M2L-s1 healthy
  (~640 GFLOPS) but **m2l-red 21.4 ms (42%) + p2p 20.0 ms (39%)**:
  - m2l-red was thread-per-cell (6143 threads, serial DIM chains) →
    thread-per-(cell, coef), 153k threads, contiguous slot reads:
    **21.4 → 1.9 ms (11x)**.
  - P2P thread-per-body → per-leaf workgroup, partner tile in 10 KB
    shared mem (chunked ×512, any size): 20.0 → 11.9 ms.
  - Census then showed the real pathology: 12 halo leaves (lone bodies
    in huge cells fail MAC vs everything) hold 100k partners each —
    39% of P2P iters in 0.37% of bodies, serial 100k-chains as the
    occupancy tail. Fix: `split_fat` (>4096 partners → 2048-slice jobs,
    one WG each, scratch partials + reduce; 7 storage bindings to fit
    the adapter's 8/stage limit): **P2P → 2.3 ms total (7x)**.
  - End-to-end `--gpuloop`: **100k 39.0 ms/step (25.6/s) vs S3 83.9
    (11.9/s) vs CPU 172 (5.8/s) — 2.1x on S3, 4.4x on CPU**; 30k
    19.3 ms (51.7/s). Triangle unchanged (mean 5.86e-4; p4 vs CPU-FMM
    max 8.5e-6; frozen-K4 still on truncation floor).
  - Deliberately NOT done: S1 mutual-kernel halving / reciprocal hoist
    (s1 is 6% — Amdahl says no), Kohnke symmetry classes (adaptive
    octree has no 189-stencil to symmetrize; the planned shared-mem
    M2L tiling died to measurement — s1 was never the wall), P2P
    double-single distances (fp32 max-tail 3e-2 stands; SHADER_F64
    absent, DS upload is S5 scope). Remaining wall is CPU traversal
    (~27 ms amortized) — S5 (GPU traversal) territory.

## Notes

- `ROT_P` rotation path is CPU-only; GPU uses naive convolution (more
  regular for SIMT anyway).
- Bit-compat contract explicitly does NOT extend to GPU (fp32).
- Long-term home may become `crates/fmm-gpu`; stays in `visual` until S3.
- [x] **S5 — smooth playout (render side only).** Exact-window rate +
  adaptive gap-covering target + weak-spring cursor (playout 0.97–1.02x @100k
  pattern in simulation), drop-newest snapshot send, dead prev-buffer upload
  cut (halved frame bytes, pixel-identical). Sync refresh hitches remain
  (140 ms @100k) but ride inside the buffer invisibly.
- [ ] **Pipelined refresh — REVERTED, do not retry without the secular gate.**
  Moving the rebuild to a background thread looked like a free 2.3x
  (100k 29→13 ms avg, max 140→18 ms) but stretched tree age to k+build
  steps and disks evaporated secularly past age ~8 (sync k=8 bounded 200
  steps at max|v|~3.2; sync k=16 pinned at the 5.0 clamp). Instantaneous
  force error is flat to k=16 and does NOT predict heating. Rule: installed
  age never exceeds k. Gate: `spiral_30k_200steps_stays_bound` (ignored,
  release). First lever on failure: k 8→4.
