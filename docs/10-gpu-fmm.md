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
- [ ] **S2b — GPU sweeps + P2P.** P2M/M2M upsweep, L2L/L2P downsweep
  level-sweep kernels; near-field P2P kernel. Gate: full-eval rel err
  ~1e-3 vs direct.
- [ ] **S3 — visual-engine integration.** `GpuBackend` behind `Physics`
  (or parallel to it): CPU tree build (4 ms @100k, keep) + traversal,
  GPU resident moments/locals, download accels only.
- [ ] **S4 — kernel tuning.** Shared-mem tiling across pairs sharing a
  source cell, Kohnke symmetry classes (docs/03 §2.1), occupancy sweep.
  Target: saturate VRAM bandwidth on replay.

## Notes

- `ROT_P` rotation path is CPU-only; GPU uses naive convolution (more
  regular for SIMT anyway).
- Bit-compat contract explicitly does NOT extend to GPU (fp32).
- Long-term home may become `crates/fmm-gpu`; stays in `visual` until S3.
