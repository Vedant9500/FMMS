# 05 — Energy efficiency (FLOP/watt)

## 1. Where energy actually goes

- On GPU nodes, GPUs draw ~75% of node power; CPU ~5–10%.
- At exascale scale, energy efficiency (FLOP/s/Watt) is the real cost metric.

## 2. Measured levers, ranked by payoff

### 2.1 GPU offload (biggest lever)

SuperMUC-NG Phase 2 (CPU Xeon Platinum 8480+ vs Intel GPU Max 1550):

- GPU offload: **4–12× higher throughput**, **up to 15× better energy
  efficiency** vs CPU-only for the same code.
- Energy efficiency tracks occupancy: under-filled GPU tiles erode the
  advantage. Saturate the device (larger N or larger leaf q).

### 2.2 GPU power capping (free energy)

- Compute-bound kernels (GEMM) are **up to 30% more energy-efficient at
  55–70% of TDP** than at full power. Faster ≠ more energy-efficient.
- Power-capping all GPUs: +24.3% energy efficiency at −26% throughput.
  Capping a subset: +9–12% efficiency at −12% throughput (better trade-off).
- Frequency capping often beats power capping for compute-bound codes
  (up to +13.2% efficiency with zero performance loss, per Frontier study).
- Implement via `nvidia-smi -pl <watts>` / CUDA `cuDeviceSetLimit` /
  p3em + EAR for instrumentation.

### 2.3 CPU power capping (free)

- Capping CPU to ~48% TDP (60/125 W): **+21% energy efficiency with zero
  performance impact** when compute runs on GPU.

### 2.4 DVFS on memory-bound phases

- Roofline rule: if a kernel is memory-bound (arithmetic intensity < machine
  balance), lowering clock frequency cuts energy with negligible performance
  loss; if compute-bound, run at max sustained frequency.
- Function-by-function DVFS is practical on CPUs (~10 µs switch cost,
  −9% energy), not yet worth it on GPUs.

### 2.5 Precision (the algorithmic energy lever)

- FP32 vs FP64 on consumer GPUs: 3–4× speed → ~3× less energy per
  simulation (GENGA measured energy reduction from FP32 long-range forces).
- Fewer operations = fewer joules. FMM's O(N) vs O(N²) is the ultimate
  energy win: ~N× fewer FLOPs at large N.

## 3. Measurement tooling

| Tool | What it gives |
|---|---|
| RAPL MSRs (`perf`, `turbostat`) | package + DRAM energy (Intel CPUs) |
| `nvidia-smi` / `nvml` | GPU power/energy, cap control |
| `p3em`, `EAR` | phase-accurate node energy attribution |
| `perf stat energy` | quick per-run joules |
| IPC + Roofline | which phase is worth tuning |

## 4. Design implications for FMMS

1. Default target = GPU-accelerated; CPU-only mode is a fallback for small N.
2. Sweep power cap at install time per GPU model; store best (cap, ε) config.
3. Power-cap the CPU whenever GPU path active.
4. Track energy/Joule as a first-class benchmark metric next to GFLOPS.
5. Prefer FP32 kernels (02 §5) on consumer GPUs; FP64 only for accumulation.

## 5. Expectation setting

Instruction-level tricks (unrolling, FMA ordering, avoiding one div) are
fraction-of-a-percent energy items. Algorithm (FMM vs N²), precision (FP32 vs
FP64), and device selection (GPU vs CPU) are the 10–100× items. Spend the
research budget there first.