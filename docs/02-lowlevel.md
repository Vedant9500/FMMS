# 02 — Low-level CPU optimization

## 1. The fast-math truth table (set expectations)

| "Trick" | Reality |
|---|---|
| Replace float multiply by bitshift | Nonsense on IEEE-754 floats; FP mult is one instruction, same latency class as add. No gain. |
| `i = 0x5F3759DF - (i>>1)` fast inverse sqrt | Historically great; **obsolete**. Hardware `rsqrtss/rsqrtps` (SSE) does the same in HW, 12-bit accuracy, ~4× faster than the magic-constant code. |
| Avoid division | Real. Replace `1/r` with reciprocal-sqrt: `rsqrt` (12-bit) + **one Newton–Raphson step → 24-bit**. Force loop does one `rsqrt` + a few FMAs instead of `sqrt` + `div`. |
| FMA (fused multiply-add) | The single most important instruction-level win in the P2P kernel. Compilers emit it with `-ffast-math` / `-march=native`. |

## 2. SIMD vectorization

Measured speedups (AVX-512 n-body study, Xeon Phi + Cascade Lake):

| Technique | Speedup over naive |
|---|---|
| SSE | 3.9× |
| AVX2 | 7.4× |
| AVX-512 | 15.1× |
| Cache blocking (BS=16) | 2.9× (additive) |
| Loop unrolling | +9% |

AVX-512: 8 FP64 or 16 FP32 FMA-lanes per instruction (32 FLOPS/instruction FP32).
Phantom-GRAPE AVX library: 2×10⁹ interactions/s/core = 75 GFLOPS (38 flops per
interaction), 20× vs no-SIMD, 2× vs SSE.

exafmm P2P kernel: no-SIMD 48.7 → SSE 189 → AVX 321 GFLOPS (6.6× total).

**Rules:**
- Write kernels with explicit intrinsics (`<immintrin.h>`); don't trust
  auto-vectorization for the inner loop.
- Interleave loads with FMAs to hide latency; aim for near-peak issue rate.
- Use `rsqrt` + NR iteration, with FP64 accumulation.
- Loop counters stride by the vector width (8/16), handle remainders with
  narrower vectors (SSE then scalar), as exafmm does.
- `-O3 -march=native -ffast-math -funroll-loops` baseline; consider ICC/oneAPI
  which outperforms GCC on these loops.

## 3. Data layout: SoA, not AoS

Structure-of-Arrays (`x[] y[] z[] m[] vx[] vy[] vz[]`) vs Array-of-Structures.
Measured in the tycho/nbody benchmark: **SoA ≈ 2.3× faster than AoS**
(836 GFLOPS vs 289 GFLOPS at 98k particles, 16 threads).

Reasons: contiguous vector loads, no strided gather, better prefetching,
simpler auto-vectorization, smaller cache footprint per lane.

Extra: 64-byte aligned allocations (`aligned_alloc`, `posix_memalign`), and
mark alignment assumptions (`__assume_aligned`) so the compiler can use aligned
loads.

## 4. Cache optimization

- **Arithmetic intensity (Roofline):** kernel flops ÷ bytes moved. Must exceed
  machine balance (FLOP/s ÷ bytes/s, e.g. ~24 flops/byte for GPUs, ~10-20 for
  CPUs) to be compute-bound.
- **Tiling/blocking:** process particle data in tiles that fit L2 (BS≈16 bodies
  per tile). 2.9× average speedup.
- **Software prefetching** of the next tile while computing the current one.
- **SoA ordering + Morton (Z-order) keys** keep spatially close bodies adjacent
  in memory → near-field P2P walks hit contiguous cache lines.
- **NUMA awareness:** first-touch allocation, thread pinning, place leaf cells
  and their moments on the socket that owns them.

## 5. Precision strategy (this is a big lever on GPU too)

| Phase | Precision | Why |
|---|---|---|
| Position subtraction, acceleration accumulation | FP64 | rounding critical (this is the "distance" path) |
| Force inner math (rsqrt, FMAs) | FP32 | consumer GPUs 3–4× faster in FP32; force error budget absorbed by FMM approximation error |
| Intermediate moments (M2L) | FP32 or mixed | order p accuracy dominates anyway |
| Integrator state | FP64 | energy/angular-momentum conservation |
| Double-single emulation | 2×FP32 ≈ FP64 | Sapporo/GRAPHE route; ~20% perf loss vs FP64-native GPU, only if HW lacks FP64 |

Guideline from GENGA: FP32 long-range forces give ~3–4× speedup on consumer
GPUs with *statistically indistinguishable* outcomes; angular-momentum error
~2 orders worse but still acceptable. On data-center GPUs (A100) FP32≈FP64, so
the lever only applies to consumer parts.

Pitfall (HiGPUs/DS): low-precision force eval makes the Aarseth timestep
criterion generate spurious tiny timesteps → use the PRS23 criterion (see
`06-integrators.md`) which is robust to rounding.

## 6. Kernels that need hand-tuning (priority order)

1. **P2P** (near-field direct summation) — highest SIMD payoff. Per interaction:
   `dx = x_i - x_j; r2 = dx²+dy²+dz²; rinv = rsqrt(r2); f = m_j·rinv³`.
   That's the whole loop; make it 8 bodies per AVX-512 iteration.
2. **M2L** — see `03-gpu.md`; on CPU use rotation-accelerated O(p³) or BLAS
   matrix form (PVFMM caches the M2L operator reuse pattern).
3. **L2P/P2M** — cheap; keep them simple, let compiler vectorize.

## 7. Profiling workflow

- **Roofline model** first: classify kernels compute- vs memory-bound.
- `perf stat` counters: IPC, L1/L2/L3 miss rates, branch mispredictions.
- Intel Advisor / `-qopt-report` for vectorization feedback.
- Only then write intrinsics or assembly. Measure every change in isolation.