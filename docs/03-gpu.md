# 03 — GPU / CUDA design

## 1. When GPU is the right target

- N > ~10⁴ (adaptive GPU FMM beats even optimized direct N² at ~3500).
- Inhomogeneous / large systems (GROMACS FMM >2× PME on multidroplet
  aerosols; small homogeneous systems still favor PME-style).
- Consumer GPUs: FP32 path gives 3–4× over FP64 (see 02 §5).

Measured headroom: single-GPU FMM ~30–70× a single CPU core (spherical-harmonic
FMM), 11× an SSE-optimized serial CPU (adaptive FMM). GROMACS GPU FMM: linear
scaling for N > 30k, α≈1.

## 2. The M2L problem (the whole GPU game)

M2L dominates: in PBC it needs **189 transformations per box** vs 1 each for
M2M and L2L — 94.5× the operations. A naive M2L kernel is:

- memory-bound (arithmetic intensity far below machine balance),
- uses <10% of issued instructions on useful FP (rest is integer bookkeeping),
- dominated by integer index arithmetic: 31 int-adds, 16 int-mults, 9 modulos,
  11 int-divides per complex multiply.

Goal: **make M2L compute-bound.** Proven approaches, in increasing payoff:

### 2.1 Exploit operator symmetry (Kohnke et al. 2021) — best documented

- The 189/316 M2L operators for a box reduce to a small set of symmetry
  classes (8-fold, 4-fold, 2-fold); precompute distinct operators only.
- Layout moments/operators in shared memory with a **presorted, coalesced
  layout** (Darve's GPU2–GPU4: block source clusters so the same operator
  serves many (source,target) pairs → arithmetic intensity 2 → 21–32 flops/word).
- Result: **compute-bound**, ~70% of GPU peak (8 TFLOPS on the tested part),
  warp divergence eliminated, near-zero integer overhead.

### 2.2 CUDA Dynamic Parallelism (Kohnke approach 2)

- Parent kernel computes octree indices/interaction sets on the host/prepass;
  child kernels (2×2×2 blocks) each do 8 O(p⁴) transforms with moments cached
  in shared memory.
- Best at high p (p ≳ 10). Poor occupancy at low p (single-warp blocks).

### 2.3 Unified-memory quick port (approach 1)

- Least effort, decent speedup, but memory-bound. Good as a first milestone,
  not the final kernel.

### 2.4 On-the-fly operator recomputation vs precompute

- Precompute-and-reuse wins (Takahashi): keeps M2L compute-bound while avoiding
  the memory traffic of storing 316 operators.
- Requires near-uniform distribution (exploits stencil symmetry).

## 3. Kernel strategy summary

| Kernel | Design |
|---|---|
| Tree build | Morton sort + radix sort, warp-level prefix sums, `ballot()`/mask to remove branches (Bonsai style). Or build on CPU, upload once. |
| P2M / M2M / L2L / L2P | one block per box; shared-memory staging; trivial, keep simple |
| M2L | symmetric precomputed operators, shared-memory tiling, coalesced layout (2.1) |
| P2P | cache source tile in shared memory (size = threads), one thread per target, symmetric upper-triangle only for CPU version |
| U2U / D2D (up/down pass) | sequential in classic impls; parallelize if profitable |

Persistent kernels + CUDA graphs + streams reduce launch overhead; page-locked
(pinned) host memory and multi-stream overlap for the host↔device transfer of
positions.

## 4. Precision on GPU

- FP32 inner loops, FP64 accumulation (double-single where needed) — the
  HiGPUs pattern: FP64 only for distance computation and force accumulation.
- `-use_fast_math`, `__frsqrt_rn` / `rsqrtf` for reciprocal sqrt (12-bit +
  NR step = 24-bit).
- Mixed precision in GROMACS FMM: p=8 matches PME accuracy; energy drift
  unchanged.

## 5. Multi-GPU / distributed

- FMM communication graph has a small bisector (Teng: O(N^{2/3})) → much
  better scaling than PME's all-to-all.
- 32-GPU single-cluster impls hit 44% efficiency @ 10⁶, 66% @ 10⁷. 256M
  particles over 256 GPUs demonstrated (Blanchard et al.).
- Partition by Morton-order range; use geometric graph partitioning (Zoltan) for
  load balance on non-uniform data.
- Copy exact reduction patterns (MPI_Allreduce only for the reduced
  acceleration set; each rank updates its own positions → no broadcast).

## 6. Alternatives when GPU underutilizes

- Small N (< 10⁴): GPU direct N² kernel (one block per target, 64 threads)
  beats FMM — keep a direct-mode fallback.
- CPU + GPU hybrid (exascale FMM, Lashuk 2013): run ULI/WLI/XLI on GPU, VLI
  (memory-bound, diagonal translation) on CPU, M2L on CPU for small p.
- Larger leaf q on the fast device (algorithmic tuning synergy).