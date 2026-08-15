# 07 — Parallelization

## 1. Parallelism structure of FMM

FMM exposes natural data parallelism: at each tree level, interactions are
independent per (source,target) octant pair. Two parallelism levels:

1. **Across octants** — the dual tree walk is task-parallel.
2. **Across the rows of an operator** — inside M2L/P2P mat-vecs (SIMD + threads).

Up/down passes (M2M, L2L) are sequentially dependent in the classic algorithm;
parallelize with task-based schemes (rake/compress methods, Fortin–Touche
atomic task updates) only if profiling justifies it.

## 2. Shared-memory (one node)

- **Task-based** dual tree walk (OpenMP tasks with `atomic`, or Rayon/TBB):
  the recommended pattern — dynamic load balancing across cell pairs, which is
  essential for non-uniform data (Dehnen 2002; exafmm).
- **Morton-order** sorting (Z-order curve) for tree build and to keep adjacent
  cells contiguous in memory.
- NUMA: pin threads, first-touch allocation, place cell data on owning socket.
- OpenMP scheduling: for the direct N² fallback use `static,chunk=8` (chunk ≥
  cache line to avoid false sharing); for the FMM walk use tasks.

## 3. Distributed (multi-node / multi-GPU)

- FMM's communication graph has a **small bisector** O(N^{2/3}) — this is why
  FMM scales better than PME (which does all-to-all). This is the *scientific
  justification* for choosing FMM on clusters.
- Partition: Morton-range splits + geometric graph partitioning (Zoltan) for
  load balance on non-uniform distributions.
- Communication pattern:
  - Local tree build → exchange boundary cells' multipoles.
  - Reduce acceleration contributions once (MPI_Allreduce on the small
    acceleration array).
  - Each rank updates its own particles' positions; **no broadcast** of the
    full state (avoids the biggest bandwidth waste; measured in fbaru/nbody-1).
- Hybrid MPI+OpenMP beats pure MPI for memory scalability (FMB vs GADGET-2
  finding); pure-MPI wins only on very large uniform problems.

## 4. GPU-specific parallelism

- One block per box for tree ops; P2P with shared-memory source tiling.
- **M2L:** the three proven schemes are in 03 §2 (symmetric/coalesced,
  dynamic-parallelism, unified-memory). Symmetric precomputed-operator kernel
  is the production choice.
- Multi-GPU: 32-GPU runs reach 44% efficiency @ 10⁶ and 66% @ 10⁷ particles;
  256M over 256 GPUs demonstrated. Use NVIDIA collective communication
  (`nccl`) for the boundary exchange.
- Persistent kernels + CUDA graphs to amortize launch overhead.
- Host↔device: pinned memory + multi-stream overlap of position upload with
  previous step's force download.

## 5. Load balance specifics

- Non-uniform data (galaxies): fixed interaction lists (traditional FMM)
  badly underutilize nodes → prefer the **dual-tree-walk + dynamic MAC**
  (falcON-style), which adapts per-interaction.
- GPU leaf size q is a load-balance knob too: larger q on the fast device
  reduces tree depth and skew (Lashuk 2013 synergy).

## 6. Practical scaling numbers (benchmarks to reproduce)

- CPU: exafmm-t single socket 14-core: 1M particles @7 digits in 0.95 s.
- GPU single: GROMACS FMM α≈1 scaling for N>30k on V100; FMM beats PME on
  inhomogeneous systems on one node.
- CPU vs GPU: GPU 32× single-thread CPU, 4.3× 4-thread SSE CPU (Lashuk 2013).