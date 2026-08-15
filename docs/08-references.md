# 08 — Annotated bibliography

## Core algorithm

- **Greengard & Rokhlin 1987** — "A fast algorithm for particle simulations",
  J. Comput. Phys. 73:325. The original FMM. (Foundation.)
- **Cheng, Greengard & Rokhlin 1999** — "A fast adaptive multipole algorithm in
  three dimensions". J. Comput. Phys. 155:468. Rotation-based O(p³) M2L, adaptive
  octree, error bounds.
- **Greengard & Rokhlin 1997** — "A new version of the fast multipole method...",
  Acta Numerica 6:229. Plane-wave/diagonal M2L.
- **Barnes & Hut 1986** — "A hierarchical O(N log N) force-calculation
  algorithm", Nature 324:446. The treecode baseline.

## Dehnen lineage (recommended architecture)

- **Dehnen 2000** — "A very fast and momentum-conserving tree code", ApJ 536:L39.
  Cartesian Taylor FMM with mutual cell–cell interactions → momentum conservation.
- **Dehnen 2002** — "A hierarchical O(N) force-calculation algorithm", J. Comput.
  Phys. 179:27. Dual tree walk, MAC, mass-dependent opening angle.
- **Dehnen 2014** — "A fast multipole method for stellar dynamics", arXiv:1405.2255,
  Comput. Astrophys. Cosmol. 1:1. Error-controlled MAC, smallest-enclosing-sphere
  expansion centers, solid harmonics, sub-linear N^0.87 scaling. **Primary design
  reference.**

## FMM performance studies / implementations

- **CMU DIMACS** "A practical comparison of N-body algorithms" —
  https://www.cs.cmu.edu/~scandal/papers/dimacs-nbody.html — FMM vs BH vs PMTA
  flop counts vs accuracy; the N > 10⁴ crossover data.
- **Fortin et al. 2011** — A&A 530. Comparison of falcON, GADGET-2, FMB; falcON
  fastest serial.
- **exafmm-t (JOSS 03145)** — https://joss.theoj.org/papers/10.21105/joss.03145.pdf —
  OpenMP+AVX-512 FMM, 0.95 s/1M/7-digits on 14-core.
- **Yokota & Barba** — task-based FMM, AVX kernels, low-accuracy optimization
  (thesis T2R2 / exafmm-dev lineage).
- **Taichi / Mukherjee et al. 2021** — ApJ 922, arXiv. Collisional FMM with solid
  harmonics + rotation M2L + error-controlled MAC; >10× faster than NBODY6++GPU.
- **Lashuk et al. 2013** — "A massively parallel adaptive FMM on heterogeneous
  architectures", arXiv. MPI+OpenMP+CUDA; GPU 32× single CPU core; q-tuning synergy.

## GPU / M2L

- **Kohnke et al. 2021** — "A CUDA fast multipole method with highly efficient M2L
  far field evaluation", IJHPCA (mpinat.mpg.de PDF). Three M2L schemes; symmetry
  exploitation → compute-bound M2L. **Primary GPU reference.**
- **GROMACS GPU FMM** — PMC7660746 (pmc.ncbi.nlm.nih.gov). FMM vs PME; p=8 matches
  PME accuracy; inhomogeneous systems.
- **Darve 2010** — CRAS Mecanique (comptes-rendus PDF). M2L blocking GPU1–GPU4,
  arithmetic intensity analysis.
- **Goude & Engblom 2012** — "Adaptive fast multipole methods on the GPU",
  J. Supercomputing. All-steps-on-GPU adaptive FMM; 11× vs SSE CPU.
- **Bonsai / Bedorf et al.** — GPU treecode, <50 ms per 10⁶ particles @ 3 digits,
  branchless warp tricks.

## Low-level / SIMD

- **Phantom-GRAPE** — arXiv:1203.4037. AVX force library, 2×10⁹ interactions/s/core,
  rsqrt+NR.
- **AVX-512 N-body optimization** — ResearchGate 341347623. SSE/AVX/AVX-512
  speedups, blocking.
- **tycho/nbody** — github.com/tycho/nbody. SoA vs AoS benchmarks.
- **SimdNBodyKernels** — github.com/markstock/SimdNBodyKernels. Ready SIMD
  kernels, ispc.
- **Fast inverse square root** — en.wikipedia.org/wiki/Fast_inverse_square_root +
  Lomont analysis. Hardware rsqrt supersedes it.

## Kernel-independent FMM

- **Ying, Biros, Zorin 2004** — KIFMM (kernel-independent, uses equivalent
  densities / Chebyshev).
- **Fong & Darve 2009** — "The black-box fast multipole method", J. Comput. Phys.
  228:8712.
- **Messner et al. 2012** — "Optimized M2L kernels for the Chebyshev interpolation
  based FMM" (hal-00746089). 16 operators instead of 316, IAblk variant.
- **ScalFMM** — people.bordeaux.inria.fr/coulaud/Softwares/scalFMM.html.

## Precision / mixed precision

- **GENGA (Brasser et al. 2023)** — A&A 678. FP32 long-range forces: 3–4× faster,
  outcomes statistically identical.
- **Sapporo / Sapporo2** — arXiv:1510.04068. Double-single precision emulation,
  GRAPE-compatible; energy error ~10⁻¹² DP vs ~10⁻¹⁰ DS.
- **HiGPUs** — arXiv:1304.1966. FP32/FP64 hybrid force kernel; DS pitfalls with
  Aarseth criterion.

## Energy efficiency

- **SuperMUC-NG Phase 2** — arXiv:2606.23265. GPU offload 4–12× throughput,
  up to 15× energy efficiency; occupancy sensitivity.
- **GPU power capping (GEMM)** — inria.hal.science/hal-04883872. 55–70% TDP
  optimal for compute-bound; CPU capping +21% no perf loss.
- **Frontier power/frequency capping** — OSTI 3005429. Frequency capping
  +13.2% efficiency, no perf loss.
- **DVFS evaluation** — arXiv:1703.02788. Memory-bound → lower clock saves energy.
- **Zeus** — NSDI'23. Power-limit + batch-size energy optimization (GPU).

## Integrators

- **Makino & Aarseth 1992** — 4th-order Hermite, PASJ 44:141.
- **Nitadori & Aarseth 2012** / 6th-8th order Hermite (S1384107608000171).
- **Pelupessy et al. 2012** — HUAYNO, Hamiltonian-splitting individual timesteps,
  momentum conservation.
- **Rein & Spiegel 2015 / PRS23** — arXiv:2401.02849. Robust timestep criterion;
  IAS15.

## Tree/partitioning

- **Teng 1998** — FMM communication-graph bisector O(N^{2/3}) — the scaling
  argument for FMM on clusters.
- **Zoltan** (Sandia) — geometric graph partitioning / load balancing.
- **NEMO toolkit / falcON** — reference test models (Plummer, Hernquist).