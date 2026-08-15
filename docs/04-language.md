# 04 — Language and toolchain

## 1. The question: Rust, C++, or Fortran?

Data points:

| Study | Result |
|---|---|
| NPB-Rust (arXiv 2502.15536) | Rust sequential 1.23% slower than Fortran, **5.59% faster than C++**; Rayon slower than OpenMP in irregular cases |
| Rewrite it in Rust (arXiv 2410.19146) | CBET physics code: Rust 1.6×–5.6× faster than C++ (clang), fewer cache accesses, 4.2× fewer branches |
| 1D heat / stencil comparison | Rust, C++, Chapel, Charm++, HPX are the top tier, statistically similar |
| SC24 HPCCG Rust | approaches C++ and Kokkos in scaling and performance |
| Rust GPU ecosystem | CUDA bindings immature; no production-grade CUDA kernel story |

### Verdict for this project

- **CPU core:** Rust is the strongest choice — equal or better performance,
  memory safety, no runtime cost, excellent concurrency ergonomics (Rayon for
  shared memory; `rsmpi`/`mpi` crate for distributed). Bounds-check overhead is
  real in hot loops but removable via `unsafe`/iteration patterns (56.9% in the
  worst NPB kernel).
- **GPU kernels:** CUDA is a C++ ecosystem. **Recommended split:** Rust host
  + C/C++ CUDA kernels exposed via FFI (`#[no_mangle] extern "C"`). The Rust
  side owns tree/walk/integration; the CUDA side owns P2P/M2L kernels.
- Keep the door open: `-Ctarget-cpu=native`, `-C opt-level=3`, LTO.

## 2. Compiler flags (CPU)

```
# Rust
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
target-cpu = "native"

# C/C++ (GCC/Clang)
-O3 -march=native -ffast-math -funroll-loops -flto

# Intel oneAPI/ICC (often fastest on these loops)
-O3 -xHOST -qopt-report -ffast-math -qopt-mem-layout-trans=4
```

Note: `-ffast-math` is required for FMA contraction and rsqrt substitution, but
it changes FP semantics — audit with a golden test (compare against a
`-O0 -fno-fast-math` reference for the first N steps).

## 3. Ecosystem / libraries to reuse (don't reinvent)

| Need | Library |
|---|---|
| Reference FMM implementations to study | ExaFMM (C++/Python), PVFMM (C++/MPI), ScalFMM (C++, MPI/OpenMP, bbFMM+harmonic), falcON (C++), GemsFMM (CUDA), Taichi (C++, OpenMP+AVX) |
| BLAS M2L | MKL / OpenBLAS / Eigen (when using matrix-form M2L) |
| Task parallelism | Rayon (Rust), OpenMP tasks / Intel TBB (C++) |
| Distributed | MPI (rsmpi / mpi crate, OpenMPI) |
| Tree/graph partitioning | Zoltan (load balance), Morton/radix sort (hand-rolled, 40 lines) |
| Timestep/integrator reference | REBOUND (IAS15), HUAYNO, NEMO toolkit |
| CGAL smallest-enclosing-sphere | Dehnen 2014 uses it for expansion centers (optional; O(N) approximation is fine) |

## 4. Build & CI

- CMake or Cargo + a CUDA sub-build; one `build.rs` wiring the FFI.
- Benchmark harness from day one: `criterion` (Rust) or `hyperfine`;
  run on a fixed machine, pin frequency, report GFLOPS and energy.
- Golden tests: energy conservation, momentum conservation, Plummer/Hernquist
  test models from NEMO.

## 5. Energy of language choice

Language runtime is a negligible fraction of total energy at scale; the big
energy levers are algorithm, precision, and hardware selection (see 05). Rust's
lower branch/miss and cache-access counts (measured in CBET) do translate to
marginally lower per-iteration energy, but do not design around it.