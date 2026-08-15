# 01 — Algorithms: FMM and its variants

## 1. Problem statement

Compute the mutual gravitational force between N bodies:

```
a_i = Σ_{j≠i} G m_j (x_j - x_i) / |x_j - x_i|³
```

Direct summation is O(N²) and dominates everything for N ≳ 10³.

## 2. Algorithm landscape

### 2.1 Barnes–Hut treecode — O(N log N)

- Recursive octree of spatial cells; monopole/quadrupole moments per cell.
- Per target, tree walk using opening-angle criterion `θ = D/r`.
- Good at low accuracy (`ε ≈ 10⁻³`), simple to implement.
- **Downside:** each target traverses its own path; no cell→cell reuse.

### 2.2 Fast Multipole Method — O(N)

- (Greengard & Rokhlin 1987; Cheng, Greengard & Rokhlin 1999.)
- Same octree, but *sources AND sinks are both grouped into cells*.
- Well-separated cell pairs interact via multipole→local (M2L) expansions;
  near neighbors interact directly (P2P).
- Five operators:
  - **P2M** particle→multipole (upward)
  - **M2M** multipole→multipole (shift up the tree)
  - **M2L** multipole→local (the expensive far-field step)
  - **L2L** local→local (shift down the tree)
  - **L2P** local→particle (evaluate) + **P2P** direct near-field
- Complexity O(N), constant ~p² or p⁴ depending on M2L implementation.

### 2.3 When each wins (empirical, CMU DIMACS + A&A 2011)

| Regime | Winner |
|---|---|
| N < 10⁴, any accuracy | Direct summation (small) |
| ε ≈ 10⁻³ (low accuracy), any N | Barnes–Hut treecode (2–4× faster) |
| ε ≲ 10⁻⁵, N > 10⁴ | **FMM** |
| Very large N, high accuracy | **FMM**, increasingly dominant |

FMM becomes effective only for N ≳ 10⁴ (tree-building overhead). Below that,
GPU *direct* summation is still the right tool.

## 3. FMM variants to choose between

### 3.1 Classical spherical-harmonics FMM (Greengard–Rokhlin, Cheng 1999)

- Multipole expansion in spherical harmonics.
- **M2L via rotation:** rotate source moments to align z-axis with the
  interaction direction → translation → rotate back. Cost **O(p³)** instead of
  naive **O(p⁴)**. (Cheng 1999, Dehnen 2014, used by Taichi.)
- Solid harmonics (traceless Cartesian tensors) are the recommended basis
  (Dehnen 2014 Table 3; Mukherjee/Taichi 2021).
- Precomputable rotation/swapping matrices (Taichi saves them for p ≤ 30).

### 3.2 falcON / Dehnen dual-tree-walk FMM — RECOMMENDED baseline

- Cartesian Taylor expansion of fixed order p, accuracy controlled by a
  multipole-acceptance criterion (MAC) rather than fixed interaction lists.
- **Mutual cell→cell interactions**: A→B and B→A computed simultaneously →
  Newton's third law satisfied by construction → momentum exactly conserved.
- Dual tree walk replaces rigid interaction lists with a *dynamic* acceptance
  criterion based on error estimates per interaction.
- Empirically fastest serial formulation (A&A 2011); scaling N^0.87
  (sub-linear!) at ε ~ 10⁻⁷ (Dehnen 2014).
- Error-controlled MAC (Dehnen 2014; Taichi 2021): balance absolute errors of
  individual interactions using the cells' multipole power `P_{n,A}` and
  smallest-enclosing-sphere radii. Only two free parameters: expansion order p
  and target relative error ε.

### 3.3 Kernel-independent / black-box FMM (KIFMM, bbFMM)

- No analytic expansion needed — works for any kernel evaluated numerically.
- Chebyshev interpolation of the far field + SVD compression of M2L
  (Fong & Darve 2009; Messner 2012).
- M2L becomes dense mat-vec, amenable to BLAS. Symmetry: only **16 unique
  operators** instead of 316 (IAblk variant).
- Useful if we ever want non-1/r kernels (softened Plummer, Yukawa). More
  flops than harmonic FMM for the same accuracy.

### 3.4 Plane-wave FMM (Greengard–Rokhlin 1997)

- Diagonal M2L via exponential plane-wave expansions → O(p² log p) cost.
- Best absolute far-field speed at high p, but complex to implement and
  parallelize; accuracy coupling is trickier.

### 3.5 Recommendation

**Primary: Dehnen-style dual-tree-walk FMM with Cartesian/solid-harmonic
expansion, rotation-accelerated M2L (O(p³)), error-controlled MAC, and mutual
interactions (momentum conservation).** This is the architecture used by the
fastest published serial codes (falcON, Taichi) and it degrades gracefully:
same tree can run in low-p "treecode mode" for cheap timesteps.

## 4. Operator cost structure (where the time goes)

| Operator | Relative cost | Optimization lever |
|---|---|---|
| M2L | ~94× M2M+L2L combined (PBC, 189 neighbors/box) | rotation O(p³), symmetry, BLAS, plane waves |
| P2P (near field) | second biggest | SIMD vectorization (321 GFLOPS demonstrated), cache blocking |
| P2M / M2M / L2L / L2P | small | negligible if M2L is fixed |

P2P and M2L must be tuned in concert: larger leaf size q shrinks the tree
(memory-bound steps) but inflates P2P and M2L flops. **Faster hardware allows
a larger optimal q** — algorithmic tuning and hardware speed are coupled.

## 5. Key tuning parameters

- **p** — expansion order (accuracy). Optimal p depends on target ε
  (Dehnen 2014: p≈10 optimal for ε=10^-6.25; p≥16 for ε<10^-8).
- **q / N_leaf** — max particles per leaf (32–64 typical).
- **MAC / opening criterion** — θ or error-based (prefer error-based).
- **Expansion centers** — center-of-mass makes dipole vanish (cheaper);
  smallest-enclosing-sphere minimizes error (Dehnen 2014). Use CoM for the
  low-accuracy mode, sphere-center for high accuracy.

## 6. Practical performance references

- **exafmm-t**: 1M particles, 0.95 s @ 7 digits on 14-core i9-7940X
  (OpenMP + AVX-512 + PVFMM M2L cache scheme).
- **Taichi**: >10× faster than NBODY6++GPU for 10⁶ stars; energy conserved to
  1 part in 10¹² at p, ε tuned.
- **GPU FMM (GROMACS/ScaFaCoS)**: ~1/3 PME speed on small homogeneous systems,
  >2× PME on large inhomogeneous (multidroplet aerosol).