# 06 — Time integration

The force solver (FMM) is the expensive part; the integrator decides how often
you pay for it.

## 1. Options

| Scheme | Order | Collisional? | Notes |
|---|---|---|---|
| Leapfrog / KDK | 2nd | no | symplectic, cheap, fixed global step; for collisionless dynamics |
| Hermite (4th, Makino–Aarseth 1992) | 4th | yes | computes jerk directly; ~2× timestep of Aarseth at same accuracy for ~<2× cost → standard choice |
| Hermite 6th/8th (Nitadori–Aarseth) | 6th/8th | yes | only ~2× flops of 4th order; better for high accuracy; more parallelism |
| Hamiltonian-splitting individual timesteps (HUAYNO) | 2nd base | both | momentum + angular momentum conserved to machine precision; symplectic structure preserved |
| IAS15 (REBOUND) | 15th | both | Gauss–Radau, adaptive, near machine precision; not symplectic but converges so tightly it behaves so |

## 2. Timestep strategy

- **Block timesteps** (power-of-2 hierarchy, `Δt = 2^k · Δt_min`): groups
  particles by timestep → efficient parallel force evaluation; the standard for
  both collisional (Hermite) and collisionless (KDK) codes.
- **Individual timesteps** concentrate work where dynamics are fast; the force
  must be recomputed only for active particles + neighbors.
- **Criterion choice matters for precision:** the classic Aarseth criterion
  (5th derivative) is fragile under low-precision force evaluation and can
  produce spurious tiny steps (HiGPUs observed this in FP32). The **PRS23
  criterion** (uses 2nd–4th derivatives) is robust to rounding and resolves
  close encounters 10⁸× further from the coordinate origin. **Recommended.**

## 3. Interaction with FMM

- For collisionless runs: FMM + KDK leapfrog (momentum-conserving if mutual
  FMM — falcON/Dehnen gives Newton's third law by construction).
- For collisional runs: FMM + Hermite 4/6/8 with block steps (Taichi model).
  Taichi demonstrates FMM can be as accurate as direct summation for N>10⁴,
  matching NBODY6++GPU dynamics at >10× speed.
- FMM efficiency is higher when many particles share a block (force eval done
  once per block, not per particle).
- Predictor stage: positions of all particles needed at the active block's time
  → predict via Taylor expansion (cheap), evaluate forces with FMM, correct.

## 4. Regularization for close encounters (optional but important for collisional)

- Kustaanheimo–Stiefel (KS) regularization handles binaries/close encounters —
  without it, individual timesteps collapse to machine epsilon. FMM collisional
  codes (Taichi) note this as their current main limitation; a hybrid
  "direct for small-N subsystems + FMM for the field" (Ahmad–Cohen scheme) is
  the documented path.

## 5. Energy conservation targets (from literature)

- FMM (Taichi): relative energy error < 10⁻⁴ at core collapse; 1 part in 10¹²
  with tight p, ε.
- Momentum: exact (machine precision) when mutual FMM is used.
- Angular momentum: sensitive to FP32 forces (~2 orders worse); still
  acceptable per GENGA.