# 11 — Visual System Roadmap (stars → gas → bodies)

Live track plan for `crates/visual`. Companion to `09-visual-system.md`
(architecture) and `ROADMAP.md` Part 2 (phase gates). Status: V1 star
look landed (2026-09-07).

## S1 — Stars, done

- Cosmic OBAFGKM sampling (`ic.rs`: `SpecClass`, young/old abundance
  tables, M ~76%, O ~0 @100k) with log-uniform Teff/L per class.
- Display decoupled from gravity: dyn masses stay 10:1 (stable);
  spectral looks rank-matched onto mass (zero physics change).
- 7-stop blackbody ramp, log-L brightness, size carries rank
  (`0.18+3.0mag²`), halo/spikes gated to heroes, bloom restrained.
- `teff`/`lum` ride Sim→Static→App; overlay shows live OBAFGKM+G census.
- Probe rigs: `--mode ramp --count K` (per-class color), `--exposure`,
  cluster/sparse field stats.

## S2 — Stars, future (one session)

In priority order:

1. Evolution clock — per-star age + lifetime track. O/B go supernova
   (flash → neutron star / stellar BH marker); G/K → giants → white
   dwarfs. Biggest remaining realism gap.
2. Missing end-states — white dwarfs (faint blue), supergiants Ia/Ib
   for star-forming arms (no Rigel-class yet).
3. Spatial age structure — OB associations (blue clumps on arms), red
   globulars in halo. Cluster seeding in `ic.rs`.
4. Metallicity tint — halo Pop-II redder, arm Pop-I bluer. LUT tweak.
5. Polish (V4) — unresolved binaries (brighter points), Cepheid
   pulsation in shader, zoom-melt to integrated light, select-a-star
   readout (uses plumbed `teff`/`lum`).

## S3 — Gas & dust (next session). Research banked 2026-09-07.

Astrophysics (Lynds 1972 survey of 33 spirals; HII-region refs; CCM
R_V=3.1 extinction; NASA M51/M42/Horsehead/Tarantula frames in
/tmp/opencode/ref_*.jpg):

- HII knots always embedded in / beside heavy obscuration; lanes
  define arms better than knots. Knots sit on inside edge of luminous
  arms, at lane branchings + feather crossings; Sa/Sb central zone
  empty of HII.
- Palette (true-color lean; several refs are multi-λ composites):
  HII salmon-pink (Hα 656nm + Hβ pink shift), OIII teal cores near
  hottest stars, reflection blue (Rayleigh, M45), dust
  brown-black occluder, old-dust faint red/brown veil.
- Extinction: blue blocked hardest (R_V 3.1 diffuse → 5+ dense) =
  dim + redden. V1 = occlusion cue only; per-star reddening is V2
  (needs dust column per star).

Technique ladder (cost ↑): noise-billboard impostors (galaxy scale)
→ 13-quad stars (3DWorld, fly-through) → bounding-sphere raymarch
(Gaia Sky/Duke shaders) → full-res volumetrics (defer: no budget at
100k stars + 60fps on laptop). Gas pass at quarter res like bloom;
stars stay full res. SpaceEngine precedent: sprites + impostors for
real/galaxy-scale, raymarch for procedural close-ups only.

G1 build (cheap, fits current pipeline):

- Emission knots: additive FBM-noise billboards (~200–600), anchored
  to actual O/B positions from census, pink shells + teal cores,
  subtle domain warp for wisps.
- Dust lanes: alpha-blend dark quads on inner arm edge (reuse arm
  phase `2φ−3ln(r+.15)` with offset), brown-black, FBM-eroded;
  drawn after stars = occlusion without depth buffer.
- Gas static analytic (like dark halo), not simulated. No SPH.
- Probe: reuse ramp rig pattern — knot color frames + lane
  occlusion frames.

G2 (later): 13-quad impostors for fly-through, screen-space dust
mask sampled in billboard shader for true per-star reddening,
split-res gas pass, SNR hook into S2 evolution clock.

## S4 — Later (see ROADMAP.md Part 2)

- V1 precision/LOD: camera rebase already in; octree LOD + frustum
  culling for >10M bodies.
- V2 bodies: sphere impostors → mesh terrain, planet shaders.
- V3 zero-copy CUDA↔Vulkan; V4 sky field, UI polish, energy report.
