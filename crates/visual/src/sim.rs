use fmm::fmm::Fmm;
use fmm::models::Particle;
use std::collections::VecDeque;

pub trait Physics {
    fn accel(&mut self, pos: &[[f64; 3]], mass: &[f64]) -> Vec<[f64; 3]>;
}

/// Unsoftened Newton has no ceiling: one deep encounter kicks Δv ~ 10³,
/// the drift flings the body out, the tree box balloons, and Morton
/// resolution at the nucleus collapses (overfull leaves). Direct carries
/// its own softening; the FMM backends cap |a| instead — same role,
/// zero library churn, science gates untouched.
pub const ACCEL_CAP: f64 = 50.0;

/// Integrator guardrails (Power et al. 2003): dt << local orbital time,
/// i.e. dt <= ETA * sqrt(EPS_LEN / a_max). The speed slider requests dt;
/// the step clamps to this — speed saturates instead of evaporating.
pub const DT_ETA: f64 = 0.25;
pub const DT_EPS_LEN: f64 = 0.02;

pub fn cap_accel(a: &mut [[f64; 3]]) {
    for v in a.iter_mut() {
        let s2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if !s2.is_finite() {
            *v = [0.0; 3];
        } else if s2 > ACCEL_CAP * ACCEL_CAP {
            let s = ACCEL_CAP / s2.sqrt();
            v[0] *= s;
            v[1] *= s;
            v[2] *= s;
        }
    }
}

pub struct Direct {
    pub soft2: f64,
}

impl Physics for Direct {
    fn accel(&mut self, pos: &[[f64; 3]], mass: &[f64]) -> Vec<[f64; 3]> {
        let n = pos.len();
        let nt = thread_count();
        if n < 600 || nt < 2 {
            let mut a = vec![[0.0; 3]; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = pos[j][0] - pos[i][0];
                    let dy = pos[j][1] - pos[i][1];
                    let dz = pos[j][2] - pos[i][2];
                    let r2 = dx * dx + dy * dy + dz * dz + self.soft2;
                    let inv = 1.0 / r2.sqrt();
                    let inv3 = inv * inv * inv;
                    let si = mass[j] * inv3;
                    let sj = mass[i] * inv3;
                    a[i][0] += si * dx;
                    a[i][1] += si * dy;
                    a[i][2] += si * dz;
                    a[j][0] -= sj * dx;
                    a[j][1] -= sj * dy;
                    a[j][2] -= sj * dz;
                }
            }
            return a;
        }
        let mut a = vec![[0.0; 3]; n];
        let chunk = n.div_ceil(nt);
        let s2 = self.soft2;
        std::thread::scope(|scope| {
            for (slot, out) in a.chunks_mut(chunk).enumerate() {
                let start = slot * chunk;
                scope.spawn(move || {
                    for (k, ak) in out.iter_mut().enumerate() {
                        let i = start + k;
                        let px = pos[i][0];
                        let py = pos[i][1];
                        let pz = pos[i][2];
                        let mut ax = 0.0;
                        let mut ay = 0.0;
                        let mut az = 0.0;
                        for j in 0..n {
                            if j == i {
                                continue;
                            }
                            let dx = pos[j][0] - px;
                            let dy = pos[j][1] - py;
                            let dz = pos[j][2] - pz;
                            let r2 = dx * dx + dy * dy + dz * dz + s2;
                            let inv = 1.0 / r2.sqrt();
                            let s = mass[j] * inv * inv * inv;
                            ax += s * dx;
                            ay += s * dy;
                            az += s * dz;
                        }
                        *ak = [ax, ay, az];
                    }
                });
            }
        });
        a
    }
}

pub struct FmmBackend {
    pub p: usize,
    pub eps: f64,
    pub n_leaf: usize,
    fmm: Option<Fmm>,
    parts: Vec<Particle>,
}

impl FmmBackend {
    pub fn new(p: usize, eps: f64, n_leaf: usize) -> Self {
        Self {
            p,
            eps,
            n_leaf,
            fmm: None,
            parts: Vec::new(),
        }
    }
}

impl Physics for FmmBackend {
    fn accel(&mut self, pos: &[[f64; 3]], mass: &[f64]) -> Vec<[f64; 3]> {
        let n = pos.len();
        self.parts.clear();
        self.parts.reserve(n);
        for i in 0..n {
            self.parts.push(Particle {
                mass: mass[i],
                pos: pos[i],
            });
        }
        let mut out = match &mut self.fmm {
            Some(f) => {
                f.refresh(&self.parts);
                f.run_threaded(&self.parts, thread_count()).accel
            }
            None => {
                let f = Fmm::new(&self.parts, self.p, self.eps, self.n_leaf);
                let out = f.run_threaded(&self.parts, thread_count()).accel;
                self.fmm = Some(f);
                out
            }
        };
        cap_accel(&mut out);
        out
    }
}

fn thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(12))
        .unwrap_or(1)
}

pub struct Sim {
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub mass: Vec<f64>,
    pub temp: Vec<f32>,
    pub mag: Vec<f32>,
    // NOTE: no teff/lum vectors. They cost 8B/star (8MB at 1M) and were only
    // used once for the UI census. Census is computed at init from `Star`
    // and shipped in `Static`; render keeps temp/mag only.
    pub census: [usize; 8],
    pub gas: Vec<crate::ic::Gas>,
    pub gas_om: f32,
    pub prev: Vec<[f64; 3]>,
    pub prev_vel: Vec<[f64; 3]>,
    pub time: f64,
    pub dt: f64,
    pub steps: u64,
    backend: Backend,
    acc: Option<Vec<[f64; 3]>>,
    vmax: Option<f64>,
    halo: Option<crate::ic::Halo>,
    /// Live BH point masses for gas: (body index, mass). Indices fixed
    /// (masses never change); positions read live from self.pos.
    bh: Vec<(usize, f64)>,
    /// Frozen disk+bulge curve (total minus halo minus BH).
    gas_tab: crate::ic::GasTable,
    /// M2 shock trough follows analytic arms (spiral/BH presets only).
    gas_arms: bool,
    /// M3 feedback cavity radius: wind expels knots inside it.
    r_clear: f64,
    /// ISCO radius: knots inside plunge (accreted, size 0). BH 0.06,
    /// spiral/merger 0.04, cluster 0 (no gas anyway). Kills the central
    /// clamp pile-up: gas heats at the rim then disappears like infall.
    r_isco: f64,
    /// Alive-knot target for fountain respawn (accreted knots recycle
    /// outside so the disk never drains).
    gas_target: usize,
    /// M4 turbulence source (fixed seed: trajectories stay deterministic).
    grng: fmm::models::Rng,
}

/// BH softening length² + gas speed cap: knots are tracers, never slingshots.
const GAS_SOFT2: f64 = 1e-4;
const GAS_VMAX: f64 = 4.0;
/// M2 shock trough: width (phase rad), pull strength, brake core/amount.
/// Brake blends the relax target toward pattern speed inside the lane:
/// gas lingers (jam) but never stops (vc > Ωp·R inside CR always wins).
const ARM_W: f64 = 0.5;
const ARM_A: f64 = 0.15;
const BRAKE_W: f64 = 0.15;
const BRAKE_K: f64 = 0.9;
/// M3 wind scale; M4 turbulence kick factor (equilibrium
/// ~8% of circular against relaxation) + thin-disk vertical restoring.
const WIND_A: f64 = 2.0;
const TURB_C: f64 = 0.07;
const SETTLE_K: f64 = 0.5;
/// Conserved-gas coalescence (supernova swell, not pop): mass ~ size³.
/// Mergers move VOLUME gradually over ~5 sweeps (~40 steps) so the render
/// swells smoothly + flashes white-blue; radii grow as cbrt, so clouds get
/// visibly bigger and STAY bigger. (Diameter-add was tried: it doubles area
/// per merger, which forced a dispersal drain that evaporated every giant —
/// "blobs stay blob". Volume-add conserves mass, so no drain is needed.)
/// Same-class pairs in visual contact transfer; anything else in contact
/// (other class, or big at cap) pushes apart instead — no two clouds share
/// space, no mass deleted. Loser keeps a dead slot (size 0) so snap indices
/// stay stable for the render interpolator; fountain respawn reuses them.
/// STRIDE desyncs the sweep across steps (no global 8-step pulse).
const MERGE_STRIDE: u64 = 8;
const MERGE_XFER: f64 = 0.3;
const MIN_ALIVE: f64 = 0.008;
/// Contact at half the summed sizes: gas quads render ~size wide, so this
/// is visual touch. (Old 0.25 needed 2x visual overlap before anything
/// happened — blobs sat inside each other untouched.)
const TOUCH_F: f64 = 0.5;
/// Overlap fraction pushed apart per sweep + approach-velocity damping.
const EXCL_F: f64 = 0.25;
const EXCL_DAMP: f64 = 0.2;
const FLASH_TAU: f64 = 0.35;
/// Condensation only: sub-equilibrium knots grow toward class equilibrium
/// (fresh + respawned knots settle in). Nothing evaporates — merged giants
/// persist; the budget stays bounded by conserved volume + caps, not a drain.
const DISP_TAU: f64 = 0.5;
const EM_MAX: f64 = 0.18;
const DUST_MAX: f64 = 0.16;

fn gas_eq(kind: f32) -> f64 {
    if kind < 2.5 {
        0.025
    } else {
        0.05
    }
}

fn gas_class(kind: f32) -> u8 {
    if kind < 2.5 {
        0
    } else {
        1
    }
}

fn gas_cap(kind: f32) -> f64 {
    if kind < 2.5 {
        EM_MAX
    } else {
        DUST_MAX
    }
}

/// One deterministic O(n²) sweep; call every MERGE_EVERY steps.
/// Gradual volume transfer (no pop); pairs the sweep skips for transfer
/// get exclusion instead (never shared space). Strided variant spreads the
/// sweep across steps (outer rows rotate by step): same pairs, same cadence
/// per pair, but no global pulse where the whole disk merges + flashes in
/// sync (the "heartbeat").
/// Full-sweep wrapper: unit tests use this (also silences dead-code in
/// non-test builds, where the app runs strided).
#[allow(dead_code)]
pub fn merge_gas(gas: &mut [crate::ic::Gas]) {
    merge_gas_stride(gas, 0, 1);
}

pub fn merge_gas_stride(gas: &mut [crate::ic::Gas], step: u64, stride: u64) {
    let n = gas.len();
    let mut i = (step % stride.max(1)) as usize;
    while i < n {
        if gas[i].size > 0.0 {
            for j in (i + 1)..n {
                if gas[j].size <= 0.0 {
                    continue;
                }
                let sa = gas[i].size as f64;
                let sb = gas[j].size as f64;
                let dx = gas[i].pos[0] - gas[j].pos[0];
                let dy = gas[i].pos[1] - gas[j].pos[1];
                let dz = gas[i].pos[2] - gas[j].pos[2];
                let touch = (sa + sb) * TOUCH_F;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 >= touch * touch {
                    continue;
                }
                // Bigger VOLUME accretes; ties go to the lower index.
                let va = sa * sa * sa;
                let vb = sb * sb * sb;
                let (big, small, v_big, v_small, s_big, s_small) = if va >= vb {
                    (i, j, va, vb, sa, sb)
                } else {
                    (j, i, vb, va, sb, sa)
                };
                let cap = gas_cap(gas[big].kind);
                let room_v = cap * cap * cap - v_big;
                if gas_class(gas[i].kind) == gas_class(gas[j].kind) && room_v > 1e-12 {
                    // Gradual volume transfer: a fraction of the small knot
                    // per sweep so size/pos glide instead of popping.
                    let mut xfer_v = (v_small * MERGE_XFER).min(room_v);
                    // Final gulp when the remainder is a wisp, or it all
                    // fits: no lingering micro-knot.
                    if v_small - xfer_v < MIN_ALIVE.powi(3)
                        || (v_small <= room_v + 1e-15 && s_small < 0.02)
                    {
                        xfer_v = v_small.min(room_v);
                    }
                    if xfer_v <= 1e-15 {
                        continue;
                    }
                    // Drift the big knot toward the meal by the transferred
                    // volume fraction: smooth glide, never a teleport.
                    let v_big2 = v_big + xfer_v;
                    let v_left = v_small - xfer_v;
                    for d in 0..3 {
                        gas[big].pos[d] =
                            (gas[big].pos[d] * v_big + gas[small].pos[d] * xfer_v) / v_big2;
                        gas[big].vel[d] =
                            (gas[big].vel[d] * v_big + gas[small].vel[d] * xfer_v) / v_big2;
                    }
                    let grew = v_big2.cbrt() - s_big;
                    gas[big].size = v_big2.cbrt() as f32;
                    gas[big].flash = (gas[big].flash + (grew * 8.0) as f32).min(1.0);
                    if v_left < MIN_ALIVE.powi(3) {
                        // Wisp remainder joins the big knot (room checked).
                        gas[big].size = (v_big2 + v_left).cbrt().min(cap) as f32;
                        gas[small].size = 0.0;
                    } else {
                        gas[small].size = v_left.cbrt() as f32;
                    }
                } else {
                    // Exclusion: separate along the line of centers (light
                    // moves more) + damp approach velocity so pairs settle
                    // instead of oscillating. Mass untouched.
                    let dist = d2.sqrt().max(1e-9);
                    let overlap = touch - dist;
                    let ma = va.max(1e-18);
                    let mb = vb.max(1e-18);
                    let push = overlap * EXCL_F;
                    let (nx, ny, nz) = (dx / dist, dy / dist, dz / dist);
                    let wi = mb / (ma + mb);
                    let wj = ma / (ma + mb);
                    gas[i].pos[0] += nx * push * wi;
                    gas[i].pos[1] += ny * push * wi;
                    gas[i].pos[2] += nz * push * wi;
                    gas[j].pos[0] -= nx * push * wj;
                    gas[j].pos[1] -= ny * push * wj;
                    let rvn = (gas[i].vel[0] - gas[j].vel[0]) * nx
                        + (gas[i].vel[1] - gas[j].vel[1]) * ny
                        + (gas[i].vel[2] - gas[j].vel[2]) * nz;
                    if rvn < 0.0 {
                        gas[i].vel[0] -= nx * rvn * EXCL_DAMP * wi;
                        gas[i].vel[1] -= ny * rvn * EXCL_DAMP * wi;
                        gas[i].vel[2] -= nz * rvn * EXCL_DAMP * wi;
                        gas[j].vel[0] += nx * rvn * EXCL_DAMP * wj;
                        gas[j].vel[1] += ny * rvn * EXCL_DAMP * wj;
                        gas[j].vel[2] += nz * rvn * EXCL_DAMP * wj;
                    }
                }
            }
        }
        i += stride.max(1) as usize;
    }
}

fn sstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a).max(1e-18)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn wrap_pi(x: f64) -> f64 {
    let t = std::f64::consts::TAU;
    ((x + std::f64::consts::PI) % t + t) % t - std::f64::consts::PI
}

enum Backend {
    Direct(Direct),
    Fmm(FmmBackend),
    Gpu(crate::gpufmm::GpuBackend),
}

impl Sim {
    pub fn new(
        n: usize,
        dt: f64,
        use_fmm: bool,
        gpu: Option<crate::gpufmm::GpuBackend>,
        preset: crate::ic::Preset,
    ) -> Self {
        let (stars, gas, halo) = crate::ic::generate(preset, n, 42);
        let gas_om = crate::ic::pattern_speed(preset);
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut temp = Vec::with_capacity(n);
        let mut mag = Vec::with_capacity(n);
        let mut census = [0usize; 8];
        for st in &stars {
            pos.push(st.pos);
            vel.push(st.vel);
            mass.push(st.mass);
            temp.push(st.temp);
            mag.push(st.mag);
            // Census folded in here: no resident teff/lum vectors (8MB at 1M).
            if let Some(k) = crate::ic::class_index(st.teff, st.lum) {
                census[k] += 1;
            }
        }
        // `stars` (72B/star transient) freed here, before tree/backend allocs.
        remove_com(&mut pos, &mut vel, &mass);
        if preset == crate::ic::Preset::Cluster && n <= 5000 {
            rescale_virial(&mut vel, &pos, &mass);
        }
        let prev = pos.clone();
        let prev_vel = vel.clone();
        let backend = if let Some(g) = gpu {
            Backend::Gpu(g)
        } else if use_fmm || n > 1500 {
            if n > 10000 {
                Backend::Fmm(FmmBackend::new(4, 1e-2, 64))
            } else {
                Backend::Fmm(FmmBackend::new(6, 1e-4, 16))
            }
        } else {
            Backend::Direct(Direct { soft2: 1e-6 })
        };
        let vmax = if n > 20000 { Some(5.0) } else { None };
        // BH bodies sit last in each generator (spiral_with / BH arm push
        // theirs after disk+bulge; merger concatenates two such halves).
        let bh: Vec<(usize, f64)> = match preset {
            crate::ic::Preset::Cluster => Vec::new(),
            crate::ic::Preset::Spiral | crate::ic::Preset::BlackHole => {
                if n > 8 {
                    vec![(n - 1, mass[n - 1])]
                } else {
                    Vec::new()
                }
            }
            crate::ic::Preset::Merger => {
                if n > 16 {
                    vec![(n / 2 - 1, mass[n / 2 - 1]), (n - 1, mass[n - 1])]
                } else {
                    Vec::new()
                }
            }
        };
        let gas_tab = crate::ic::gas_disk_table(preset);
        let gas_arms = matches!(
            preset,
            crate::ic::Preset::Spiral | crate::ic::Preset::BlackHole
        );
        let r_clear = match preset {
            crate::ic::Preset::Spiral => 0.07,
            crate::ic::Preset::BlackHole => 0.15,
            _ => 0.0,
        };
        let r_isco = match preset {
            crate::ic::Preset::BlackHole => 0.06,
            crate::ic::Preset::Spiral | crate::ic::Preset::Merger => 0.04,
            crate::ic::Preset::Cluster => 0.0,
        };
        let gas_target = gas.len();
        Self {
            pos,
            vel,
            mass,
            temp,
            mag,
            prev,
            prev_vel,
            time: 0.0,
            dt,
            steps: 0,
            backend,
            acc: None,
            vmax,
            halo,
            census,
            gas,
            gas_om,
            bh,
            gas_tab,
            gas_arms,
            r_clear,
            r_isco,
            gas_target,
            grng: fmm::models::Rng::new(1234),
        }
    }

    pub fn n(&self) -> usize {
        self.pos.len()
    }

    fn add_halo(&self, a: &mut [[f64; 3]]) {
        if let Some(h) = &self.halo {
            for (v, p) in a.iter_mut().zip(self.pos.iter()) {
                let ha = crate::ic::halo_accel(h, p);
                v[0] += ha[0];
                v[1] += ha[1];
                v[2] += ha[2];
            }
        }
    }

    /// Backend accel with GPU→CPU fallback: if the pair/cell counts need
    /// buffers past the device limits, swap in the CPU FMM backend
    /// permanently and compute fresh. No garbage step: the swap happens
    /// before any accel is consumed.
    fn backend_accel(&mut self) -> Vec<[f64; 3]> {
        let over = match &mut self.backend {
            Backend::Direct(d) => return d.accel(&self.pos, &self.mass),
            Backend::Fmm(f) => return f.accel(&self.pos, &self.mass),
            Backend::Gpu(g) => match g.try_accel(&self.pos, &self.mass) {
                Ok(a) => return a,
                Err(e) => e,
            },
        };
        eprintln!("GPU FMM over budget ({over}); falling back to CPU FMM");
        self.backend = Backend::Fmm(FmmBackend::new(4, 1e-2, 64));
        match &mut self.backend {
            Backend::Fmm(f) => f.accel(&self.pos, &self.mass),
            _ => unreachable!(),
        }
    }

    pub fn step(&mut self) {
        self.prev.clone_from(&self.pos);
        self.prev_vel.clone_from(&self.vel);
        let a0 = match self.acc.take() {
            // Cache already carries halo (applied when stored below).
            Some(a) => a,
            None => {
                let mut a = self.backend_accel();
                self.add_halo(&mut a);
                a
            }
        };
        // Adaptive clamp: requested dt (speed slider) can exceed stability
        // near the center; shrink to the eta criterion instead of blowing up.
        let mut amax: f64 = 1e-9;
        for v in &a0 {
            amax = amax.max(v[0].abs()).max(v[1].abs()).max(v[2].abs());
        }
        let dt = self.dt.min(DT_ETA * (DT_EPS_LEN / amax).sqrt());
        for i in 0..self.pos.len() {
            self.vel[i][0] += a0[i][0] * dt * 0.5;
            self.vel[i][1] += a0[i][1] * dt * 0.5;
            self.vel[i][2] += a0[i][2] * dt * 0.5;
            self.pos[i][0] += self.vel[i][0] * dt;
            self.pos[i][1] += self.vel[i][1] * dt;
            self.pos[i][2] += self.vel[i][2] * dt;
        }
        let mut a1 = self.backend_accel();
        self.add_halo(&mut a1);
        for i in 0..self.pos.len() {
            self.vel[i][0] += a1[i][0] * dt * 0.5;
            self.vel[i][1] += a1[i][1] * dt * 0.5;
            self.vel[i][2] += a1[i][2] * dt * 0.5;
        }
        if let Some(vmax) = self.vmax {
            let v2max = vmax * vmax;
            for v in self.vel.iter_mut() {
                let s2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if s2 > v2max {
                    let s = vmax / s2.sqrt();
                    v[0] *= s;
                    v[1] *= s;
                    v[2] *= s;
                }
            }
        }
        self.step_gas(dt);
        // Strided sweep every step (1/8 of rows): same per-pair cadence as
        // the old every-8th global sweep, but events spread uniformly in
        // time instead of one synchronized whole-disk pulse. Respawn
        // trickles (cap 2/step ≈ old 8/sweep average) for the same reason.
        merge_gas_stride(&mut self.gas, self.steps, MERGE_STRIDE);
        self.respawn_gas(2);
        self.acc = Some(a1);
        self.time += dt;
        self.steps += 1;
    }

    /// M1 test-particle gas: live halo + live BH point masses + frozen
    /// disk table, relaxed toward circular on 0.3 orbits (collisional
    /// stand-in: shear + BH whirl emerge, slingshots impossible).
    /// BH realism: ISCO plunge recycles to the disk (size intact, flash up),
    /// wind is a smooth bipolar fountain (poles vent, plane leaks through
    /// as nuclear spirals), flash carries merge glow + tidal heat at rim.
    /// Gas mass is conserved everywhere: merges move volume, exclusion
    /// only pushes, nothing evaporates or shreds.
    fn step_gas(&mut self, dt: f64) {
        if self.gas.is_empty() {
            return;
        }
        let pos = &self.pos;
        let halo = &self.halo;
        let bh = &self.bh;
        let tab = &self.gas_tab;
        let arms = &self.gas_arms;
        let gas_om = &self.gas_om;
        let time = &self.time;
        let rc = self.r_clear;
        let risco = self.r_isco;
        // Fountain snapshot for plunge recycle: live sibling positions.
        // Owned Vec, so the &mut loop below stays borrow-clean.
        let siblings: Vec<[f64; 3]> = if risco > 0.0 {
            self.gas
                .iter()
                .filter(|g| g.size > 0.0)
                .map(|g| g.pos)
                .collect()
        } else {
            Vec::new()
        };
        let grng = &mut self.grng;
        for g in self.gas.iter_mut() {
            if g.size <= 0.0 {
                continue;
            }
            let p = g.pos;
            let r3raw = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            // Nearest-BH distance drives ISCO recycle + tidal heat floor
            // (merger BHs live off-origin; single-BH presets sit ~at origin).
            let mut d_bh = f64::INFINITY;
            for &(bi, _) in bh.iter() {
                let q = pos[bi];
                let dx = p[0] - q[0];
                let dy = p[1] - q[1];
                let dz = p[2] - q[2];
                d_bh = d_bh.min((dx * dx + dy * dy + dz * dz).sqrt());
            }
            if !d_bh.is_finite() {
                d_bh = r3raw;
            }
            // Plunge recycle (fountain): no stable orbit inside ISCO, but
            // gas is never deleted — the knot rejoins the disk at a live
            // sibling with its size intact and a feedback flash. Kills the
            // central clamp pile-up without vanish-on-arrival.
            if risco > 0.0 && d_bh < risco {
                if !siblings.is_empty() {
                    let k = (grng.unit() * siblings.len() as f64) as usize % siblings.len();
                    let sp = siblings[k];
                    g.pos = [
                        sp[0] + 0.15 * crate::ic::gauss(grng),
                        sp[1] + 0.15 * crate::ic::gauss(grng),
                        sp[2] * 0.7 + 0.03 * crate::ic::gauss(grng),
                    ];
                    for d in 0..3 {
                        g.vel[d] *= 0.3;
                    }
                    g.flash = 1.0;
                }
                continue;
            }
            // Condensation only: sub-equilibrium knots grow toward eq.
            // Merged giants persist (no evaporation, no tidal shredding —
            // both deleted mass); caps + conserved volume bound the budget.
            {
                let eq = gas_eq(g.kind);
                let s = g.size as f64;
                if s < eq {
                    let k = (dt / DISP_TAU).min(1.0);
                    g.size = (s + (eq - s) * k) as f32;
                }
            }
            let r3 = r3raw.max(0.03);
            let rr = (p[0] * p[0] + p[1] * p[1]).sqrt().max(0.03);
            let mut ax = 0.0;
            let mut ay = 0.0;
            let mut az = 0.0;
            if let Some(h) = halo {
                let ha = crate::ic::halo_accel(h, &p);
                ax += ha[0];
                ay += ha[1];
                az += ha[2];
            }
            let mut vbh2 = 0.0;
            for &(bi, bm) in bh.iter() {
                let q = pos[bi];
                let dx = p[0] - q[0];
                let dy = p[1] - q[1];
                let dz = p[2] - q[2];
                let s2 = dx * dx + dy * dy + dz * dz + GAS_SOFT2;
                let s = bm / (s2 * s2.sqrt());
                ax -= s * dx;
                ay -= s * dy;
                az -= s * dz;
                vbh2 += bm / s2.sqrt();
            }
            let vd2 = tab.v2(rr);
            let s = vd2 / rr;
            ax -= s * p[0];
            ay -= s * p[1];
            let vh2 = match halo {
                Some(h) => crate::ic::halo_v2(h, r3),
                None => 0.0,
            };
            let vt0 = (vd2 + vh2 + vbh2).sqrt();
            // M3 wind: smooth bipolar fountain. Gaussian profile (no step
            // edge to pile up on) × polar bias (poles vent, disk plane
            // leaks through and feeds nuclear spirals).
            let mut wind_prof = 0.0;
            let mut wind_bip = 1.0;
            if rc > 0.0 {
                let cos_t = (p[2] / r3).abs().clamp(0.0, 1.0);
                wind_bip = 0.25 + 0.75 * cos_t;
                wind_prof = (-0.5 * (r3 / rc).powi(2)).exp();
                let w = WIND_A / rc * wind_prof * wind_bip;
                ax += w * p[0];
                ay += w * p[1];
                az += w * p[2];
            }
            // M4 thin-disk restoring (razor-thin disk pull missing above).
            az -= p[2] * vt0 * vt0 / (rr * rr) * SETTLE_K;
            // M2 spiral shock: trough attraction toward the lane phase
            // (orbit crowding) + traffic-jam brake toward pattern speed +
            // shock dissipation (kills libration so knots settle in lane).
            // Pattern rotates rigidly at gas_om; gas flows through it.
            let mut vt_mul = 1.0;
            let mut vt_add = 0.0;
            let mut shock = 0.0;
            if *arms {
                let omp = *gas_om as f64;
                let phi = p[1].atan2(p[0]);
                let psi = 2.0 * (phi - omp * *time) - 3.0 * (rr + 0.15).ln();
                // Lanes repeat every π (two arms); fold before measuring.
                let d = wrap_pi(2.0 * (psi + 0.55)) / 2.0;
                let gr = -3.0 / (rr + 0.15);
                let gp = 2.0 / rr;
                let gn = (gr * gr + gp * gp).sqrt();
                let pull = (-0.5 * (d / ARM_W).powi(2)).exp() * (d / ARM_W);
                let c = phi.cos();
                let ss = phi.sin();
                ax += -ARM_A * pull * (gr * c - gp * ss) / gn;
                ay += -ARM_A * pull * (gr * ss + gp * c) / gn;
                let b = (-0.5 * (d / BRAKE_W).powi(2)).exp() * BRAKE_K;
                vt_mul = 1.0 - b;
                vt_add = b * omp * rr;
                shock = b;
            }
            let a2 = ax * ax + ay * ay + az * az;
            if !a2.is_finite() {
                ax = 0.0;
                ay = 0.0;
                az = 0.0;
            } else if a2 > ACCEL_CAP * ACCEL_CAP {
                let k = ACCEL_CAP / a2.sqrt();
                ax *= k;
                ay *= k;
                az *= k;
            }
            let vt = (vd2 + vh2 + vbh2).sqrt() * vt_mul + vt_add;
            let sgn = if p[0] * g.vel[1] - p[1] * g.vel[0] >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let tx = -p[1] / rr * sgn;
            let ty = p[0] / rr * sgn;
            let tau = 0.3 * 2.0 * std::f64::consts::PI * rr / vt.max(0.05);
            let k = ((dt / tau) * (1.0 + 20.0 * shock)).min(1.0);
            g.vel[0] += ax * dt + (vt * tx - g.vel[0]) * k;
            g.vel[1] += ay * dt + (vt * ty - g.vel[1]) * k;
            g.vel[2] += az * dt - g.vel[2] * k;
            // M4 turbulence: smooth fountain boost (no 4x step edge to
            // jitter across). Peak ~2.5x at the polar centre, 1x outside.
            let tc = TURB_C * vt * dt.sqrt() * (1.0 + 1.5 * wind_prof * wind_bip);
            g.vel[0] += tc * crate::ic::gauss(grng);
            g.vel[1] += tc * crate::ic::gauss(grng);
            g.vel[2] += tc * crate::ic::gauss(grng);
            let v2 = g.vel[0] * g.vel[0] + g.vel[1] * g.vel[1] + g.vel[2] * g.vel[2];
            if v2 > GAS_VMAX * GAS_VMAX {
                let s = GAS_VMAX / v2.sqrt();
                g.vel[0] *= s;
                g.vel[1] *= s;
                g.vel[2] *= s;
            }
            g.pos[0] += g.vel[0] * dt;
            g.pos[1] += g.vel[1] * dt;
            g.pos[2] += g.vel[2] * dt;
            // Flash: merge glow decays (supernova fade); tidal heat at the
            // rim glides both ways (slow attack ~0.25 sim) so eccentric
            // divers brighten smoothly instead of strobing each periapsis.
            let decayed = g.flash * (-(dt / FLASH_TAU)).exp() as f32;
            let heat_in = risco.max(0.02);
            let heat = (1.0 - sstep(heat_in, 0.35, d_bh) as f32) * 0.85;
            if heat > decayed {
                g.flash = decayed + (heat - decayed) * (dt * 4.0).min(1.0) as f32;
            } else {
                g.flash = decayed;
            }
        }
    }

    /// Fountain respawn: accreted/merged dead slots rain back onto live
    /// gas (sibling positions + jitter) so the disk never drains. Keeps
    /// the dead kind (em/dust balance) but rolls a fresh noise seed.
    /// Capped per call so refills glide in instead of popping.
    fn respawn_gas(&mut self, cap: usize) {
        if self.gas_target == 0 || self.gas.is_empty() {
            return;
        }
        let alive: Vec<usize> = self
            .gas
            .iter()
            .enumerate()
            .filter(|(_, g)| g.size > 0.0)
            .map(|(i, _)| i)
            .collect();
        let mut need = self.gas_target.saturating_sub(alive.len()).min(cap);
        if need == 0 || alive.is_empty() {
            return;
        }
        // Snapshot siblings first (immutable) so the refill loop can hold
        // mutable knot borrows without aliasing.
        let sib_pos: Vec<[f64; 3]> = alive.iter().map(|&i| self.gas[i].pos).collect();
        let sib_vel: Vec<[f64; 3]> = alive.iter().map(|&i| self.gas[i].vel).collect();
        let grng = &mut self.grng;
        for g in self.gas.iter_mut() {
            if need == 0 {
                break;
            }
            if g.size > 0.0 {
                continue;
            }
            let k = (grng.unit() * sib_pos.len() as f64) as usize % sib_pos.len();
            let (sp, sv) = (sib_pos[k], sib_vel[k]);
            // Wide jitter: fresh knots must land outside giants' merge
            // radii, or refill just feeds the nearest monster.
            let j = 0.15;
            g.pos = [
                sp[0] + j * crate::ic::gauss(grng),
                sp[1] + j * crate::ic::gauss(grng),
                sp[2] * 0.7 + 0.03 * crate::ic::gauss(grng),
            ];
            g.vel = [
                sv[0] + 0.05 * crate::ic::gauss(grng),
                sv[1] + 0.05 * crate::ic::gauss(grng),
                0.02 * crate::ic::gauss(grng),
            ];
            g.size = (0.015 + 0.02 * grng.unit()) as f32;
            g.seed = grng.unit() as f32;
            g.flash = 0.0;
            need -= 1;
        }
    }
}

pub struct Snap {
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub gas: Vec<[f64; 3]>,
    pub gas_size: Vec<f32>,
    pub gas_flash: Vec<f32>,
    pub time: f64,
    pub steps: u64,
}

pub struct Static {
    pub temp: Vec<f32>,
    pub mag: Vec<f32>,
    pub census: [usize; 8],
    pub gas: Vec<crate::ic::Gas>,
    pub gas_om: f32,
    pub n: usize,
}

pub struct HistSnap {
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub gas: Vec<[f64; 3]>,
    pub gas_size: Vec<f32>,
    pub gas_flash: Vec<f32>,
    pub time: f64,
    pub steps: u64,
    pub arrival: std::time::Instant,
}

/// Advance a sim-time playout cursor. Occupancy feedback (0.25x..1.5x)
/// keeps bracketing snaps live through bursty arrivals; past the newest
/// snap the cursor glides on ballistic extrapolation (snap velocities, via
/// `bracket`'s extra term) for up to CURSOR_EXTRA_WALL of wall time.
/// That allowance covers arrival gaps from slow GPU-refresh steps at large
/// N: the old hard 2-step ceiling froze motion whenever snaps stalled and
/// then fast-forwarded on catch-up — the periodic breathe. Drift during a
/// gap is bounded by rate*EXTRA_WALL and blends back on resume.
pub const CURSOR_EXTRA_WALL: f64 = 0.5;

pub fn advance_cursor(
    cur: f64,
    front: f64,
    back: f64,
    rate: f64,
    dt: f64,
    wall_dt: f64,
    target_wall: f64,
) -> f64 {
    let target = (rate * target_wall).clamp(2.0 * dt, 8.0 * dt);
    let occ = (back - cur).max(0.0);
    let factor = (occ / target).clamp(0.25, 1.5);
    let allow = (2.0 * dt).max(rate * CURSOR_EXTRA_WALL);
    (cur + wall_dt * rate * factor).clamp(front, back + allow)
}

/// Bracket a sim time against history: (lower index, blend, extrapolate).
pub fn bracket(hist: &VecDeque<HistSnap>, t: f64) -> (usize, f64, f64) {
    let n = hist.len();
    if n == 0 {
        return (0, 1.0, 0.0);
    }
    let mut lo = 0;
    while lo + 1 < n && hist[lo + 1].time <= t {
        lo += 1;
    }
    if lo + 1 >= n {
        return (n - 1, 1.0, (t - hist[n - 1].time).max(0.0));
    }
    let span = (hist[lo + 1].time - hist[lo].time).max(1e-12);
    (lo, ((t - hist[lo].time) / span).clamp(0.0, 1.0), 0.0)
}

pub enum Msg {
    Static(Static),
    Snap(Snap),
}

pub struct Ctl {
    pub paused: std::sync::atomic::AtomicBool,
    pub speed: std::sync::Mutex<f32>,
    pub dt: std::sync::Mutex<f64>,
}

impl Ctl {
    pub fn new() -> Self {
        Self {
            paused: std::sync::atomic::AtomicBool::new(false),
            speed: std::sync::Mutex::new(1.0),
            dt: std::sync::Mutex::new(0.004),
        }
    }
}

fn snap_of(sim: &Sim) -> Snap {
    Snap {
        pos: sim.pos.clone(),
        vel: sim.vel.clone(),
        gas: sim.gas.iter().map(|g| g.pos).collect(),
        gas_size: sim.gas.iter().map(|g| g.size).collect(),
        gas_flash: sim.gas.iter().map(|g| g.flash).collect(),
        time: sim.time,
        steps: sim.steps,
    }
}

pub fn physics_loop(mut sim: Sim, tx: std::sync::mpsc::SyncSender<Msg>, ctl: std::sync::Arc<Ctl>) {
    let init_dt = *ctl.dt.lock().unwrap();
    sim.dt = init_dt;
    let n = sim.n();
    tx.send(Msg::Static(Static {
        temp: sim.temp.clone(),
        mag: sim.mag.clone(),
        census: sim.census,
        gas: sim.gas.clone(),
        gas_om: sim.gas_om,
        n,
    }))
    .ok();
    loop {
        if ctl.paused.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }
        // Speed scales the step size; render stretches snapshots over
        // arrival intervals so motion stays smooth at any rate.
        let speed = *ctl.speed.lock().unwrap();
        sim.dt = *ctl.dt.lock().unwrap() * speed.max(0.01) as f64;
        sim.step();
        let done = tx.send(Msg::Snap(snap_of(&sim))).is_err();
        if done {
            break;
        }
    }
}

fn remove_com(pos: &mut [[f64; 3]], vel: &mut [[f64; 3]], mass: &[f64]) {
    let mut mp = [0.0; 3];
    let mut mv = [0.0; 3];
    let mut mtot = 0.0;
    for i in 0..mass.len() {
        mtot += mass[i];
        for d in 0..3 {
            mp[d] += mass[i] * pos[i][d];
            mv[d] += mass[i] * vel[i][d];
        }
    }
    for i in 0..mass.len() {
        for d in 0..3 {
            pos[i][d] -= mp[d] / mtot;
            vel[i][d] -= mv[d] / mtot;
        }
    }
}

fn rescale_virial(vel: &mut [[f64; 3]], pos: &[[f64; 3]], mass: &[f64]) {
    let mut t = 0.0;
    let mut w = 0.0;
    for i in 0..mass.len() {
        t +=
            0.5 * mass[i] * (vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1] + vel[i][2] * vel[i][2]);
        for j in (i + 1)..mass.len() {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            let r = (dx * dx + dy * dy + dz * dz + 1e-12).sqrt();
            w -= mass[i] * mass[j] / r;
        }
    }
    if t > 0.0 && w < 0.0 {
        let s = (-0.5 * w / t).sqrt() * 0.9;
        for v in vel.iter_mut() {
            v[0] *= s;
            v[1] *= s;
            v[2] *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ic::Preset;
    use fmm::models::Rng;

    #[test]
    fn kdk_advances_finite() {
        let mut sim = Sim::new(128, 0.004, false, None, Preset::Cluster);
        for _ in 0..3 {
            sim.step();
        }
        assert!((sim.time - 0.012).abs() < 1e-12, "t={}", sim.time);
        assert_eq!(sim.steps, 3);
        for i in 0..sim.n() {
            for d in 0..3 {
                assert!(sim.pos[i][d].is_finite());
                assert!(sim.vel[i][d].is_finite());
            }
        }
    }

    #[test]
    fn direct_parallel_matches_serial() {
        let n = 700usize;
        let mut rng = Rng::new(7);
        let mut pos = vec![[0.0; 3]; n];
        let mut mass = vec![0.0; n];
        for i in 0..n {
            pos[i] = [
                rng.unit() * 2.0 - 1.0,
                rng.unit() * 2.0 - 1.0,
                rng.unit() * 2.0 - 1.0,
            ];
            mass[i] = 1.0 / n as f64;
        }
        let mut d = Direct { soft2: 1e-6 };
        let got = d.accel(&pos, &mass);
        let mut want = vec![[0.0; 3]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r2 = dx * dx + dy * dy + dz * dz + 1e-6;
                let inv = 1.0 / r2.sqrt();
                let inv3 = inv * inv * inv;
                let si = mass[j] * inv3;
                let sj = mass[i] * inv3;
                want[i][0] += si * dx;
                want[i][1] += si * dy;
                want[i][2] += si * dz;
                want[j][0] -= sj * dx;
                want[j][1] -= sj * dy;
                want[j][2] -= sj * dz;
            }
        }
        for i in 0..n {
            for k in 0..3 {
                assert!((got[i][k] - want[i][k]).abs() < 1e-9, "{i} {k}");
            }
        }
    }

    #[test]
    fn cached_accel_matches_fresh() {
        let mut a = Sim::new(64, 0.004, false, None, Preset::Cluster);
        let mut b = Sim::new(64, 0.004, false, None, Preset::Cluster);
        for _ in 0..4 {
            a.step();
        }
        b.step();
        b.acc = None;
        for _ in 0..3 {
            b.step();
        }
        for i in 0..64 {
            for d in 0..3 {
                assert!((a.pos[i][d] - b.pos[i][d]).abs() < 1e-12, "{i} {d}");
                assert!((a.vel[i][d] - b.vel[i][d]).abs() < 1e-12, "{i} {d}");
            }
        }
    }

    #[test]
    fn large_n_init_skips_virial() {
        let t = std::time::Instant::now();
        let sim = Sim::new(6000, 0.004, false, None, Preset::Cluster);
        assert_eq!(sim.n(), 6000);
        assert!(t.elapsed() < std::time::Duration::from_secs(20));
        for i in 0..sim.n() {
            for d in 0..3 {
                assert!(sim.pos[i][d].is_finite());
            }
        }
    }

    #[test]
    fn presets_run_finite() {
        for preset in [
            Preset::Cluster,
            Preset::Spiral,
            Preset::Merger,
            Preset::BlackHole,
        ] {
            let mut sim = Sim::new(64, 0.004, false, None, preset);
            for _ in 0..3 {
                sim.step();
            }
            assert_eq!(sim.steps, 3);
            // Census folded at init: every non-BH star classified exactly once.
            let bh = match preset {
                Preset::Cluster => 0,
                Preset::Merger => 2,
                _ => 1,
            };
            assert_eq!(sim.census.iter().sum::<usize>(), 64 - bh, "{preset:?}");
            for i in 0..sim.n() {
                for d in 0..3 {
                    assert!(sim.pos[i][d].is_finite(), "{preset:?} {i} {d}");
                }
            }
        }
    }

    #[test]
    fn accel_cap_bounds_huge_and_nonfinite() {
        let mut a = vec![
            [1e6, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
            [f64::NAN, 1.0, 2.0],
            [3.0, 4.0, 0.0],
        ];
        cap_accel(&mut a);
        for v in &a {
            for &x in v {
                assert!(x.is_finite());
            }
            let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(s <= ACCEL_CAP + 1e-9, "{s}");
        }
        assert_eq!(a[3], [3.0, 4.0, 0.0]);
    }

    #[test]
    fn dt_clamp_caps_requested_step() {
        // Speed 10x requests dt=0.16; near the BH the eta criterion must shrink it.
        let mut sim = Sim::new(256, 0.16, false, None, Preset::Spiral);
        sim.step();
        assert!(sim.time < 0.05, "t={}", sim.time);
        assert!(sim.time > 0.0);
        for i in 0..sim.n() {
            for d in 0..3 {
                assert!(sim.pos[i][d].is_finite());
            }
        }
    }

    #[test]
    fn physics_loop_streams_snaps() {
        let sim = Sim::new(128, 0.004, false, None, Preset::Cluster);
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let ctl = std::sync::Arc::new(Ctl::new());
        std::thread::spawn(move || physics_loop(sim, tx, ctl));
        let mut snaps = 0;
        let mut last_steps = 0;
        let t = std::time::Instant::now();
        while snaps < 3 && t.elapsed() < std::time::Duration::from_secs(15) {
            match rx.recv_timeout(std::time::Duration::from_secs(15)) {
                Ok(Msg::Static(s)) => assert_eq!(s.n, 128),
                Ok(Msg::Snap(s)) => {
                    assert!(s.steps > last_steps);
                    last_steps = s.steps;
                    for i in 0..s.pos.len() {
                        for d in 0..3 {
                            assert!(s.pos[i][d].is_finite());
                            assert!(s.vel[i][d].is_finite());
                        }
                    }
                    snaps += 1;
                }
                Err(_) => break,
            }
        }
        assert_eq!(snaps, 3);
    }

    #[test]
    fn bracket_holds_blends_extrapolates() {
        use std::time::Instant;
        let now = Instant::now();
        let snap = |t: f64| HistSnap {
            pos: vec![[t, 0.0, 0.0]],
            vel: vec![[1.0, 0.0, 0.0]],
            gas: vec![[t, 0.0, 0.0]],
            gas_size: vec![0.05],
            gas_flash: vec![0.0],
            time: t,
            steps: 0,
            arrival: now,
        };
        let h: VecDeque<HistSnap> = [0.0, 0.01, 0.02].into_iter().map(snap).collect();
        assert_eq!(bracket(&h, -1.0), (0, 0.0, 0.0));
        let (lo, al, ex) = bracket(&h, 0.015);
        assert_eq!(lo, 1);
        assert!((al - 0.5).abs() < 1e-12, "{al}");
        assert_eq!(ex, 0.0);
        let (lo, al, ex) = bracket(&h, 0.035);
        assert_eq!((lo, al), (2, 1.0));
        assert!((ex - 0.015).abs() < 1e-12, "{ex}");
        assert_eq!(bracket(&VecDeque::new(), 0.0), (0, 1.0, 0.0));
    }

    #[test]
    fn extrapolation_stays_bounded() {
        let got = advance_cursor(1.0, 0.0, 1.0, 0.2, 0.01, 5.0, 0.30);
        // Allowance is max(2dt, rate*0.5 wall) = 0.1 here, not 2dt = 0.02.
        assert!((got - 1.1).abs() < 1e-12, "{got}");
    }

    #[test]
    fn cursor_glides_through_arrival_gap() {
        // Newest snap frozen at t=1.0 for a full wall-second (slow refresh
        // at large N): the cursor must keep gliding past the old back+2dt
        // ceiling at a steady crawl, never pin and never run past the
        // rate*EXTRA_WALL allowance.
        let mut cur = 1.0;
        let mut first = 0.0;
        let mut last = 0.0;
        for f in 0..60 {
            let next = advance_cursor(cur, 0.0, 1.0, 0.2, 0.01, 1.0 / 60.0, 0.30);
            if f == 0 {
                first = next - cur;
            }
            last = next - cur;
            cur = next;
        }
        assert!(cur > 1.02, "pinned at the old 2-step ceiling: {cur}");
        assert!(
            cur <= 1.0 + 0.2 * CURSOR_EXTRA_WALL + 1e-9,
            "ran past allowance: {cur}"
        );
        assert!(
            (first - last).abs() < 1e-12,
            "glide pulsed: {first} vs {last}"
        );
    }

    #[test]
    fn gas_shears_differentially() {
        let mut sim = Sim::new(512, 0.004, false, None, Preset::Spiral);
        // Respawn recycles dead slots at sibling positions (teleports slot
        // indices by design); shear tracks slot continuity, so disable it.
        sim.gas_target = 0;
        assert!(!sim.gas.is_empty());
        let band = |sim: &Sim, lo: f64, hi: f64| {
            let mut ids = Vec::new();
            for (i, g) in sim.gas.iter().enumerate() {
                let r = (g.pos[0] * g.pos[0] + g.pos[1] * g.pos[1]).sqrt();
                if g.size > 0.0 && (lo..=hi).contains(&r) {
                    ids.push((i, g.pos[1].atan2(g.pos[0])));
                }
            }
            ids
        };
        let inner0 = band(&sim, 0.4, 1.0);
        let outer0 = band(&sim, 2.0, 3.5);
        assert!(inner0.len() > 5 && outer0.len() > 5);
        for _ in 0..100 {
            sim.step();
        }
        let swept = |ids: &[(usize, f64)]| {
            let mut s = 0.0;
            let mut n = 0;
            for &(i, a0) in ids {
                let g = &sim.gas[i];
                if g.size <= 0.0 {
                    continue;
                }
                s += g.pos[1].atan2(g.pos[0]) - a0;
                n += 1;
            }
            (s / n as f64, n)
        };
        let (di, ni) = swept(&inner0);
        let (douts, no) = swept(&outer0);
        // Rate-based: dt clamp shrinks steps near the nucleus; ratios and
        // rates are immune. Inner Ω(0.7)≈1.0, outer Ω(2.75)≈0.27.
        let t = sim.time.max(1e-9);
        assert!(ni > 3 && no > 3, "ni={ni} no={no}");
        assert!(di / t > 0.4, "inner rate {}", di / t);
        assert!(di > 3.0 * douts, "inner {di} outer {douts}");
    }

    #[test]
    fn gas_stays_bounded_and_finite() {
        for preset in [Preset::Spiral, Preset::Merger, Preset::BlackHole] {
            let mut sim = Sim::new(256, 0.004, false, None, preset);
            for _ in 0..200 {
                sim.step();
            }
            assert!(!sim.gas.is_empty(), "{preset:?}");
            for g in &sim.gas {
                for d in 0..3 {
                    assert!(g.pos[d].is_finite() && g.vel[d].is_finite(), "{preset:?}");
                }
                let r = (g.pos[0] * g.pos[0] + g.pos[1] * g.pos[1] + g.pos[2] * g.pos[2]).sqrt();
                assert!((0.005..=12.0).contains(&r), "{preset:?} r={r}");
                let sp = (g.vel[0] * g.vel[0] + g.vel[1] * g.vel[1] + g.vel[2] * g.vel[2]).sqrt();
                assert!(sp <= GAS_VMAX + 1e-9, "{preset:?} sp={sp}");
            }
        }
    }

    #[test]
    fn merge_conserves_volume_and_grows() {
        use crate::ic::Gas;
        let mut v = vec![
            Gas {
                pos: [0.0, 0.0, 0.0],
                vel: [1.0, 0.0, 0.0],
                size: 0.04,
                kind: 0.0,
                seed: 0.1,
                flash: 0.0,
            },
            Gas {
                pos: [0.02, 0.0, 0.0],
                vel: [0.0, 2.0, 0.0],
                size: 0.06,
                kind: 0.0,
                seed: 0.2,
                flash: 0.0,
            },
            Gas {
                pos: [5.0, 0.0, 0.0],
                vel: [0.0; 3],
                size: 0.05,
                kind: 0.0,
                seed: 0.3,
                flash: 0.0,
            },
        ];
        let vol = |v: &[Gas]| v.iter().map(|g| (g.size as f64).powi(3)).sum::<f64>();
        let v0 = vol(&v);
        // Gradual: one sweep moves a volume fraction (no pop), repeated
        // sweeps converge on the volume sum (cbrt), not the diameter sum.
        merge_gas(&mut v);
        assert!(v[1].size > 0.06, "grew {}", v[1].size);
        assert!(v[0].size > 0.0 && v[0].size < 0.04, "{}", v[0].size);
        assert!(v[1].flash > 0.0, "merge pumps flash");
        assert!((vol(&v) - v0).abs() < 1e-9, "volume conserved");
        let step1 = v[1].size;
        for _ in 0..20 {
            merge_gas(&mut v);
        }
        let want = (0.04f64.powi(3) + 0.06f64.powi(3)).cbrt();
        assert!((v[1].size as f64 - want).abs() < 1e-6, "{}", v[1].size);
        assert_eq!(v[0].size, 0.0);
        assert_eq!(v[2].size, 0.05);
        assert!((vol(&v) - v0).abs() < 1e-6, "still conserved");
        // Drifted toward the meal but never teleported past either start.
        assert!((0.0..=0.02).contains(&v[1].pos[0]), "{}", v[1].pos[0]);
        assert!(v[1].pos[0] < 0.02 && v[1].pos[0] > 0.0);
        assert!(step1 < v[1].size);
        assert_eq!(v[1].kind, 0.0);
        assert_eq!(v[1].seed, 0.2);
    }

    #[test]
    fn merge_single_sweep_never_pops() {
        use crate::ic::Gas;
        let mk = |size: f32| Gas {
            pos: [0.0; 3],
            vel: [0.0; 3],
            size,
            kind: 0.0,
            seed: 0.0,
            flash: 0.0,
        };
        // Worst case: two near-cap knots. One sweep must move < 60% of the
        // small knot's volume (supernova swell, not explosion pop).
        let mut v = vec![mk(0.10), mk(0.10)];
        merge_gas(&mut v);
        let (a, b) = (v[0].size as f64, v[1].size as f64);
        assert!((a - 0.10).abs() + (b - 0.10).abs() > 1e-9, "must transfer");
        assert!(
            ((a.powi(3) + b.powi(3)) - 2.0 * 0.10f64.powi(3)).abs() < 1e-9,
            "volume conserved {a}+{b}"
        );
        assert!(
            (a - 0.10).abs() < 0.06 && (b - 0.10).abs() < 0.06,
            "{a} {b}"
        );
    }

    #[test]
    fn merge_excludes_and_respects_caps() {
        use crate::ic::Gas;
        let mk = |kind: f32, size: f32| Gas {
            pos: [0.0; 3],
            vel: [0.0; 3],
            size,
            kind,
            seed: 0.0,
            flash: 0.0,
        };
        // Different classes never transfer — but they must not share space:
        // exclusion pushes them apart with sizes untouched.
        let mut v = vec![mk(0.0, 0.05), mk(3.0, 0.05)];
        v[1].pos = [0.01, 0.0, 0.0];
        merge_gas(&mut v);
        assert_eq!((v[0].size, v[1].size), (0.05, 0.05));
        let d = (v[1].pos[0] - v[0].pos[0]).abs();
        assert!(d > 0.01, "pushed apart {d}");
        // Full merge when the volume sum fits under the cap: big converges
        // on cbrt(v0+v1), small vanishes, volume conserved throughout.
        let mut w = vec![mk(0.0, 0.12), mk(0.0, 0.10)];
        for _ in 0..30 {
            merge_gas(&mut w);
        }
        let want = (0.12f64.powi(3) + 0.10f64.powi(3)).cbrt();
        assert!(want < 0.18, "fits under cap");
        assert!((w[0].size as f64 - want).abs() < 1e-6, "{}", w[0].size);
        assert_eq!(w[1].size, 0.0);
        // At-cap big: no transfer, exclusion instead. Sizes frozen, pair
        // separates, volume exactly conserved.
        let mut u = vec![mk(0.0, 0.18), mk(0.0, 0.05)];
        u[1].pos = [0.02, 0.0, 0.0];
        let vu = (u[0].size as f64).powi(3) + (u[1].size as f64).powi(3);
        merge_gas(&mut u);
        assert_eq!((u[0].size, u[1].size), (0.18, 0.05));
        let du = (u[1].pos[0] - u[0].pos[0]).abs();
        assert!(du > 0.02, "separated {du}");
        assert!(
            ((u[0].size as f64).powi(3) + (u[1].size as f64).powi(3) - vu).abs() < 1e-12,
            "conserved"
        );
        let mut t1 = vec![mk(0.0, 0.05), mk(0.0, 0.05)];
        let mut t2 = t1.clone();
        merge_gas(&mut t1);
        merge_gas(&mut t2);
        assert_eq!(t1[0].size, t2[0].size);
        assert_eq!(t1[0].pos, t2[0].pos);
    }

    #[test]
    fn dead_knots_flagged_in_snap() {
        // Regression: merged-away knots must read size 0 on the snap wire,
        // or the app resurrects them as frozen statues from init meta.
        let mut sim = Sim::new(128, 0.004, false, None, Preset::Spiral);
        let mut pair = None;
        for (i, a) in sim.gas.iter().enumerate() {
            for (j, b) in sim.gas.iter().enumerate().skip(i + 1) {
                let em = a.kind < 2.5 && b.kind < 2.5;
                let du = a.kind >= 2.5 && b.kind >= 2.5;
                if em || du {
                    pair = Some((i, j));
                    break;
                }
            }
            if pair.is_some() {
                break;
            }
        }
        let (i, j) = pair.expect("same-class pair");
        sim.gas[j].pos = sim.gas[i].pos;
        // Gradual merge: shrink the meal to a wisp so one sweep finishes it.
        sim.gas[j].size = 0.01;
        sim.gas[j].flash = 0.0;
        merge_gas(&mut sim.gas);
        assert!(sim.gas.iter().filter(|g| g.size <= 0.0).count() >= 1);
        let snap = snap_of(&sim);
        assert_eq!(snap.gas.len(), sim.gas.len());
        assert_eq!(snap.gas_size.len(), sim.gas.len());
        assert_eq!(snap.gas_flash.len(), sim.gas.len());
        assert!(snap.gas_size.iter().filter(|&&s| s <= 0.0).count() >= 1);
    }

    #[test]
    fn long_run_no_blob() {
        let mut sim = Sim::new(512, 0.004, false, None, Preset::Spiral);
        for _ in 0..300 {
            sim.step();
        }
        let mut mx = 0.0f32;
        let mut alive = 0;
        for g in &sim.gas {
            assert!(g.pos.iter().all(|v| v.is_finite()));
            assert!(g.vel.iter().all(|v| v.is_finite()));
            if g.size > 0.0 {
                alive += 1;
                mx = mx.max(g.size);
            }
        }
        assert!(alive > 0);
        assert!(mx <= 0.22 + 1e-6, "max size {mx}");
        // Merging must not eat the disk: most knots survive 300 steps.
        let frac = alive as f64 / sim.gas.len() as f64;
        assert!(frac > 0.5, "alive frac {frac}");
    }

    #[test]
    fn emission_budget_bounded_and_steady() {
        // Locks the heartbeat + white-donut regressions: total emission
        // budget (Σ size²·(1+1.5·flash), the shader brightness proxy) must
        // neither run away under fountain refill nor pulse on any cadence.
        let mut sim = Sim::new(512, 0.004, false, None, Preset::Spiral);
        let budget = |sim: &Sim| {
            sim.gas
                .iter()
                .filter(|g| g.kind < 2.5 && g.size > 0.0)
                .map(|g| {
                    let s = g.size as f64;
                    s * s * (1.0 + 1.5 * g.flash as f64)
                })
                .sum::<f64>()
        };
        let b0 = budget(&sim);
        let mut series = Vec::new();
        for _ in 0..300 {
            sim.step();
            series.push(budget(&sim));
        }
        let tail = &series[100..];
        let (mn, mx) = tail
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
        // No pulse on any cadence (the heartbeat); equilibrium flat or
        // draining, never accumulating (the white donut). Seeded start b0
        // is deliberately small, so compare tail against itself, not b0.
        assert!(mx / mn < 1.6, "pulsing: min {mn} max {mx}");
        let q1: f64 = tail[..50].iter().sum::<f64>() / 50.0;
        let q4: f64 = tail[150..].iter().sum::<f64>() / 50.0;
        assert!(q4 <= 1.05 * q1, "accumulating {q1} -> {q4}");
        let _ = b0;
        let mean: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
        let jab = tail
            .windows(2)
            .fold(0.0f64, |a, w| a.max((w[1] - w[0]).abs()));
        assert!(jab / mean < 0.08, "step jitter {jab} mean {mean}");
    }

    #[test]
    fn gas_jams_in_lanes() {
        let mut on = Sim::new(512, 0.004, false, None, Preset::Spiral);
        let mut off = Sim::new(512, 0.004, false, None, Preset::Spiral);
        off.gas_arms = false;
        for _ in 0..200 {
            on.step();
            off.step();
        }
        // Jammed fraction inside CR (r < 1.8): knots crawling near
        // pattern speed. Outskirts excluded (slow pattern-relative anyway).
        let jammed = |sim: &Sim| {
            let mut n = 0;
            let mut k = 0;
            for g in &sim.gas {
                if g.kind < 2.5 || g.size <= 0.0 {
                    continue;
                }
                let r = (g.pos[0] * g.pos[0] + g.pos[1] * g.pos[1]).sqrt().max(0.05);
                if r >= 1.8 {
                    continue;
                }
                let phi = g.pos[1].atan2(g.pos[0]);
                let psi = 2.0 * (phi - sim.gas_om as f64 * sim.time) - 3.0 * (r + 0.15).ln();
                if (wrap_pi(2.0 * (psi + 0.55)) / 2.0).abs() > 0.3 {
                    continue;
                }
                let lz = g.pos[0] * g.vel[1] - g.pos[1] * g.vel[0];
                let vr = (g.pos[0] * g.vel[0] + g.pos[1] * g.vel[1]) / r;
                if (2.0 * (lz / (r * r) - sim.gas_om as f64) - 3.0 * vr / (r + 0.15)).abs() < 0.5 {
                    k += 1;
                }
                n += 1;
            }
            (k as f64 / n as f64, n)
        };
        let (jon, non) = jammed(&on);
        let (joff, noff) = jammed(&off);
        assert!(non > 20 && noff > 20);
        assert!(jon > joff + 0.04, "on={jon} off={joff}");
    }

    #[test]
    fn cavity_stays_open_and_ring_persists() {
        let mut sim = Sim::new(512, 0.004, false, None, Preset::Spiral);
        for _ in 0..300 {
            sim.step();
        }
        let r2d = |g: &crate::ic::Gas| (g.pos[0] * g.pos[0] + g.pos[1] * g.pos[1]).sqrt();
        let inside = sim.gas.iter().filter(|g| r2d(g) < 0.07).count();
        assert!(
            (inside as f64) < 0.03 * sim.gas.len() as f64,
            "inside={inside} n={}",
            sim.gas.len()
        );
        let ring = sim
            .gas
            .iter()
            .filter(|g| (0.25..=0.35).contains(&r2d(g)))
            .count() as f64
            / (std::f64::consts::PI * (0.35 * 0.35 - 0.25 * 0.25));
        let ctrl = sim
            .gas
            .iter()
            .filter(|g| (0.50..=0.70).contains(&r2d(g)))
            .count() as f64
            / (std::f64::consts::PI * (0.70 * 0.70 - 0.50 * 0.50));
        assert!(ring > 1.3 * ctrl, "ring={ring} ctrl={ctrl}");
        // Soft, not a detached annulus: feathered ring never runs away, and
        // the feed band stays populated (no moat between cavity and ring).
        assert!(ring < 5.0 * ctrl, "hard annulus ring={ring} ctrl={ctrl}");
        let fed = sim
            .gas
            .iter()
            .filter(|g| (0.07..0.20).contains(&r2d(g)))
            .count() as f64
            / (std::f64::consts::PI * (0.20 * 0.20 - 0.07 * 0.07));
        assert!(fed > 0.3 * ring, "moat fed={fed} ring={ring}");
    }

    #[test]
    fn plunge_recycles_instead_of_clamping() {
        // Real infall without deletion: a knot inside ISCO rejoins the disk
        // at a live sibling with its size intact + feedback flash, instead
        // of piling into a flickering clamped statue or vanishing.
        let mut sim = Sim::new(256, 0.004, false, None, Preset::BlackHole);
        sim.gas_target = 0;
        let bh_pos = sim.pos[sim.bh[0].0];
        sim.gas[0].pos = [bh_pos[0] + 0.02, bh_pos[1], bh_pos[2]];
        sim.gas[0].vel = [0.0; 3];
        sim.gas[0].size = 0.05;
        sim.gas[1].pos = [bh_pos[0] + 0.09, bh_pos[1], bh_pos[2]];
        sim.gas[1].vel = [0.0, 1.5, 0.0];
        sim.gas[1].size = 0.05;
        sim.gas[1].flash = 0.0;
        sim.step();
        assert_eq!(sim.gas[0].size, 0.05, "plunged knot keeps its mass");
        assert_eq!(sim.gas[0].flash, 1.0, "feedback flash on recycle");
        let r0 =
            (sim.gas[0].pos[0] * sim.gas[0].pos[0] + sim.gas[0].pos[1] * sim.gas[0].pos[1]).sqrt();
        assert!(r0 > 0.06, "rejoined the disk, r={r0}");
        assert!(sim.gas[1].size > 0.0, "rim survives");
        // Tidal heat glides up smoothly (no periapsis strobe): ~a minute of
        // steps to approach the rim floor.
        for _ in 0..60 {
            sim.step();
        }
        assert!(sim.gas[1].flash > 0.2, "tidal heat {}", sim.gas[1].flash);
    }

    #[test]
    fn inner_clouds_persist() {
        // No shredding, no evaporation: a big knot deep in the well keeps
        // its size indefinitely (mass is conserved everywhere now).
        let mut sim = Sim::new(256, 0.004, false, None, Preset::Spiral);
        sim.gas_target = 0;
        for g in sim.gas.iter_mut().skip(1) {
            g.size = 0.0;
        }
        sim.gas[0].size = 0.15;
        sim.gas[0].flash = 0.0;
        for _ in 0..100 {
            sim.gas[0].pos = [0.15, 0.0, 0.0];
            sim.gas[0].vel = [0.0, 0.71, 0.0];
            sim.step();
        }
        assert!(
            ((sim.gas[0].size as f64) - 0.15).abs() < 1e-6,
            "persisted {}",
            sim.gas[0].size
        );
    }

    #[test]
    fn flash_decays_without_pump() {
        let mut sim = Sim::new(256, 0.004, false, None, Preset::Spiral);
        sim.gas_target = 0;
        // Outer knot: heat floor is 0 there, so only exponential decay acts.
        let mut k = 0;
        for (i, g) in sim.gas.iter().enumerate() {
            let r = (g.pos[0] * g.pos[0] + g.pos[1] * g.pos[1]).sqrt();
            if r > 1.5 {
                k = i;
                break;
            }
        }
        sim.gas[k].flash = 1.0;
        sim.step();
        assert!(
            sim.gas[k].flash < 1.0 && sim.gas[k].flash > 0.9,
            "fade {}",
            sim.gas[k].flash
        );
    }

    #[test]
    fn respawn_recycles_dead() {
        let mut sim = Sim::new(256, 0.004, false, None, Preset::Spiral);
        // Deterministic: kill 10, one sweep refills the 8-slot cap, the
        // next finishes the rest. No dynamics involved.
        for g in sim.gas.iter_mut().take(10) {
            g.size = 0.0;
        }
        let alive0 = sim.gas.iter().filter(|g| g.size > 0.0).count();
        sim.respawn_gas(8);
        let alive1 = sim.gas.iter().filter(|g| g.size > 0.0).count();
        assert_eq!(alive1, alive0 + 8, "{alive0} -> {alive1}");
        sim.respawn_gas(8);
        let alive2 = sim.gas.iter().filter(|g| g.size > 0.0).count();
        assert_eq!(alive2, alive0 + 10, "{alive1} -> {alive2}");
        for g in &sim.gas {
            assert!(g.pos.iter().all(|v| v.is_finite()));
            assert!((0.0..=1.0).contains(&g.flash), "{}", g.flash);
            assert!((0.0..=0.22).contains(&g.size), "{}", g.size);
        }
    }

    #[test]
    fn gas_thin_and_stirred() {
        let mut sim = Sim::new(512, 0.004, false, None, Preset::Spiral);
        for _ in 0..200 {
            sim.step();
        }
        let mut mz = 0.0;
        let mut disp = 0.0;
        let mut n = 0;
        for g in &sim.gas {
            let r = (g.pos[0] * g.pos[0] + g.pos[1] * g.pos[1]).sqrt().max(0.05);
            if !(0.5..=3.0).contains(&r) {
                continue;
            }
            mz += g.pos[2].abs();
            let vr = (g.pos[0] * g.vel[0] + g.pos[1] * g.vel[1]) / r;
            let sp = (g.vel[0] * g.vel[0] + g.vel[1] * g.vel[1]).sqrt().max(1e-9);
            disp += vr.abs() / sp;
            n += 1;
        }
        mz /= n as f64;
        disp /= n as f64;
        assert!(mz < 0.10, "mz={mz}");
        assert!((0.02..=0.30).contains(&disp), "disp={disp}");
    }

    #[test]
    fn gas_whirls_near_bh() {
        let mut sim = Sim::new(256, 0.004, false, None, Preset::BlackHole);
        let mut inner = 0;
        for (i, g) in sim.gas.iter().enumerate() {
            let r = (g.pos[0] * g.pos[0] + g.pos[1] * g.pos[1]).sqrt();
            let rb = (sim.gas[inner].pos[0] * sim.gas[inner].pos[0]
                + sim.gas[inner].pos[1] * sim.gas[inner].pos[1])
                .sqrt();
            if r < rb {
                inner = i;
            }
        }
        let ang = |p: &[f64; 3]| p[1].atan2(p[0]);
        let a0 = ang(&sim.gas[inner].pos);
        for _ in 0..50 {
            sim.step();
        }
        let swept = (ang(&sim.gas[inner].pos) - a0).abs();
        assert!(swept > 0.2, "swept={swept}");
    }

    #[test]
    fn cursor_rides_bursty_arrivals_uniformly() {
        use std::time::Duration;
        let dt = 0.01;
        let frame = 1.0 / 144.0;
        let gaps = [0.017, 0.017, 0.017, 0.140];
        let t0 = std::time::Instant::now();
        let mut hist = VecDeque::new();
        let mut sim_time = 0.0;
        let mut next_at = 0.0;
        let mut gi = 0;
        let mut now = 0.0;
        let mut rate = 0.0;
        let mut sdt = dt;
        let mut cur: Option<f64> = None;
        let mut deltas = Vec::new();
        let mut prev = 0.0;
        for _ in 0..900 {
            now += frame;
            while next_at <= now {
                sim_time += dt;
                hist.push_back(HistSnap {
                    pos: vec![[sim_time, 0.0, 0.0]],
                    vel: vec![[1.0, 0.0, 0.0]],
                    gas: vec![[sim_time, 0.0, 0.0]],
                    gas_size: vec![0.05],
                    gas_flash: vec![0.0],
                    time: sim_time,
                    steps: 0,
                    arrival: t0 + Duration::from_secs_f64(next_at),
                });
                next_at += gaps[gi % 4];
                gi += 1;
                while hist.len() > 24 {
                    hist.pop_front();
                }
            }
            if hist.len() >= 2 {
                let (f, b) = (hist.front().unwrap(), hist.back().unwrap());
                let span = (b.time - f.time).max(1e-12);
                sdt = (span / (hist.len() as f64 - 1.0)).max(1e-9);
                let wall = (b.arrival - f.arrival).as_secs_f64().max(1e-6);
                let inst = span / wall;
                rate = if rate <= 0.0 {
                    inst
                } else {
                    rate * 0.85 + inst * 0.15
                };
            }
            let (f, b) = (hist.front().unwrap(), hist.back().unwrap());
            let mut rt = match cur {
                Some(t) => t,
                None => {
                    let target = (rate * 0.30).clamp(2.0 * sdt, 8.0 * sdt);
                    (b.time - target).max(f.time)
                }
            };
            rt = advance_cursor(rt, f.time, b.time, rate, sdt, frame, 0.30);
            rt = rt.max(hist.front().unwrap().time);
            while hist.len() >= 3 && hist[1].time <= rt {
                hist.pop_front();
            }
            deltas.push(rt - prev);
            prev = rt;
            cur = Some(rt);
        }
        let tail = &deltas[300..];
        let (mn, mx) = tail
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(a, b), &d| (a.min(d), b.max(d)));
        assert!(mn > 0.0, "cursor stalls: min {mn}");
        assert!(mx / mn < 3.0, "pulsing: min {mn} max {mx}");
    }
}
