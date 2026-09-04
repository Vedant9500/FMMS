use fmm::fmm::Fmm;
use fmm::models::{plummer, Particle, Rng};

pub trait Physics {
    fn accel(&mut self, pos: &[[f64; 3]], mass: &[f64]) -> Vec<[f64; 3]>;
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
        match &mut self.fmm {
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
        }
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
    pub prev: Vec<[f64; 3]>,
    pub prev_vel: Vec<[f64; 3]>,
    pub time: f64,
    pub dt: f64,
    pub steps: u64,
    backend: Backend,
    acc: Option<Vec<[f64; 3]>>,
    vmax: Option<f64>,
}

enum Backend {
    Direct(Direct),
    Fmm(FmmBackend),
    Gpu(crate::gpufmm::GpuBackend),
}

impl Sim {
    pub fn new(n: usize, dt: f64, use_fmm: bool, gpu: Option<crate::gpufmm::GpuBackend>) -> Self {
        let parts = plummer(n, [0.0; 3], 42);
        let mut rng = Rng::new(1234);
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        for p in &parts {
            pos.push(p.pos);
            mass.push(p.mass);
            let r2 = p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2];
            let sig2 = 1.0 / (6.0 * (1.0 + r2).sqrt());
            let sig = sig2.sqrt();
            let mut gauss = || {
                let u1 = rng.unit().max(1e-12);
                let u2 = rng.unit();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            vel.push([sig * gauss(), sig * gauss(), sig * gauss()]);
        }
        remove_com(&mut pos, &mut vel, &mass);
        if n <= 5000 {
            rescale_virial(&mut vel, &pos, &mass);
        }
        let mut temp = Vec::with_capacity(n);
        let mut mag = Vec::with_capacity(n);
        for _ in 0..n {
            let u1 = rng.unit().max(1e-12);
            let u2 = rng.unit();
            let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            temp.push((0.5 + 0.22 * g).clamp(0.02, 1.0) as f32);
        }
        for _ in 0..n {
            mag.push((rng.unit().powf(3.0) * 0.9 + 0.1) as f32);
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
        }
    }

    pub fn n(&self) -> usize {
        self.pos.len()
    }

    pub fn step(&mut self) {
        self.prev.clone_from(&self.pos);
        self.prev_vel.clone_from(&self.vel);
        let dt = self.dt;
        let a0 = match self.acc.take() {
            Some(a) => a,
            None => match &mut self.backend {
                Backend::Direct(d) => d.accel(&self.pos, &self.mass),
                Backend::Fmm(f) => f.accel(&self.pos, &self.mass),
                Backend::Gpu(g) => g.accel(&self.pos, &self.mass),
            },
        };
        for i in 0..self.pos.len() {
            self.vel[i][0] += a0[i][0] * dt * 0.5;
            self.vel[i][1] += a0[i][1] * dt * 0.5;
            self.vel[i][2] += a0[i][2] * dt * 0.5;
            self.pos[i][0] += self.vel[i][0] * dt;
            self.pos[i][1] += self.vel[i][1] * dt;
            self.pos[i][2] += self.vel[i][2] * dt;
        }
        let a1 = match &mut self.backend {
            Backend::Direct(d) => d.accel(&self.pos, &self.mass),
            Backend::Fmm(f) => f.accel(&self.pos, &self.mass),
            Backend::Gpu(g) => g.accel(&self.pos, &self.mass),
        };
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
        self.acc = Some(a1);
        self.time += dt;
        self.steps += 1;
    }
}

pub struct Snap {
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub time: f64,
    pub steps: u64,
}

pub struct Static {
    pub temp: Vec<f32>,
    pub mag: Vec<f32>,
    pub n: usize,
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

pub fn physics_loop(mut sim: Sim, tx: std::sync::mpsc::Sender<Msg>, ctl: std::sync::Arc<Ctl>) {
    let init_dt = *ctl.dt.lock().unwrap();
    sim.dt = init_dt;
    let n = sim.n();
    tx.send(Msg::Static(Static {
        temp: sim.temp.clone(),
        mag: sim.mag.clone(),
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
        let done = tx
            .send(Msg::Snap(Snap {
                pos: sim.pos.clone(),
                vel: sim.vel.clone(),
                time: sim.time,
                steps: sim.steps,
            }))
            .is_err();
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

    #[test]
    fn kdk_advances_finite() {
        let mut sim = Sim::new(128, 0.004, false, None);
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
        let mut a = Sim::new(64, 0.004, false, None);
        let mut b = Sim::new(64, 0.004, false, None);
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
        let sim = Sim::new(6000, 0.004, false, None);
        assert_eq!(sim.n(), 6000);
        assert!(t.elapsed() < std::time::Duration::from_secs(20));
        for i in 0..sim.n() {
            for d in 0..3 {
                assert!(sim.pos[i][d].is_finite());
            }
        }
    }

    #[test]
    fn physics_loop_streams_snaps() {
        let sim = Sim::new(128, 0.004, false, None);
        let (tx, rx) = std::sync::mpsc::channel();
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
}
