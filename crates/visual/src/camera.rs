use glam::{Mat4, Vec3};

pub struct Camera {
    pub target: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            yaw: 0.6,
            pitch: 0.45,
            dist: 6.0,
            fov_y: 55.0,
        }
    }

    pub fn eye(&self) -> [f64; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let d = self.dist as f64;
        let sy = sy as f64;
        let cy = cy as f64;
        let sp = sp as f64;
        let cp = cp as f64;
        [
            self.target[0] + d * cp * sy,
            self.target[1] + d * sp,
            self.target[2] + d * cp * cy,
        ]
    }

    pub fn forward(&self) -> Vec3 {
        let e = self.eye();
        let f = Vec3::new(
            (self.target[0] - e[0]) as f32,
            (self.target[1] - e[1]) as f32,
            (self.target[2] - e[2]) as f32,
        );
        f.normalize()
    }

    pub fn basis(&self) -> (Vec3, Vec3) {
        let fwd = self.forward();
        let world_up = Vec3::Y;
        let right = fwd.cross(world_up).normalize_or(Vec3::X);
        let up = right.cross(fwd).normalize_or(Vec3::Y);
        (right, up)
    }

    /// View-projection for camera-rebased positions.
    ///
    /// Buffers hold `world - eye` (see `rebase`), so the camera sits at the
    /// origin and looks at `target - eye`. Using a full world-space
    /// `look_at(eye, target)` here would subtract the eye offset twice and
    /// make the scene swim when orbiting/panning.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let e = self.eye();
        let rel = Vec3::new(
            (self.target[0] - e[0]) as f32,
            (self.target[1] - e[1]) as f32,
            (self.target[2] - e[2]) as f32,
        );
        let view = Mat4::look_at_rh(Vec3::ZERO, rel, Vec3::Y);
        let proj = Mat4::perspective_rh_gl(
            self.fov_y.to_radians(),
            aspect,
            (self.dist * 1e-3).max(1e-4),
            (self.dist * 50.0 + 100.0).max(800.0),
        );
        proj * view
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.dist = (self.dist * (1.0 - delta * 0.1)).clamp(0.2, 200.0);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up) = self.basis();
        let s = self.dist * 0.0015;
        let t = [
            self.target[0] - (right.x * dx - up.x * dy) as f64 * s as f64,
            self.target[1] - (right.y * dx - up.y * dy) as f64 * s as f64,
            self.target[2] - (right.z * dx - up.z * dy) as f64 * s as f64,
        ];
        self.target = t;
    }
}

pub fn rebase(world: &[f64; 3], eye: &[f64; 3]) -> [f32; 3] {
    [
        (world[0] - eye[0]) as f32,
        (world[1] - eye[1]) as f32,
        (world[2] - eye[2]) as f32,
    ]
}
