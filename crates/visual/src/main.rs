mod camera;
mod gpu;
mod render;
mod sim;

use camera::{rebase, Camera};
use gpu::{GpuItem, GpuState, Uniform};
use sim::{physics_loop, Ctl, Msg, Sim};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

struct Args {
    n: usize,
    dt: f64,
    fmm: bool,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let mut n = 1500usize;
    let mut dt = None::<f64>;
    let mut fmm = false;
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--n" => {
                n = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(n);
                i += 2;
            }
            "--dt" => {
                dt = a.get(i + 1).and_then(|s| s.parse().ok()).or(dt);
                i += 2;
            }
            "--fmm" => {
                fmm = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    let dt = dt.unwrap_or(if n > 20000 { 0.01 } else { 0.004 });
    Args { n, dt, fmm }
}

fn post_binds(
    device: &wgpu::Device,
    tex_layout: &wgpu::BindGroupLayout,
    composite_layout: &wgpu::BindGroupLayout,
    uni_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    state: &GpuState,
    uni_h: &wgpu::Buffer,
    uni_v: &wgpu::Buffer,
) -> (
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
) {
    let tex = |label: &str, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    };
    let bind_bright = tex("bind-bright", &state.hdr_view);
    let bind_blur_h = tex("bind-blur-h", &state.bright_view);
    let bind_blur_v = tex("bind-blur-v", &state.blur_a);
    let bind_blur_h2 = tex("bind-blur-h2", &state.blur_b);
    let uni = |label: &str, buf: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: uni_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    };
    let bind_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind-composite"),
        layout: composite_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&state.hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&state.blur_b),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (
        bind_bright,
        bind_blur_h,
        uni(&"bind-blur-uh", uni_h),
        bind_blur_v,
        uni(&"bind-blur-uv", uni_v),
        bind_blur_h2,
        bind_composite,
    )
}

fn bg_stars() -> Vec<([f64; 3], f32, f32)> {
    let mut rng = fmm::models::Rng::new(99);
    let mut out = Vec::with_capacity(2200);
    for _ in 0..2200 {
        let z = 2.0 * rng.unit() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * rng.unit();
        let s = (1.0 - z * z).sqrt();
        let r = 300.0;
        out.push((
            [r * s * phi.cos(), r * s * phi.sin(), r * z],
            rng.unit() as f32,
            (rng.unit().powf(2.0) * 0.35 + 0.05) as f32,
        ));
    }
    out
}

struct Graphics {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    gpu_state: GpuState,
    billboard: wgpu::RenderPipeline,
    bright: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    tonemap: wgpu::RenderPipeline,
    tex_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    uni_layout: wgpu::BindGroupLayout,
    bind_bright: wgpu::BindGroup,
    bind_bright_uni: wgpu::BindGroup,
    bind_blur_h: wgpu::BindGroup,
    bind_blur_uh: wgpu::BindGroup,
    bind_blur_v: wgpu::BindGroup,
    bind_blur_uv: wgpu::BindGroup,
    bind_blur_h2: wgpu::BindGroup,
    bind_composite: wgpu::BindGroup,
    bind_bloom_uni: wgpu::BindGroup,
    blur_uni_h: wgpu::Buffer,
    blur_uni_v: wgpu::Buffer,
    bright_uni: wgpu::Buffer,
    bloom_uni: wgpu::Buffer,
    bg_prev: wgpu::Buffer,
    bg_curr: wgpu::Buffer,
    bg_bind: wgpu::BindGroup,
    bg_n: usize,
    sampler: wgpu::Sampler,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    egui_ctx: egui::Context,
}

struct App {
    args: Option<Args>,
    gfx: Option<Graphics>,
    rx: Option<std::sync::mpsc::Receiver<Msg>>,
    ctl: Option<Arc<Ctl>>,
    camera: Camera,
    last: Instant,
    paused: bool,
    speed: f32,
    point_size: f32,
    exposure: f32,
    bloom: f32,
    threshold: f32,
    radius: f32,
    fps: f32,
    phys_rate: f32,
    dragging: Option<MouseButton>,
    last_cursor: Option<winit::dpi::PhysicalPosition<f64>>,
    modifiers: winit::keyboard::ModifiersState,
    bg: Vec<([f64; 3], f32, f32)>,
    stage_prev: Vec<GpuItem>,
    stage_curr: Vec<GpuItem>,
    stage_bg: Vec<GpuItem>,
    r0pos: Vec<[f64; 3]>,
    r1pos: Vec<[f64; 3]>,
    r0vel: Vec<[f64; 3]>,
    r1vel: Vec<[f64; 3]>,
    temp: Vec<f32>,
    mag: Vec<f32>,
    body_n: usize,
    blend: f64,
    interval_ema: f64,
    last_arrival: Option<Instant>,
    sim_time: f64,
    sim_steps: u64,
}

impl App {
    fn new(args: Args) -> Self {
        let point_size = (0.035 * (1500.0 / args.n.max(1) as f32).sqrt()).clamp(0.004, 0.035);
        Self {
            args: Some(args),
            gfx: None,
            rx: None,
            ctl: None,
            camera: Camera::new(),
            last: Instant::now(),
            paused: false,
            speed: 1.0,
            point_size,
            exposure: 1.4,
            bloom: 0.35,
            threshold: 1.5,
            radius: 2.0,
            fps: 60.0,
            phys_rate: 0.0,
            dragging: None,
            last_cursor: None,
            modifiers: winit::keyboard::ModifiersState::default(),
            bg: bg_stars(),
            stage_prev: Vec::new(),
            stage_curr: Vec::new(),
            stage_bg: Vec::new(),
            r0pos: Vec::new(),
            r1pos: Vec::new(),
            r0vel: Vec::new(),
            r1vel: Vec::new(),
            temp: Vec::new(),
            mag: Vec::new(),
            body_n: 0,
            blend: 1.0,
            interval_ema: 0.25,
            last_arrival: None,
            sim_time: 0.0,
            sim_steps: 0,
        }
    }

    fn init_graphics(&mut self, event_loop: &ActiveEventLoop) {
        let args = self.args.take().unwrap();
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("fmms-visual V0"))
                .unwrap(),
        );
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fmms"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        }))
        .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let (uniform_layout, storage_layout) = render::layouts(&device);
        let tex_layout = render::tex_layout(&device, "tex-layout");
        let composite_layout = render::composite_layout(&device);
        let uni_layout = render::blur_uni_layout(&device);
        let mut gpu_state = GpuState::new(
            &device,
            &uniform_layout,
            &storage_layout,
            args.n,
            config.width,
            config.height,
        );
        let billboard = render::billboard_pipeline(
            &device,
            gpu_state.hdr_format,
            &uniform_layout,
            &storage_layout,
        );
        let bright =
            render::bright_pipeline(&device, gpu_state.hdr_format, &tex_layout, &uni_layout);
        let blur = render::blur_pipeline(&device, gpu_state.hdr_format, &tex_layout, &uni_layout);
        let tonemap = render::tonemap_pipeline(&device, format, &composite_layout, &uni_layout);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post-smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blur_uni_h = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur-uni-h"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_uni_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur-uni-v"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bright_uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bright-uni"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bloom_uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom-uni"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uni_bind = |label: &str, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &uni_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        };
        let bind_bright_uni = uni_bind("bind-bright-uni", &bright_uni);
        let bind_bloom_uni = uni_bind("bind-bloom-uni", &bloom_uni);
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(2048),
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
        let sim = Sim::new(args.n, args.dt, args.fmm);
        gpu_state.n = sim.n();
        let (tx, rx) = std::sync::mpsc::channel();
        let ctl = Arc::new(Ctl::new());
        *ctl.dt.lock().unwrap() = args.dt;
        *ctl.speed.lock().unwrap() = self.speed;
        let ctl_thread = ctl.clone();
        std::thread::Builder::new()
            .name("physics".into())
            .spawn(move || physics_loop(sim, tx, ctl_thread))
            .expect("physics thread");
        let bg_n = 2200usize;
        let bg_size = (bg_n * std::mem::size_of::<GpuItem>()) as u64;
        let bg_prev = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bg-prev"),
            size: bg_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_curr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bg-curr"),
            size: bg_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-bind"),
            layout: &storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bg_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bg_curr.as_entire_binding(),
                },
            ],
        });
        let (
            bind_bright,
            bind_blur_h,
            bind_blur_uh,
            bind_blur_v,
            bind_blur_uv,
            bind_blur_h2,
            bind_composite,
        ) = post_binds(
            &device,
            &tex_layout,
            &composite_layout,
            &uni_layout,
            &sampler,
            &gpu_state,
            &blur_uni_h,
            &blur_uni_v,
        );
        write_post_uniforms(
            &queue,
            &config,
            &blur_uni_h,
            &blur_uni_v,
            &bright_uni,
            &bloom_uni,
            1.4,
            1.5,
            0.35,
        );
        let gfx = Graphics {
            window,
            surface,
            device,
            queue,
            config,
            size,
            gpu_state,
            billboard,
            bright,
            blur,
            tonemap,
            tex_layout,
            composite_layout,
            uni_layout,
            bind_bright,
            bind_bright_uni,
            bind_blur_h,
            bind_blur_uh,
            bind_blur_v,
            bind_blur_uv,
            bind_blur_h2,
            bind_composite,
            bind_bloom_uni,
            blur_uni_h,
            blur_uni_v,
            bright_uni,
            bloom_uni,
            bg_prev,
            bg_curr,
            bg_bind,
            bg_n,
            sampler,
            egui_renderer,
            egui_state,
            egui_ctx,
        };
        self.rx = Some(rx);
        self.ctl = Some(ctl);
        self.gfx = Some(gfx);
        self.last = Instant::now();
    }

    fn rebuild_hdr_bind(&mut self) {
        let gfx = self.gfx.as_mut().unwrap();
        let (
            bind_bright,
            bind_blur_h,
            bind_blur_uh,
            bind_blur_v,
            bind_blur_uv,
            bind_blur_h2,
            bind_composite,
        ) = post_binds(
            &gfx.device,
            &gfx.tex_layout,
            &gfx.composite_layout,
            &gfx.uni_layout,
            &gfx.sampler,
            &gfx.gpu_state,
            &gfx.blur_uni_h,
            &gfx.blur_uni_v,
        );
        gfx.bind_bright = bind_bright;
        gfx.bind_blur_h = bind_blur_h;
        gfx.bind_blur_uh = bind_blur_uh;
        gfx.bind_blur_v = bind_blur_v;
        gfx.bind_blur_uv = bind_blur_uv;
        gfx.bind_blur_h2 = bind_blur_h2;
        gfx.bind_composite = bind_composite;
        let exposure = self.exposure;
        let threshold = self.threshold;
        let bloom = self.bloom;
        write_post_uniforms(
            &gfx.queue,
            &gfx.config,
            &gfx.blur_uni_h,
            &gfx.blur_uni_v,
            &gfx.bright_uni,
            &gfx.bloom_uni,
            exposure,
            threshold,
            bloom,
        );
    }

    fn render_frame(&mut self) {
        let now = Instant::now();
        let real_dt = (now - self.last).as_secs_f64();
        self.last = now;
        if real_dt > 0.0 {
            self.fps = self.fps * 0.95 + (1.0 / real_dt.max(1e-6)) as f32 * 0.05;
        }
        // Drain physics snapshots, keep newest. Stretch each snapshot over
        // its arrival interval so motion stays smooth at any step rate.
        if let Some(rx) = self.rx.as_ref() {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Static(s) => {
                        self.temp = s.temp;
                        self.mag = s.mag;
                        self.body_n = s.n;
                    }
                    Msg::Snap(s) => {
                        let arrival = Instant::now();
                        if let Some(last) = self.last_arrival {
                            let dt = (arrival - last).as_secs_f64().max(1e-3);
                            self.interval_ema = self.interval_ema * 0.9 + dt * 0.1;
                            self.phys_rate = (1.0 / self.interval_ema.max(1e-3)) as f32;
                        }
                        self.last_arrival = Some(arrival);
                        if self.r1pos.is_empty() {
                            self.r0pos = s.pos.clone();
                            self.r0vel = s.vel.clone();
                            self.r1pos = s.pos;
                            self.r1vel = s.vel;
                            self.blend = 1.0;
                        } else {
                            let b = self.blend.clamp(0.0, 1.0);
                            for i in 0..self.r0pos.len().min(s.pos.len()) {
                                for d in 0..3 {
                                    self.r0pos[i][d] += (self.r1pos[i][d] - self.r0pos[i][d]) * b;
                                    self.r0vel[i][d] += (self.r1vel[i][d] - self.r0vel[i][d]) * b;
                                }
                            }
                            self.r1pos = s.pos;
                            self.r1vel = s.vel;
                            self.blend = 0.0;
                        }
                        self.sim_time = s.time;
                        self.sim_steps = s.steps;
                    }
                }
            }
        }
        if !self.paused {
            self.blend = (self.blend + real_dt / self.interval_ema.max(1e-3)).min(1.0);
        }

        let gfx = self.gfx.as_mut().unwrap();
        let eye = self.camera.eye();
        let n = self.body_n.min(self.r1pos.len());
        let alpha = self.blend.clamp(0.0, 1.0) as f32;
        self.stage_prev.clear();
        self.stage_curr.clear();
        self.stage_prev.reserve(n);
        self.stage_curr.reserve(n);
        for i in 0..n {
            let a = alpha as f64;
            let p0 = self.r0pos[i];
            let p1 = self.r1pos[i];
            let v0 = self.r0vel[i];
            let v1 = self.r1vel[i];
            let w = [
                p0[0] + (p1[0] - p0[0]) * a,
                p0[1] + (p1[1] - p0[1]) * a,
                p0[2] + (p1[2] - p0[2]) * a,
            ];
            let v = [
                v0[0] + (v1[0] - v0[0]) * a,
                v0[1] + (v1[1] - v0[1]) * a,
                v0[2] + (v1[2] - v0[2]) * a,
            ];
            let psp = (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]).sqrt() as f32;
            let sp = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() as f32;
            self.stage_prev.push(GpuItem {
                pos: rebase(&p0, &eye),
                speed: psp,
                misc: [self.temp[i], self.mag[i], 0.0, 0.0],
            });
            self.stage_curr.push(GpuItem {
                pos: rebase(&w, &eye),
                speed: sp,
                misc: [self.temp[i], self.mag[i], 0.0, 0.0],
            });
        }
        let bg_n = self.bg.len().min(gfx.bg_n);
        self.stage_bg.clear();
        self.stage_bg.reserve(bg_n);
        for (w, t, m) in self.bg.iter().take(bg_n) {
            self.stage_bg.push(GpuItem {
                pos: rebase(w, &eye),
                speed: 0.0,
                misc: [*t, *m, 0.001 + *m * 0.002, 0.0],
            });
        }
        gfx.queue
            .write_buffer(&gfx.bg_prev, 0, bytemuck::cast_slice(&self.stage_bg));
        gfx.queue
            .write_buffer(&gfx.bg_curr, 0, bytemuck::cast_slice(&self.stage_bg));
        let aspect = gfx.config.width as f32 / gfx.config.height.max(1) as f32;
        let vp = self.camera.view_proj(aspect);
        let (right, up) = self.camera.basis();
        let uni = Uniform {
            view_proj: vp.to_cols_array_2d(),
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            // Staging buffers already hold the interpolated state; the
            // shader must not interpolate a second time (that would square
            // the blend into an easing curve).
            params: [self.point_size, 1.0, self.exposure, 0.0],
        };
        gfx.gpu_state
            .upload(&gfx.queue, &self.stage_prev, &self.stage_curr, &uni);
        write_post_uniforms(
            &gfx.queue,
            &gfx.config,
            &gfx.blur_uni_h,
            &gfx.blur_uni_v,
            &gfx.bright_uni,
            &gfx.bloom_uni,
            self.exposure,
            self.threshold,
            self.bloom,
        );

        let raw_input = gfx.egui_state.take_egui_input(&gfx.window);
        let fps = self.fps;
        let phys = self.phys_rate;
        let bytes_kb = gfx.gpu_state.bytes_per_frame / 1024;
        let sim_time = self.sim_time;
        let sim_steps = self.sim_steps;
        // run_ui (not manual begin/end_pass): registers the root background
        // rect, so is_pointer_over_egui is false in the 3D view and
        // EventResponse::consumed stops claiming every press/wheel.
        let mut full = gfx.egui_ctx.run_ui(raw_input, |ui| {
            egui::Window::new("fmms V0").show(ui.ctx(), |ui| {
                ui.label(format!("bodies: {n}"));
                ui.label(format!("fps: {fps:.0}  phys: {phys:.1}/s"));
                ui.label(format!("sim t: {sim_time:.2}"));
                ui.label(format!("steps: {sim_steps}"));
                ui.label(format!("upload/frame: {bytes_kb} KB"));
                ui.checkbox(&mut self.paused, "paused (space)");
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=40.0).text("speed"));
                ui.add(egui::Slider::new(&mut self.point_size, 0.002..=0.2).text("size"));
                ui.add(egui::Slider::new(&mut self.exposure, 0.2..=5.0).text("exposure"));
                ui.add(egui::Slider::new(&mut self.bloom, 0.0..=2.0).text("bloom"));
                ui.add(egui::Slider::new(&mut self.threshold, 0.1..=2.0).text("threshold"));
                ui.add(egui::Slider::new(&mut self.radius, 1.0..=3.0).text("bloom size"));
                egui::CollapsingHeader::new("controls")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label("orbit: LMB / MMB drag");
                        ui.label("pan: RMB drag, Shift+LMB drag");
                        ui.label("zoom: wheel, Ctrl+MMB drag");
                    });
            });
        });
        // Forward UI state to the physics thread (checked every step).
        if let Some(ctl) = self.ctl.as_ref() {
            ctl.paused
                .store(self.paused, std::sync::atomic::Ordering::Relaxed);
            if let Ok(mut s) = ctl.speed.lock() {
                *s = self.speed;
            }
        }
        gfx.egui_state
            .handle_platform_output(&gfx.window, full.platform_output);
        let tris = gfx.egui_ctx.tessellate(full.shapes, full.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gfx.config.width, gfx.config.height],
            pixels_per_point: full.pixels_per_point,
        };

        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.device, &gfx.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gfx.gpu_state.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: render::SPACE_BG[0] as f64,
                            g: render::SPACE_BG[1] as f64,
                            b: render::SPACE_BG[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&gfx.billboard);
            pass.set_bind_group(0, &gfx.gpu_state.uniform_bind, &[]);
            pass.set_bind_group(1, &gfx.gpu_state.storage_bind, &[]);
            pass.draw(0..4, 0..n as u32);
            pass.set_bind_group(1, &gfx.bg_bind, &[]);
            pass.draw(0..4, 0..bg_n as u32);
        }
        post_pass(
            &mut enc,
            "bright",
            &gfx.gpu_state.bright_view,
            &gfx.bright,
            &[&gfx.bind_bright, &gfx.bind_bright_uni],
        );
        let pairs = self.radius.round().clamp(1.0, 3.0) as u32;
        for i in 0..pairs {
            let h_bind = if i == 0 {
                &gfx.bind_blur_h
            } else {
                &gfx.bind_blur_h2
            };
            post_pass(
                &mut enc,
                "blur-h",
                &gfx.gpu_state.blur_a,
                &gfx.blur,
                &[h_bind, &gfx.bind_blur_uh],
            );
            post_pass(
                &mut enc,
                "blur-v",
                &gfx.gpu_state.blur_b,
                &gfx.blur,
                &[&gfx.bind_blur_v, &gfx.bind_blur_uv],
            );
        }
        post_pass(
            &mut enc,
            "tonemap",
            &view,
            &gfx.tonemap,
            &[&gfx.bind_composite, &gfx.bind_bloom_uni],
        );
        let user_cmds = {
            for (id, deltas) in std::mem::take(&mut full.textures_delta.set) {
                for delta in &deltas {
                    gfx.egui_renderer
                        .update_texture(&gfx.device, &gfx.queue, id, delta);
                }
            }
            for id in std::mem::take(&mut full.textures_delta.free) {
                gfx.egui_renderer.free_texture(&id);
            }
            gfx.egui_renderer
                .update_buffers(&gfx.device, &gfx.queue, &mut enc, &tris, &screen)
        };
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            gfx.egui_renderer.render(&mut pass, &tris, &screen);
        }
        gfx.queue
            .submit(user_cmds.into_iter().chain([enc.finish()]));
        gfx.window.pre_present_notify();
        gfx.queue.present(frame);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_none() {
            self.init_graphics(event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.gfx.is_none() {
            return;
        }
        let window = self.gfx.as_ref().unwrap().window.clone();
        let response = self
            .gfx
            .as_mut()
            .unwrap()
            .egui_state
            .on_window_event(&window, &event);
        let consumed = response.consumed;
        if response.repaint {
            window.request_redraw();
        }
        // Per egui-winit docs (EventResponse::consumed): only pass to the
        // camera when egui did NOT consume the event. An active drag keeps
        // pointer capture until its Release, even over UI.
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                {
                    let gfx = self.gfx.as_mut().unwrap();
                    gfx.size = size;
                    gfx.config.width = size.width.max(1);
                    gfx.config.height = size.height.max(1);
                    gfx.surface.configure(&gfx.device, &gfx.config);
                    let (w, h) = (gfx.config.width, gfx.config.height);
                    gfx.gpu_state.resize(&gfx.device, w, h);
                }
                self.rebuild_hdr_bind();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if consumed {
                    return;
                }
                if event.state == ElementState::Pressed
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Space))
                {
                    self.paused = !self.paused;
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed {
                    // Press on UI belongs to egui; never start a camera drag.
                    if consumed {
                        return;
                    }
                    self.dragging = Some(button);
                } else if self.dragging == Some(button) || !consumed {
                    // Release always ends capture, even over UI.
                    self.dragging = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(last) = self.last_cursor {
                    let dx = (position.x - last.x) as f32;
                    let dy = (position.y - last.y) as f32;
                    let shift = self.modifiers.shift_key();
                    let ctrl = self.modifiers.control_key();
                    match self.dragging {
                        Some(MouseButton::Left) | Some(MouseButton::Middle) if shift => {
                            self.camera.pan(dx, dy)
                        }
                        Some(MouseButton::Middle) if ctrl => self.camera.zoom(dy * 0.03),
                        Some(MouseButton::Left) | Some(MouseButton::Middle) => {
                            self.camera.orbit(dx, dy)
                        }
                        Some(MouseButton::Right) => self.camera.pan(dx, dy),
                        _ => {}
                    }
                }
                self.last_cursor = Some(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Wheel over UI (e.g. hovering a slider) belongs to egui;
                // plain wheel elsewhere zooms, no mouse button needed.
                if consumed {
                    return;
                }
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 50.0, p.y as f32 / 50.0),
                };
                if self.modifiers.shift_key() {
                    self.camera.pan(dx * 30.0, -dy * 30.0);
                } else {
                    self.camera.zoom(dx + dy);
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                self.gfx.as_ref().unwrap().window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = self.gfx.as_ref() {
            gfx.window.request_redraw();
        }
    }
}

fn post_pass(
    enc: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    binds: &[&wgpu::BindGroup],
) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    for (i, b) in binds.iter().enumerate() {
        pass.set_bind_group(i as u32, Some(*b), &[]);
    }
    pass.draw(0..3, 0..1);
}

fn write_post_uniforms(
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    blur_h: &wgpu::Buffer,
    blur_v: &wgpu::Buffer,
    bright: &wgpu::Buffer,
    bloom: &wgpu::Buffer,
    exposure: f32,
    threshold: f32,
    strength: f32,
) {
    let bw = (config.width / 4).max(1) as f32;
    let bh = (config.height / 4).max(1) as f32;
    queue.write_buffer(
        bright,
        0,
        bytemuck::bytes_of(&[1.0 / bw, 1.0 / bh, 1.0 / exposure.max(1e-3), threshold]),
    );
    queue.write_buffer(bloom, 0, bytemuck::bytes_of(&[strength, 0.0f32, 0.0, 0.0]));
    queue.write_buffer(
        blur_h,
        0,
        bytemuck::bytes_of(&[1.0f32, 0.0, 1.0 / bw, 1.0 / bh]),
    );
    queue.write_buffer(
        blur_v,
        0,
        bytemuck::bytes_of(&[0.0f32, 1.0, 1.0 / bw, 1.0 / bh]),
    );
}

fn main() {
    let args = parse_args();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(args);
    event_loop.run_app(&mut app).unwrap();
}
