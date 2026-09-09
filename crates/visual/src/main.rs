mod camera;
mod gpu;
mod gpufmm;
mod ic;
mod render;
mod sim;

use camera::{rebase, Camera};
use gpu::{GpuItem, GpuState, Uniform};
use sim::{advance_cursor, bracket, physics_loop, target_wall_for_gap, Ctl, HistSnap, Msg, RateWindow, Sim};
use std::collections::VecDeque;
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
    gpu: bool,
    preset: ic::Preset,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let mut n = 1500usize;
    let mut dt = None::<f64>;
    let mut fmm = false;
    let mut gpu = false;
    let mut preset = ic::Preset::Cluster;
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
            "--preset" => {
                preset = a
                    .get(i + 1)
                    .and_then(|s| ic::Preset::from_str(s))
                    .unwrap_or(preset);
                i += 2;
            }
            "--fmm" => {
                fmm = true;
                i += 1;
            }
            "--gpu" => {
                gpu = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    let dt = dt.unwrap_or(if n > 20000 { 0.01 } else { 0.004 });
    Args {
        n,
        dt,
        fmm,
        gpu,
        preset,
    }
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
    gas_uni: &wgpu::Buffer,
) -> (
    wgpu::BindGroup,
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&state.gas_em_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&state.gas_dust_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (
        bind_blur_h,
        uni(&"bind-blur-uh", uni_h),
        bind_blur_v,
        uni(&"bind-blur-uv", uni_v),
        bind_blur_h2,
        bind_composite,
        tex("bind-dust-mask", &state.gas_dust_view),
        uni(&"bind-gas-uni", gas_uni),
    )
}

fn gas_pair(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    cap: usize,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
    let size = (cap * std::mem::size_of::<GpuItem>()) as u64;
    let mk = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let prev = mk("gas-prev");
    let curr = mk("gas-curr");
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gas-bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: curr.as_entire_binding(),
            },
        ],
    });
    (prev, curr, bind)
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
    bind_bright_uni: wgpu::BindGroup,
    bind_blur_h: wgpu::BindGroup,
    bind_blur_uh: wgpu::BindGroup,
    bind_blur_v: wgpu::BindGroup,
    bind_blur_uv: wgpu::BindGroup,
    bind_blur_h2: wgpu::BindGroup,
    bind_composite: wgpu::BindGroup,
    bind_bloom_uni: wgpu::BindGroup,
    bind_dust_mask: wgpu::BindGroup,
    bind_gas_uni: wgpu::BindGroup,
    gas_uni: wgpu::Buffer,
    blur_uni_h: wgpu::Buffer,
    blur_uni_v: wgpu::Buffer,
    bright_uni: wgpu::Buffer,
    bloom_uni: wgpu::Buffer,
    bg_prev: wgpu::Buffer,
    bg_curr: wgpu::Buffer,
    bg_bind: wgpu::BindGroup,
    bg_n: usize,
    emission: wgpu::RenderPipeline,
    dust: wgpu::RenderPipeline,
    gas_em_prev: wgpu::Buffer,
    gas_em_curr: wgpu::Buffer,
    gas_em_bind: wgpu::BindGroup,
    gas_dust_prev: wgpu::Buffer,
    gas_dust_curr: wgpu::Buffer,
    gas_dust_bind: wgpu::BindGroup,
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
    stage_curr: Vec<GpuItem>,
    stage_bg: Vec<GpuItem>,
    stage_gas_em: Vec<GpuItem>,
    stage_gas_dust: Vec<GpuItem>,
    temp: Vec<f32>,
    mag: Vec<f32>,
    census: [usize; 8],
    gas: Vec<ic::Gas>,
    gas_om: f32,
    body_n: usize,
    hist: VecDeque<HistSnap>,
    render_t: Option<f64>,
    sim_rate: f64,
    rate_win: RateWindow,
    cursor_slow: f64,
    sim_dt: f64,
    interval_ema: f64,
    last_arrival: Option<Instant>,
    sim_time: f64,
    sim_steps: u64,
}

/// Snap history cap: must hold the adaptive playout target (worst gap +
/// margin, up to 1.0s wall) plus catch-up margin.
/// Snap history cap: must hold the target buffer plus catch-up margin.
const HIST_CAP: usize = 24;
/// Physics channel bound: backpressure when steps outrun drains. Was 16:
/// at 1M a queued snap is ~48MB (pos+vel), so 16 deep = ~768MB of stale
/// snaps. 4 keeps the playout buffer fed (HIST_CAP=24 drains faster than
/// physics produces) while capping staleness at ~192MB worst case.
const CHAN_CAP: usize = 4;

impl App {
    fn new(args: Args) -> Self {
        let point_size = (0.035 * (1500.0 / args.n.max(1) as f32).sqrt()).clamp(0.004, 0.035);
        let sim_dt = args.dt;
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
            exposure: 1.0,
            bloom: 0.12,
            threshold: 1.5,
            radius: 1.0,
            fps: 60.0,
            phys_rate: 0.0,
            dragging: None,
            last_cursor: None,
            modifiers: winit::keyboard::ModifiersState::default(),
            bg: bg_stars(),
            stage_curr: Vec::new(),
            stage_bg: Vec::new(),
            stage_gas_em: Vec::new(),
            stage_gas_dust: Vec::new(),
            temp: Vec::new(),
            mag: Vec::new(),
            census: [0; 8],
            gas: Vec::new(),
            gas_om: 0.0,
            body_n: 0,
            hist: VecDeque::new(),
            render_t: None,
            sim_rate: 0.0,
            rate_win: RateWindow::new(4.0),
            cursor_slow: 0.0,
            sim_dt,
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
        let (device, queue) = {
            // The GPU FMM stages M2L pairs in one scratch buffer
            // (b-slots = pairs x 25 x 8B: ~280MB at 300k bodies), past the
            // 256MiB default max_buffer_size. Ask the adapter for up to 2GiB
            // (clamped to what it reports); retry with defaults on refusal.
            let alim = adapter.limits();
            let want_buf = alim.max_buffer_size.min(2 * 1024 * 1024 * 1024);
            let want_bind = alim
                .max_storage_buffer_binding_size
                .min(2 * 1024 * 1024 * 1024)
                .min(want_buf);
            let mut limits = wgpu::Limits::default();
            limits.max_buffer_size = want_buf.max(limits.max_buffer_size);
            limits.max_storage_buffer_binding_size =
                want_bind.max(limits.max_storage_buffer_binding_size);
            let desc = wgpu::DeviceDescriptor {
                label: Some("fmms"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            };
            match pollster::block_on(adapter.request_device(&desc)) {
                Ok(d) => d,
                Err(_) => {
                    eprintln!("high GPU buffer limits refused; retrying with defaults");
                    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                        label: Some("fmms"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        experimental_features: Default::default(),
                        memory_hints: Default::default(),
                        trace: Default::default(),
                    }))
                    .unwrap()
                }
            }
        };
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
            &tex_layout,
            &uni_layout,
        );
        let bright = render::bright_pipeline(
            &device,
            gpu_state.hdr_format,
            &composite_layout,
            &uni_layout,
        );
        let blur = render::blur_pipeline(&device, gpu_state.hdr_format, &tex_layout, &uni_layout);
        let tonemap = render::tonemap_pipeline(&device, format, &composite_layout, &uni_layout);
        let emission = render::emission_pipeline(
            &device,
            gpu_state.hdr_format,
            &uniform_layout,
            &storage_layout,
        );
        let dust = render::dust_pipeline(
            &device,
            gpu_state.hdr_format,
            &uniform_layout,
            &storage_layout,
        );
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
        // G2 gas params: [redden strength, full/mask res scale, 0, 0].
        let gas_uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gas-uni"),
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
        // Refresh every 8 steps: frozen-tree mean error 5.5e-3 sits on the
        // p=4/eps=1e-2 truncation floor (5.6e-3) out to k=16 (`probe
        // --gpureuse`), while refresh CPU (prepass+record ~120ms @100k)
        // amortizes 30->15ms. k=8 measured 28ms/step vs 43ms at k=4 @100k.
        let gpu_backend = if args.gpu || args.n > 30000 {
            Some(gpufmm::GpuBackend::new(&device, &queue, 8))
        } else {
            None
        };
        let sim = Sim::new(args.n, args.dt, args.fmm, gpu_backend, args.preset);
        gpu_state.n = sim.n();
        let (tx, rx) = std::sync::mpsc::sync_channel(CHAN_CAP);
        let ctl = Arc::new(Ctl::new());
        *ctl.dt.lock().unwrap() = args.dt;
        *ctl.speed.lock().unwrap() = self.speed;
        let ctl_thread = ctl.clone();
        std::thread::Builder::new()
            .name("physics".into())
            .spawn(move || physics_loop(sim, tx, ctl_thread))
            .expect("physics thread");
        let bg_n = 2200usize;
        let (gas_em_prev, gas_em_curr, gas_em_bind) = gas_pair(&device, &storage_layout, 1024);
        let (gas_dust_prev, gas_dust_curr, gas_dust_bind) =
            gas_pair(&device, &storage_layout, 1024);
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
            bind_blur_h,
            bind_blur_uh,
            bind_blur_v,
            bind_blur_uv,
            bind_blur_h2,
            bind_composite,
            bind_dust_mask,
            bind_gas_uni,
        ) = post_binds(
            &device,
            &tex_layout,
            &composite_layout,
            &uni_layout,
            &sampler,
            &gpu_state,
            &blur_uni_h,
            &blur_uni_v,
            &gas_uni,
        );
        write_post_uniforms(
            &queue,
            &config,
            &blur_uni_h,
            &blur_uni_v,
            &bright_uni,
            &bloom_uni,
            1.0,
            1.5,
            0.12,
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
            bind_bright_uni,
            bind_blur_h,
            bind_blur_uh,
            bind_blur_v,
            bind_blur_uv,
            bind_blur_h2,
            bind_composite,
            bind_bloom_uni,
            bind_dust_mask,
            bind_gas_uni,
            gas_uni,
            blur_uni_h,
            blur_uni_v,
            bright_uni,
            bloom_uni,
            bg_prev,
            bg_curr,
            bg_bind,
            bg_n,
            emission,
            dust,
            gas_em_prev,
            gas_em_curr,
            gas_em_bind,
            gas_dust_prev,
            gas_dust_curr,
            gas_dust_bind,
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
            bind_blur_h,
            bind_blur_uh,
            bind_blur_v,
            bind_blur_uv,
            bind_blur_h2,
            bind_composite,
            bind_dust_mask,
            bind_gas_uni,
        ) = post_binds(
            &gfx.device,
            &gfx.tex_layout,
            &gfx.composite_layout,
            &gfx.uni_layout,
            &gfx.sampler,
            &gfx.gpu_state,
            &gfx.blur_uni_h,
            &gfx.blur_uni_v,
            &gfx.gas_uni,
        );
        gfx.bind_blur_h = bind_blur_h;
        gfx.bind_blur_uh = bind_blur_uh;
        gfx.bind_blur_v = bind_blur_v;
        gfx.bind_blur_uv = bind_blur_uv;
        gfx.bind_blur_h2 = bind_blur_h2;
        gfx.bind_composite = bind_composite;
        gfx.bind_dust_mask = bind_dust_mask;
        gfx.bind_gas_uni = bind_gas_uni;
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
        // Drain physics snapshots into a short history. Render trails by a
        // fixed lag and interpolates between bracketing snaps, so bursty
        // arrivals (slow refresh step + fast frozen steps) render uniformly.
        if let Some(rx) = self.rx.as_ref() {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Static(s) => {
                        self.census = s.census;
                        self.temp = s.temp;
                        self.mag = s.mag;
                        self.gas_om = s.gas_om;
                        self.gas = s.gas;
                        self.body_n = s.n;
                    }
                    Msg::Snap(s) => {
                        let arrival = Instant::now();
                        self.rate_win.push(arrival, s.time);
                        if let Some(last) = self.last_arrival {
                            let dt = (arrival - last).as_secs_f64().max(1e-3);
                            self.interval_ema = self.interval_ema * 0.9 + dt * 0.1;
                            self.phys_rate = (1.0 / self.interval_ema.max(1e-3)) as f32;
                        }
                        self.last_arrival = Some(arrival);
                        self.hist.push_back(HistSnap {
                            pos: s.pos,
                            vel: s.vel,
                            gas: s.gas,
                            gas_size: s.gas_size,
                            gas_flash: s.gas_flash,
                            time: s.time,
                            steps: s.steps,
                            arrival,
                        });
                        while self.hist.len() > HIST_CAP {
                            self.hist.pop_front();
                        }
                        if let Some(h) = self.hist.back() {
                            self.sim_time = h.time;
                            self.sim_steps = h.steps;
                        }
                    }
                }
            }
        }
        // Sim-time cursor + jitter buffer. Rate comes from an exact 4s
        // window (burst-proof mean); the buffer target covers the worst
        // recent gap + margin, so refresh bursts never starve the bracket;
        // the weak spring on filtered occupancy centers the level with <5%
        // ripple instead of hunting. Startup seeds from the instant rate
        // until the window fills.
        let tw = target_wall_for_gap(self.rate_win.max_gap(now));
        if let (Some(f), Some(b)) = (self.hist.front(), self.hist.back()) {
            let n = self.hist.len();
            if n >= 2 {
                let span = (b.time - f.time).max(1e-12);
                self.sim_dt = (span / (n as f64 - 1.0)).max(1e-9);
                let r = self.rate_win.rate();
                if r > 0.0 {
                    self.sim_rate = r;
                } else {
                    let wall = (b.arrival - f.arrival).as_secs_f64().max(1e-6);
                    self.sim_rate = span / wall;
                }
            }
            let mut rt = match self.render_t {
                Some(t) => t,
                None => {
                    let target =
                        (self.sim_rate * tw).clamp(2.0 * self.sim_dt, 12.0 * self.sim_dt);
                    if self.paused {
                        b.time
                    } else {
                        (b.time - target).max(f.time)
                    }
                }
            };
            if !self.paused && self.sim_rate > 0.0 {
                rt = advance_cursor(
                    rt,
                    f.time,
                    b.time,
                    self.sim_rate,
                    self.sim_dt,
                    real_dt.min(0.25),
                    tw,
                    &mut self.cursor_slow,
                );
            }
            while self.hist.len() > HIST_CAP {
                self.hist.pop_front();
            }
            if let Some(h) = self.hist.front() {
                rt = rt.max(h.time);
            }
            while self.hist.len() >= 3 && self.hist[1].time <= rt {
                self.hist.pop_front();
            }
            self.render_t = Some(rt);
        }
        let (lo, alpha, extra) = match self.render_t {
            Some(t) => bracket(&self.hist, t),
            None => (0, 1.0, 0.0),
        };
        let (p0, p1, v0, v1) = match (self.hist.get(lo), self.hist.get(lo + 1)) {
            (Some(a), Some(b)) => (&a.pos[..], &b.pos[..], &a.vel[..], &b.vel[..]),
            (Some(a), None) => (&a.pos[..], &a.pos[..], &a.vel[..], &a.vel[..]),
            _ => (&[][..], &[][..], &[][..], &[][..]),
        };

        let gfx = self.gfx.as_mut().unwrap();
        let eye = self.camera.eye();
        let n = self.body_n.min(p0.len().min(p1.len()));
        // Curr-only staging: the shader reads curr directly (alpha=1), so
        // the old prev buffer (build + upload, 32B/body/frame) was pure
        // waste. .speed is likewise never read by any shader: no sqrt.
        self.stage_curr.clear();
        self.stage_curr.reserve(n);
        for i in 0..n {
            let a = alpha;
            let q0 = p0[i];
            let q1 = p1[i];
            let u0 = v0[i];
            let u1 = v1[i];
            let v = [
                u0[0] + (u1[0] - u0[0]) * a,
                u0[1] + (u1[1] - u0[1]) * a,
                u0[2] + (u1[2] - u0[2]) * a,
            ];
            let w = [
                q0[0] + (q1[0] - q0[0]) * a + v[0] * extra,
                q0[1] + (q1[1] - q0[1]) * a + v[1] * extra,
                q0[2] + (q1[2] - q0[2]) * a + v[2] * extra,
            ];
            self.stage_curr.push(GpuItem {
                pos: rebase(&w, &eye),
                speed: 0.0,
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
        self.stage_gas_em.clear();
        self.stage_gas_dust.clear();
        // M1: gas rides live snaps (interpolated like stars), not the old
        // rigid pattern carousel. Meta (kind/seed) stays static; size and
        // flash ride the snaps (gradual merge swell + supernova fade).
        let (g0, g1) = match (self.hist.get(lo), self.hist.get(lo + 1)) {
            (Some(a), Some(b)) => (&a.gas[..], &b.gas[..]),
            (Some(a), None) => (&a.gas[..], &a.gas[..]),
            _ => (&[][..], &[][..]),
        };
        let (s0, s1) = match (self.hist.get(lo), self.hist.get(lo + 1)) {
            (Some(a), Some(b)) => (&a.gas_size[..], &b.gas_size[..]),
            (Some(a), None) => (&a.gas_size[..], &a.gas_size[..]),
            _ => (&[][..], &[][..]),
        };
        let (f0, f1) = match (self.hist.get(lo), self.hist.get(lo + 1)) {
            (Some(a), Some(b)) => (&a.gas_flash[..], &b.gas_flash[..]),
            (Some(a), None) => (&a.gas_flash[..], &a.gas_flash[..]),
            _ => (&[][..], &[][..]),
        };
        let gn = self.gas.len().min(
            g0.len()
                .min(g1.len())
                .min(s0.len().min(s1.len()).min(f0.len().min(f1.len()))),
        );
        for i in 0..gn {
            let g = &self.gas[i];
            // Live size rides the snaps: merged-away knots read 0 here
            // (init meta would resurrect them as frozen statues).
            let size = s0[i] + (s1[i] - s0[i]) * alpha as f32;
            if size <= 0.0 {
                continue;
            }
            let flash = (f0[i] + (f1[i] - f0[i]) * alpha as f32).clamp(0.0, 1.0);
            let (a, b) = (g0[i], g1[i]);
            let w = [
                a[0] + (b[0] - a[0]) * alpha,
                a[1] + (b[1] - a[1]) * alpha,
                a[2] + (b[2] - a[2]) * alpha,
            ];
            let item = GpuItem {
                pos: rebase(&w, &eye),
                speed: 0.0,
                misc: [g.seed, size, g.kind, flash],
            };
            if g.kind < 2.5 {
                if self.stage_gas_em.len() < 1024 {
                    self.stage_gas_em.push(item);
                }
            } else if self.stage_gas_dust.len() < 1024 {
                self.stage_gas_dust.push(item);
            }
        }
        // Back-to-front blend order: far quads first so near translucent
        // dust/emission composites correctly (~2k items, sort is trivial).
        let fwd = self.camera.forward();
        let depth = |it: &GpuItem| it.pos[0] * fwd.x + it.pos[1] * fwd.y + it.pos[2] * fwd.z;
        self.stage_gas_em
            .sort_by(|a, b| depth(b).partial_cmp(&depth(a)).unwrap());
        self.stage_gas_dust
            .sort_by(|a, b| depth(b).partial_cmp(&depth(a)).unwrap());
        gfx.queue.write_buffer(
            &gfx.gas_em_prev,
            0,
            bytemuck::cast_slice(&self.stage_gas_em),
        );
        gfx.queue.write_buffer(
            &gfx.gas_em_curr,
            0,
            bytemuck::cast_slice(&self.stage_gas_em),
        );
        gfx.queue.write_buffer(
            &gfx.gas_dust_prev,
            0,
            bytemuck::cast_slice(&self.stage_gas_dust),
        );
        gfx.queue.write_buffer(
            &gfx.gas_dust_curr,
            0,
            bytemuck::cast_slice(&self.stage_gas_dust),
        );
        let em_n = self.stage_gas_em.len();
        let dust_n = self.stage_gas_dust.len();
        let aspect = gfx.config.width as f32 / gfx.config.height.max(1) as f32;
        let vp = self.camera.view_proj(aspect);
        let (right, up) = self.camera.basis();
        let uni = Uniform {
            view_proj: vp.to_cols_array_2d(),
            // .w carries eye XZ so gas shaders can anchor noise in true
            // world space (staging holds rebased world-eye positions).
            cam_right: [right.x, right.y, right.z, eye[0] as f32],
            cam_up: [up.x, up.y, up.z, eye[2] as f32],
            // Staging buffers already hold the interpolated state; the
            // shader must not interpolate a second time (that would square
            // the blend into an easing curve).
            params: [
                self.point_size,
                1.0,
                self.exposure,
                self.render_t.unwrap_or(0.0) as f32,
            ],
        };
        gfx.gpu_state
            .upload(&gfx.queue, &self.stage_curr, &uni);
        gfx.queue.write_buffer(
            &gfx.gas_uni,
            0,
            bytemuck::bytes_of(&[1.0f32, 4.0, 0.0, 0.0]),
        );
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
                ui.label(format!(
                    "OBAFGKM+G: {} {} {} {} {} {} {} {}",
                    self.census[0],
                    self.census[1],
                    self.census[2],
                    self.census[3],
                    self.census[4],
                    self.census[5],
                    self.census[6],
                    self.census[7]
                ));
                ui.label(format!("fps: {fps:.0}  phys: {phys:.1}/s"));
                ui.label(format!("sim t: {sim_time:.2}"));
                ui.label(format!("steps: {sim_steps}"));
                ui.label(format!("upload/frame: {bytes_kb} KB"));
                ui.checkbox(&mut self.paused, "paused (space)");
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=10.0).text("speed"));
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
            // G2 split-res gas pass: emission HDR + dust mask at quarter res.
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gas-em"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gfx.gpu_state.gas_em_view,
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
            pass.set_pipeline(&gfx.emission);
            pass.set_bind_group(0, &gfx.gpu_state.uniform_bind, &[]);
            pass.set_bind_group(1, &gfx.gas_em_bind, &[]);
            pass.draw(0..52, 0..em_n as u32);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gas-dust"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gfx.gpu_state.gas_dust_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // rgb = 0 tau, alpha = far (Min blend keeps nearest).
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 10000.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&gfx.dust);
            pass.set_bind_group(0, &gfx.gpu_state.uniform_bind, &[]);
            pass.set_bind_group(1, &gfx.gas_dust_bind, &[]);
            pass.draw(0..52, 0..dust_n as u32);
        }
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
            pass.set_bind_group(2, &gfx.bind_dust_mask, &[]);
            pass.set_bind_group(3, &gfx.bind_gas_uni, &[]);
            pass.draw(0..4, 0..n as u32);
            pass.set_bind_group(1, &gfx.bg_bind, &[]);
            pass.draw(0..4, 0..bg_n as u32);
        }
        post_pass(
            &mut enc,
            "bright",
            &gfx.gpu_state.bright_view,
            &gfx.bright,
            &[&gfx.bind_composite, &gfx.bind_bright_uni],
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
