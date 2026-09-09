use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuItem {
    pub pos: [f32; 3],
    pub speed: f32,
    pub misc: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniform {
    pub view_proj: [[f32; 4]; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub params: [f32; 4],
}

pub struct GpuState {
    /// Stale by design: kept allocated+bound so the storage layout (and the
    /// gas bind groups sharing it) never churns. Nothing stages, uploads,
    /// or shades from it since the billboard went curr-only.
    #[allow(dead_code)]
    pub prev_buf: wgpu::Buffer,
    pub curr_buf: wgpu::Buffer,
    pub uniform_buf: wgpu::Buffer,
    pub uniform_bind: wgpu::BindGroup,
    pub storage_bind: wgpu::BindGroup,
    pub hdr_tex: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub hdr_format: wgpu::TextureFormat,
    pub bright_tex: wgpu::Texture,
    pub bright_view: wgpu::TextureView,
    pub blur_a_tex: wgpu::Texture,
    pub blur_a: wgpu::TextureView,
    pub blur_b_tex: wgpu::Texture,
    pub blur_b: wgpu::TextureView,
    /// Quarter-res split gas pass (G2): emission HDR + dust extinction mask.
    /// Dust mask rgb = accumulated optical depth (CCM-tinted), alpha =
    /// nearest dust view-depth (Min blend, cleared to far).
    pub gas_em_tex: wgpu::Texture,
    pub gas_em_view: wgpu::TextureView,
    pub gas_dust_tex: wgpu::Texture,
    pub gas_dust_view: wgpu::TextureView,
    pub n: usize,
    pub bytes_per_frame: usize,
}

fn target(
    device: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

impl GpuState {
    pub fn new(
        device: &wgpu::Device,
        uniform_layout: &wgpu::BindGroupLayout,
        storage_layout: &wgpu::BindGroupLayout,
        n: usize,
        w: u32,
        h: u32,
    ) -> Self {
        let size = (n.max(1) * std::mem::size_of::<GpuItem>()) as u64;
        let prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prev"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let curr_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("curr"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform"),
            contents: bytemuck::bytes_of(&Uniform {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                cam_right: [1.0, 0.0, 0.0, 0.0],
                cam_up: [0.0, 1.0, 0.0, 0.0],
                params: [1.0, 0.0, 1.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform-bind"),
            layout: uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let storage_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("storage-bind"),
            layout: storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prev_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: curr_buf.as_entire_binding(),
                },
            ],
        });
        let hdr_format = wgpu::TextureFormat::Rgba16Float;
        let (hdr_tex, hdr_view) = target(device, "hdr", w, h, hdr_format);
        let bw = (w / 4).max(1);
        let bh = (h / 4).max(1);
        let (bright_tex, bright_view) = target(device, "bright", bw, bh, hdr_format);
        let (blur_a_tex, blur_a) = target(device, "blur-a", bw, bh, hdr_format);
        let (blur_b_tex, blur_b) = target(device, "blur-b", bw, bh, hdr_format);
        let (gas_em_tex, gas_em_view) = target(device, "gas-em", bw, bh, hdr_format);
        let (gas_dust_tex, gas_dust_view) = target(device, "gas-dust", bw, bh, hdr_format);
        Self {
            prev_buf,
            curr_buf,
            uniform_buf,
            uniform_bind,
            storage_bind,
            hdr_tex,
            hdr_view,
            hdr_format,
            bright_tex,
            bright_view,
            blur_a_tex,
            blur_a,
            blur_b_tex,
            blur_b,
            gas_em_tex,
            gas_em_view,
            gas_dust_tex,
            gas_dust_view,
            n,
            bytes_per_frame: 0,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let (hdr_tex, hdr_view) = target(device, "hdr", w, h, self.hdr_format);
        self.hdr_tex = hdr_tex;
        self.hdr_view = hdr_view;
        let bw = (w / 4).max(1);
        let bh = (h / 4).max(1);
        let (bright_tex, bright_view) = target(device, "bright", bw, bh, self.hdr_format);
        self.bright_tex = bright_tex;
        self.bright_view = bright_view;
        let (blur_a_tex, blur_a) = target(device, "blur-a", bw, bh, self.hdr_format);
        self.blur_a_tex = blur_a_tex;
        self.blur_a = blur_a;
        let (blur_b_tex, blur_b) = target(device, "blur-b", bw, bh, self.hdr_format);
        self.blur_b_tex = blur_b_tex;
        self.blur_b = blur_b;
        let (gas_em_tex, gas_em_view) = target(device, "gas-em", bw, bh, self.hdr_format);
        self.gas_em_tex = gas_em_tex;
        self.gas_em_view = gas_em_view;
        let (gas_dust_tex, gas_dust_view) = target(device, "gas-dust", bw, bh, self.hdr_format);
        self.gas_dust_tex = gas_dust_tex;
        self.gas_dust_view = gas_dust_view;
    }

    /// Per-frame upload: curr (CPU-interpolated) + uniform only. prev_buf
    /// stays allocated+bound for layout stability but is never written or
    /// read (billboard takes curr directly at alpha=1). Halves upload bytes.
    pub fn upload(&mut self, queue: &wgpu::Queue, curr: &[GpuItem], uniform: &Uniform) {
        queue.write_buffer(&self.curr_buf, 0, bytemuck::cast_slice(curr));
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniform));
        self.bytes_per_frame =
            curr.len() * std::mem::size_of::<GpuItem>() + std::mem::size_of::<Uniform>();
    }
}
