pub const SPACE_BG: [f32; 4] = [0.004, 0.004, 0.018, 1.0];

pub const BILLBOARD: &str = r#"
struct Uniform {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniform;
struct Item { pos: vec3<f32>, speed: f32, misc: vec4<f32> };
@group(1) @binding(0) var<storage, read> prev: array<Item>;
@group(1) @binding(1) var<storage, read> curr: array<Item>;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tm: vec3<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Out {
    let alpha = u.params.y;
    let p = mix(prev[ii].pos, curr[ii].pos, alpha);
    let tm = mix(prev[ii].misc.xyz, curr[ii].misc.xyz, alpha);
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5),
        vec2<f32>(-0.5, 0.5), vec2<f32>(0.5, 0.5));
    let c = corners[vi % 4u];
    let base = u.params.x * (0.35 + tm.y * tm.y * 2.4);
    let quad = mix(base, tm.z * length(p), step(0.0001, tm.z));
    let size = quad * 2.0;
    let world = p + (u.cam_right.xyz * c.x + u.cam_up.xyz * c.y) * size;
    var o: Out;
    o.clip = u.view_proj * vec4<f32>(world, 1.0);
    o.uv = c + vec2<f32>(0.5, 0.5);
    o.tm = tm;
    return o;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    let d = length(centered) * 2.0;
    let mag = clamp(in.tm.y, 0.0, 1.0);
    let ck = mix(6.0, 14.0, smoothstep(0.0, 1.0, mag));
    let core = exp(-d * d * ck);
    let halo = pow(max(0.0, 1.0 - d), 3.0) * 0.3 * smoothstep(0.6, 0.95, mag);
    let sx = exp(-abs(centered.x) * 30.0) * exp(-abs(centered.y) * 7.0);
    let sy = exp(-abs(centered.y) * 30.0) * exp(-abs(centered.x) * 7.0);
    let spike_gate = smoothstep(0.5, 0.9, mag);
    let spikes = (sx + sy) * spike_gate * 0.55;
    let total = core + halo + spikes;
    let edge = smoothstep(0.003, 0.03, total);
    let t = clamp(in.tm.x, 0.0, 1.0);
    var col = mix(vec3<f32>(0.45, 0.62, 1.0), vec3<f32>(0.9, 0.93, 1.0), smoothstep(0.05, 0.45, t));
    col = mix(col, vec3<f32>(1.0, 0.85, 0.65), smoothstep(0.55, 0.8, t));
    col = mix(col, vec3<f32>(1.0, 0.45, 0.25), smoothstep(0.85, 1.0, t));
    let core_col = mix(col, vec3<f32>(1.0), 0.55);
    let exposure = u.params.z;
    let bright = 0.35 + mag * 2.2;
    let hdr = (core * core_col * 2.2 + halo * col * 1.4 + spikes * mix(col, vec3<f32>(1.0), 0.5) * 2.4) * (exposure * bright);
    return vec4<f32>(hdr * edge, 1.0);
}
"#;

pub const TONEMAP: &str = r#"
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_smp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_smp: sampler;
@group(1) @binding(0) var<uniform> b: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(hdr_tex));
    let uv = fc.xy / dim;
    let c = textureSample(hdr_tex, hdr_smp, uv).rgb
        + textureSample(bloom_tex, bloom_smp, uv).rgb * b.x;
    let mapped = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(mapped, 1.0);
}
"#;

pub const BRIGHT: &str = r#"
struct BrightP { dst_inv: vec2<f32>, inv_exp: f32, thresh: f32 };
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_smp: sampler;
@group(1) @binding(0) var<uniform> p: BrightP;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = fc.xy * p.dst_inv;
    let fpx = 1.0 / vec2<f32>(textureDimensions(hdr_tex));
    let h = vec2<f32>(fpx.x * 0.5, 0.0);
    let v = vec2<f32>(0.0, fpx.y * 0.5);
    var c = textureSample(hdr_tex, hdr_smp, uv - h - v).rgb;
    c += textureSample(hdr_tex, hdr_smp, uv + h - v).rgb;
    c += textureSample(hdr_tex, hdr_smp, uv - h + v).rgb;
    c += textureSample(hdr_tex, hdr_smp, uv + h + v).rgb;
    c *= 0.25;
    let lum = dot(c * p.inv_exp, vec3<f32>(0.2126, 0.7152, 0.0722));
    let k = smoothstep(p.thresh, p.thresh + 0.6, lum);
    return vec4<f32>(c * k, 1.0);
}
"#;

pub const BLUR: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(1) @binding(0) var<uniform> u: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(src_tex));
    let uv = fc.xy / dim;
    let step = u.xy * u.zw;
    var c = textureSample(src_tex, src_smp, uv).rgb * 0.227;
    c += textureSample(src_tex, src_smp, uv + step * 1.384).rgb * 0.316;
    c += textureSample(src_tex, src_smp, uv - step * 1.384).rgb * 0.316;
    c += textureSample(src_tex, src_smp, uv + step * 3.230).rgb * 0.070;
    c += textureSample(src_tex, src_smp, uv - step * 3.230).rgb * 0.070;
    return vec4<f32>(c, 1.0);
}
"#;

pub fn billboard_pipeline(
    device: &wgpu::Device,
    hdr_format: wgpu::TextureFormat,
    uniform_layout: &wgpu::BindGroupLayout,
    storage_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("billboard"),
        source: wgpu::ShaderSource::Wgsl(BILLBOARD.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("billboard-layout"),
        bind_group_layouts: &[Some(uniform_layout), Some(storage_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("billboard"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: hdr_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub fn tonemap_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    bind_layout: &wgpu::BindGroupLayout,
    uni_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    post_pipeline(
        device,
        "tonemap",
        TONEMAP,
        surface_format,
        &[bind_layout, uni_layout],
        "fs",
    )
}

pub fn bright_pipeline(
    device: &wgpu::Device,
    hdr_format: wgpu::TextureFormat,
    bind_layout: &wgpu::BindGroupLayout,
    uni_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    post_pipeline(
        device,
        "bright",
        BRIGHT,
        hdr_format,
        &[bind_layout, uni_layout],
        "fs",
    )
}

pub fn blur_pipeline(
    device: &wgpu::Device,
    hdr_format: wgpu::TextureFormat,
    tex_layout: &wgpu::BindGroupLayout,
    uni_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    post_pipeline(
        device,
        "blur",
        BLUR,
        hdr_format,
        &[tex_layout, uni_layout],
        "fs",
    )
}

fn post_pipeline(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    target: wgpu::TextureFormat,
    bind_layouts: &[&wgpu::BindGroupLayout],
    entry: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let owned: Vec<Option<&wgpu::BindGroupLayout>> =
        bind_layouts.iter().map(|l| Some(*l)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &owned,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: target,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub fn tex_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub fn composite_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub fn blur_uni_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blur-uni-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

pub fn layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let storage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("storage-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    (uniform, storage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_parse() {
        for (name, src) in [
            ("billboard", BILLBOARD),
            ("tonemap", TONEMAP),
            ("bright", BRIGHT),
            ("blur", BLUR),
        ] {
            naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
