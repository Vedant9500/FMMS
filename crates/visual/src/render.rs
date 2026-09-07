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
// G2 dust mask (quarter-res): rgb = optical depth, a = nearest dust depth.
@group(2) @binding(0) var dust_tex: texture_2d<f32>;
@group(2) @binding(1) var dust_smp: sampler;
// G2 gas params: x = redden strength, y = full/mask resolution scale.
@group(3) @binding(0) var<uniform> g: vec4<f32>;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tm: vec3<f32>,
    @location(2) vdepth: f32,
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
    let base = u.params.x * (0.18 + tm.y * tm.y * 3.0);
    let quad = mix(base, tm.z * length(p), step(0.0001, tm.z));
    let size = quad * 2.0;
    let world = p + (u.cam_right.xyz * c.x + u.cam_up.xyz * c.y) * size;
    var o: Out;
    o.clip = u.view_proj * vec4<f32>(world, 1.0);
    o.uv = c + vec2<f32>(0.5, 0.5);
    o.tm = tm;
    o.vdepth = length(p);
    return o;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    let d = length(centered) * 2.0;
    let mag = clamp(in.tm.y, 0.0, 1.0);
    let ck = mix(2.0, 12.0, smoothstep(0.1, 0.9, mag));
    let core = exp(-d * d * ck);
    let halo = pow(max(0.0, 1.0 - d), 3.0) * 0.22 * smoothstep(0.65, 0.95, mag);
    let sx = exp(-abs(centered.x) * 30.0) * exp(-abs(centered.y) * 7.0);
    let sy = exp(-abs(centered.y) * 30.0) * exp(-abs(centered.x) * 7.0);
    let spike_gate = smoothstep(0.7, 0.95, mag);
    let spikes = (sx + sy) * spike_gate * 0.55;
    let total = core + halo + spikes;
    let edge = smoothstep(0.003, 0.03, total);
    let t = clamp(in.tm.x, 0.0, 1.0);
    var col = mix(vec3<f32>(0.62, 0.70, 1.0), vec3<f32>(0.70, 0.78, 1.0), smoothstep(0.0, 0.15, t));
    col = mix(col, vec3<f32>(0.82, 0.87, 1.0), smoothstep(0.15, 0.30, t));
    col = mix(col, vec3<f32>(0.97, 0.97, 1.0), smoothstep(0.30, 0.45, t));
    col = mix(col, vec3<f32>(1.0, 0.94, 0.84), smoothstep(0.45, 0.60, t));
    col = mix(col, vec3<f32>(1.0, 0.78, 0.55), smoothstep(0.60, 0.78, t));
    col = mix(col, vec3<f32>(1.0, 0.47, 0.16), smoothstep(0.78, 1.0, t));
    let core_col = mix(col, vec3<f32>(1.0), mix(0.22, 0.0, smoothstep(0.4, 0.9, t)));
    let exposure = u.params.z;
    let bright = 0.45 + 0.55 * mag;
    var hdr = (core * core_col * 1.15 + halo * col * 1.2 + spikes * mix(col, vec3<f32>(1.0), 0.5) * 2.0) * (exposure * bright);
    // G2 per-star reddening: dust mask stores CCM-tinted optical depth +
    // nearest dust depth; only stars behind the dust get extinguished.
    let mdim = vec2<f32>(textureDimensions(dust_tex));
    let m = textureSample(dust_tex, dust_smp, in.clip.xy / (mdim * max(g.y, 1.0)));
    let behind = smoothstep(m.a - 0.15, m.a + 0.05, in.vdepth);
    hdr *= mix(vec3<f32>(1.0), exp(-m.rgb), behind * clamp(g.x, 0.0, 1.0));
    return vec4<f32>(hdr * edge, 1.0);
}
"#;

pub const GAS_NOISE: &str = r#"
fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}
fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        v += a * vnoise(q);
        q *= 2.03;
        a *= 0.5;
    }
    return v;
}
// World-anchored gas coords: rebased center + eye = true world, plus the
// fragment's world-space offset from center so noise stays fixed in world
// space at any zoom (no per-quad tiling, no texel magnification).
// Anisotropic: compressed along arm tangent -> filaments stretch along arm.
fn gas_q(wc: vec2<f32>, loc: vec2<f32>, seed: f32, eye: vec2<f32>) -> vec2<f32> {
    let w = wc + eye;
    let p = (w + loc) * 9.0 + seed * 13.0;
    let rl = max(length(w), 1e-3);
    let rd = w / rl;
    let tg = vec2<f32>(-rd.y, rd.x) * 0.94 + rd * 0.34;
    let nd = vec2<f32>(-tg.y, tg.x);
    return tg * dot(p, tg) * 0.55 + nd * dot(p, nd) * 1.35;
}
// G2 13-plane impostor orientations: 3 axial + 6 edge-diagonal +
// 4 corner-diagonal. Any sightline pierces several planes, so gas keeps
// volume on fly-through and never pops at the near plane.
const NORMS: array<vec3f, 13> = array<vec3f, 13>(
    vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0),
    vec3f(0.7071, 0.7071, 0.0), vec3f(0.7071, -0.7071, 0.0),
    vec3f(0.7071, 0.0, 0.7071), vec3f(0.7071, 0.0, -0.7071),
    vec3f(0.0, 0.7071, 0.7071), vec3f(0.0, 0.7071, -0.7071),
    vec3f(0.5774, 0.5774, 0.5774), vec3f(0.5774, 0.5774, -0.5774),
    vec3f(0.5774, -0.5774, 0.5774), vec3f(-0.5774, 0.5774, 0.5774),
);
// Domain-warped density with multiplicative porosity: product of two
// offset octaves keeps mean ~constant while punching true holes.
fn gas_d(p: vec2<f32>, tm: f32) -> f32 {
    let s = vec2<f32>(tm * 0.02, tm * 0.013);
    let w = vec2<f32>(fbm(p + s), fbm(p + vec2<f32>(5.2, 1.3) - s));
    let d = fbm(p + 1.6 * w + s);
    let m = (0.35 + 0.65 * fbm(p * 2.3 + vec2<f32>(4.7, 2.9)))
        * (0.35 + 0.65 * fbm(p * 4.9 + vec2<f32>(1.7, 8.4)));
    return d * smoothstep(0.08, 0.45, m);
}
"#;

pub const GAS_EMISSION: &str = r#"
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
    @location(1) tm: vec4<f32>,
    @location(2) wpos: vec3<f32>,
    @location(3) lpos: vec3<f32>,
};

// G2: 13 intersecting quads per instance (52 verts). World-anchored noise
// makes each plane slice different density -> cheap volumetric read.
// misc.w carries the supernova/tidal flash: swell + white-blue + 4x glow.
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Out {
    let p = curr[ii].pos;
    let tm = curr[ii].misc.xyz;
    let fl = curr[ii].misc.w;
    let qid = vi / 4u;
    let ci = vi % 4u;
    let n = NORMS[qid % 13u];
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let t = normalize(cross(n, up));
    let b = cross(n, t);
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5),
        vec2<f32>(-0.5, 0.5), vec2<f32>(0.5, 0.5));
    let c = corners[ci];
    let off = (t * c.x + b * c.y) * tm.y * (1.0 + fl * 0.3);
    var o: Out;
    o.clip = u.view_proj * vec4<f32>(p + off, 1.0);
    o.uv = c + vec2<f32>(0.5, 0.5);
    o.tm = vec4<f32>(tm, fl);
    o.wpos = p;
    o.lpos = off;
    return o;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    let r = length(centered) * 2.0;
    let eye = vec2<f32>(u.cam_right.w, u.cam_up.w);
    let q = gas_q(in.wpos.xz, in.lpos.xz, in.tm.x, eye) + in.lpos.y * 2.7;
    let d = gas_d(q, u.params.w);
    let rim = exp(-pow((r - 0.55) * 2.4, 2.0));
    let core = exp(-r * r * 4.0) * 0.55;
    let body = smoothstep(1.0, 0.25, r) * smoothstep(0.30, 0.70, d) * (rim * 0.95 + core);
    var col = mix(vec3<f32>(1.0, 0.36, 0.42), vec3<f32>(1.0, 0.72, 0.75), smoothstep(0.7, 0.0, r) * smoothstep(0.4, 0.75, d));
    if (in.tm.z > 0.5 && in.tm.z < 1.5) {
        col = mix(vec3<f32>(0.35, 0.85, 0.78), vec3<f32>(1.0, 0.36, 0.42), smoothstep(0.05, 0.5, r));
    } else if (in.tm.z > 1.5) {
        col = vec3<f32>(0.45, 0.60, 1.0);
    }
    let fl = clamp(in.tm.w, 0.0, 1.0);
    col = mix(col, vec3<f32>(0.75, 0.85, 1.0), fl * 0.7);
    // EM_NORM: ~13 planes pierce a central sightline; normalize to one.
    // Flash glow capped at 2.5x so overlapping giants can't white-clip.
    return vec4<f32>(col * body * 0.15 * 0.85 * u.params.z * (1.0 + fl * 1.5), 1.0);
}
"#;

pub const GAS_DUST: &str = r#"
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
    @location(1) tm: vec4<f32>,
    @location(2) wpos: vec3<f32>,
    @location(3) lpos: vec3<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Out {
    let p = curr[ii].pos;
    let tm = curr[ii].misc.xyz;
    let fl = curr[ii].misc.w;
    let qid = vi / 4u;
    let ci = vi % 4u;
    let n = NORMS[qid % 13u];
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let t = normalize(cross(n, up));
    let b = cross(n, t);
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5),
        vec2<f32>(-0.5, 0.5), vec2<f32>(0.5, 0.5));
    let c = corners[ci];
    let off = (t * c.x + b * c.y) * tm.y;
    var o: Out;
    o.clip = u.view_proj * vec4<f32>(p + off, 1.0);
    o.uv = c + vec2<f32>(0.5, 0.5);
    o.tm = vec4<f32>(tm, fl);
    o.wpos = p;
    o.lpos = off;
    return o;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    let r = length(centered) * 2.0;
    let eye = vec2<f32>(u.cam_right.w, u.cam_up.w);
    let q = gas_q(in.wpos.xz, in.lpos.xz, in.tm.x, eye) + in.lpos.y * 2.7;
    let d = gas_d(q, u.params.w);
    let shape = smoothstep(0.42, 0.8, d) * smoothstep(1.0, 0.35, r);
    // CCM-tinted optical depth (blue dies first) + nearest depth for the
    // per-star behind-test. DUST_NORM same 13-plane accounting as emission.
    let tau = shape * vec3<f32>(0.55, 0.75, 1.0) * 0.9 * 0.12;
    let dep = mix(10000.0, length(in.wpos), step(0.003, shape));
    return vec4<f32>(tau, dep);
}
"#;

pub const TONEMAP: &str = r#"
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_smp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_smp: sampler;
// G2 split-res gas: quarter-res emission HDR + dust optical-depth mask.
@group(0) @binding(4) var em_tex: texture_2d<f32>;
@group(0) @binding(5) var em_smp: sampler;
@group(0) @binding(6) var dust_tex: texture_2d<f32>;
@group(0) @binding(7) var dust_smp: sampler;
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
    let transmit = exp(-textureSample(dust_tex, dust_smp, uv).rgb);
    let c = (textureSample(hdr_tex, hdr_smp, uv).rgb
        + textureSample(em_tex, em_smp, uv).rgb) * transmit
        + textureSample(bloom_tex, bloom_smp, uv).rgb * b.x;
    let mapped = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(mapped, 1.0);
}
"#;

pub const BRIGHT: &str = r#"
struct BrightP { dst_inv: vec2<f32>, inv_exp: f32, thresh: f32 };
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_smp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_smp: sampler;
@group(0) @binding(4) var em_tex: texture_2d<f32>;
@group(0) @binding(5) var em_smp: sampler;
@group(0) @binding(6) var dust_tex: texture_2d<f32>;
@group(0) @binding(7) var dust_smp: sampler;
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
    // G2: bloom sees the same extinguished composite the eye sees.
    let transmit = exp(-textureSample(dust_tex, dust_smp, uv).rgb);
    c = (c + textureSample(em_tex, em_smp, uv).rgb) * transmit;
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
    tex_layout: &wgpu::BindGroupLayout,
    uni_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("billboard"),
        source: wgpu::ShaderSource::Wgsl(BILLBOARD.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("billboard-layout"),
        bind_group_layouts: &[
            Some(uniform_layout),
            Some(storage_layout),
            Some(tex_layout),
            Some(uni_layout),
        ],
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

fn gas_pipeline(
    device: &wgpu::Device,
    label: &str,
    src: String,
    hdr_format: wgpu::TextureFormat,
    uniform_layout: &wgpu::BindGroupLayout,
    storage_layout: &wgpu::BindGroupLayout,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(uniform_layout), Some(storage_layout)],
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
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: hdr_format,
                blend: Some(blend),
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

pub fn emission_pipeline(
    device: &wgpu::Device,
    hdr_format: wgpu::TextureFormat,
    uniform_layout: &wgpu::BindGroupLayout,
    storage_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    gas_pipeline(
        device,
        "emission",
        format!("{GAS_NOISE}{GAS_EMISSION}"),
        hdr_format,
        uniform_layout,
        storage_layout,
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        },
    )
}

pub fn dust_pipeline(
    device: &wgpu::Device,
    hdr_format: wgpu::TextureFormat,
    uniform_layout: &wgpu::BindGroupLayout,
    storage_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    gas_pipeline(
        device,
        "dust",
        format!("{GAS_NOISE}{GAS_DUST}"),
        hdr_format,
        uniform_layout,
        storage_layout,
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            // Nearest dust depth wins regardless of draw order.
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Min,
            },
        },
    )
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
    let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let smp = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite-layout"),
        // hdr, hdr-smp, bloom, bloom-smp, gas-em, gas-em-smp, dust, dust-smp
        entries: &[
            tex(0),
            smp(1),
            tex(2),
            smp(3),
            tex(4),
            smp(5),
            tex(6),
            smp(7),
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
        for (name, src) in [
            ("emission", format!("{GAS_NOISE}{GAS_EMISSION}")),
            ("dust", format!("{GAS_NOISE}{GAS_DUST}")),
        ] {
            naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
