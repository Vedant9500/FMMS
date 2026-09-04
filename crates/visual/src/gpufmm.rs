pub const P: usize = 4;
pub const DIM: usize = (P + 1) * (P + 1);

pub const WGSL_HELPERS: &str = r#"
fn idx(n: u32, m: i32) -> u32 {
    return u32(i32(n * (n + 1u)) + m);
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn upsilon_table(x: f32, y: f32, z: f32) -> array<vec2<f32>, DIM> {
    let r2 = x * x + y * y + z * z;
    var u: array<vec2<f32>, DIM>;
    u[idx(0u, 0)] = vec2<f32>(1.0, 0.0);
    for (var m = 0u; m <= P; m++) {
        let mi = i32(m);
        if (m > 0u) {
            u[idx(m, mi)] = cmul(vec2<f32>(x, y) * (1.0 / f32(2u * m)), u[idx(m - 1u, mi - 1)]);
        }
        for (var n = m + 1u; n <= P; n++) {
            let a = f32(2u * n - 1u) * z;
            let c = f32(n * n - m * m);
            var p2 = vec2<f32>(0.0, 0.0);
            if (n >= m + 2u) {
                p2 = u[idx(n - 2u, mi)];
            }
            u[idx(n, mi)] = (u[idx(n - 1u, mi)] * a - p2 * r2) / c;
        }
    }
    for (var m = 1u; m <= P; m++) {
        let mi = i32(m);
        var sgn = -1.0;
        if (m % 2u == 0u) {
            sgn = 1.0;
        }
        for (var n = m; n <= P; n++) {
            let cc = u[idx(n, mi)];
            u[idx(n, -mi)] = vec2<f32>(cc.x, -cc.y) * sgn;
        }
    }
    return u;
}
"#;

pub const M2L_WGSL: &str = r#"
const P: u32 = 4u;
const DIM: u32 = 25u;

@group(0) @binding(0) var<storage, read> moments: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> srcs: array<u32>;
@group(0) @binding(2) var<storage, read> rs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

fn idx(n: u32, m: i32) -> u32 {
    return u32(i32(n * (n + 1u)) + m);
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pi = gid.x;
    if (pi >= u.z) {
        return;
    }
    let moff = srcs[pi] * u.y;
    let ooff = pi * u.y;
    let x = rs[pi].x;
    let y = rs[pi].y;
    let z = rs[pi].z;
    let r2 = x * x + y * y + z * z;
    let ir = inverseSqrt(r2);

    var th: array<vec2<f32>, DIM>;
    th[idx(0u, 0)] = vec2<f32>(ir, 0.0);
    for (var n = 1u; n <= P; n++) {
        let a = f32(2u * n - 1u) * z;
        let b = f32((n - 1u) * (n - 1u));
        let p1 = th[idx(n - 1u, 0)];
        var p2 = vec2<f32>(0.0, 0.0);
        if (n >= 2u) {
            p2 = th[idx(n - 2u, 0)];
        }
        th[idx(n, 0)] = (p1 * a - p2 * b) / r2;
    }
    for (var m = 1u; m <= P; m++) {
        let mi = i32(m);
        th[idx(m, mi)] = cmul(vec2<f32>(x, y) * (f32(2u * m - 1u) / r2), th[idx(m - 1u, mi - 1)]);
        for (var n = m + 1u; n <= P; n++) {
            let a = f32(2u * n - 1u) * z;
            let b = f32((n - 1u) * (n - 1u) - m * m);
            th[idx(n, mi)] = (th[idx(n - 1u, mi)] * a - th[idx(n - 2u, mi)] * b) / r2;
        }
    }
    for (var m = 1u; m <= P; m++) {
        let mi = i32(m);
        var sgn = -1.0;
        if (m % 2u == 0u) {
            sgn = 1.0;
        }
        for (var n = m; n <= P; n++) {
            let c = th[idx(n, mi)];
            th[idx(n, -mi)] = vec2<f32>(c.x, -c.y) * sgn;
        }
    }

    for (var n = 0u; n <= P; n++) {
        let ni = i32(n);
        for (var m = -ni; m <= ni; m++) {
            var fb = vec2<f32>(0.0, 0.0);
            for (var k = 0u; k <= P - n; k++) {
                let ki = i32(k);
                for (var l = -ki; l <= ki; l++) {
                    let t = th[idx(n + k, m + l)];
                    let mc = moments[moff + idx(k, l)];
                    fb += cmul(vec2<f32>(mc.x, -mc.y), t);
                }
            }
            out[ooff + idx(n, m)] = fb;
        }
    }
}
"#;

pub const M2L_ACC_WGSL: &str = r#"
const P: u32 = 4u;
const DIM: u32 = 25u;

@group(0) @binding(0) var<storage, read> moments: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gsrc: array<u32>;
@group(0) @binding(3) var<storage, read> gstart: array<u32>;
@group(0) @binding(4) var<storage, read> gcount: array<u32>;
@group(0) @binding(5) var<storage, read_write> locals: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> u: vec4<u32>;

fn idx(n: u32, m: i32) -> u32 {
    return u32(i32(n * (n + 1u)) + m);
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= u.x) {
        return;
    }
    let loff = c * u.y;
    let cnt = gcount[c];
    if (cnt == 0u) {
        for (var i = 0u; i < DIM; i++) {
            locals[loff + i] = vec2<f32>(0.0, 0.0);
        }
        return;
    }
    let dst = centers[c].xyz;
    var acc: array<vec2<f32>, DIM>;
    for (var i = 0u; i < DIM; i++) {
        acc[i] = vec2<f32>(0.0, 0.0);
    }
    var th: array<vec2<f32>, DIM>;
    let start = gstart[c];
    for (var j = 0u; j < cnt; j++) {
        let src = gsrc[start + j];
        let r = dst - centers[src].xyz;
        let x = r.x;
        let y = r.y;
        let z = r.z;
        let r2 = x * x + y * y + z * z;
        let ir = inverseSqrt(r2);
        th[idx(0u, 0)] = vec2<f32>(ir, 0.0);
        for (var n = 1u; n <= P; n++) {
            let a = f32(2u * n - 1u) * z;
            let b = f32((n - 1u) * (n - 1u));
            let p1 = th[idx(n - 1u, 0)];
            var p2 = vec2<f32>(0.0, 0.0);
            if (n >= 2u) {
                p2 = th[idx(n - 2u, 0)];
            }
            th[idx(n, 0)] = (p1 * a - p2 * b) / r2;
        }
        for (var m = 1u; m <= P; m++) {
            let mi = i32(m);
            th[idx(m, mi)] = cmul(vec2<f32>(x, y) * (f32(2u * m - 1u) / r2), th[idx(m - 1u, mi - 1)]);
            for (var n = m + 1u; n <= P; n++) {
                let a = f32(2u * n - 1u) * z;
                let b = f32((n - 1u) * (n - 1u) - m * m);
                th[idx(n, mi)] = (th[idx(n - 1u, mi)] * a - th[idx(n - 2u, mi)] * b) / r2;
            }
        }
        for (var m = 1u; m <= P; m++) {
            let mi = i32(m);
            var sgn = -1.0;
            if (m % 2u == 0u) {
                sgn = 1.0;
            }
            for (var n = m; n <= P; n++) {
                let cc = th[idx(n, mi)];
                th[idx(n, -mi)] = vec2<f32>(cc.x, -cc.y) * sgn;
            }
        }
        let moff = src * u.y;
        for (var n = 0u; n <= P; n++) {
            let ni = i32(n);
            for (var m = -ni; m <= ni; m++) {
                var fb = vec2<f32>(0.0, 0.0);
                for (var k = 0u; k <= P - n; k++) {
                    let ki = i32(k);
                    for (var l = -ki; l <= ki; l++) {
                        let t = th[idx(n + k, m + l)];
                        let mc = moments[moff + idx(k, l)];
                        fb += cmul(vec2<f32>(mc.x, -mc.y), t);
                    }
                }
                acc[idx(n, m)] += fb;
            }
        }
    }
    for (var i = 0u; i < DIM; i++) {
        locals[loff + i] = acc[i];
    }
}
"#;

pub fn p2m_source(p: u32) -> String {
    let d = (p + 1) * (p + 1);
    format!(
        r#"const P: u32 = {p}u;
const DIM: u32 = {d}u;
{pre}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> leaf_ids: array<u32>;
@group(0) @binding(2) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cfirst: array<u32>;
@group(0) @binding(4) var<storage, read> ccount: array<u32>;
@group(0) @binding(5) var<storage, read_write> moments: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let li = gid.x;
    if (li >= u.y) {{
        return;
    }}
    let c = leaf_ids[li];
    let off = c * u.x;
    let z = centers[c].xyz;
    var acc: array<vec2<f32>, DIM>;
    for (var i = 0u; i < DIM; i++) {{
        acc[i] = vec2<f32>(0.0, 0.0);
    }}
    for (var k = cfirst[c]; k < cfirst[c] + ccount[c]; k++) {{
        let r = pos[k].xyz - z;
        let ut = upsilon_table(r.x, r.y, r.z);
        let m = pos[k].w;
        for (var i = 0u; i < DIM; i++) {{
            acc[i] += ut[i] * m;
        }}
    }}
    for (var i = 0u; i < DIM; i++) {{
        moments[off + i] = acc[i];
    }}
}}
"#,
        pre = WGSL_HELPERS,
        p = p,
        d = d,
    )
}

pub fn m2m_source(p: u32) -> String {
    let d = (p + 1) * (p + 1);
    format!(
        r#"const P: u32 = {p}u;
const DIM: u32 = {d}u;
{pre}
@group(0) @binding(0) var<storage, read_write> moments: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> children: array<u32>;
@group(0) @binding(3) var<storage, read> level_cells: array<u32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= u.y) {{
        return;
    }}
    let c = level_cells[u.z + i];
    let off = c * u.x;
    let z = centers[c].xyz;
    var has_child = false;
    for (var ci = 0u; ci < 8u; ci++) {{
        if (bitcast<i32>(children[c * 8u + ci]) >= 0) {{
            has_child = true;
        }}
    }}
    if (!has_child) {{
        return;
    }}
    var acc: array<vec2<f32>, DIM>;
    for (var q = 0u; q < DIM; q++) {{
        acc[q] = vec2<f32>(0.0, 0.0);
    }}
    for (var ci = 0u; ci < 8u; ci++) {{
        let ch = bitcast<i32>(children[c * 8u + ci]);
        if (ch < 0) {{
            continue;
        }}
        let chu = u32(ch);
        let d = centers[chu].xyz - z;
        let ut = upsilon_table(d.x, d.y, d.z);
        let coff = chu * u.x;
        for (var n = 0u; n <= P; n++) {{
            let ni = i32(n);
            for (var m = -ni; m <= ni; m++) {{
                var s = vec2<f32>(0.0, 0.0);
                for (var k = 0u; k <= n; k++) {{
                    let ki = i32(k);
                    let nk = n - k;
                    let lo = max(-ki, m - i32(nk));
                    let hi = min(ki, m + i32(nk));
                    for (var l = lo; l <= hi; l++) {{
                        s += cmul(ut[idx(k, l)], moments[coff + idx(nk, m - l)]);
                    }}
                }}
                acc[idx(n, m)] += s;
            }}
        }}
    }}
    for (var q = 0u; q < DIM; q++) {{
        moments[off + q] = acc[q];
    }}
}}
"#,
        pre = WGSL_HELPERS,
        p = p,
        d = d,
    )
}

pub fn l2l_source(p: u32) -> String {
    let d = (p + 1) * (p + 1);
    format!(
        r#"const P: u32 = {p}u;
const DIM: u32 = {d}u;
{pre}
@group(0) @binding(0) var<storage, read_write> locals: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> children: array<u32>;
@group(0) @binding(3) var<storage, read> level_cells: array<u32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= u.y) {{
        return;
    }}
    let c = level_cells[u.z + i];
    let off = c * u.x;
    var has_child = false;
    for (var ci = 0u; ci < 8u; ci++) {{
        if (bitcast<i32>(children[c * 8u + ci]) >= 0) {{
            has_child = true;
        }}
    }}
    if (!has_child) {{
        return;
    }}
    var f: array<vec2<f32>, DIM>;
    for (var q = 0u; q < DIM; q++) {{
        f[q] = locals[off + q];
    }}
    let z = centers[c].xyz;
    for (var ci = 0u; ci < 8u; ci++) {{
        let ch = bitcast<i32>(children[c * 8u + ci]);
        if (ch < 0) {{
            continue;
        }}
        let chu = u32(ch);
        let r = z - centers[chu].xyz;
        let ut = upsilon_table(r.x, r.y, r.z);
        let coff = chu * u.x;
        for (var n = 0u; n <= P; n++) {{
            let ni = i32(n);
            for (var m = -ni; m <= ni; m++) {{
                var s = vec2<f32>(0.0, 0.0);
                for (var k = 0u; k <= P - n; k++) {{
                    let ki = i32(k);
                    for (var l = -ki; l <= ki; l++) {{
                        let uc = ut[idx(k, l)];
                        s += cmul(vec2<f32>(uc.x, -uc.y), f[idx(n + k, m + l)]);
                    }}
                }}
                locals[coff + idx(n, m)] += s;
            }}
        }}
    }}
}}
"#,
        pre = WGSL_HELPERS,
        p = p,
        d = d,
    )
}

pub fn l2p_source(p: u32) -> String {
    let d = (p + 1) * (p + 1);
    format!(
        r#"const P: u32 = {p}u;
const DIM: u32 = {d}u;
{pre}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cell_of_slot: array<u32>;
@group(0) @binding(2) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> locals: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> accel: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let s = gid.x;
    if (s >= u.y) {{
        return;
    }}
    let c = cell_of_slot[s];
    let off = c * u.x;
    let r = centers[c].xyz - pos[s].xyz;
    let ut = upsilon_table(r.x, r.y, r.z);
    var psi00 = vec2<f32>(0.0, 0.0);
    for (var k = 0u; k <= P; k++) {{
        let ki = i32(k);
        for (var l = -ki; l <= ki; l++) {{
            let uc = ut[idx(k, l)];
            psi00 += cmul(vec2<f32>(uc.x, -uc.y), locals[off + idx(k, l)]);
        }}
    }}
    var psi11 = vec2<f32>(0.0, 0.0);
    var psi10 = vec2<f32>(0.0, 0.0);
    for (var k = 0u; k <= P - 1u; k++) {{
        let ki = i32(k);
        for (var l = -ki; l <= ki; l++) {{
            let uc = ut[idx(k, l)];
            let ucc = vec2<f32>(uc.x, -uc.y);
            psi11 += cmul(ucc, locals[off + idx(1u + k, 1 + l)]);
            psi10 += cmul(ucc, locals[off + idx(1u + k, l)]);
        }}
    }}
    accel[s] = vec4<f32>(-psi11.x, -psi11.y, -psi10.x, psi00.x);
}}
"#,
        pre = WGSL_HELPERS,
        p = p,
        d = d,
    )
}

pub fn p2p_source(p: u32) -> String {
    let d = (p + 1) * (p + 1);
    format!(
        r#"const P: u32 = {p}u;
const DIM: u32 = {d}u;
{pre}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> nstart: array<u32>;
@group(0) @binding(2) var<storage, read> ncount: array<u32>;
@group(0) @binding(3) var<storage, read> nlist: array<u32>;
@group(0) @binding(4) var<storage, read> cfirst: array<u32>;
@group(0) @binding(5) var<storage, read> ccount: array<u32>;
@group(0) @binding(6) var<storage, read_write> accel: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> u: vec4<u32>;

fn interact(px: vec3<f32>, qx: vec3<f32>, qm: f32) -> vec4<f32> {{
    let d = qx - px;
    let r2 = dot(d, d);
    var inv = inverseSqrt(r2);
    inv = inv * (1.5 - 0.5 * r2 * inv * inv);
    inv = inv * (1.5 - 0.5 * r2 * inv * inv);
    let inv3 = inv * inv * inv;
    return vec4<f32>(qm * inv3 * d, qm * inv);
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let s = gid.x;
    if (s >= u.x) {{
        return;
    }}
    let p = pos[s].xyz;
    var a = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let start = nstart[s];
    let cnt = ncount[s];
    for (var t = 0u; t < cnt; t++) {{
        let leaf = nlist[start + t];
        for (var k = cfirst[leaf]; k < cfirst[leaf] + ccount[leaf]; k++) {{
            if (k == s) {{
                continue;
            }}
            a += interact(p, pos[k].xyz, pos[k].w);
        }}
    }}
    accel[s] += a;
}}
"#,
        pre = WGSL_HELPERS,
        p = p,
        d = d,
    )
}

pub fn sweep_layout(
    device: &wgpu::Device,
    label: &str,
    n_storage: u32,
    rw_binding: u32,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::new();
    for b in 0..n_storage {
        let ro = b != rw_binding;
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if ro {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: n_storage,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub fn sweep_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipe_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub fn m2l_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_only {
                wgpu::BufferBindingType::Storage { read_only: true }
            } else {
                wgpu::BufferBindingType::Storage { read_only: false }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("m2l-layout"),
        entries: &[
            storage(0, true),
            storage(1, true),
            storage(2, true),
            storage(3, false),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub const M2L_RED_WGSL: &str = r#"
const DIM: u32 = 25u;

@group(0) @binding(0) var<storage, read> slots: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gstart: array<u32>;
@group(0) @binding(2) var<storage, read> gcount: array<u32>;
@group(0) @binding(3) var<storage, read_write> locals: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= u.x) {
        return;
    }
    let loff = c * u.y;
    let start = gstart[c];
    let cnt = gcount[c];
    var acc: array<vec2<f32>, DIM>;
    for (var i = 0u; i < DIM; i++) {
        acc[i] = vec2<f32>(0.0, 0.0);
    }
    for (var j = 0u; j < cnt; j++) {
        let soff = (start + j) * u.y;
        for (var i = 0u; i < DIM; i++) {
            acc[i] += slots[soff + i];
        }
    }
    for (var i = 0u; i < DIM; i++) {
        locals[loff + i] = acc[i];
    }
}
"#;

pub fn m2l_acc_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_only {
                wgpu::BufferBindingType::Storage { read_only: true }
            } else {
                wgpu::BufferBindingType::Storage { read_only: false }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("m2l-acc-layout"),
        entries: &[
            storage(0, true),
            storage(1, true),
            storage(2, true),
            storage(3, true),
            storage(4, true),
            storage(5, false),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn m2l_acc_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("m2l-acc"),
        source: wgpu::ShaderSource::Wgsl(M2L_ACC_WGSL.into()),
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("m2l-acc-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("m2l-acc"),
        layout: Some(&pipe_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub fn m2l_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("m2l"),
        source: wgpu::ShaderSource::Wgsl(M2L_WGSL.into()),
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("m2l-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("m2l"),
        layout: Some(&pipe_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub fn m2l_red_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_only {
                wgpu::BufferBindingType::Storage { read_only: true }
            } else {
                wgpu::BufferBindingType::Storage { read_only: false }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("m2l-red-layout"),
        entries: &[
            storage(0, true),
            storage(1, true),
            storage(2, true),
            storage(3, false),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn m2l_red_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("m2l-red"),
        source: wgpu::ShaderSource::Wgsl(M2L_RED_WGSL.into()),
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("m2l-red-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("m2l-red"),
        layout: Some(&pipe_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m2l_shader_parses() {
        naga::front::wgsl::parse_str(M2L_WGSL).expect("m2l wgsl");
        assert_eq!(DIM, 25);
    }

    #[test]
    fn m2l_acc_shader_parses() {
        naga::front::wgsl::parse_str(M2L_ACC_WGSL).expect("m2l-acc wgsl");
    }

    #[test]
    fn m2l_red_shader_parses() {
        naga::front::wgsl::parse_str(M2L_RED_WGSL).expect("m2l-red wgsl");
    }

    #[test]
    fn sweep_shaders_parse() {
        naga::front::wgsl::parse_str(&p2m_source(4)).expect("p2m wgsl");
        naga::front::wgsl::parse_str(&m2m_source(4)).expect("m2m wgsl");
        naga::front::wgsl::parse_str(&l2l_source(4)).expect("l2l wgsl");
        naga::front::wgsl::parse_str(&l2p_source(4)).expect("l2p wgsl");
        naga::front::wgsl::parse_str(&p2p_source(4)).expect("p2p wgsl");
    }
}
