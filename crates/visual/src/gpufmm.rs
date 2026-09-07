//! GPU-resident FMM backend (visual preset p=4 / eps=1e-2 / n_leaf=64).
//!
//! Algorithm: plain spherical-harmonic FMM in f32, evaluated directly
//! (O(p^4) translations, DIM=25). Unlike the CPU crate there is deliberately
//! NO Wigner rotation: at p=4 the rotation bookkeeping (swap matrices,
//! two extra passes) costs more than the direct contraction, and theta
//! tables stay register-resident per thread.
//!
//! Refresh model: the tree + interaction lists rebuild on CPU every `k`
//! steps (`try_accel_with_refresh`); positions upload every step and
//! moments/locals recompute on GPU. Non-refresh steps cost one SoA shuffle
//! plus small uploads on CPU.
//!
//! Per-pass uniform `vec4<u32>` meanings (positional fields!):
//!   p2m: [DIM, nleaf, _, _]      one thread per leaf
//!   m2m: [DIM, nlevel, level_off, _]  one thread per level cell
//!   s1:  [_, DIM, npairs, _]     one thread per directed M2L pair
//!   red: [ncells, DIM, _, _]     one thread per (cell, moment)
//!   l2l: [DIM, nlevel, level_off, _]
//!   l2p: [DIM, nslots, _, _]     one thread per body
//!   p2p: [nleaf, _, _, _]        one workgroup per leaf ordinal
//!   fat: [njobs, _, _, _]        one workgroup per fat job
//!   fred:[nfat, _, _, _]         one workgroup per fat leaf
//!
//! Buffer size scalings (d=25): b-slots = npairs*d*8B dominates
//! (~4.3 pairs/body measured: ~280MB@300k, ~0.9GB@1M). `ensure()` checks
//! every buffer against the device buffer+binding limits and returns
//! OverBudget instead of hitting wgpu validation.
//!
//! Precision: f32 end to end (f64 in, f32 on device, f64 out). P2P uses a
//! 1e-12 softening + 2x Newton-refined rsqrt; coincident bodies are finite
//! (no NaN) at negligible force error.

pub const P: usize = 4;
pub const DIM: usize = (P + 1) * (P + 1);
pub const EPS: f64 = 1e-2;
pub const N_LEAF: usize = 64;

use fmm::tree::Octree;

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

/// Directed-pair M2L: one thread per pair. Templated on P like the sweep
/// shaders (was a hardcoded-DIM const duplicating idx/cmul).
pub fn m2l_source() -> String {
    let p = P as u32;
    let d = DIM as u32;
    format!(
        r#"const P: u32 = {p}u;
const DIM: u32 = {d}u;
{pre}
@group(0) @binding(0) var<storage, read> moments: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> srcs: array<u32>;
@group(0) @binding(2) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> dsts: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pi = gid.x;
    if (pi >= u.z) {{
        return;
    }}
    // Separation from cell centers on-device: saves a per-pair CPU loop +
    // a vec4 upload per refresh (u32 dst list instead).
    let src = srcs[pi];
    let r = centers[dsts[pi]].xyz - centers[src].xyz;
    let moff = src * u.y;
    let ooff = pi * u.y;
    let x = r.x;
    let y = r.y;
    let z = r.z;
    let r2 = x * x + y * y + z * z;
    let ir = inverseSqrt(r2);

    var th: array<vec2<f32>, DIM>;
    th[idx(0u, 0)] = vec2<f32>(ir, 0.0);
    for (var n = 1u; n <= P; n++) {{
        let a = f32(2u * n - 1u) * z;
        let b = f32((n - 1u) * (n - 1u));
        let p1 = th[idx(n - 1u, 0)];
        var p2 = vec2<f32>(0.0, 0.0);
        if (n >= 2u) {{
            p2 = th[idx(n - 2u, 0)];
        }}
        th[idx(n, 0)] = (p1 * a - p2 * b) / r2;
    }}
    for (var m = 1u; m <= P; m++) {{
        let mi = i32(m);
        th[idx(m, mi)] = cmul(vec2<f32>(x, y) * (f32(2u * m - 1u) / r2), th[idx(m - 1u, mi - 1)]);
        for (var n = m + 1u; n <= P; n++) {{
            let a = f32(2u * n - 1u) * z;
            let b = f32((n - 1u) * (n - 1u) - m * m);
            th[idx(n, mi)] = (th[idx(n - 1u, mi)] * a - th[idx(n - 2u, mi)] * b) / r2;
        }}
    }}
    for (var m = 1u; m <= P; m++) {{
        let mi = i32(m);
        var sgn = -1.0;
        if (m % 2u == 0u) {{
            sgn = 1.0;
        }}
        for (var n = m; n <= P; n++) {{
            let c = th[idx(n, mi)];
            th[idx(n, -mi)] = vec2<f32>(c.x, -c.y) * sgn;
        }}
    }}

    for (var n = 0u; n <= P; n++) {{
        let ni = i32(n);
        for (var m = -ni; m <= ni; m++) {{
            var fb = vec2<f32>(0.0, 0.0);
            for (var k = 0u; k <= P - n; k++) {{
                let ki = i32(k);
                for (var l = -ki; l <= ki; l++) {{
                    let t = th[idx(n + k, m + l)];
                    let mc = moments[moff + idx(k, l)];
                    fb += cmul(vec2<f32>(mc.x, -mc.y), t);
                }}
            }}
            out[ooff + idx(n, m)] = fb;
        }}
    }}
}}
"#,
        pre = WGSL_HELPERS,
        p = p,
        d = d,
    )
}

pub fn m2l_red_source() -> String {
    let d = DIM as u32;
    format!(
        r#"const DIM: u32 = {d}u;

@group(0) @binding(0) var<storage, read> slots: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gstart: array<u32>;
@group(0) @binding(2) var<storage, read> gcount: array<u32>;
@group(0) @binding(3) var<storage, read_write> locals: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = gid.x;
    let c = t / u.y;
    if (c >= u.x) {{
        return;
    }}
    let i = t % u.y;
    let start = gstart[c];
    let cnt = gcount[c];
    var acc = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < cnt; j++) {{
        acc += slots[(start + j) * u.y + i];
    }}
    locals[c * u.y + i] = acc;
}}
"#,
        d = d,
    )
}

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

pub const P2P_WG: u32 = 64;
/// Partner-body count above which a leaf is split across workgroups.
pub const FAT_THR: u32 = 4096;
/// Partner bodies per fat job (one workgroup each).
pub const FAT_CHUNK: u32 = 2048;

pub fn p2p_source(_p: u32) -> String {
    format!(
        r#"const TILE: u32 = 512u;
const WG: u32 = {wg}u;
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> leaf_ids: array<u32>;
@group(0) @binding(2) var<storage, read> cfirst: array<u32>;
@group(0) @binding(3) var<storage, read> ccount: array<u32>;
@group(0) @binding(4) var<storage, read> pb_start: array<u32>;
@group(0) @binding(5) var<storage, read> pb_count: array<u32>;
@group(0) @binding(6) var<storage, read> pb_list: array<u32>;
@group(0) @binding(7) var<storage, read_write> accel: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> u: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, TILE>;
var<workgroup> tids: array<u32, TILE>;

fn interact(px: vec3<f32>, qx: vec3<f32>, qm: f32) -> vec4<f32> {{
    let d = qx - px;
    // 1e-12 softening: coincident bodies stay finite (no inf/NaN poisoning
    // the step). Force error ~1e-12 relative; self-skip is by slot id.
    let r2 = dot(d, d) + 1e-12;
    var inv = inverseSqrt(r2);
    inv = inv * (1.5 - 0.5 * r2 * inv * inv);
    inv = inv * (1.5 - 0.5 * r2 * inv * inv);
    let inv3 = inv * inv * inv;
    return vec4<f32>(qm * inv3 * d, qm * inv);
}}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let wg = wid.x;
    if (wg >= u.x) {{
        return;
    }}
    let leaf = leaf_ids[wg];
    let cf = cfirst[leaf];
    let cc = ccount[leaf];
    let base = pb_start[wg];
    let total = pb_count[wg];
    let li = lid.x;
    let first = min(li, cc);
    let nown = min((cc - first + WG - 1u) / WG, 4u);
    var acc: array<vec4<f32>, 4>;
    var own_p: array<vec3<f32>, 4>;
    var own_s: array<u32, 4>;
    for (var i = 0u; i < nown; i++) {{
        let s = cf + li + i * WG;
        own_s[i] = s;
        own_p[i] = pos[s].xyz;
        acc[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }}
    let nchunk = (total + TILE - 1u) / TILE;
    for (var ch = 0u; ch < nchunk; ch++) {{
        let cbase = ch * TILE;
        var len = TILE;
        if (cbase + len > total) {{
            len = total - cbase;
        }}
        for (var i = li; i < len; i += WG) {{
            let s = pb_list[base + cbase + i];
            tile[i] = pos[s];
            tids[i] = s;
        }}
        workgroupBarrier();
        for (var i = 0u; i < nown; i++) {{
            let p = own_p[i];
            let s = own_s[i];
            var b0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var b1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var b2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var b3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var j = 0u;
            let main4 = len & 4294967292u;
            for (; j < main4; j += 4u) {{
                if (tids[j] != s) {{
                    b0 += interact(p, tile[j].xyz, tile[j].w);
                }}
                if (tids[j + 1u] != s) {{
                    b1 += interact(p, tile[j + 1u].xyz, tile[j + 1u].w);
                }}
                if (tids[j + 2u] != s) {{
                    b2 += interact(p, tile[j + 2u].xyz, tile[j + 2u].w);
                }}
                if (tids[j + 3u] != s) {{
                    b3 += interact(p, tile[j + 3u].xyz, tile[j + 3u].w);
                }}
            }}
            for (; j < len; j++) {{
                if (tids[j] != s) {{
                    b0 += interact(p, tile[j].xyz, tile[j].w);
                }}
            }}
            acc[i] += b0 + b1 + b2 + b3;
        }}
        workgroupBarrier();
    }}
    for (var i = 0u; i < nown; i++) {{
        accel[own_s[i]] += acc[i];
    }}
}}
"#,
        wg = P2P_WG
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
            storage(3, true),
            storage(4, false),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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

pub fn m2l_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("m2l"),
        source: wgpu::ShaderSource::Wgsl(m2l_source().into()),
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
        source: wgpu::ShaderSource::Wgsl(m2l_red_source().into()),
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
        naga::front::wgsl::parse_str(&m2l_source()).expect("m2l wgsl");
        assert_eq!(DIM, 25);
    }

    #[test]
    fn m2l_red_shader_parses() {
        naga::front::wgsl::parse_str(&m2l_red_source()).expect("m2l-red wgsl");
    }

    #[test]
    fn sweep_shaders_parse() {
        naga::front::wgsl::parse_str(&p2m_source(4)).expect("p2m wgsl");
        naga::front::wgsl::parse_str(&m2m_source(4)).expect("m2m wgsl");
        naga::front::wgsl::parse_str(&l2l_source(4)).expect("l2l wgsl");
        naga::front::wgsl::parse_str(&l2p_source(4)).expect("l2p wgsl");
        naga::front::wgsl::parse_str(&p2p_source(4)).expect("p2p wgsl");
        naga::front::wgsl::parse_str(&fat_p2p_source()).expect("fat p2p wgsl");
        naga::front::wgsl::parse_str(&fat_red_source()).expect("fat red wgsl");
    }

    #[test]
    fn group_m2l_directed_covers_both_ways() {
        let pairs = vec![(0u32, 1u32), (1u32, 2u32)];
        let g = group_m2l(&pairs, 3);
        assert_eq!(g.total, 4);
        let mut got = g.gsrc.clone();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 1, 2]);
        assert_eq!(g.counts, vec![1, 2, 1]);
        assert_eq!(g.starts, vec![0, 1, 3]);
    }

    #[test]
    fn p2p_bodies_expand_partners_and_self() {
        let p2p = vec![(0u32, 0u32), (0u32, 1u32)];
        let pb = build_p2p_bodies(&p2p, 2, &[0, 1], &[0, 3], &[3, 2]);
        assert_eq!(pb.start, vec![0, 5]);
        assert_eq!(pb.count, vec![5, 3]);
        assert_eq!(&pb.list[0..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&pb.list[5..8], &[0, 1, 2]);
    }

    #[test]
    fn fat_split_zeroes_main_and_covers_range() {        let mut pl = PbLists {
            start: vec![0, 4],
            count: vec![4, 2],
            list: vec![0, 1, 2, 3, 4, 5],
        };
        let fj = split_fat(&mut pl, &[0, 1], &[0, 4], &[4, 2], 3, 2).expect("fat split");
        assert_eq!(pl.count, vec![0, 2]);
        assert_eq!((fj.nfat, fj.njobs), (1, 2));
        assert_eq!(fj.job_s0, vec![0, 0]);
        assert_eq!(fj.job_cc, vec![4, 4]);
        assert_eq!(fj.job_off, vec![0, 2]);
        assert_eq!(fj.job_len, vec![2, 2]);
        assert_eq!(fj.fat_start, vec![0, 2]);
        assert_eq!((fj.fat_cf, fj.fat_cc), (vec![0], vec![4]));
        assert_eq!(fj.iters, 16);
    }

    /// GPU-vs-CPU fidelity: same coarse algorithm (p=4/eps=1e-2/n_leaf=64),
    /// f32 vs f64 arithmetic. Needs a GPU adapter:
    /// `cargo test -p fmms-visual -- --ignored gpu_matches_cpu`.
    /// Guards every kernel/shader change (bindings, WG size, softening,
    /// on-device r) against silent numerical breakage.
    #[test]
    #[ignore]
    fn gpu_matches_cpu_fmm() {
        let n = 2000;
        let parts = fmm::models::plummer(n, [0.0; 3], 7);
        let pos: Vec<[f64; 3]> = parts.iter().map(|p| p.pos).collect();
        let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        // CPU reference, then the same accel cap the GPU path applies.
        let fmm = fmm::fmm::Fmm::new(&parts, P, EPS, N_LEAF);
        let mut want = fmm.run_threaded(&parts, 4).accel;
        crate::sim::cap_accel(&mut want);
        // Headless device (same as probe harness).
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("acc-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        }))
        .unwrap();
        let mut be = GpuBackend::new(&device, &queue, u64::MAX);
        let got = be.accel_with_refresh(&pos, &mass, true);
        assert_eq!(got.len(), n);
        let (mut max_rel, mut sum_rel, mut cnt) = (0.0f64, 0.0f64, 0usize);
        for i in 0..n {
            let w = want[i];
            let g = got[i];
            let wn = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            if wn < 1e-9 {
                continue;
            }
            let d = ((g[0] - w[0]).powi(2) + (g[1] - w[1]).powi(2) + (g[2] - w[2]).powi(2)).sqrt();
            let r = d / wn;
            max_rel = max_rel.max(r);
            sum_rel += r;
            cnt += 1;
        }
        let mean = sum_rel / cnt as f64;
        eprintln!("gpu-vs-cpu: max rel {max_rel:.3e} mean {mean:.3e} over {cnt}");
        // Measured ~6e-6 max / ~1e-7 mean; 100x headroom catches real
        // breakage (swapped bindings, wrong WG size) without flaking.
        assert!(max_rel < 1e-4, "max rel err {max_rel}");
        assert!(mean < 1e-5, "mean rel err {mean}");
    }
}

/// Directed M2L pair groups keyed by destination cell.
pub struct M2lGroups {
    pub gsrc: Vec<u32>,
    pub gdst: Vec<u32>,
    pub starts: Vec<u32>,
    pub counts: Vec<u32>,
    pub total: usize,
}

/// Counting-sort undirected M2L pairs into directed per-dst groups.
pub fn group_m2l(pairs: &[(u32, u32)], ncells: usize) -> M2lGroups {
    let mut counts = vec![0u32; ncells];
    for &(a, b) in pairs {
        counts[a as usize] += 1;
        counts[b as usize] += 1;
    }
    let mut starts = vec![0u32; ncells];
    let mut acc = 0u32;
    for c in 0..ncells {
        starts[c] = acc;
        acc += counts[c];
    }
    let total = acc as usize;
    let mut cursors = starts.clone();
    let mut gsrc = vec![0u32; total];
    let mut gdst = vec![0u32; total];
    for &(a, b) in pairs {
        gsrc[cursors[b as usize] as usize] = a;
        gdst[cursors[b as usize] as usize] = b;
        cursors[b as usize] += 1;
        gsrc[cursors[a as usize] as usize] = b;
        gdst[cursors[a as usize] as usize] = a;
        cursors[a as usize] += 1;
    }
    M2lGroups {
        gsrc,
        gdst,
        starts,
        counts,
        total,
    }
}

/// Flat partner-body CSR per leaf ordinal for the tiled P2P kernel.
/// `start[w]`/`count[w]` index `list` with body slots; includes the leaf's
/// own bodies (self-skip happens on GPU).
pub struct PbLists {
    pub start: Vec<u32>,
    pub count: Vec<u32>,
    pub list: Vec<u32>,
}

/// Expand recorded leaf pairs into per-leaf partner-body lists.
/// `leaf_ids[w]` is the cell of leaf ordinal w.
/// Two-pass counting sort (no per-cell allocation): first pass counts
/// partner bodies per ordinal, second fills. Same output order as the old
/// Vec-of-Vecs version (leaves in order, partners in record order, bodies
/// ascending) so kernel behavior is unchanged.
pub fn build_p2p_bodies(
    p2p: &[(u32, u32)],
    ncells: usize,
    leaf_ids: &[u32],
    leaf_first: &[u32],
    leaf_count: &[u32],
) -> PbLists {
    let mut ord_of = vec![u32::MAX; ncells];
    for (w, &l) in leaf_ids.iter().enumerate() {
        ord_of[l as usize] = w as u32;
    }
    let bodies = |q: u32| leaf_count[q as usize] as usize;
    let mut count = vec![0u32; leaf_ids.len()];
    for &(a, b) in p2p {
        // P2P pairs are leaf-leaf by construction (walk invariant); anything
        // else would break the per-ordinal kernel tables downstream.
        debug_assert!(ord_of[a as usize] != u32::MAX && ord_of[b as usize] != u32::MAX);
        count[ord_of[a as usize] as usize] += bodies(b) as u32;
        if a != b {
            count[ord_of[b as usize] as usize] += bodies(a) as u32;
        }
    }
    let mut start = vec![0u32; leaf_ids.len()];
    let mut acc = 0usize;
    for w in 0..leaf_ids.len() {
        start[w] = acc as u32;
        acc += count[w] as usize;
    }
    let mut list = vec![0u32; acc];
    let mut cursor = start.clone();
    let mut push_bodies = |ord: u32, q: u32, cursor: &mut [u32]| {
        let first = leaf_first[q as usize] as usize;
        let n = leaf_count[q as usize] as usize;
        let at = cursor[ord as usize] as usize;
        for k in 0..n {
            list[at + k] = (first + k) as u32;
        }
        cursor[ord as usize] += n as u32;
    };
    for &(a, b) in p2p {
        push_bodies(ord_of[a as usize], b, &mut cursor);
        if a != b {
            push_bodies(ord_of[b as usize], a, &mut cursor);
        }
    }
    PbLists { start, count, list }
}

/// Fat-leaf split: leaves with more than `thr` partner bodies are removed
/// from the main tables (count zeroed) and spread across one workgroup per
/// `chunk` partners. Returns per-job tables plus per-fat-leaf CSR.
pub struct FatJobs {
    pub job_s0: Vec<u32>,
    pub job_cc: Vec<u32>,
    pub job_off: Vec<u32>,
    pub job_len: Vec<u32>,
    pub fat_start: Vec<u32>,
    pub fat_cf: Vec<u32>,
    pub fat_cc: Vec<u32>,
    pub nfat: usize,
    pub njobs: usize,
    #[allow(dead_code)]
    pub bodies: u64,
    #[allow(dead_code)]
    pub iters: u64,
}

pub fn split_fat(
    pl: &mut PbLists,
    leaf_ids: &[u32],
    leaf_first: &[u32],
    leaf_count: &[u32],
    thr: u32,
    chunk: u32,
) -> Result<FatJobs, GpuError> {
    let mut job_s0 = Vec::new();
    let mut job_cc = Vec::new();
    let mut job_off = Vec::new();
    let mut job_len = Vec::new();
    let mut fat_start = Vec::new();
    let mut fat_cf = Vec::new();
    let mut fat_cc = Vec::new();
    let (mut bodies, mut iters) = (0u64, 0u64);
    for (w, &l) in leaf_ids.iter().enumerate() {
        let total = pl.count[w];
        if total <= thr {
            continue;
        }
        fat_start.push(job_s0.len() as u32);
        let cf = leaf_first[l as usize];
        let cc = leaf_count[l as usize];
        // Kernels address 64 lanes max: graceful Err (caller falls back to
        // CPU) instead of the old mid-run assert. Evolution can overfill
        // leaves the init-time build never had.
        if cc > 64 {
            return Err(GpuError::OverfullLeaf { leaf: l, count: cc });
        }
        fat_cf.push(cf);
        fat_cc.push(cc);
        bodies += cc as u64;
        let mut off = pl.start[w];
        let mut left = total;
        while left > 0 {
            let len = left.min(chunk);
            job_s0.push(cf);
            job_cc.push(cc);
            job_off.push(off);
            job_len.push(len);
            iters += len as u64 * cc as u64;
            off += len;
            left -= len;
        }
        pl.count[w] = 0;
    }
    fat_start.push(job_s0.len() as u32);
    let (nfat, njobs) = (fat_cf.len(), job_s0.len());
    Ok(FatJobs {
        job_s0,
        job_cc,
        job_off,
        job_len,
        fat_start,
        fat_cf,
        fat_cc,
        nfat,
        njobs,
        bodies,
        iters,
    })
}

/// One workgroup per fat job: tile one partner slice, partial sums per own
/// body into scratch (reduced by `fat_red_source`).
pub fn fat_p2p_source() -> String {
    r#"const TILE: u32 = 512u;
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> job_s0: array<u32>;
@group(0) @binding(2) var<storage, read> job_cc: array<u32>;
@group(0) @binding(3) var<storage, read> job_off: array<u32>;
@group(0) @binding(4) var<storage, read> job_len: array<u32>;
@group(0) @binding(5) var<storage, read> pb_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> u: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, TILE>;
var<workgroup> tids: array<u32, TILE>;

fn interact(px: vec3<f32>, qx: vec3<f32>, qm: f32) -> vec4<f32> {
    let d = qx - px;
    // Same 1e-12 softening as the main P2P kernel (see above).
    let r2 = dot(d, d) + 1e-12;
    var inv = inverseSqrt(r2);
    inv = inv * (1.5 - 0.5 * r2 * inv * inv);
    inv = inv * (1.5 - 0.5 * r2 * inv * inv);
    let inv3 = inv * inv * inv;
    return vec4<f32>(qm * inv3 * d, qm * inv);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = wid.x;
    if (j >= u.x) {
        return;
    }
    let s0 = job_s0[j];
    let cc = job_cc[j];
    let li = lid.x;
    let on = li < cc;
    var p = vec3<f32>(0.0, 0.0, 0.0);
    var s = 0u;
    if (on) {
        s = s0 + li;
        p = pos[s].xyz;
    }
    let base = job_off[j];
    let total = job_len[j];
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let nchunk = (total + TILE - 1u) / TILE;
    for (var ch = 0u; ch < nchunk; ch++) {
        let cbase = ch * TILE;
        var len = TILE;
        if (cbase + len > total) {
            len = total - cbase;
        }
        for (var i = li; i < len; i += 64u) {
            let q = pb_list[base + cbase + i];
            tile[i] = pos[q];
            tids[i] = q;
        }
        workgroupBarrier();
        if (on) {
            var b0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var b1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var b2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var b3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var k = 0u;
            let main4 = len & 4294967292u;
            for (; k < main4; k += 4u) {
                if (tids[k] != s) {
                    b0 += interact(p, tile[k].xyz, tile[k].w);
                }
                if (tids[k + 1u] != s) {
                    b1 += interact(p, tile[k + 1u].xyz, tile[k + 1u].w);
                }
                if (tids[k + 2u] != s) {
                    b2 += interact(p, tile[k + 2u].xyz, tile[k + 2u].w);
                }
                if (tids[k + 3u] != s) {
                    b3 += interact(p, tile[k + 3u].xyz, tile[k + 3u].w);
                }
            }
            for (; k < len; k++) {
                if (tids[k] != s) {
                    b0 += interact(p, tile[k].xyz, tile[k].w);
                }
            }
            acc += b0 + b1 + b2 + b3;
        }
        workgroupBarrier();
    }
    if (on) {
        scratch[j * 64u + li] = acc;
    }
}
"#
    .to_string()
}

/// One workgroup per fat leaf: sum its jobs' partials into accel.
pub fn fat_red_source() -> String {
    r#"@group(0) @binding(0) var<storage, read> scratch: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> fat_start: array<u32>;
@group(0) @binding(2) var<storage, read> fat_cf: array<u32>;
@group(0) @binding(3) var<storage, read> fat_cc: array<u32>;
@group(0) @binding(4) var<storage, read_write> accel: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let f = wid.x;
    if (f >= u.x) {
        return;
    }
    let li = lid.x;
    if (li >= fat_cc[f]) {
        return;
    }
    var a = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var j = fat_start[f]; j < fat_start[f + 1u]; j++) {
        a += scratch[j * 64u + li];
    }
    accel[fat_cf[f] + li] += a;
}
"#
    .to_string()
}

/// Flat GPU-side cell tables, rebuilt on tree refresh.
pub struct CellTables {
    pub centers: Vec<[f32; 4]>,
    pub cfirst: Vec<u32>,
    pub ccount: Vec<u32>,
    pub children: Vec<u32>,
    pub leaf_ids: Vec<u32>,
    pub level_cells: Vec<u32>,
    pub level_off: Vec<usize>,
    pub cell_of_slot: Vec<u32>,
}

pub fn cell_tables(tree: &Octree, nslots: usize) -> CellTables {
    let ncells = tree.cells.len();
    let centers: Vec<[f32; 4]> = tree
        .cells
        .iter()
        .map(|c| [c.cen[0] as f32, c.cen[1] as f32, c.cen[2] as f32, 0.0])
        .collect();
    let cfirst: Vec<u32> = tree.cells.iter().map(|c| c.first as u32).collect();
    let ccount: Vec<u32> = tree.cells.iter().map(|c| c.count as u32).collect();
    let mut children = vec![u32::MAX; ncells * 8];
    let mut leaf_ids = Vec::new();
    for (c, cell) in tree.cells.iter().enumerate() {
        for k in 0..8 {
            if cell.children[k] >= 0 {
                children[c * 8 + k] = cell.children[k] as u32;
            }
        }
        if tree.is_leaf(c) {
            leaf_ids.push(c as u32);
        }
    }
    let (level_off, level_idx) = tree.levels_csr();
    let level_cells: Vec<u32> = level_idx.iter().map(|&c| c as u32).collect();
    let mut cell_of_slot = vec![0u32; nslots];
    for &l in &leaf_ids {
        let c = l as usize;
        for k in tree.cells[c].first..tree.cells[c].first + tree.cells[c].count {
            cell_of_slot[k] = l;
        }
    }
    CellTables {
        centers,
        cfirst,
        ccount,
        children,
        leaf_ids,
        level_cells,
        level_off,
        cell_of_slot,
    }
}

fn nthreads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(12))
        .unwrap_or(1)
}

/// GPU-resident FMM backend (visual preset p=4/eps=1e-2/n_leaf=64).
///
/// The tree + interaction lists refresh every `k` steps (temporal
/// coherence: bodies move ~v·dt per step, far less than a cell); positions
/// upload fresh every step and moments/locals recompute on GPU, so per-step
/// CPU work is one SoA shuffle plus small uploads.
/// A GPU FMM buffer would exceed the device's buffer/binding limits.
/// Carries which buffer and by how much, so callers can fall back to CPU.
#[derive(Clone, Copy, Debug)]
pub struct OverBudget {
    pub buffer: &'static str,
    pub bytes: u64,
    pub limit: u64,
}

impl std::fmt::Display for OverBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} needs {} MiB, device limit {} MiB",
            self.buffer,
            self.bytes / (1024 * 1024),
            self.limit / (1024 * 1024)
        )
    }
}

/// Fallible GPU FMM failure: over-budget buffers or a broken tree invariant.
/// Both map to the CPU FMM fallback in the app (never a mid-run panic).
#[derive(Clone, Copy, Debug)]
pub enum GpuError {
    OverBudget(OverBudget),
    OverfullLeaf { leaf: u32, count: u32 },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::OverBudget(e) => write!(f, "over budget: {e}"),
            GpuError::OverfullLeaf { leaf, count } => {
                write!(f, "leaf {leaf} holds {count} bodies (kernel takes 64)")
            }
        }
    }
}

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    p2m_pipe: wgpu::ComputePipeline,
    m2m_pipe: wgpu::ComputePipeline,
    l2l_pipe: wgpu::ComputePipeline,
    l2p_pipe: wgpu::ComputePipeline,
    p2p_pipe: wgpu::ComputePipeline,
    fat_pipe: wgpu::ComputePipeline,
    fatred_pipe: wgpu::ComputePipeline,
    s1_pipe: wgpu::ComputePipeline,
    red_pipe: wgpu::ComputePipeline,
    p2m_layout: wgpu::BindGroupLayout,
    m2m_layout: wgpu::BindGroupLayout,
    l2l_layout: wgpu::BindGroupLayout,
    l2p_layout: wgpu::BindGroupLayout,
    p2p_layout: wgpu::BindGroupLayout,
    fat_layout: wgpu::BindGroupLayout,
    fatred_layout: wgpu::BindGroupLayout,
    s1_layout: wgpu::BindGroupLayout,
    red_layout: wgpu::BindGroupLayout,
    bufs: Option<GpuBufs>,
    tables: Option<(CellTables, M2lGroups, PbLists, FatJobs, Vec<u32>)>,
    parts: Vec<fmm::models::Particle>,
    step: u64,
    k: u64,
    /// Per-pass bind groups + frozen uniforms, rebuilt on every refresh.
    /// Tables (and hence counts/offsets) only change on refresh, so the
    /// ~30 buffer+bindgroup creations per step collapse to zero on
    /// non-refresh steps. Invalidated implicitly: rebuilt wherever tables are.
    binds: Option<PassBinds>,
}

/// One cached dispatch: pipeline selector + thread count. The bind group
/// lives in `PassBinds.bg` at the same index.
#[derive(Clone, Copy)]
enum PassKind {
    P2m,
    M2m,
    S1,
    Red,
    L2l,
    L2p,
    P2p,
    Fat,
    Fred,
}

struct PassBinds {
    uni: Vec<wgpu::Buffer>,
    bg: Vec<wgpu::BindGroup>,
    cmds: Vec<(PassKind, u32)>,
}

struct GpuBufs {
    pos: wgpu::Buffer,
    cen: wgpu::Buffer,
    cf: wgpu::Buffer,
    cc: wgpu::Buffer,
    ch: wgpu::Buffer,
    leaves: wgpu::Buffer,
    levels: wgpu::Buffer,
    cos: wgpu::Buffer,
    mom: wgpu::Buffer,
    loc: wgpu::Buffer,
    acc: wgpu::Buffer,
    gsrc: wgpu::Buffer,
    gdst: wgpu::Buffer,
    slots: wgpu::Buffer,
    gs: wgpu::Buffer,
    gc: wgpu::Buffer,
    pbs: wgpu::Buffer,
    pbc: wgpu::Buffer,
    pbl: wgpu::Buffer,
    fjob_s0: wgpu::Buffer,
    fjob_cc: wgpu::Buffer,
    fjob_o: wgpu::Buffer,
    fjob_n: wgpu::Buffer,
    fscr: wgpu::Buffer,
    ffstart: wgpu::Buffer,
    ffcf: wgpu::Buffer,
    ffcc: wgpu::Buffer,
    rb: wgpu::Buffer,
    cap_cells: usize,
    cap_slots: usize,
    cap_pairs: usize,
    cap_pbl: usize,
    cap_fjobs: usize,
    cap_nfat: usize,
}

impl GpuBackend {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, k: u64) -> Self {
        let p2m_layout = sweep_layout(device, "p2m", 6, 5);
        let m2m_layout = sweep_layout(device, "m2m", 4, 0);
        let l2l_layout = sweep_layout(device, "l2l", 4, 0);
        let l2p_layout = sweep_layout(device, "l2p", 5, 4);
        let p2p_layout = sweep_layout(device, "p2p", 8, 7);
        let fat_layout = sweep_layout(device, "fat", 7, 6);
        let fatred_layout = sweep_layout(device, "fatred", 5, 4);
        let s1_layout = m2l_layout(device);
        let red_layout = m2l_red_layout(device);
        let p = P as u32;
        Self {
            device: device.clone(),
            queue: queue.clone(),
            p2m_pipe: sweep_pipeline(device, "p2m", &p2m_source(p), &p2m_layout),
            m2m_pipe: sweep_pipeline(device, "m2m", &m2m_source(p), &m2m_layout),
            l2l_pipe: sweep_pipeline(device, "l2l", &l2l_source(p), &l2l_layout),
            l2p_pipe: sweep_pipeline(device, "l2p", &l2p_source(p), &l2p_layout),
            p2p_pipe: sweep_pipeline(device, "p2p", &p2p_source(p), &p2p_layout),
            fat_pipe: sweep_pipeline(device, "fat", &fat_p2p_source(), &fat_layout),
            fatred_pipe: sweep_pipeline(device, "fatred", &fat_red_source(), &fatred_layout),
            s1_pipe: m2l_pipeline(device, &s1_layout),
            red_pipe: m2l_red_pipeline(device, &red_layout),
            p2m_layout,
            m2m_layout,
            l2l_layout,
            l2p_layout,
            p2p_layout,
            fat_layout,
            fatred_layout,
            s1_layout,
            red_layout,
            bufs: None,
            tables: None,
            parts: Vec::new(),
            step: 0,
            k: k.max(1),
            binds: None,
        }
    }

    fn ensure(
        &mut self,
        ncells: usize,
        nslots: usize,
        total: usize,
        pbl: usize,
        fjobs: usize,
        nfat: usize,
    ) -> Result<(), OverBudget> {
        let ok = match &self.bufs {
            Some(b) => {
                ncells <= b.cap_cells
                    && nslots <= b.cap_slots
                    && total <= b.cap_pairs
                    && pbl <= b.cap_pbl
                    && fjobs <= b.cap_fjobs
                    && nfat <= b.cap_nfat
            }
            None => false,
        };
        if ok {
            return Ok(());
        }
        // Budget pre-check: every buffer below must fit max_buffer_size, and
        // every STORAGE-bound one (all but b-rb) must fit
        // max_storage_buffer_binding_size. Failing here returns OverBudget
        // instead of dying inside wgpu validation.
        let d = DIM as u64;
        let nc = ncells as u64;
        let ns = nslots as u64;
        let tt = total.max(1) as u64;
        let pb = pbl.max(1) as u64;
        let fj = fjobs.max(1) as u64;
        let nf = nfat as u64;
        // (label, bytes, bound_as_storage)
        let sizes: [(&str, u64, bool); 24] = [
            ("b-pos", ns * 16, true),
            ("b-cen", nc * 16, true),
            ("b-cf", nc * 4, true),
            ("b-cc", nc * 4, true),
            ("b-ch", nc * 32, true),
            ("b-leaves", nc * 4, true),
            ("b-levels", nc * 4, true),
            ("b-cos", ns * 4, true),
            ("b-mom", nc * d * 8, true),
            ("b-loc", nc * d * 8, true),
            ("b-acc", ns * 16, true),
            ("b-rb", ns * 16, false),
            ("b-gsrc", tt * 4, true),
            ("b-gdst", tt * 4, true),
            ("b-slots", tt * d * 8, true),
            ("b-gs", nc * 4, true),
            ("b-gc", nc * 4, true),
            ("b-pbs", nc * 4, true),
            ("b-pbc", nc * 4, true),
            ("b-pbl", pb * 4, true),
            ("b-fjob", fj * 4, true),
            ("b-fscr", fj * 64 * 16, true),
            ("b-ffstart", (nf + 1) * 4, true),
            ("b-ffcc", nf * 4, true),
        ];
        let lim = self.device.limits();
        for &(label, bytes, storage) in &sizes {
            let cap = if storage {
                lim.max_storage_buffer_binding_size
            } else {
                lim.max_buffer_size
            };
            if bytes.max(16) > cap {
                return Err(OverBudget {
                    buffer: label,
                    bytes: bytes.max(16),
                    limit: cap,
                });
            }
        }
        let d = DIM;
        let mk = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        self.bufs = Some(GpuBufs {
            pos: mk("b-pos", (nslots * 16) as u64, wgpu::BufferUsages::COPY_DST),
            cen: mk("b-cen", (ncells * 16) as u64, wgpu::BufferUsages::COPY_DST),
            cf: mk("b-cf", (ncells * 4) as u64, wgpu::BufferUsages::COPY_DST),
            cc: mk("b-cc", (ncells * 4) as u64, wgpu::BufferUsages::COPY_DST),
            ch: mk("b-ch", (ncells * 32) as u64, wgpu::BufferUsages::COPY_DST),
            leaves: mk(
                "b-leaves",
                (ncells * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            levels: mk(
                "b-levels",
                (ncells * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            cos: mk("b-cos", (nslots * 4) as u64, wgpu::BufferUsages::COPY_DST),
            mom: mk(
                "b-mom",
                (ncells * d * 8) as u64,
                wgpu::BufferUsages::empty(),
            ),
            loc: mk(
                "b-loc",
                (ncells * d * 8) as u64,
                wgpu::BufferUsages::empty(),
            ),
            acc: mk("b-acc", (nslots * 16) as u64, wgpu::BufferUsages::COPY_SRC),
            rb: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("b-rb"),
                size: ((nslots * 16) as u64).max(16),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            gsrc: mk(
                "b-gsrc",
                (total.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            gdst: mk(
                "b-gdst",
                (total.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            slots: mk(
                "b-slots",
                (total.max(1) * d * 8) as u64,
                wgpu::BufferUsages::empty(),
            ),
            gs: mk("b-gs", (ncells * 4) as u64, wgpu::BufferUsages::COPY_DST),
            gc: mk("b-gc", (ncells * 4) as u64, wgpu::BufferUsages::COPY_DST),
            pbs: mk("b-pbs", (ncells * 4) as u64, wgpu::BufferUsages::COPY_DST),
            pbc: mk("b-pbc", (ncells * 4) as u64, wgpu::BufferUsages::COPY_DST),
            pbl: mk(
                "b-pbl",
                (pbl.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            fjob_s0: mk(
                "b-fjob-s0",
                (fjobs.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            fjob_cc: mk(
                "b-fjob-cc",
                (fjobs.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            fjob_o: mk(
                "b-fjob-o",
                (fjobs.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            fjob_n: mk(
                "b-fjob-n",
                (fjobs.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            fscr: mk(
                "b-fscr",
                (fjobs.max(1) * 64 * 16) as u64,
                wgpu::BufferUsages::empty(),
            ),
            ffstart: mk(
                "b-ffstart",
                ((nfat + 1).max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            ffcf: mk(
                "b-ffcf",
                (nfat.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            ffcc: mk(
                "b-ffcc",
                (nfat.max(1) * 4) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            cap_cells: ncells,
            cap_slots: nslots,
            cap_pairs: total.max(1),
            cap_pbl: pbl.max(1),
            cap_fjobs: fjobs.max(1),
            cap_nfat: nfat.max(1),
        });
        Ok(())
    }

    fn upload<T: bytemuck::Pod>(&self, buf: &wgpu::Buffer, data: &[T]) {
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
    }

    /// (Re)build every pass's bind group + frozen uniform. Call after the
    /// refresh uploads, before encoding: counts/offsets come from the fresh
    /// tables, buffers from `self.bufs`. Same pass order as the old inline
    /// encoder (ordering inside one encoder is preserved).
    fn build_binds(
        &self,
        ct: &CellTables,
        g: &M2lGroups,
        fj: &FatJobs,
        ncells: u32,
        nslots: u32,
        d: u32,
    ) -> PassBinds {
        use wgpu::util::DeviceExt;
        let b = self.bufs.as_ref().expect("bufs for binds");
        let nleaf = ct.leaf_ids.len() as u32;
        let nlevels = ct.level_off.len() - 1;
        let mut pb = PassBinds {
            uni: Vec::new(),
            bg: Vec::new(),
            cmds: Vec::new(),
        };
        let add = |pb: &mut PassBinds,
                       label: &str,
                       layout: &wgpu::BindGroupLayout,
                       bufs: &[&wgpu::Buffer],
                       vals: [u32; 4],
                       kind: PassKind,
                       count: u32| {
            let ub = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g-uni"),
                contents: bytemuck::bytes_of(&vals),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mut all: Vec<&wgpu::Buffer> = bufs.to_vec();
            all.push(&ub);
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &all
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            pb.uni.push(ub);
            pb.bg.push(bind);
            pb.cmds.push((kind, count));
        };
        add(
            &mut pb,
            "p2m",
            &self.p2m_layout,
            &[&b.pos, &b.leaves, &b.cen, &b.cf, &b.cc, &b.mom],
            [d, nleaf, 0, 0],
            PassKind::P2m,
            nleaf,
        );
        for l in (0..nlevels).rev() {
            let (lo, ct_) = (
                ct.level_off[l] as u32,
                (ct.level_off[l + 1] - ct.level_off[l]) as u32,
            );
            if ct_ == 0 {
                continue;
            }
            add(
                &mut pb,
                "m2m",
                &self.m2m_layout,
                &[&b.mom, &b.cen, &b.ch, &b.levels],
                [d, ct_, lo, 0],
                PassKind::M2m,
                ct_,
            );
        }
        add(
            &mut pb,
            "m2ls1",
            &self.s1_layout,
            &[&b.mom, &b.gsrc, &b.cen, &b.gdst, &b.slots],
            [0, d, g.total.max(1) as u32, 0],
            PassKind::S1,
            g.total.max(1) as u32,
        );
        add(
            &mut pb,
            "m2lred",
            &self.red_layout,
            &[&b.slots, &b.gs, &b.gc, &b.loc],
            [ncells, d, 0, 0],
            PassKind::Red,
            ncells * d,
        );
        for l in 0..nlevels {
            let (lo, ct_) = (
                ct.level_off[l] as u32,
                (ct.level_off[l + 1] - ct.level_off[l]) as u32,
            );
            if ct_ == 0 {
                continue;
            }
            add(
                &mut pb,
                "l2l",
                &self.l2l_layout,
                &[&b.loc, &b.cen, &b.ch, &b.levels],
                [d, ct_, lo, 0],
                PassKind::L2l,
                ct_,
            );
        }
        add(
            &mut pb,
            "l2p",
            &self.l2p_layout,
            &[&b.pos, &b.cos, &b.cen, &b.loc, &b.acc],
            [d, nslots, 0, 0],
            PassKind::L2p,
            nslots,
        );
        add(
            &mut pb,
            "p2p",
            &self.p2p_layout,
            &[
                &b.pos, &b.leaves, &b.cf, &b.cc, &b.pbs, &b.pbc, &b.pbl, &b.acc,
            ],
            [nleaf, 0, 0, 0],
            PassKind::P2p,
            // One WG=P2P_WG-wide workgroup per leaf ordinal.
            nleaf * P2P_WG,
        );
        add(
            &mut pb,
            "fat",
            &self.fat_layout,
            &[
                &b.pos, &b.fjob_s0, &b.fjob_cc, &b.fjob_o, &b.fjob_n, &b.pbl, &b.fscr,
            ],
            [fj.njobs as u32, 0, 0, 0],
            PassKind::Fat,
            fj.njobs as u32 * 64,
        );
        add(
            &mut pb,
            "fatred",
            &self.fatred_layout,
            &[&b.fscr, &b.ffstart, &b.ffcf, &b.ffcc, &b.acc],
            [fj.nfat as u32, 0, 0, 0],
            PassKind::Fred,
            fj.nfat as u32 * 64,
        );
        pb
    }

    /// Isolated submit+poll for profiling (probe-only in the app binary).
    #[allow(dead_code)]
    fn submit(
        &self,
        label: &str,
        pipe: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        bufs: &[&wgpu::Buffer],
        vals: &[u32; 4],
        count: u32,
    ) -> f64 {
        use wgpu::util::DeviceExt;
        let ub = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g-uni"),
                contents: bytemuck::bytes_of(vals),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut all: Vec<&wgpu::Buffer> = bufs.to_vec();
        all.push(&ub);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &all
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            p.set_pipeline(pipe);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let t = std::time::Instant::now();
        let idx = self.queue.submit([enc.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(std::time::Duration::from_secs(120)),
            })
            .unwrap();
        t.elapsed().as_secs_f64() * 1000.0
    }

    /// Table counts for profiling (probe-only in the app binary).
    #[allow(dead_code)]
    pub fn stats(&self) -> (usize, usize, usize, usize, usize) {
        let (ct, g, pl, _, _) = self.tables.as_ref().expect("no tables");
        (
            ct.centers.len(),
            g.total,
            ct.leaf_ids.len(),
            ct.level_off.len() - 1,
            pl.list.len(),
        )
    }

    /// P2P work distribution: (max cc, avg cc, max partner bodies, avg, max product).
    #[allow(dead_code)]
    pub fn p2p_dist(&self) -> (u32, f64, u32, f64, u64) {
        let (ct, _, pl, _, _) = self.tables.as_ref().expect("no tables");
        let (mut mcc, mut scc) = (0u32, 0u64);
        for &l in &ct.leaf_ids {
            let c = ct.ccount[l as usize];
            mcc = mcc.max(c);
            scc += c as u64;
        }
        let (mut mt, mut st, mut mx) = (0u32, 0u64, 0u64);
        for (w, &l) in ct.leaf_ids.iter().enumerate() {
            let t = pl.count[w];
            mt = mt.max(t);
            st += t as u64;
            mx = mx.max(t as u64 * ct.ccount[l as usize] as u64);
        }
        let n = ct.leaf_ids.len().max(1);
        (mcc, scc as f64 / n as f64, mt, st as f64 / n as f64, mx)
    }

    /// Fat-leaf census: (split leaves, their bodies, their iters).
    #[allow(dead_code)]
    pub fn fat_census(&self) -> (usize, u64, u64) {
        let (_, _, _, fj, _) = self.tables.as_ref().expect("no tables");
        (fj.nfat, fj.bodies, fj.iters)
    }

    /// One full step with each kernel group submitted + polled separately.
    /// Returns per-stage wall ms. Tables must exist; positions re-uploaded
    /// like the batched path. Probe-only in the app binary.
    #[allow(dead_code)]
    pub fn profile_step(&mut self, pos: &[[f64; 3]], mass: &[f64]) -> Vec<(&'static str, f64)> {
        let n = pos.len();
        self.parts.clear();
        for i in 0..n {
            self.parts.push(fmm::models::Particle {
                mass: mass[i],
                pos: pos[i],
            });
        }
        let (ncells, nslots, nlevels, d, nleaf) = {
            let (ct, _, _, _, order) = self.tables.as_ref().expect("no tables");
            let bodies = fmm::bodies::Bodies::from_parts(&self.parts, order);
            let pos: Vec<[f32; 4]> = (0..n)
                .map(|k| {
                    [
                        bodies.x[k] as f32,
                        bodies.y[k] as f32,
                        bodies.z[k] as f32,
                        bodies.m[k] as f32,
                    ]
                })
                .collect();
            let b = self.bufs.as_ref().unwrap();
            self.upload(&b.pos, &pos);
            (
                ct.centers.len(),
                n as u32,
                ct.level_off.len() - 1,
                DIM as u32,
                ct.leaf_ids.len(),
            )
        };
        let b = self.bufs.as_ref().unwrap();
        let mut out = Vec::new();
        out.push((
            "p2m",
            self.submit(
                "p2m",
                &self.p2m_pipe,
                &self.p2m_layout,
                &[&b.pos, &b.leaves, &b.cen, &b.cf, &b.cc, &b.mom],
                &[d, nleaf as u32, 0, 0],
                nleaf as u32,
            ),
        ));
        let mut t_m2m = 0.0;
        {
            let (ct, _, _, _, _) = self.tables.as_ref().unwrap();
            for l in (0..nlevels).rev() {
                let (lo, ct_) = (
                    ct.level_off[l] as u32,
                    (ct.level_off[l + 1] - ct.level_off[l]) as u32,
                );
                if ct_ == 0 {
                    continue;
                }
                t_m2m += self.submit(
                    "m2m",
                    &self.m2m_pipe,
                    &self.m2m_layout,
                    &[&b.mom, &b.cen, &b.ch, &b.levels],
                    &[d, ct_, lo, 0],
                    ct_,
                );
            }
        }
        out.push(("m2m-all", t_m2m));
        let total = self.tables.as_ref().unwrap().1.total;
        out.push((
            "m2l-s1",
            self.submit(
                "m2ls1",
                &self.s1_pipe,
                &self.s1_layout,
                &[&b.mom, &b.gsrc, &b.cen, &b.gdst, &b.slots],
                &[0, d, total.max(1) as u32, 0],
                total.max(1) as u32,
            ),
        ));
        out.push((
            "m2l-red",
            self.submit(
                "m2lred",
                &self.red_pipe,
                &self.red_layout,
                &[&b.slots, &b.gs, &b.gc, &b.loc],
                &[ncells as u32, d, 0, 0],
                ncells as u32 * d,
            ),
        ));
        let mut t_l2l = 0.0;
        {
            let (ct, _, _, _, _) = self.tables.as_ref().unwrap();
            for l in 0..nlevels {
                let (lo, ct_) = (
                    ct.level_off[l] as u32,
                    (ct.level_off[l + 1] - ct.level_off[l]) as u32,
                );
                if ct_ == 0 {
                    continue;
                }
                t_l2l += self.submit(
                    "l2l",
                    &self.l2l_pipe,
                    &self.l2l_layout,
                    &[&b.loc, &b.cen, &b.ch, &b.levels],
                    &[d, ct_, lo, 0],
                    ct_,
                );
            }
        }
        out.push(("l2l-all", t_l2l));
        out.push((
            "l2p",
            self.submit(
                "l2p",
                &self.l2p_pipe,
                &self.l2p_layout,
                &[&b.pos, &b.cos, &b.cen, &b.loc, &b.acc],
                &[d, nslots, 0, 0],
                nslots,
            ),
        ));
        out.push((
            "p2p",
            self.submit(
                "p2p",
                &self.p2p_pipe,
                &self.p2p_layout,
                &[
                    &b.pos, &b.leaves, &b.cf, &b.cc, &b.pbs, &b.pbc, &b.pbl, &b.acc,
                ],
                &[nleaf as u32, 0, 0, 0],
                nleaf as u32 * P2P_WG,
            ),
        ));
        let (njobs, nfat) = {
            let (_, _, _, fj, _) = self.tables.as_ref().unwrap();
            (fj.njobs as u32, fj.nfat as u32)
        };
        out.push((
            "p2p-fat",
            self.submit(
                "fat",
                &self.fat_pipe,
                &self.fat_layout,
                &[
                    &b.pos, &b.fjob_s0, &b.fjob_cc, &b.fjob_o, &b.fjob_n, &b.pbl, &b.fscr,
                ],
                &[njobs, 0, 0, 0],
                njobs * 64,
            ),
        ));
        out.push((
            "p2p-fred",
            self.submit(
                "fatred",
                &self.fatred_pipe,
                &self.fatred_layout,
                &[&b.fscr, &b.ffstart, &b.ffcf, &b.ffcc, &b.acc],
                &[nfat, 0, 0, 0],
                nfat * 64,
            ),
        ));
        let t = std::time::Instant::now();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prof-copy"),
            });
        enc.copy_buffer_to_buffer(&b.acc, 0, &b.rb, 0, (n * 16) as u64);
        let idx = self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        b.rb.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(std::time::Duration::from_secs(120)),
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        b.rb.unmap();
        out.push(("readback", t.elapsed().as_secs_f64() * 1000.0));
        out
    }

    /// Force evaluation with explicit refresh control (harness use).
    /// Err(GpuError) on over-budget buffers or a broken tree invariant;
    /// callers fall back to CPU instead of hitting wgpu validation.
    /// bufs/tables are left untouched on Err.
    pub fn try_accel_with_refresh(
        &mut self,
        pos: &[[f64; 3]],
        mass: &[f64],
        refresh: bool,
    ) -> Result<Vec<[f64; 3]>, GpuError> {
        let n = pos.len();
        let nt = nthreads();
        self.parts.clear();
        self.parts.reserve(n);
        for i in 0..n {
            self.parts.push(fmm::models::Particle {
                mass: mass[i],
                pos: pos[i],
            });
        }
        if refresh || self.tables.is_none() {
            let tree = Octree::build(&self.parts, N_LEAF);
            let bodies = fmm::bodies::Bodies::from_parts(&self.parts, &tree.order);
            let d = DIM;
            let ncells = tree.cells.len();
            let mut moments = vec![fmm::harmonic::C::ZERO; ncells * d];
            fmm::moments::compute_moments_par(&bodies, &tree, P, &mut moments, nt);
            let pows = fmm::moments::multipole_power(&moments, ncells, P);
            let binom = fmm::moments::binomials(P);
            let (_, fmin) = fmm::mac::f_prepass_par(&bodies, &tree, nt);
            let rec = fmm::walk::record_pairs_par(
                &bodies, &tree, P, EPS, &pows, &binom, &fmin, &moments, nt,
            );
            let g = group_m2l(&rec.pairs, ncells);
            let ct = cell_tables(&tree, n);
            let mut pl = build_p2p_bodies(&rec.p2p, ncells, &ct.leaf_ids, &ct.cfirst, &ct.ccount);
            // Kernels address 64 lanes max: graceful fallback, not a mid-run
            // panic (split_fat re-checks the fat path identically).
            if let Some(&l) = ct
                .leaf_ids
                .iter()
                .find(|&&l| ct.ccount[l as usize] > 64)
            {
                return Err(GpuError::OverfullLeaf {
                    leaf: l,
                    count: ct.ccount[l as usize],
                });
            }
            let fj = split_fat(
                &mut pl,
                &ct.leaf_ids,
                &ct.cfirst,
                &ct.ccount,
                FAT_THR,
                FAT_CHUNK,
            )?;
            self.ensure(ncells, n, g.total, pl.list.len(), fj.njobs, fj.nfat)
                .map_err(GpuError::OverBudget)?;
            let b = self.bufs.as_ref().unwrap();
            self.upload(&b.cen, &ct.centers);
            self.upload(&b.cf, &ct.cfirst);
            self.upload(&b.cc, &ct.ccount);
            self.upload(&b.ch, &ct.children);
            let leaf_u32: Vec<u32> = ct.leaf_ids.clone();
            self.upload(&b.leaves, &leaf_u32);
            let lvl_u32: Vec<u32> = ct.level_cells.iter().map(|&c| c).collect();
            self.upload(&b.levels, &lvl_u32);
            self.upload(&b.cos, &ct.cell_of_slot);
            self.upload(&b.gsrc, &g.gsrc);
            self.upload(&b.gdst, &g.gdst);
            self.upload(&b.gs, &g.starts);
            self.upload(&b.gc, &g.counts);
            self.upload(&b.pbs, &pl.start);
            self.upload(&b.pbc, &pl.count);
            self.upload(&b.pbl, &pl.list);
            self.upload(&b.fjob_s0, &fj.job_s0);
            self.upload(&b.fjob_cc, &fj.job_cc);
            self.upload(&b.fjob_o, &fj.job_off);
            self.upload(&b.fjob_n, &fj.job_len);
            self.upload(&b.ffstart, &fj.fat_start);
            self.upload(&b.ffcf, &fj.fat_cf);
            self.upload(&b.ffcc, &fj.fat_cc);
            // Freeze pass binds for the next k steps (counts/offsets are
            // refresh-stable; buffers outlive in bufs + binds).
            let binds = self.build_binds(&ct, &g, &fj, ncells as u32, n as u32, DIM as u32);
            self.binds = Some(binds);
            self.tables = Some((ct, g, pl, fj, tree.order.clone()));
        }
        let (_ct, _g, _pl, _fj, order) = self.tables.as_ref().unwrap();
        let bodies = fmm::bodies::Bodies::from_parts(&self.parts, order);
        let pos: Vec<[f32; 4]> = (0..n)
            .map(|k| {
                [
                    bodies.x[k] as f32,
                    bodies.y[k] as f32,
                    bodies.z[k] as f32,
                    bodies.m[k] as f32,
                ]
            })
            .collect();
        let b = self.bufs.as_ref().unwrap();
        self.upload(&b.pos, &pos);
        // Encode from cached binds: zero buffer/bindgroup allocation on
        // non-refresh steps (all ~30 were created per step before).
        let binds = self.binds.as_ref().expect("binds built on refresh");
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-step"),
            });
        for (ci, &(kind, count)) in binds.cmds.iter().enumerate() {
            // Layouts were validated at bind-group creation; encode needs
            // only the pipeline + cached group.
            let pipe: &wgpu::ComputePipeline = match kind {
                PassKind::P2m => &self.p2m_pipe,
                PassKind::M2m => &self.m2m_pipe,
                PassKind::S1 => &self.s1_pipe,
                PassKind::Red => &self.red_pipe,
                PassKind::L2l => &self.l2l_pipe,
                PassKind::L2p => &self.l2p_pipe,
                PassKind::P2p => &self.p2p_pipe,
                PassKind::Fat => &self.fat_pipe,
                PassKind::Fred => &self.fatred_pipe,
            };
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            p.set_pipeline(pipe);
            p.set_bind_group(0, &binds.bg[ci], &[]);
            p.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&b.acc, 0, &b.rb, 0, (n * 16) as u64);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        b.rb.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(120)),
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let data = b.rb.get_mapped_range(..).expect("map").to_vec();
        b.rb.unmap();
        let got: &[[f32; 4]] = bytemuck::cast_slice(&data);
        self.step += 1;
        // Scatter slot order back to particle-id order (Physics contract,
        // matching Fmm::run_threaded).
        let order = &self.tables.as_ref().unwrap().4;
        let mut out = vec![[0.0; 3]; n];
        for (k, &id) in order.iter().enumerate() {
            out[id as usize] = [got[k][0] as f64, got[k][1] as f64, got[k][2] as f64];
        }
        crate::sim::cap_accel(&mut out);
        Ok(out)
    }

    /// Compatibility wrapper: panics with the budget message instead of a
    /// wgpu validation error. Probe harnesses use this; the app uses
    /// `try_accel` and falls back to CPU.
    #[allow(dead_code)]
    pub fn accel_with_refresh(
        &mut self,
        pos: &[[f64; 3]],
        mass: &[f64],
        refresh: bool,
    ) -> Vec<[f64; 3]> {
        self.try_accel_with_refresh(pos, mass, refresh)
            .unwrap_or_else(|e| panic!("GPU FMM over budget, use CPU FMM: {e}"))
    }
}

impl GpuBackend {
    /// Physics-trait entry point: refresh every k-th step.
    #[allow(dead_code)]
    pub fn accel(&mut self, pos: &[[f64; 3]], mass: &[f64]) -> Vec<[f64; 3]> {
        let refresh = self.step % self.k == 0;
        self.accel_with_refresh(pos, mass, refresh)
    }

    /// Fallible refresh-every-k-th-step entry point for the app: on
    /// Err(GpuError) the caller swaps in the CPU FMM backend.
    pub fn try_accel(
        &mut self,
        pos: &[[f64; 3]],
        mass: &[f64],
    ) -> Result<Vec<[f64; 3]>, GpuError> {
        let refresh = self.step % self.k == 0;
        self.try_accel_with_refresh(pos, mass, refresh)
    }

    /// Frozen-tree slot→particle-id order (None before first refresh).
    /// Used by validation harnesses; the app path always gets id order.
    #[allow(dead_code)]
    pub fn order(&self) -> Option<&[u32]> {
        self.tables.as_ref().map(|(_, _, _, _, o)| o.as_slice())
    }
}
