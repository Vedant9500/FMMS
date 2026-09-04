#[allow(dead_code)]
#[path = "../camera.rs"]
mod camera;
#[allow(dead_code)]
#[path = "../gpu.rs"]
mod gpu;
#[allow(dead_code)]
#[path = "../gpufmm.rs"]
mod gpufmm;
#[path = "../render.rs"]
mod render;

use camera::{rebase, Camera};
use gpu::{GpuItem, GpuState, Uniform};
use std::time::Duration;

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

fn craft_count(mode: &str, eye: &[f64; 3], count: usize) -> Vec<GpuItem> {
    let mut rng = fmm::models::Rng::new(1234);
    let mut out = Vec::new();
    let mut push = |w: [f64; 3], t: f32, m: f32, size_l: f32| {
        out.push(GpuItem {
            pos: rebase(&w, eye),
            speed: 0.0,
            misc: [t, m, size_l, 0.0],
        });
    };
    match mode {
        "single" => push([0.0, 0.0, 0.0], 0.3, 1.0, 0.0),
        "sparse" => {
            for _ in 0..150 {
                push(
                    [
                        rng.unit() * 40.0 - 20.0,
                        rng.unit() * 40.0 - 20.0,
                        rng.unit() * 40.0 - 20.0,
                    ],
                    rng.unit() as f32,
                    (rng.unit().powf(3.0) * 0.9 + 0.1) as f32,
                    0.0,
                );
            }
            for (w, t, m) in bg_stars() {
                push(w, t, m, 0.001 + m * 0.002);
            }
        }
        _ => {
            for _ in 0..count {
                let mut g = || {
                    let u1 = rng.unit().max(1e-12);
                    let u2 = rng.unit();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                push(
                    [g() * 0.8, g() * 0.8, g() * 0.8],
                    rng.unit() as f32,
                    (rng.unit().powf(3.0) * 0.9 + 0.1) as f32,
                    0.0,
                );
            }
            for (w, t, m) in bg_stars() {
                push(w, t, m, 0.001 + m * 0.002);
            }
        }
    }
    out
}

fn m2l_bench(device: &wgpu::Device, queue: &wgpu::Queue, npairs: usize) {
    use fmm::harmonic::dim;
    let p = gpufmm::P;
    let d = dim(p);
    assert_eq!(d, gpufmm::DIM);
    let parts = fmm::models::plummer(20000, [0.0; 3], 42);
    let tree = fmm::tree::Octree::build(&parts, 64);
    let bodies = fmm::bodies::Bodies::from_parts(&parts, &tree.order);
    let ncells = tree.cells.len();
    let mut moments = vec![fmm::harmonic::C::ZERO; ncells * d];
    fmm::moments::compute_moments(&bodies, &tree, p, &mut moments);
    println!("m2l: ncells={ncells} dim={d}");

    let mut rng = fmm::models::Rng::new(7);
    let mut pairs = Vec::with_capacity(npairs);
    while pairs.len() < npairs {
        let a = (rng.unit() * ncells as f64) as usize % ncells;
        let b = (rng.unit() * ncells as f64) as usize % ncells;
        if a != b {
            pairs.push((a, b));
        }
    }

    // CPU reference on a validation subset (isolated locals per pair).
    let nv = npairs.min(2000);
    let mut cpu_out = vec![[0.0f32; 2]; nv * d];
    let t = std::time::Instant::now();
    for (k, &(a, b)) in pairs.iter().enumerate().take(nv) {
        let mut locals = vec![fmm::harmonic::C::ZERO; ncells * d];
        fmm::translate::m2l_one_sided(&tree, a, b, &moments, &mut locals, p);
        for i in 0..d {
            let c = locals[b * d + i];
            cpu_out[k * d + i] = [c.re as f32, c.im as f32];
        }
    }
    let cpu_val_ms = t.elapsed().as_secs_f64() * 1000.0;
    // CPU throughput loop (shared sink cell; accumulation is part of kernel).
    let b_sink = pairs[0].1;
    let mut locals = vec![fmm::harmonic::C::ZERO; ncells * d];
    let t = std::time::Instant::now();
    for &(a, _) in &pairs {
        fmm::translate::m2l_one_sided(&tree, a, b_sink, &moments, &mut locals, p);
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / npairs as f64 * 1000.0;
    println!("cpu one-sided: {cpu_ms:.3} us/pair (validate loop {cpu_val_ms:.1} ms for {nv})");

    // GPU buffers.
    let m32: Vec<[f32; 2]> = moments.iter().map(|c| [c.re as f32, c.im as f32]).collect();
    let srcs: Vec<u32> = pairs.iter().map(|&(a, _)| a as u32).collect();
    let rs: Vec<[f32; 4]> = pairs
        .iter()
        .map(|&(a, b)| {
            let ca = tree.cells[a].cen;
            let cb = tree.cells[b].cen;
            [
                (cb[0] - ca[0]) as f32,
                (cb[1] - ca[1]) as f32,
                (cb[2] - ca[2]) as f32,
                0.0,
            ]
        })
        .collect();
    let mkbuf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    use wgpu::util::DeviceExt;
    let mom_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("m2l-mom"),
        contents: bytemuck::cast_slice(&m32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("m2l-src"),
        contents: bytemuck::cast_slice(&srcs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let r_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("m2l-r"),
        contents: bytemuck::cast_slice(&rs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = mkbuf(
        "m2l-out",
        (npairs * d * 8) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("m2l-uni"),
        contents: bytemuck::bytes_of(&[ncells as u32, d as u32, npairs as u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = gpufmm::m2l_layout(device);
    let pipe = gpufmm::m2l_pipeline(device, &layout);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("m2l-bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: mom_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: r_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: uni_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("m2l") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("m2l"), timestamp_writes: None });
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((npairs as u32).div_ceil(64), 1, 1);
    }
    let rb = mkbuf(
        "m2l-rb",
        (npairs * d * 8) as u64,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, (npairs * d * 8) as u64);
    let t = std::time::Instant::now();
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    rb.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(60)) })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    let data = rb.get_mapped_range(..).expect("map").to_vec();
    rb.unmap();
    let gpu_out: &[[f32; 2]] = bytemuck::cast_slice(&data);

    let mut max_rel: f32 = 0.0;
    let mut max_abs: f32 = 0.0;
    for k in 0..nv {
        for i in 0..d {
            let c = cpu_out[k * d + i];
            let g = gpu_out[k * d + i];
            let ar = (g[0] - c[0]).abs();
            let ai = (g[1] - c[1]).abs();
            max_abs = max_abs.max(ar).max(ai);
            let denom = c[0].abs().max(c[1].abs()).max(1e-6);
            max_rel = max_rel.max(ar / denom).max(ai / denom);
        }
    }
    println!("gpu one-sided: {:.3} us/pair ({npairs} pairs in {gpu_ms:.1} ms)", gpu_ms * 1000.0 / npairs as f64);
    println!("vs cpu: max_abs={max_abs:e} max_rel={max_rel:e}");
}

fn m2lacc_bench(device: &wgpu::Device, queue: &wgpu::Queue) {
    use fmm::harmonic::{dim, idx, C};
    let p = gpufmm::P;
    let d = dim(p);
    let n = 30000usize;
    let nt = 12usize;
    let parts = fmm::models::plummer(n, [0.0; 3], 42);
    let tree = fmm::tree::Octree::build(&parts, 64);
    let bodies = fmm::bodies::Bodies::from_parts(&parts, &tree.order);
    let ncells = tree.cells.len();
    let mut moments = vec![C::ZERO; ncells * d];
    fmm::moments::compute_moments(&bodies, &tree, p, &mut moments);
    let pows = fmm::moments::multipole_power(&moments, ncells, p);
    let binom = fmm::moments::binomials(p);
    let (_, fmin) = fmm::mac::f_prepass_par(&bodies, &tree, nt);

    let t = std::time::Instant::now();
    let rec = fmm::walk::record_pairs_par(&bodies, &tree, p, 1e-2, &pows, &binom, &fmin, &moments, nt);
    let record_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("record: {} pairs in {record_ms:.1} ms", rec.pairs.len());

    // Group directed pairs by dst cell (CPU counting sort).
    let t = std::time::Instant::now();
    let mut counts = vec![0u32; ncells];
    for &(a, b) in &rec.pairs {
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
    for &(a, b) in &rec.pairs {
        gsrc[cursors[b as usize] as usize] = a;
        gdst[cursors[b as usize] as usize] = b;
        cursors[b as usize] += 1;
        gsrc[cursors[a as usize] as usize] = b;
        gdst[cursors[a as usize] as usize] = a;
        cursors[a as usize] += 1;
    }
    let group_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("group: {total} directed in {group_ms:.1} ms");

    // Reference A: fp64 one-sided accumulation over the same groups.
    let t = std::time::Instant::now();
    let mut th = vec![C::ZERO; d];
    let mut locals_a = vec![C::ZERO; ncells * d];
    for c in 0..ncells {
        for s in 0..counts[c] as usize {
            let src = gsrc[starts[c] as usize + s] as usize;
            let cc = tree.cells[c].cen;
            let cs = tree.cells[src].cen;
            fmm::harmonic::theta_table_into([cc[0] - cs[0], cc[1] - cs[1], cc[2] - cs[2]], p, &mut th);
            for nn in 0..=p {
                for m in -(nn as i32)..=nn as i32 {
                    let mut fb = C::ZERO;
                    for k in 0..=(p - nn) {
                        for l in -(k as i32)..=k as i32 {
                            let t = th[idx(nn + k, m + l)];
                            fb = fb.add(moments[src * d + idx(k, l)].conj().mul(t));
                        }
                    }
                    let o = c * d + idx(nn, m);
                    locals_a[o] = locals_a[o].add(fb);
                }
            }
        }
    }
    let refa_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("cpu refA (serial one-sided): {refa_ms:.1} ms");

    // Reference B: full CPU parallel replay locals.
    let mut b2 = bodies.clone();
    let mut locals_b = vec![C::ZERO; ncells * d];
    let t = std::time::Instant::now();
    fmm::walk::dual_walk_par(&mut b2, &tree, p, 1e-2, &pows, &binom, &fmin, &moments, &mut locals_b, nt);
    let walk_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("cpu dual_walk_par total: {walk_ms:.1} ms");

    // GPU path: S1 kernel over grouped pairs -> per-pair slots, then
    // segmented reduce into per-cell locals.
    use wgpu::util::DeviceExt;
    let m32: Vec<[f32; 2]> = moments.iter().map(|c| [c.re as f32, c.im as f32]).collect();
    let rslots: Vec<[f32; 4]> = (0..total)
        .map(|j| {
            let ca = tree.cells[gsrc[j] as usize].cen;
            let cb = tree.cells[gdst[j] as usize].cen;
            [
                (cb[0] - ca[0]) as f32,
                (cb[1] - ca[1]) as f32,
                (cb[2] - ca[2]) as f32,
                0.0,
            ]
        })
        .collect();
    let init = |label: &str, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let mom_buf = init("acc-mom", bytemuck::cast_slice(&m32));
    let src_buf = init("acc-gsrc", bytemuck::cast_slice(&gsrc));
    let r_buf = init("acc-r", bytemuck::cast_slice(&rslots));
    let slots_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("acc-slots"),
        size: (total * d * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let s1_uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("acc-s1uni"),
        contents: bytemuck::bytes_of(&[0u32, d as u32, total as u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let s1_layout = gpufmm::m2l_layout(device);
    let s1_pipe = gpufmm::m2l_pipeline(device, &s1_layout);
    let s1_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("acc-s1bind"),
        layout: &s1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: mom_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: r_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: slots_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: s1_uni.as_entire_binding() },
        ],
    });
    let gstart_buf = init("acc-gstart", bytemuck::cast_slice(&starts));
    let gcount_buf = init("acc-gcount", bytemuck::cast_slice(&counts));
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("acc-out"),
        size: (ncells * d * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let red_uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("acc-reduni"),
        contents: bytemuck::bytes_of(&[ncells as u32, d as u32, 0u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let red_layout = gpufmm::m2l_red_layout(device);
    let red_pipe = gpufmm::m2l_red_pipeline(device, &red_layout);
    let red_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("acc-redbind"),
        layout: &red_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: slots_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gstart_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gcount_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: red_uni.as_entire_binding() },
        ],
    });
    let up_mb = (m32.len() * 8 + rslots.len() * 16 + gsrc.len() * 4 + starts.len() * 8) as f64 / 1e6;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("acc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("acc-s1"), timestamp_writes: None });
        pass.set_pipeline(&s1_pipe);
        pass.set_bind_group(0, &s1_bind, &[]);
        pass.dispatch_workgroups((total as u32).div_ceil(64), 1, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("acc-red"), timestamp_writes: None });
        pass.set_pipeline(&red_pipe);
        pass.set_bind_group(0, &red_bind, &[]);
        pass.dispatch_workgroups(((ncells * d) as u32).div_ceil(64), 1, 1);
    }
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("acc-rb"),
        size: (ncells * d * 8) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, (ncells * d * 8) as u64);
    let t = std::time::Instant::now();
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    rb.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(60)) })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    let data = rb.get_mapped_range(..).expect("map").to_vec();
    rb.unmap();
    let gpu_out: &[[f32; 2]] = bytemuck::cast_slice(&data);
    let down_mb = data.len() as f64 / 1e6;
    println!("gpu replay: {gpu_ms:.1} ms (up {up_mb:.1} MB, down {down_mb:.1} MB)");

    let cmp = |name: &str, refs: &[C]| {
        let mut max_abs: f32 = 0.0;
        let mut max_rel: f32 = 0.0;
        for i in 0..ncells * d {
            let c = refs[i];
            let g = gpu_out[i];
            let ar = (g[0] - c.re as f32).abs();
            let ai = (g[1] - c.im as f32).abs();
            max_abs = max_abs.max(ar).max(ai);
            let denom = (c.re as f32).abs().max((c.im as f32).abs()).max(1e-6);
            max_rel = max_rel.max(ar / denom).max(ai / denom);
        }
        println!("vs {name}: max_abs={max_abs:e} max_rel={max_rel:e}");
    };
    cmp("refA serial-one-sided", &locals_a);
    cmp("refB dual_walk_par", &locals_b);
}
fn gpufull_bench(device: &wgpu::Device, queue: &wgpu::Queue, n: usize, reference: &str, p: usize, eps: f64) {
    use fmm::harmonic::{dim, C};
    let d = dim(p);
    let nt = 12usize;
    let parts = fmm::models::plummer(n, [0.0; 3], 42);
    let tree = fmm::tree::Octree::build(&parts, 64);
    let bodies = fmm::bodies::Bodies::from_parts(&parts, &tree.order);
    let ncells = tree.cells.len();
    let nslots = bodies.x.len();

    // CPU cell tables.
    let centers: Vec<[f32; 4]> = tree.cells.iter()
        .map(|c| [c.cen[0] as f32, c.cen[1] as f32, c.cen[2] as f32, 0.0]).collect();
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
    let nlevels = level_off.len() - 1;
    let mut cell_of_slot = vec![0u32; nslots];
    for &l in &leaf_ids {
        let c = l as usize;
        for k in tree.cells[c].first..tree.cells[c].first + tree.cells[c].count {
            cell_of_slot[k] = l;
        }
    }
    let pos: Vec<[f32; 4]> = (0..nslots)
        .map(|k| [bodies.x[k] as f32, bodies.y[k] as f32, bodies.z[k] as f32, bodies.m[k] as f32])
        .collect();

    // Traversal: M2L pairs + P2P leaf pairs.
    let moments_cpu = {
        let mut m = vec![C::ZERO; ncells * d];
        fmm::moments::compute_moments(&bodies, &tree, p, &mut m);
        m
    };
    let pows = fmm::moments::multipole_power(&moments_cpu, ncells, p);
    let binom = fmm::moments::binomials(p);
    let (_, fmin) = fmm::mac::f_prepass_par(&bodies, &tree, nt);
    let t = std::time::Instant::now();
    let rec = fmm::walk::record_pairs_par(&bodies, &tree, p, eps, &pows, &binom, &fmin, &moments_cpu, nt);
    let record_ms = t.elapsed().as_secs_f64() * 1000.0;

    // M2L groups by dst.
    let mut counts = vec![0u32; ncells];
    for &(a, b) in &rec.pairs {
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
    for &(a, b) in &rec.pairs {
        gsrc[cursors[b as usize] as usize] = a;
        gdst[cursors[b as usize] as usize] = b;
        cursors[b as usize] += 1;
        gsrc[cursors[a as usize] as usize] = b;
        gdst[cursors[a as usize] as usize] = a;
        cursors[a as usize] += 1;
    }

    // P2P partner-body CSR per leaf ordinal (tiled kernel) + fat split.
    let mut pb = gpufmm::build_p2p_bodies(&rec.p2p, ncells, &leaf_ids, &cfirst, &ccount);
    let fj = gpufmm::split_fat(&mut pb, &leaf_ids, &cfirst, &ccount, gpufmm::FAT_THR, gpufmm::FAT_CHUNK);
    println!("gpufull n={n}: cells={ncells} leaves={} m2l={} directed={total} p2pbodies={} record={record_ms:.1}ms",
        leaf_ids.len(), rec.pairs.len(), pb.list.len());

    // GPU buffers.
    use wgpu::util::DeviceExt;
    let sbuf = |label: &str, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let pos_buf = sbuf("g-pos", bytemuck::cast_slice(&pos));
    let cen_buf = sbuf("g-cen", bytemuck::cast_slice(&centers));
    let cf_buf = sbuf("g-cfirst", bytemuck::cast_slice(&cfirst));
    let cc_buf = sbuf("g-ccount", bytemuck::cast_slice(&ccount));
    let ch_buf = sbuf("g-children", bytemuck::cast_slice(&children));
    let leaf_buf = sbuf("g-leaves", bytemuck::cast_slice(&leaf_ids));
    let lvl_buf = sbuf("g-levels", bytemuck::cast_slice(&level_idx.iter().map(|&c| c as u32).collect::<Vec<_>>()));
    let cos_buf = sbuf("g-cos", bytemuck::cast_slice(&cell_of_slot));
    let mom_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g-mom"), size: (ncells * d * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let loc_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g-loc"), size: (ncells * d * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let acc_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g-acc"), size: (nslots * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let rslots: Vec<[f32; 4]> = (0..total)
        .map(|j| {
            let ca = tree.cells[gsrc[j] as usize].cen;
            let cb = tree.cells[gdst[j] as usize].cen;
            [(cb[0] - ca[0]) as f32, (cb[1] - ca[1]) as f32, (cb[2] - ca[2]) as f32, 0.0]
        })
        .collect();
    let gsrc_buf = sbuf("g-gsrc", bytemuck::cast_slice(&gsrc));
    let gr_buf = sbuf("g-gr", bytemuck::cast_slice(&rslots));
    let slots_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g-slots"), size: (total.max(1) * d * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let gs_buf = sbuf("g-gstart", bytemuck::cast_slice(&starts));
    let gc_buf = sbuf("g-gcount", bytemuck::cast_slice(&counts));
    let pbs_buf = sbuf("g-pbstart", bytemuck::cast_slice(&pb.start));
    let pbc_buf = sbuf("g-pbcount", bytemuck::cast_slice(&pb.count));
    let pbl_buf = sbuf("g-pblist", bytemuck::cast_slice(&pb.list));
    let fjs_buf = sbuf("g-fjob-s0", bytemuck::cast_slice(&fj.job_s0));
    let fjc_buf = sbuf("g-fjob-cc", bytemuck::cast_slice(&fj.job_cc));
    let fjo_buf = sbuf("g-fjob-o", bytemuck::cast_slice(&fj.job_off));
    let fjn_buf = sbuf("g-fjob-n", bytemuck::cast_slice(&fj.job_len));
    let fscr_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g-fscr"), size: (fj.njobs.max(1) * 64 * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let ffs_buf = sbuf("g-ffstart", bytemuck::cast_slice(&fj.fat_start));
    let ffc_buf = sbuf("g-ffcf", bytemuck::cast_slice(&fj.fat_cf));
    let ffx_buf = sbuf("g-ffcc", bytemuck::cast_slice(&fj.fat_cc));
    let bind2 = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g-bind"),
            layout,
            entries: &bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            }).collect::<Vec<_>>(),
        })
    };

    let p2m_l = gpufmm::sweep_layout(device, "p2m", 6, 5);
    let p2m_p = gpufmm::sweep_pipeline(device, "p2m", &gpufmm::p2m_source(p as u32), &p2m_l);
    let m2m_l = gpufmm::sweep_layout(device, "m2m", 4, 0);
    let m2m_p = gpufmm::sweep_pipeline(device, "m2m", &gpufmm::m2m_source(p as u32), &m2m_l);
    let l2l_l = gpufmm::sweep_layout(device, "l2l", 4, 0);
    let l2l_p = gpufmm::sweep_pipeline(device, "l2l", &gpufmm::l2l_source(p as u32), &l2l_l);
    let l2p_l = gpufmm::sweep_layout(device, "l2p", 5, 4);
    let l2p_p = gpufmm::sweep_pipeline(device, "l2p", &gpufmm::l2p_source(p as u32), &l2p_l);
    let p2p_l = gpufmm::sweep_layout(device, "p2p", 8, 7);
    let p2p_p = gpufmm::sweep_pipeline(device, "p2p", &gpufmm::p2p_source(p as u32), &p2p_l);
    let fat_l = gpufmm::sweep_layout(device, "fat", 7, 6);
    let fat_p = gpufmm::sweep_pipeline(device, "fat", &gpufmm::fat_p2p_source(), &fat_l);
    let fatred_l = gpufmm::sweep_layout(device, "fatred", 5, 4);
    let fatred_p = gpufmm::sweep_pipeline(device, "fatred", &gpufmm::fat_red_source(), &fatred_l);
    let s1_l = gpufmm::m2l_layout(device);
    let s1_p = gpufmm::m2l_pipeline(device, &s1_l);
    let red_l = gpufmm::m2l_red_layout(device);
    let red_p = gpufmm::m2l_red_pipeline(device, &red_l);

    let t = std::time::Instant::now();
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gfull") });
    // NOTE: each dispatch gets its OWN uniform buffer. A shared buffer
    // rewritten between dispatches does NOT work: all writes land before
    // the single submit, so every pass would read the last value.
    let mkuni = |vals: &[u32; 4]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("g-uni"),
            contents: bytemuck::bytes_of(vals),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };
    let mut keepalive: Vec<wgpu::Buffer> = Vec::new();
    let mut keepbind: Vec<wgpu::BindGroup> = Vec::new();
    let pass = |enc: &mut wgpu::CommandEncoder, label: &str, pipe: &wgpu::ComputePipeline,
                layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer], vals: &[u32; 4], n: u32,
                keepalive: &mut Vec<wgpu::Buffer>, keepbind: &mut Vec<wgpu::BindGroup>| {
        let ub = mkuni(vals);
        let mut all: Vec<&wgpu::Buffer> = bufs.to_vec();
        all.push(&ub);
        let bg = bind2(layout, &all);
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            p.set_pipeline(pipe);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        keepalive.push(ub);
        keepbind.push(bg);
    };
    pass(&mut enc, "p2m", &p2m_p, &p2m_l, &[&pos_buf, &leaf_buf, &cen_buf, &cf_buf, &cc_buf, &mom_buf],
        &[d as u32, leaf_ids.len() as u32, 0, 0], leaf_ids.len() as u32, &mut keepalive, &mut keepbind);
    for l in (0..nlevels).rev() {
        let lo = level_off[l];
        let ct = level_off[l + 1] - lo;
        let (lo, ct) = (lo as u32, ct as u32);
        if ct == 0 { continue; }
        pass(&mut enc, "m2m", &m2m_p, &m2m_l, &[&mom_buf, &cen_buf, &ch_buf, &lvl_buf],
            &[d as u32, ct, lo, 0], ct, &mut keepalive, &mut keepbind);
    }
    pass(&mut enc, "m2ls1", &s1_p, &s1_l, &[&mom_buf, &gsrc_buf, &gr_buf, &slots_buf],
        &[0u32, d as u32, total as u32, 0u32], total.max(1) as u32, &mut keepalive, &mut keepbind);
    pass(&mut enc, "m2lred", &red_p, &red_l, &[&slots_buf, &gs_buf, &gc_buf, &loc_buf],
        &[ncells as u32, d as u32, 0u32, 0u32], (ncells * d) as u32, &mut keepalive, &mut keepbind);
    for l in 0..nlevels {
        let lo = level_off[l];
        let ct = level_off[l + 1] - lo;
        let (lo, ct) = (lo as u32, ct as u32);
        if ct == 0 { continue; }
        pass(&mut enc, "l2l", &l2l_p, &l2l_l, &[&loc_buf, &cen_buf, &ch_buf, &lvl_buf],
            &[d as u32, ct, lo, 0], ct, &mut keepalive, &mut keepbind);
    }
    pass(&mut enc, "l2p", &l2p_p, &l2p_l, &[&pos_buf, &cos_buf, &cen_buf, &loc_buf, &acc_buf],
        &[d as u32, nslots as u32, 0, 0], nslots as u32, &mut keepalive, &mut keepbind);
    pass(&mut enc, "p2p", &p2p_p, &p2p_l, &[&pos_buf, &leaf_buf, &cf_buf, &cc_buf, &pbs_buf, &pbc_buf, &pbl_buf, &acc_buf],
        &[leaf_ids.len() as u32, 0, 0, 0], leaf_ids.len() as u32 * 64, &mut keepalive, &mut keepbind);
    pass(&mut enc, "fat", &fat_p, &fat_l, &[&pos_buf, &fjs_buf, &fjc_buf, &fjo_buf, &fjn_buf, &pbl_buf, &fscr_buf],
        &[fj.njobs as u32, 0, 0, 0], fj.njobs as u32 * 64, &mut keepalive, &mut keepbind);
    pass(&mut enc, "fatred", &fatred_p, &fatred_l, &[&fscr_buf, &ffs_buf, &ffc_buf, &ffx_buf, &acc_buf],
        &[fj.nfat as u32, 0, 0, 0], fj.nfat as u32 * 64, &mut keepalive, &mut keepbind);
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g-rb"), size: (nslots * 16) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&acc_buf, 0, &rb, 0, (nslots * 16) as u64);
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    rb.map_async(wgpu::MapMode::Read, .., move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(120)) }).unwrap();
    rx.recv().unwrap().unwrap();
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    let data = rb.get_mapped_range(..).expect("map").to_vec();
    rb.unmap();
    let gpu_acc: &[[f32; 4]] = bytemuck::cast_slice(&data);
    println!("gpu full eval: {gpu_ms:.1} ms");

    if reference == "direct" {
        let t = std::time::Instant::now();
        let (da, _) = fmm::direct::direct(&parts);
        println!("cpu direct: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        let (max_rel, mean_rel) = rel_err(&gpu_acc, &da, &tree.order, nslots);
        println!("gpu vs direct: max_rel={max_rel:e} mean_rel={mean_rel:e}");
    } else {
        let t = std::time::Instant::now();
        let fmm = fmm::fmm::Fmm::new(&parts, p, eps, 64);
        let out = fmm.run_threaded(&parts, nt);
        println!("cpu fmm: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        let (max_rel, mean_rel) = rel_err_gpu_fmm(&gpu_acc, &out.accel, &tree.order, nslots);
        println!("gpu vs cpu-fmm: max_rel={max_rel:e} mean_rel={mean_rel:e}");
        let t = std::time::Instant::now();
        let (da, _) = fmm::direct::direct(&parts);
        println!("cpu direct: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        let (tmax, tmean) = rel_err_fmm_direct(&out.accel, &da, n);
        println!("cpu-fmm vs direct (truncation floor): max={tmax:e} mean={tmean:e}");
    }
}

fn rel_err(gpu_acc: &[[f32; 4]], da: &[[f64; 3]], order: &[u32], nslots: usize) -> (f64, f64) {
    let mut max_rel = 0.0f64;
    let mut sum_rel = 0.0f64;
    for k in 0..nslots {
        let id = order[k] as usize;
        let dx = gpu_acc[k][0] as f64 - da[id][0];
        let dy = gpu_acc[k][1] as f64 - da[id][1];
        let dz = gpu_acc[k][2] as f64 - da[id][2];
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        let mag = (da[id][0] * da[id][0] + da[id][1] * da[id][1] + da[id][2] * da[id][2]).sqrt().max(1e-9);
        max_rel = max_rel.max(err / mag);
        sum_rel += err / mag;
    }
    (max_rel, sum_rel / nslots as f64)
}

fn rel_err_gpu_fmm(gpu_acc: &[[f32; 4]], fa: &[[f64; 3]], order: &[u32], nslots: usize) -> (f64, f64) {
    let mut max_rel = 0.0f64;
    let mut sum_rel = 0.0f64;
    for k in 0..nslots {
        let id = order[k] as usize;
        let dx = gpu_acc[k][0] as f64 - fa[id][0];
        let dy = gpu_acc[k][1] as f64 - fa[id][1];
        let dz = gpu_acc[k][2] as f64 - fa[id][2];
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        let mag = (fa[id][0] * fa[id][0] + fa[id][1] * fa[id][1] + fa[id][2] * fa[id][2]).sqrt().max(1e-9);
        max_rel = max_rel.max(err / mag);
        sum_rel += err / mag;
    }
    (max_rel, sum_rel / nslots as f64)
}

fn rel_err_fmm_direct(fa: &[[f64; 3]], da: &[[f64; 3]], n: usize) -> (f64, f64) {
    let mut max_rel = 0.0f64;
    let mut sum_rel = 0.0f64;
    for id in 0..n {
        let dx = fa[id][0] - da[id][0];
        let dy = fa[id][1] - da[id][1];
        let dz = fa[id][2] - da[id][2];
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        let mag = (da[id][0] * da[id][0] + da[id][1] * da[id][1] + da[id][2] * da[id][2]).sqrt().max(1e-9);
        max_rel = max_rel.max(err / mag);
        sum_rel += err / mag;
    }
    (max_rel, sum_rel / n as f64)
}

fn gpuloop_bench(device: &wgpu::Device, queue: &wgpu::Queue, n: usize, steps: usize) {
    let parts = fmm::models::plummer(n, [0.0; 3], 42);
    let pos: Vec<[f64; 3]> = parts.iter().map(|p| p.pos).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let mut be = gpufmm::GpuBackend::new(device, queue, 4);
    be.accel(&pos, &mass);
    let t = std::time::Instant::now();
    let mut per = Vec::with_capacity(steps);
    for _ in 0..steps {
        let s = std::time::Instant::now();
        be.accel(&pos, &mass);
        per.push(s.elapsed().as_secs_f64() * 1000.0);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!("gpuloop n={n} steps={steps} k=4: {ms:.1} ms/step = {:.1} steps/s", 1000.0 / ms);
    if steps <= 16 {
        let s: Vec<String> = per.iter().map(|v| format!("{v:.0}")).collect();
        println!("  per-step ms: {}", s.join(" "));
    }
}

fn gpuprof_bench(device: &wgpu::Device, queue: &wgpu::Queue, n: usize, reps: usize) {
    let parts = fmm::models::plummer(n, [0.0; 3], 42);
    let pos: Vec<[f64; 3]> = parts.iter().map(|p| p.pos).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let mut be = gpufmm::GpuBackend::new(device, queue, u64::MAX);
    be.accel_with_refresh(&pos, &mass, true);
    let (ncells, total, nleaf, nlevels, nlist) = be.stats();
    let (mcc, avgcc, mt, at, mx) = be.p2p_dist();
    println!("gpuprof n={n}: cells={ncells} leaves={nleaf} levels={nlevels} directed={total} p2pnlist={nlist}");
    println!("  leaf cc: max={mcc} avg={avgcc:.1} | partners: max={mt} avg={at:.0} | max prod={mx}");
    let (nl, nb, ni) = be.fat_census();
    println!("  fat-split: leaves={nl} bodies={nb} iters={ni}");
    println!("f64: {} timestamp: {}", device.features().contains(wgpu::Features::SHADER_F64),
        device.features().contains(wgpu::Features::TIMESTAMP_QUERY));
    let mut best: Vec<(&str, f64)> = be.profile_step(&pos, &mass).into_iter().map(|(k, _)| (k, f64::MAX)).collect();
    for _ in 0..reps {
        for (i, (k, v)) in be.profile_step(&pos, &mass).into_iter().enumerate() {
            if v < best[i].1 {
                best[i] = (k, v);
            }
        }
    }
    let sum: f64 = best.iter().map(|(_, v)| v).sum();
    for (k, v) in &best {
        println!("  {k:8} {v:7.2} ms ({:.0}%)", 100.0 * v / sum.max(1e-9));
    }
    println!("  {:8} {sum:.2} ms split-total", "sum");
}

fn gpureuse_check(device: &wgpu::Device, queue: &wgpu::Queue, k: u64) {
    let n = 30000usize;
    let parts = fmm::models::plummer(n, [0.0; 3], 42);
    let pos0: Vec<[f64; 3]> = parts.iter().map(|p| p.pos).collect();
    let mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    // Random-walk drift proxy: K steps x N(0, 0.007) per dim (~v*dt per step).
    let mut rng = fmm::models::Rng::new(5);
    let gauss = |rng: &mut fmm::models::Rng| {
        let u1 = rng.unit().max(1e-12);
        let u2 = rng.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 0.007
    };
    let mut pos = pos0.clone();
    for _ in 0..k {
        for p in pos.iter_mut() {
            for d in 0..3 {
                p[d] += gauss(&mut rng);
            }
        }
    }
    let mut be = gpufmm::GpuBackend::new(device, queue, u64::MAX);
    let a_fresh = be.accel_with_refresh(&pos, &mass, true);
    let mut be2 = gpufmm::GpuBackend::new(device, queue, u64::MAX);
    be2.accel_with_refresh(&pos0, &mass, true);
    let a_frozen = be2.accel_with_refresh(&pos, &mass, false);
    let fmm = fmm::fmm::Fmm::new(
        &pos.iter().zip(mass.iter()).map(|(p, m)| fmm::models::Particle { mass: *m, pos: *p }).collect::<Vec<_>>(),
        4, 1e-2, 64,
    );
    let tmp_parts: Vec<fmm::models::Particle> = pos.iter().zip(mass.iter())
        .map(|(p, m)| fmm::models::Particle { mass: *m, pos: *p }).collect();
    let out = fmm.run_threaded(&tmp_parts, 12);
    let cmp = |name: &str, a: &[[f64; 3]], b: &[[f64; 3]]| {
        let (mut mx, mut sm) = (0.0f64, 0.0f64);
        for i in 0..n {
            let e = ((a[i][0] - b[i][0]).powi(2) + (a[i][1] - b[i][1]).powi(2) + (a[i][2] - b[i][2]).powi(2)).sqrt();
            let m = (b[i][0].powi(2) + b[i][1].powi(2) + b[i][2].powi(2)).sqrt().max(1e-9);
            mx = mx.max(e / m);
            sm += e / m;
        }
        println!("{name}: max={mx:e} mean={:e}", sm / n as f64);
    };
    cmp("frozen-K vs cpu-fmm-fresh", &a_frozen, &out.accel);
    cmp("fresh-gpu vs cpu-fmm-fresh", &a_fresh, &out.accel);
}

fn phys_bench(n: usize, p: usize, eps: f64, n_leaf: usize, iters: usize, nt: usize) {
    let parts = fmm::models::plummer(n, [0.0; 3], 42);
    let mut f = fmm::fmm::Fmm::new(&parts, p, eps, n_leaf);
    // warmup (allocates thread pools/scratch on first run)
    let _ = f.run_threaded(&parts, nt);
    f.refresh(&parts);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f.refresh(&parts);
    }
    let ms_r = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = f.run_threaded(&parts, nt);
    }
    let ms_f = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("phys n={n} p={p} eps={eps} n_leaf={n_leaf} threads={nt}: refresh {ms_r:.1} + force {ms_f:.1} ms/eval");
}

fn save_ppm(path: &str, w: u32, h: u32, rgba: &[u8], stride: usize) {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h as usize {
        let row = &rgba[y * stride..y * stride + w as usize * 4];
        for px in row.chunks_exact(4) {
            buf.extend_from_slice(&px[..3]);
        }
    }
    let mut f = std::fs::File::create(path).unwrap();
    use std::io::Write;
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    f.write_all(&buf).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut w = 800u32;
    let mut h = 600u32;
    let mut mode = "sparse".to_string();
    let mut out = "/tmp/opencode/probe".to_string();
    let mut bloom = 0.35f32;
    let mut threshold = 1.5f32;
    let mut pairs = 2u32;
    let mut dump_stages = false;
    let mut m2l_mode: Option<usize> = None;
    let mut m2lacc_mode = false;
    let mut gpufull_mode: Option<(usize, String, usize, f64)> = None;
    let mut gpuloop_mode: Option<(usize, usize)> = None;
    let mut gpuprof_mode: Option<(usize, usize)> = None;
    let mut gpureuse_mode: Option<u64> = None;
    let mut count = 1500usize;
    let mut point_size = 0.035f32;
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--w" => {
                w = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(w);
                i += 2;
            }
            "--h" => {
                h = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(h);
                i += 2;
            }
            "--mode" => {
                mode = a.get(i + 1).cloned().unwrap_or(mode);
                i += 2;
            }
            "--out" => {
                out = a.get(i + 1).cloned().unwrap_or(out);
                i += 2;
            }
            "--bloom" => {
                bloom = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(bloom);
                i += 2;
            }
            "--threshold" => {
                threshold = a
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(threshold);
                i += 2;
            }
            "--pairs" => {
                pairs = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(pairs);
                i += 2;
            }
            "--dump" => {
                dump_stages = true;
                i += 1;
            }
            "--count" => {
                count = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(count);
                i += 2;
            }
            "--size" => {
                point_size = a
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(point_size);
                i += 2;
            }
            "--phys" => {
                let get = |k: usize, d: &str| a.get(i + k).cloned().unwrap_or(d.to_string());
                let n: usize = get(1, "100000").parse().unwrap_or(100000);
                let p: usize = get(2, "4").parse().unwrap_or(4);
                let eps: f64 = get(3, "0.01").parse().unwrap_or(0.01);
                let nl: usize = get(4, "64").parse().unwrap_or(64);
                let it: usize = get(5, "5").parse().unwrap_or(5);
                let nt: usize = get(6, "12").parse().unwrap_or(12);
                phys_bench(n, p, eps, nl, it, nt);
                return;
            }
            "--m2l" => {
                let np: usize = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(50000);
                m2l_mode = Some(np);
                i += 2;
            }
            "--m2lacc" => {
                m2lacc_mode = true;
                i += 1;
            }
            "--gpufull" => {
                let nn: usize = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8192);
                let rf: String = a.get(i + 2).cloned().unwrap_or("direct".to_string());
                let pp: usize = a.get(i + 3).and_then(|s| s.parse().ok()).unwrap_or(4);
                let ee: f64 = a.get(i + 4).and_then(|s| s.parse().ok()).unwrap_or(0.01);
                gpufull_mode = Some((nn, rf, pp, ee));
                i += 5;
            }
            "--gpuloop" => {
                let nn: usize = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(100000);
                let st: usize = a.get(i + 2).and_then(|s| s.parse().ok()).unwrap_or(8);
                gpuloop_mode = Some((nn, st));
                i += 3;
            }
            "--gpureuse" => {
                let kk: u64 = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4);
                gpureuse_mode = Some(kk);
                i += 2;
            }
            "--gpuprof" => {
                let nn: usize = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(100000);
                let rr: usize = a.get(i + 2).and_then(|s| s.parse().ok()).unwrap_or(5);
                gpuprof_mode = Some((nn, rr));
                i += 3;
            }
            _ => i += 1,
        }
    }
    let exposure = 1.4f32;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter");
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("probe"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: Default::default(),
        memory_hints: Default::default(),
        trace: Default::default(),
    }))
    .unwrap();
    if let Some(np) = m2l_mode {
        m2l_bench(&device, &queue, np);
        return;
    }
    if m2lacc_mode {
        m2lacc_bench(&device, &queue);
        return;
    }
    if let Some((nn, rf, pp, ee)) = gpufull_mode {
        gpufull_bench(&device, &queue, nn, &rf, pp, ee);
        return;
    }
    if let Some((nn, st)) = gpuloop_mode {
        gpuloop_bench(&device, &queue, nn, st);
        return;
    }
    if let Some((nn, rr)) = gpuprof_mode {
        gpuprof_bench(&device, &queue, nn, rr);
        return;
    }
    if let Some(kk) = gpureuse_mode {
        gpureuse_check(&device, &queue, kk);
        return;
    }

    let camera = Camera::new();
    let eye = camera.eye();
    let items = craft_count(&mode, &eye, count);
    let n = items.len();
    println!("items: {n}");

    let (uniform_layout, storage_layout) = render::layouts(&device);
    let tex_layout = render::tex_layout(&device, "tex-layout");
    let composite_layout = render::composite_layout(&device);
    let uni_layout = render::blur_uni_layout(&device);
    let mut state = GpuState::new(&device, &uniform_layout, &storage_layout, n, w, h);
    state.n = n;
    let billboard =
        render::billboard_pipeline(&device, state.hdr_format, &uniform_layout, &storage_layout);
    let bright = render::bright_pipeline(&device, state.hdr_format, &tex_layout, &uni_layout);
    let blur = render::blur_pipeline(&device, state.hdr_format, &tex_layout, &uni_layout);
    let final_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let tonemap = render::tonemap_pipeline(&device, final_format, &composite_layout, &uni_layout);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("post-smp"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let mkuni = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let blur_uni_h = mkuni("blur-uni-h");
    let blur_uni_v = mkuni("blur-uni-v");
    let bright_uni = mkuni("bright-uni");
    let bloom_uni = mkuni("bloom-uni");
    let tex_bind = |label: &str, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    };
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
    let final_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("final"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: final_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_bright = tex_bind("bind-bright", &state.hdr_view);
    let bind_blur_h = tex_bind("bind-blur-h", &state.bright_view);
    let bind_blur_v = tex_bind("bind-blur-v", &state.blur_a);
    let bind_blur_h2 = tex_bind("bind-blur-h2", &state.blur_b);
    let bind_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind-composite"),
        layout: &composite_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&state.hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&state.blur_b),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let bind_bright_uni = uni_bind("bind-bright-uni", &bright_uni);
    let bind_bloom_uni = uni_bind("bind-bloom-uni", &bloom_uni);
    let bind_blur_uh = uni_bind("bind-blur-uh", &blur_uni_h);
    let bind_blur_uv = uni_bind("bind-blur-uv", &blur_uni_v);

    let aspect = w as f32 / h.max(1) as f32;
    let vp = camera.view_proj(aspect);
    let (right, up) = camera.basis();
    state.upload(
        &queue,
        &items,
        &items,
        &Uniform {
            view_proj: vp.to_cols_array_2d(),
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            params: [point_size, 1.0, exposure, 0.0],
        },
    );
    let bw = (w / 4).max(1) as f32;
    let bh = (h / 4).max(1) as f32;
    queue.write_buffer(
        &bright_uni,
        0,
        bytemuck::bytes_of(&[1.0 / bw, 1.0 / bh, 1.0 / exposure.max(1e-3), threshold]),
    );
    queue.write_buffer(
        &bloom_uni,
        0,
        bytemuck::bytes_of(&[bloom, 0.0f32, 0.0, 0.0]),
    );
    queue.write_buffer(
        &blur_uni_h,
        0,
        bytemuck::bytes_of(&[1.0f32, 0.0, 1.0 / bw, 1.0 / bh]),
    );
    queue.write_buffer(
        &blur_uni_v,
        0,
        bytemuck::bytes_of(&[0.0f32, 1.0, 1.0 / bw, 1.0 / bh]),
    );

    let mut enc =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("p") });
    let post = |enc: &mut wgpu::CommandEncoder,
                label: &str,
                target: &wgpu::TextureView,
                pipeline: &wgpu::RenderPipeline,
                binds: &[&wgpu::BindGroup]| {
        let clear = if label == "hdr" {
            wgpu::Color {
                r: render::SPACE_BG[0] as f64,
                g: render::SPACE_BG[1] as f64,
                b: render::SPACE_BG[2] as f64,
                a: 1.0,
            }
        } else {
            wgpu::Color::BLACK
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        for (k, b) in binds.iter().enumerate() {
            pass.set_bind_group(k as u32, Some(*b), &[]);
        }
        if label == "hdr" {
            pass.set_bind_group(0, &state.uniform_bind, &[]);
            pass.set_bind_group(1, &state.storage_bind, &[]);
            pass.draw(0..4, 0..n as u32);
        } else {
            pass.draw(0..3, 0..1);
        }
    };
    post(&mut enc, "hdr", &state.hdr_view, &billboard, &[]);
    post(
        &mut enc,
        "bright",
        &state.bright_view,
        &bright,
        &[&bind_bright, &bind_bright_uni],
    );
    for k in 0..pairs {
        let hb = if k == 0 { &bind_blur_h } else { &bind_blur_h2 };
        post(
            &mut enc,
            "blur-h",
            &state.blur_a,
            &blur,
            &[hb, &bind_blur_uh],
        );
        post(
            &mut enc,
            "blur-v",
            &state.blur_b,
            &blur,
            &[&bind_blur_v, &bind_blur_uv],
        );
    }
    post(
        &mut enc,
        "tonemap",
        &final_view,
        &tonemap,
        &[&bind_composite, &bind_bloom_uni],
    );

    let copy_tex = |enc: &mut wgpu::CommandEncoder,
                    tex: &wgpu::Texture,
                    tw: u32,
                    th: u32,
                    bpp: u32,
                    label: &str| {
        let rb = tw * bpp;
        let rp = rb.div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (rp * th) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(rp),
                    rows_per_image: Some(th),
                },
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );
        (buf, rp)
    };
    let (readback, pad) = copy_tex(&mut enc, &final_tex, w, h, 4, "readback");
    let bw = (w / 4).max(1);
    let bh = (h / 4).max(1);
    let bright_rb = if dump_stages {
        Some(copy_tex(
            &mut enc,
            &state.bright_tex,
            bw,
            bh,
            8,
            "bright-rb",
        ))
    } else {
        None
    };
    let blur_rb = if dump_stages {
        Some(copy_tex(&mut enc, &state.blur_b_tex, bw, bh, 8, "blur-rb"))
    } else {
        None
    };
    let t_gpu = std::time::Instant::now();
    let idx = queue.submit([enc.finish()]);
    let map_one = |buf: &wgpu::Buffer| {
        let (tx2, rx2) = std::sync::mpsc::channel();
        buf.map_async(wgpu::MapMode::Read, .., move |res| {
            let _ = tx2.send(res);
        });
        rx2
    };
    let rx0 = map_one(&readback);
    let rx1 = bright_rb.as_ref().map(|(b, _)| map_one(b));
    let rx2 = blur_rb.as_ref().map(|(b, _)| map_one(b));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    rx0.recv().unwrap().unwrap();
    println!(
        "render submit+poll: {:.1} ms",
        t_gpu.elapsed().as_secs_f64() * 1000.0
    );
    let _ = idx;
    let read_mapped = |buf: &wgpu::Buffer| buf.get_mapped_range(..).expect("map").to_vec();
    let data = read_mapped(&readback);
    readback.unmap();
    if let Some((buf, rp)) = bright_rb {
        rx1.unwrap().recv().unwrap().unwrap();
        let d = read_mapped(&buf);
        buf.unmap();
        let mut hdr = Vec::new();
        hdr.extend_from_slice(&bw.to_le_bytes());
        hdr.extend_from_slice(&bh.to_le_bytes());
        hdr.extend_from_slice(&rp.to_le_bytes());
        std::fs::write(format!("{out}-{mode}.bright-f16r"), hdr).unwrap();
        std::fs::write(format!("{out}-{mode}.bright-f16"), d).unwrap();
        println!("saved bright dump {bw}x{bh}");
    }
    if let Some((buf, rp)) = blur_rb {
        rx2.unwrap().recv().unwrap().unwrap();
        let d = read_mapped(&buf);
        buf.unmap();
        std::fs::write(format!("{out}-{mode}.blur-f16"), d).unwrap();
        println!("saved blur dump rp={rp}");
    }
    let path = format!("{out}-{mode}.ppm");
    save_ppm(&path, w, h, &data, pad as usize);
    let px: Vec<u8> = data
        .chunks(pad as usize)
        .flat_map(|row| row[..w as usize * 4].to_vec())
        .collect();
    let lum: Vec<f32> = px.chunks_exact(4).map(|c| c[0] as f32).collect();
    let max = lum.iter().cloned().fold(0.0f32, f32::max);
    let mean = lum.iter().sum::<f32>() / lum.len() as f32;
    let lit = lum.iter().filter(|v| **v > 8.0).count();
    println!("saved {path} max={max:.0} mean={mean:.2} lit={lit}");
}
