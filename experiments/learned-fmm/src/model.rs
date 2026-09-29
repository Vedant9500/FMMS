use burn::{
    module::Module,
    nn::{Gelu, Linear, LinearConfig},
    tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor, TensorData},
};
use fmm::models::Particle;

use crate::InteractionGraph;

#[derive(Clone, Copy)]
pub struct TreeConfig {
    pub emb: usize,
    pub hidden: usize,
    pub rounds: usize,
    pub pool: u8,
    pub h_trunk: usize,
    pub h_head: usize,
    pub mono: u8,
    pub geom: u8,
    pub equiv: u8,
    pub up_mode: u8,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    act: Gelu,
}

impl<B: Backend> Mlp<B> {
    pub fn new(d_in: usize, hidden: usize, d_out: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(d_in, hidden).init(device),
            l2: LinearConfig::new(hidden, d_out).init(device),
            act: Gelu::new(),
        }
    }
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l2.forward(self.act.forward(self.l1.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct TreeModel<B: Backend> {
    leaf: Mlp<B>,
    up: Mlp<B>,
    up_enc: Mlp<B>,
    up_agg: Mlp<B>,
    msg: Mlp<B>,
    down: Mlp<B>,
    msg2: Mlp<B>,
    down2: Mlp<B>,
    msg3: Mlp<B>,
    down3: Mlp<B>,
    dec: Mlp<B>,
}

impl<B: Backend> TreeModel<B> {
    pub fn new(cfg: &TreeConfig, device: &B::Device) -> Self {
        let d = cfg.emb;
        let ht = if cfg.h_trunk > 0 { cfg.h_trunk } else { cfg.hidden };
        let hh = if cfg.h_head > 0 { cfg.h_head } else { cfg.hidden };
        Self {
            leaf: Mlp::new(6, ht, d, device),
            up: Mlp::new(8 * (d + 3), ht, d, device),
            up_enc: Mlp::new(d + 3, ht, ht, device),
            up_agg: Mlp::new(ht, ht, d, device),
            msg: Mlp::new(2 * d + 4, ht, d, device),
            down: Mlp::new(3 * d, hh, d, device),
            msg2: Mlp::new(2 * d + 4, ht, d, device),
            down2: Mlp::new(3 * d, hh, d, device),
            msg3: Mlp::new(2 * d + 4, ht, d, device),
            down3: Mlp::new(3 * d, hh, d, device),
            dec: Mlp::new(d + 7, hh, 3, device),
        }
    }

    fn row(data: Vec<f32>, d: usize, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(data, [1, d]), device)
    }

    pub fn forward(
        &self,
        cfg: &TreeConfig,
        parts: &[Particle],
        g: &InteractionGraph,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let emb = cfg.emb;
        let rounds = cfg.rounds;
        let n = parts.len();
        let nc = g.nodes.len();
        let zero = Self::row(vec![0.0; emb], emb, device);
        let mut up: Vec<Tensor<B, 2>> = vec![zero.clone(); nc];
        let mut leaf_row = vec![0usize; nc];
        let mut leaves = Vec::new();
        for (i, nd) in g.nodes.iter().enumerate() {
            if nd.is_leaf {
                leaf_row[i] = leaves.len();
                leaves.push(i);
            }
        }
        if !leaves.is_empty() {
            let mut flat = Vec::with_capacity(leaves.len() * 6);
            for &i in &leaves {
                let nd = &g.nodes[i];
                let lf = nd.leaf.as_ref().unwrap();
                let inv = 1.0 / (nd.rho + 1e-12);
                flat.extend_from_slice(&[
                    (lf.count as f32 / 16.0),
                    ((lf.com[0] - nd.center[0]) * inv) as f32,
                    ((lf.com[1] - nd.center[1]) * inv) as f32,
                    ((lf.com[2] - nd.center[2]) * inv) as f32,
                    (lf.total_mass) as f32,
                    (lf.spread * inv) as f32,
                ]);
            }
            let out: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(flat, [leaves.len(), 6]), device);
            let out = self.leaf.forward(out);
            for (&i, row) in leaves.iter().zip(out.split(1, 0)) {
                up[i] = row;
            }
        }
        let max_level = g.nodes.iter().map(|nd| nd.level).max().unwrap_or(0);
        let mut parent_of = vec![0usize; nc];
        for (q, nd) in g.nodes.iter().enumerate() {
            for c in nd.children.iter().flatten() {
                parent_of[*c] = q;
            }
        }
        for l in (0..=max_level).rev() {
            let cells: Vec<usize> = (0..nc)
                .filter(|&i| g.nodes[i].level == l && !g.nodes[i].is_leaf)
                .collect();
            if cells.is_empty() {
                continue;
            }
            if cfg.up_mode == 1 {
                let htu = if cfg.h_trunk > 0 { cfg.h_trunk } else { cfg.hidden };
                let mut enc_rows: Vec<Tensor<B, 2>> = Vec::new();
                let mut bounds = Vec::with_capacity(cells.len() + 1);
                let mut nrows = 0;
                for &i in &cells {
                    bounds.push(nrows);
                    for c in g.nodes[i].children.iter().flatten() {
                        let cc = g.nodes[*c].center;
                        let pc = g.nodes[i].center;
                        let rel = vec![
                            (cc[0] - pc[0]) as f32,
                            (cc[1] - pc[1]) as f32,
                            (cc[2] - pc[2]) as f32,
                        ];
                        let geom = Self::row(rel, 3, device);
                        enc_rows.push(Tensor::cat(vec![up[*c].clone(), geom], 1));
                        nrows += 1;
                    }
                }
                bounds.push(nrows);
                let enc = self.up_enc.forward(Tensor::cat(enc_rows, 0));
                let mut sums = Vec::with_capacity(cells.len());
                for b in 0..cells.len() {
                    let (s, e) = (bounds[b], bounds[b + 1]);
                    if e > s {
                        sums.push(enc.clone().narrow(0, s, e - s).sum_dim(0));
                    } else {
                        sums.push(Self::row(vec![0.0; htu], htu, device));
                    }
                }
                let agg = self.up_agg.forward(Tensor::cat(sums, 0));
                for (&i, row) in cells.iter().zip(agg.split(1, 0)) {
                    up[i] = row;
                }
            } else {
                let mut inputs = Vec::with_capacity(cells.len());
                for &i in &cells {
                    let mut slot_rows: Vec<Tensor<B, 2>> = Vec::with_capacity(8);
                    for o in 0..8 {
                        match g.nodes[i].children[o] {
                            Some(c) => {
                                let cc = g.nodes[c].center;
                                let pc = g.nodes[i].center;
                                let rel = vec![
                                    (cc[0] - pc[0]) as f32,
                                    (cc[1] - pc[1]) as f32,
                                    (cc[2] - pc[2]) as f32,
                                ];
                                let geom = Self::row(rel, 3, device);
                                slot_rows.push(Tensor::cat(vec![up[c].clone(), geom], 1));
                            }
                            None => slot_rows.push(Self::row(vec![0.0; emb + 3], emb + 3, device)),
                        }
                    }
                    inputs.push(Tensor::cat(slot_rows, 1));
                }
                let out = self.up.forward(Tensor::cat(inputs, 0));
                for (&i, row) in cells.iter().zip(out.split(1, 0)) {
                    up[i] = row;
                }
            }
        }
        let inter = self.interact(emb, &up, &self.msg, cfg.pool, cfg.geom, g, device);
        let local = self.propagate(emb, &up, &inter, &self.down, g, &parent_of, max_level, device);
        let local = if rounds >= 2 {
            let inter2 = self.interact(emb, &local, &self.msg2, cfg.pool, cfg.geom, g, device);
            self.propagate(emb, &up, &inter2, &self.down2, g, &parent_of, max_level, device)
        } else {
            local
        };
        let local = if rounds >= 3 {
            let inter3 = self.interact(emb, &local, &self.msg3, cfg.pool, cfg.geom, g, device);
            self.propagate(emb, &up, &inter3, &self.down3, g, &parent_of, max_level, device)
        } else {
            local
        };
        let mut leaf_of = vec![0i64; n];
        for (i, nd) in g.nodes.iter().enumerate() {
            if nd.is_leaf {
                for &pid in &nd.pids {
                    leaf_of[pid] = i as i64;
                }
            }
        }
        let local_stack = Tensor::cat(local, 0);
        let li: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(leaf_of.clone(), [n]), device);
        let gathered = local_stack.select(0, li);
        let mut mono = vec![[0.0f64; 3]; nc];
        if cfg.mono == 1 || cfg.equiv == 1 {
            for (b, srcs) in g.incoming.iter().enumerate() {
                let cb = g.nodes[b].center;
                let mut acc = [0.0f64; 3];
                for &a in srcs {
                    let ca = g.nodes[a].center;
                    let r = [
                        cb[0] - ca[0],
                        cb[1] - ca[1],
                        cb[2] - ca[2],
                    ];
                    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                    let s = g.nodes[a].mass / (r2 * r2.sqrt() + 1e-30);
                    acc[0] += s * r[0];
                    acc[1] += s * r[1];
                    acc[2] += s * r[2];
                }
                mono[b] = acc;
            }
            let mut m = 0.0f64;
            for v in &mono {
                m += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            }
            let inv = 1.0 / (m / nc.max(1) as f64 + 1e-30);
            for v in mono.iter_mut() {
                v[0] *= inv;
                v[1] *= inv;
                v[2] *= inv;
            }
        }
        let mut feats = Vec::with_capacity(n * 7);
        for (i, p) in parts.iter().enumerate() {
            let nd = &g.nodes[leaf_of[i] as usize];
            let inv = 1.0 / (nd.rho + 1e-12);
            let mo = mono[leaf_of[i] as usize];
            feats.extend_from_slice(&[
                ((p.pos[0] - nd.center[0]) * inv) as f32,
                ((p.pos[1] - nd.center[1]) * inv) as f32,
                ((p.pos[2] - nd.center[2]) * inv) as f32,
                (p.mass * n as f64) as f32,
                mo[0] as f32,
                mo[1] as f32,
                mo[2] as f32,
            ]);
        }
        let fm: Tensor<B, 2> = Tensor::from_data(TensorData::new(feats, [n, 7]), device);
        let inp = Tensor::cat(vec![gathered, fm], 1);
        let gates = self.dec.forward(inp);
        if cfg.equiv == 1 {
            let mut us = Vec::with_capacity(n * 3);
            let mut ms = Vec::with_capacity(n * 3);
            let mut ws = Vec::with_capacity(n * 3);
            for (i, p) in parts.iter().enumerate() {
                let li = leaf_of[i] as usize;
                let nd = &g.nodes[li];
                let o = [
                    p.pos[0] - nd.center[0],
                    p.pos[1] - nd.center[1],
                    p.pos[2] - nd.center[2],
                ];
                let mo = mono[li];
                let no = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() + 1e-30;
                let nm = (mo[0] * mo[0] + mo[1] * mo[1] + mo[2] * mo[2]).sqrt() + 1e-30;
                let u = [o[0] / no, o[1] / no, o[2] / no];
                let m = [mo[0] / nm, mo[1] / nm, mo[2] / nm];
                let w = [
                    u[1] * m[2] - u[2] * m[1],
                    u[2] * m[0] - u[0] * m[2],
                    u[0] * m[1] - u[1] * m[0],
                ];
                us.extend_from_slice(&[u[0] as f32, u[1] as f32, u[2] as f32]);
                ms.extend_from_slice(&[m[0] as f32, m[1] as f32, m[2] as f32]);
                ws.extend_from_slice(&[w[0] as f32, w[1] as f32, w[2] as f32]);
            }
            let u: Tensor<B, 2> = Tensor::from_data(TensorData::new(us, [n, 3]), device);
            let m: Tensor<B, 2> = Tensor::from_data(TensorData::new(ms, [n, 3]), device);
            let w: Tensor<B, 2> = Tensor::from_data(TensorData::new(ws, [n, 3]), device);
            let g1 = gates.clone().narrow(1, 0, 1);
            let g2 = gates.clone().narrow(1, 1, 1);
            let g3 = gates.narrow(1, 2, 1);
            g1.mul(u) + g2.mul(m) + g3.mul(w)
        } else {
            gates
        }
    }

    fn interact(
        &self,
        emb: usize,
        states: &[Tensor<B, 2>],
        msg: &Mlp<B>,
        pool: u8,
        geom_mode: u8,
        g: &InteractionGraph,
        device: &B::Device,
    ) -> Vec<Tensor<B, 2>> {
        let nc = g.nodes.len();
        let zero = Self::row(vec![0.0; emb], emb, device);
        let stack = Tensor::cat(states.to_vec(), 0);
        let mut src_idx = Vec::new();
        let mut dst_idx = Vec::new();
        let mut bounds = Vec::with_capacity(nc + 1);
        let mut ecount = 0;
        for (b, srcs) in g.incoming.iter().enumerate() {
            bounds.push(ecount);
            for &a in srcs {
                src_idx.push(a as i64);
                dst_idx.push(b as i64);
                ecount += 1;
            }
        }
        bounds.push(ecount);
        let mut inter: Vec<Tensor<B, 2>> = vec![zero; nc];
        if ecount > 0 {
            let si: Tensor<B, 1, Int> =
                Tensor::from_data(TensorData::new(src_idx, [ecount]), device);
            let di: Tensor<B, 1, Int> =
                Tensor::from_data(TensorData::new(dst_idx.clone(), [ecount]), device);
            let su = stack.clone().select(0, si);
            let du = stack.clone().select(0, di);
            let mut geom = Vec::with_capacity(ecount * 4);
            let mut wraw = Vec::with_capacity(ecount);
            for (b, srcs) in g.incoming.iter().enumerate() {
                let cb = g.nodes[b].center;
                let rhob = g.nodes[b].rho;
                for &a in srcs {
                    let ca = g.nodes[a].center;
                    let rhoa = g.nodes[a].rho;
                    let dx = [
                        (cb[0] - ca[0]) * 0.5,
                        (cb[1] - ca[1]) * 0.5,
                        (cb[2] - ca[2]) * 0.5,
                    ];
                    let r = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
                    let sep = (r / (rhoa + rhob + 1e-12) + 1.0).log10();
                    if geom_mode == 1 {
                        let s = rhoa + rhob + 1e-30;
                        geom.extend_from_slice(&[
                            (r * 0.5) as f32,
                            sep as f32,
                            (rhoa / s) as f32,
                            (rhob / s) as f32,
                        ]);
                    } else {
                        geom.extend_from_slice(&[
                            dx[0] as f32,
                            dx[1] as f32,
                            dx[2] as f32,
                            sep as f32,
                        ]);
                    }
                    wraw.push((g.nodes[a].mass / (r * r + 1e-12)) as f32);
                }
            }
            let gm: Tensor<B, 2> = Tensor::from_data(TensorData::new(geom, [ecount, 4]), device);
            let out = msg.forward(Tensor::cat(vec![su, du, gm], 1));
            let out = if pool == 1 {
                let mut wv = vec![1.0f32; ecount];
                for b in 0..nc {
                    let (s, e) = (bounds[b], bounds[b + 1]);
                    if e > s {
                        let m: f32 = wraw[s..e].iter().sum::<f32>() / (e - s) as f32;
                        let inv = 1.0 / m.max(1e-30);
                        for e in s..e {
                            wv[e] = (wraw[e] * inv).clamp(0.0, 1e4);
                        }
                    }
                }
                let wm: Tensor<B, 2> =
                    Tensor::from_data(TensorData::new(wv, [ecount, 1]), device);
                out.mul(wm)
            } else {
                out
            };
            let didx: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(dst_idx, [ecount, 1]), device)
                    .repeat(&[1, emb]);
            let mut scl = Vec::with_capacity(nc);
            for b in 0..nc {
                let (s, e) = (bounds[b], bounds[b + 1]);
                scl.push(if e > s {
                    1.0 / ((e - s) as f32).sqrt()
                } else {
                    0.0
                });
            }
            let sc: Tensor<B, 2> = Tensor::from_data(TensorData::new(scl, [nc, 1]), device);
            let base: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(vec![0.0f32; nc * emb], [nc, emb]), device);
            let summed = base
                .scatter(0, didx, out, IndexingUpdateOp::Add)
                .mul(sc);
            inter = summed.split(1, 0);
        }
        inter
    }

    fn propagate(
        &self,
        emb: usize,
        up: &[Tensor<B, 2>],
        inter: &[Tensor<B, 2>],
        down_mlp: &Mlp<B>,
        g: &InteractionGraph,
        parent_of: &[usize],
        max_level: u32,
        device: &B::Device,
    ) -> Vec<Tensor<B, 2>> {
        let nc = g.nodes.len();
        let zero = Self::row(vec![0.0; emb], emb, device);
        let mut local: Vec<Tensor<B, 2>> = vec![zero.clone(); nc];
        local[0] = down_mlp
            .forward(Tensor::cat(vec![zero.clone(), up[0].clone(), inter[0].clone()], 1));
        for l in 1..=max_level {
            let cells: Vec<usize> = (0..nc)
                .filter(|&i| g.nodes[i].level == l)
                .collect();
            if cells.is_empty() {
                continue;
            }
            let mut inputs = Vec::with_capacity(cells.len());
            for &i in &cells {
                let q = parent_of[i];
                inputs.push(Tensor::cat(
                    vec![local[q].clone(), up[i].clone(), inter[i].clone()],
                    1,
                ));
            }
            let out = down_mlp.forward(Tensor::cat(inputs, 0));
            for (&i, row) in cells.iter().zip(out.split(1, 0)) {
                local[i] = row;
            }
        }
        local
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_graph, generate, normalize_to_unit_box, ConfigTag};
    use burn::backend::{Autodiff, NdArray};
    use fmm::tree::Octree;

    type TB = Autodiff<NdArray>;

    #[test]
    fn forward_finite_and_deterministic() {        let device = Default::default();
        let cfg = TreeConfig { emb: 8, hidden: 16, rounds: 2, pool: 0, h_trunk: 0, h_head: 0, mono: 0, geom: 0, equiv: 0, up_mode: 0 };
        let mut parts = generate(ConfigTag::Cube, 64, 31);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let g = build_graph(&parts, &tree, 2, 1e-3);
        let model = TreeModel::<TB>::new(&cfg, &device);
        let a = model.forward(&cfg, &parts, &g, &device);
        let b = model.forward(&cfg, &parts, &g, &device);
        let da: Vec<f32> = a.into_data().into_vec().unwrap();
        let db: Vec<f32> = b.into_data().into_vec().unwrap();
        assert_eq!(da.len(), 64 * 3);
        assert_eq!(da, db);
        for &x in &da {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn weighted_pool_forward_finite() {
        let device = Default::default();
        let cfg = TreeConfig { emb: 8, hidden: 16, rounds: 2, pool: 1, h_trunk: 0, h_head: 0, mono: 0, geom: 0, equiv: 0, up_mode: 0 };
        let mut parts = generate(ConfigTag::Cube, 64, 31);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let g = build_graph(&parts, &tree, 2, 1e-3);
        let model = TreeModel::<TB>::new(&cfg, &device);
        let a = model.forward(&cfg, &parts, &g, &device);
        let da: Vec<f32> = a.into_data().into_vec().unwrap();
        assert_eq!(da.len(), 64 * 3);
        for &x in &da {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn mono_forward_finite() {
        let device = Default::default();
        let cfg = TreeConfig { emb: 8, hidden: 16, rounds: 1, pool: 0, h_trunk: 0, h_head: 0, mono: 1, geom: 0, equiv: 0, up_mode: 0 };
        let mut parts = generate(ConfigTag::Plummer, 64, 37);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let g = build_graph(&parts, &tree, 2, 1e-3);
        let model = TreeModel::<TB>::new(&cfg, &device);
        let a = model.forward(&cfg, &parts, &g, &device);
        let da: Vec<f32> = a.into_data().into_vec().unwrap();
        assert_eq!(da.len(), 64 * 3);
        for &x in &da {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn equiv_forward_finite() {
        let device = Default::default();
        let cfg = TreeConfig { emb: 8, hidden: 16, rounds: 1, pool: 0, h_trunk: 0, h_head: 0, mono: 0, geom: 1, equiv: 1, up_mode: 0 };
        let mut parts = generate(ConfigTag::Mixed, 64, 41);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let g = build_graph(&parts, &tree, 2, 1e-3);
        let model = TreeModel::<TB>::new(&cfg, &device);
        let a = model.forward(&cfg, &parts, &g, &device);
        let da: Vec<f32> = a.into_data().into_vec().unwrap();
        assert_eq!(da.len(), 64 * 3);
        for &x in &da {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn scatter_add_matches_manual_sum() {
        use burn::tensor::TensorData;
        let device = Default::default();
        let vals: Tensor<TB, 2> = Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
            &device,
        );
        let idx: Tensor<TB, 2, Int> =
            Tensor::from_data(TensorData::new(vec![0i64, 1, 0, 1, 0, 1], [3, 2]), &device);
        let base: Tensor<TB, 2> =
            Tensor::from_data(TensorData::new(vec![0.0f32; 4], [2, 2]), &device);
        let out = base.scatter(0, idx, vals, IndexingUpdateOp::Add);
        let v: Vec<f32> = out.into_data().into_vec().unwrap();
        assert_eq!(v, vec![9.0, 0.0, 0.0, 12.0]);
    }

    #[test]
    fn deepsets_up_forward_finite() {
        let device = Default::default();
        let cfg = TreeConfig { emb: 8, hidden: 16, rounds: 1, pool: 0, h_trunk: 0, h_head: 0, mono: 0, geom: 0, equiv: 0, up_mode: 1 };
        let mut parts = generate(ConfigTag::Mixed, 96, 43);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let g = build_graph(&parts, &tree, 2, 1e-3);
        let model = TreeModel::<TB>::new(&cfg, &device);
        let a = model.forward(&cfg, &parts, &g, &device);
        let da: Vec<f32> = a.into_data().into_vec().unwrap();
        assert_eq!(da.len(), 96 * 3);
        for &x in &da {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn training_decreases_loss_on_one_sample() {
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};
        use burn::tensor::{Tensor, TensorData};
        let device = Default::default();
        let cfg = TreeConfig { emb: 16, hidden: 32, rounds: 1, pool: 0, h_trunk: 0, h_head: 0, mono: 0, geom: 0, equiv: 0, up_mode: 0 };
        let mut parts = generate(ConfigTag::Cube, 32, 33);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let (_, far, stats) = crate::split_near_far(&parts, &tree);
        let scale = stats.rms_far.max(1e-12);
        let mut target = Vec::with_capacity(32 * 3);
        for i in 0..32 {
            target.extend_from_slice(&[
                (far[i][0] / scale) as f32,
                (far[i][1] / scale) as f32,
                (far[i][2] / scale) as f32,
            ]);
        }
        let g = build_graph(&parts, &tree, 2, 1e-3);
        let mut model = TreeModel::<TB>::new(&cfg, &device);
        let mut optim = AdamConfig::new().init();
        let tgt: Tensor<TB, 2> = Tensor::from_data(TensorData::new(target, [32, 3]), &device);
        let mut first = 0.0;
        let mut last = 0.0;
        for s in 0..400 {
            let pred = model.forward(&cfg, &parts, &g, &device);
            let diff = pred - tgt.clone();
            let loss = diff.clone().powf_scalar(2.0).mean()
                / tgt.clone().powf_scalar(2.0).mean().clamp_min(1e-12);
            let v = loss.clone().into_scalar();
            assert!(v.is_finite(), "non-finite loss at step {s}");
            if s == 0 {
                first = v;
            }
            last = v;
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(1e-3, model, grads);
        }
        assert!(last < 0.5 * first, "first={first} last={last}");
    }
}
