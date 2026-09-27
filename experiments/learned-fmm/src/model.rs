use burn::{
    module::Module,
    nn::{Gelu, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use fmm::models::Particle;

use crate::InteractionGraph;

pub struct TreeConfig {
    pub emb: usize,
    pub hidden: usize,
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
    msg: Mlp<B>,
    down: Mlp<B>,
    dec: Mlp<B>,
}

impl<B: Backend> TreeModel<B> {
    pub fn new(cfg: &TreeConfig, device: &B::Device) -> Self {
        let d = cfg.emb;
        let h = cfg.hidden;
        Self {
            leaf: Mlp::new(6, h, d, device),
            up: Mlp::new(8 * (d + 3), h, d, device),
            msg: Mlp::new(2 * d + 4, h, d, device),
            down: Mlp::new(3 * d, h, d, device),
            dec: Mlp::new(d + 4, h, 3, device),
        }
    }

    fn row(data: Vec<f32>, d: usize, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(data, [1, d]), device)
    }

    pub fn forward(
        &self,
        emb: usize,
        parts: &[Particle],
        g: &InteractionGraph,
        device: &B::Device,
    ) -> Tensor<B, 2> {
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
            for (r, &i) in leaves.iter().enumerate() {
                up[i] = out.clone().narrow(0, r, 1);
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
            for (r, &i) in cells.iter().enumerate() {
                up[i] = out.clone().narrow(0, r, 1);
            }
        }
        let up_stack = Tensor::cat(up.clone(), 0);
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
        let mut inter: Vec<Tensor<B, 2>> = vec![zero.clone(); nc];
        if ecount > 0 {
            let si: Tensor<B, 1, Int> =
                Tensor::from_data(TensorData::new(src_idx, [ecount]), device);
            let di: Tensor<B, 1, Int> =
                Tensor::from_data(TensorData::new(dst_idx, [ecount]), device);
            let su = up_stack.clone().select(0, si);
            let du = up_stack.clone().select(0, di);
            let mut geom = Vec::with_capacity(ecount * 4);
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
                    geom.extend_from_slice(&[
                        dx[0] as f32,
                        dx[1] as f32,
                        dx[2] as f32,
                        sep as f32,
                    ]);
                }
            }
            let gm: Tensor<B, 2> = Tensor::from_data(TensorData::new(geom, [ecount, 4]), device);
            let msg = self.msg.forward(Tensor::cat(vec![su, du, gm], 1));
            for b in 0..nc {
                let (s, e) = (bounds[b], bounds[b + 1]);
                if e > s {
                    let k = (e - s) as f32;
                    inter[b] = msg.clone().narrow(0, s, e - s).sum_dim(0).mul_scalar(1.0 / k.sqrt());
                }
            }
        }
        let mut local: Vec<Tensor<B, 2>> = vec![zero.clone(); nc];
        local[0] = self
            .down
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
            let out = self.down.forward(Tensor::cat(inputs, 0));
            for (r, &i) in cells.iter().enumerate() {
                local[i] = out.clone().narrow(0, r, 1);
            }
        }
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
        let mut feats = Vec::with_capacity(n * 4);
        for (i, p) in parts.iter().enumerate() {
            let nd = &g.nodes[leaf_of[i] as usize];
            let inv = 1.0 / (nd.rho + 1e-12);
            feats.extend_from_slice(&[
                ((p.pos[0] - nd.center[0]) * inv) as f32,
                ((p.pos[1] - nd.center[1]) * inv) as f32,
                ((p.pos[2] - nd.center[2]) * inv) as f32,
                (p.mass * n as f64) as f32,
            ]);
        }
        let fm: Tensor<B, 2> = Tensor::from_data(TensorData::new(feats, [n, 4]), device);
        let inp = Tensor::cat(vec![gathered, fm], 1);
        self.dec.forward(inp)
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
        let cfg = TreeConfig { emb: 8, hidden: 16 };
        let mut parts = generate(ConfigTag::Cube, 64, 31);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, 16);
        let g = build_graph(&parts, &tree, 2, 1e-3);
        let model = TreeModel::<TB>::new(&cfg, &device);
        let a = model.forward(cfg.emb, &parts, &g, &device);
        let b = model.forward(cfg.emb, &parts, &g, &device);
        let da: Vec<f32> = a.into_data().into_vec().unwrap();
        let db: Vec<f32> = b.into_data().into_vec().unwrap();
        assert_eq!(da.len(), 64 * 3);
        assert_eq!(da, db);
        for &x in &da {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn training_decreases_loss_on_one_sample() {
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};
        use burn::tensor::{Tensor, TensorData};
        let device = Default::default();
        let cfg = TreeConfig { emb: 16, hidden: 32 };
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
            let pred = model.forward(cfg.emb, &parts, &g, &device);
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
