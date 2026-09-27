use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::Backend, Tensor, TensorData},
};
use learned_fmm::{generate, normalize_to_unit_box, split_near_far, ConfigTag};
use fmm::tree::Octree;

type B = Autodiff<NdArray>;

#[derive(Module, Debug)]
struct FlatMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> FlatMlp<B> {
    fn new(device: &B::Device, hidden: usize) -> Self {
        Self {
            l1: LinearConfig::new(4, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, 3).init(device),
            relu: Relu::new(),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.l1.forward(x));
        let x = self.relu.forward(self.l2.forward(x));
        self.l3.forward(x)
    }
}

fn arg_val(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = arg_val(&args, "--n", "256").parse().unwrap();
    let steps: usize = arg_val(&args, "--steps", "2000").parse().unwrap();
    let lr: f64 = arg_val(&args, "--lr", "1e-3").parse().unwrap();
    let hidden: usize = arg_val(&args, "--hidden", "64").parse().unwrap();
    let device = Default::default();
    let mut parts = generate(ConfigTag::Cube, n, 7);
    normalize_to_unit_box(&mut parts);
    let tree = Octree::build(&parts, 16);
    let (_, far, stats) = split_near_far(&parts, &tree);
    let scale = stats.rms_far.max(1e-12);
    let mut xs = Vec::with_capacity(n * 4);
    let mut ys = Vec::with_capacity(n * 3);
    for i in 0..n {
        xs.extend_from_slice(&[
            parts[i].pos[0] as f32,
            parts[i].pos[1] as f32,
            parts[i].pos[2] as f32,
            (parts[i].mass * n as f64) as f32,
        ]);
        ys.extend_from_slice(&[
            (far[i][0] / scale) as f32,
            (far[i][1] / scale) as f32,
            (far[i][2] / scale) as f32,
        ]);
    }
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::new(xs, [n, 4]), &device);
    let y: Tensor<B, 2> = Tensor::from_data(TensorData::new(ys, [n, 3]), &device);
    let norm = y.clone().powf_scalar(2.0).mean().into_scalar().max(1e-12);
    let mut model = FlatMlp::new(&device, hidden);
    let mut optim = AdamConfig::new().init();
    for s in 0..steps {
        let out = model.forward(x.clone());
        let diff = out - y.clone();
        let loss = diff.powf_scalar(2.0).mean() / norm;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
        if s % 200 == 0 || s + 1 == steps {
            let v = loss.into_scalar();
            println!("step {s}: rel_mse={v:.6}");
        }
    }
}
