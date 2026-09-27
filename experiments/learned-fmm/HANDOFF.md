# HANDOFF — learned-FMM experiment (paused mid-ablation, 2026-09-28)

Read this first in a new session. Everything below is measured, not guessed.
Companions: `README.md` (idea + verdicts), `LIMITATIONS.md` (open risks).

## 1. What this is (30 seconds)

Tree-structured far-field surrogate for FMMS: same octree + same
well-separatedness rule as real FMM, but every analytic operator (P2M / M2M /
M2L / L2L / L2P) replaced by a learned MLP. Hybrid by design: exact P2P near
field, network predicts only the smooth far field. FMMS stays pristine —
this crate (`experiments/learned-fmm`, detached `[workspace]`) depends on
`fmm` as a library and never patches it. Rust + burn, no Python anywhere.

## 2. Exact state when paused

- Commit stack (master): `781521e` dataset → `602c574` burn spike →
  `ec97631` graph extractor → `90c3bef` tree model → `308510f` ablation
  verdicts. The rounds refactor + minibatch trainer + deadlock guard are
  committed with this doc.
- **Killed mid-flight:** 1-vs-2-round ablation under the new minibatch
  harness (user stopped for handoff). r1 logged 1 epoch (train rel 0.79 —
  minibatch starts slower per-epoch than per-sample SGD: 56 steps vs 900,
  expected); r2 was still in epoch 0. NEITHER LEG COMPLETED — the
  1-vs-2 question is still open. Relaunch commands in §6.
- Last completed results: data ablation 250→0.38 / 500→0.29 / 1000→0.30
  (test blended; data saturates ~500 for the 40k model); capacity leg
  160k params → test 0.28 (weak response → architectural suspect, hence
  the rounds ablation).

## 3. Repo map (experiment dir only; root FMMS untouched)

- `src/lib.rs` — configs (`generate`: plummer/cube/clumps/mixed, deterministic
  `Rng`), unit-box normalize, Shoemake `random_rotate`, `split_near_far`
  (leaf-θ near/far), `SnapshotData` binary I/O (`LFMMD001`, f32),
  `extract_nodes` (octant-indexed children + leaf pids), `build_graph`
  (same-level directed lists via full `mac::accept`), `walk_coverage`
  (cross-check vs walk's recorded pairs). 12 tests.
- `src/model.rs` — `TreeModel`: 5 MLPs (leaf/up/msg/down/decode) + `msg2`/
  `down2` for round 2; `TreeConfig { emb, hidden, rounds }`; level-batched +
  `select`/`gather` forward. 2 tests (determinism incl. rounds=2; learning
  regression 1.0→~0.3, margin checked over 5 runs).
- `src/bin/dataset.rs` — corpus generator. Manifest now has
  `id,config,n,seed,p,eps,n_leaf,rotate,rms_*,far_frac,min_r,file`.
- `src/bin/graph.rs` — topology stats; `--walk` compares lists vs the
  walk's own M2L pairs + prints far_frac joint row.
- `src/bin/flat.rs` — flat-MLP memorize spike (backend validation, keep).
- `src/bin/train.rs` — trainer. Regenerates everything from manifest seeds
  (row-0 dump cross-check ~2.5e-8); loss = relMSE + logcosh with per-component
  logging; `--test-frac` (tail rows held out); `--rounds`, `--batch` (default
  16), `--threads` (default 1 — see deadlock); `--probe` diagnostics.
  NO checkpoint saving yet (runs are disposable; add burn Record when curves
  justify keeping weights).

## 4. Corpora and logs (all under /tmp = EPHEMERAL, regenerate on loss)

- `/tmp/opencode/learned-fmm-1k` (31MB): 1000 snaps, N 128–1024, seed 1000.
- `/tmp/opencode/learned-fmm-250` (7.7M, seed 2000), `-500` (16M, seed 3000).
- `/tmp/opencode/learned-fmm/` pilot 64 + every train log (`train-*.log`).
- Regen: `./target/release/dataset --out DIR --snapshots K --n-min 128
  --n-max 1024 --p 8 --eps 1e-5 [--seed S]` (~1s per 1k snaps at N≤1k).
- Key numbers transcribed here so logs can die: §2 + README verdicts.

## 5. Commands (run from `experiments/learned-fmm`)

```sh
cargo test --release --lib            # 14 tests, fast (burn cached)
cargo build --release --all-targets   # must show ZERO warning lines
./target/release/train --data DIR --epochs 30 --emb 32 --hidden 64 \
  --rounds R --batch 16 --lr 1e-2 2>&1 | tee /tmp/opencode/learned-fmm/NAME.log
```
Log convention (user-mandated): EVERY long run gets `| tee <stable .log path>`,
NEVER `| tail` (it swallows output until exit — burned us twice). Tail via
`tail -f` on the log path.

## 6. Open threads, in order

1. **Relaunch the killed ablation** (r1 + r2 commands as in §5 with
   `--rounds 1` / `--rounds 2`, logs `train-1k-mb-r1/r2.log`; run both at
   once — separate single-threaded processes, 2 cores, ~6GB total RSS).
   Verdict table: r2 clearly < 0.29 → multi-hop confirmed, scale rounds;
   r2 ≈ r1 → topology hypothesis fails, look at features/encodings next.
2. Whichever way (1) goes, the next fork is recorded in README/LIMITATIONS.
3. Eval harness (held-out error vs `direct` + `Fmm` p=3,5,8 + flat MLP;
   KDK energy-drift rollout reusing `energy_gate.rs` warm-start pattern).
4. Persistence (burn Record), weight-init seeding (currently unseeded —
   runs not bitwise reproducible), train/test split hygiene.
5. 20k corpus stays ON HOLD until a model re-opens the data gap.

## 7. DO NOT RETRY (all measured, all reverted, see LIMITATIONS)

- `burn-ndarray/multi-threads`: parallelizes only batched 3D+ matmuls;
  all our Linears are 2D (serial by construction). Zero effect.
- OpenBLAS: 2.7× CPU for zero wall gain (serial-OpenBLAS flat too) plus an
  `LD_LIBRARY_PATH` wart. Bottleneck is serial tiny-op overhead, not kernels.
- `--threads > 1`: burn 0.21 Autodiff DEADLOCKS on concurrent `backward`
  (7+ min frozen, all threads futex_wait, 31MB RSS, zero progress).
  Data-parallel across *processes* is fine; across threads is not.
- Raw MSE, softened labels, 20k corpus at current sizes, GPU at N≤1k
  (launch-bound — would underwhelm like threads did).

## 8. Gotchas (each one bitten already)

- `pkill -f <pattern>` matches your own shell when the pattern is in its
  cmdline (killed my own shell once). Always use the bracket trick:
  `pkill -f "[t]arget/release/train ..."`.
- `tail -100 test` rows are the LARGEST N (N ramps with row id) — test loss
  sits *below* train throughout. Quirk of the split, not a bug.
- Blended loss = relMSE + logcosh; old logs (1k 30ep run) lack the rel split.
  Compare blended-to-blended across those.
- lr scaling: per-sample SGD used 1e-3; minibatch-16 needs ~1e-2 (probed
  3e-3/1e-2 on pilot). Retune if batch changes again.
- Step cost: ~12ms N≤512 pilot, ~2.3s N=4k edge-heavy (194k edges); epoch
  ~45s (r1) / ~95s (r2) on 1k corpus. RSS ~3GB single-threaded on N≤1k —
  box has 15GB with chrome hogging ~11GB: keep thread/process counts low.
- `user≈real` in `time` output = single-threaded (expected now).
- Machine: Ryzen 7 7840HS (16 threads), RTX 4060 Laptop 8GB (unused),
  `/opt/cuda` + vulkan present, rustc 1.97.1, crates.io reachable (one
  transient DNS flake seen; retry works).
- Repo rules: zero `cargo build` warnings (bar); FMMS crates off-limits;
  matching `AGENTS.md` idioms (terse, no comments except safety-critical).

## 9. burn 0.21 API notes (all verified against vendored sources)

- Backend `Autodiff<NdArray>`, device `Default::default()` (Cpu).
- `LinearConfig::new(in, out).init(&device)`; `Gelu::new().forward(x)`;
  `#[derive(Module)]` structs hold only Modules (dims live in `TreeConfig`).
- `Tensor::from_data(TensorData::new(vec, [r, c]), &device)`;
  `select(0, idx_int_tensor)` row-gather; `narrow(0, s, len)` views;
  `sum_dim(0)` keeps rank (`[k,D]`→`[1,D]`); `Tensor::cat(vec, dim)`;
  `mul_scalar`; `cosh().log()`; `.into_scalar()` (numel 1);
  `.into_data().into_vec()`; index tensors are `Tensor<B,1,Int>` from
  `Vec<i64>`.
- `AdamConfig::new().init()` → `optim.step(lr, model, grads)`;
  `GradientsParams::from_grads(grads, &model)` skips params with no grads
  (proven: rounds=1 trains fine with msg2/down2 present-but-unused).
- Minibatch accumulation: `GradientsAccumulator::<TreeModel<BT>>::new()`,
  `.accumulate(&model, grads)`, `.grads()`; scale loss by `1/B` BEFORE
  backward for mean-gradient semantics.
- `&TreeModel` is `Sync` (shared across `thread::scope` compiles) — but see
  deadlock above: sharing is legal, concurrent backward hangs.
