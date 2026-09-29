# Ablation log — learned-FMM experiment

Running record of every training leg + verdict. Numbers are blended loss
(relMSE + logcosh) unless noted; test = tail 10% holdout (largest-N rows,
so test sits below train — split quirk, not a bug). Full train logs live
under `/tmp/opencode/learned-fmm/` (EPHEMERAL); this file is the durable
record. Updated after every leg.

## Data ablation — 40k model (emb32/hidden64, rounds 1, 30 epochs)

| Date | Corpus | Test blended | Verdict |
|---|---|---|---|
| 2026-09-27 | 250 snaps (seed 2000) | 0.38 | real gain 250→500 |
| 2026-09-27 | 500 snaps (seed 3000) | 0.29 | — |
| 2026-09-27 | 1000 snaps (seed 1000, N128–1024) | 0.30 | 500→1000 is noise (±0.03) → data saturates ~500 for 40k model; 20k corpus ON HOLD |

## Capacity leg — 1k corpus, 30 epochs

| Date | Config | Train → Test blended (rel) | Verdict |
|---|---|---|---|
| 2026-09-27 | emb64/hidden128 ≈160k, ~85s/epoch | 0.322→0.308 train, 0.30→0.28 test (rel 0.248) | 4× params for ~5% → constraint is ARCHITECTURAL, not data/size. Suspect: single message-passing round → H3 |

## H3 — rounds 1 vs 2 (minibatch-16, lr 1e-2, emb32/hidden64, 1k corpus seed 1000, 30 epochs)

Command: `./target/release/train --data /tmp/opencode/learned-fmm-1k --epochs 30 --emb 32 --hidden 64 --rounds R --batch 16 --lr 1e-2`

| Leg | Date | Params | Epoch cost | Final train (rel) | Final test (rel) | Mean-last-5 test | Best test |
|---|---|---|---|---|---|---|---|
| r1 | 2026-09-28 | 39.9k | ~51s | 0.34878 (0.3035) | 0.28762 (0.2505) | 0.30589 | 0.28762 |
| r2 | 2026-09-28 | 54.7k (+msg2/down2) | ~96s | 0.27662 (0.2404) | 0.23796 (0.2074) | 0.27314 | 0.23796 |

Δ test = 0.050 final / 0.033 mean-last-5 (kill threshold |Δ|<0.015).
Train gap matches test gap (rules out under-training).
Per-param efficiency vs capacity leg: +37% params for 0.05 vs +300% for 0.02 → effect is architectural, not capacity.

**Verdict: multi-hop CONFIRMED. Scale rounds next (r3 with msg3/down3).**
Caveats: single run each, unseeded init — repeat before publishing.

| Leg | Date | Params | Epoch cost | Final train (rel) | Final test (rel) | Mean-last-5 test | Best test |
|---|---|---|---|---|---|---|---|
| r3 | 2026-09-28 | 69.5k (+msg3/down3) | ~140s | 0.27841 (0.2421) | 0.29070 (0.2531) | 0.27771 | 0.25593 (ep27) |

r3 run died silently at epoch 29/30 (1 epoch short, likely OOM — deeper graph;
verdict uses epochs 0–28, plenty). r3 ≈ r2 (best 0.256 vs 0.238, mean-last-5
0.278 vs 0.273 — within noise, predicted +0.01–0.02 gain did NOT materialize).

**Verdict: routing saturates at 2 rounds. Do NOT scale to r4; pivot to
H1 (attention) / H2 (decoder) / H5 (monopole) — the missing gain is in
message content and readout, not more hops.**

## H1 — 1/r²-weighted pooling (rounds 2, emb32/hidden64, batch-16, lr 1e-2)

Command: `./target/release/train --data /tmp/opencode/learned-fmm-1k --epochs 30 --emb 32 --hidden 64 --rounds 2 --pool 1 --batch 16 --lr 1e-2`
(zero new params: fixed `w=(M/r²)/mean(M/r²)` per dst replaces uniform `1/√k`)

| Leg | Final train (rel) | Final test (rel) | Mean-last-5 test | Best test |
|---|---|---|---|---|
| r2 uniform | 0.27662 (0.2404) | 0.23796 (0.2074) | 0.27314 | 0.23796 |
| H1 weighted | 0.27519 (0.2393) | 0.22294 (0.1942) | 0.24542 | 0.22294 (ep29) |

Δ best = 0.015, Δ mean-last-5 = 0.028 (kill threshold <0.01).
Train endpoints match (0.275 vs 0.277 — same fit, better generalization).

**Verdict: H1 HOLDS (no washout — dense fan-in was lossy). Effect is modest
on best (+6%) but consistent on smoothed (+10%) at zero param cost.
Needs a repeat run (unseeded init) before claiming; keep pool=1 as default
for all downstream legs (H2/H5/H4/H7/H6 run weighted).**

## H2 — fat decoder at fixed budget (rounds 2, pool 1, trunk-32/head-128)

Command: `... --rounds 2 --pool 1 --h-trunk 32 --h-head 128 ...`
(56.1k params, +2.5% vs H1 baseline — placement test, not size)

| Leg | Final train (rel) | Final test (rel) | Mean-last-5 test | Best test |
|---|---|---|---|---|
| H1 uniform-h64 | 0.27519 (0.2393) | 0.22294 (0.1942) | 0.24542 | 0.22294 |
| H2 fat-head | 0.28098 (0.2442) | 0.24250 (0.2112) | 0.24804 | 0.22078 (ep23) |

Δ best = 0.002, Δ smoothed = −0.003 (kill threshold <0.01).

**Verdict: H2 KILLED. Capacity placement is irrelevant at fixed budget —
the decoder is not the load-bearing stage. The bottleneck is message
content / geometry encoding, not where the params sit. Revert to uniform
hidden (flags stay available); do not buy a bigger decoder.**

## H5 — analytic monopole features (rounds 2, pool 1, mono 1)

Command: `... --rounds 2 --pool 1 --mono 1 ...`
(per-leaf monopole ΣM·R/R³, self-standardized, +3 decoder inputs;
+192 params, +0.35% — negligible vs effect)

| Leg | Final train (rel) | Final test (rel) | Mean-last-5 test | Best test |
|---|---|---|---|---|
| H1 no-mono | 0.27519 (0.2393) | 0.22294 (0.1942) | 0.24542 | 0.22294 |
| H5 mono | 0.26602 (0.2311) | 0.20797 (0.1813) | 0.21506 | 0.20797 (ep29) |

Δ best = 0.015, Δ smoothed = 0.030 (kill threshold <0.01).
Train fits better too (0.266 vs 0.275) — more signal, not just
generalization. Still descending at ep29.

**Verdict: H5 HOLDS. The free embedding was wasting capacity re-learning
Newton's leading order; handing it the monopole frees the trunk for
dipole+ corrections. Keep mono=1 as default for H4/H7/H6.**

## H4-stage-1 — invariant-only messages, negative control (rounds 2, pool 1, geom 1)

Command: `... --rounds 2 --pool 1 --geom 1 ...`
(msg geometry `[r, sep, ρa/ρb, ρb/ρa]`, direction dropped; direct vector head)

| Leg | Final train | Final test | Best test |
|---|---|---|---|
| H1 raw-Cartesian | 0.275 | 0.223 | 0.223 |
| H4-s1 invariant | 0.532 | 0.464 | 0.464 |

2× worse, plateaus ~0.46–0.53 from epoch 13 on (never learns).

**Verdict: control behaves as predicted — direction is load-bearing, a
scalar trunk cannot emit the vector field. Non-vacuity proven: the
equivariant head (stage 2) is worth testing, since raw direction carries
real signal the model currently memorizes per-rotation.**
