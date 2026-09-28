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
