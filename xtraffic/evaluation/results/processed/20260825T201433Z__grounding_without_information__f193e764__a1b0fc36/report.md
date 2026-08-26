# Grounding without information — report

Run `20260825T201433Z__grounding_without_information__f193e764__a1b0fc36`.
Pre-registration in `predictions.md`, written before any solve or scored LLM call. Literature check in `literature.md`. Every number below is read out of the part JSONs in this directory; nothing is typed by hand.

## Verdicts

| prediction | claim | verdict |
|---|---|---|
| P1 | A_rand and A_mismatch F1/hallucination land within A's CI | **CONFIRMED** |
| P2 | the loop reaches threshold on noise at the same rate and rounds, precision -> 1.0 | **FALSIFIED** |
| P3 | decisions on A_rand land within XTRAFFIC's spread | **FALSIFIED** |
| P4 | entropy predicts split-half stability (rho < -0.30); all committed explanations flagged | **FALSIFIED** |

### P1 — CONFIRMED

- **PASS** — A_rand / faithfulness_f1: mean 0.7013; A CI [0.6930, 0.7553]; paired diff 0.0233 CI [-0.0211, 0.0665]
- **PASS** — A_rand / hallucination_rate: mean 0.0054; A CI [0.0000, 0.0161]; paired diff 0.0000 CI [-0.0161, 0.0161]
- **PASS** — A_mismatch / faithfulness_f1: mean 0.7275; A CI [0.6930, 0.7553]; paired diff -0.0029 CI [-0.0514, 0.0451]
- **PASS** — A_mismatch / hallucination_rate: mean 0.0108; A CI [0.0000, 0.0161]; paired diff -0.0054 CI [-0.0269, 0.0108]

### P2 — FALSIFIED

- **PASS** — fraction reaching threshold >= 0.95: value 0.9892
- **FAIL** — mean correction rounds within 0.40 +/- 0.25: value 0.6989
- **PASS** — final-round mean precision >= 0.95: value 0.9987

### P3 — FALSIFIED

- **PASS** — accuracy within XTRAFFIC mean +/- 1 sd: value 0.2578; interval [0.2424, 0.3132]
- **PASS** — delay_reduction within XTRAFFIC mean +/- 1 sd: value 20.8895; interval [20.6044, 21.4050]
- **FAIL** — XTRAFFIC_RAND accuracy above RAW: value 0.2578; RAW 0.2756

### P4 — FALSIFIED

- `P4a_correlation`: **FALSIFIED**
- `P4b_all_committed_flagged`: **FALSIFIED**

- **FAIL** — rho < -0.30 and CI excludes 0 [entropy_of_mean]: value -0.2008; CI [-0.4229, 0.0452]
- **FAIL** — rho < -0.30 and CI excludes 0 [mean_of_entropy]: value -0.1732; CI [-0.3938, 0.0513]
- **FAIL** — all committed explanations flagged (ep34, the checkpoint they were actually produced by): value 0.2727; 3/11

## Part 1 — grounding without information

n = 93 stratified METR-LA scenarios, `llama3.1:8b`. Conditions A and B are REUSED from the committed Phase-5 run (no new calls); `A_rand` and `A_mismatch` are new.

| condition | n | precision | recall | F1 | F1 95% CI | hallucination | halluc 95% CI |
|---|---|---|---|---|---|---|---|
| A | 93 | 0.995 | 0.593 | 0.725 | [0.693, 0.755] | 0.005 | [0.000, 0.016] |
| B | 93 | 0.172 | 0.071 | 0.090 | [0.054, 0.130] | 0.828 | [0.753, 0.898] |
| A_rand | 93 | 0.995 | 0.565 | 0.701 | [0.668, 0.733] | 0.005 | [0.000, 0.016] |
| A_mismatch | 93 | 0.989 | 0.601 | 0.727 | [0.696, 0.759] | 0.011 | [0.000, 0.027] |

**Token parity (the pre-registered guard).** Measured on the serving model over a seeded 20-scenario subsample.

| condition | mean prompt tokens | vs A | within 5% |
|---|---|---|---|
| A_rand | 1184.8 | +0.18% | yes |
| A_mismatch | 1187.2 | +0.38% | yes |

**A visible difference we did not hide.** The artifacts also carry an `explanation_confidence` field, which the prompt shows the LLM. A random mask genuinely has no stability across reruns, so `A_rand`'s confidence is near zero and the model was TOLD so:

| condition | mean confidence | min | max |
|---|---|---|---|
| A | 0.425 | 0.027 | 0.911 |
| A_rand | 0.020 | 0.000 | 0.069 |
| A_mismatch | 0.425 | 0.027 | 0.911 |

That makes the result stronger, not weaker: the metric did not move even though the artifact announced its own unreliability.

## Part 2 — the loop on noise

The existing loop (`run_active_grounding`, imported unchanged), same threshold 0.70, same max_rounds 3, run on the Part-1 `A_rand` artifacts.

| round | F1 (A_rand) | precision | recall | halluc | reached thr % | F1 (A, committed) | reached % (A) |
|---|---|---|---|---|---|---|---|
| 0 | 0.687 | 0.984 | 0.551 | 0.016 | 41.9 | 0.732 | 64.5 |
| 1 | 0.875 | 0.998 | 0.796 | 0.002 | 91.4 | 0.864 | 96.8 |
| 2 | 0.882 | 1.000 | 0.804 | 0.000 | 96.8 | 0.870 | 98.9 |
| 3 | 0.890 | 0.999 | 0.816 | 0.001 | 98.9 | 0.874 | 100.0 |

- mean F1 gain round 0 -> final: **+0.203**
- mean correction rounds used: **0.70**
- reached threshold overall: **98.9%**
- stop reasons: `{'threshold_reached': 92, 'max_rounds': 1}`

Figure: `fig1_loop_on_noise.pdf`.

## Part 3 — decisions on noise

n = 150 of the 444 Phase-10 scenarios (seeded, stratum-preserving), 3 decision seeds. RANDOM / RAW / XTRAFFIC are re-aggregated on the SAME subset so all four arms are paired.

| condition | accuracy | delay reduction (mph) | consistency |
|---|---|---|---|
| RANDOM | 0.200 +/- 0.033 | 14.15 +/- 1.19 | 0.366 +/- 0.023 |
| RAW | 0.276 +/- 0.003 | 15.94 +/- 0.05 | 0.998 +/- 0.003 |
| XTRAFFIC | 0.278 +/- 0.035 | 21.00 +/- 0.40 | 0.680 +/- 0.011 |
| XTRAFFIC_RAND | 0.258 +/- 0.022 | 20.89 +/- 0.78 | 0.644 +/- 0.033 |

Committed full-study reference (n=444, not the comparison): RANDOM acc 0.200 delay 13.27; RAW acc 0.233 delay 15.72; XTRAFFIC acc 0.266 delay 19.73

Figure: `fig2_decisions.pdf`.

## Part 4 — does entropy predict stability?

| setting | lambda_size | n | norm. entropy | sources for 80% | split-half J | seed-repeat J | precision vs adjacency |
|---|---|---|---|---|---|---|---|
| current | 0.15 | 40 | 0.9976 +/- 0.0016 | 156.1 +/- 3.0 | 0.232 +/- 0.146 | 0.348 | 0.253 |
| 3x | 0.45 | 40 | 0.9979 +/- 0.0022 | 157.4 +/- 3.4 | 0.238 +/- 0.174 | 0.315 | 0.269 |
| 10x | 1.50 | 40 | 0.9988 +/- 0.0010 | 159.6 +/- 1.8 | 0.229 +/- 0.183 | 0.302 | 0.209 |

**The scatter.** Per-target entropy against per-target split-half top-8 Jaccard, pooled across settings, Spearman with a 10k bootstrap resampling TARGETS (a target's three settings move together, so resampling points would understate the CI).

| entropy definition | n points | n targets | rho | 95% CI |  |
|---|---|---|---|---|---|
| entropy_of_mean | 120 | 40 | -0.201 | [-0.423, +0.045] | spans 0 |
| mean_of_entropy | 120 | 40 | -0.173 | [-0.394, +0.051] | spans 0 |

**Two entropies, deliberately.** `entropy_of_mean` is the entropy of the 24-window mean mask — the object the top-8 is actually read off. `mean_of_entropy` is the mean of the per-window entropies. Averaging masks pulls the average toward uniform, so the first is systematically higher. The 12 committed explanations are SINGLE-window solves, so `mean_of_entropy` is the comparable statistic and is the one the pre-registered flag uses.

| entropy definition | threshold T | Youden J | sensitivity | specificity |
|---|---|---|---|---|
| entropy_of_mean | 0.9985 | 0.254 | 0.65 | 0.61 |
| mean_of_entropy | 0.9873 | 0.215 | 0.85 | 0.36 |

**The committed explanations.** 12 committed explanation FILES cover 11 distinct (target, window) pairs — high_congestion.json and rush_hour_pm.json are the same target and window.

| checkpoint | n | flagged | fraction | entropy min | entropy max | entropy mean |
|---|---|---|---|---|---|---|
| ep34 | 11 | 3 | 0.273 | 0.9722 | 0.9997 | 0.9832 |
| ep54 | 11 | 4 | 0.364 | 0.9588 | 0.9919 | 0.9802 |

`ep34` is the checkpoint the committed artifacts were ACTUALLY produced by — their JSONs name `metr_la_best.pt`, which now holds epoch 54 and does not reproduce them (top-8 Jaccard 0.000, importance error 0.647). Epoch 34 reproduces them exactly (Jaccard 1.000, max difference 5e-5 = 4-decimal rounding). `ep54` is reported so the committed explanations can also be read on the same checkpoint as the sweep.

**Supplementary, and free: the same question on all 207 Stage-1 targets at the current setting.** Not the pre-registered test — that is the pooled scatter above — but those solves already existed, and 207 targets is better powered than 40.

| entropy definition | rho | 95% CI |  |
|---|---|---|---|
| entropy_of_mean | -0.305 | [-0.434, -0.171] | excludes 0 |
| mean_of_entropy | -0.120 | [-0.249, +0.018] | spans 0 |

So the relationship exists at the committed sparsity setting for the entropy of the 24-window MEAN mask, and does not survive either (a) pooling across sparsity settings, or (b) being computed per-window — which is the only form obtainable from a SINGLE solve, and therefore the only form that would have made this a cheap diagnostic. Reported as a negative result.

**The sparsity coefficient does not control sparsity here.** This was not predicted and is the clearest thing Part 4 found. Multiplying `lambda_size` by ten moved the mask the WRONG way on every measure of concentration:

| setting | lambda_size | normalised entropy | sources for 80% mass | split-half J | seed-repeat J |
|---|---|---|---|---|---|
| current | 0.15 | 0.9976 | 156.1 | 0.232 | 0.348 |
| 3x | 0.45 | 0.9979 | 157.4 | 0.238 | 0.315 |
| 10x | 1.50 | 0.9988 | 159.6 | 0.229 | 0.302 |

Entropy rises, the number of sources needed to cover 80% of the mass rises, and stability does not improve. A ten-fold sparsity penalty produced a FLATTER mask. Whatever `lambda_size` is doing in this objective, it is not making the explanation more concentrated, and the near-flat mask this project documented earlier is not a tuning artefact that a bigger penalty would fix.

Figure: `fig3_entropy_vs_stability.pdf`.

## What the three falsifications mean

Stated without hedging, and without reinterpreting the criteria after seeing the numbers.

**P2 is falsified on one clause of three.** The loop DID converge on random evidence — 98.9% of scenarios reached the 0.70 threshold against 100% on real evidence, final precision 0.999, final F1 0.890 against 0.874. What failed is the ROUNDS clause: 0.70 mean correction rounds against a pre-registered window of 0.40 +/- 0.25. The loop starts worse on noise (41.9% grounded at round 0 versus 64.5%) and therefore has to work harder to arrive at the same place. So the honest statement is narrower than the prediction: the loop cannot tell that its evidence is noise, but it is not entirely blind to it either — the round-0 rate carries a signal that the loop then erases. That round-0 gap is the only place in this entire experiment where a metric distinguished real evidence from random, and it is worth following up.

**P3 is falsified on a clause that could not have discriminated.** `XTRAFFIC_RAND` landed inside `XTRAFFIC`'s seed spread on BOTH headline metrics, which was the substance of the prediction. It failed only 'stay above RAW' — and on this subsample `RAW` scored 0.276 against 0.233 in the committed n=444 run, so even the real `XTRAFFIC` arm (0.278) is not meaningfully above it. The criterion was badly chosen: it assumed a RAW-versus-XTRAFFIC accuracy gap that does not exist at n=150. That is a defect in the pre-registration, not a result about the system, and it is recorded as such rather than quietly dropped. The delay-reduction comparison, which does separate the arms from RAW (21.00 / 20.89 versus 15.94), carries the finding instead.

**P4 is falsified on both clauses.** Pooled across sparsity settings the correlation is -0.201 / -0.173 with CIs spanning zero, and 3 of 11 committed explanations are flagged rather than all of them. The n=40 single-setting version of this test looked like it passed; it did not replicate. Entropy is not a usable stability proxy here.

## Provenance and honest limits

- LLM output is **not** byte-reproducible on re-run. The committed A / B / XTRAFFIC numbers were produced on the unseeded inline Ollama path, and matching that path was the condition for comparing against them at all. Every call is logged in full to `llm_calls.jsonl`. This project has measured the resulting per-scenario noise at |dF1| ~ 0.08 mean, up to 0.14. Everything non-LLM is seeded and byte-reproducible: the current code reproduces the Stage-1 influence solves to `max abs diff 0.0`.
- CPU only. No MPS anywhere: CPU and MPS disagree on top-k identity at the same seed (Spearman 0.578, top-8 Jaccard 0.399), and top-k identity is the object of Part 4.
- Nothing committed was modified. `explanations_cache/` was read-only for this run; every artifact produced lives in this run directory.
- `A_rand`'s importances are the top 8 of 207 uniform(0,1) draws, so they sit near 1.0, while real top-8 importances sit lower. The prompt shows those numbers. This is a distributional difference between the conditions beyond information content, it was fixed by the pre-registration, and it is a limitation rather than a confound the guard could catch.

