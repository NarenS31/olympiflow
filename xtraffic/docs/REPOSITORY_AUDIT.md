# XTraffic — Repository Audit (Phase 0)

**Date:** 2026-08-21
**Auditor:** Claude Opus 5, acting as research-engineering assistant, under the
role boundaries stated in the Phase-0 brief.
**Method:** read-only inspection. **No file was modified, no experiment was run,
no result was regenerated.** Every number quoted below was read off an artifact
already on disk, or off source code. Where a number appears only in prose
(`CLAUDE.md`, `LAB_NOTEBOOK.md`) and not in a machine-readable artifact, it is
marked as such.

**What this document is not:** it is not a reproduction. I did not rerun a single
experiment, so I cannot certify that any committed number would reappear. Several
findings below are specifically about why some of them would *not*.

---

## 0. Orientation

| | |
|---|---|
| Project root | `/Users/narensara/Desktop/OlympiFlow/xtraffic` |
| Git root | `/Users/narensara/Desktop/OlympiFlow` (one level up; repo also holds the older OlympiFlow React/FastAPI app) |
| Branch | `main`, clean working tree (`git status` empty for `xtraffic/`) |
| HEAD | `a5ba31d` — *"Phase 11 corrected: resolver fixed, PEMS-BAY/Chicago region tables, outlier resolved, Chicago dropped from faithfulness table"* |
| Other branch | `xtraffic-phase-1` (local + remote) |
| Tracked files under `xtraffic/` | 2,497 |
| Python | 60 files, 18,275 lines |
| Language constraint | Python 3.9 (`typing.Optional`, no `X \| Y`) |

The repository is **substantial, unusually well-documented, and internally
honest**. `CLAUDE.md` runs to ~700 lines of dated status log that repeatedly
flags its own weaknesses, negative results, and superseded numbers. That is rare
and it is an asset. Most of the problems below are *scientific-design* problems
and *reproducibility-infrastructure* problems, not sloppiness.

---

## 1. Repository map

### 1.1 Data layer — `data/`

| Path | Lines | Role |
|---|---|---|
| `data/pipelines/metr_la.py` | 45 | METR-LA download → windowed tensors. Thin wrapper over `_csv_common`. |
| `data/pipelines/pems_bay.py` | 39 | Same, PEMS-BAY (325 sensors). |
| `data/pipelines/chicago.py` | 182 | Chicago Traffic Tracker via **live Socrata API**. Segment-nodes. |
| `data/pipelines/power_grid.py` | 946 | IEEE 14-bus. Real topology + real AC power flow (`pandapower.runpp`), **synthetic demand**. |
| `data/pipelines/_csv_common.py` | 111 | Shared CSV → tensor path (METR-LA / PEMS-BAY). |
| `data/pipelines/_modality_common.py` | 161 | Sidecar modality writer (`mod_<name>.npz`), shared windowing/scaling. |
| `data/pipelines/weather.py` | 123 | Open-Meteo ERA5 archive → per-node weather sidecar. |
| `data/pipelines/events.py` | 106 | Curated 2012 LA events → proximity-decayed sidecar. |
| `data/pipelines/transit.py` | 105 | GTFS static → nearby-stop-count sidecar. |
| `data/pipelines/sanity_check.py` | 74 | Shape/NaN/split-order assertions across datasets. |
| `data/pipelines/second_domain.py` | 88 | Phase-19 exploration stub. |
| `data/pipelines/DIFFERENCES_POWER_GRID.md` | — | Cross-domain honesty note. |

**On disk (gitignored):** `data/raw/` — metr_la 100M, pems_bay 82M, chicago 13M,
power_grid 4.2M. `data/processed/` — all four datasets present with
`train/val/test.npz`, `adjacency.npy`, `scaler.json`, `node_meta.json`,
`stats.json`. METR-LA additionally carries three sidecars (`mod_weather`,
`mod_events`, `mod_transit`); power_grid carries `mod_power`.

### 1.2 Model layer — `models/gnn/`

| Path | Lines | Role |
|---|---|---|
| `stgnn.py` | 301 | `XTrafficSTGNN`. Graph WaveNet base + 3 modifications: learned semantic co-movement edges (`A = σ(α)·A_phys + (1−σ(α))·A_sem`), multi-scale dilations `[1,2,4,8]`, per-modality gated fusion with `None`-tolerant gates. Ablation switches `use_semantic`, `use_multiscale`, `sidecar_modalities`. |
| `train.py` | 294 | Adam + clipping, masked MAE, early stop on val MAE, per-epoch CSV. **Only file in the repo that calls a `set_seed()`.** |
| `loaders.py` | 155 | `make_fusion_loaders`, `build_modality_dict`. |
| `baselines.py` | 142 | Historical average, linear regression, persistence. |
| `evaluate.py` | 118 | Checkpoint → JSON metrics row; fusion-aware. |

**Checkpoints** (`models/gnn/checkpoints/`, gitignored, 8 files):
`metr_la_best.pt` (epoch 54, val MAE 2.8747), `metr_la_best_epoch34_ARCHIVE.pt`
(epoch 34, val MAE 2.9027), `metr_la_fusion_best.pt`, `power_grid_best.pt`
(epoch 6, val MAE 0.0017 pu), `pems_bay_zero_shot.pt`, `chicago_zero_shot.pt`,
`chicago_fine_tuned.pt`, `power_grid_smoke.pt`.

Checkpoints store `config`, `scaler`, `n_nodes`, `epoch`, `val_mae`. They do
**not** store git commit, package versions, or a data hash.

### 1.3 Explanation layer — `models/explainer/`

| Path | Lines | Role |
|---|---|---|
| `explain.py` | 356 | Pure-PyTorch GNNExplainer reimplementation (learned node + edge mask logits, preservation + sparsity + entropy loss, 200 Adam steps @ lr 0.01). `ExplanationBuilder` composes the JSON. Propagation path = BFS shortest path; lag = windowed cross-correlation; confidence = mean Jaccard of top-k over K reruns. |
| `schema.py` | 106 | The explanation JSON **contract** + validator. Required-keys check; extra keys permitted (this is how Phase 18's `uncertainty` block attaches additively). |
| `node_names.py` | 267 | Offline lat/lon → region label, per city (`_LA_REGIONS`, `_PEMS_BAY_REGIONS`, `_CHICAGO_REGIONS`) + compass fallback; power-grid bus naming. |
| `shap_explainer.py` | 250 | SHAP `KernelExplainer` over nodes, exposing the same `explain_target()` signature so it drops into `ExplanationBuilder`. |
| `counterfactual.py` | 535 | Minimum-uplift search that flips a congested target to free flow. **Does rerun the frozen GNN** (`predict_target_mph`) at every search step. |
| `uncertain_explainer.py` | 414 | K=10 reruns → per-node top-k frequency → CORE / PERIPHERAL / NOISE tiers. |
| `scenarios.py`, `generate.py` | 198 | Deterministic scenario selection + explanation generation. |

### 1.4 LLM advisory layer — `models/advisor/`

| Path | Lines | Role |
|---|---|---|
| `advisor.py` | 834 | The prompt system. Ollama `/api/generate`, `format=json`, temperature 0.1. Prompt order SYSTEM → CITY CONTEXT → EXPLANATION → TASK. Advisory contract `{reasoning, cited_causes[], recommendations[]}` + `validate_advisory` + 2-retry self-correct on malformed JSON. Prompt variants: `build_prompt` (A), `build_prompt_condition` (A/B/C/C_RICH/D_CONTRA), `build_prompt_counterfactual`, `build_prompt_uncertain`. |
| `active_grounding.py` | 723 | Score → if F1 < 0.7 name the missed top-k nodes → re-prompt, ≤3 rounds. Logs a per-round convergence curve. |
| `knowledge_base.py` | 146 | Keyword + region-tag retrieval over per-city KB JSON, top-k=4. |
| `domains.py` | 175 | Per-domain vocabulary (unit, node noun, stress noun, operator role). Exists because a traffic-worded prompt told the LLM a substation was doing "0.88 mph". |
| `pipeline.py` | 94 | `advise(city, node_id, timestamp)` = GNN → explainer → advisor. |
| `kb/` | — | `la.json` (11 chunks), `chicago.json` (9), `power_grid.json` (12), `README.md` template. **No PEMS-BAY KB** (deliberate — see §5). |

### 1.5 Evaluation layer — `evaluation/` (32 files)

| Path | Lines | Role |
|---|---|---|
| `faithfulness.py` | 590 | **The metric.** `NodeTable` + `resolve_location` (5-rung ladder) + `score_advisory`. Discussed at length in §4. |
| `run_faithfulness_study.py` | 441 | Stratified sampler (5 tod-bands × 3 congestion terciles), conditions A/B/C/C_RICH/D_CONTRA, caching, CSV/JSON/PDF. |
| `bootstrap_ci.py` | 373 | Paired-by-scenario nonparametric bootstrap, 10k iters, 95% CI on per-condition means and on paired differences. |
| `cross_city_faithfulness.py` | 1020 | 3 cities × 2 LLMs master table + `RANDOM_region` / `RANDOM_target_region` chance controls. |
| `power_grid_faithfulness.py` | 759 | Cross-domain A/B + `RANDOM_bus` / `RANDOM_zone` chance controls + paired bootstrap. |
| `active_grounding_power_grid.py` | 805 | Phase-16 loop rerun on the grid, two vocabulary variants + a measured noise floor. |
| `sim_eval.py` | 923 | 444-scenario simulated decision evaluation; 5 actions; model-in-the-loop ground truth. |
| `sparse_regime.py` | 816 | Sensor-density sweep 100→10%, prediction + faithfulness degradation. |
| `shap_comparison.py` | 367 | GNNExplainer vs SHAP vs no-explainer, n=28. |
| `counterfactual_study.py` | 497 | n=20 counterfactual validity + narration faithfulness. |
| `uncertainty_study.py` | 601 | n=20 A-vs-U, hedging analysis. |
| `failure_modes.py` | 499 | Six-category condition-A failure taxonomy over the 7/93 failures. |
| `explainer_metrics.py` | 210 | Fidelity+ / Fidelity− / sparsity / stability vs random baseline. |
| `cross_city.py` | 253 | Zero-shot / fine-tuned / from-scratch weight transfer. |
| `fusion_comparison.py` | 101 | Traffic-only vs fusion checkpoint, side by side. |
| `ablations.py` | 249 | 7 model variants × 3 seeds + pipeline conditions. |
| `validate_resolver.py` | 101 | Resolver gate against hand-labeled sets. |
| `verify_traffic_unchanged.py` | 175 | Byte-identity regression gate on 12 committed explanations × 5 rendered artifacts + 3 prompt constants. |
| `refresh_explanation_names.py` | 124 | Rewrites cached `node_name` strings after a region-table change. |
| `make_paper_artifacts.py` | 560 | Regenerates 7 figures + 5 tables; writes loud `PENDING` placeholders for missing upstream results. |
| `human_study/` (6 files, 1,022) | | Retired FastAPI expert-study toolkit (see §7). |

### 1.6 Configuration — `configs/` (13 YAML)

`data.yaml` (10.6K, all four datasets + sidecars), `advisor.yaml`,
`ablations.yaml`, `counterfactual.yaml`, `cross_city_faith.yaml`,
`human_study.yaml`, `power_grid_faith.yaml`, `sim_eval.yaml`,
`sparse_regime.yaml`, `train_metr_la.yaml`, `train_metr_la_fusion.yaml`,
`train_power_grid.yaml`, `uncertainty.yaml`.

The "no magic numbers in code" rule is genuinely honoured — thresholds,
temperatures, budgets, and seeds live in YAML with explanatory comments.

### 1.7 Orchestration

`run_phases.sh` (5.0K) and `run_queue.sh` (11.4K) — bash sequencers that run
studies one at a time because they share the single local Ollama server. Both
rely on the studies' own resumability (per-decision JSONL + cache).

---

## 2. Dependency map

From `requirements.txt` (all pinned, with rationale comments — good):

```
torch==2.2.2              # last clean cp39 wheel line
torch-geometric==2.5.3    # pinned as a pair with torch
numpy==1.26.4             # <2 required by torch 2.2 ABI
pandas==2.1.4
scipy==1.11.4             # cross-correlation for propagation lag
rapidfuzz==3.6.1          # OPTIONAL — falls back to stdlib difflib
pandapower==2.14.11       # IEEE 14-bus case + AC power flow
shap==0.44.1              # KernelExplainer baseline
pyyaml==6.0.1
requests==2.31.0
matplotlib==3.8.2
fastapi==0.110.0 / uvicorn==0.29.0 / python-multipart==0.0.9   # retired human study
```

**Notes and risks**

1. **`torch-geometric` is declared but never imported.** Phase 3 deliberately
   reimplemented GNNExplainer in pure PyTorch. Harmless, but it misrepresents the
   real dependency surface.
2. **`rapidfuzz` is optional and the fallback is not equivalent.**
   `faithfulness.py:60-74` prefers `rapidfuzz.token_set_ratio` (order-insensitive,
   ignores extra words) and falls back to `difflib.SequenceMatcher` (character
   n-gram ratio). These are *different similarity functions* compared against the
   *same* threshold of 82. A machine without `rapidfuzz` will silently produce a
   different resolver, therefore different faithfulness numbers. The backend is
   recorded in `FUZZ_BACKEND` but is **not written into any result file.**
   → **This is a reproducibility defect, not a cosmetic one.**
3. **No `scikit-learn`, no `statsmodels`.** There is therefore no multiple-comparison
   correction, no mixed-effects model, and no Krippendorff/Cohen agreement
   machinery available. Phases 4 and 11 of the new plan will need additions.
4. **Ollama is an undeclared dependency** with no version pinning. `ollama list`
   on this machine right now: `llama3.1:8b`, `mistral:7b`, `gemma2:9b`,
   `phi3:medium`, `llama3.2:3b`, `llama3.2:latest`, `llama3:latest`,
   `aria_distilled:latest`. **`qwen2.5:7b`, which `configs/ablations.yaml:54`
   still names, is not installed.** Model *digests* (available from
   `/api/tags`) are never recorded in results.
5. **No test dependency** (`pytest` absent) because there are no tests.

---

## 3. Existing experiments

Phase numbering below is the project's own (`CLAUDE.md`), which is not
contiguous — there is no written Phase 14 or 15, and "Phase 15b" is a bundle of
edits to existing files.

| # | Experiment | Entry point | n | Artifact on disk | Status |
|---|---|---|---|---|---|
| 1 | Data pipelines ×4 | `data/pipelines/*.py` | — | `data/processed/*` (gitignored) | Built; Chicago not reproducible (§5.1) |
| 2 | ST-GNN training | `models/gnn/train.py` | — | `metr_la_best.pt` ep54 | Built |
| 2b | Baselines | `models/gnn/baselines.py` | — | `baseline_{historical_average,linear_regression,persistence}_{metr_la,power_grid}.json` | Built |
| 3 | Explainer metrics | `evaluation/explainer_metrics.py` | 90 valid / 100 | `explainer_metrics_metr_la.{json,csv}` | Run |
| 4 | Advisory demo | `models/advisor/demo.py` | 6 | `results/advisories/*.json` | Run — **only place full advisory text is stored** |
| 5 | Faithfulness A/B/C | `run_faithfulness_study.py` | 93 | `faithfulness_summary.json`, `faithfulness_per_scenario.csv` (280 rows) | Run — **superseded, see §5.4** |
| 6 | Fusion comparison | `fusion_comparison.py` | — | `results/fusion/comparison.{csv,json}` | Run — **negative result** |
| 6b | Cross-city transfer | `cross_city.py` | — | `chicago_{zero_shot,fine_tuned}.pt` | Partially run; no transfer table found |
| 7 | Ablation matrix | `ablations.py` | 7 variants × 3 seeds | `paper/tables/table4_ablation.tex` | **PENDING placeholder — never run** |
| 8 | Human study toolkit | `human_study/` | 0 real participants | demo data only | **Retired** |
| 9 | Paper artifacts | `make_paper_artifacts.py` | — | `paper/figures/`, `paper/tables/` | Run |
| 10 | Simulated decision eval | `sim_eval.py` | 444 × 3 seeds × {RAW, XTRAFFIC} = 2,664 (+RANDOM) | `sim_eval_summary.json`, `decisions.jsonl` (2,736 lines) | Run |
| 11 | Cross-city × cross-model | `cross_city_faithfulness.py` | 93 / 85 / 33 × 2 models | `cross_city_faithfulness_master.json` + 6 per-scenario CSVs | Run, then **corrected and rerun** |
| 12 | SHAP comparison | `shap_comparison.py` | 28 | `table_shap_comparison.tex` + summary | Run |
| 13 | Failure taxonomy | `failure_modes.py` | 7 failures of 93 | `failure_modes.json`, 3 regenerated advisories | Run |
| 14 | Sparse-sensor regime | `sparse_regime.py` | 5 densities | `sparse_regime_{summary.json,per_density.csv}` | Run — **not mentioned in the Phase-0 brief** |
| 15b | C_RICH + D_CONTRA + bootstrap CI | `run_faithfulness_study.py --out-tag 15b` | 93 | `faithfulness_summary_15b.json`, `faithfulness_bootstrap_ci.json` | Run |
| 16 | Active grounding loop | `active_grounding.py` | 93 stratified (+5 smoke) | `active_grounding_summary.json` | Run |
| 17 | Counterfactuals | `counterfactual_study.py` | 20 | `counterfactual_summary.json` | Run |
| 18 | Uncertainty tiers | `uncertainty_study.py` | 20 | `uncertainty_summary.json` | Run |
| 19 | Power-grid cross-domain | `power_grid_faithfulness.py` | 20 | `table_power_grid_faith.tex` + summary | Run |
| 20 | Active grounding cross-domain | `active_grounding_power_grid.py` | 20 × 2 variants | `..._summary.json`, `..._traces.json` | Run |

**Total LLM decisions on disk:** ~4,013 JSONL lines + ~1,000 cached decision JSONs.

---

## 4. The faithfulness metric, read closely

This matters more than anything else in the audit, because the new central
hypothesis is about *unsupported claims* and this is the only thing currently
measuring them.

`evaluation/faithfulness.py:481` — `score_advisory(explanation, advisory, table)`:

```
cause_precision    = |cited causes c : resolve(c) ∩ topk ≠ ∅| / |cited causes|
cause_recall       = |topk nodes covered by ∪ resolve(c)| / |topk|
faithfulness_f1    = harmonic mean
hallucination_rate = 1 − cause_precision            (line 540)
quantitative_fidelity = fraction of stated numbers within ±10% of ANY
                        reference quantity in the explanation JSON
```

### 4.1 What it actually measures

**One claim type only: "node X is among the causes."** The unit of analysis is an
entry in the LLM's own `cited_causes[]` array. It is not a claim extracted from
prose — it is a field the model was instructed to fill. Nothing in the pipeline
parses the `reasoning` string for assertions.

Not covered at all: edge existence, edge direction, path existence, temporal /
lag claims, confidence-tier claims, prediction claims, counterfactual claims,
domain-terminology claims. The explanation JSON *contains* `top_edges`,
`propagation_path`, and `propagation_lag_minutes`, and the rendered prompt shows
them to the LLM — but **no metric ever checks whether the LLM described them
correctly.** An advisory can invent an edge direction or a propagation order and
score hallucination 0.000.

### 4.2 Alignment, not correctness

The reference set is the explainer's top-k — i.e. *what the LLM was shown*. The
project states this itself, repeatedly and to its credit (`CLAUDE.md`, Phase 12
and Phase 19 caveats). The sharpest evidence is the project's own Phase-12
diagnostic: GNNExplainer and SHAP top-k sets have **Jaccard 0.022** (n=28) — very
nearly disjoint — yet produce **identical** precision (0.982) and hallucination
(0.018). Two mutually contradictory "explanations" both score as perfectly
faithful. That is a property of the metric, not of the method.

### 4.3 A refusal scores as a maximal hallucination

`faithfulness.py:540`:

```python
hallucination = (1.0 - precision) if n_causes else 1.0
```

If the model declines to answer, or emits an empty `cited_causes` list, it scores
**hallucination 1.000, precision 0.000, F1 0.000** — the same as confidently
inventing four fake causes. The new plan's **Condition N (no-answer control)
cannot be scored by this metric as written.** This is a genuine blocker for
Phase 5, and it also means the current condition-B numbers conflate "fabricated"
with "declined / emitted nothing parseable."

Related: `advisory_error` is a logged column, so parse failures are
distinguishable in the CSV — but they are not distinguished in the headline mean.

### 4.4 Region-level credit is a documented generosity with a measured cost

A citation resolves to the *set* of nodes in a region and "hits" if that set
intersects top-k. The project discovered empirically how much this inflates
things: the Phase-11 investigation found a control citing **only the target's own
region** — zero causal content — scored F1 0.510 on PEMS-BAY, reproducing the
reported outlier cell to three decimals. That control (`RANDOM_target_region`) is
now implemented and is the correct floor. **It exists only in
`cross_city_faithfulness.py` and `power_grid_faithfulness.py`** — the Phase-5,
15b, 12, 13, 16, 17, 18 studies have no chance floor at all.

Measured floors: METR-LA 0.278, PEMS-BAY 0.269, Chicago 0.565. Chicago's margin
over floor is +0.06 and it is correctly excluded from the faithfulness table.

### 4.5 Quantitative fidelity is very generous

`faithfulness.py:558-563`: a stated number is faithful if within ±10% of **any**
reference quantity — target current speed, target predicted speed, all eight
top-node speeds, and the lag. With ~11 reference values spread over a plausible
range, a substantial fraction of arbitrary two-digit numbers will match
something. The in-code comment acknowledges the generosity. This number should
not be quoted as evidence of numerical accuracy.

### 4.6 The resolver is validated, unevenly

`validate_resolver.py` gates against hand-labeled sets:
METR-LA 29/30 (96.7%), PEMS-BAY 32/32, Chicago 32/32, power grid 22/22. Half of
each non-METR-LA set is refusals by design — a good decision, because a resolver
that matches everything would erase condition B.

But: the METR-LA set is 30 cases against a 207-node / ~20-region universe, and it
was hand-labeled by the same person who defined the regions. It gates the ladder,
not the *region table*, which is the thing that actually produced the Phase-11
outlier.

---

## 5. Reproducibility problems

Ordered by severity. Each is a concrete, fixable defect.

### 5.1 BLOCKER — Chicago data cannot be reproduced, ever

`data/pipelines/chicago.py:57`:

```python
params = {"$limit": ds["page_limit"], "$order": "time DESC"}
```

The pipeline requests the **50,000 most recent rows** from a live Socrata
endpoint. Chicago Traffic Tracker is a rolling feed. Running this today returns
2026 data; the committed Chicago results were built from a July-2025 window. There
is no date filter, no snapshot, no checksum. The processed tensors exist on disk
but are gitignored.

**Consequence:** every Chicago number in the repository is unreproducible from
source. (Chicago is already excluded from the faithfulness table for an unrelated
structural reason, but it still carries a prediction-transfer claim.)

**Fix:** add `$where` date bounds to `configs/data.yaml`, and hash-stamp the
downloaded CSV. Then either re-derive (accepting new numbers) or archive the
existing raw pull as the frozen snapshot.

### 5.2 BLOCKER — no LLM seed; every LLM number is one unreplicable draw

`models/advisor/advisor.py:650-657` builds the Ollama payload as:

```python
"options": {"temperature": self.temperature}    # 0.1
```

**No `seed`. No `num_predict`. No `top_p`, `top_k`, `num_ctx`, or
`repeat_penalty`.** Temperature 0.1 is not 0, so decoding is stochastic, and even
at temperature 0 Ollama is not bit-reproducible across model reloads without an
explicit seed.

The project *measured* this itself (Phase 20): re-running the same scenario with
the same prompt and the same model gave **F1 0.857 vs 0.333** on scenario
idx 1912, and a mean round-0 |diff| of 0.078 (std 0.132) across 20 scenarios.

**Consequences:**
- No committed LLM result is reproducible.
- Every study is n=1 draw per (scenario, condition). Single-draw noise of ~0.08
  F1 is the same order as several reported effects (e.g. the Phase-12 A-vs-SHAP
  gap of +0.05, the Phase-20 variant contrast of 0.041 — the latter the project
  correctly reported as a null).
- Token budget is uncontrolled, so the "matched token budget" requirement of the
  new ablation matrix is currently unimplementable.

**Fix:** pass `seed`, `num_predict`, `top_p`, `top_k`, `num_ctx` explicitly from
config; record the resolved model **digest** from `/api/tags`; run k≥5 draws per
cell and treat draw as a nested random factor.

### 5.3 BLOCKER — raw LLM output is not persisted at scale

Inspected every cache format:

| Study | n decisions | What is stored | Raw text? |
|---|---|---|---|
| Phase 4 demo | 6 | full `{prediction, explanation, advisory, context_used, model}` | **Yes** |
| Phase 5/15b | 282 | `cited_locations` + metrics | No `reasoning`, no `recommendations`, no prompt |
| Phase 10 sim_eval | 2,736 | `chosen`, `ground_truth`, `correct`, `delay_reduction` | **No text at all** |
| Phase 11 cross-city | 995 | metrics + `cited_locations` + `resolution_methods` | No |
| Phase 16 active grounding | 93 | curve + `final_advisory` | Final round only |
| Phase 18 uncertainty | 20 | metrics only | No |

So of roughly 4,000 LLM generations, **the full text of about 100 survives.**

**Consequences:**
- Any new metric (atomic-claim decomposition, edge/path/temporal verification,
  refusal detection) **cannot be applied retrospectively.** It requires
  regeneration, which per §5.2 will not reproduce the committed numbers.
- Blinded human annotation (new Phase 4) has almost no material to annotate.
- The project already paid this cost once: diagnosing the Phase-11 outlier
  "required re-running the LLM."

**Fix:** persist prompt, raw response string, parsed object, model digest, and
sampling params for every call, before any further experiment runs.

### 5.4 HIGH — committed paper artifacts are built from superseded numbers

`make_paper_artifacts.py:424` reads
`results/faithfulness/faithfulness_summary.json`. That file holds the
**pre-correction** Phase-5 numbers (A F1 0.7246 / halluc 0.005; B halluc 0.8280).
The corrected values, after the three resolver defects were fixed, are in
`cross_city_faithfulness_master.json`: **A F1 0.7232 / B halluc 0.8387**.

`evaluation/paper/tables/table3_faithfulness.tex` on disk currently reads
`B: no-expl. 0.172 & 0.071 & 0.090 & 0.828` — the superseded row. Neither the
JSON nor the `.tex` carries any superseded marker; the only warning lives in
`CLAUDE.md` prose, which lists 8 artifacts owed a rerun (Phases 5/15b, 12, 13,
14, 16, 18, and Phase 20's METR-LA row).

**Note for the brief:** the "82.8%" figure in the Phase-0 brief is this
superseded number. The defensible corrected figure is **83.9%**. The difference
is small and in the project's favour; the problem is provenance, not magnitude.

### 5.5 HIGH — no experiment provenance capture whatsoever

Grepped the entire codebase for `git rev-parse`, `commit_hash`, `sys.version`,
`platform.` — **zero hits.** No result file records:

git commit · Python version · package versions · OS/device · Ollama model digest
· data snapshot hash · prompt-template hash · wall-clock · resolver fuzzy backend

Explanations record `"model_checkpoint": "metr_la_best.pt"` — a *filename*, not a
hash or epoch. Since `metr_la_best.pt` was overwritten at least twice (3-epoch
placeholder → epoch 34 → epoch 54), it is **impossible to tell from an artifact
which model produced it.** The project hit this exact bug in Phase 17: six cached
explanations predated the checkpoint they claimed and were silently stale;
`build_or_load_explanation` caches on *filename presence only*.

Per `CLAUDE.md`, results from Phases 3/5/10/11/12/13/16/17/18 were produced on
epoch-34 while the live `metr_la_best.pt` is now epoch-54. **Rerunning any of them
today silently uses a different model.**

### 5.6 HIGH — zero automated tests

```
$ find . -name "test_*.py" -o -name "*_test.py" -o -type d -name tests
(nothing)
```

18,275 lines of research code, including a metric with a five-rung resolution
ladder and several documented near-miss correctness bugs, with **no unit tests.**
The two regression gates that exist (`verify_traffic_unchanged.py`,
`validate_resolver.py`) are valuable but are golden-file / accuracy gates, not
tests. `verify_traffic_unchanged.py` demonstrably had a coverage hole: its probe
list contained only bare region names and bare ids, so it **passed throughout the
entire period the parenthetical resolver bug was live.**

### 5.7 MEDIUM — determinism is partial even on the PyTorch side

`models/gnn/train.py:41` is the only `set_seed()`:

```python
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

Missing: `torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic/benchmark`, `CUBLAS_WORKSPACE_CONFIG`,
DataLoader `worker_init_fn` and `generator=`. `cross_city.py:59` reimplements a
weaker `set_seed` (no CUDA). `explainer_metrics.py:87` hardcodes
`torch.manual_seed(1234)`; `explain.py:137` seeds per call. **No study seeds
Python/NumPy globally before sampling scenarios** — sampling happens to use
explicit `random.Random(seed)` / `np.random.RandomState(seed)` instances, which is
actually the better pattern, but it is inconsistent.

Additionally: training ran on **Apple MPS locally and CUDA/T4 on Colab**, and MPS
falls back to CPU for `einsum`. Cross-device bitwise reproducibility is not
achievable and is not documented as a limitation.

### 5.8 MEDIUM — results are overwritten in place

Results land in fixed paths (`results/faithfulness/faithfulness_summary.json`).
Collision is avoided by ad-hoc `--out-tag` suffixes (`_15b`, `_dcontra`,
`_smoke`) rather than by versioned run directories. There is no run id, no
manifest, no append-only guarantee. `--smoke` writes `*_smoke.pt` specifically
because a smoke run once clobbered a real checkpoint.

### 5.9 MEDIUM — no data integrity checks

No `sha256`/`md5` anywhere. Two of three METR-LA/PEMS-BAY URLs point at
`raw.githubusercontent.com/.../master/...` — a **mutable branch ref.** GTFS feeds
(`gtfs_bus.zip`, `google_transit.zip`) are live and change weekly.

### 5.10 LOW — stale config references

`configs/ablations.yaml:54` names `qwen2.5:7b`, which is not installed. The
ablation harness has never been run, so this has not surfaced as an error.

---

## 6. Claims currently supported by code + artifacts

"Supported" here means: **an artifact exists on disk containing this number, and
code exists that produced it.** It does *not* mean reproducible (see §5), and it
does *not* mean the claim's scope is justified (see §7).

| # | Claim | Evidence on disk | n | Caveat |
|---|---|---|---|---|
| C1 | Removing the explanation raises unsupported-citation rate from ~0.005 to ~0.84 on METR-LA / llama3.1:8b | `cross_city_faithfulness_master.json` (corrected) | 93 | Single draw; "unsupported" = one claim type only |
| C2 | The A-vs-B gap holds on 2 models × 2 cities | same, 4 cells | 93 / 85 | Chicago excluded on chance-floor grounds — correctly |
| C3 | The A-vs-B gap holds on a non-traffic graph (IEEE 14-bus) | `results/power_grid_faithfulness/summary.json` | 20 | Synthetic demand; k=4 of 14 nodes; one model |
| C4 | Condition A beats an explicit chance floor by +0.35–0.50 F1 | cross-city + power-grid chance rows | 93 / 85 / 20 | Only implemented in those two studies |
| C5 | A vs B differences have bootstrap 95% CIs excluding 0 | `faithfulness_bootstrap_ci.json`; power-grid: 9/9 CIs exclude 0 | 93 / 20 | Clustered by scenario only, not by draw/model/domain; **no multiple-comparison correction** |
| C6 | Richer city context does *not* change faithfulness (C_RICH ≈ A, CI spans 0) | `faithfulness_summary_15b.json` | 93 | Strong, clean negative result |
| C7 | Contradictory city context does not degrade faithfulness (D_CONTRA halluc 0.000, 93/93) | `faithfulness_summary_dcontra.json` | 93 | Notable robustness result |
| C8 | The active-grounding loop raises F1 0.732 → 0.874 while hallucination falls 0.005 → 0.000 | `active_grounding_summary.json` | 93 | **See §7.1 — the brief's 0.438 → 0.912 is a different, selected run** |
| C9 | The loop transfers to the power grid with no retuning, converging in exactly 1 round | `active_grounding_power_grid_summary.json` | 20 (4 and 2 engaged) | Ceiling effect explicitly acknowledged; n_engaged tiny |
| C10 | GNNExplainer and SHAP select near-disjoint structures (Jaccard 0.022) yet yield identical precision/hallucination | `shap_comparison` summary | 28 | Best read as a **limitation of the metric** |
| C11 | Counterfactual validity is 40% (8/20); all 8 valid ones are target-only | `counterfactual_summary.json` | 20 | Honestly reported as a weakness |
| C12 | Uncertainty framing makes the model hedge more on peripheral than core causes (+0.13) but slightly raises hallucination (0.000 → 0.104) | `uncertainty_summary.json` | 20 | Concentrated in 4/20 |
| C13 | Condition-A residual failures are incompleteness, not fabrication (5/7 under-citation, 0 geographic) | `failure_modes.json` | 7 of 93 | Taxonomy applied by rule, single draw |
| C14 | Full pipeline improves simulated decision quality over prediction-only and random (acc 0.266 / 0.233 / 0.200; delay 19.7 / 15.7 / 13.3) | `sim_eval_summary.json`, `decisions.jsonl` | 444 × 3 seeds | **Model-in-the-loop ground truth** — see §7.4 |
| C15 | METR-LA prediction is competitive with, not better than, published Graph WaveNet (30-min MAE 3.196 vs 3.07) | `eval_metr_la_on_metr_la.json` | — | Project corrected its own earlier overclaim |
| C16 | Heterogeneous fusion is a wash on METR-LA (ΔMAE −0.002/−0.004/+0.007) | `results/fusion/comparison.json` | — | Clean negative result, well reported |
| C17 | On the power grid the ST-GNN loses to persistence overall | `baseline_persistence_power_grid.json` vs `eval_power_grid_on_power_grid.json` | — | Honestly reported |
| C18 | Explainer beats random on Fidelity+ (1.393 vs 0.843) | `explainer_metrics_metr_la.json` | 90 | **Fidelity− is inconclusive (12.93 ≈ 12.84 random)** |
| C19 | Faithfulness degrades measurably as sensor density falls | `sparse_regime_summary.json` | 5 densities | Not mentioned in the brief |

---

## 7. Claims NOT currently supported

### 7.1 The brief's headline active-grounding figure is a selected subpopulation

The brief states: *"Active grounding loop improves claim-level F1 from
approximately 0.438 to 0.912."*

`active_grounding_summary.json` on disk says:

```json
"n_scenarios": 93, "select": "stratified",
"per_round": [{"round": 0, "mean_f1": 0.7323, ...}],
"mean_f1_gain": 0.1415, "frac_reached_overall": 1.0
```

0.438 → 0.912 is the **n=5 smoke run**, launched with `--select low-f1`, on the
five *worst* condition-A failures from Phase 5. `CLAUDE.md` labels it a smoke test
and the full run superseded it. Quoting it as the headline is **selecting on the
dependent variable**: those five were chosen *because* they scored low, so
regression to the mean guarantees a large apparent gain.

**The defensible claim is +0.141 (0.732 → 0.874, n=93).** The engaged-subpopulation
result can be reported *as such*, clearly labelled, alongside the full-sample
number.

### 7.2 "82.8%" is the superseded value

See §5.4. Corrected: 83.9%. Report the corrected one and cite the correction.

### 7.3 No claim about *unsupported claims in general* is currently supported

The central hypothesis is about **unsupported explanation claims**. The system
measures **one claim type** (cause-node membership in top-k), extracted from a
**self-declared JSON field**, against **what the model was shown**. Nothing
currently supports statements about:

- edge, direction, or path claims (the data is shown to the LLM, never checked)
- temporal / lag claims
- confidence-tier claims
- refusals (§4.3 — they score as maximal hallucination)
- claims made in the free `reasoning` prose that never enter `cited_causes`

### 7.4 The decision-quality result is internal consistency, not planning benefit

`sim_eval.py` scores candidate interventions **with the same GNN** that produces
the predictions the LLM sees. The best-scoring action is the ground truth. The
project states this limitation in-file and in config. But two further issues are
visible in the artifacts and are *not* yet addressed:

- **RAW consistency is 0.998** — the prediction-only condition picks essentially
  the same action every time. The comparison is therefore partly "does the
  pipeline break a degenerate constant policy," not "does explanation help."
- The action ground truth was **calibrated** (`budget=45`, `cap=24`) via sweeps
  until the ground-truth distribution looked like "a real 4-way decision problem."
  `CLAUDE.md` argues the mechanism is physical rather than hand-tuned, which is
  reasonable — but the calibration used the same scenario pool that was then
  evaluated. **Whether the simulator was tuned on the test scenarios is a
  question the Phase-13 audit in the new plan must answer, and the current
  evidence says: partly yes.**

### 7.5 "Domain-agnostic" is not earned

`CLAUDE.md` Phase 19 proposes the framing *"mathematical GNN-explanation
grounding is a domain-agnostic mechanism for eliminating LLM hallucination."*
Evidence: 3 traffic datasets (2 of which are the same sensor-network family, the
third excluded) + 1 synthetic-demand 14-bus grid, n=20, one model. Per the brief's
own claim-discipline rules, this must be **"transferred across the tested
graph-based dynamical systems."** Also note the word *eliminating* — rate is 0.005
and 0.050, not 0.

### 7.6 Nothing distinguishes structure from its confounds

The whole point of the new hypothesis is to separate structured evidence from
*more information*, *more retries*, *cleaner formatting*, *domain vocabulary*,
*explainer artifacts*, and *refusal*. Current conditions:

| Needed contrast | Exists? |
|---|---|
| A. Ungrounded | Yes (condition B) |
| B. Prose evidence, same facts | **No** |
| C. JSON evidence, no relations | **No** |
| D. Full graph evidence | Yes (condition A) |
| E. Shuffled edges | **No** |
| F. Shuffled node labels | **No** |
| G/H. Confidence on/off | **No** |
| I. Retrieval-only | Partial — condition B retrieves KB but also drops the prediction framing |
| J. Generic retry | **No** |
| K. Self-critique | **No** |
| L. Active grounding | Yes (Phase 16) |
| M. Wrong evidence | Partial — `D_CONTRA` corrupts *context*, never the *evidence* |
| N. No-answer control | **No**, and unscoreable as the metric stands (§4.3) |

**Condition B confounds at least four variables at once** (no explanation, no
prediction-derived retrieval, different task text, different prompt length). It
cannot isolate structure.

### 7.7 No ground truth anywhere

Every dataset is observational. Nothing in the repository has a known causal
subgraph, known causal path, or known relevant feature set. So "does the explainer
recover the true structure" and "does the LLM describe the true structure" are
currently **unanswerable**. This is exactly the gap the new Phase 2 synthetic
benchmark fills, and it is the single highest-value addition available.

### 7.8 No human validation of the automated metric

Zero human annotations exist. The `human_study/` toolkit was built, smoke-tested
on synthetic demo data, and retired without a single real participant. There is
therefore no evidence that the deterministic verifier agrees with human judgement
of "unsupported."

### 7.9 Statistical issues not yet addressed

- **Pseudoreplication.** 93 scenarios are stratified samples from *one* METR-LA
  test split, sharing one graph, one model, one checkpoint. They are not 93
  independent observations of "an LLM explaining a graph prediction." Bootstrap
  CIs computed over them will be too narrow for any population-level claim.
- **No multiple-comparison correction** anywhere (grep: no `bonferroni`, `holm`,
  `fdr`, `benjamini`). Phase 19 alone reports 9 simultaneous CIs.
- **Single draw per cell** — LLM sampling variance (measured at ~0.08 F1) is not
  in any interval.
- **No prespecified analysis plan.** Thresholds (F1 0.7, fuzzy 82, top-k 8 vs 4,
  flip 35 mph, budget 45 / cap 24) were all chosen during exploration, several
  explicitly by sweeping until the result looked reasonable. That is legitimate
  exploratory work but it means **no current result is confirmatory.**

---

## 8. Verified / needs rerun / informal

### 8.1 Existing verified results
*(artifact on disk, produced by committed code, internally consistent)*

- Prediction metrics, METR-LA epoch-54 and power grid — `eval_*.json` + baselines.
- Explainer metrics vs random (Fidelity+ 1.393 vs 0.843; Fidelity− inconclusive).
- Corrected cross-city × cross-model faithfulness, 6 cells + chance floors.
- Power-grid faithfulness + 2 chance controls + 9 paired bootstrap CIs.
- Active grounding n=93 and its cross-domain n=20 (both variants + noise floor).
- C_RICH / D_CONTRA at n=93 with CIs.
- SHAP comparison n=28 incl. the Jaccard-0.022 diagnostic.
- Counterfactual n=20 (validity 40%).
- Uncertainty n=20 (hedging gap +0.13).
- Failure taxonomy (7 failures).
- sim_eval n=444 × 3 seeds.
- Sparse-regime sweep.
- Fusion comparison (negative).

**All of the above are single-draw on the LLM axis and carry no provenance
stamp.** "Verified" means the artifact exists and matches its code — not that it
would recur.

### 8.2 Results that need rerunning

| Result | Why |
|---|---|
| Phase 5 A/B/C (`faithfulness_summary.json`) | Scored with the pre-correction resolver; paper Table 3 is built from it |
| Phase 15b, 12, 13, 14, 16, 18, and Phase 20's METR-LA row | Same — the 8 artifacts `CLAUDE.md` lists as owed |
| Everything produced on epoch-34 | Live `metr_la_best.pt` is epoch-54; §5.5 |
| All Chicago results | Source data not reproducible; §5.1 |
| Phase 7 ablation matrix | **Never run** — `table4_ablation.tex` is a PENDING placeholder |
| Cross-city transfer table | Checkpoints exist, no results table found |
| Fidelity+ on SHAP top-k | Named as the natural follow-up, never run |
| Explainer metrics on power_grid | Blocked on the degenerate lowest-voltage target rule |

### 8.3 Informal observations only (prose, no artifact)

- "Beats published Graph WaveNet" — **retracted by the project itself**; corrected
  to "competitive" (3.196 vs 3.07).
- Learned modality gates (traffic 0.598 / transit 0.502 / events 0.304 / weather
  0.303) — read manually out of the checkpoint;
  `comparison.json.final_modality_gates` is `null`.
- Learned α drift 0.49 → 0.39 — from a 3-epoch placeholder run; the per-epoch CSV
  for the real run lives Colab-side.
- "Visibility is a dead channel" — stated, never quantified in an artifact.
- Phase-13 mitigations (drop target from cause list, gate free-flow advisories) —
  proposed, only the under-citation one (Phase 16) was built.
- Phase-17 "45% flip rate, mild 93% / medium 31% / deep 17%" — calibration console
  output, not a committed artifact.
- The entire Phase-8 human study — toolkit only, zero real data.

---

## 9. Ten things to fix before any expensive experiment runs

Ranked by "how much later work is invalidated if this is skipped."

1. **Persist everything per LLM call** — prompt, raw response, parsed object,
   model digest, sampling params, timings. Without this, §5.3 repeats forever.
2. **Seed and fully parameterise the Ollama call**; record the model digest.
3. **Run k≥5 draws per cell** and treat draw as a nested random factor.
4. **Stamp provenance** on every artifact: git SHA, Python + package versions,
   checkpoint hash (not filename), data hash, prompt-template hash, fuzzy backend.
5. **Freeze the Chicago snapshot** (date-bounded query + checksum) or drop
   Chicago from confirmatory claims.
6. **Content-address checkpoints**; make explanation caches key on checkpoint
   hash, not filename presence.
7. **Write unit tests before touching the metric** — the resolver ladder, the
   scoring math, the `n_causes == 0` branch, and the region tables.
8. **Fix the refusal semantics** (§4.3) before condition N is designed. Refusal
   must be its own outcome category, not hallucination 1.0.
9. **Mark superseded artifacts in the artifacts themselves**, and repoint
   `make_paper_artifacts.py`.
10. **Write the prespecified analysis plan before the confirmatory run** — after
    that, exploratory and confirmatory results must never be merged.

---

## 10. Honest summary

XTraffic is a real, working, three-layer system with far more genuine
experimental content than its stage would suggest, and — unusually — a written
record that repeatedly catches and corrects its own errors. The Phase-11
resolver investigation in particular is a piece of serious scientific work: an
outlier was found, traced to three distinct measurement defects, reproduced with
a purpose-built control to three decimal places, and the affected numbers were
rerun and revised *downward*. That is the behaviour the whole project should be
judged on.

The gap between what exists and what the new hypothesis requires is nevertheless
large, and it is concentrated in three places:

1. **The metric is narrower than the hypothesis.** It checks one claim type, from
   a self-declared field, against what the model was shown. The hypothesis is
   about unsupported claims in general, against something true.
2. **The experimental design cannot isolate structure.** Condition B changes at
   least four things at once. There is no prose-matched, format-matched,
   retry-matched, or corrupted-evidence control, so "structure matters" is not yet
   separable from "more information," "cleaner format," or "more attempts."
3. **The reproducibility floor is too low to support confirmatory claims.** No LLM
   seed, no raw output, no provenance, no tests, one draw per cell, and one
   dataset that cannot be re-downloaded.

None of these is fatal and none requires discarding existing work. The synthetic
benchmark (new Phase 2) supplies the ground truth that fixes (1); the matched
ablation matrix (new Phase 5) fixes (2); Phase 1 fixes (3). The existing studies
become the *exploratory* record — which is a legitimate and valuable thing for
them to be, provided they are labelled as such and never presented as
confirmatory.

**Recommended sequence:** Phase 1 (reproducibility) → Phase 3 (claim verifier,
test-first) → Phase 2 (synthetic ground truth) → Phase 5 (matched ablations) →
Phase 11 (analysis plan) → everything else. Rationale and alternatives in
`docs/EXPERIMENT_PLAN.md`.

**Nothing in this repository was modified during this audit.**

---

## 11. Addenda — findings logged after the audit was written

Sections 0–10 above are the 2026-08-21 audit as delivered and are unchanged.
Findings that surface later are appended here, dated, rather than edited into the
original text, so the audit's own record stays intact.

### 11.1 MEDIUM — `verify_traffic_unchanged.py` selects its inputs by filename sort order

*Logged 2026-08-24, during the explanation-network analysis. Not fixed.*

`evaluation/verify_traffic_unchanged.py:67-72`:

```python
def _explanation_files(limit: int = 6) -> List[str]:
    files: List[str] = []
    for pat in ("evaluation/results/faithfulness/explanations_cache/*.json",
                "evaluation/results/explanations/*.json"):
        files += sorted(glob.glob(os.path.join(PKG_ROOT, pat)))[:limit]
    return files
```

The gate's 12 golden inputs are whatever the first six lexically-sorted files in
each directory happen to be. `explanations_cache/` currently holds 93 files named
`metr_la_<window>_<target>.json`, so the selection is decided by string ordering
of the window index — the current first entry is `metr_la_1154_197.json`.

**Why it matters:** any future study that writes an explanation into
`explanations_cache/` with a lower-sorting name (`metr_la_1000_*.json`, or
anything beginning with a digit below `1154`) silently changes *which* artifacts
the regression gate compares, without changing the gate's code or its golden
file. The gate would then either fail for a reason unrelated to the change under
test, or — worse — pass while no longer covering the artifacts it was baselined
on. This is the same failure mode §5.6 already records for this file: its probe
list once contained only bare region names and bare ids, so it passed throughout
the period the parenthetical resolver bug was live.

**Fix (not applied):** pin the 12 filenames explicitly in the module, or store
them in the golden JSON alongside the hashes, so the gate's input set is data
rather than an accident of directory listing order.

**Current exposure:** none. The explanation-network analysis writes only into its
own run directory and never into `explanations_cache/`; the gate was confirmed
PASSING before and after that work.
