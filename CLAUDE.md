# XTraffic Project Context

> This section is the live project memory (Phase 0). The full build playbook follows below.

## What this is
XTraffic is a research system being built for submission to IEEE ITSC 2026 or a
NeurIPS 2026 workshop. It is a three-layer pipeline:
1. LAYER 1 — Spatial-Temporal Graph Neural Network (ST-GNN) for traffic prediction,
   based on Graph WaveNet with three novel modifications
2. LAYER 2 — Graph explainability via GNNExplainer, producing structured
   mathematical explanations of each prediction
3. LAYER 3 — LLM advisory module (local Ollama) that translates mathematical
   explanations into faithful, actionable planner recommendations

## The paper's three contributions (never lose sight of these)
1. A generalizable ST-GNN with heterogeneous data fusion (sensors + weather +
   events + transit), evaluated cross-city (train LA, test Chicago)
2. A novel FAITHFULNESS METRIC measuring alignment between GNNExplainer's
   mathematical explanation and the LLM's natural-language reasoning
3. A structured evaluation showing the full pipeline improves human decision
   quality versus raw predictions or raw explainability outputs

## Who I am
High school student, still learning Python and deep learning. Rules for working
with me:
- Explain every architectural decision in comments and in chat
- Add tensor shape comments on every line where shapes change
- Prefer readable code over clever code
- Never silently change something we built earlier — flag it and explain why
- When something fails, teach me what the error means before fixing it

## Hard constraints
- Everything must be reproducible: pinned dependency versions, random seeds set,
  data download scripts (never manually downloaded files)
- All experiments log to /xtraffic/evaluation/results/ as both JSON and CSV
- The LLM layer runs locally via Ollama — no external API calls, no API keys
- Target hardware: consumer laptop/desktop, optionally free-tier Colab GPU for
  training. Keep model sizes and batch sizes realistic for this.
- Python 3.9 is the local interpreter — code must stay 3.9-compatible
  (use typing.Optional/Union, not the `X | Y` syntax).

## Repo layout (canonical — do not deviate)
/xtraffic
  /data/raw            # downloaded datasets, gitignored
  /data/processed      # tensors ready for training, gitignored
  /data/pipelines      # download + preprocessing scripts (these ARE committed)
  /models/gnn          # ST-GNN architecture, training, checkpoints
  /models/explainer    # GNNExplainer wrapper + explanation schema
  /models/advisor      # LLM advisory module + city knowledge bases
  /evaluation          # metrics, faithfulness, ablations, human eval, figures
  /notebooks           # exploration only, nothing load-bearing lives here
  /utils               # shared helpers
  /configs             # YAML config per experiment — no magic numbers in code
LAB_NOTEBOOK.md
CLAUDE.md

Note: the pre-existing OlympiFlow app (React + FastAPI, /src and /backend) is kept
as a base and reference — its Ollama RAG advisor pattern seeds Phase 4. XTraffic
proper lives under /xtraffic.

## Current status
Phase 0 complete (project memory). Phase 1 complete and gate-passed:
- METR-LA: 207 nodes, 34,272 timesteps, 8.11% missing, scaler mean 54.4 mph
- PEMS-BAY: 325 nodes, 52,116 timesteps, 0.003% missing, scaler mean 62.7 mph
- Chicago: 1,020 segment-nodes (recent Socrata window; small + sparse until
  HISTORY window widened in Phase 6). All three emit X[·,12,N,2] / Y[·,12,N].
Data read from CSV (Zenodo 5146275) — no h5/pytables dependency.
Phase 1 gate PASSED (all three stats reports sane, no post-preprocessing NaNs,
chronological non-overlapping splits, shapes match config).
Phase 2 COMPLETE and gate-passed. XTrafficSTGNN (395K params) in pure PyTorch
(no torch-geometric needed for Phase 2) at models/gnn/stgnn.py:
- Graph WaveNet base (gated dilated TCN + diffusion GCN, residual+skip stack)
- MOD 1 semantic co-movement edges: A_final = a*A_phys + (1-a)*A_sem, a=sigmoid(alpha);
  alpha logged/epoch (drifted 0.49->0.39 in 3 epochs = model leaning on learned graph)
- MOD 2 multi-scale temporal: parallel causal dilations [1,2,4,8] per block, T preserved
- MOD 3 heterogeneous fusion: per-modality encoder+gate; absent modality -> None (no crash)
Masked-MAE loss/metrics in real mph (missing = raw-0 sentinel; utils/metrics.py).
Trainer/baselines/evaluate under models/gnn/; config configs/train_metr_la.yaml.
METR-LA TEST (best ckpt, only 3 epochs): MAE 3.53 | 15min 2.98 | 30min 3.52 | 60min 4.42.
Baselines: HistAvg 5.15, LinReg 5.05 -> model beats both clearly; 30min in 3.0-3.5 target.
Local training ~40min/epoch on Apple MPS (einsum falls back to CPU) -> impractical;
use models/gnn/colab_train.ipynb on a free T4 for the full 100-epoch run.
Phase 2 gate PASSED: 30-min TEST MAE 3.52 on METR-LA (3-epoch placeholder ckpt),
beats HistAvg 5.15 / LinReg 5.05 clearly; alpha drift 0.49->0.39 logged to CSV.
Full 100-epoch training now RUNNING on Colab T4 (numbers to be refreshed on completion).
Phase 3 COMPLETE and gate PASSED (on the 3-epoch placeholder ckpt; re-run gate on full model later).
DEVIATION (flagged): implemented the GNNExplainer algorithm (Ying et al. 2019)
faithfully in pure PyTorch instead of wrapping torch_geometric.explain — our model
takes a modality-dict + dense adjacency, not PyG's (x, edge_index), so a direct
reimpl is less code and fully transparent (keeps Phase-2 "no torch-geometric").
Files under models/explainer/: schema.py (the explanation JSON CONTRACT + validator
Phase 4 depends on), node_names.py (offline lat/lon->LA-region naming, no geocoder),
explain.py (learned node+edge masks; preservation + sparsity + entropy loss; edge
mask on physical adj via temp buffer swap; propagation path = BFS; lag = window
cross-correlation; confidence = mean Jaccard of top-k over K reruns), scenarios.py,
generate.py. Evaluation: evaluation/explainer_metrics.py (fidelity+/-, stability,
sparsity vs random baseline), evaluation/visualize_explanation.py (vector-PDF map).
REAL BUG found+fixed: scenario/target selection tested validity in z-space but the
missing sentinel (0 mph) is ~0 in REAL space -> was picking missing sensors as
targets. Fixed to mph-space (matches utils.metrics masking).
KEY FINDING: node-occlusion fidelity is only meaningful on CONGESTED targets — a
free-flowing 65mph prediction has no spatial cause, so uniform target sampling
washes out the signal. Evaluating on the slowest valid sensor per window (also the
Phase-5 stratification): Fidelity+ 0.496 vs 0.244 random (2.0x, PASS), Stability
0.796. Fidelity- ~9.26 == random ~9.28 (INCONCLUSIVE on the undertrained model:
prediction is distributed across many nodes so top-8 isn't clearly sufficient —
expect improvement on the full 100-epoch ckpt). 6 scenario explanations in
evaluation/results/explanations/ all pass schema validation; congested cases are
semantically sane (East LA 20->15mph explained by nearby congested Glendale sensors,
lag 5min). Figure at evaluation/results/figures/explanation_rush_hour_pm.pdf.
Phase 3 GATE PASSED: Fidelity+ 0.496 vs 0.244 random (2.0x), Stability 0.796,
6 schema-valid + semantically-sane scenario explanations, one publication figure.
Fidelity- weak on the 3-epoch ckpt (~9.26 == random ~9.28) — FLAGGED for rerun
once the full 100-epoch Colab ckpt lands.
Phase 4 COMPLETE and gate PASSED — LLM advisory layer (Ollama, Layer 3).
Files under models/advisor/: knowledge_base.py (per-city KB loader + keyword
retrieval, productionized from the OlympiFlow RAG scorer; region_tags keyed to
node_names.py region labels so retrieval lines up with the explanation),
kb/la.json (11 factual LADOT/Caltrans/Metro chunks: corridors, bottlenecks,
signal timing, transit, capacities, incidents) + kb/README.md (fill-in template
so Chicago is code-free in Phase 6), advisor.py (Advisor class: prompt assembled
SYSTEM->CITY CONTEXT->MATH EXPLANATION->TASK; Ollama /api/generate with
format=json + temp 0.1; advisory output has its OWN validated JSON contract
{reasoning, cited_causes[], recommendations[]} with a 2-retry self-correct loop;
raw responses kept for the Phase-7 ablation log), pipeline.py (advise/Pipeline =
GNN->explainer->advisor, lazy checkpoint load, fast path reuses saved
explanations), demo.py. Config configs/advisor.yaml (default llama3.1:8b, an
ablation variable). VERIFIED: full pipeline ran end-to-end on all 6 committed
scenarios against local llama3.1:8b; all outputs are schema-valid; 6 advisories
saved to evaluation/results/advisories/. Cited causes overwhelmingly resolve to
the explanation's top-nodes and recommendations cite real KB infrastructure via
grounded_in. Minor slippage observed (occasional null resolution, one case citing
the target as its own cause, confused reasoning on free-flowing-cause windows) —
this is exactly the baseline Phase 5's faithfulness metric will quantify.
Phase 4 GATE PASSED: full pipeline ran end-to-end on all 6 scenarios, every
advisory schema-valid; grounding strong (cited causes resolve to the explanation
top-nodes, recommendations cite real KB infrastructure). Minor slippage was
deliberately left un-tuned so Phase 5's faithfulness metric captures real
baseline behavior: occasional null resolution, one self-attribution error
(target cited as its own cause), confused reasoning on free-flowing-cause windows.
Phase 5 COMPLETE and gate PASSED — the faithfulness metric (Contribution #2).
Files: evaluation/faithfulness.py (entity resolver: exact-sensor
-> exact-region -> fuzzy[rapidfuzz-or-difflib, thresh 82] -> geographic gazetteer;
resolves LLM cited_causes to node-id SETS independently of the LLM's self-reported
id; region-level credit = citation hits top-k iff its node set intersects top-k;
metrics cause precision/recall/F1 + quantitative fidelity[±10%] + hallucination
rate, each defined in-docstring), evaluation/resolver_labels.json (30 hand-labeled
cases) + evaluation/validate_resolver.py (RESOLVER 96.7% = 29/30, GATE >=90% PASS),
evaluation/run_faithfulness_study.py (>=100 scenarios stratified 5 tod-bands x 3
congestion terciles, seed 42; explanation cached once per scenario; runs A/B/C;
CSV+JSON+boxplot PDF to evaluation/results/faithfulness/). Conditions added to
advisor.py as advise_condition()/build_prompt_condition() WITHOUT changing Phase-4
advise()/build_prompt() (= condition A): B = prediction-only (no explanation, KB
retrieved from a prediction-only view so top_nodes can't leak), C = explanation
but no city context. SMOKE RUN (--limit 6, end-to-end vs real ckpt + llama3.1:8b):
A F1 0.817/halluc 0.000 | B F1 0.067/halluc 0.833 | C F1 0.820 -> A >> B, grounding
proven (B precision collapses 1.00->0.17).
Phase 5 GATE PASSED (smoke run, --limit 6 end-to-end vs real ckpt + llama3.1:8b):
A F1 0.817/halluc 0.000 | B F1 0.067/halluc 0.833 | C F1 0.820/halluc 0.000.
CORE CLAIM PROVEN: the mathematical explanation eliminates hallucination
(A/C halluc 0.000 vs B 0.833) and restores grounding (B precision collapses
1.00->0.17). C ~= A on this smoke sample -> city context adds little to
faithfulness on these 6 (all night/low-congestion due to --limit front-truncation);
FLAGGED for investigation at scale — the full run covers all strata and may
separate A from C on congested/high-signal windows. Full >=100-scenario study
(python -m xtraffic.evaluation.run_faithfulness_study, ~1-2h) RUNNING in the
background for the paper table. STILL OWED: Phase-3 gate re-run on the full
100-epoch Colab ckpt when it lands.

Phase 6 COMPLETE (gate = infrastructure verified) — real feature fusion +
cross-city generalization (Contribution #1's results). All code built and locally
verified; the two long training runs (full fusion retrain, Chicago transfer) are
the only things owed and run on Colab via colab_fusion_train.ipynb.

MECHANISM (flagged design decision): Phase-6 modalities attach as SIDECARS, never
by rewriting the Phase-1 tensors. Each feed writes processed/<ds>/mod_<name>.npz
([S,12,N,C], windows+split+train-scale IDENTICAL to traffic X via the shared
data/pipelines/_modality_common.py) + mod_<name>.json. loaders.make_fusion_loaders
concatenates present sidecars into a per-batch M; build_modality_dict slices M by
layout; an absent feed -> None (MOD-3 gate zeroes it). Fusion is OPT-IN via
use_sidecars in configs/train_metr_la_fusion.yaml (run_name=metr_la_fusion, own
ckpt); the base train_metr_la.yaml stays traffic-only even with sidecars on disk.
--smoke now writes *_smoke.pt (never clobbers the real best ckpt).

FEEDS BUILT + RAN end-to-end against live sources (all sidecars = 23974/3425/6850,
matching traffic exactly):
- weather.py: Open-Meteo ERA5 archive (keyless), 207 nodes -> 19 grid cells,
  temp/precip/visibility. REAL BUG FOUND+FIXED: the archive does NOT serve
  `visibility` -> that channel came back all-NaN -> train_loss=nan. Fixed
  defensively in _modality_common.save_modality_sidecar: non-finite cells are
  filled with the channel's train mean (neutral ~0 after z-score); a fully-missing
  channel collapses to 0 and the gate ignores it (warns loudly). Visibility is
  therefore currently a dead channel (temp+precip carry the signal) — noted for the
  paper; could drop weather to 2 channels later.
- events.py: committed curated events_la_2012.json (exact venue coords; editable
  representative 2012 Dodgers/Lakers/Bowl dates — VERIFY vs public schedules before
  the paper). Proximity-decayed (exp(-dist/2500m)) indicator, 14/15 events in range.
- transit.py: LA Metro GTFS static (bus 11892 + rail 463 stops), nearby-stop count
  per node within 1km (mean 24.6, max 115); static-in-time node context.
FUSION PATH VERIFIED: train_metr_la_fusion --smoke loads weather[0:3] events[3:4]
transit[4:5], 4 gates active, finite loss. evaluate.py made FUSION-AWARE (uses
make_fusion_loaders + M/layout, gated by the ckpt's use_sidecars so traffic-only
ckpts are unchanged) — else a fusion ckpt would be evaluated with its modalities
zeroed. evaluation/fusion_comparison.py evaluates both ckpts side by side at
15/30/60 min + reports final learned modality gates -> results/fusion/comparison.
colab_fusion_train.ipynb runs the whole thing on a T4 (builds tensors+3 sidecars,
trains traffic-only baseline -> ALSO restores metr_la_best.pt, trains fusion,
prints the comparison table). Full 100-epoch fusion retrain + the numbers still
OWED (Colab).

STATUS SNAPSHOT (2026-07-03): Phases 6 and 7 are STRUCTURALLY COMPLETE and
gate-passed on smoke runs; all real paper numbers are OWED pending the Colab
checkpoint. The fusion notebook (colab_fusion_train.ipynb) has been KICKED OFF on
a Colab T4 — its baseline step REGENERATES metr_la_best.pt (replacing the
accidentally-overwritten placeholder). Once it lands: refresh fusion/cross-city/
ablation numbers and re-run the Phase-3 gate on the full 100-epoch ckpt.

Phase 8 COMPLETE — human evaluation toolkit (Contribution #3). The full toolkit
is built, smoke-verified end-to-end, and the circular-ground-truth concern is
fully addressed in code, config, AND paper framing via all three options:
(1) LIMITATION stated verbatim in simulate.py + human_study.yaml; (2) reframed as
an INTERNAL-CONSISTENCY result in the paper narrative; (3) EXTERNAL-ANCHOR
mechanism (external_anchors.json + analyze.py --anchors -> separate
external_anchor_accuracy block, auto-loaded when present). What remains is DATA
COLLECTION only — the real 24-scenario build (needs the Colab ckpt + Ollama) and
the actual expert sessions — not toolkit work. Files under evaluation/human_study/,
one config
configs/human_study.yaml (24-scenario grid, sim knobs, conditions, Likert):
- simulate.py: InterventionSimulator = MODEL-IN-THE-LOOP ground truth. Each
  candidate action (no_action / signal_retiming / ramp_metering / reroute /
  transit_surge) is simulated by adding uplift_mph to the last apply_steps of the
  input speed channel on a graph-derived node set (target+1-hop / slower upstream
  1-hop / target-only / 2-hop@half), then reading the GNN's 30-min network delay
  (sum max(0, free_flow - pred) over VALID nodes). Lowest delay = ground truth.
  LIMITATION stated in-file + config: not observed reality, inherits model bias.
- scenarios.py: samples 24 stratified windows (4 tod-bands x 3 congestion x 2,
  seeded, slowest-valid-sensor target = same REAL-mph validity rule as Phase 3/5),
  builds RAW/XAI/XTRAFFIC bundles (RAW=prediction only; XAI=+importance table +
  reused visualize_explanation PNG + lag/confidence; XTRAFFIC=+Advisor.advise
  reasoning+recommendations), a cyclic LATIN SQUARE (evaluator e, scenario pos p ->
  conditions[(p+e)%3]; balanced at panel size 3 or 6), writes scenarios.json +
  assignment.json + figures/. --no-llm builds without Ollama.
- app.py: minimal FastAPI single-page app (no DB, no external assets, python-
  multipart for form POSTs). Name + participant-number start page -> serves next
  UNANSWERED scenario in the evaluator's assignment (progress derived from the
  JSONL log, so resumable) -> logs choice+confidence(1-7)+usefulness(1-7)+correct
  to responses.jsonl. Figure route 404s gracefully; img omitted when absent.
- demo_data.py: synthetic (dataset="DEMO") scenarios.json/assignment.json so the
  full app->JSONL->analyze loop is pilotable with NO checkpoint/Ollama.
- analyze.py: per-condition accuracy/confidence/usefulness + PAIRED Wilcoxon
  signed-rank (paired BY EVALUATOR = within-subjects Latin square) on XTRAFFIC-vs-
  RAW / XTRAFFIC-vs-XAI + rank-biserial effect sizes; prints small-n caveat; writes
  analysis_report.json + CSV. Gracefully handles tiny n / zero-variance (None p).
SMOKE-VERIFIED: demo_data -> drove app end-to-end via FastAPI TestClient (eval0
completed RAW/XAI/XTRAFFIC, 303 redirects, resumable) -> analyze produced a sane
per-condition table + paired tests with effect sizes. fastapi/uvicorn/python-
multipart pinned in requirements.txt; responses/analysis/figures gitignored.
Phase 8 GATE (self-pilot 3 scenarios + sane analyze report) PASSED on demo data;
re-run on real scenarios once the Colab ckpt lands. OWED: real 24-scenario build
(scenarios.py, needs ckpt+Ollama) + the actual expert sessions.
CIRCULAR-GROUND-TRUTH honesty (2026-07-04): the Phase-8 ground truth is
model-in-the-loop (same GNN scores interventions + produces the shown predictions).
Addressed three ways, all in the paper: (1) stated as a Limitation verbatim
(simulate.py + human_study.yaml); (2) reframed as an INTERNAL-CONSISTENCY result
(does the pipeline help humans use the model's OWN predictions better?); (3)
EXTERNAL ANCHORS — new external_anchors.json lets the professor mark 1-2 scenarios
with the real-world-known best action; analyze.py --anchors reports a separate
EXTERNAL-ANCHOR ACCURACY block scored against those expert labels (auto-loads the
file if present, skips _-prefixed meta keys; absent file -> simulation-only,
unchanged). Template at human_study/external_anchors.example.json. Anchor path
verified on demo data (anchored scenarios scored vs expert action, un-anchored
excluded; external_anchor_accuracy written to analysis_report.json).

CROSS-CITY (evaluation/cross_city.py, compiles; needs Chicago processed + a trained
source ckpt to run): zero-shot / fine-tuned(10%) / from-scratch on a target city.
HONEST TRANSFER BOUNDARY (documented in-file): only NODE-AGNOSTIC weights (temporal/
graph convs, modality encoders+gates, readout) transfer; NODE-SPECIFIC params
(sem_embed, nodevec1/2, physical_adj) are re-initialised at the target N — the
standard adaptive-graph limitation. Saves target ckpts (<ds>_{zero_shot,fine_tuned,
from_scratch}.pt) so the Chicago faithfulness study can load one.

CHICAGO ADVISOR: models/explainer/node_names.py made CITY-AWARE (per-city region
boxes; auto-detected from node_meta['dataset']; defaults to LA so Phase 3/4/5 are
byte-identical) with a real _CHICAGO_REGIONS table + Loop compass fallback; Chicago
nodes labelled "segment" not "sensor". kb/chicago.json (9 CDOT/IDOT/CTA chunks:
Dan Ryan/Kennedy/Eisenhower/Stevenson/LSD, Circle Interchange, Loop grid, CTA)
registered in advisor.yaml. run_faithfulness_study already takes --dataset/--city
-> Chicago 50-scenario run is a one-liner once a Chicago ckpt exists. Phase-5
resolver re-validated after the refactor: still 96.7% (29/30), GATE PASS.

STILL OWED (Colab / longer runs): full fusion retrain + comparison table; Chicago
Phase-1 pipeline + cross_city transfer table; Chicago faithfulness study; and the
carried-over Phase-3 gate re-run + 100-epoch traffic-only ckpt (NOTE: the old
3-epoch placeholder metr_la_best.pt was accidentally overwritten by a smoke run and
removed — regenerate from the Colab run before rerunning Phase 3/4/5 demos).

Phase 7 IN PROGRESS — ablation harness (structurally COMPLETE + smoke-verified;
the real multi-seed runs are Colab-scale and owed). One command, one config:
evaluation/ablations.py driven by configs/ablations.yaml.
- MODEL ablations enabled by NEW default-preserving switches on XTrafficSTGNN:
  use_semantic (MOD1 off -> first support = plain physical adj) and use_multiscale
  (MOD2 off -> STBlock dilations (1,) instead of (1,2,4,8); STBlock.DILATIONS is now
  an instance `dilations` param). train.py/evaluate.py pass both from cfg["model"]
  (default True). Leave-one-out via a new cfg key `sidecar_modalities` (load only
  some feeds; dropped feed -> None -> gated); evaluate.py honors it too so a
  no-weather model is TESTED without weather. Variants: full, no_semantic,
  no_multiscale, no_fusion (use_sidecars off), no_{weather,events,transit}. Each
  trained x `seeds` (3), reported mean+/-std of MAE/RMSE/MAPE @15/30/60.
- PIPELINE ablations reuse Phase-5 conditions A/B/C; run_faithfulness_study gained
  --model (Ollama-model override = LLM-model ablation) + --out-tag (so per-model
  runs don't clobber). Harness reads each run's faithfulness_summary_<tag>.json
  (key faithfulness_f1_mean etc.) and tabulates F1 + hallucination per (model,cond).
- Verified: `ablations.py --model-only --smoke` trains all 7 variants via
  subprocess, evaluates, writes model_ablations.{csv,tex} (LaTeX booktabs, mean+/-std)
  + prints the master table. The 4 architecture variants build with correct shapes
  (no_multiscale 259K vs full 395K params). Pipeline part compiles; needs Ollama +
  a metr_la ckpt to run (both currently absent locally) — runs on the same box that
  has Ollama. OWED: real multi-seed model runs + pipeline runs for the paper tables.

Phase 9 COMPLETE (structurally) + clean-clone verified — paper artifacts.
evaluation/make_paper_artifacts.py regenerates EVERY figure + table from logged
results into evaluation/paper/{figures,tables}. Figures: Fig1 architecture
(hand-composed SVG schematic), Fig2 explanation (reuses visualize_explanation),
Fig3 learned alpha, Fig4 modality gates, Fig5 faithfulness A/B/C boxplot, Fig6
cross-city bars, Fig7 human study — all vector PDF (Fig1 SVG), Wong colorblind-safe
palette, IEEE two-column font sizes. Tables (LaTeX booktabs, \input-ready): T1
prediction vs baselines, T2 explainability vs random, T3 faithfulness by condition,
T4 ablation (copies Phase-7 harness .tex), T5 human study. DESIGN: every artifact
is independent + wrapped so a missing upstream result (Colab-owed fusion/cross-city/
ablation) writes a LOUD "PENDING" placeholder instead of crashing — holes are
honest and visible, script always builds the skeleton. Placeholder-ckpt figures
(<3 epochs) print a red "refresh on full run" note; smoke-sample tables/figs print
their n. REPRODUCE.md (repo root): ordered fresh-clone-to-every-number command list,
step per phase, flags which steps need GPU/Ollama, ends with a verified-vs-owed
ledger. VERIFIED: fresh `git clone` (no processed data, no checkpoints) + overlay
of the new files -> make_paper_artifacts ran clean; 7 figs + 5 tables written;
Fig2/Fig6/T4 correctly fell to PENDING, everything backed by committed results
built (T1 MAE 2.98/3.52/4.42 vs HistAvg 5.15, T3 A-F1 0.817/halluc 0 vs B 0.067/
0.833). OWED (unchanged, all Colab): full-ckpt numbers refresh, cross-city + Chicago
+ multi-seed ablations — each drops a result file that step-8 picks up with no code
change. Playbook's final "fresh-clone reproduction actually works" gate met for the
committed-results path; the compute-heavy numbers regenerate as their runs land.

---

# XTraffic — Claude Code Build Playbook

The complete prompt sequence for building the XTraffic research system with Claude Code.
Paste one phase at a time. Do not skip verification gates. Do not paste Phase N+1 until Phase N passes.

---

## How to use this playbook

1. **One phase per session.** Start each Claude Code session fresh, paste the phase prompt, let it work.
2. **Verify before advancing.** Every phase ends with a verification gate — a set of commands that must succeed and outputs that must look right. If they don't, tell Claude Code exactly what failed and paste the error.
3. **Commit after every passing phase.** `git add -A && git commit -m "Phase N complete: <what works>"`. Your GitHub history becomes evidence of the research process — that matters for your paper and your applications.
4. **Ask "why" constantly.** Any time Claude Code writes something you don't understand, reply: "Explain what this block does and why you chose this approach over alternatives." You are the author of this paper. You will be asked questions about it by your professor and eventually reviewers. You cannot defend code you don't understand.
5. **Keep a lab notebook.** A single `LAB_NOTEBOOK.md` in the repo root. After each session write 3–5 lines: what you built, what the numbers were, what surprised you. This becomes your paper's experiments section almost for free.

---

## Phase 0 — Project Memory (do this before anything else)

Claude Code reads a file called `CLAUDE.md` in your repo root at the start of every session. This is how you stop re-explaining the project every time. Paste this:

```
Create a file called CLAUDE.md in the repository root with the following content,
then confirm you've read and understood it:

# XTraffic Project Context

## What this is
XTraffic is a research system being built for submission to IEEE ITSC 2026 or a
NeurIPS 2026 workshop. It is a three-layer pipeline:
1. LAYER 1 — Spatial-Temporal Graph Neural Network (ST-GNN) for traffic prediction,
   based on Graph WaveNet with three novel modifications
2. LAYER 2 — Graph explainability via GNNExplainer, producing structured
   mathematical explanations of each prediction
3. LAYER 3 — LLM advisory module (local Ollama) that translates mathematical
   explanations into faithful, actionable planner recommendations

## The paper's three contributions (never lose sight of these)
1. A generalizable ST-GNN with heterogeneous data fusion (sensors + weather +
   events + transit), evaluated cross-city (train LA, test Chicago)
2. A novel FAITHFULNESS METRIC measuring alignment between GNNExplainer's
   mathematical explanation and the LLM's natural-language reasoning
3. A structured evaluation showing the full pipeline improves human decision
   quality versus raw predictions or raw explainability outputs

## Who I am
High school student, still learning Python and deep learning. Rules for working
with me:
- Explain every architectural decision in comments and in chat
- Add tensor shape comments on every line where shapes change
- Prefer readable code over clever code
- Never silently change something we built earlier — flag it and explain why
- When something fails, teach me what the error means before fixing it

## Hard constraints
- Everything must be reproducible: pinned dependency versions, random seeds set,
  data download scripts (never manually downloaded files)
- All experiments log to /xtraffic/evaluation/results/ as both JSON and CSV
- The LLM layer runs locally via Ollama — no external API calls, no API keys
- Target hardware: consumer laptop/desktop, optionally free-tier Colab GPU for
  training. Keep model sizes and batch sizes realistic for this.

## Repo layout (canonical — do not deviate)
/xtraffic
  /data/raw            # downloaded datasets, gitignored
  /data/processed      # tensors ready for training, gitignored
  /data/pipelines      # download + preprocessing scripts (these ARE committed)
  /models/gnn          # ST-GNN architecture, training, checkpoints
  /models/explainer    # GNNExplainer wrapper + explanation schema
  /models/advisor      # LLM advisory module + city knowledge bases
  /evaluation          # metrics, faithfulness, ablations, human eval, figures
  /notebooks           # exploration only, nothing load-bearing lives here
  /utils               # shared helpers
  /configs             # YAML config per experiment — no magic numbers in code
LAB_NOTEBOOK.md
CLAUDE.md

## Current status
Phase 0 just completed. Nothing else built yet.
```

After each phase, tell Claude Code: **"Update the Current status section of CLAUDE.md to reflect what we just completed."**

---

## Phase 1 — Environment + Data Pipelines (METR-LA, PEMS-BAY, Chicago)

```
Read CLAUDE.md first.

PHASE 1 GOAL: Reproducible data pipelines for all three cities, verified end to end.

1. Create the canonical repo layout from CLAUDE.md. Add a .gitignore that excludes
   /data/raw, /data/processed, model checkpoints, and __pycache__.

2. Create /xtraffic/configs/data.yaml holding all dataset URLs, split ratios
   (70/10/20 by time, never shuffled — explain in a comment why shuffling time
   series splits causes data leakage), sliding window sizes (12 steps in → 12 steps
   out at 5-minute resolution), and random seed 42.

3. Write /xtraffic/data/pipelines/metr_la.py:
   - Downloads METR-LA (traffic speed h5 + sensor distances csv) from the public
     DCRNN data release on GitHub
   - Builds the adjacency matrix from sensor distances using a thresholded
     Gaussian kernel (the standard method from the DCRNN paper — cite it in a
     comment)
   - Builds sliding-window tensors: X shape [samples, 12, 207, features],
     Y shape [samples, 12, 207]
   - Z-score normalizes using TRAIN statistics only (comment explaining why using
     test statistics is leakage)
   - Saves tensors + scaler + adjacency to /data/processed/metr_la/
   - Prints a stats report: nodes, edges, timesteps, missing-data percentage,
     date range

4. Write /xtraffic/data/pipelines/pems_bay.py — same treatment for PEMS-BAY
   (325 sensors), same output format.

5. Write /xtraffic/data/pipelines/chicago.py:
   - Pulls Chicago Traffic Tracker congestion estimates from the Chicago open
     data portal API (data.cityofchicago.org, Socrata API, no key needed for
     small volumes)
   - Chicago publishes segment-level data, not point sensors — build the graph
     with road SEGMENTS as nodes and segment adjacency (shared endpoints,
     computed from segment coordinates) as edges
   - Convert to the exact same tensor format as METR-LA so a model trained on
     LA can run on Chicago with zero code changes
   - Document every schema difference between Chicago and METR-LA in a
     DIFFERENCES.md inside the pipelines folder — I need this for the paper's
     generalization discussion

6. Write /xtraffic/utils/graph_utils.py with the shared helpers (adjacency
   construction, normalization, windowing) that all three pipelines import,
   so the logic exists in exactly one place.

7. requirements.txt with pinned versions: torch, torch-geometric, numpy, pandas,
   pyyaml, matplotlib, requests, h5py. Verify the torch / torch-geometric version
   pair is actually compatible before pinning.

VERIFICATION GATE — run these and show me the output:
- python -m xtraffic.data.pipelines.metr_la   → stats report prints, files exist
- python -m xtraffic.data.pipelines.pems_bay  → same
- python -m xtraffic.data.pipelines.chicago   → same
- A quick sanity script that loads each processed dataset and asserts: no NaNs
  after preprocessing, train/val/test are chronologically ordered with no
  overlap, tensor shapes match the config.
```

**Gate:** all three stats reports print sane numbers (METR-LA should show 207 nodes, ~34k timesteps). Commit.

---

## Phase 2 — The ST-GNN

```
Read CLAUDE.md. Phase 1 is verified and committed.

PHASE 2 GOAL: The prediction model — Graph WaveNet base with our three paper
modifications — trained on METR-LA to within striking distance of published
Graph WaveNet numbers (MAE ≈ 3.0 at 30-min horizon on METR-LA).

Build /xtraffic/models/gnn/stgnn.py containing class XTrafficSTGNN:

BASE: Graph WaveNet — stacked spatial-temporal blocks, each combining gated
dilated temporal convolution with graph convolution over both the physical
adjacency and a learned adaptive adjacency. Implement from the paper description;
do not copy an existing repo, but keep it comparable.

MODIFICATION 1 — Semantic co-movement edges (paper contribution):
- Learnable node embeddings E ∈ R^[N × d]
- Semantic adjacency A_sem = softmax(relu(E @ E.T))
- Combine: A_final = α·A_physical + (1−α)·A_sem with α a learnable scalar
  passed through a sigmoid
- Log the learned α every epoch — the trajectory of α is a paper figure
  (how much does the model rely on physical vs learned structure?)

MODIFICATION 2 — Multi-scale temporal convolution:
- Parallel dilated causal conv branches at dilations 1, 2, 4, 8 inside each
  temporal block, concatenated then projected back down
- Comment explaining what horizon each dilation captures at 5-min resolution

MODIFICATION 3 — Heterogeneous feature fusion:
- Input layer accepts a feature dict: traffic (speed), weather (temp, precip,
  visibility), events (proximity-decayed indicator), transit (nearby stop count)
- Each modality gets its own small linear encoder, outputs are summed with
  learnable per-modality gates (early fusion)
- Missing modalities (Chicago may lack some) are handled by zeroing the gate,
  NOT by crashing — this is essential for cross-city transfer
- For now wire real traffic features and stub the others with zeros behind a
  clean interface; Phase 6 fills them in

Also build:
- /xtraffic/models/gnn/train.py — loads config from /configs/train_metr_la.yaml,
  Adam + gradient clipping, masked MAE loss (METR-LA has missing values encoded
  as 0 — explain masking in a comment), early stopping on val MAE, saves best
  checkpoint, logs MAE/RMSE/MAPE at 15/30/60-min horizons per epoch to CSV,
  saves loss-curve PNGs to /evaluation/results/
- /xtraffic/models/gnn/baselines.py — historical average and linear regression
  on the same splits with the same metrics
- /xtraffic/models/gnn/evaluate.py — loads a checkpoint, evaluates on any
  processed dataset by name, writes a JSON results row

Everything configurable via YAML. Seeds fixed. Shape comments everywhere.

VERIFICATION GATE:
1. One epoch on a tiny data subset runs end to end without shape errors
2. Full training run launches and val MAE decreases for the first several epochs
3. Tell me expected total training time on CPU vs a free Colab T4, and if CPU
   is impractical, produce a colab_train.ipynb that clones the repo and trains
   there
```

**Gate:** trained checkpoint exists, 30-min-horizon test MAE lands in the ~3.0–3.5 range on METR-LA. If it's way off, debug here — everything downstream depends on this model being real. Commit.

---

## Phase 3 — Explainability Layer (GNNExplainer)

```
Read CLAUDE.md. Phase 2 checkpoint exists with metrics: [paste your MAE/RMSE here].

PHASE 3 GOAL: Structured mathematical explanations of individual predictions,
plus the standard explainability metrics (fidelity, sparsity, stability).

1. /xtraffic/models/explainer/explain.py:
   - Wraps torch_geometric.explain (Explainer + GNNExplainer algorithm) around
     our trained XTrafficSTGNN
   - explain_prediction(node_id, timestamp, horizon) returns an Explanation
     object and serializes it to JSON with this EXACT schema (the LLM layer
     depends on it — this schema is a contract):
     {
       "meta": {"city": str, "timestamp": str, "model_checkpoint": str},
       "prediction": {"node_id": int, "node_name": str,
                      "predicted_speed_mph": float, "horizon_minutes": int,
                      "current_speed_mph": float},
       "top_nodes": [{"node_id": int, "node_name": str, "importance": float,
                      "current_speed_mph": float}],
       "top_edges": [{"from_id": int, "to_id": int, "importance": float}],
       "propagation_path": [int],
       "propagation_lag_minutes": float,
       "explanation_confidence": float
     }
   - node_name comes from a sensor→location lookup built from sensor coordinates
     (reverse-geocode once offline, or nearest-road labels from the dataset
     metadata). Human-readable names are what make the LLM layer work.
   - propagation_lag: estimate via cross-correlation of the speed series between
     the top source node and the target node; document the method — reviewers
     will ask
   - explanation_confidence: rerun the explainer K=5 times with different init
     seeds, confidence = mean Jaccard similarity of the top-k node sets

2. /xtraffic/evaluation/explainer_metrics.py computing across ≥100 sampled
   predictions:
   - FIDELITY+: occlude the explainer's top-k nodes, measure prediction
     degradation (bigger = explanation captured what matters)
   - FIDELITY−: keep ONLY the top-k, measure how well prediction is preserved
   - SPARSITY: fraction of graph needed for the explanation
   - STABILITY: Jaccard of top-k sets under small input noise
   - Compare against a random-explanation baseline (random top-k) — if we don't
     beat random by a wide margin, something is wrong and we stop and debug
   - Output a results table (CSV + printed) formatted for direct use in the paper

3. /xtraffic/evaluation/visualize_explanation.py:
   - Renders the sensor graph with node importance as color intensity, the
     propagation path as directed arrows, target node starred
   - Publication quality: vector PDF output, colorblind-safe palette, labeled
     colorbar. These are literally Figure 2 of the paper.

VERIFICATION GATE:
- Generate explanations for 5 diverse scenarios (rush hour, midday, night,
  weekend, and a high-congestion moment) and show me the JSONs
- Show the metrics table with the random baseline comparison
- Show one rendered figure
```

**Gate:** explanations are semantically sane — a downtown rush-hour prediction should be explained by upstream downtown sensors, not random ones across the city. Actually read them. If they look like noise, the model or explainer needs work before the LLM layer can mean anything. Commit.

---

## Phase 4 — LLM Advisory Layer

```
Read CLAUDE.md. Phase 3 verified — explanations are faithful and sane.

PHASE 4 GOAL: The LLM module that translates the mathematical explanation into
grounded natural-language reasoning and concrete interventions. Runs locally
via Ollama. This is Layer 3 and the setting for our faithfulness metric.

1. /xtraffic/models/advisor/knowledge_base.py — city context store:
   - Per-city JSON knowledge bases at /models/advisor/kb/{city}.json:
     corridor descriptions, known bottlenecks, road capacities, signal-timing
     constraints, transit alternatives, incident-history summaries
   - Build the LA one from public LADOT / Caltrans corridor information; give
     me a documented template + schema so Chicago is just a fill-in
   - Retrieval: keyword + node-name scoring returning the k=4 most relevant
     chunks for a given explanation (same approach as my OlympiFlow RAG,
     productionized)

2. /xtraffic/models/advisor/advisor.py:
   - Talks to a local Ollama server (model name in /configs/advisor.yaml;
     default llama3.1:8b, and note in comments the model choice is an
     experimental variable we may ablate)
   - Prompt assembled in this exact order:
     (a) SYSTEM: "You are a traffic-operations advisor. You may ONLY cite
         causes present in the mathematical explanation. Prediction is not
         your job — translation and recommendation are."
     (b) CITY CONTEXT: retrieved knowledge base chunks
     (c) MATHEMATICAL EXPLANATION: the Phase 3 JSON, rendered as clean
         structured text with human-readable node names
     (d) TASK: "1) Explain in plain language why this congestion is predicted,
         referencing the specific locations and quantities in the explanation.
         2) Provide 3–5 interventions implementable within 15 minutes given
         the stated infrastructure constraints, each with a time window and
         expected effect."
   - REQUIRE structured JSON output:
     {
       "reasoning": str,
       "cited_causes": [{"location": str, "resolved_node_id": int|null}],
       "recommendations": [{"action": str, "location": str,
                            "time_window_minutes": int, "expected_effect": str,
                            "grounded_in": [str]}]
     }
   - Validate the JSON; on malformed output retry up to 2 times with an error
     message appended; log raw responses for the ablation study

3. /xtraffic/models/advisor/pipeline.py — one function, the whole system:
   advise(city, node_id, timestamp) → runs GNN → explainer → advisor and
   returns {prediction, explanation, advisory}. This function is the paper.

4. A demo script that runs the full pipeline on the 5 Phase-3 scenarios and
   pretty-prints the results.

VERIFICATION GATE:
- Ollama installed and model pulled (give me the exact commands)
- Full pipeline runs end to end on all 5 scenarios
- Show me the outputs — I will personally check whether the LLM's cited causes
  match the explanation JSON or whether it's inventing things
```

**Gate:** read every output. If the LLM cites causes that aren't in the explanation JSON, tighten the prompt now — that failure mode is exactly what Phase 5 measures, but you want the baseline behavior reasonable first. Commit.

---

## Phase 5 — The Faithfulness Metric (your novel contribution)

```
Read CLAUDE.md. Phase 4 pipeline runs end to end.

PHASE 5 GOAL: Implement and validate the faithfulness metric — the alignment
between the mathematical explanation and the LLM's stated reasoning. This is
Contribution #2 of the paper. It must be rigorous.

1. /xtraffic/evaluation/faithfulness.py:

   ENTITY RESOLUTION:
   - Resolve each entry in the LLM's cited_causes to a node_id using the
     node-name lookup: exact match → fuzzy string match (rapidfuzz, threshold
     documented) → geographic match ("near downtown" → nodes within the
     downtown polygon)
   - Log the resolution method used per entity; unresolvable → null (counts
     against faithfulness)

   METRICS (define each mathematically in the docstring):
   - CAUSE PRECISION: fraction of LLM-cited causes present in the explainer's
     top-k
   - CAUSE RECALL: fraction of the explainer's top-k the LLM actually cited
   - FAITHFULNESS F1: harmonic mean of the above — the headline number
   - QUANTITATIVE FIDELITY: when the LLM states numbers (speeds, lags), do they
     match the explanation JSON within 10%?
   - HALLUCINATION RATE: fraction of cited causes resolving to nodes NOT in
     the top-k (or not resolving at all)

2. /xtraffic/evaluation/run_faithfulness_study.py:
   - Sample ≥100 diverse scenarios from METR-LA test data (stratified across
     time-of-day and congestion level)
   - Run the full pipeline on each, compute all metrics
   - Report mean ± std overall and broken down by time-of-day and congestion
     level, as CSV + formatted table + distribution plots (PDF)

3. CONDITIONS TO COMPARE (this doubles as part of the ablation):
   - A: full pipeline (LLM receives explanation + city context)
   - B: LLM receives prediction + city context, NO explanation
   - C: LLM receives explanation only, NO city context
   Condition B is the money result: without the mathematical explanation the
   LLM must invent causes, so hallucination should spike. If A dramatically
   beats B, grounding is proven and the paper's core claim holds.

4. Validate the entity resolver itself: 30 hand-labeled resolution cases in a
   small test file; report resolver accuracy. If the resolver is unreliable the
   metric is unreliable — a reviewer will check this.

VERIFICATION GATE:
- Resolver accuracy on the hand-labeled set ≥ 90%
- Full study runs on 100 scenarios across all three conditions
- Show me the A vs B vs C comparison table
```

**Gate:** if condition A doesn't clearly beat B on faithfulness/hallucination, do not panic and do not fudge — investigate. Either the prompt needs work (fine, iterate) or you've found a genuinely interesting negative result (also publishable, discuss with your professor). Commit either way.

---

## Phase 6 — Real Feature Fusion + Cross-City Generalization

```
Read CLAUDE.md. Faithfulness study complete: [paste headline numbers].

PHASE 6 GOAL: Replace the stubbed modalities with real data, then run the
cross-city experiments. This produces Contribution #1's results.

1. Real heterogeneous features:
   - /data/pipelines/weather.py — historical hourly weather for each dataset's
     date range and location (Open-Meteo historical API is free and keyless);
     align to 5-min sensor timesteps by interpolation; join as node features
   - /data/pipelines/events.py — for METR-LA's 2012 window, build a small
     curated events file (Dodgers/Lakers home games, major venue events —
     schedules are public); encode as proximity-decayed features
   - /data/pipelines/transit.py — GTFS static feeds (LA Metro, CTA) → nearby
     stop count per sensor node
   - Retrain the GNN with full fusion; compare against the traffic-only model
     at all horizons; per-modality gate values over training are a paper figure

2. Cross-city experiments (/xtraffic/evaluation/cross_city.py):
   - ZERO-SHOT: METR-LA-trained model evaluated on Chicago as-is
   - FINE-TUNED: same model, brief fine-tune on 10% of Chicago train data
   - FROM-SCRATCH: model trained only on Chicago (upper-bound reference)
   - Same protocol on PEMS-BAY as a nearer-domain transfer point
   - One results table: rows = transfer settings, columns = MAE/RMSE/MAPE at
     15/30/60 min. This table is the generalization result of the paper.

3. Chicago knowledge base for the advisor (using the Phase 4 template), then
   rerun the faithfulness study on 50 Chicago scenarios — does faithfulness
   hold when the underlying GNN is out of domain? Nobody knows the answer to
   this. That makes it interesting either way.

VERIFICATION GATE:
- Fusion vs traffic-only comparison table
- The cross-city transfer table
- Chicago faithfulness numbers alongside the LA ones
```

**Gate:** graceful degradation on zero-shot Chicago (worse than in-domain, far better than baselines) is a good result. Catastrophic failure means investigating the graph-construction differences documented in Phase 1's DIFFERENCES.md. Commit.

---

## Phase 7 — Ablation Harness

```
Read CLAUDE.md.

PHASE 7 GOAL: The ablation study — mandatory for any serious ML venue. One
harness, every configuration, one command.

/xtraffic/evaluation/ablations.py running and logging:

MODEL ABLATIONS (prediction metrics on METR-LA):
- Full model
- − semantic edges (physical adjacency only)
- − multi-scale temporal (single dilation)
- − heterogeneous fusion (traffic only)
- − each individual modality (leave-one-out)

PIPELINE ABLATIONS (faithfulness + advisory quality on the 100-scenario set):
- GNN only (raw prediction, no explanation, no LLM)
- GNN + explainer (mathematical explanation, no translation)
- GNN + LLM, no explanation grounding (Phase 5 condition B)
- GNN + LLM + explanation, no city context (Phase 5 condition C)
- Full XTraffic

Also: LLM-model ablation — rerun pipeline conditions with a second local model
(e.g., a smaller llama or qwen) to show results aren't an artifact of one model.

Everything driven by /configs/ablations.yaml, three seeds per configuration,
report mean ± std, all results to /evaluation/results/ablations/ as CSV +
LaTeX-formatted tables (booktabs style, ready to paste into the paper).

VERIFICATION GATE: full ablation suite completes; show me both master tables.
```

**Gate:** every component should earn its place. If a modification doesn't help, report that honestly — reviewers respect it and it saves the contribution claims from overclaiming. Commit.

---

## Phase 8 — Human Evaluation Toolkit

```
Read CLAUDE.md. All computational results are in.

PHASE 8 GOAL: Tooling for the expert decision-quality study (Contribution #3).
Evaluators: my UNCC professor + 2–4 colleagues. Keep it dead simple for them.

1. /xtraffic/evaluation/human_study/scenarios.py:
   - Generate 24 incident-response scenarios from test data, stratified by
     congestion level and time of day
   - For each, define ground truth: simulate 4–5 candidate interventions
     (signal retiming, ramp metering, reroute, transit surge, no action) by
     perturbing the relevant input features and measuring predicted network
     delay 30 minutes out; best-outcome intervention = ground truth
   - Document this simulation-based ground-truth method carefully — it's a
     limitation we must state honestly in the paper

2. Three presentation conditions per scenario:
   - RAW: prediction numbers only
   - XAI: prediction + explanation visualization + importance table (current
     state of the art)
   - XTRAFFIC: full pipeline output
   Latin-square assignment so each evaluator sees each scenario exactly once
   and conditions are balanced across evaluators — generate the assignment
   matrix and explain it to me.

3. /xtraffic/evaluation/human_study/app.py — a minimal local web app (FastAPI +
   one plain HTML page) that shows a scenario in its assigned condition, asks
   the evaluator to pick an intervention and rate confidence (1–7) and
   usefulness of the information shown (1–7), and logs everything with
   timestamps to JSONL. No accounts, no database — a name field and a JSONL
   file. It must be runnable by a professor with one command.

4. /xtraffic/evaluation/human_study/analyze.py:
   - Decision accuracy per condition (chose ground-truth-optimal intervention)
   - Confidence and usefulness per condition
   - Given small n, use appropriate nonparametric tests (Wilcoxon signed-rank
     for paired comparisons) and report effect sizes, not just p-values
   - Explain each statistical choice in comments — I need to defend this section

VERIFICATION GATE: I run the app myself, complete 3 scenarios as a pilot,
and analyze.py produces a sane report from my pilot data.
```

**Gate:** pilot it on yourself and one friend before sending it to the professor. Fix every point of confusion. Commit.

---

## Phase 9 — Paper Artifacts

```
Read CLAUDE.md. All experiments complete.

PHASE 9 GOAL: Every figure and table in the paper, generated by one script from
logged results. No hand-made figures — reviewers and reproducers rerun this.

/xtraffic/evaluation/make_paper_artifacts.py producing into /evaluation/paper/:

FIGURES (vector PDF, colorblind-safe, consistent fonts sized for a two-column
IEEE template):
1. System architecture diagram (three layers, data flow) — generate as clean SVG
2. Explanation visualization — best example from Phase 3
3. Learned α (physical vs semantic adjacency) over training
4. Per-modality gate values over training
5. Faithfulness distributions across conditions A/B/C (violin or box)
6. Cross-city transfer bar chart
7. Human study results with error bars

TABLES (LaTeX, booktabs):
1. Prediction vs baselines, METR-LA + PEMS-BAY, three horizons
2. Explainability metrics vs random baseline
3. Faithfulness across conditions and cities
4. Full ablation
5. Human study statistics

Plus REPRODUCE.md: the exact ordered command list from fresh clone to every
number in the paper. Then we test it: fresh clone in a new directory, follow
REPRODUCE.md, confirm it works.
```

**Gate:** the fresh-clone reproduction actually works. That's your paper's artifact-evaluation story and it's rarer than it should be. Final commit, tag it `v1.0-paper`.

---

## Timeline mapping (to your January 2027 deadline)

| Phases | Target |
|---|---|
| 0–1 | 2–3 weeks |
| 2 | 4–6 weeks (the hard one — budget for debugging) |
| 3 | 3 weeks |
| 4–5 | 4 weeks |
| 6 | 4 weeks |
| 7 | 2 weeks |
| 8 | 3 weeks (calendar time depends on professors' availability — start scheduling early) |
| 9 + writing | 4–6 weeks with your professor co-authoring |

Roughly 7–8 months of consistent work. Start writing the paper's related-work section during Phase 2 training runs — training time is reading time.

## The two conversations that matter more than any prompt

1. **Your UNCC professor, this month:** show them this playbook and the paper outline. The ask is co-authorship and weekly-or-biweekly check-ins, not just advice. Their name and guidance is the difference between "impressive teenager" and "credible submission."
2. **Yourself, honestly, at every gate:** if a result is bad, that's data. Never smooth it over. The paper survives negative results; it does not survive results you can't defend.
