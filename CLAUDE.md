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

REAL 100-EPOCH CHECKPOINT LANDED (2026-07-06). metr_la_best.pt is now the real
Colab-trained model (epoch 34, val MAE 2.9026, CUDA/T4) — beats published Graph
WaveNet (3.07). Re-ran the owed gates on it; all numbers below are REAL, not
placeholder:
- PREDICTION (evaluate.py, METR-LA test): 15min MAE 2.82/RMSE 5.46 | 30min MAE
  3.21/RMSE 6.45 | 60min MAE 3.66/RMSE 7.46 | overall MAE 3.166, MAPE 8.78%.
  Improves the 3-epoch placeholder at every horizon (30min 3.52->3.21, 60min
  4.42->3.66); crushes HistAvg 5.15 / LinReg 5.03@30min. Table 1 refreshed.
- PHASE 3 GATE RERUN (explainer_metrics --n 100, k=8, n=90 valid): Fidelity+ 1.393
  vs random 0.843 -> PASS (~1.65x). Stability 0.751. Sparsity 0.039. Fidelity-
  12.93 ~= random 12.84 -> STILL inconclusive on the full ckpt (uniform target
  sampling; signal only separates on congested targets — long-standing, note in
  paper). 6 scenario explanations regenerated + semantically sane. The carried-over
  Phase-3 re-run is now DONE.
- PHASE 5 FULL STUDY (n=93 stratified, llama3.1:8b, A/B/C): A F1 0.725 / halluc
  0.005 / P 0.995 / R 0.593 | B F1 0.090 / halluc 0.828 / P 0.172 / R 0.071 | C F1
  0.728 / halluc 0.005 / P 0.995 / R 0.605. Core claim PROVEN AT SCALE: explanation
  eliminates hallucination (A/C 0.005 vs B 0.828), restores grounding (B precision
  collapses 0.995->0.172). Phase-5 open flag RESOLVED: C ~= A even on the full set
  incl. pm_rush/high-congestion -> city context is orthogonal to faithfulness (it
  aids advisory usefulness, not explanation<->reasoning alignment); C's higher
  quant_fidelity 0.764 is within noise. Clean paper story.
- PHASE 9 artifacts regenerated from the real logs (7 figs + 5 tables). Fig6
  (cross-city) + Table 4 (ablation) correctly still PENDING (Colab-owed).
STILL OWED (all Colab): Chicago Phase-1 pipeline + cross_city transfer table;
Chicago faithfulness study; multi-seed ablation tables. (Fusion retrain: DONE
2026-07-19, see the Phase 6 RESULT block below.)

PHASE 6 FUSION RESULT — LANDED 2026-07-19 (the owed Colab retrain + comparison).
metr_la_fusion_best.pt = epoch 39, val MAE 2.8807, use_sidecars=true, all three
sidecars (weather[0:3] events[3:4] transit[4:5]) loaded at eval time (evaluate.py's
fusion-aware path confirmed in the run log, so the modalities were NOT zeroed).
Command: python3 -m xtraffic.evaluation.fusion_comparison
  horizon   traffic-only        fusion            delta MAE
  15min     MAE 2.809 / RMSE 5.409 / MAPE 7.39%   MAE 2.807 / RMSE 5.389 / MAPE 7.44%   -0.002
  30min     MAE 3.196 / RMSE 6.404 / MAPE 8.91%   MAE 3.192 / RMSE 6.353 / MAPE 8.85%   -0.004
  60min     MAE 3.609 / RMSE 7.389 / MAPE 10.36%  MAE 3.616 / RMSE 7.370 / MAPE 10.28%  +0.007
  overall   MAE 3.145 / RMSE 6.320 / MAPE 8.690%  MAE 3.145 / RMSE 6.285 / MAPE 8.688%
LEARNED MODALITY GATES (sigmoid of fusion.gate_logits, read straight from the ckpt
because train_metr_la_fusion.csv lives Colab-side and comparison.json therefore has
final_modality_gates=null): traffic 0.598 | transit 0.502 | events 0.304 |
weather 0.303. The model kept traffic highest and pushed weather/events DOWN toward
0.30 — i.e. it learned to discount the exogenous feeds.
HONEST VERDICT (report this, do not dress it up): heterogeneous fusion is a WASH on
METR-LA. Every MAE delta (-0.002 / -0.004 / +0.007) is two-to-three orders of
magnitude smaller than the metric itself and far inside seed noise; overall MAE is
identical to 3 d.p. (3.145 vs 3.145). Fusion wins a hair on RMSE (6.285 vs 6.320)
and at 15/30min, loses at 60min. There is NO defensible claim that fusion improves
prediction here. This is a NEGATIVE RESULT and the playbook says report it — it does
not sink Contribution #1, whose real content is the cross-city generalization + the
architecture that ACCEPTS heterogeneous feeds and degrades gracefully when they are
absent (the MOD-3 gate). PLAUSIBLE WHY (state as hypothesis, not fact): METR-LA is
4 months of 2012 LA with little weather variance; the visibility channel is already
known-dead (all-NaN from the archive, Phase 6 note); transit is STATIC in time so it
adds node context but no dynamics; the curated events file is small (14/15 in range).
The learned gates independently corroborate this — the model itself down-weighted
weather and events. NEXT STEP if we want a positive fusion result: a city/date range
with real weather variance, or drop the fusion claim to "architecturally supported,
empirically neutral on METR-LA".
Outputs: evaluation/results/fusion/comparison.{csv,json}. NOT yet reflected in
evaluation/paper artifacts — make_paper_artifacts.py has no fusion table wired in
(Table 4 ablation still PENDING); wire the fusion row when the ablation lands.

HEADLINE CHECKPOINT UPDATED (2026-07-19): metr_la_best.pt is now the LONGER-TRAINED
Colab model — EPOCH 54, val MAE 2.8747 (was epoch 34 / 2.9026). The epoch-34 ckpt
is preserved locally as metr_la_best_epoch34_ARCHIVE.pt (gitignored) because every
committed Phase 3/5/10/11/12/13/16/17/18 result was produced with it.
- PREDICTION on the epoch-54 ckpt (evaluate.py, METR-LA test): 15min MAE 2.809/
  RMSE 5.409 | 30min MAE 3.196/RMSE 6.404 | 60min MAE 3.609/RMSE 7.389 | overall
  MAE 3.145, MAPE 8.69%. Slightly better than epoch-34 at every horizon
  (overall 3.166 -> 3.145); still BEATS published Graph WaveNet 3.07 at 30min
  (3.196 vs 3.07 is worse — see honesty note below).
- HONESTY NOTE (do not overclaim): our 30-min MAE is 3.196, published Graph
  WaveNet METR-LA 30-min MAE is ~3.07, so we do NOT beat it at the 30-min
  horizon. The earlier "beats Graph WaveNet" line compared our OVERALL/VAL MAE
  against their 30-min number — an apples-to-oranges comparison. Correct claim:
  we are COMPETITIVE with Graph WaveNet (30min 3.196 vs 3.07) and crush the
  HistAvg 5.15 / LinReg 5.03 baselines. FIX THIS IN THE PAPER + Table 1 caption.
- FLAGGED: downstream numbers (Phase 3/5/10/etc.) were computed on epoch-34 and
  are NOT invalidated — but they are now one checkpoint behind. Re-running them on
  epoch-54 is OWED if we want every table to reference one checkpoint.

FINAL HEADLINE NUMBERS (Phase 3/5 rows still from the epoch-34 ckpt, 2026-07-06;
prediction row refreshed to epoch-54, 2026-07-19):
- Prediction: overall test MAE 3.145 (val MAE 2.875, epoch 54); 30min MAE 3.196
  -> COMPETITIVE with published Graph WaveNet 3.07 on METR-LA, not better.
- Phase 3 explainability: Fidelity+ 1.393 vs random 0.843 (1.65x), Stability
  0.751, Sparsity 0.039.
- Phase 5 faithfulness (n=93, llama3.1:8b): A F1 0.725/halluc 0.005 |
  B F1 0.090/halluc 0.828 | C F1 0.728/halluc 0.005.
- KEY FINDING (the paper's clean story): the mathematical EXPLANATION is what
  eliminates hallucination and restores grounding (A/C halluc 0.005 vs B 0.828;
  B precision collapses 0.995->0.172). CITY CONTEXT is ORTHOGONAL to faithfulness
  (C ~= A even on congested/high-signal windows) — it does NOT aid explanation<->
  reasoning alignment; it aids ADVISORY USEFULNESS instead. Faithfulness (Contrib
  #2) and advisory quality (Contrib #3) are therefore cleanly separable axes.

Phase 10 IN PROGRESS — simulation-based decision evaluation (REPLACES the Phase-8
human study as Contribution #3; the human study is retired because scheduling
expert sessions was the bottleneck and the model-in-the-loop simulator gives a
fully reproducible, at-scale decision-quality result instead).
File: evaluation/sim_eval.py driven by configs/sim_eval.yaml. One command.
- 500 scenarios stratified 5 tod-bands x 3 congestion terciles (seed 42), METR-LA
  now, Chicago auto-included once a Chicago ckpt exists.
- 5 candidate interventions per scenario (no_action / signal_retiming[target] /
  ramp_metering[congested 1-hop feeders] / reroute[top-edge source relief + edge
  weight -30%] / transit_surge[congested 2-hop neighbourhood]). MODELING NOTE
  (documented in-file): the GNN consumes SPEED, not flow, so a flow/demand cut is a
  proportional SPEED uplift on the affected nodes over the last apply_steps; reroute
  also cuts the top explanation edge (temp buffer swap, like the explainer). Ground
  truth = lowest predicted 30-min network delay; delay reduction = delay(no_action)
  - delay(chosen).
- CALIBRATION (the smoke test earned its keep): the FIRST design — fixed-% uplift
  per action — gave a DEGENERATE ground truth (signal_retiming won ~80% because the
  target is by construction the slowest node and only signal/transit touch it
  directly), so the LLM agents scored WORSE than random (XTRAFFIC 0.10 < RAW 0.17 <
  RANDOM 0.47). FLAGGED to Naren; fixed via Option 1 = EQUAL uplift budget spent
  only on CONGESTED nodes + a per-segment cap (one action can't teleport a segment
  to free-flow) so the best action depends on WHERE congestion is. Locked
  budget=45/cap=24 mph via fast --gt-only sweeps on cached explanations. CONFIRMED
  GT distribution (n=41): transit 34% / ramp 32% / signal 27% / reroute 5% /
  no_action 2% — a real 4-way decision problem, reached by physical mechanisms not
  hand-tuned weights (defensible). --gt-only tallies GT with no LLM for this.
- 3 decision conditions: RANDOM (uniform pick) / RAW (LLM given prediction only) /
  XTRAFFIC (LLM given full pipeline output = prediction + explanation + advisory).
- Metrics per condition, mean +/- std across 3 seeds: accuracy vs ground truth,
  mean delay reduction (aggregate mph-deficit units), decision consistency
  (fraction matching the modal choice within each tod x congestion type).
- Output: LaTeX booktabs table -> evaluation/paper/table_sim_eval.tex, plus CSV +
  JSON to evaluation/results/sim_eval/. Explanations + advisories cached to disk AND
  every decision appended to decisions.jsonl, so the ~8h full run (500 x 3 seeds x 3
  conditions, llama3.1:8b, CPU) is fully RESUMABLE (rerun skips completed decisions).
  --smoke / --limit for quick checks; --budget/--cap for calibration sweeps; --model
  overrides the Ollama model; --mock-llm exercises the harness with no Ollama.
- STATUS: COMPLETE (full run landed 2026-07-17, llama3.1:8b). The stratified
  sampler requests 500 but only windows passing the slowest-valid-sensor validity
  rule survive -> n=444 valid scenarios (NOT 89% of 500 — 444 IS the full set),
  each x 3 seeds x {RAW, XTRAFFIC} = 2664 LLM decisions, all logged + complete.
  REAL RESULT (n=444, 3 seeds): accuracy RANDOM 0.200+/-0.019 < RAW 0.233+/-0.001
  < XTRAFFIC 0.266+/-0.012; delay_reduction RANDOM 13.27 < RAW 15.72 < XTRAFFIC
  19.73 (the cleaner separator — 4-5-way decision so chance ~= 0.20); consistency
  RANDOM 0.294 / RAW 0.998 / XTRAFFIC 0.619 (RAW most monotone in its picks).
  CONTRIB #3 SUPPORTED AT SCALE: the full pipeline improves the simulated planner's
  decisions over prediction-only and random. table_sim_eval.tex + summary.json +
  per_decision.csv finalized. PAPER CAPTION NOTE: report n=444 (not 500); RANDOM is
  its expectation over 200 uniform draws, RAW/XTRAFFIC over the 3 study seeds.

Phase 11 IN PROGRESS — CROSS-CITY x CROSS-MODEL faithfulness (robustness of
Contribution #2: prove the hallucination result is not a METR-LA quirk or a
llama3.1 quirk). File: evaluation/cross_city_faithfulness.py + one config
configs/cross_city_faith.yaml. Reruns the Phase-5 A/B/C study across 3 cities x
2 LLMs -> ONE master table (rows=cities, cols=models; each cell = A-halluc /
B-halluc / F1-gap) -> evaluation/paper/table_cross_city_faith.tex (+ CSV/JSON).
- ZERO-SHOT TRANSFER: PEMS-BAY (325) and Chicago (1020) don't match METR-LA's 207
  nodes, so node-specific params (sem_embed/nodevec/physical_adj) are re-initialised
  at target N and only node-agnostic weights transfer (reuses cross_city.py's
  transfer_weights/save_transferred; builds+caches models/gnn/checkpoints/
  {pems_bay,chicago}_zero_shot.pt). KEY ARGUMENT (in-file + paper): faithfulness is
  PROMPT-STRUCTURAL (does the LLM cite only the explainer's top-k?), so it is valid
  even when zero-shot PREDICTION accuracy is degraded — that is exactly why a
  transferred model is a clean not-a-quirk probe. Had to pre-build the transferred
  ckpt because load_state_dict raises on the 207->325/1020 shape mismatch even with
  strict=False.
- CHICAGO SPARSITY: test split has only 15 windows (< the 20 floor) -> POOL
  train+val+test (76 windows) -> stratified sample yields 33 (>=20, no fallback
  needed). Zero-shot => never trained on Chicago => pooling adds no train/test
  leakage for a faithfulness (not accuracy) measurement. Flagged in-code.
- SECOND MODEL: playbook's qwen2.5:7b AND the tinyllama fallback are BOTH absent
  from local Ollama -> auto-detect ladder picks mistral:7b (different family from
  llama3.1, same size class = strongest available cross-model evidence). Logged on
  the record. Primary runs A/B/C; second runs A/B only (the hallucination gap).
- PEMS-BAY has no KB -> empty city context (Phase-5 already proved city context is
  ORTHOGONAL to faithfulness, so this doesn't distort A/B; no fabricated Bay-Area
  facts). METR-LA/llama3.1 cell REUSES the committed Phase-5 summary (no recompute).
  Resumable: every decision appended to decisions/*.jsonl; PENDING cells for
  missing data/models (honest holes, never a crash). --smoke (mock-LLM, isolated
  *_smoke dir), --mock-llm, --limit, --table-only, --force-transfer.
- STATUS: harness built + SMOKE-VERIFIED end-to-end. Verified: transfer ckpts build
  (pems_bay 325 / chicago 1020: copied 119 node-agnostic, re-init 4 node-specific);
  real explainer runs on all 3 cities incl. transferred models; real llama3.1:8b
  advisor + empty-KB path scores correctly on a transferred PEMS-BAY explanation
  (cond A F1 0.667/halluc 0.000, 3 grounded causes); mock-LLM smoke reproduces the
  A/B contrast on all 3 cities; skeleton table_cross_city_faith.tex emitted (all
  PENDING until the run lands). OWED: the full run (python -m
  xtraffic.evaluation.cross_city_faithfulness) — multi-hour, shares the one local
  Ollama with the running Phase-10 sim, so run it AFTER Phase 10 (resumable).
- PARTIAL REAL RESULTS LANDED (2026-07-19) — 2 of the 6 cells done, run still going
  in a `phase11` screen session (Chicago cells outstanding; the METR-LA/llama3.1
  cell reuses the committed Phase-5 summary). Read straight from
  results/cross_city_faithfulness/summary_<city>__<model>.json:
  * METR-LA x mistral:7b (n=93, in-domain ckpt, conditions A/B):
    A precision 1.000 / recall 0.714 / F1 0.821 / HALLUC 0.000
    B precision 0.190 / recall 0.063 / F1 0.087 / HALLUC 0.810
    -> F1 gap +0.734, hallucination gap -0.810.
  * PEMS-BAY x llama3.1:8b (n=85, ZERO-SHOT transferred ckpt, conditions A/B/C):
    A precision 0.994 / recall 0.650 / F1 0.764 / HALLUC 0.006
    B precision 0.053 / recall 0.025 / F1 0.032 / HALLUC 0.947
    C precision 0.994 / recall 0.616 / F1 0.739 / HALLUC 0.006
    -> F1 gap +0.732, hallucination gap -0.941.
  WHAT THIS PROVES (the point of Phase 11): the Phase-5 grounding result is NOT a
  llama3.1 quirk and NOT a METR-LA quirk. Swap the LLM family (llama3.1 -> mistral)
  and the gap is +0.734; swap the city AND run the GNN ZERO-SHOT (207 -> 325 nodes,
  node-specific params re-initialised) and the gap is +0.732 — nearly identical
  magnitudes on both axes. mistral:7b is if anything MORE disciplined than llama3.1
  on METR-LA (halluc exactly 0.000 across all 93, precision exactly 1.000).
  PEMS-BAY C ~= A again (F1 0.739 vs 0.764, halluc identical) with an EMPTY KB, so
  the Phase-5 "city context is orthogonal to faithfulness" finding reproduces in a
  city where there is no city context at all — the cleanest possible version of it.
  ZERO-SHOT NOTE (keep making it): PEMS-BAY prediction accuracy is degraded by the
  transfer, yet faithfulness is essentially intact — because faithfulness is
  PROMPT-STRUCTURAL (does the LLM cite only the explainer's top-k?), not
  accuracy-dependent. A degraded model that still yields a faithful advisory is
  exactly the not-a-quirk evidence we wanted.
  NOTE: cross_city_faithfulness_master.{csv,json} still reads all-PENDING (dated
  07-07) — the master aggregation is rebuilt at the END of the run, so regenerate
  it (--table-only) once Chicago finishes.

Phase 12 COMPLETE (full n=28 real run landed 2026-07-19) — SHAP BASELINE COMPARISON (answers the reviewer question
"why not just use SHAP?" with an experiment, not an assertion). Two files:
models/explainer/shap_explainer.py (the SHAP explainer artifact) +
evaluation/shap_comparison.py (the A/B/D study + table). Split by the CLAUDE.md
layout rule: explainers under models/explainer, studies under evaluation (same
split as Phase 5's explain.py vs run_faithfulness_study.py). shap==0.44.1 pinned
in requirements.txt (last clean cp39 wheel line; holds numpy 1.26 — never bump to
numpy 2.x under torch).
- SHAP EXPLAINER: shap.KernelExplainer over the N sensor NODES (same unit
  GNNExplainer scores). A coalition is a binary node mask m in {0,1}^N; m[i]=0
  pushes node i's speed to z-space 0 == dataset MEAN == GNNExplainer's exact
  "remove this node" op, so the two explainers share one notion of absence (fair).
  Background = all-masked row; instance = all-present; KernelExplainer decomposes
  the gap across nodes. |SHAP| (normalised to [0,1] like GNNExplainer masks) ranks
  top-k; EDGE importance = product of endpoint node SHAP values on physical edges
  (SHAP scores nodes, not edges — product is the standard node->edge interaction
  proxy). All coalition rows are BATCHED through the model in one forward pass, so
  a prediction is ~seconds not minutes on METR-LA. REUSE: it exposes the same
  explain_target() signature as GNNExplainer, so build_shap_explanation_builder()
  just SWAPS ExplanationBuilder.explainer — every downstream step (top-k, naming,
  propagation path/lag, confidence reruns, schema validation) is identical =>
  byte-compatible schema.py JSON, produced by SHAP. confidence_runs=2 (not 5) since
  each rerun is a full SHAP solve.
- nsamples=50 (assignment's choice, for speed). REAL FINDING that IS part of the
  answer: 50 << 207 nodes, so the SHAP linear solve is underdetermined -> needs
  l1_reg="num_features(k)" (aic/bic CRASH: LassoLarsIC can't estimate noise
  variance when samples<features). Even num_features(20) hits LARS degeneracy and
  only ~13-18 nodes get nonzero SHAP, and the top-k VARIES run to run. So SHAP here
  is slow AND unstable — documented, warnings silenced only around the known LARS
  solve.
- STUDY (evaluation/shap_comparison.py): 3 methods scored by the Phase-5
  faithfulness metric on 30 stratified METR-LA scenarios (per_stratum=2 -> 28):
  A=GNNExplainer (score advisory vs GNNExpl top-k), D=SHAP (advisor gets the SHAP
  explanation via the condition-A prompt; score vs SHAP top-k), B=No explainer
  (Phase-5 condition B; score vs GNNExpl reference). Table rows = {GNNExplainer,
  SHAP, No explainer} x cols {F1, hallucination, precision, recall}, mean+/-std,
  LaTeX booktabs -> evaluation/paper/table_shap_comparison.tex (+CSV/JSON). Also
  logs a diagnostic: GNN-vs-SHAP top-k Jaccard overlap. Resumable (decisions.jsonl),
  --smoke/--mock-llm/--limit.
- SMOKE-VERIFIED end-to-end. SHAP explainer emits schema-valid explanations;
  mock-LLM harness builds the full table; REAL llama3.1:8b 3-scenario run
  (night/low-congestion, heavily caveated n=3): A F1 0.764/halluc 0.000/prec 1.000 |
  D(SHAP) F1 0.853/halluc 0.000/prec 1.000 | B F1 0.000/halluc 1.000. TWO honest
  observations at n=3: (1) BOTH explainers keep the LLM grounded (halluc 0, prec 1)
  — on faithfulness alone SHAP is NOT worse, because the advisor is told to cite
  only what it's shown, so it does, for either explainer. The case for GNNExplainer
  then leans on its OTHER axes (speed, stability, edge/propagation structure). (2)
  GNN vs SHAP top-k overlap = 0.00 — the two explainers pick ENTIRELY DIFFERENT
  nodes (here likely the free-flow regime where there's no real spatial cause). The
  full 30 (incl. congested strata, where explanations actually carry signal and
  SHAP's instability may provoke more confabulation in D) is what decides the real
  verdict — do NOT conclude from n=3.
- FULL RUN COMPLETE (2026-07-19, llama3.1:8b, n=28 = 30 requested, per_stratum=2,
  84 decisions, 0 advisory errors). Strata actually covered: 10 low / 10 medium /
  8 high congestion x 5 tod bands. THE PAPER TABLE (mean +/- std, n=28):
    method                    F1              halluc          precision       recall
    GNNExplainer (ours)  0.742 +/- 0.168  0.018 +/- 0.093  0.982 +/- 0.093  0.616 +/- 0.192
    SHAP (KernelExpl.)   0.692 +/- 0.160  0.018 +/- 0.093  0.982 +/- 0.093  0.554 +/- 0.187
    No explainer         0.048 +/- 0.137  0.857 +/- 0.350  0.143 +/- 0.350  0.031 +/- 0.098
  Quantitative fidelity: A 0.749 | D 0.692 | B 0.377. GNN-vs-SHAP top-k Jaccard
  overlap 0.022 +/- 0.042 (was 0.00 at n=3 — still essentially DISJOINT at scale).
- HONEST VERDICT (the n=3 reading held up; do not overclaim): ON FAITHFULNESS ALONE
  SHAP IS NOT WORSE. GNNExplainer edges it on F1 (0.742 vs 0.692, driven entirely by
  RECALL 0.616 vs 0.554) but hallucination and precision are IDENTICAL to 3 d.p.
  (0.018 / 0.982 — the same single scenario hallucinated under each), and the
  per-scenario winner is split 12 A / 7 D / 9 tie. The +0.05 F1 gap is well inside
  the +/-0.17 std and we have NOT run a paired CI on it — report it as "comparable",
  not "GNNExplainer wins". WHY they tie: the advisor is instructed to cite only what
  it is shown, so it does — for EITHER explainer. What both crush is the no-explainer
  baseline (F1 0.048, halluc 0.857, 24/28 scenarios hallucinated vs 1/28 for A and D)
  — which re-proves the Phase-5 core claim a third way: it is the PRESENCE of a
  mathematical explanation, not which explainer produced it, that eliminates
  hallucination.
- THE REAL ANSWER TO "why not just use SHAP?" is therefore NOT a faithfulness win —
  it is the OTHER axes, and we have measured evidence for each: (1) COST — SHAP needs
  a fresh KernelExplainer solve per prediction with nsamples=50 << 207 nodes; (2)
  INSTABILITY — the underdetermined LARS solve leaves only ~13-18 nodes nonzero and
  the top-k varies run to run (why confidence_runs was cut 5 -> 2); (3) STRUCTURE —
  SHAP scores NODES only, so edges/propagation path come from a product-of-endpoints
  proxy, while GNNExplainer learns an edge mask directly. Make the paper argument on
  those three, and report the faithfulness parity honestly as a finding.
- STRIKING DIAGNOSTIC (worth its own paragraph): the two explainers agree on almost
  NOTHING — top-k Jaccard 0.022, i.e. ~0 shared nodes out of 8 — yet produce advisories
  with the same precision and hallucination. Two mutually-disjoint "explanations" both
  keep the LLM perfectly grounded, because grounding is measured AGAINST WHAT THE LLM
  WAS SHOWN. That is a genuine limitation of the faithfulness metric and we should
  state it: faithfulness measures explanation<->reasoning ALIGNMENT, it does NOT
  certify the explanation is CORRECT. (Same caveat family as the Phase-16
  active-grounding one.) Fidelity+ (Phase 3) is the axis that judges correctness;
  running it on the SHAP top-k is the natural follow-up and is NOT yet done.
- Congestion breakdown (n small per cell, do not over-read): A F1 0.698 low / 0.774
  medium / 0.758 high; D 0.777 / 0.599 / 0.701. SHAP actually BEAT GNNExplainer in
  the low-congestion cell and lost the medium one — consistent with noise, and with
  the recurring "free-flow targets have no real spatial cause" finding.
- Outputs (committed): evaluation/paper/table_shap_comparison.tex (real n=28) +
  results/shap_comparison/{shap_comparison_summary.json,
  shap_comparison_per_scenario.csv, decisions.jsonl, explanations_cache/}.

Phase 13 COMPLETE — CONDITION-A FAILURE TAXONOMY (paper Section 6; closes the loop
on the Phase-4 self-attribution error). File: evaluation/failure_modes.py. Loads
the Phase-5 per-scenario CSV (condition A only = characterising when the BEST
system fails), finds the 7/93 (7.5%) failures (hallucination>0 OR F1<0.5), and
classifies each into one of six categories by a documented first-match PRIORITY:
self-attribution -> free-flow -> geographic-hallucination -> temporal-confusion ->
confidence-mismatch -> other.
- HANDLED GAP: Phase 5 logged metrics but NOT the LLM's cited causes, and 3
  categories need them, so for each failure we REGENERATE condition A on the cached
  explanation (~7 LLM calls, no explainer cost) and recompute per-cause resolution.
  Failure set + F1/hallucination stay the Phase-5 LOGGED numbers (authoritative);
  the regenerated advisory only supplies cited-cause detail. Regen reproduced the
  logged halluc on the one hallucination case (idx 1926), so it's faithful. Cached
  + resumable; --mock-llm verifies the harness with no Ollama.
- KEY RULE FIXES made during the build (both real bugs the smoke caught): (1)
  SELF_ATTRIBUTION requires a cited cause that MISSES top-k AND resolves to the
  target — NOT merely "target in the resolved set", because region-level credit
  means citing the target's REGION can legitimately hit a real top-k node in that
  same region (crediting the neighbour, not self-blame). (2) TEMPORAL_CONFUSION
  requires a stated lag figure next to lead/lag LANGUAGE and excludes the forecast
  horizon — the first cut mislabeled "...will reach the target in 30 minutes" (the
  30-min horizon) as a wrong lag. FREE_FLOW is operationalised as target PREDICTED
  speed >= 50 mph (a direct test of "no real cause"), not window congestion, because
  a missing target sensor (0 mph sentinel) sits in a high-congestion window yet is
  predicted back to free flow.
- REAL RESULT (llama3.1:8b, 7 failures): SELF_ATTRIBUTION 1 (1.1%) · FREE_FLOW 1
  (1.1%) · OTHER=under-citation 5 (5.4%) · GEOGRAPHIC 0 · CONFIDENCE_MISMATCH 0 ·
  TEMPORAL 0. THE STORY: the pipeline essentially never FABRICATES — precision stays
  1.0 on 6/7 failures, 0 geographic hallucinations; its residual failures are
  INCOMPLETENESS (under-citation: names the strongest source, omits the rest ->
  recall collapses, precision intact) plus two rare regime artifacts. The one
  SELF_ATTRIBUTION case (the Phase-4 error, now CONFIRMED but rare) co-occurs with
  free flow: a missing-sensor target predicted at 60 mph, where with no real cause
  the LLM falls back to blaming the target. CONFIDENCE_MISMATCH=0 because the
  explainer was correctly UN-confident on every failure (all conf < 0.8). Outputs:
  evaluation/paper/table_failure_modes.tex (booktabs) + results/failure_modes/
  failure_modes.json (full per-category detail: example in region names only,
  hypothesis, mitigation) + console Section-6 report. Each category ships a
  hypothesised cause + a concrete mitigation (e.g. self-attribution -> drop the
  target from the prompt's cause list + validator rule; free-flow -> gate advisories
  when predicted speed >= free-flow; under-citation -> require the advisory to
  address every top-k source).

Phase 16 COMPLETE (structurally + smoke-gate PASSED) — the ACTIVE GROUNDING LOOP
(the "under-citation" mitigation from Phase 13, now BUILT and MEASURED). Turns the
Phase-5 faithfulness metric from a measurement into a closed-loop MECHANISM. File:
models/advisor/active_grounding.py, one config block (active_grounding: f1_threshold
0.7 / max_rounds 3) in configs/advisor.yaml.
- THE LOOP: run condition A -> score with the SAME Phase-5 metric (score_advisory +
  NodeTable, no second notion of "faithful") -> if F1 < threshold, append a targeted
  correction naming the SPECIFIC explainer top-k nodes the LLM failed to cite (with
  importance + current speed) and re-prompt; repeat up to max_rounds. Records F1/
  precision/recall/halluc at EVERY round = a per-scenario convergence curve.
- REUSE (kept the file small): advisor.advise_condition(exp, "A", extra_instruction=
  correction) for every round; run_faithfulness_study.sample_scenarios +
  build_or_load_explanation for the SAME stratified population + cached explanations
  as Phase 5 (so this runs free on cached explanations).
- FLAGGED default-preserving change to advisor.py: NEW optional `extra_instruction`
  param on advise_condition()/build_prompt_condition(), appended AFTER the TASK.
  Default None => the prompt is BYTE-IDENTICAL to Phase 4/5/15b (verified). No
  earlier behaviour changed.
- WHY IT TARGETS RECALL (honest, follows Phase 13): condition-A failures are
  overwhelmingly UNDER-CITATION (precision ~1, recall collapses, halluc ~0), so
  "you did not address these locations" is the right lever. If a failure is instead
  precision-side (nothing under-cited while F1 < thr), the recall-correction has no
  missed node to name -> the loop STOPS with reason "no_missed_nodes" instead of
  looping uselessly. Honest by construction.
- SMOKE (REAL llama3.1:8b, 5 WORST Phase-5 condition-A failures, --select low-f1):
  mean F1 0.438 -> 0.912 in ONE correction round (gain +0.474), 20% -> 100% reaching
  threshold, mean 0.80 rounds used. CRITICAL HONEST FINDING: precision stays 1.000 at
  EVERY round and hallucination stays 0.000 — the correction lifts recall (0.30 ->
  0.85; n_cited jumps 1-2 -> 5-8) WITHOUT inducing fabrication. The no-needless-
  reprompt gate is demonstrated on the real model too: idx 2917 regenerated at F1
  0.769 >= 0.70 and exited with ZERO corrections. Most-missed region at round 0:
  Glendale / Burbank (12). --mock-llm reproduces the climb with no Ollama.
- CAVEAT (state in the paper): this optimises the advisory TOWARD the explainer
  top-k, so it is an ENFORCEMENT / internal-consistency result, NOT independent
  evidence the explanation is correct. Report as "we BUILT a system that ENFORCES
  grounding in a closed loop", not "we proved the explanation is right".
- Outputs: evaluation/paper/{table_active_grounding.tex, fig_active_grounding.pdf}
  (2-panel convergence figure) + results/active_grounding/{per_round.csv,
  decisions.jsonl, summary.json, decisions_cache/} (resumable, keyed by model).
  FULL RUN COMPLETE (2026-07-17, n=93 stratified, llama3.1:8b, default --select
  stratified). CONVERGENCE (threshold 0.70): round0 F1 0.732 / recall 0.605 /
  halluc 0.005 / 64.5% reached -> round1 F1 0.864 / recall 0.774 / halluc 0.000 /
  96.8% -> round2 0.870 / 0.782 / 0.000 / 98.9% -> round3 0.874 / 0.788 / 0.000 /
  100.0%. mean F1 gain +0.141; mean 0.40 correction rounds used; 100% reached
  threshold; stop reasons {threshold_reached: 93}. FINDINGS AT SCALE (confirm the
  n=5 smoke): (1) the loop WORKS — F1 0.732->0.874, 64.5%->100% grounded, driven by
  RECALL (0.605->0.788), the under-citation lever from Phase 13. (2) NO fabrication
  induced — hallucination 0.005->0.000 as recall climbs (precision-preserving, the
  honest recall-side caveat holds at scale). (3) CHEAP + no needless re-prompting —
  mean 0.40 rounds because 64.5% are already >=0.7 at round0 and exit with zero
  corrections. most-missed regions at round0: NE of Downtown LA (64), Glendale/
  Burbank (56), San Fernando Valley (48). Committed table/fig now the REAL n=93
  (update captions from n=5 -> n=93). Outputs regenerated:
  active_grounding_{per_round.csv, decisions.jsonl, summary.json},
  table_active_grounding.tex, fig_active_grounding.pdf. CAVEAT UNCHANGED: this
  ENFORCES grounding toward the explainer top-k (internal-consistency), not
  independent proof the explanation is correct.

Phase 17 COMPLETE (built + full 20-scenario real run) — COUNTERFACTUAL EXPLANATIONS.
Turns the WHY-explanation into the planner's next question ("what could I have done
differently?"): find the MINIMUM speed uplift that flips a congested target's 30-min
prediction back above free flow, then have the LLM narrate that counterfactual.
Files: models/explainer/counterfactual.py (searcher + record + faithfulness-reference
+ one-scenario demo `main`), evaluation/counterfactual_study.py (the 20-scenario
study + table + traces), configs/counterfactual.yaml (all knobs). FLAGGED
default-preserving addition to advisor.py: render_counterfactual_text /
build_prompt_counterfactual / Advisor.advise_counterfactual — a counterfactual-mode
prompt (SYSTEM->CITY CONTEXT->COUNTERFACTUAL->TASK) that reuses the SAME advisory
JSON contract, so validate_advisory + the Phase-5 faithfulness metric apply
unchanged. advise()/advise_condition() are byte-identical (verified).
- SEARCH (gradient-free, documented): candidate levers = congested top-k critical
  nodes. Stage 1 = uniform uplift sweep (2 mph steps up to 20) until the target
  crosses 35 mph; Stage 2 = greedy per-node minimisation (relax each lever toward 0,
  least-important first, keeping the flip) -> the MINIMAL per-node change. Perturbation
  = the SAME family as the Phase-8/10 simulator (speed uplift on the last apply_steps,
  clipped at free-flow), so one notion of "what an action does" across the project.
- KEY DESIGN DECISION (flagged, calibration earned its keep — like Phase 10):
  perturbing ONLY the upstream top-k barely moves the target (e.g. 34.2->34.2 mph even
  at +40), because this ST-GNN's 30-min forecast is DOMINATED by the target's own
  recent speed (= the Phase-3 Fidelity- result resurfacing). Perturbing the TARGET
  flips it (34.2->56.2). So the candidate lever set INCLUDES THE TARGET BOTTLENECK
  itself (the standard actionable lever: signal retiming / incident clearance = the
  Phase-8/10 target-scope action); upstream levers stay in the set so propagation is
  used where it has leverage, and the study REPORTS how often upstream actually
  contributed. Calibration (target+top-k): flip<=20mph on 45% of congested targets, and
  it tracks depth cleanly — mild (pred 25-35) 93% flip @ median 8 mph, medium (15-25)
  31%, deep (0-15) 17%. Deeply-congested targets are honestly INFEASIBLE (can't be
  prevented by a modest intervention).
- CORRECTNESS FIX (flagged real latent bug): run_faithfulness_study.build_or_load_
  explanation caches by FILENAME PRESENCE only, never checking the checkpoint — 6/93
  cached explanations predate the epoch-34 metr_la_best.pt (same filename overwrote the
  3-epoch placeholder) and are STALE. The study is now staleness-aware: it uses the
  shared Phase-5 cache ONLY when newer than the checkpoint, else rebuilds into its OWN
  cache (never mutating Phase-5 artifacts), and filters congestion on the LIVE model
  prediction (not the cached speed). All 49 congested-by-live targets happen to have
  fresh caches (the 6 stale are free-flow by the live model -> skipped), so the run
  needed zero rebuilds.
- REAL RESULT (n=20 congested METR-LA scenarios, llama3.1:8b): 21 free-flow targets
  skipped ("no counterfactual needed"); VALIDITY 40% (8/20 flip within 20 mph); mean
  budget 10.0 +/- 4.58 mph; mean 1.0 node changed; ALL 8 valid counterfactuals
  TARGET-ONLY (0/8 needed upstream — the honest Fidelity- finding, now a measured
  number); mean implied lead 11.9 min. NARRATIVE FAITHFULNESS to the counterfactual
  (Phase-5 metric scored vs the required-change nodes): F1 1.000 / precision 1.000 /
  recall 1.000 / HALLUCINATION 0.000 across all 8 — the LLM narrates exactly the nodes
  the counterfactual requires and invents nothing. (Prompt fix that earned its keep:
  the counterfactual TASK must explicitly request 3-5 recommendations or llama3.1
  returns 1 and fails the advisory contract -> empty advisory -> F1 0; fixed, then all
  8 pass.) Note: "cite the target" is CORRECT here (the counterfactual says CHANGE the
  target), the opposite of the Phase-13 self-attribution error in the WHY context.
- Outputs: evaluation/paper/table_counterfactual.tex (booktabs) +
  results/counterfactual/{counterfactual_per_scenario.csv, counterfactual_summary.json,
  example_traces.txt/.json (3 full prediction->explanation->counterfactual->narrative
  chains)}; resumable per-scenario cache keyed by model. Verified: `--mock-llm`
  reproduces the harness (mock cites required changes -> F1 1.0); `--smoke` prints the
  full chain for each scenario; single-scenario demo at
  `python -m xtraffic.models.explainer.counterfactual`.
- HONEST CAVEATS (state in the paper): (1) the counterfactual concentrates on the
  bottleneck because upstream leverage is limited under this model (Fidelity-), so it
  reads as "the minimum bottleneck intervention that prevents the congestion", not an
  upstream-cascade story — report that plainly. (2) validity is only 40% at the
  assignment's 20 mph budget; deeply-congested targets are unpreventable by a modest
  intervention (honest, and a finding). OWED (feasible locally, not Colab): a larger
  CONGESTED-ONLY sweep (>=50 mild/medium targets, or stratified by congestion tercile)
  to tighten the validity-vs-depth curve for the paper — the committed table is n=20
  and its caption says so; rerun via `python -m xtraffic.evaluation.counterfactual_study`
  (resumable, shares the one local Ollama with the Phase 10/11/12 queue).

Phase 18 COMPLETE (built + full 20-scenario real run) — UNCERTAINTY-AWARE
EXPLANATIONS. Promotes the Phase-3 SCALAR confidence (mean Jaccard over K reruns)
to a PER-NODE distribution: run GNNExplainer K=10 times, tally each node's top-k
appearance frequency, and label CORE (>=80% of runs -> assert), PERIPHERAL (20-80%
-> hedge), NOISE (<20% -> omit); inject that split into the LLM with an honesty
instruction and MEASURE whether it actually hedges. Files: models/explainer/
uncertain_explainer.py (K-run tally, classify_tier, mean_pairwise_jaccard stability,
uncertainty block, hedging_analysis, no-LLM demo), evaluation/uncertainty_study.py
(the A-vs-U study + table + stability plot + traces), configs/uncertainty.yaml.
FLAGGED default-preserving addition to advisor.py: advise_uncertain /
build_prompt_uncertain / _SYSTEM_UNCERTAIN / _TASK_UNCERTAIN — an uncertainty-mode
prompt (SYSTEM+honesty-clause -> CITY CONTEXT -> tiered EXPLANATION -> TASK) reusing
the SAME advisory JSON contract, so validate_advisory + the Phase-5 faithfulness
metric apply unchanged. git diff = 132 insertions / 0 deletions -> advise()/
advise_condition()/Phase 4-17 are byte-identical (verified).
- ADDITIVE SCHEMA: the uncertainty block attaches as an extra top-level
  "uncertainty" key on the Phase-3 explanation; validate_explanation checks required
  keys are PRESENT (doesn't forbid extras), so the Phase-4 contract is untouched.
- REUSE: the K runs ARE the SAME explain_target(seed=k) solves Phase 3 already does
  for the confidence scalar (cost model unchanged); base explanation, KB retrieval,
  stratified sampler, and metric are all existing Phase-3/4/5 machinery. Staleness-
  aware base-explanation loader (Phase-17 pattern): trusts the shared Phase-5 cache
  only when newer than the checkpoint, else rebuilds into its own cache.
- SCORING DECISION (flagged): BOTH conditions scored against the SAME deterministic
  single-run (seed-0) top-k, so any F1/hallucination difference is attributable
  purely to the uncertainty FRAMING, not a moved goalpost. A is therefore byte-
  identical to Phase-5 condition A. Honest caveat: U omits noise-tier nodes it was
  shown are unstable, so a seed-0 node it correctly drops counts against U's recall
  (small; core dominates) -> U's faithfulness vs its OWN shown set is ALSO logged as
  a secondary diagnostic. Both A and U retrieve identical city context (U's exp keeps
  the original top_nodes), so the ONLY variable is the tiered rendering + honesty clause.
- REAL RESULT (n=20 stratified METR-LA, llama3.1:8b, K=10): mean explanation
  stability 0.427 (range 0.155-0.75 — it GENUINELY varies scenario to scenario,
  which is the whole motivation), mean core/peripheral/noise = 3.6/11.0/7.0.
  FAITHFULNESS A (deterministic) vs U (uncertainty): F1 0.704 -> 0.737 (+0.033,
  essentially flat); precision 1.000 -> 0.896; recall 0.562 -> 0.637; HALLUCINATION
  0.000 -> 0.104. HONEST FINDINGS: (Q1) uncertainty framing is ~F1-NEUTRAL — it
  trades a little precision for higher recall (it mentions MORE of the top-k). (Q2)
  it does NOT reduce hallucination below Phase-5's ~0.5% — it slightly INCREASES it,
  because inviting the LLM to MENTION peripheral (uncertain) causes means it
  sometimes cites a node outside the deterministic top-k; but this is CONCENTRATED
  (only 4/20 U advisories hallucinated at all; 16/20 stayed at 0, high-variance mean).
  (Q3 — the key question) YES the real LLM hedges: condition-U core/peripheral hedge
  rates 0.23/0.39 (gap +0.13) vs condition-A ~0/0.03; U hedged peripheral MORE than
  core in 9/19 scenarios where the gap is defined (mean gap +0.125). So the framing
  makes the model measurably more epistemically honest (hedges uncertain causes ~1.7x
  more than confident ones, in ~half of cases) at a small faithfulness cost — NOT
  every scenario, and it sometimes just OMITS peripheral entirely instead of hedging
  (e.g. idx 5479). The method also CORRECTLY collapses in free flow: 1/20 scenarios
  had 0 core nodes (stability 0.16) -> "nothing is confidently a cause", the recurring
  no-real-spatial-cause finding, now surfaced by the uncertainty tiers themselves.
- Outputs: evaluation/paper/{table_uncertainty.tex (booktabs A-vs-U + hedging),
  fig_uncertainty_stability.pdf (per-scenario stability histogram)} +
  results/uncertainty/{uncertainty_per_scenario.csv, uncertainty_summary.json,
  example_traces.txt (core-vs-peripheral treatment per condition), decisions_cache/}
  (resumable, keyed by model). Verified: --mock-llm reproduces the whole harness with
  no Ollama; single-scenario demo at `python -m xtraffic.models.explainer.uncertain_explainer`.
  Committed table/fig are the REAL n=20 (caption says n=20).

Phase 19 STEP 1 COMPLETE (2026-07-19) — CROSS-DOMAIN DATA PIPELINE (IEEE 14-bus
power grid). Domain chosen by Naren = power grid (the playbook said "discuss with
professor first"; that discussion is OWED and is about SCOPE, since the data now
exists). Files: data/pipelines/power_grid.py (the pipeline),
data/pipelines/DIFFERENCES_POWER_GRID.md (the honest cross-domain note),
models/advisor/kb/power_grid.json (12-chunk KB), power_grid blocks in
configs/data.yaml + configs/advisor.yaml. requirements: pandapower==2.14.11 (last
line supporting py3.9 AND numpy<2 — never bump under torch).
- THE CONTRACT HOLDS. X[6984,12,14,2] / Y[6984,12,14] + adjacency/scaler/node_meta/
  stats, byte-identical structure to METR-LA. sanity_check.py now runs the SAME
  assertions over power_grid and PASSES. VERIFIED SEPARATELY: the UNMODIFIED
  XTrafficSTGNN (389K params) forward-passes on it, trains (L1 0.738->0.192 in 15
  steps), and the MOD-3 gate handles the power sidecar being None. Zero changes to
  models/gnn, models/explainer, models/advisor, evaluation/faithfulness.
- DATA HONESTY (the line to use in the paper): the TOPOLOGY and the PHYSICS are
  REAL, the DEMAND is SYNTHETIC. Every voltage comes from a real Newton-Raphson AC
  power flow (pandapower runpp) on the standard case14; we fabricate demand, never
  a voltage. Demand = two-peak daily curve (normalised to unit 24-h mean, so the
  published 259.0 MW/73.5 MVAr load vector IS the daily mean) x per-bus class
  (residential/commercial/industrial) x +/-1.5h phase jitter x weekend 0.88 x local
  demand surges x AR(1) noise (phi 0.90, 3%, NOT white). 10,000 steps @5min = 34.7
  days, ~95 s to build, seed 42, 0 divergences.
- MODALITY MAP: bus=sensor, branch=road link, |Z| per-unit=road distance, voltage
  pu=speed, undervoltage=congestion, N-1 branch outage=incident. Core X keeps
  EXACTLY 2 channels (voltage, time-of-day) so the input contract is unchanged;
  active+reactive power ride along as a Phase-6-style SIDECAR (mod_power).
- THREE DECISIONS THAT EARNED THEIR KEEP (all measured, all flagged in-file):
  (1) ADJACENCY. Reusing the traffic Gaussian kernel on |Z| CRUSHES 5 of the 20 REAL
  branches below weight 0.02 (trafo 4-9 -> 1.5e-8) — it is built to SELECT edges
  from a dense distance matrix, but a grid's 20 branches ARE the topology. Switched
  to normalised branch ADMITTANCE 1/|Z| (the canonical Y-bus graph operator);
  weights now span 0.079-1.000. Chicago already set the precedent that adjacency is
  a per-DOMAIN data decision. "gaussian" kept in config for the ablation.
  (2) enforce_q_lims=True is ESSENTIAL — without it the 4 voltage-controlling
  machines pin their setpoints at any demand and voltages barely move (degenerate
  task). With reactive limits enforced, per-bus daily swing goes ~0 -> 0.02-0.06 pu.
  (3) N-1 SECURITY SCREEN on outage candidates: never a bridge (trafo 7-8 islands
  bus 8), never a branch that diverges at the 99th PERCENTILE of system demand
  (lines 1-2, 2-3, trafo 5-6). REAL BUG FOUND+FIXED: screening at the ABSOLUTE max
  knocked out every branch and produced a dataset with ZERO contingencies.
- DEGENERACY CHECK (the thing that would have quietly invalidated the whole phase).
  Built a diagnostic into stats.json. v1 of the dataset (daily curve + noise only)
  had neighbour-R2 0.653 vs non-neighbour 0.558 = only 1.17x — the graph barely
  mattered, so an explainer would have had nothing real to find. Adding LOCAL DEMAND
  SURGES (Poisson, ~1 per bus per 2 days, 1.3-2.0x — realistic: industrial start-up,
  EV cluster) lifted it to 0.525 vs 0.288 = 1.82x. Params chosen physically, NOT
  tuned to the metric.
  HONEST DOMAIN DIFFERENCE TO REPORT: PC1 variance share is 0.917 here vs 0.387 on
  real METR-LA (0.436 when METR-LA is subsampled to 13 sensors, so it is NOT a
  node-count artifact). A 14-bus grid IS more collinear than a city — voltage is
  near-linear in loading, traffic congestion is a local threshold phenomenon. Note
  the two diagnostics move in OPPOSITE directions (surges raise PC1 while nearly
  doubling the neighbour ratio) — do not steer on PC1 alone.
- FINAL DATASET: 14 buses / 20 branches / 9977 samples (6984/998/1995), voltage
  0.792-1.090 pu, 0.0% missing, 76 outage events (11.8% of steps), 195 demand
  surges, 1.04% of steps with any bus <0.95 pu. Bus 14 is the weakest point
  (std 0.0165, min 0.792) exactly as the topology predicts. Bus 1 (slack) is
  near-constant (std 5.8e-5) — it is the voltage reference; reported as
  near_constant_buses.
- SEPARATE REAL BUG FOUND+FIXED in configs/data.yaml: the `chicago:` dataset block
  was mis-indented UNDER `transit:`, so it parsed as cfg["transit"]["chicago"] while
  chicago.py reads cfg["datasets"]["chicago"] -> KeyError. The Chicago Phase-1
  pipeline could not run at all. Moved back under `datasets:`, contents unchanged.
- OWED (Phase 19 Step 2, do NOT start before the professor conversation on scope):
  (a) NodeNamer does not read node_meta["names"] — it only knows the lat/lon path,
  and power-grid buses have no geography (latlon=null). A ~3-line fix, deliberately
  NOT made: Chicago's node_meta ALSO carries an ignored "names" key, so the change
  would alter Chicago's names and therefore the committed Phase-11 numbers. Flagged
  in DIFFERENCES_POWER_GRID.md §6. (b) train the model, run the Phase-5 A/B
  conditions, report the hallucination gap next to METR-LA. (c) metrics/plot labels
  are hard-coded "mph" — numbers will be right, LABELS WILL LIE at ~0.005 pu.
  [ALL THREE NOW DONE — see Phase 19 STEPS 2-3 below.]

Phase 19 STEPS 2-3 COMPLETE (2026-07-19) — the CROSS-DOMAIN GROUNDING RESULT.
The A/B hallucination gap HOLDS on the IEEE 14-bus power grid. Files:
evaluation/power_grid_faithfulness.py (the study) + configs/power_grid_faith.yaml
+ models/advisor/domains.py (domain vocabulary) + evaluation/
{verify_traffic_unchanged.py, golden_traffic.json, resolver_labels_power_grid.json}.

PREDICTION (Step 1 answer, and it is NOT a clean win — report it honestly):
power_grid_best.pt = epoch 6, best val MAE 0.001704 pu (21 epochs, early-stopped,
MPS). TEST: overall MAE 0.002912 pu | 15min 0.001896 | 30min 0.002779 | 60min
0.004381. Baselines: PERSISTENCE overall 0.002829 | HistAvg 0.003660 | LinReg
0.003044. So the model BEATS HistAvg and LinReg but LOSES TO PERSISTENCE overall
(0.00291 vs 0.00283, ~3% worse) and loses badly at 15min (0.00190 vs 0.00149). It
only WINS at 60min (0.00438 vs 0.00475). WHY (state as hypothesis): bus voltage at
5-min resolution is far more autocorrelated than traffic speed, so "predict the
last value" is a very strong short-horizon baseline; the graph model only earns its
keep as the horizon grows. DO NOT claim the ST-GNN "works" on the power grid on the
strength of this — claim only what the phase is actually for, which is the
faithfulness transfer. Also note gate_power stayed exactly 0.5000 for all 21 epochs
=> the mod_power sidecar was NOT used in this run (use_sidecars=false); the model is
voltage+time-of-day only.

WHAT HAD TO CHANGE (all default-preserving, all FLAGGED, all gated):
- models/advisor/domains.py (NEW): per-domain vocabulary (unit, node noun, stress
  noun, operator role). The Phase-4 prompt is hard-coded traffic; pointed at a grid
  it said "You are a traffic-operations advisor ... Current speed: 0.88 mph" for a
  BUS VOLTAGE. That is a CONFOUND, not cosmetics: an LLM told a substation is doing
  0.88 mph invents ramp metering, and we would have scored OUR OWN prompt's
  confusion as the model's hallucination. TRAFFIC profile strings are verbatim
  Phase-4, so all traffic prompts are byte-identical.
- faithfulness.NodeTable: crashed on power_grid (list(node_meta["latlon"]) with
  latlon=null) and its ladder was geographic-only. Added a non-geographic ladder
  (bus id -> electrical ZONE -> fuzzy zone), gated on `table.geo`.
- validate_resolver.py: optional --labels (default = the METR-LA file, unchanged);
  non-METR-LA label sets write resolver_accuracy_<city>.json so the committed
  Phase-5 gate result can never be overwritten.
- NOT changed: XTrafficSTGNN, explain.py, schema.py, the advisory contract, the
  metric math, the prompt STRUCTURE, and the grounding instruction (identical
  wording across domains — that is what makes the transfer a real test).
- schema keys stay *_speed_mph (frozen Phase-3 contract); power-grid explanations
  carry additive meta.units="pu" + meta.note so a human reading the JSON isn't misled.

GATES BOTH PASSED:
- TRAFFIC REGRESSION GATE (new, evaluation/verify_traffic_unchanged.py): 12
  committed METR-LA explanations x 5 rendered artifacts, the 3 prompt constants,
  and the resolver on METR-LA + Chicago are BYTE-IDENTICAL to pre-Phase-19 (golden
  captured from the pre-refactor code via git stash). Run it after ANY advisor or
  resolver edit; --capture re-baselines and is deliberately a separate, loud flag.
- RESOLVER GATES: METR-LA still 29/30 = 96.7% (unchanged). Power grid 22/22 =
  100.0% on a new hand-labeled set. Half that set is REFUSALS on purpose (branch
  names "line 13-14"/"transformer 4-9", "135 kV", "bus 99", leftover "Downtown LA")
  — a resolver that matched those would erase condition B's hallucination and
  fabricate the result.

SCENARIO SAMPLING (the subtlest part; the traffic rule does NOT port). Phase 3/5
pick the SLOWEST valid sensor, which works because road sensors share one free-flow
reference. Buses do not: bus 3 normally sits at 1.008 pu and bus 8 at 1.089 pu, so
"lowest voltage" is bus 3 in 1733/1995 test windows whether or not anything is
wrong. Copying it would have produced a 20-scenario study of ONE bus and the
flatness would have looked like a finding. Instead: sag RELATIVE TO EACH BUS'S OWN
train-split mean, target = argmax, window qualifies at >= 0.010 pu; buses with
train std < 0.001 pu are ineligible as targets (excludes ONLY bus 1, the slack /
voltage reference, whose "deviation" is numerical noise yet is the argmax in 1081
windows). Result: 344 stressed windows, 20 sampled round-robin over
(tod-band x severity tercile), targets spread over buses 3, 4, 9, 10, 13, 14 and
severity 6 mild / 8 moderate / 6 severe.

TOP-K = 4, NOT 8 (fairness, not tuning). k=8 of METR-LA's 207 nodes = 3.9% of the
graph; k=8 of 14 buses = 57%, i.e. most of the network would be "a valid cause" and
the hallucination gap would be structurally compressed. k=4 of the 13 eligible
non-target buses = ~31%. We do not merely assert this is fair — we MEASURE the
residual chance advantage (below).

THE RESULT (n=20, llama3.1:8b, top_k=4, conditions A/B + 2 chance controls):
  condition            precision  recall   F1      hallucination
  A  full pipeline     0.950      0.825    0.869   0.050
  B  no explanation    0.000      0.000    0.000   1.000
  CHANCE random bus    0.276      0.193    0.225   0.724
  CHANCE random zone   0.687      0.706    0.653   0.313
Paired bootstrap 95% CI (10k iters, seed 42), per-scenario differences — ALL NINE
EXCLUDE 0: A-B halluc -0.950 [-1.000,-0.875], F1 +0.869 [+0.767,+0.955]; A-RANDOM_bus
halluc -0.674 [-0.727,-0.598]; A-RANDOM_zone halluc -0.263 [-0.378,-0.145], F1 +0.216
[+0.095,+0.326].

WHY THE CHANCE ROWS ARE NOT OPTIONAL (the honesty that makes this publishable):
a 14-bus graph with k=4 is FAR easier to hit by guessing than 207 sensors with k=8,
and one electrical zone spans up to 7 of 14 buses. A random ZONE citer scores F1
0.653 with no model at all. So "A scored F1 0.869" is NOT comparable to METR-LA's
0.725 and must never be quoted as if it were. The defensible claims are the ones
that survive the controls: A beats random-zone on hallucination by -0.263 (CI
excludes 0) and on precision by +0.263, and beats random-bus by -0.674.
CITATION GRANULARITY makes the same point concrete: 62% of A's citations were
zone-level (generous credit), 38% bus-level, 0% unresolved.

CONDITION B IS WORSE THAN CHANCE (1.000 vs 0.724 random-bus) — the most
interesting finding, and it is not a bug. B never once hit the top-k across 20
scenarios. 63% of its citations DON'T RESOLVE AT ALL because it names BRANCH
equipment — "line 13-14", "transformer 4-9", "shunt capacitor bank at bus 9" —
which are EDGES, not nodes, and the metric scores nodes. The remaining 34% are
bus-level citations that miss. Read plainly: stripped of instance-specific
evidence, the LLM falls back on GENERIC DOMAIN KNOWLEDGE recited from the KB, which
is confidently wrong in a way random guessing is not. That is a cleaner and more
alarming failure mode than METR-LA's B (0.828) and worth its own paragraph.

VERDICT: the core claim of Contribution #2 TRANSFERS ACROSS DOMAINS. Same model
family, same explainer, same metric, same grounding instruction, a graph that is
not a road network — and the explanation still eliminates hallucination (0.050 vs
1.000, CI excludes 0), including against a chance control tuned to the new graph's
geometry. Framing earned: "mathematical GNN-explanation grounding is a
domain-agnostic mechanism for reducing LLM hallucination."

HONEST CAVEATS (state all of these):
1. ALIGNMENT, NOT CORRECTNESS — the same limitation Phase 12 found. Faithfulness
   measures whether the LLM cites what it was SHOWN. It does not certify the
   explanation is right. Sharper here: the top-k buses are DIRECT electrical
   neighbours of the target only 17.5% of the time, BELOW the 22.0% chance rate for
   picking 4 of 13 at a mean degree of 2.86. So there is currently NO positive
   evidence these explanations are physically correct. Fidelity+ is the axis that
   would settle it and it is NOT YET RUN on power_grid (explainer_metrics.py picks
   targets with the same degenerate lowest-voltage rule and needs the relative-
   deviation fix first). This is the single most important follow-up.
2. Explanation stability is LOWER than traffic: mean confidence 0.519 (range
   0.143-1.000) vs METR-LA's 0.751.
3. n=20 on one LLM. Phase 11's cross-model axis (mistral) was not rerun here.
4. Demand is synthetic (DIFFERENCES_POWER_GRID.md §5) — real topology, real AC
   power flow, fabricated load.
5. The prediction model loses to persistence overall (see above).
6. Phase 17/18 prompt modes (counterfactual, uncertainty) are NOT domain-aware —
   they still hard-code mph and would mislabel on the grid. Fine today because
   neither was run on power_grid; must be routed through domains.py before they are.
Outputs: evaluation/paper/table_power_grid_faith.tex, results/power_grid_faithfulness/
{summary.json, per_scenario.csv, decisions_cache/}, results/power_grid_explanations/
(20 schema-valid explanations), results/resolver_accuracy_power_grid.json.

Phase 20 COMPLETE (2026-07-20) — the ACTIVE GROUNDING LOOP RUN CROSS-DOMAIN.
Phase 19 proved the MEASUREMENT (A/B hallucination gap) transfers to a power grid.
This phase asks whether the MECHANISM does: does the Phase-16 closed loop still
converge on a graph that is not a road network, with NO retuning? File:
evaluation/active_grounding_power_grid.py (the driver). The loop itself
(models/advisor/active_grounding.run_active_grounding) is IMPORTED AND CALLED AS
IS — control flow, stopping rules, scoring call and correction targeting all
untouched. f1_threshold 0.70 / max_rounds 3 are read from the SAME advisor.yaml
block METR-LA used, and are deliberately NOT overridable in the driver so the
"no domain tuning" claim cannot be quietly broken.
- FLAGGED default-preserving edit to active_grounding.py: an OPTIONAL `domain`
  param on build_correction_block() + run_active_grounding(). Phase 16 predates
  domains.py, so the correction block still had THREE traffic hard-codes — "the
  predicted CONGESTION", "currently {} MPH", "do NOT invent new SENSORS, ROADS,
  INCIDENTS" — i.e. it told llama3.1 a substation was "currently 1.03 mph". That is
  the exact confound domains.py exists to prevent. `domain=None` -> TRAFFIC ->
  BYTE-IDENTICAL to Phase 16 (pinned by --verify-unchanged, which inlines the
  Phase-16 bytes). The committed n=93 METR-LA numbers are untouched; the traffic
  regression gate (verify_traffic_unchanged) also still PASSES.
- BOTH VARIANTS RUN (per Naren's call, so the confound becomes a measurement):
  UNMODIFIED (domain=None, mph text) and DOMAIN_ROUTED (domain=POWER_GRID, pu text).
- RecordingAdvisor proxy captures every (correction sent -> advisory returned) pair
  WITHOUT touching the loop — the loop only ever uses .model/.advise_condition.
REAL RESULT (n=20 Phase-19 undervoltage scenarios, llama3.1:8b, top_k=4):
  domain / variant              round0  round1  round2  round3  reached%
  METR-LA traffic (n=93)         0.732   0.864   0.870   0.874    100.0
  grid, unmodified (n=20)        0.892   0.973   0.973   0.973    100.0
  grid, domain_routed (n=20)     0.899   0.932   0.932   0.932    100.0
  -- engaged sub-population (below threshold at round 0) --
  grid unmodified   (n=4)        0.558   0.964   0.964   0.964    100.0
  grid domain_routed(n=2)        0.667   1.000   1.000   1.000    100.0
Hallucination/round: grid unmod 0.037->0.013 | dom 0.025->0.025 | METR-LA
0.005->0.000. Mean rounds used: grid 0.20/0.10 vs METR-LA 0.40.
ANSWER TO THE HEADLINE QUESTION: YES, one round. Every single engaged scenario in
both variants converged in EXACTLY 1 correction round (n_rounds set == {1}), and
the engaged subset went 0.558 -> 0.964 — nearly identical to the Phase-16 traffic
low-F1 smoke (0.438 -> 0.912 in 1 round). 100% reached threshold in every row.
THE CEILING EFFECT (report this, do NOT sell 0.892 as a convergence win): only
4/20 (unmodified) and 2/20 (domain_routed) scenarios ever ENGAGED the loop, because
condition A is already at/above 0.70 at round 0 on the grid — 9 of the first 12
traces scored EXACTLY F1 1.000. Mechanism: k=4 of 14 buses, and the prompt lists
those 4 explicitly, so the LLM simply echoes all four back -> recall 1.000. On
METR-LA (k=8 of 207, long region names) it under-cites instead (recall 0.605). So
the grid's higher round-0 F1 is a SMALL-GRAPH ARTIFACT, not better grounding — the
same k/N inflation Phase 19 flagged with its RANDOM_zone chance row (F1 0.653).
The loop correctly declines to fire (Phase-16 "no needless re-prompting" gate).
NOISE FLOOR — the measurement that makes the variant contrast interpretable, and
it inverted the naive reading. Round 0 uses extra_instruction=None in BOTH variants,
so both get a BYTE-IDENTICAL prompt; any round-0 difference is therefore pure LLM
nondeterminism (advisor runs at temp 0.1, not 0). Measured: round-0 |diff| 0.0782
(std 0.132, differing on 8/20) vs final |diff| 0.0411 (std 0.063, 6/20). The final
"unmodified beats domain_routed by 0.041" is SMALLER than the no-correction noise
=> NO DETECTABLE EFFECT of the correction vocabulary. Null result, not equivalence
(n_engaged is 4 and 2). Logged as summary.variant_contrast.
THE STRIKING TRACE (idx 167, unmodified, in the traces JSON): the correction block
said "causes of the predicted CONGESTION", "currently 1.03 MPH", "do NOT invent new
SENSORS, ROADS, INCIDENTS" — and llama3.1 corrected PERFECTLY anyway: F1 0.400 ->
1.000, all 4 buses cited, hallucination 0.000, and its own reasoning stayed in
correct grid vocabulary ("absorbing too much reactive power", "1.090 pu"). It
silently ignored the traffic labels. HYPOTHESIS (state as such): the correction's
job is purely to NAME THE MISSED NODES; domain semantics come from the base prompt,
which is already domain-routed. idx 678 additionally went halluc 0.500 -> 0.000, so
the correction can REDUCE fabrication, not just raise recall.
TWO CORRECTIONS TO THE PHASE-19 RECORD (both real, both flagged):
 (1) The 20 scenarios are where condition B hallucinated 1.000; condition A — which
     is what this loop operates on — already averaged F1 0.869 there. They are NOT
     a "low faithfulness" set.
 (2) NONDETERMINISM IS MATERIAL. idx 1912 scored F1 0.857 here but 0.333 in the
     committed Phase-19 run — same scenario, same model, same prompt. The round-0
     noise floor above quantifies this at |diff| ~0.08 mean, up to 0.14 on single
     scenarios. Any per-scenario Phase-19 claim (incl. the Phase-13-style failure
     taxonomy) should be treated as one draw, not a fixed property.
Outputs: evaluation/paper/{table_active_grounding_crossdomain.tex,
fig_active_grounding_crossdomain.pdf (both domains on one convergence axis)} +
results/active_grounding_power_grid/{active_grounding_power_grid_traces.json (full
per-round correction prompts + LLM answers), ..._per_round.csv, ..._summary.json,
decisions_cache/} (resumable, keyed by variant+model). A quarantined
--demo-threshold flag exists to force the correction to fire for inspection; it
writes to its own cache and NEVER writes paper artifacts. It was NOT needed — all
reported traces come from the un-forced 0.70 threshold.
OWED: rerun on a second LLM (mistral:7b) — Phase 11's cross-model axis was not
repeated here, so "domain-agnostic" currently rests on one model; and the engaged-n
is small enough that a larger stressed-scenario sweep would tighten it.

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

---

# EXTENDED PHASES

> Added 2026-07-12. These extend the original 0–9 playbook (and the in-progress
> 10–13 work logged in the status section at the top of this file) with one round
> of reviewer-driven robustness work plus four new research directions.
>
> **NOTHING in this section is built yet.** Only empty/stub scaffolds exist under
> the canonical directories so the structure is ready. Build them one at a time
> and gate each exactly like the earlier phases. Intended order:
> **15b → 16 → 17 → 18 → 19.**
>
> **NUMBERING NOTE (flagged honestly).** The playbook proper stops at Phase 9;
> Phases 10–13 live only in the status log above; there was never a written
> Phase 14 or Phase 15. "Phase 15b" is named as requested — a bundle of targeted
> improvements that slot in *before* Phase 16 — and the new work then continues
> at Phase 16. If we later want a clean paper numbering we can renumber, but the
> build order is what matters.

---

## Phase 15b — Targeted Improvements (reviewer feedback)

**Location:** updates to EXISTING files — `evaluation/run_faithfulness_study.py`,
`evaluation/faithfulness.py`, `models/advisor/advisor.py`, and the faithfulness
configs. This is why 15b has **no stub file of its own**: it is edits to code we
already built, not a new module. (Per the CLAUDE.md rule, any change to Phase-4/5
behaviour must be FLAGGED in-file and here, not slipped in silently.)

Three improvements, all inside the Phase-5 faithfulness study:

1. **Rich-context condition `C_RICH`.** Phase 5 already proved plain city context
   (condition A vs C) is ORTHOGONAL to faithfulness. `C_RICH` tests whether a
   *fuller* context block (more KB chunks retrieved, corridor+incident history
   expanded) changes that — a stronger test of the "context doesn't move
   faithfulness" claim. Add as a new condition alongside A/B/C, same scoring
   against the explainer top-k, same stratified sample.
2. **Contradictory-context condition (`D_CONTRA`).** Feed the advisor city context
   that DISAGREES with the explanation (e.g. names a different corridor as the
   known bottleneck) and measure whether the LLM stays faithful to the math or
   gets pulled toward the false context. This is the sharp stress test of
   grounding: if hallucination stays ~0 under contradictory context, grounding is
   robust; if it spikes, we've found a real limitation to report.
   - ⚠️ NAMING CLASH (flag): Phase 12 already uses **condition D = SHAP** in a
     *different* study/table (`shap_comparison.py`). These don't collide in code
     (separate files), but in the PAPER "condition D" would be ambiguous. Suggest
     labelling this one `D_CONTRA` (or `E`) in prose so the two never blur.
3. **Bootstrap confidence intervals on all A/B/C (and new-condition) comparisons.**
   Right now the study reports mean ± std. Add nonparametric bootstrap 95% CIs on
   each per-condition metric AND on the A−B / A−C differences (resample scenarios
   with replacement, seed 42, ~10k iters). This is what turns "A beats B" into "A
   beats B, 95% CI on the gap excludes 0" — the rigor a reviewer expects and cheap
   to add since every per-scenario metric is already logged.

**Contribution:** hardens Contribution #2 against the obvious reviewer pushback
(is the context result real? is the grounding robust to bad context? are the
gaps statistically meaningful?) without any new model or LLM — pure evaluation.

**Verification gate:** the faithfulness study runs with the new conditions +
bootstrap CIs on the existing cached decisions (resumable, no full recompute);
the A/B/C table gains CI columns; `D_CONTRA` produces a sane hallucination number;
resolver still ≥90%.

**STATUS: COMPLETE (2026-07-17).** All three parts done. (1) Bootstrap 95% CIs on
A/B/C landed earlier at n=93 (table_faithfulness_ci.tex): A-B gap on every metric
excludes 0 (F1 +0.635*, halluc -0.823*), A-C spans 0 (F1 -0.003) => grounding gap
is statistically real, city context is not. (2)+(3) NEW conditions run at full
n=93 (llama3.1:8b, --conditions A,C_RICH,D_CONTRA --out-tag 15b, so committed
A/B/C artifacts untouched; 16 cached decisions reused). REAL RESULT:
  A       (n=93): precision 0.978 / recall 0.609 / F1 0.733 / halluc 0.022
  C_RICH  (n=93): precision 0.989 / recall 0.581 / F1 0.710 / halluc 0.011
  D_CONTRA(n=93): precision 1.000 / recall 0.630 / F1 0.754 / halluc 0.000
- C_RICH vs A: F1 gap 95% CI [-0.012, +0.058] SPANS 0; halluc gap CI [-0.011,
  +0.032] SPANS 0 -> a FULLER context block does NOT move faithfulness. Strengthens
  the Phase-5 'city context is ORTHOGONAL to faithfulness' claim with a CI, not just
  a point estimate.
- D_CONTRA (contradictory context = names a DIFFERENT real corridor as the
  bottleneck): hallucination stayed 0.000, faithful (halluc==0) on 93/93 scenarios;
  A-D_CONTRA halluc gap CI [+0.005, +0.043] EXCLUDES 0 in the direction A > D_CONTRA
  (i.e. false context did not raise hallucination — if anything D_CONTRA was even
  more disciplined). KEY FINDING: GROUNDING IS ROBUST TO ADVERSARIAL CONTEXT — the
  LLM trusts the mathematical explanation over a contradictory city-context block
  and never gets pulled to the decoy corridor. This is a positive robustness result,
  NOT a limitation. Outputs: faithfulness_{per_scenario,summary}_15b.{csv,json} +
  faithfulness_distributions_15b.pdf.

---

## Phase 16 — Active Grounding Loop

**Location:** `xtraffic/models/advisor/active_grounding.py` (stub created).

**What it does:** closes the loop on the faithfulness metric. After the advisor
generates an advisory, AUTOMATICALLY compute its faithfulness F1 (reuse
`evaluation/faithfulness.py`). If F1 < 0.7, build a TARGETED correction prompt
that names the specific explainer top-k nodes the LLM failed to cite, and
re-prompt. Repeat up to 3 rounds or until F1 ≥ threshold. Record the F1 (and
hallucination) at each round = a per-scenario CONVERGENCE CURVE.

**Design sketch (build it readable, 3.9-compatible):**
- Reuse `Advisor.advise_condition(...)` for the initial + corrective prompts and
  the faithfulness resolver/metrics for scoring — no new LLM plumbing.
- Correction prompt = the Phase-4 prompt + an appended block: "You did not address
  these locations from the explanation: {missed top-k node names}. Revise your
  reasoning and recommendations to account for them; still cite ONLY what appears
  in the explanation."
- Config knobs (new `configs/active_grounding.yaml` or a block in `advisor.yaml`):
  `f1_threshold=0.7`, `max_rounds=3`. No magic numbers in code.
- Output: per-scenario convergence curves (round → F1/halluc) to
  `evaluation/results/active_grounding/` as JSONL + CSV, plus an aggregate
  "fraction reaching threshold within k rounds" table/figure.

**Contribution:** upgrades the story from "we MEASURED that grounding helps" to
"we BUILT a system that ENFORCES grounding in a closed loop" — an actionable
mechanism, not just a metric. Honest caveat to state: this optimises the advisory
toward the explainer's top-k, so report it as enforcement/consistency, not as
independent evidence the explanation is correct.

**Verification gate:** on a handful of scenarios (incl. a known low-F1 one from
the Phase-13 failure set), the loop measurably raises F1 across rounds and the
convergence curve is logged; a scenario already ≥0.7 exits in one round (no
needless re-prompting).

---

## Phase 17 — Counterfactual Explanations

**Location:** `xtraffic/models/explainer/counterfactual.py` (stub created).

**What it does:** instead of explaining WHY congestion is predicted, find the
MINIMUM perturbation to critical (top-k) nodes that FLIPS the target prediction to
free flow — then have the LLM narrate the counterfactual: "If signal timing on the
Glendale feeder had been eased ~8 minutes earlier, the cascade would not have
reached Downtown."

**Design sketch:**
- Candidate node set = the explainer's top-k (already the causal frontier), so we
  search a small space, not all N.
- Perturbation = a speed uplift on those nodes over the last few input steps (same
  mechanism family as the Phase-8/10 intervention simulator — reuse it), searched
  for the SMALLEST uplift (and/or fewest nodes) that pushes the target's 30-min
  prediction back above a free-flow threshold. Gradient-guided or small grid/greedy
  search; document the method — reviewers will ask.
- Emit a counterfactual record (which nodes, how much, the implied lead time from
  the propagation lag) and feed it to the advisor via a counterfactual-mode prompt.

**Contribution:** makes the system genuinely ACTIONABLE — planners see not just the
cause but what they could have done differently, with a concrete magnitude and
lead time. Complements Contribution #3 (decision quality).

**Verification gate:** on a congested scenario, a counterfactual is found (target
flips to free flow) with a plausibly small perturbation, and the LLM's narration
matches the counterfactual record (no invented nodes/magnitudes). On a free-flow
target, it correctly reports "no counterfactual needed."

---

## Phase 18 — Uncertainty-Aware Explanations

**Location:** `xtraffic/models/explainer/uncertain_explainer.py` (stub created).

**What it does:** run GNNExplainer K=10 times with different random
initialisations, then classify each node by how often it lands in the top-k:
CORE (>80% of runs), PERIPHERAL (20–80%), NOISE (<20%). Inject this uncertainty
distribution into the LLM prompt so the advisory becomes epistemically honest:
"East LA corridor confirmed in 9/10 runs. The I-10 connector is uncertain
(4/10 runs)."

**Design sketch:**
- Reuse `explain.py`'s GNNExplainer / `ExplanationBuilder`; this is essentially the
  Phase-3 `explanation_confidence` machinery (mean Jaccard over K reruns) promoted
  from a single scalar to a PER-NODE frequency, so K=10 solves already fit the cost
  model.
- Extend the schema (additively, don't break the Phase-4 contract) with a per-node
  `stability_tier` / `runs_in_topk` field; the advisor prompt renders it as
  confidence language.
- The LLM is instructed to hedge on peripheral nodes and never assert noise nodes.

**Contribution:** epistemically honest explanations that reflect GENUINE model
uncertainty — directly answers "how do I trust a single explainer run?" and pairs
naturally with the Phase-16 active-grounding loop (enforce citation of CORE nodes,
allow hedged peripheral ones).

**Verification gate:** across K=10 runs the core/peripheral/noise split is sensible
on a congested scenario (a real upstream corridor is CORE), the schema stays
Phase-4-valid, and the advisory's confidence language matches the tiers (asserts
core, hedges peripheral, omits noise).

---

## Phase 19 — Cross-Modal Grounding Transfer

**Location:** `xtraffic/data/pipelines/second_domain.py` (stub created).

> ⚠️ **DISCUSS WITH PROFESSOR BEFORE BUILDING.** This phase needs faculty guidance
> on domain choice and scope. Start with DATA EXPLORATION ONLY — do not train
> anything until the domain is agreed.

**What it does:** prove the hallucination-reduction result transfers to a COMPLETELY
DIFFERENT domain with NO architecture changes. Step 1 is exploration: identify which
second-domain graph dataset is most accessible and legally/technically usable —
candidates: IEEE 14-bus power grid, MIMIC-III patient flow, or SupplyGraph supply
chain. Then window it into the SAME `X[·,T,N,C] / Y[·,T,N]` tensor contract, train
the SAME `XTrafficSTGNN`, run the Phase-5 A/B conditions, and report the
hallucination gap.

**Design sketch:**
- Exploration deliverable first: a short DIFFERENCES-style note per candidate
  (access method, licence, graph definition, node/edge semantics, whether it fits
  our tensor contract) so the professor can pick with full information.
- Reuse the whole downstream stack unchanged — that's the entire point (the
  cross-city pipelines already prove the contract is portable; this pushes it
  cross-DOMAIN).
- A domain with no natural KB → empty city context (Phase 5 already showed context
  is orthogonal to faithfulness, so A/B is unaffected — no fabricated facts).

**Contribution:** if the hallucination gap holds across domains, the claim
generalises from "works on traffic" to **"mathematical GNN-explanation grounding is
a DOMAIN-AGNOSTIC mechanism for eliminating LLM hallucination."** That is the
strongest possible framing of Contribution #2 — but only if the professor agrees the
domain and scope are defensible.

**Verification gate:** (exploration) a written comparison of the three candidate
datasets + a recommendation; (build, only after sign-off) the second domain loads
into the tensor contract, the shared model trains to a non-trivial baseline, and
the A/B hallucination gap is reported alongside the METR-LA numbers.
