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
NEXT: Phase 2 — the ST-GNN (needs `pip install -r xtraffic/requirements.txt`,
torch + torch-geometric; training likely on Colab T4).

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
