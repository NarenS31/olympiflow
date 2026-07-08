# XTraffic — Lab Notebook

Running log of what was built, the numbers, and what surprised us. 3–5 lines per
session. This becomes the paper's experiments section almost for free.

---

## 2026-07-01 — Phase 0 + Phase 1 (data pipelines)
- Established project memory (Phase 0) by prepending the XTraffic context to
  `CLAUDE.md`; kept the full build playbook below it.
- Built the canonical `/xtraffic` scaffold on top of the existing OlympiFlow repo
  (OlympiFlow kept as a base + Ollama-RAG reference for Phase 4).
- Wrote reproducible pipelines: `metr_la.py`, `pems_bay.py` (shared `_h5_common.py`),
  and `chicago.py` (segment-graph, Socrata API). All emit one tensor contract:
  `X [samples,12,N,2]`, `Y [samples,12,N]`, plus adjacency/scaler/node_meta/stats.
- Key design decisions: chronological (never shuffled) 70/10/20 split to avoid
  temporal leakage; z-score scaler fit on TRAIN speeds only; Gaussian-kernel
  adjacency (DCRNN Eq. 10) for LA/Bay, shared-endpoint binary graph for Chicago.
- Schema divergences documented in `data/pipelines/DIFFERENCES.md` for the paper's
  generalization section.
- **Gate PASSED.** Numbers: METR-LA 207 nodes / 34,272 steps / 8.11% missing;
  PEMS-BAY 325 nodes / 52,116 steps / 0.003% missing; Chicago 1,020 segment-nodes
  / 99 steps / 66% missing (portal feed was stale to ~2026-04-30, so only a few
  hours came back — enough to prove the tensor contract, not to train on).
- **Surprise:** the DCRNN `.h5` files 404'd on Zenodo; the record actually hosts
  CSV. Switched to CSV, which also let us drop the pytables dependency entirely.
- **TODO next session (Phase 2):** `pip install -r xtraffic/requirements.txt`
  (torch + torch-geometric), then build the ST-GNN. Also widen the Chicago
  Socrata window before it's usable for cross-city training (Phase 6).

## Phase 2 — the ST-GNN (2026-07-03)
- Built `XTrafficSTGNN` (395K params) in **pure PyTorch** — Phase 2 needs no
  torch-geometric (that's a Phase 3 dep); the model is just dilated convs + a
  diffusion GCN written as an einsum over `[N,N]` supports.
- Base = Graph WaveNet (gated dilated TCN + adaptive-adjacency GCN, residual+skip).
  Three modifications wired in and each is doing something measurable:
  - MOD 1 semantic co-movement edges: `A_final = a·A_phys + (1-a)·A_sem`,
    `a = sigmoid(alpha)`. **alpha drifted 0.49 → 0.39 over 3 epochs** — the model
    is actively choosing the *learned* graph over physical adjacency. That
    trajectory is a paper figure basically for free.
  - MOD 2 multi-scale temporal: parallel causal dilations [1,2,4,8] per block
    (≈10/20/40/80 min of history), left-padded so T stays 12 — no receptive-field
    length bookkeeping.
  - MOD 3 heterogeneous fusion: per-modality encoder + learnable sigmoid gate; an
    absent modality is passed as `None` and simply contributes nothing (this is
    what will let the LA-trained model run on Chicago unchanged in Phase 6).
- Masking subtlety: missing readings are real-0 mph, which z-scores to a sentinel
  ≈ -2.79. Loss + all metrics reconstruct real mph and mask `|real| > 0`, so we
  never train on the data-collection artifact. Metrics reported in mph to compare
  with published GWN.
- **Numbers (METR-LA test, best checkpoint, only 3 epochs):**
  MAE 3.53 | 15min 2.98 | 30min 3.52 | 60min 4.42.
  Baselines: HistAvg 5.15, LinReg 5.05. Model beats both clearly; 30-min MAE is
  inside the 3.0–3.5 target. Published GWN is 2.69/3.07/3.53 — 3 epochs got us
  most of the way; a full 100-epoch Colab run should close it.
- **Surprise / lesson:** Apple MPS is ~40 min/epoch because the graph-diffusion
  einsum falls back to CPU on MPS. Local full training is impractical → wrote
  `models/gnn/colab_train.ipynb` for a free T4. (The 40k-second epoch timers in
  the log are inflated by laptop sleep between sessions, not real compute.)
- **Gate PASSED** (shapes clean end-to-end; train loss ↓ monotonically; val MAE ↓
  overall to 3.23; beats baselines; 30-min in range).
- **TODO next session (Phase 3):** `pip install torch-geometric==2.5.3`, then the
  GNNExplainer wrapper + the explanation-JSON schema (the Phase 4 LLM contract).

## Phase 3 — explainability layer (2026-07-03)
- **Deviation, flagged:** did NOT install/wrap `torch_geometric.explain`.
  Reimplemented the GNNExplainer algorithm (Ying et al., NeurIPS 2019) in pure
  PyTorch against our own model — our forward takes a *modality dict* + *dense
  adjacency*, not PyG's `(x, edge_index)`, so wrapping would mean writing a shim
  anyway. Cite as a faithful reimplementation. Keeps the no-torch-geometric line.
- Learns two soft masks by freezing the model and optimising only the masks:
  a NODE feature mask (scales each node's z-scored speed toward 0 = the mean =
  "remove this node's info") and an EDGE mask over the physical adjacency
  (installed via a temporary buffer swap so grads flow, restored in a `finally`).
  Loss = prediction-preservation + sparsity + entropy (pushes masks toward 0/1).
- `schema.py` pins the explanation JSON as a hard contract with a validator — the
  Phase-4 LLM depends on this shape, so it's the interface, not a detail.
  Human-readable node names come from offline lat/lon→LA-region tagging
  (`node_names.py`) — no external geocoder (reproducibility + no-API-calls rule).
  Propagation path = BFS over the physical graph; lag = window cross-correlation
  (temporal precedence, NOT a causal claim — documented); confidence = mean
  Jaccard of the top-k node set over K=4–5 reruns with different inits.
- **REAL BUG I caught by reading outputs (the gate works!):** every scenario came
  back "now 0.0 mph". Cause: I tested sensor validity in *z-space* (`|z|>eps`),
  but the missing sentinel is 0 mph in *real* space (≈ −2.79 in z-space). So the
  "slowest" node was always a *missing* sensor. Fixed to test mph, matching
  `utils.metrics`. After the fix, targets are genuinely congested and lag/
  confidence actually vary.
- **Key finding:** node-occlusion fidelity is only meaningful where there's
  congestion to explain. A free-flowing 65 mph prediction has no spatial cause,
  so occluding neighbours does nothing — uniform target sampling averages the
  signal to noise (Fidelity+ 0.24 vs 0.18 random, marginal). Restricting to the
  slowest valid sensor per window (= Phase-5's congestion stratification):
  **Fidelity+ 0.496 vs 0.244 random (2.0×, PASS), Stability 0.796.**
- **Honest weak spot:** Fidelity− ≈ random (9.26 vs 9.28). On the 3-epoch model
  the prediction is spread across many nodes, so top-8 isn't clearly *sufficient*.
  Not fudging it — expect it to separate on the full 100-epoch checkpoint; re-run
  the gate then.
- 6 scenario explanations written, all schema-valid; congested cases read sanely
  (East LA 20→15 mph explained by nearby congested Glendale/Burbank sensors,
  5-min lag). Figure 2 rendered as vector PDF (geographic map, viridis, arrows).
- **Gate PASSED** (Fidelity+ clearly beats random; explanations semantically sane
  on congested targets; schema validated), with the Fidelity− caveat noted.
- **TODO next session (Phase 4):** Ollama advisory layer, consuming the
  explanation JSON contract. Also re-run this gate on the full-model checkpoint.

## Phase 4 — LLM Advisory Layer (Ollama, Layer 3)
- Built the advisor that turns the Phase-3 explanation JSON into (a) plain-language
  reasoning and (b) 3–5 concrete interventions. Runs 100% locally via Ollama.
- **Retrieval:** productionized the OlympiFlow keyword scorer. KB chunks in
  `kb/la.json` carry `region_tags` keyed to the *same* region strings
  `node_names.py` emits, so an East-LA/Glendale explanation pulls the
  Glendale/Burbank + San Fernando Valley + bottleneck chunks — verified it does.
- **Prompt order (fixed):** SYSTEM (may ONLY cite causes in the math explanation)
  → CITY CONTEXT (retrieved KB) → MATH EXPLANATION (rendered with node *names*)
  → TASK. Temperature 0.1 — we want faithful translation, not creativity.
- **Advisory has its own JSON contract** ({reasoning, cited_causes[],
  recommendations[]}), validated with a 2-retry self-correct loop that feeds the
  parse/shape error back to the model. Ollama `format=json` cuts malformed output.
- **Gate:** ran the full GNN→explainer→advisor chain on all 6 committed scenarios
  against `llama3.1:8b`. Every output schema-valid; 6 advisories saved.
- **Read the outputs (the real gate):** grounding is strong — cited causes
  overwhelmingly resolve to the explanation's top nodes, quoted speeds come from
  the injected top-8 (not invented), recommendations cite real LA infrastructure
  (I-5/SR-134 metering, ATSAC on Sunset/Santa Monica, Metro Expo/Wilshire diversion).
- **Honest slippage (what surprised me):** on free-flowing-cause windows (midday)
  the LLM mislabels 56 mph as a "cause of congestion"; occasionally emits a null
  resolution or cites the target node as its own cause. Not fixing by over-tuning
  the prompt — this IS the baseline Phase 5's faithfulness metric exists to measure.
- **TODO next session (Phase 5):** entity resolver + cause precision/recall/F1 +
  hallucination rate across conditions A/B/C on ≥100 scenarios. Also still owe the
  Phase-3 gate re-run on the full 100-epoch checkpoint.

## Phase 5 — the Faithfulness Metric (Contribution #2) (2026-07-03)
- Built `evaluation/faithfulness.py`: resolve each LLM `cited_causes[i].location`
  back to sensor node_ids, then score against the explainer's top-k. Metrics,
  each defined in its docstring: **cause precision** (of what the LLM claimed, how
  much was really in the explanation), **recall** (of the explanation's top-k, how
  much the LLM covered), **faithfulness F1** (headline), **quantitative fidelity**
  (do stated speeds/lag match within 10%), **hallucination rate** (= 1 − precision).
- **Design choice, documented:** we resolve INDEPENDENTLY of the LLM's own
  `resolved_node_id` — trusting the model's self-report would let it grade its own
  homework. And because METR-LA node names are region-tagged and the LLM cites at
  region granularity, a citation resolves to the *set* of nodes in that region and
  counts as a hit iff that set intersects top-k (a deliberate, stated generosity).
- **Entity resolver** = an escalating ladder: exact sensor id → exact region →
  fuzzy (rapidfuzz if installed, else stdlib difflib fallback, threshold 82) →
  geographic gazetteer ("downtown", "the Valley" → region). Every citation logs
  which rung fired.
- **Resolver validated** on 30 hand-labeled cases (`resolver_labels.json`):
  **96.7% (29/30), PASS (gate ≥90%)**. The one miss is honest — "just north of
  downtown" resolves to Downtown via the `downtown` alias instead of "N of
  Downtown LA". Regions absent from this METR-LA subset (Santa Monica, LAX, Long
  Beach) correctly resolve to null.
- **Conditions A/B/C** added to the advisor WITHOUT touching Phase-4 behavior
  (`advise()`/`build_prompt()` are untouched = condition A). B = prediction only,
  no explanation (and city context is retrieved from a prediction-only view so
  top_nodes can't leak); C = explanation but no city context.
- **`run_faithfulness_study.py`** samples ≥100 scenarios stratified over 5
  time-of-day bands × 3 congestion terciles (slowest-valid-sensor target, seed 42),
  builds each explanation once (cached), runs A/B/C, aggregates mean±std overall +
  by band + by congestion, writes CSV/JSON + boxplot PDF.
- **Smoke run (`--limit 6`, end-to-end vs the real checkpoint + `llama3.1:8b`)** —
  the money result is already stark: **A F1 0.817, halluc 0.000 | B F1 0.067,
  halluc 0.833 | C F1 0.820**. Strip the explanation (B) and the LLM invents
  causes — precision collapses 1.00→0.17. **A ≫ B → grounding proven.** (Caveat:
  `--limit` truncates the front of the stratum list, so these 6 are all
  night/low-congestion; the ≥100 gate run covers all strata.)
- **Gate status:** resolver ≥90% PASS; metric computes on real advisories; A/B/C
  runs end-to-end; A clearly beats B. Still owe the **full ≥100-scenario run**
  (`python -m xtraffic.evaluation.run_faithfulness_study`, ~1–2 h local) for the
  paper table, and the Phase-3 gate re-run on the full 100-epoch checkpoint.
- **Surprise:** even the 3-epoch model produces near-perfect precision under
  condition A — the prompt's "only cite causes in the explanation" constraint plus
  `format=json` is doing real work. The interesting variance is in *recall* (the
  LLM cites ~70% of top-k, not all 8), which is the honest headroom to discuss.

---

## Phase 6 — Real feature fusion + cross-city (session: infrastructure)

- **Built the modality SIDECAR mechanism.** Phase-6 feeds never touch the Phase-1
  tensors — each writes `processed/<ds>/mod_<name>.npz` shaped `[S,12,N,C]`, windowed
  and split *identically* to traffic X (shared `_modality_common.py`), z-scored on
  TRAIN only. `make_fusion_loaders` concatenates present sidecars → per-batch `M`;
  absent feed → `None` → MOD-3 gate zeroes it. Fusion is opt-in (`use_sidecars` in
  `train_metr_la_fusion.yaml`); base config stays traffic-only. Verified both paths.
- **All three real feeds RAN against live sources**, every sidecar = 23974/3425/6850
  (matches traffic exactly): weather (Open-Meteo ERA5, 19 grid cells), events
  (curated 2012 venues, proximity-decayed, 14/15 in range), transit (LA Metro GTFS,
  12,355 stops → mean 24.6 stops/node).
- **REAL BUG found + fixed:** fusion smoke → `train_loss=nan`. Traced to the weather
  **visibility** channel: Open-Meteo's *archive* API doesn't serve `visibility`, so
  it came back all-NaN and one NaN poisons every gradient. Fixed defensively at the
  shared writer — non-finite cells fill with the channel's train mean (~0 after
  z-score), fully-missing channels collapse to 0 and the gate ignores them. Visibility
  is now effectively a dead channel; temp+precip carry the weather signal.
- **Also fixed a footgun I hit:** `--smoke` was overwriting the real `metr_la_best.pt`.
  Smoke now writes `*_smoke.pt`. (Cost: the old 3-epoch placeholder ckpt was lost
  before I caught it — regenerate from the Colab run.)
- **Cross-city (`cross_city.py`):** zero-shot / fine-tune / from-scratch. Honest
  transfer boundary — only node-agnostic conv weights transfer; the learned graphs
  (sem_embed, nodevec, physical_adj) are re-initialised at the target node count.
  Compiles; needs Chicago processed + a source ckpt to produce the table.
- **Chicago advisor:** made `node_names.py` city-aware (real Chicago region boxes,
  auto-detected from `node_meta['dataset']`, LA behaviour unchanged); added
  `kb/chicago.json` (CDOT/IDOT/CTA). Resolver re-validated: still 96.7% PASS.
- **Gate status:** infra complete + locally verified; owed on Colab — full fusion
  retrain + comparison table, Chicago transfer table, Chicago faithfulness study.
- **Surprise:** the archive API silently dropping `visibility` is exactly the kind of
  real-data gap the paper's "heterogeneous fusion must not crash on missing modalities"
  design is *for* — the same None-gate mechanism that handles Chicago's absent feeds
  also absorbed a missing channel within LA. The architecture's robustness earned itself.

---

## Phase 6 wrap + Phase 7 — comparison notebook & ablation harness (session)

- **Fusion Colab notebook** (`colab_fusion_train.ipynb`): builds the traffic tensors
  + all three sidecars, trains the traffic-only baseline (also *restores* the lost
  `metr_la_best.pt`), trains the fusion model, and prints the fusion-vs-traffic-only
  table. Made **`evaluate.py` fusion-aware** first — a fusion checkpoint must be
  evaluated *with* its sidecars, else we'd zero the modalities it trained on.
  `fusion_comparison.py` lays the two checkpoints side by side + reports the learned
  modality gates.
- **Phase 7 ablation harness** (`evaluation/ablations.py`, `configs/ablations.yaml`):
  one command runs every configuration.
  - Added default-preserving architecture switches to the model: `use_semantic`
    (MOD 1) and `use_multiscale` (MOD 2). Confirmed shapes: `no_multiscale` drops
    to 259K params (single dilation) vs 395K full. Leave-one-out via a new
    `sidecar_modalities` config key.
  - Pipeline ablations reuse the Phase-5 A/B/C conditions; gave the faithfulness
    study a `--model` (LLM-model ablation) and `--out-tag` flag.
  - **Smoke-verified end to end:** `--model-only --smoke` trained all 7 model
    variants via subprocess, evaluated each, and wrote `model_ablations.{csv,tex}`
    (LaTeX booktabs, mean±std). Numbers are throwaway (1 epoch); the *structure* is
    what the smoke proves.
- **Gate status:** Phase 7 structurally complete + smoke-passed. Owed: the real
  multi-seed model runs and the Ollama pipeline runs (need a checkpoint + Ollama) for
  the paper tables — Colab-scale, same as everything else.
- **Reflection:** the ablation switches were cheap to add *because* the three
  contributions were built as separable modules from the start (MOD 1/2/3). Turning
  each off is one flag, not a rewrite — the Phase-2 modularity paid off two phases later.

---

### 2026-07-04 — Phase 8 honest ground-truth circularity: the three-way fix

- **Problem named.** The Phase-8 ground truth is *model-in-the-loop*: the same GNN
  both produces the predictions evaluators see AND scores which intervention is
  "best". Internally consistent, but circular by construction — a reviewer will
  flag it, so we address it head-on rather than bury it.
- **The paper does all three, not one:**
  1. **State it** as a Limitation verbatim (was already in `simulate.py` +
     `human_study.yaml`; kept).
  2. **Reframe as internal consistency** — the answerable question is "does the
     full pipeline help humans use the *model's own* predictions better than raw
     numbers / explanations alone?" Valid decision-quality result on its own terms.
  3. **External anchors** — new: `external_anchors.json` lets the professor mark
     1–2 scenarios where the real-world correct action is known from professional
     experience. `analyze.py` now reports a separate **EXTERNAL-ANCHOR ACCURACY**
     block scored against those expert labels — a small out-of-model validation set.
- **Built:** `_load_anchors` / `_anchor_accuracy` + `--anchors` flag in
  `analyze.py` (auto-loads `external_anchors.json` from the results dir if present;
  skips `_`-prefixed meta keys); `external_anchors.example.json` fill-in template;
  expanded the limitation comment in `configs/human_study.yaml` to spell out all
  three mitigations.
- **Verified:** seeded demo scenarios + synthetic responses + a 2-scenario anchor
  file → `analyze` printed the anchor block (anchored scenarios scored vs the
  expert action, un-anchored excluded), wrote `external_anchor_accuracy` into
  `analysis_report.json`. Cleaned up the seeded artifacts afterward.
- **Reflection:** option 3 costs one conversation during the eval session and zero
  code risk (the anchor path is additive — absent file → simulation-only, exactly
  as before). Cheap insurance that turns a known weakness into a validation point.

---

## 2026-07-06 — Real 100-epoch checkpoint: Phase 3 + 5 gate rerun, artifacts refresh
- The real trained checkpoint landed from Colab (T4, CUDA): `metr_la_best.pt`,
  **epoch 34, val MAE 2.9026** — beats published Graph WaveNet (3.07). All the
  "owed on the full ckpt" numbers below are now REAL, not placeholder.
- **Prediction (re-ran `evaluate.py` on the real ckpt, METR-LA test):**
  15min MAE **2.82** / RMSE 5.46 · 30min MAE **3.21** / RMSE 6.45 ·
  60min MAE **3.66** / RMSE 7.46 · overall MAE 3.166, MAPE 8.78%. Improves on the
  old 3-epoch placeholder at every horizon (30min 3.52→3.21, 60min 4.42→3.66) and
  crushes HistAvg 5.15 / LinReg (5.03 @30min). Table 1 refreshed.
- **Phase 3 gate rerun** (`explainer_metrics --n 100`, k=8, n=90 valid targets):
  Fidelity+ **1.393 ± 2.68** vs random **0.843 ± 1.35** → PASS (beats random ~1.65×).
  Stability **0.751 ± 0.15**. Sparsity 0.039. Fidelity- 12.93 ≈ random 12.84 —
  still inconclusive even on the full ckpt (uniform target sampling; the signal
  only separates on congested targets, as noted since Phase 3). 6 scenario
  explanations regenerated, semantically sane (high-congestion East LA 20→20.8mph
  explained by nearby congested Glendale/SFV sensors, lag 30min).
- **Phase 5 faithfulness study** (full stratified set, **n=93**, llama3.1:8b,
  conditions A/B/C):
  - A (full): P **0.995** · R 0.593 · **F1 0.725** · halluc **0.005** · quant 0.698
  - B (no explanation): P 0.172 · R 0.071 · **F1 0.090** · halluc **0.828** · quant 0.360
  - C (no city context): P 0.995 · R 0.605 · **F1 0.728** · halluc 0.005 · quant 0.764
  - **Core claim proven at scale:** the mathematical explanation eliminates
    hallucination (A/C 0.005 vs B 0.828) and restores grounding (B precision
    collapses 0.995→0.172). Much cleaner than the 6-scenario smoke run.
- **Surprise / flag resolved:** the Phase-5 open question — "does city context (A
  vs C) matter at scale on congested windows?" — is answered: **C ≈ A** even with
  pm_rush/high-congestion strata included. Faithfulness measures explanation↔
  reasoning alignment, which city context is orthogonal to; C's slightly higher
  quantitative_fidelity (0.764) is within noise. City context helps *advisory
  usefulness*, not *faithfulness* — a clean, defensible story for the paper.
- **Phase 9 artifacts** regenerated from the real logs: 7 figures + 5 tables.
  Fig6 (cross-city) and Table 4 (ablation) correctly stay PENDING — still Colab-owed.
- **Still owed (unchanged, all Colab):** full fusion retrain + comparison table;
  Chicago Phase-1 pipeline + cross_city transfer table; Chicago faithfulness study;
  multi-seed ablation tables.

## Phase 10 — simulation-based decision evaluation (2026-07-06)
- **Goal:** replace the human study as Contribution #3 with a reproducible,
  at-scale decision-quality experiment. 500 stratified scenarios; 3 decision
  agents (RANDOM / RAW-LLM / XTRAFFIC-LLM) pick one of 5 interventions; score vs a
  model-in-the-loop ground truth (lowest predicted 30-min network delay).
- **What the smoke test caught (this is why we smoke first):** the first design —
  each intervention a fixed-% speed uplift on its node set — gave a DEGENERATE
  ground truth. `signal_retiming` won ~80% of scenarios because the target is by
  construction the slowest node (biggest delay contributor) and only signal/transit
  touch it directly. The LLM agents reasoned toward reroute/ramp_metering (the
  plausible traffic-engineering actions) and so scored *worse than random*
  (XTRAFFIC 0.10 < RAW 0.17 < RANDOM 0.47). An indefensible Contribution #3.
- **Fix (chosen with Naren — Option 1, equal-budget + two physical mechanisms):**
  every active intervention spends the SAME uplift budget, but (1) only on its
  CONGESTED nodes (spending on already-fast nodes is wasted), and (2) with a
  per-segment cap (one action can't teleport a segment to free-flow). Reroute =
  source-node relief + top-edge weight cut. Now the best action depends on WHERE
  congestion sits, not on which action hits the worst node.
- **Calibrated on cached explanations** (fast --gt-only sweeps over budget/cap).
  Locked budget=45 mph, cap=24 mph. **Confirmed GT distribution (n=41): transit
  34% · ramp 32% · signal 27% · reroute 5% · no_action 2%** — a genuine 4-way
  decision problem, reached by physical mechanisms, NOT by hand-tuning outcome
  weights (important for defensibility).
- **Harness:** evaluation/sim_eval.py + configs/sim_eval.yaml. Explanations +
  advisories cached; every decision appended to decisions.jsonl so the ~8h full
  run (500 scenarios × 3 seeds × 3 conditions, llama3.1:8b) is fully resumable.
  Outputs CSV + JSON + booktabs table_sim_eval.tex. Full run KICKED OFF; numbers
  land here on completion.

---

## Phase 11 — cross-city × cross-model faithfulness (2026-07-07)
- **Goal:** stress-test Contribution #2. Phase 5 proved "the math explanation kills
  LLM hallucination" on ONE city + ONE model. A reviewer will ask: METR-LA quirk?
  llama3.1 quirk? So rerun the A/B/C study on 3 cities × 2 LLMs → one master table.
  File: `evaluation/cross_city_faithfulness.py` + `configs/cross_city_faith.yaml`.
- **Zero-shot transfer, done honestly.** PEMS-BAY (325) and Chicago (1020) don't
  match METR-LA's 207 nodes, so I reuse `cross_city.py`'s transfer: node-agnostic
  weights copy, node-specific embeddings re-init at target N (copied 119, re-init 4).
  **The argument that makes this valid:** faithfulness is a *prompt-structural*
  property — condition A shows the LLM the explainer's top-k and forbids inventing
  causes; B hides them. Whether the LLM obeys is independent of how *accurate* the
  transferred prediction is. So a possibly-inaccurate zero-shot model is a perfectly
  clean probe of the grounding effect. (Bonus gotcha learned: `load_state_dict`
  raises on the 207→325 shape mismatch *even with strict=False*, so I pre-build the
  transferred checkpoint at the target N instead of loading cross-N.)
- **Chicago is tiny:** 15 test windows (< my 20 floor). Pooled train+val+test = 76 →
  stratified sample gives 33 (≥20, no fallback). Zero-shot ⇒ never trained on
  Chicago ⇒ pooling adds no leakage for a *faithfulness* (not accuracy) measurement.
- **Surprise:** the playbook's second model `qwen2.5:7b` AND its `tinyllama` fallback
  are BOTH missing from local Ollama. Auto-detect ladder picked **mistral:7b** —
  actually the *better* choice (different family from llama3.1, same size class →
  stronger "not two llamas" evidence). Logged the substitution on the record.
- **Smoke-verified end-to-end** (mock-LLM, isolated dir): transfer ckpts build,
  real explainer runs on all 3 cities incl. the 1020-node Chicago transfer, metric +
  aggregation + LaTeX table all produce the expected A-halluc 0 / B-halluc 1 shape.
  Also ran ONE real llama3.1:8b advisory on a transferred PEMS-BAY explanation with
  the empty-KB path → schema-valid, cond A F1 0.667 / halluc 0.000, 3 grounded
  causes. Skeleton `table_cross_city_faith.tex` emitted (all PENDING until the run).
- **Owed:** the full run — multi-hour and it shares the one local Ollama with the
  running Phase-10 sim, so it goes AFTER Phase 10 (fully resumable, so it can be
  killed/resumed freely). Numbers land here on completion.

---

## Phase 12 — SHAP baseline comparison ("why not just use SHAP?") (2026-07-07)
- **Goal:** answer the inevitable reviewer question with an EXPERIMENT, not a
  sentence. Build a SHAP explainer that plugs into the exact same pipeline, feed
  its output to the same LLM advisor, and measure whether the reasoning stays as
  faithful. Files: `models/explainer/shap_explainer.py` (explainer) +
  `evaluation/shap_comparison.py` (study/table). Pinned `shap==0.44.1` (last clean
  cp39 wheel; it held numpy at 1.26 — bumping numpy to 2.x would break torch).
- **The clean design idea:** SHAP scores the same unit GNNExplainer does — the N
  nodes. A coalition is a binary node mask; masking a node pushes its speed to the
  z-space mean, which is *literally GNNExplainer's own "remove this node" op*. So
  the two explainers share one definition of "absence" → a fair, controlled swap.
  Because I gave `ShapExplainer` the same `explain_target()` signature, I reuse
  100% of `ExplanationBuilder` (naming, top-k, path, lag, confidence, schema
  validation) — the ONLY thing that changes is the importances. Byte-compatible
  schema JSON, produced by SHAP.
- **What the smoke taught me (nsamples=50 is the assignment's speed choice):** 50
  coalitions for 207 nodes is underdetermined, so SHAP needs an L1 sparse solve.
  `aic`/`bic` literally *crash* (LassoLarsIC can't estimate noise variance when
  samples < features); `num_features(k)` works but LARS goes degenerate and only
  ~13–18 nodes get nonzero SHAP, and the top-k *changes between runs*. So SHAP here
  is slow **and** unstable — which is itself part of the answer, and I documented it
  rather than hiding it.
- **Real 3-scenario smoke (n=3, night/low-congestion — do NOT over-read):**
  GNNExplainer F1 0.764 / halluc 0.000 · SHAP F1 0.853 / halluc 0.000 · No-explainer
  F1 0.000 / halluc 1.000. Two honest takeaways: (1) on *faithfulness*, SHAP is NOT
  worse — both explainers keep the LLM grounded, because the advisor is instructed
  to cite only what it's shown, and it obeys for either. So the argument for
  GNNExplainer leans on its *other* axes (speed, stability, edge/propagation
  structure). (2) **GNN vs SHAP top-k overlap = 0.00** — the two explainers pick
  completely different nodes (free-flow regime here, where there's no real spatial
  cause). Whether that 0-overlap + SHAP's instability makes the LLM confabulate more
  under *congestion* is exactly what the full 30 (stratified) will show.
- **Owed:** the full 30-scenario real run (`python -m
  xtraffic.evaluation.shap_comparison`, ~84 LLM calls). Resumable — the smoke's 3
  decisions are reused. Runs in the same Ollama queue after Phase 10/11.

---

## Phase 13 — condition-A failure taxonomy (paper Section 6) (2026-07-07)
- **Goal:** an honest account of when the FULL pipeline (condition A) still fails —
  reviewers trust a system whose authors mapped its limits. Also closes the loop on
  the Phase-4 self-attribution error: does it survive into the tuned study?
  File: `evaluation/failure_modes.py`.
- **Data reality I had to handle:** Phase 5 logged the per-scenario *metrics* but
  threw away the LLM's cited causes, and 3 of the 6 categories need them. So for
  each of the 7 failures I regenerate condition A on the *cached explanation* (~7
  LLM calls, no explainer cost) to recover the cited-cause detail; the failure set
  and F1/hallucination stay Phase-5's logged numbers. The regen reproduced the
  logged 0.50 hallucination on the one hallucination case, so it's faithful.
- **Two real bugs the smoke caught (this is why I smoke):**
  1. My first SELF_ATTRIBUTION rule ("target in the resolved set") false-fired: the
     mock cited "San Fernando Valley", which via *region-level credit* resolves to a
     set containing both a real top-k sensor AND the target (same region) → wrongly
     flagged. Fix: self-attribution requires a cited cause that *misses* top-k and
     resolves to the target — i.e. it's a specific kind of hallucination, not just
     "the target appeared in a region set."
  2. TEMPORAL_CONFUSION false-fired on all 3 "correct-node" failures because the
     advisory says "...will reach the target in **30 minutes**" — that's the forecast
     *horizon*, not a propagation lag. Fix: only count a "N min" as a lag claim if
     lead/lag *language* sits next to it, and exclude the horizon. After the fix,
     temporal = 0 (there genuinely are no lag-contradiction claims here).
- **Result (real llama3.1:8b, 7 failures / 93 = 7.5%):** SELF_ATTRIBUTION 1 (1.1%),
  FREE_FLOW 1 (1.1%), OTHER = under-citation 5 (5.4%), and GEOGRAPHIC / TEMPORAL /
  CONFIDENCE_MISMATCH all **0**. **The story I did NOT expect to be this clean:** the
  pipeline essentially never *fabricates* — precision stays 1.0 on 6/7 failures and
  there are zero geographic hallucinations. Its failures are *incompleteness*
  (it names the strongest source and drops the rest → recall collapses while
  precision holds) plus two rare regime artifacts. The single self-attribution case
  (the Phase-4 bug, confirmed but rare) turns out to co-occur with free flow: a
  missing-sensor target predicted at 60 mph, where with no real cause the LLM falls
  back to blaming the target. And CONFIDENCE_MISMATCH = 0 because the explainer was
  correctly *un*-confident (conf < 0.8) on every single failure — a nice consistency
  result on its own.
- **Output:** `table_failure_modes.tex` + `failure_modes.json` (per-category example
  in region names only, hypothesised cause, concrete mitigation) + a Section-6
  console report. Complete — no owed run (only 7 LLM calls, already done).
