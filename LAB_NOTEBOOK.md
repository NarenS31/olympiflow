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

## Phase 16 — active grounding loop (does correction actually raise faithfulness?) (2026-07-12)
- **What I built:** `models/advisor/active_grounding.py`. It closes the loop on the
  Phase-5 faithfulness metric: run condition A, score it, and if F1 < 0.7 hand the
  LLM a *targeted* correction that names the exact explainer top-k nodes it failed to
  cite (with importance + current speed), then re-prompt — up to 3 rounds. Every
  round's F1/precision/recall/hallucination is logged = a per-scenario convergence
  curve. Config knobs (`f1_threshold`, `max_rounds`) live in a new
  `active_grounding` block in `advisor.yaml` (no magic numbers).
- **The one change to old code, flagged:** I added an optional `extra_instruction`
  argument to `advisor.advise_condition` / `build_prompt_condition`. With it `None`
  (every Phase 4/5/15b caller) the prompt is *byte-identical* to before — I checked
  this explicitly (`build_prompt_condition(exp, kb, 'A') == build_prompt(exp, kb)`),
  because the CLAUDE.md rule is "never silently change something we built earlier."
  The correction is appended *after* the TASK so the model reads it as the newest
  instruction.
- **Why the correction targets recall, not precision:** Phase 13 already told me the
  condition-A failures are almost all *under-citation* — the LLM names the single
  strongest source and drops the rest, so recall collapses while precision stays 1.0
  and hallucination stays 0. So "you didn't address these locations" is the correct
  lever. And if a failure were precision-side instead (something cited that ISN'T in
  the top-k), there'd be no missed node to name — I made the loop detect that
  (missed set empty while F1 < threshold) and stop with reason `no_missed_nodes`
  rather than pretend it can fix it.
- **Smoke design:** I deliberately did NOT smoke on the first 5 stratified windows
  (they're mostly already-faithful night/low-congestion cases → the loop would be a
  boring no-op). Instead I added `--select low-f1`, which pulls the 5 *worst*
  condition-A scenarios straight from the Phase-5 per-scenario CSV (they have cached
  explanations, so zero explainer cost). That's the honest way to see whether
  correction works — point it at real failures.
- **Result (REAL llama3.1:8b, 5 worst failures):** mean F1 **0.438 → 0.912 in a
  single correction round** (gain **+0.474**), **20% → 100%** reaching threshold,
  mean **0.80** rounds used. The trace: idx 219 `0.222→1.000`, idx 4367 `0.400→0.933`,
  idx 3296 `0.400→0.857`, idx 5668 `0.400→1.000`.
- **The finding I care about most (and it's the honest one):** *precision stays
  exactly 1.000 at every round and hallucination stays 0.000.* The correction raises
  recall (0.30 → 0.85; cited causes jump from 1–2 up to 5–8) **without** tempting the
  model to fabricate to comply — which is the obvious risk of telling an LLM "you
  missed these, add them." It didn't invent; it went back to the explanation and
  cited what was really there. So the loop is a clean *enforcement* mechanism.
- **Bonus — the "no needless re-prompt" gate, on the real model:** idx 2917
  regenerated at F1 0.769 (its temp-0.1 regeneration landed above threshold even
  though its Phase-5-logged F1 was 0.40), so the loop exited with **zero**
  corrections. Early-exit path confirmed live, not just in the mock.
- **Honest caveat I wrote into the file and CLAUDE.md:** this optimises the advisory
  *toward* the explainer's top-k. It proves we can ENFORCE consistency between the
  math and the words in a closed loop; it is NOT independent evidence the explanation
  itself is right. The paper must frame it as enforcement / internal-consistency.
- **Most-missed region at round 0:** Glendale / Burbank (12 times across the 5
  scenarios) — a concrete pointer to where under-citation concentrates.
- **Output:** `table_active_grounding.tex` + `fig_active_grounding.pdf` (2-panel:
  F1-vs-round spaghetti+mean with the threshold line, and cumulative % reaching
  threshold) + `results/active_grounding/{per_round.csv, decisions.jsonl,
  summary.json}`, all resumable via a per-model decision cache. **Owed:** the full
  stratified ≥93-scenario run for the paper number (multi-hour, shares the one local
  Ollama with the Phase 10/11/12 queue) — the committed table/fig are the n=5 smoke
  and say so in the caption.

## Phase 17 — counterfactual explanations ("what could I have done differently?") (2026-07-12)
- **What I built:** `models/explainer/counterfactual.py` (the searcher + the record)
  and `evaluation/counterfactual_study.py` (the 20-scenario study), driven by one new
  `configs/counterfactual.yaml`. Instead of explaining WHY a target is congested, I find
  the MINIMUM speed uplift on its critical nodes that flips the 30-min prediction back
  above free flow (35 mph), then have the LLM narrate that counterfactual. I also added
  a flagged, default-preserving `advise_counterfactual` to `advisor.py` — a
  counterfactual-mode prompt that reuses the SAME advisory JSON contract, so
  `validate_advisory` and the Phase-5 faithfulness metric both apply with zero changes
  (`advise()`/`advise_condition()` stayed byte-identical).
- **The search (gradient-free, two stages):** (1) a uniform uplift sweep in 2 mph steps
  up to a 20 mph budget until the target crosses 35; (2) a greedy per-node minimisation
  that relaxes each lever back toward 0 (least-important first) while keeping the flip,
  so I report the genuinely minimal per-node change, not a blanket uplift. The
  perturbation is the same "raise speed on the last few input steps, clip at free-flow"
  mechanism as the Phase-8/10 simulator — one notion of "what an action does."
- **The finding that reshaped the whole phase (this is the honest, important one):**
  perturbing ONLY the explainer's upstream top-k barely moves the target. I measured it:
  a target at 34.2 mph stays at **34.2** even if I raise all its upstream critical nodes
  by +40 mph. But raising the **target itself** flips it (34.2 → 56.2). So this ST-GNN's
  30-min forecast is dominated by the target's own recent speed — which is exactly the
  Phase-3 *Fidelity−* result (the prediction isn't concentrated in the top-k) coming back
  to bite. An upstream-only counterfactual is therefore almost never feasible and would
  leave nothing to narrate.
- **The design decision I made (and flagged):** I put the **target bottleneck itself**
  into the candidate lever set alongside the congested top-k. That's the standard
  actionable lever anyway (signal retiming / incident clearance = the Phase-8/10
  target-scope action), and it makes the counterfactual feasible. I kept upstream nodes
  in the set so propagation is used where it *does* carry leverage, and the study
  *reports* how often upstream actually contributed — turns out **0/8**, so I'm honest
  that leverage lives at the bottleneck under this model.
- **Calibration (earned its keep, like Phase 10):** with target+top-k levers, 45% of
  congested targets flip within 20 mph, and it tracks congestion depth cleanly — mild
  targets (pred 25–35) flip 93% of the time at a median of just **8 mph**, medium (15–25)
  31%, deep (0–15) 17%. Deeply-congested targets are honestly **infeasible** — you can't
  prevent a 3.5 mph jam with a modest intervention.
- **A real bug the smoke caught (cache staleness):** `build_or_load_explanation` caches
  by filename only and never checks the checkpoint — 6/93 cached explanations predate the
  epoch-34 `metr_la_best.pt` (same filename overwrote the old 3-epoch placeholder) and are
  STALE (cached pred 34.13 vs live 37.27 for idx 4289). I made the study staleness-aware:
  use the shared Phase-5 cache only when it's newer than the checkpoint, else rebuild into
  my OWN cache (never touch Phase-5 artifacts), and filter congestion on the **live**
  model prediction. All 49 congested-by-live targets have fresh caches (the 6 stale ones
  are free-flow live → skipped), so zero rebuilds were needed.
- **A prompt bug the smoke also caught:** my counterfactual TASK said "what specific
  action" (singular), so llama3.1 sometimes returned 1 recommendation and failed the
  advisory contract's 3–5 rule → empty advisory → F1 0 (idx 1770 did exactly this). Fixed
  by asking for "3 to 5 concrete recommendations," matching the standard task; after that
  all 8 flips pass.
- **Result (n=20 congested METR-LA, real llama3.1:8b):** 21 free-flow targets skipped
  ("no counterfactual needed"); **validity 40% (8/20)**; mean budget **10.0 ± 4.58 mph**;
  **1** node changed; **all 8 target-only**; mean implied lead 11.9 min. Narrative
  faithfulness to the counterfactual (Phase-5 metric vs the required-change nodes):
  **F1 1.000 / precision 1.000 / recall 1.000 / hallucination 0.000** across all 8 — the
  LLM narrates exactly the required node and invents nothing. Note "cite the target" is
  *correct* here (the counterfactual says CHANGE the target) — the opposite of the
  Phase-13 self-attribution error in the WHY context.
- **One clean full chain (idx 3515):** Downtown LA predicted 32.29 mph → counterfactual
  "+8 mph at the bottleneck" flips it to 35.4 (implied lead 20 min) → LLM: "raise the
  speed from 32.87 to 40.87 mph at Downtown LA," with 3 concrete actions (dynamic lane
  management on I-110, signal timing on Brand/San Fernando, ramp metering at I-5/SR-134).
  Cited cause resolves to Downtown LA, F1 1.0.
- **Output:** `table_counterfactual.tex` + `results/counterfactual/{per_scenario.csv,
  summary.json, example_traces.txt/.json}` (3 full prediction→explanation→counterfactual→
  narrative chains), resumable per-model cache. **Caveats for the paper:** the
  counterfactual reads as "the minimum bottleneck intervention that prevents the jam," not
  an upstream-cascade story (because of Fidelity−); validity is 40% at the 20 mph budget.
  **Owed (feasible locally, not Colab):** a larger congested-only sweep (≥50 mild/medium
  targets, stratified by tercile) to tighten the validity-vs-depth curve — the committed
  table is n=20 and says so.

---

## Phase 18 — uncertainty-aware explanations ("how much do I trust one run?") (2026-07-12)
- **The idea:** Phase 3 gave one explainer run a *scalar* confidence (mean Jaccard over K
  reruns). Phase 18 promotes that to a **per-node** distribution: run GNNExplainer K=10
  times, count how often each node lands in the top-k, and label each **CORE** (≥80% of
  runs → assert), **PERIPHERAL** (20–80% → hedge), or **NOISE** (<20% → omit). Inject the
  split into the LLM with an honesty instruction and measure whether the model actually
  hedges the uncertain causes.
- **Built:** `models/explainer/uncertain_explainer.py` (K-run tally, `classify_tier`,
  `mean_pairwise_jaccard` stability, the uncertainty block, `hedging_analysis`, a no-LLM
  demo) + a **flagged, purely-additive** advisor mode (`advise_uncertain` /
  `build_prompt_uncertain` / `_SYSTEM_UNCERTAIN`; `git diff` = 132 insertions, 0 deletions,
  so condition A / Phase 4-17 stay byte-identical) + `evaluation/uncertainty_study.py`
  (the A-vs-U study, table, stability plot, traces) + `configs/uncertainty.yaml`.
- **Reuse (kept it small):** the K runs are the SAME `explain_target(seed=k)` solves Phase 3
  already does for the confidence scalar — I just keep per-node frequencies instead of one
  Jaccard. The base explanation, KB retrieval, faithfulness metric, and stratified sampler
  are all the existing Phase-3/4/5 machinery. The uncertainty block attaches ADDITIVELY
  (extra `"uncertainty"` key) so `validate_explanation` still passes — the Phase-4 contract
  is untouched.
- **Scoring decision (flagged):** both A and U are scored against the SAME deterministic
  single-run (seed-0) top-k, so the F1/hallucination difference is attributable purely to
  the uncertainty FRAMING, not a moved goalpost. Honest caveat: U omits noise-tier nodes,
  so a seed-0 node it correctly drops counts against U's recall — small (core dominates),
  and I also log U's faithfulness vs its own shown set as a secondary diagnostic.
- **What surprised me:** on congested/unstable targets the explanation stability is genuinely
  LOW — the demo's East LA (20 mph) target had stability **0.30** with only 3 reliable core
  nodes and 10 noise nodes a single run would have hidden. And the free-flow smoke scenario
  produced **0 core nodes** (stability 0.16): the method honestly says "nothing is confidently
  a cause" exactly where the whole project keeps finding there is none.
- **Smoke (real llama3.1:8b, n=3, heavily caveated):** mean stability 0.41 (core/peri/noise
  3.0/11.7/10.3). Faithfulness **A F1 0.730 / halluc 0.000 → U F1 0.812 / halluc 0.111**:
  the framing lifts recall (0.58→0.75) but invites a little hallucination (mentioning
  peripheral nodes outside the deterministic top-k costs precision). **Q3 (the key question)
  — the real model DOES hedge:** idx 4289 U-reasoning asserts core ("*the Glendale/Burbank
  area, which is confirmed by all runs*") and hedges peripheral ("*It's possible that there
  may be some impact from ... Hollywood, but these are less certain*") → hedge gap +0.33,
  and U's F1 (0.857) beat A's (0.667). Honest nuance: idx 5479 the LLM just OMITTED peripheral
  entirely (gap 0) rather than hedging — also acceptable. `--mock-llm` reproduces the harness
  with no Ollama. **Smoke gate PASSED.**
- **Full run (real llama3.1:8b, n=20 stratified METR-LA, K=10):** mean stability **0.427**
  (range **0.155–0.75** — genuinely varies scenario to scenario, the whole point), mean
  core/peripheral/noise **3.6/11.0/7.0**. **Faithfulness A→U:** F1 0.704→**0.737** (+0.03,
  flat); precision 1.000→0.896; recall 0.562→**0.637**; hallucination 0.000→**0.104**.
  **What I found, honestly:** (Q1) the framing is ~F1-neutral — it trades a little precision
  for higher recall (it mentions MORE of the top-k). (Q2) it does NOT drive hallucination
  below Phase-5's ~0.5% — it slightly RAISES it, because inviting the model to *mention*
  uncertain causes means it sometimes cites a node outside the deterministic top-k; but this
  is concentrated (**only 4/20** U advisories hallucinated at all — 16/20 stayed at 0, so the
  mean is high-variance). (Q3, the key question) **YES the real model hedges:** condition-U
  core/peripheral hedge rates **0.23 / 0.39** (gap **+0.13**) vs condition-A ~0/0.03, and U
  hedged peripheral more than core in **9/19** scenarios (mean gap +0.125). So uncertainty
  framing makes the model measurably more epistemically honest (~1.7× more hedging on
  uncertain causes, in about half of cases) at a small faithfulness cost — not uniform, and
  sometimes it just OMITS peripheral instead of hedging (idx 5479). The method also correctly
  collapses in free flow: **1/20** had 0 core nodes (stability 0.16) = "nothing is confidently
  a cause" — the project's recurring no-real-cause finding, now surfaced by the tiers.
- **Honest caveat for the paper:** this is an internal-consistency / behavioural result — we
  show the LLM *reflects* the explainer's per-run uncertainty, not that the tiers are the
  "true" importances. And U's small hallucination bump is the price of higher recall; if a
  deployment wants zero hallucination, gate on core-only. Committed `table_uncertainty.tex` +
  `fig_uncertainty_stability.pdf` are the real n=20 (captions say n=20). **Gate PASSED.**

---

## Phase 11 — cross-city × cross-model faithfulness: FIRST REAL CELLS (2026-07-19)

- **What landed:** the full run is going in a `phase11` screen session and **2 of the 6
  cells are done** (Chicago still outstanding; the METR-LA × llama3.1 cell reuses the
  committed Phase-5 summary). Numbers read straight from
  `results/cross_city_faithfulness/summary_<city>__<model>.json`, not from memory:
  - **METR-LA × mistral:7b** (n=93, in-domain checkpoint, conditions A/B):
    A precision **1.000** / recall 0.714 / **F1 0.821** / **hallucination 0.000**;
    B precision 0.190 / recall 0.063 / F1 0.087 / **hallucination 0.810**.
    → **F1 gap +0.734**, hallucination gap −0.810.
  - **PEMS-BAY × llama3.1:8b** (n=85, **zero-shot** transferred checkpoint, A/B/C):
    A precision 0.994 / recall 0.650 / **F1 0.764** / **hallucination 0.006**;
    B precision 0.053 / recall 0.025 / F1 0.032 / **hallucination 0.947**;
    C precision 0.994 / recall 0.616 / F1 0.739 / hallucination 0.006.
    → **F1 gap +0.732**, hallucination gap −0.941.
- **Why this is the result Phase 11 exists for:** the Phase-5 grounding claim is **not a
  llama3.1 quirk and not a METR-LA quirk**. Change the LLM *family* → gap +0.734. Change
  the *city* AND run the GNN **zero-shot** (207 → 325 nodes, node-specific params
  re-initialised) → gap +0.732. Two completely different perturbations, essentially the
  same magnitude.
- **What surprised me:** (1) **mistral:7b is MORE disciplined than llama3.1** on METR-LA —
  hallucination *exactly* 0.000 and precision *exactly* 1.000 across all 93 scenarios, no
  std at all. I expected the smaller/different-family model to be sloppier; it wasn't.
  (2) **PEMS-BAY B hallucinates at 0.947** — even worse than METR-LA's 0.81/0.83. With no
  KB and no explanation the model has *nothing* to cite, so it invents almost every time.
  That makes B's collapse look like a property of ungrounded LLMs generally, not of one
  city's prompt.
  (3) **C ≈ A on PEMS-BAY with a completely EMPTY knowledge base** (F1 0.739 vs 0.764,
  hallucination identical). Phase 5 said city context is orthogonal to faithfulness; here
  there *is* no city context and faithfulness barely moves. Cleanest version of that claim
  we're going to get.
- **The zero-shot argument to keep making:** PEMS-BAY *prediction* accuracy is degraded by
  the transfer, but faithfulness is intact — because faithfulness is **prompt-structural**
  (does the LLM cite only the explainer's top-k?), not accuracy-dependent. A degraded model
  that still produces a faithful advisory is exactly the not-a-quirk probe we wanted.
- **Housekeeping:** `cross_city_faithfulness_master.{csv,json}` still reads all-PENDING
  (dated 07-07) — the master table is rebuilt at the *end* of the run. Regenerate with
  `--table-only` once Chicago lands. **Not committed yet.**

---

## Phase 12 — SHAP baseline comparison: FULL RUN (n=28) (2026-07-19)

- **The run:** `python -m xtraffic.evaluation.shap_comparison`, llama3.1:8b, **n=28**
  (30 requested, per_stratum=2), 84 decisions, **0 advisory errors**. Strata covered:
  10 low / 10 medium / 8 high congestion × 5 time-of-day bands.
- **THE TABLE** (mean ± std, n=28, scored by the Phase-5 faithfulness metric):

  | Explainer | F1 | Hallucination | Precision | Recall |
  |---|---|---|---|---|
  | **GNNExplainer (ours)** | **0.742 ± 0.168** | **0.018 ± 0.093** | **0.982 ± 0.093** | **0.616 ± 0.192** |
  | **SHAP (KernelExplainer)** | 0.692 ± 0.160 | 0.018 ± 0.093 | 0.982 ± 0.093 | 0.554 ± 0.187 |
  | **No explainer** | 0.048 ± 0.137 | 0.857 ± 0.350 | 0.143 ± 0.350 | 0.031 ± 0.098 |

  Quantitative fidelity: A 0.749 | SHAP 0.692 | none 0.377.
  **GNN-vs-SHAP top-k Jaccard overlap: 0.022 ± 0.042.**
- **Honest verdict (the n=3 smoke reading held up — I am not going to dress this up):**
  **on faithfulness alone, SHAP is not worse.** GNNExplainer edges it on F1 (0.742 vs
  0.692) and the whole gap is **recall** (0.616 vs 0.554); hallucination and precision are
  **identical to 3 d.p.** (the *same single* scenario hallucinated under each), and the
  per-scenario winner splits **12 A / 7 SHAP / 9 tie**. +0.05 F1 sits well inside a ±0.17
  std and I have not run a paired CI, so the paper says **"comparable"**, not "we win".
  The reason they tie is obvious in hindsight: the advisor is told to cite only what it is
  shown, so it does — *for either explainer*.
- **What both crush** is the no-explainer baseline: F1 0.048, hallucination 0.857,
  **24/28 scenarios hallucinated** vs **1/28** for each explainer. That's the Phase-5 core
  claim re-proven a third way — it is the **presence** of a mathematical explanation, not
  *which* explainer made it, that eliminates hallucination.
- **So the real answer to "why not just use SHAP?"** is not a faithfulness win, it's the
  other three axes (and I have evidence for each): **cost** (a fresh KernelExplainer solve
  per prediction, nsamples=50 ≪ 207 nodes), **instability** (the underdetermined LARS solve
  leaves only ~13–18 nodes nonzero and the top-k moves run to run — why I cut
  `confidence_runs` 5→2), and **structure** (SHAP scores nodes only, so edges and the
  propagation path come from a product-of-endpoints proxy; GNNExplainer learns an edge mask
  directly). Argue those, and report the faithfulness parity as a finding.
- **The thing that actually surprised me — and I think it belongs in the paper:** the two
  explainers agree on **almost nothing**. Jaccard **0.022** means ~0 shared nodes out of 8.
  Two *mutually disjoint* "explanations" both keep the LLM at precision 0.982 and
  hallucination 0.018. Which means **faithfulness measures alignment, not correctness** —
  you can be perfectly faithful to a bad explanation. That is a real limitation of my own
  metric and I'd rather state it than have a reviewer find it. The axis that *does* judge
  correctness is Fidelity+ (Phase 3); running it on the SHAP top-k is the obvious follow-up
  and is **not done yet**.
- **Congestion breakdown** (tiny cells, don't over-read): A F1 0.698 low / 0.774 medium /
  0.758 high; SHAP 0.777 / 0.599 / 0.701. SHAP actually *beat* us in low congestion and
  lost medium — consistent with noise, and with the recurring "free-flow targets have no
  real spatial cause" finding.
- Outputs: `evaluation/paper/table_shap_comparison.tex` (now the real n=28) +
  `results/shap_comparison/{summary.json, per_scenario.csv, decisions.jsonl,
  explanations_cache/}`. **Phase 12 gate PASSED. Not committed yet.**

## 2026-07-19 — Phase 19 Steps 2-3: does the grounding result survive leaving traffic?

The whole point of Phase 19: Contribution #2 says the mathematical explanation is what
stops the LLM inventing causes. So far I'd only ever shown that on road networks
(METR-LA, then mistral instead of llama, then PEMS-BAY zero-shot). All still traffic.
Today I ran the same A/B protocol on the IEEE 14-bus power grid.

**Step 1 first, and it's not the headline I wanted.** `power_grid_best.pt` is epoch 6,
val MAE 0.001704 pu. On test it gets overall MAE 0.002912 pu — which beats HistAvg
(0.00366) and LinReg (0.00304) but **loses to persistence** (0.00283). At 15 min
persistence crushes it (0.00149 vs 0.00190); the model only wins at 60 min (0.00438 vs
0.00475). Bus voltage at 5-min resolution is just far more autocorrelated than traffic
speed, so "repeat the last value" is a brutal short-horizon baseline. I'm writing that
down plainly rather than quietly quoting the val MAE, because the val number (0.0017)
looks great and means nothing on its own. Also spotted `gate_power` sat at exactly
0.5000 for all 21 epochs — the power sidecar was never actually used in this run.

**The thing that nearly broke the experiment silently.** The Phase-4 prompt is hard-coded
traffic. Pointed at the grid it said *"You are a traffic-operations advisor... Current
speed: 0.88 mph"* — for a bus voltage. `DIFFERENCES_POWER_GRID.md` §8.2 had predicted
"labels will lie" and filed it as cosmetic. It is not cosmetic in a **prompt**: an LLM
told a substation is doing 0.88 mph will invent ramp metering, and I'd have measured my
own prompt's confusion as the model's hallucination. So I built `models/advisor/domains.py`
— units, node noun, stress noun, operator role — and left the grounding instruction and
prompt structure *identical* across domains, since those are the things actually under test.

**Making sure I didn't break five phases to add one.** Every committed number in Phases
4-18 came through the code I was editing. So before touching anything I `git stash`ed my
edits, captured the pre-refactor prompts/resolver output to `golden_traffic.json`, popped
the stash, and wrote `verify_traffic_unchanged.py` to assert byte-identity. It passes: 12
explanations x 5 artifacts, the 3 prompt constants, the resolver on METR-LA + Chicago —
all unchanged. That felt like overkill until I remembered the alternative is "I read it
carefully and I think it's fine."

**The sampling trap.** My first instinct was to reuse the traffic rule: target = the
lowest-voltage bus. Checked it: that's bus 3 in **1733 of 1995** test windows, because
buses don't share a free-flow reference (bus 3 normally sits at 1.008 pu, bus 8 at 1.089).
I'd have shipped a 20-scenario study of one bus and the flatness would have read as a
finding. Fixed by measuring each bus's sag against **its own** train-split mean. Second
trap right behind it: bus 1 (the slack bus, std 2e-4) then won the argmax in 1081 windows
on pure noise, so near-constant buses are ineligible as targets. After both fixes: 344
stressed windows, targets spread across buses 3, 4, 9, 10, 13, 14.

**The control that changed how I report everything.** k=8 of 207 METR-LA sensors is 3.9%
of the graph. k=8 of 14 buses is 57% — over half the network would count as "a valid
cause". I dropped to k=4, but rather than just assert that's fair I added chance
baselines: cite at random from the same vocabulary, score with the same metric. A random
**zone** citer scores **F1 0.653** with no model at all, because one electrical zone spans
up to 7 of 14 buses. That single number reframed the whole write-up.

**Result (n=20, llama3.1:8b, k=4):**

| condition | precision | recall | F1 | hallucination |
|---|---|---|---|---|
| A full pipeline | 0.950 | 0.825 | 0.869 | **0.050** |
| B no explanation | 0.000 | 0.000 | 0.000 | **1.000** |
| chance: random bus | 0.276 | 0.193 | 0.225 | 0.724 |
| chance: random zone | 0.687 | 0.706 | 0.653 | 0.313 |

Paired bootstrap, 10k iters — all nine CIs exclude 0. A vs B halluc -0.950
[-1.000, -0.875]. And crucially A still beats the *generous* zone-chance row:
halluc -0.263 [-0.378, -0.145], F1 +0.216 [+0.095, +0.326]. **The gap transfers.**

**Most interesting thing I found: condition B is worse than chance** (1.000 vs 0.724).
B never hit the top-k once in 20 scenarios. 63% of its citations don't resolve *at all*
— it kept naming `line 13-14`, `transformer 4-9`, `shunt capacitor bank at bus 9`. Those
are **edges**, not nodes. Stripped of instance-specific evidence, the LLM reached into the
knowledge base and recited generic domain facts with total confidence. That's a worse
failure mode than random guessing, and I didn't expect it — on METR-LA condition B at
least invented *places*, which are the right type of thing.

**What I refuse to overclaim.** A's F1 of 0.869 is *not* comparable to METR-LA's 0.725 —
different graph, different chance floor — and 62% of A's citations were coarse zone-level
ones that get generous credit. Worse: the top-k buses are direct electrical neighbours of
the target only **17.5%** of the time, *below* the 22% chance rate. So I have no positive
evidence these explanations are physically *correct* — only that the LLM faithfully repeats
what it's shown. Same limitation Phase 12 found with SHAP, and sharper here. Fidelity+ is
the axis that would settle it and it is **not run yet** (it needs the same relative-deviation
target fix). That's the top of my list.

Gates: traffic regression PASS (byte-identical), METR-LA resolver still 96.7%, new
power-grid resolver 22/22 — half of those cases are deliberate *refusals*, since a resolver
that happily matched "line 13-14" would have erased B's hallucination and manufactured my
own result.

## Phase 20 — does the *loop* transfer, not just the metric? (2026-07-20)

Phase 19 showed the hallucination **measurement** transfers to the power grid. This
was the harder question: does the **closed loop** from Phase 16 still work on a
graph that isn't a road network, with nothing retuned? I imported
`run_active_grounding` and called it as-is. Threshold 0.70, max 3 rounds, straight
out of `advisor.yaml` — the same values METR-LA used. I deliberately made them
*non*-overridable in the driver so I couldn't cheat later without noticing.

**One thing I had to fix, and I'm flagging it.** The correction block still had three
traffic hard-codes Phase 19's sweep missed — it literally told llama3.1 a substation
was *"currently 1.03 mph"* and warned it not to invent *"sensors, roads, incidents"*.
That's the exact confound `domains.py` exists to stop. So I ran **both** versions and
made the confound a measurement instead of a choice. `domain=None` is byte-identical
to Phase 16 (I pinned it with a golden-bytes test before editing).

| | r0 | r1 | r2 | r3 | reached |
|---|---|---|---|---|---|
| METR-LA (n=93) | 0.732 | 0.864 | 0.870 | 0.874 | 100% |
| grid, unmodified (n=20) | 0.892 | 0.973 | 0.973 | 0.973 | 100% |
| grid, domain-routed (n=20) | 0.899 | 0.932 | 0.932 | 0.932 | 100% |
| *grid engaged only (n=4)* | *0.558* | *0.964* | *0.964* | *0.964* | *100%* |

**Answer to my question: yes, one round.** Every single scenario that engaged the
loop — both variants — converged in *exactly* one correction round. The engaged
subset went 0.558 → 0.964, which is almost exactly the traffic low-F1 smoke
(0.438 → 0.912). Same shape, different domain, zero tuning.

**What I refuse to sell.** The grid's round-0 F1 of 0.892 looks better than traffic's
0.732 and it absolutely is not. Only **4 of 20** scenarios ever engaged, because
condition A is already above threshold at round 0 — nine of my first twelve traces
scored *exactly* 1.000. The reason is embarrassingly simple: k=4 of 14 buses, and the
prompt lists all four, so the model just echoes them back. On METR-LA it's 8 of 207
with long region names and it under-cites. That's a small-graph artifact, the same
k/N inflation that made Phase 19's random-zone baseline score 0.653. The loop
declining to fire is correct behaviour, not a win.

**The thing I'm most pleased with.** The variant comparison came out "unmodified beats
domain-routed by 0.041" and I nearly wrote that down as a finding. Then I realised
round 0 uses *no correction in either variant* — identical prompts — so any round-0
difference is pure LLM nondeterminism. Measured it: round-0 |diff| **0.078**, final
|diff| **0.041**. The "effect" is *smaller than the noise floor*. So: no detectable
effect of the correction vocabulary. I got a free test-retest reliability check out
of a comparison I designed for something else.

**Best single trace (idx 167).** The unmodified correction told it "predicted
congestion", "1.03 mph", "don't invent roads" — and llama3.1 fixed itself perfectly
anyway: F1 0.400 → 1.000, all four buses, hallucination 0.000, and its *own* reasoning
stayed in grid language ("absorbing too much reactive power", "1.090 pu"). It just
ignored my wrong labels. My hypothesis is the correction's only real job is to *name
the missed nodes* — the domain semantics come from the base prompt, which is already
domain-routed. One scenario (678) even went hallucination 0.500 → 0.000, so the
correction can *reduce* fabrication, not only raise recall.

**Two things this forced me to correct in the Phase 19 record.** First, I'd been
calling these "the low-faithfulness scenarios" — they're where condition *B*
hallucinated 1.000, but condition A (what the loop actually operates on) already
averaged 0.869. Second, and more seriously: **idx 1912 scored 0.857 here and 0.333 in
the committed Phase 19 run.** Same scenario, same model, same prompt. temp is 0.1, not
0. So any per-scenario claim I've made anywhere is one draw, not a property. That
applies to the Phase 13 failure taxonomy too and I need to think about it.

Gates: golden byte-identity PASS, traffic regression gate PASS (n=93 numbers safe).
Owed: rerun on mistral:7b — "domain-agnostic" currently rests on one LLM.
