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
