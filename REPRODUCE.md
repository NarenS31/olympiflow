# REPRODUCE.md — XTraffic, fresh clone to every paper number

This is the ordered command list that reproduces every figure and table in the
paper from a clean checkout. Each block says **what it produces** and **what it
needs** (some steps need a GPU or a local Ollama server; those are flagged). The
final step, `make_paper_artifacts`, turns whatever logged results exist into the
paper's figures and tables — so you can rebuild the paper skeleton immediately and
fill in the compute-heavy numbers as each run lands.

All commands are run from the **repository root** unless stated otherwise, and use
`python3` (the project targets Python 3.9).

---

## 0. Clone and set up the environment

```bash
git clone https://github.com/NarenS31/olympiflow.git
cd olympiflow
git checkout xtraffic-phase-1            # the XTraffic branch

python3 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r xtraffic/requirements.txt
```

Everything runs as a module from the repo root, e.g.
`python3 -m xtraffic.<module>`. Random seed is 42 throughout (`configs/data.yaml`).

---

## 1. Data pipelines (Phase 1) — downloads + processed tensors

Needs network access. Writes to `xtraffic/data/processed/<dataset>/` (gitignored).

```bash
python3 -m xtraffic.data.pipelines.metr_la      # 207 nodes, ~34k timesteps
python3 -m xtraffic.data.pipelines.pems_bay     # 325 nodes
python3 -m xtraffic.data.pipelines.chicago      # segment-nodes (Socrata API)
```

Each prints a stats report (nodes, edges, timesteps, missing %, date range).
Sanity gate (no post-preprocessing NaNs, chronological non-overlapping splits,
shapes match config): the pipelines assert these on the way through.

---

## 2. Train the ST-GNN (Phase 2) — Table 1, Figure 3

Full 100-epoch training is impractical on CPU (~40 min/epoch on Apple MPS). Use a
**free Colab T4** via `xtraffic/models/gnn/colab_train.ipynb` (clones the repo,
trains, downloads `metr_la_best.pt` back into
`xtraffic/models/gnn/checkpoints/`). For a local smoke run:

```bash
python3 -m xtraffic.models.gnn.train --config xtraffic/configs/train_metr_la.yaml
```

Produces `evaluation/results/train_metr_la.csv` (per-epoch metrics + learned
`alpha` + modality gates → **Figures 3, 4**).

Baselines + test evaluation (→ **Table 1**):

```bash
python3 -m xtraffic.models.gnn.baselines --dataset metr_la
python3 -m xtraffic.models.gnn.evaluate  --dataset metr_la
```

Writes `baseline_*_metr_la.json` and `eval_metr_la_on_metr_la.json`.

---

## 3. Explainability (Phase 3) — Table 2, Figure 2

Needs a checkpoint + processed metr_la.

```bash
python3 -m xtraffic.models.explainer.generate            # 6 scenario explanations
python3 -m xtraffic.evaluation.explainer_metrics --dataset metr_la   # Table 2
python3 -m xtraffic.evaluation.visualize_explanation \
    --explanation evaluation/results/explanations/rush_hour_pm.json --dataset metr_la
```

Writes `explainer_metrics_metr_la.json` (→ **Table 2**) and the explanation
figure reused as **Figure 2**.

---

## 4. LLM advisory + faithfulness (Phases 4–5) — Table 3, Figure 5

Needs a **local Ollama server** with the model pulled:

```bash
# one-time: install Ollama (https://ollama.com), then
ollama pull llama3.1:8b
ollama serve &                           # background server on :11434
```

Validate the entity resolver (gate ≥ 90%), then run the study across conditions
A/B/C:

```bash
python3 -m xtraffic.evaluation.validate_resolver         # ~96.7% (29/30)
python3 -m xtraffic.evaluation.run_faithfulness_study    # >=100 scenarios (~1-2h)
# quick check: append  --limit 6  for a 6-scenario smoke run
```

Writes `evaluation/results/faithfulness/faithfulness_summary.json` (→ **Table 3**)
and `faithfulness_per_scenario.csv` (→ **Figure 5**).

---

## 5. Fusion + cross-city (Phase 6) — Figure 4, Figure 6

Long training; use `xtraffic/models/gnn/colab_fusion_train.ipynb` on a T4.
It builds the 3 modality sidecars, retrains the fusion model, and restores the
traffic-only baseline. Then, locally:

```bash
python3 -m xtraffic.evaluation.fusion_comparison --dataset metr_la   # fusion vs traffic-only
python3 -m xtraffic.evaluation.cross_city --source metr_la --target chicago   # Figure 6
```

Writes `results/fusion/comparison.json` and `results/cross_city/*.json`.

---

## 6. Ablations (Phase 7) — Table 4

Multi-seed; Colab-scale for the model variants. Model + pipeline ablations:

```bash
python3 -m xtraffic.evaluation.ablations --config xtraffic/configs/ablations.yaml
# smoke: add  --model-only --smoke
```

Writes `results/ablations/model_ablations.tex` (→ **Table 4**).

---

## 7. Human study (Phase 8) — Table 5, Figure 7

Build scenarios (needs ckpt + Ollama), run the app for evaluators, analyze:

```bash
python3 -m xtraffic.evaluation.human_study.scenarios      # 24-scenario grid + assignment
python3 -m xtraffic.evaluation.human_study.app            # FastAPI app, evaluators use browser
python3 -m xtraffic.evaluation.human_study.analyze --anchors   # Table 5 + external-anchor block
```

To pilot the full loop with **no checkpoint/Ollama**:

```bash
python3 -m xtraffic.evaluation.human_study.demo_data      # synthetic scenarios
# ...complete a few scenarios in the app, then:
python3 -m xtraffic.evaluation.human_study.analyze
```

Writes `results/human_study/analysis_report.json` (→ **Table 5, Figure 7**).

---

## 8. Build every paper artifact (Phase 9)

```bash
python3 -m xtraffic.evaluation.make_paper_artifacts
```

Reads whatever the steps above logged into `evaluation/results/` and writes:

- `evaluation/paper/figures/` — Figure 1 (architecture, SVG) … Figure 7 (PDF)
- `evaluation/paper/tables/`  — Table 1 … Table 5 (LaTeX booktabs, `\input`-ready)

Any result not yet produced (e.g. the Colab-owed cross-city run) yields a clearly
labelled **PENDING** placeholder instead of crashing — the missing numbers are
loud and visible, never silently absent. `\input{}` the tables and
`\includegraphics{}` the figures straight into the manuscript.

---

## What is verified vs. owed

Runs locally today from committed results: steps **3, 4, 8** (given a checkpoint +
Ollama), and step **8** always (it regenerates the paper skeleton from whatever is
logged). Owed on Colab (documented above, not yet in the paper as final numbers):
the full 100-epoch traffic-only + fusion training, the cross-city transfer table,
the Chicago faithfulness study, and the multi-seed ablations. Each drops its
result file into `evaluation/results/`; re-running step 8 picks it up with no code
changes.
