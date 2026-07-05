# Phase 8 — Human Decision-Quality Study

Tooling for the expert study behind **Contribution #3**: does the full XTraffic
pipeline help a traffic operator make *better decisions* than raw predictions or
explainability output alone?

## The design in one paragraph

Each evaluator works through a set of incident-response scenarios. For every
scenario they see the information for one of three **presentation conditions** and
must choose a single intervention, then rate their confidence and how useful the
information was. A **Latin square** balances conditions across evaluators so each
person sees every scenario exactly once and each scenario appears under every
condition. We then compare decision accuracy / confidence / usefulness across
conditions.

| Condition  | What the evaluator sees                                            |
|------------|-------------------------------------------------------------------|
| `RAW`      | Prediction numbers only (control)                                 |
| `XAI`      | Prediction + explanation figure + node-importance table (SOTA)    |
| `XTRAFFIC` | Full pipeline: prediction + explanation + LLM advisory            |

## Ground truth (read this — it's a stated limitation)

We don't have a field experiment, so "the best intervention" is defined by
**simulation** (`simulate.py`): each candidate action is applied to the trained
GNN's own input window and the model's 30-min-ahead network delay is read off; the
lowest-delay action is ground truth. This is a **model-in-the-loop proxy**, not
observed reality — internally consistent and reproducible, but it inherits the
model's biases. The paper must state this honestly.

## Files

| File            | Role                                                               |
|-----------------|--------------------------------------------------------------------|
| `simulate.py`   | Intervention simulation → ground-truth-optimal action per scenario |
| `scenarios.py`  | Build 24 scenarios + RAW/XAI/XTRAFFIC bundles + Latin square       |
| `demo_data.py`  | Synthetic scenario set (no ML) to pilot the app anywhere           |
| `app.py`        | Minimal FastAPI evaluation app; logs to `responses.jsonl`          |
| `analyze.py`    | Per-condition metrics + paired Wilcoxon + effect sizes             |

Config: [`configs/human_study.yaml`](../../configs/human_study.yaml) — 24-scenario
grid, simulation knobs, conditions, Likert scale. No magic numbers in code.

## How to run

**1. Build the scenario set** (needs the trained checkpoint + local Ollama):

```bash
python -m xtraffic.evaluation.human_study.scenarios            # full 24 + advisories
python -m xtraffic.evaluation.human_study.scenarios --no-llm   # skip Ollama
```

Writes `evaluation/results/human_study/scenarios.json`, `assignment.json`, and one
explanation figure per scenario.

**2. Run the app** (one command; a professor needs no ML installed):

```bash
python -m xtraffic.evaluation.human_study.app
# open http://localhost:8000, enter name, pick participant number, go
```

Every submission appends one line to `responses.jsonl`. Participants can close the
tab and resume — progress is derived from the log.

**3. Analyse:**

```bash
python -m xtraffic.evaluation.human_study.analyze
python -m xtraffic.evaluation.human_study.analyze --exclude-degenerate  # drop 'no_action' GT
```

## Pilot it today (no checkpoint required)

The real content is owed pending the Colab checkpoint + Ollama. To exercise the
whole app → JSONL → analysis loop right now with **clearly-labelled fake data**:

```bash
python -m xtraffic.evaluation.human_study.demo_data   # writes DEMO scenarios.json
python -m xtraffic.evaluation.human_study.app         # pilot as yourself
python -m xtraffic.evaluation.human_study.analyze     # sane report from your pilot
```

Demo data is tagged `dataset: "DEMO"` — never use it for the paper.

## Statistics (defend this in the paper)

- **Nonparametric**: small panel → no normality assumption → **Wilcoxon
  signed-rank**, not a t-test.
- **Paired by evaluator**: within-subjects design → each evaluator's mean per
  condition is one paired unit; compares `XTRAFFIC vs RAW` and `XTRAFFIC vs XAI`.
- **Effect sizes reported** (matched-pairs rank-biserial) because at tiny n a
  p-value is underpowered — the effect sizes and per-condition means carry the
  story.
