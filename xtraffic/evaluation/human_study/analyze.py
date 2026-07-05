"""Phase 8 — analyse the human decision-quality study (Contribution #3).

Reads the app's JSONL log and reports, per presentation condition
(RAW / XAI / XTRAFFIC):
  * DECISION ACCURACY  — fraction of scenarios where the evaluator chose the
    ground-truth-optimal intervention (from the simulation).
  * CONFIDENCE (1-7)   — self-reported confidence in the choice.
  * USEFULNESS (1-7)   — self-reported usefulness of the information shown.

Statistics — WHY these choices (I need to defend this section):
  * n is small (a professor + a few colleagues), so we do NOT assume normality
    and avoid the paired t-test. We use the WILCOXON SIGNED-RANK TEST, the
    nonparametric paired test.
  * The design is WITHIN-SUBJECTS (Latin square): every evaluator experiences all
    three conditions, across different scenarios. So we pair BY EVALUATOR — for
    each evaluator we compute their mean accuracy/confidence/usefulness under each
    condition, then run Wilcoxon on those paired per-evaluator means for the two
    comparisons that matter: XTRAFFIC vs RAW (does the full pipeline beat raw
    numbers?) and XTRAFFIC vs XAI (does the LLM layer add value over explainability
    alone?).
  * We report EFFECT SIZE, not just p — with tiny n a p-value is underpowered and
    can mislead. We use the matched-pairs rank-biserial correlation
    r = Z / sqrt(N_pairs) style estimate (reported alongside the raw median
    difference), because effect size is what a reviewer can actually interpret at
    this sample size.

Honest caveat printed with the report: with a handful of evaluators these tests
are exploratory; the effect sizes and per-condition means carry the story, the
p-values are indicative only.

EXTERNAL-ANCHOR VALIDATION (addresses the circular-ground-truth limitation)
---------------------------------------------------------------------------
The simulation ground truth is MODEL-IN-THE-LOOP: the same GNN both produces the
predictions evaluators see and scores which intervention is "best". That is
internally consistent but circular by construction. To break the circle at a few
points, an optional `external_anchors.json` lets a domain expert (my professor)
mark scenarios where they know the REAL-WORLD correct intervention from
professional experience. When present, we ALSO report decision accuracy scored
against those expert anchors — a small out-of-model validation set. See
external_anchors.example.json for the fill-in template.

Run:
  python -m xtraffic.evaluation.human_study.analyze
  python -m xtraffic.evaluation.human_study.analyze --exclude-degenerate
  python -m xtraffic.evaluation.human_study.analyze --anchors path/to/external_anchors.json

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ...utils.io_utils import PKG_ROOT

DATA_DIR = os.path.join(PKG_ROOT, "evaluation", "results", "human_study")
RESPONSES_PATH = os.path.join(DATA_DIR, "responses.jsonl")
CONDITIONS = ["RAW", "XAI", "XTRAFFIC"]
METRICS = ["correct", "confidence", "usefulness"]   # 'correct' is 0/1 -> accuracy


def _load_responses(exclude_degenerate: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    degenerate = set()
    if exclude_degenerate:
        with open(os.path.join(DATA_DIR, "scenarios.json")) as f:
            for s in json.load(f)["scenarios"]:
                if s.get("no_action_is_best"):
                    degenerate.add(s["scenario_id"])
    with open(RESPONSES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if exclude_degenerate and r["scenario_id"] in degenerate:
                continue
            r["correct"] = 1.0 if r["correct"] else 0.0
            rows.append(r)
    return rows


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _load_anchors(path: Optional[str]) -> Dict[str, str]:
    """scenario_id -> expert-declared correct intervention (external ground truth).

    Optional. Each value may be a bare intervention name or an object with an
    `intervention` key (the template also carries `source`/`rationale` for the
    paper's audit trail). Missing/absent file -> {} (no external validation).
    """
    default = os.path.join(DATA_DIR, "external_anchors.json")
    path = path or (default if os.path.exists(default) else None)
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    anchors: Dict[str, str] = {}
    for sid, val in raw.items():
        if sid.startswith("_"):                # skip _README / meta keys
            continue
        if isinstance(val, dict):
            iv = val.get("intervention")
        else:
            iv = val
        if iv:
            anchors[sid] = iv
    return anchors


def _anchor_accuracy(rows: List[Dict[str, Any]], anchors: Dict[str, str]
                     ) -> Dict[str, Dict[str, float]]:
    """Per-condition accuracy scored against EXPERT anchors, not the simulation.

    A response counts as correct here iff the evaluator's chosen intervention
    equals the expert anchor for that scenario. Only responses on anchored
    scenarios that report a `chosen` intervention are included.
    """
    out: Dict[str, Dict[str, float]] = {}
    for c in CONDITIONS:
        hits, n = 0, 0
        for r in rows:
            if r["condition"] != c or r["scenario_id"] not in anchors:
                continue
            chosen = r.get("choice")
            if chosen is None:
                continue
            n += 1
            if chosen == anchors[r["scenario_id"]]:
                hits += 1
        out[c] = {"n": n, "accuracy": round(hits / n, 3) if n else float("nan")}
    return out


def _per_condition(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Overall mean of each metric per condition (pooled over all responses)."""
    out: Dict[str, Dict[str, float]] = {}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        out[c] = {"n": len(sub),
                  **{m: round(_mean([r[m] for r in sub]), 3) for m in METRICS}}
    return out


def _per_evaluator_means(rows: List[Dict[str, Any]], metric: str
                         ) -> Dict[str, Dict[str, float]]:
    """evaluator -> {condition: mean(metric)} — the paired units for Wilcoxon."""
    evs = sorted({r["evaluator"] for r in rows})
    out: Dict[str, Dict[str, float]] = {}
    for ev in evs:
        out[ev] = {}
        for c in CONDITIONS:
            vals = [r[metric] for r in rows if r["evaluator"] == ev and r["condition"] == c]
            if vals:
                out[ev][c] = _mean(vals)
    return out


def _wilcoxon(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank with a matched-pairs effect size.

    Returns median difference (a-b), the statistic, p-value, and rank-biserial
    effect size. Falls back gracefully when scipy is absent or n is too small.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    diffs = [x - y for x, y in pairs]
    nonzero = [d for d in diffs if d != 0]
    med = sorted(diffs)[len(diffs) // 2] if diffs else float("nan")
    result: Dict[str, Any] = {"n_pairs": len(pairs), "median_diff": round(med, 3)}
    if len(nonzero) < 1:
        result.update({"stat": None, "p_value": None, "effect_size_r": None,
                       "note": "no non-zero differences"})
        return result
    try:
        from scipy.stats import wilcoxon
        # zero_method='wilcox' drops zero-diffs (standard); may warn at tiny n.
        stat, p = wilcoxon(a[:len(pairs)], b[:len(pairs)]) if False else wilcoxon(
            [x for x, _ in pairs], [y for _, y in pairs])
        # Rank-biserial for signed-rank: r = W+/(W+ + W-) mapped to [-1,1]; we use
        # the common approximation r = stat_diff / sum_ranks via effect from stat.
        n = len(nonzero)
        total_rank = n * (n + 1) / 2.0
        # scipy returns the smaller of W+/W-; recover a signed rank-biserial.
        r = 1.0 - (2.0 * stat) / total_rank if total_rank else 0.0
        # sign it by the direction of the median difference.
        r = abs(r) * (1.0 if med >= 0 else -1.0)
        result.update({"stat": float(stat), "p_value": round(float(p), 4),
                       "effect_size_r": round(float(r), 3)})
    except Exception as e:                                # pragma: no cover
        result.update({"stat": None, "p_value": None, "effect_size_r": None,
                       "note": "scipy unavailable or failed: {}".format(e)})
    return result


def _paired_vectors(means: Dict[str, Dict[str, float]], c1: str, c2: str
                    ) -> Tuple[List[float], List[float]]:
    a, b = [], []
    for ev, per in means.items():
        if c1 in per and c2 in per:
            a.append(per[c1])
            b.append(per[c2])
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-degenerate", action="store_true",
                    help="drop scenarios whose ground truth is 'no_action'.")
    ap.add_argument("--anchors", default=None,
                    help="optional external_anchors.json (expert ground truth) "
                         "for out-of-model validation; defaults to the one in the "
                         "results dir if present.")
    args = ap.parse_args()

    if not os.path.exists(RESPONSES_PATH):
        print("No responses yet at {} — run the app first.".format(RESPONSES_PATH))
        return
    rows = _load_responses(args.exclude_degenerate)
    print("Loaded {} responses from {} evaluators.".format(
        len(rows), len({r["evaluator"] for r in rows})))

    # --- per-condition summary --------------------------------------------
    summary = _per_condition(rows)
    print("\n=== PER-CONDITION (pooled) ===")
    print("{:10s}{:>6s}{:>12s}{:>12s}{:>12s}".format(
        "condition", "n", "accuracy", "confidence", "usefulness"))
    for c in CONDITIONS:
        s = summary[c]
        print("{:10s}{:>6d}{:>12.3f}{:>12.3f}{:>12.3f}".format(
            c, s["n"], s["correct"], s["confidence"], s["usefulness"]))

    # --- paired tests (per-evaluator means) -------------------------------
    comparisons = [("XTRAFFIC", "RAW"), ("XTRAFFIC", "XAI"), ("XAI", "RAW")]
    stats: Dict[str, Any] = {}
    print("\n=== PAIRED WILCOXON (per-evaluator means; effect size = rank-biserial) ===")
    for metric in METRICS:
        means = _per_evaluator_means(rows, metric)
        label = "accuracy" if metric == "correct" else metric
        stats[label] = {}
        for c1, c2 in comparisons:
            a, b = _paired_vectors(means, c1, c2)
            res = _wilcoxon(a, b)
            stats[label]["{}_vs_{}".format(c1, c2)] = res
            print("  {:11s} {:>9s} vs {:<9s}  median_diff={:+.3f}  p={}  r={}  (n_pairs={})"
                  .format(label, c1, c2, res["median_diff"],
                          res["p_value"], res["effect_size_r"], res["n_pairs"]))

    # --- external-anchor validation (breaks the circular ground truth) ----
    anchors = _load_anchors(args.anchors)
    anchor_report: Optional[Dict[str, Dict[str, float]]] = None
    if anchors:
        anchor_report = _anchor_accuracy(rows, anchors)
        n_anchored = len({r["scenario_id"] for r in rows
                          if r["scenario_id"] in anchors})
        print("\n=== EXTERNAL-ANCHOR ACCURACY (vs EXPERT ground truth, "
              "not the simulation) ===")
        print("{} scenario(s) carry an expert anchor.".format(n_anchored))
        print("{:10s}{:>6s}{:>12s}".format("condition", "n", "accuracy"))
        for c in CONDITIONS:
            s = anchor_report[c]
            print("{:10s}{:>6d}{:>12.3f}".format(c, s["n"], s["accuracy"]))
        print("(Small out-of-model validation set — does the pipeline still help "
              "when 'best' is judged by a human expert, not the model?)")
    else:
        print("\n(No external_anchors.json found — simulation ground truth only. "
              "Add expert anchors to validate out-of-model; see "
              "external_anchors.example.json.)")

    # --- write report -----------------------------------------------------
    out_json = os.path.join(DATA_DIR, "analysis_report.json")
    with open(out_json, "w") as f:
        json.dump({"per_condition": summary, "paired_tests": stats,
                   "external_anchor_accuracy": anchor_report,
                   "n_responses": len(rows),
                   "excluded_degenerate": args.exclude_degenerate}, f, indent=2)
    out_csv = os.path.join(DATA_DIR, "analysis_per_condition.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n", "accuracy", "confidence", "usefulness"])
        for c in CONDITIONS:
            s = summary[c]
            w.writerow([c, s["n"], s["correct"], s["confidence"], s["usefulness"]])

    print("\nCAVEAT: with a small panel these p-values are exploratory; read the "
          "effect sizes and per-condition means as the primary evidence.")
    print("Wrote:\n  {}\n  {}".format(out_json, out_csv))


if __name__ == "__main__":
    main()
