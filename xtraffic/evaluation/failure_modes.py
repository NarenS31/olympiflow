"""Phase 13 — FAILURE-MODE TAXONOMY for condition A (the full pipeline).

This becomes Section 6 of the paper. A rigorous, honest account of WHEN our best
system (condition A: prediction + explanation + city context) still produces an
unfaithful advisory is worth more to a reviewer than a page of claimed strengths.
It also closes the loop on the SELF_ATTRIBUTION error we flagged in Phase 4: does
it survive into the tuned condition-A study, or not?

WHAT THIS DOES
--------------
1. Loads the Phase-5 faithfulness results (evaluation/results/faithfulness/) and
   keeps CONDITION A only -- we are characterising when the BEST system fails.
2. Finds every A scenario that FAILS: hallucination present (any cited cause not
   in the explanation's top-k) OR faithfulness F1 < 0.5.
3. Classifies each failure into exactly one of six categories (priority order,
   documented below), using the explanation JSON, the regenerated advisory, the
   logged metrics, and the road graph.
4. Reports per category: frequency (% of all A scenarios), mean F1, mean
   hallucination, one representative example (REGION NAMES only, no raw ids), a
   one-sentence hypothesised cause, and one concrete mitigation.
5. Writes a LaTeX table + a JSON with full details + a console summary.

WHY WE REGENERATE THE ADVISORIES (an honest limitation, handled)
----------------------------------------------------------------
Phase 5 logged the per-scenario METRICS but not the LLM's cited causes. Three of
the six categories (self-attribution, geographic hallucination, temporal
confusion) need the actual cited text, so for each FAILURE we re-run condition A
on the CACHED explanation (no explainer cost, ~7 LLM calls) and recompute the
per-cause resolution. The failure SET and the reported F1/hallucination remain the
Phase-5 LOGGED numbers (authoritative); the regenerated advisory only supplies the
cited-cause detail needed to classify. At temperature 0.1 the advisory is close to
the original; any disagreement (e.g. logged precision 1.0 but a regenerated miss)
is recorded in the JSON for transparency.

CLASSIFICATION RULES (priority order -- first match wins, so every failure gets
exactly one label even though the phenomena can overlap):
  1. SELF_ATTRIBUTION      : a cited cause resolves to a node set containing the
                             TARGET node (the LLM blames the target for itself).
  2. FREE_FLOW_FAILURE     : the model predicts the target at FREE FLOW
                             (predicted speed >= FREE_FLOW_MPH) -- there is no real
                             congestion to explain, so any surfaced cause is noise.
                             We use PREDICTED SPEED, not just the window's
                             congestion tercile, because a missing target sensor
                             (0 mph sentinel) sits in a "high-congestion" window yet
                             is predicted back to free flow -- window congestion
                             would mislabel it. (Flagged deviation from the literal
                             "congestion low" rule, with reason.)
  3. GEOGRAPHIC_HALLUCINATION : a cited cause misses the top-k AND is unresolved OR
                             lies > GEO_HOPS graph hops from every top-k node and
                             the target (an invented, far-away place).
  4. TEMPORAL_CONFUSION    : the cited nodes are correct (all hit the top-k) but the
                             advisory states a propagation lag that contradicts the
                             explanation's propagation_lag_minutes (>10% off).
  5. CONFIDENCE_MISMATCH   : explanation_confidence > 0.8 but F1 < 0.5 (the explainer
                             was over-confident about a bad explanation).
  6. OTHER                 : none of the above -- on this dataset this is dominated
                             by UNDER-CITATION (precision high, recall low: the LLM
                             names only the strongest source and omits the rest).

Run:
  python -m xtraffic.evaluation.failure_modes            # regenerate + classify (real LLM)
  python -m xtraffic.evaluation.failure_modes --mock-llm # harness check, no Ollama

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from ..models.advisor.advisor import Advisor
from ..models.gnn.loaders import load_adjacency, load_node_meta
from ..utils.io_utils import PKG_ROOT
from .faithfulness import NodeTable, score_advisory

# ---- Classification thresholds (no magic numbers buried in code). -----------
FREE_FLOW_MPH = 50.0    # target predicted at/above this == no congestion to explain
GEO_HOPS = 2            # a cited node farther than this from top-k/target == "not adjacent"
CONF_HIGH = 0.8         # "high" explanation_confidence for the confidence-mismatch rule
F1_LOW = 0.5            # a scenario with F1 below this is a failure
LAG_TOL = 0.10          # a stated lag within +/-10% of the explanation's is "consistent"

# ---- The six categories: (key, short definition, hypothesis, mitigation). ---
# Order here is the DISPLAY order (table rows); the PRIORITY order is separate.
CATEGORIES: List[Any] = [
    ("SELF_ATTRIBUTION", "LLM cites the target as its own cause",
     "When the upstream causal signal is weak, the model restates the target region "
     "as a cause -- the same error seen in the untuned Phase-4 advisor.",
     "Drop the target node/region from the prompt's cause list and add a system rule "
     "forbidding self-citation; reject any self-citation in the advisory validator."),
    ("TEMPORAL_CONFUSION", "correct nodes, wrong time relationship",
     "With several near-equal-importance sources the LLM garbles the propagation lag "
     "or the lead/lag ordering between source and target.",
     "Render each source's propagation lag explicitly in the prompt and require the "
     "advisory to echo it; validate stated lags against the explanation JSON."),
    ("GEOGRAPHIC_HALLUCINATION", "cited nodes not in top-k and >2 hops away",
     "The LLM pattern-matches to well-known corridors that are not in the "
     "explanation's graph neighbourhood.",
     "Constrain citations to an enumerated allow-list of the top-k region names and "
     "reject out-of-list causes in the validator."),
    ("CONFIDENCE_MISMATCH", "explanation_confidence > 0.8 but F1 < 0.5",
     "The explainer's stability-based confidence over-states reliability on "
     "diffuse-cause windows where many nodes share the influence.",
     "Recalibrate explanation_confidence against realised faithfulness and gate "
     "advisories whose confidence is below a faithfulness-predictive threshold."),
    ("FREE_FLOW_FAILURE", "target predicted at free flow -- no real spatial cause",
     "With the target predicted at free flow there is no congestion to explain, so "
     "the explainer surfaces noise that the LLM dutifully narrates.",
     "Detect free flow (predicted speed >= {:.0f} mph or a missing target sensor) and "
     "suppress causal advisories, returning 'no actionable congestion'.".format(FREE_FLOW_MPH)),
    ("OTHER", "does not fit the above (here: under-citation)",
     "The LLM names only the single strongest source and omits the rest of the "
     "top-k, so recall -- not precision -- collapses.",
     "Instruct the advisory to address every top-k source or explicitly justify each "
     "omission, and score recall inside the self-correction loop."),
]
# First-match priority (see module docstring for the reasoning).
PRIORITY = ["SELF_ATTRIBUTION", "FREE_FLOW_FAILURE", "GEOGRAPHIC_HALLUCINATION",
            "TEMPORAL_CONFUSION", "CONFIDENCE_MISMATCH", "OTHER"]


# ===========================================================================
# Loading the Phase-5 condition-A results.
# ===========================================================================
def _f(v: str) -> float:
    """Parse a CSV cell to float; blank / 'None' -> NaN (quant_fidelity can be None)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_condition_a(results_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(results_dir, "faithfulness_per_scenario.csv")
    if not os.path.exists(path):
        raise SystemExit(
            "[phase13] {} not found -- run the Phase-5 study first "
            "(python -m xtraffic.evaluation.run_faithfulness_study).".format(path))
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] == "A"]
    for r in rows:
        for k in ("cause_precision", "cause_recall", "faithfulness_f1",
                  "hallucination_rate", "quantitative_fidelity"):
            r[k] = _f(r[k])
        r["sample_index"] = int(r["sample_index"])
        r["target_node"] = int(r["target_node"])
    return rows


def is_failure(row: Dict[str, Any]) -> bool:
    """A condition-A scenario FAILS if it hallucinated any cause or F1 < 0.5."""
    hallucinated = (row["hallucination_rate"] > 0) if row["hallucination_rate"] == row["hallucination_rate"] else False
    low_f1 = (row["faithfulness_f1"] < F1_LOW) if row["faithfulness_f1"] == row["faithfulness_f1"] else False
    return hallucinated or low_f1


# ===========================================================================
# Explanation cache + advisory regeneration.
# ===========================================================================
def load_explanation(results_dir: str, dataset: str, idx: int, node: int
                     ) -> Optional[Dict[str, Any]]:
    path = os.path.join(results_dir, "explanations_cache",
                        "{}_{}_{}.json".format(dataset, idx, node))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class MockAdvisor:
    """Ollama-free stand-in for --mock-llm: cites the top-2 nodes (grounded), so the
    harness (load -> regenerate -> classify -> report) runs without a live model."""

    def __init__(self, model: str = "mock"):
        self.model = model

    def advise_condition(self, exp: Dict[str, Any], condition: str) -> Dict[str, Any]:
        top = exp.get("top_nodes", [])
        causes = [{"location": n["node_name"], "resolved_node_id": n["node_id"]}
                  for n in top[:2]]
        adv = {"reasoning": "Mock reasoning citing the top sources.",
               "cited_causes": causes,
               "recommendations": [{"action": "mock", "location": "mock",
                   "time_window_minutes": 15, "expected_effect": "mock",
                   "grounded_in": ["mock"]} for _ in range(3)]}
        return {"advisory": adv, "condition": condition, "model": self.model}


def get_or_make_advisory(advisor: Any, exp: Dict[str, Any], cache_dir: str,
                         idx: int, node: int) -> Dict[str, Any]:
    """Regenerate the condition-A advisory for one failure and cache it (resumable)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "{}_{}.json".format(idx, node))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    res = advisor.advise_condition(exp, "A")
    with open(path, "w") as f:
        json.dump(res["advisory"], f, indent=2)
    return res["advisory"]


# ===========================================================================
# Graph hops (for the geographic-hallucination test).
# ===========================================================================
def bfs_hops(adj_bool: np.ndarray, sources: Set[int]) -> np.ndarray:
    """Multi-source BFS: min hop count from any node in `sources`. inf if unreachable."""
    n = adj_bool.shape[0]
    dist = np.full(n, np.inf)
    dq: "collections.deque[int]" = collections.deque()
    for s in sources:
        if 0 <= s < n:
            dist[s] = 0
            dq.append(s)
    while dq:
        u = dq.popleft()
        for v in np.where(adj_bool[u])[0]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                dq.append(int(v))
    return dist


# ===========================================================================
# The classifier.
# ===========================================================================
def _region(name: str) -> str:
    """Strip the '(sensor 123, lat, lon)' tail so we show REGION NAMES only."""
    return name.split(" (sensor")[0].strip()


# A number+"min" is only a PROPAGATION-LAG claim if lead/lag language sits right
# next to it. This is what stops us mislabeling the forecast HORIZON ("...will
# reach the target in 30 minutes") or an intervention window ("...for 15 minutes")
# as a temporal error -- a real false positive we caught while building this.
_LAG_WORDS = ("lag", "earlier", "later", "minutes before", "minutes after",
              "minutes ago", "ahead", "lead time", "leads", "precede", "prior")


def _lag_claims(text: str, horizon_min: float) -> List[float]:
    """Stated propagation-lag figures (minutes) in the advisory reasoning: a
    '<n> min' that co-occurs with lead/lag language and is NOT the forecast horizon
    or the standard 15-minute intervention window."""
    t = (text or "").lower()
    claims: List[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?min", t):
        n = float(m.group(1))
        if n == horizon_min or n == 15.0:
            continue
        ctx = t[max(0, m.start() - 40): min(len(t), m.end() + 40)]  # +/-40 chars
        if any(w in ctx for w in _LAG_WORDS):
            claims.append(n)
    return claims


def classify(row: Dict[str, Any], exp: Dict[str, Any], advisory: Dict[str, Any],
             per_cause: List[Dict[str, Any]], hops: np.ndarray) -> str:
    """Assign one category by the documented priority order."""
    target = exp["prediction"]["node_id"]
    pred_speed = float(exp["prediction"]["predicted_speed_mph"])
    topk = {int(n["node_id"]) for n in exp.get("top_nodes", [])}
    lag = float(exp.get("propagation_lag_minutes", 0.0))
    conf = float(exp.get("explanation_confidence", 0.0))
    f1 = row["faithfulness_f1"]

    # 1. SELF_ATTRIBUTION — a cited cause that MISSES the top-k yet resolves to the
    #    target: the LLM blamed the target for itself. We require the miss because
    #    region-level credit means a citation of the target's REGION can legitimately
    #    hit a real top-k node in that same region (e.g. two San-Fernando-Valley
    #    sensors) -- that is crediting the real cause, not self-attribution. If the
    #    citation covers the target AND a genuine top-k node we give it the benefit of
    #    the doubt (documented limitation).
    if any((not pc.get("hit_topk")) and target in set(pc.get("resolved_node_ids", []))
           for pc in per_cause):
        return "SELF_ATTRIBUTION"

    # 2. FREE_FLOW_FAILURE — target predicted at free flow (or missing sensor -> free flow).
    if pred_speed >= FREE_FLOW_MPH:
        return "FREE_FLOW_FAILURE"

    # 3. GEOGRAPHIC_HALLUCINATION — a cited cause that misses top-k and is far/absent.
    for pc in per_cause:
        if pc.get("hit_topk"):
            continue                              # cited a real cause -> not hallucinated
        nids = pc.get("resolved_node_ids", [])
        if not nids:                              # unresolved -> invented place
            return "GEOGRAPHIC_HALLUCINATION"
        if min(hops[n] for n in nids) > GEO_HOPS:  # resolved but far from the explanation
            return "GEOGRAPHIC_HALLUCINATION"

    # 4. TEMPORAL_CONFUSION — correct nodes, but a stated lag contradicts the explanation.
    all_correct = all(pc.get("hit_topk") for pc in per_cause) and len(per_cause) > 0
    if all_correct and lag > 0:
        horizon = float(exp["prediction"].get("horizon_minutes", 0))
        claims = _lag_claims(advisory.get("reasoning", ""), horizon)
        if claims and all(abs(s - lag) > LAG_TOL * max(lag, 1.0) for s in claims):
            return "TEMPORAL_CONFUSION"

    # 5. CONFIDENCE_MISMATCH — explainer was over-confident about a bad explanation.
    if conf > CONF_HIGH and f1 == f1 and f1 < F1_LOW:
        return "CONFIDENCE_MISMATCH"

    # 6. OTHER — residual (here: under-citation, high precision / low recall).
    return "OTHER"


# ===========================================================================
# Representative example (region names only).
# ===========================================================================
def example_of(row: Dict[str, Any], exp: Dict[str, Any],
               advisory: Dict[str, Any]) -> Dict[str, Any]:
    topk_regions = []
    for n in exp.get("top_nodes", []):
        r = _region(n["node_name"])
        if r not in topk_regions:
            topk_regions.append(r)
    cited_regions = [_region(c.get("location", "")) for c in advisory.get("cited_causes", [])]
    return {
        "sample_index": row["sample_index"],
        "tod_band": row["tod_band"], "congestion": row["congestion"],
        "target_region": _region(exp["prediction"]["node_name"]),
        "target_predicted_mph": exp["prediction"]["predicted_speed_mph"],
        "explanation_confidence": exp.get("explanation_confidence"),
        "cited_regions": cited_regions,
        "topk_regions": topk_regions,
        "faithfulness_f1": round(row["faithfulness_f1"], 3),
        "hallucination_rate": round(row["hallucination_rate"], 3),
    }


# ===========================================================================
# Output.
# ===========================================================================
def build_table(cat_stats: Dict[str, Dict[str, Any]], n_total: int,
                n_fail: int, out_tex: str) -> None:
    lines = [
        "% Auto-generated by evaluation/failure_modes.py (Phase 13). Do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Condition-A (full pipeline) failure taxonomy on METR-LA "
        r"($n=" + str(n_total) + r"$ scenarios; " + str(n_fail) + r" failures, "
        + "{:.1f}".format(100.0 * n_fail / max(n_total, 1)) + r"\%). A failure is any "
        r"hallucinated cause or faithfulness F1 $<0.5$. Frequency is over all "
        r"condition-A scenarios; mean F1 and hallucination are over the scenarios in "
        r"each category (Phase-5 logged metrics). Categories with 0 occurrences are "
        r"reported --- their absence is itself a result.}",
        r"\label{tab:failure_modes}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Failure mode & Count & Freq. (\%) & Mean F1 & Mean halluc. \\",
        r"\midrule",
    ]
    for key, _defn, _hyp, _mit in CATEGORIES:
        s = cat_stats[key]
        label = key.replace("_", r"\_")
        if s["count"] == 0:
            lines.append("{} & 0 & 0.0 & -- & -- \\\\".format(label))
        else:
            lines.append("{} & {} & {:.1f} & {:.3f} & {:.3f} \\\\".format(
                label, s["count"], s["freq_pct"], s["mean_f1"], s["mean_halluc"]))
    lines += [
        r"\midrule",
        r"\textbf{All failures} & " + str(n_fail) + " & "
        + "{:.1f}".format(100.0 * n_fail / max(n_total, 1)) + r" & -- & -- \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 13 condition-A failure taxonomy")
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--city", default="metr_la")
    ap.add_argument("--mock-llm", action="store_true",
                    help="use a mock advisor (no Ollama) to verify the harness")
    ap.add_argument("--model", default=None, help="override the Ollama advisor model")
    args = ap.parse_args()

    results_dir = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness")
    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "failure_modes")
    os.makedirs(out_dir, exist_ok=True)
    adv_cache = os.path.join(out_dir, "failure_advisories")
    out_tex = os.path.join(PKG_ROOT, "evaluation", "paper", "table_failure_modes.tex")

    rows = load_condition_a(results_dir)
    failures = [r for r in rows if is_failure(r)]
    print("[phase13] condition-A scenarios: {} | failures (halluc>0 or F1<0.5): {} "
          "({:.1f}%)".format(len(rows), len(failures), 100.0 * len(failures) / max(len(rows), 1)))

    # Set up the advisor (real or mock) + resolver table + road graph.
    if args.mock_llm:
        advisor: Any = MockAdvisor()
    elif args.model:
        import copy

        from ..models.advisor.advisor import load_advisor_config
        cfg = copy.deepcopy(load_advisor_config())
        cfg.setdefault("ollama", {})["model"] = args.model
        advisor = Advisor(args.city, cfg=cfg)
    else:
        advisor = Advisor(args.city)
    table = NodeTable(load_node_meta(args.dataset))
    adj_bool = (load_adjacency(args.dataset).numpy() > 0)     # [N, N] road graph

    # Classify every failure.
    classified: List[Dict[str, Any]] = []
    for i, row in enumerate(failures):
        idx, node = row["sample_index"], row["target_node"]
        exp = load_explanation(results_dir, args.dataset, idx, node)
        if exp is None:
            print("  [skip] no cached explanation for {}_{}".format(idx, node))
            continue
        advisory = get_or_make_advisory(advisor, exp, adv_cache, idx, node)
        metrics = score_advisory(exp, advisory, table)         # -> per_cause detail
        topk = {int(n["node_id"]) for n in exp.get("top_nodes", [])}
        hops = bfs_hops(adj_bool, topk | {exp["prediction"]["node_id"]})
        category = classify(row, exp, advisory, metrics["per_cause"], hops)
        classified.append({
            "sample_index": idx, "target_node": node, "category": category,
            "logged_f1": row["faithfulness_f1"], "logged_halluc": row["hallucination_rate"],
            "regenerated_precision": metrics["cause_precision"],
            "regenerated_hallucination": metrics["hallucination_rate"],
            "example": example_of(row, exp, advisory),
        })
        print("  [{}/{}] idx={} {}/{} -> {}".format(
            i + 1, len(failures), idx, row["tod_band"], row["congestion"], category))

    # Aggregate per category.
    cat_stats: Dict[str, Dict[str, Any]] = {}
    for key, defn, hyp, mit in CATEGORIES:
        members = [c for c in classified if c["category"] == key]
        if members:
            f1s = [m["logged_f1"] for m in members if m["logged_f1"] == m["logged_f1"]]
            hs = [m["logged_halluc"] for m in members if m["logged_halluc"] == m["logged_halluc"]]
            rep = members[0]["example"]
        else:
            f1s, hs, rep = [], [], None
        cat_stats[key] = {
            "definition": defn, "count": len(members),
            "freq_pct": round(100.0 * len(members) / max(len(rows), 1), 2),
            "mean_f1": round(sum(f1s) / len(f1s), 3) if f1s else None,
            "mean_halluc": round(sum(hs) / len(hs), 3) if hs else None,
            "representative_example": rep,
            "hypothesized_cause": hyp,
            "mitigation": mit,
            "member_indices": [m["sample_index"] for m in members],
        }

    # Write JSON (full detail) + LaTeX table.
    summary = {
        "dataset": args.dataset,
        "n_condition_a": len(rows), "n_failures": len(failures),
        "failure_rate_pct": round(100.0 * len(failures) / max(len(rows), 1), 2),
        "thresholds": {"FREE_FLOW_MPH": FREE_FLOW_MPH, "GEO_HOPS": GEO_HOPS,
                       "CONF_HIGH": CONF_HIGH, "F1_LOW": F1_LOW},
        "advisory_source": "mock" if args.mock_llm else getattr(advisor, "model", "?"),
        "note": ("Failure set + F1/hallucination are the Phase-5 LOGGED metrics; "
                 "advisories were regenerated from cached explanations to recover "
                 "cited-cause detail for classification (see module docstring)."),
        "categories": cat_stats,
        "per_failure": classified,
    }
    with open(os.path.join(out_dir, "failure_modes.json"), "w") as f:
        json.dump(summary, f, indent=2)
    build_table(cat_stats, len(rows), len(failures), out_tex)

    # Console report (Section-6 style).
    print("\n" + "=" * 78)
    print("CONDITION-A FAILURE TAXONOMY (n={} A scenarios, {} failures = {:.1f}%)".format(
        len(rows), len(failures), summary["failure_rate_pct"]))
    print("=" * 78)
    for key, defn, _hyp, _mit in CATEGORIES:
        s = cat_stats[key]
        head = "{}  [{}]  n={} ({:.1f}% of A)".format(key, defn, s["count"], s["freq_pct"])
        print("\n" + head)
        if s["count"]:
            print("  mean F1 {} | mean halluc {}".format(s["mean_f1"], s["mean_halluc"]))
            ex = s["representative_example"]
            print("  example: {}/{} target '{}' (pred {} mph, conf {}); cited {} vs top-k {}".format(
                ex["tod_band"], ex["congestion"], ex["target_region"],
                ex["target_predicted_mph"], ex["explanation_confidence"],
                ex["cited_regions"], ex["topk_regions"][:4]))
        print("  hypothesis:  " + s["hypothesized_cause"])
        print("  mitigation:  " + s["mitigation"])
    print("\n[phase13] wrote:")
    print("  " + out_tex)
    print("  " + os.path.join(out_dir, "failure_modes.json"))


if __name__ == "__main__":
    main()
