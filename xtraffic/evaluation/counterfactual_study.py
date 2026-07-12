"""Phase 17 STUDY — evaluate counterfactual explanations at scale.

Runs the Phase-17 counterfactual pipeline on 20 CONGESTED METR-LA scenarios and
reports, exactly as the assignment asks:

  * VALIDITY RATE          — fraction where a flip-to-free-flow was achieved within
                             the perturbation budget.
  * MEAN PERTURBATION BUDGET — average total mph uplift needed (over the valid ones).
  * MEAN NODES CHANGED     — average number of upstream nodes the minimal
                             counterfactual required (over the valid ones).
  * NARRATIVE FAITHFULNESS — the SAME Phase-5 faithfulness metric, but scored
                             against the COUNTERFACTUAL's required-change nodes:
                             does the LLM's plain-language narration cite the nodes
                             the counterfactual says must change, and nothing else?

WHY THIS IS CHEAP: it reuses the Phase-5 stratified sampler at per_stratum=7, i.e.
the SAME 93-scenario population whose explanations are already cached
(explanations_cache/). So the explainer never re-runs; the only new compute is the
gradient-free search (a few hundred fast forward passes per scenario) and one LLM
narration per valid counterfactual.

Run (needs the trained checkpoint + a local Ollama with the advisor model):
  python -m xtraffic.evaluation.counterfactual_study --smoke     # 3 congested, full chain shown
  python -m xtraffic.evaluation.counterfactual_study             # the full 20
  python -m xtraffic.evaluation.counterfactual_study --mock-llm  # harness only, no Ollama
  python -m xtraffic.evaluation.counterfactual_study --model llama3.2:3b   # faster LLM

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch

from ..models.explainer.counterfactual import (
    CounterfactualSearcher, build_counterfactual, counterfactual_reference_explanation,
    format_full_chain, load_counterfactual_config, validate_counterfactual,
)
from ..models.explainer.explain import ExplanationBuilder
from ..models.gnn.loaders import _load_split, load_node_meta
from ..utils.io_utils import PKG_ROOT
from .faithfulness import NodeTable, mean_std, score_advisory
from .run_faithfulness_study import (
    _explanation_cache_path, sample_scenarios,
)

OUT_DIR = os.path.join(PKG_ROOT, "evaluation", "results", "counterfactual")
HORIZON_STEP = 6  # 30-min horizon — matches the Phase-5 explanations we reuse


# ===========================================================================
# Staleness-aware explanation loading.
#
# REAL ISSUE this guards against (flagged): evaluation/run_faithfulness_study.py's
# build_or_load_explanation caches by FILENAME PRESENCE only — it never checks
# whether the checkpoint changed. When the epoch-34 metr_la_best.pt overwrote the
# earlier 3-epoch placeholder (same filename), any explanation cached BEFORE that
# became STALE: its top_nodes / predicted speed reflect the OLD model. Reusing a
# stale explanation would perturb the WRONG "critical" nodes and mis-filter
# congestion. So here we treat the shared Phase-5 cache as valid ONLY when it is
# NEWER than the checkpoint; otherwise we rebuild into OUR OWN cache dir and never
# mutate the shared Phase-5 artifacts.
# ===========================================================================
def _own_cache_path(dataset: str, idx: int, node: int) -> str:
    d = os.path.join(OUT_DIR, "explanations_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{}_{}_{}.json".format(dataset, idx, node))


def get_explanation(builder: ExplanationBuilder, dataset: str,
                    sc: Dict[str, Any], ckpt_mtime: float) -> Dict[str, Any]:
    """Return an explanation CONSISTENT with the live checkpoint. Uses the shared
    Phase-5 cache when it is fresh (newer than the checkpoint); otherwise rebuilds
    into this study's own cache. Rebuilds are logged so a slow run is explained."""
    idx, node = sc["sample_index"], sc["target_node"]
    shared = _explanation_cache_path(dataset, idx, node)
    if os.path.exists(shared) and os.path.getmtime(shared) >= ckpt_mtime:
        with open(shared) as f:
            return json.load(f)                               # fresh shared cache -> free
    own = _own_cache_path(dataset, idx, node)
    if os.path.exists(own):
        with open(own) as f:
            return json.load(f)                               # our rebuild (always post-ckpt)
    print("    (rebuilding stale/missing explanation for idx {} node {} ...)".format(
        idx, node))
    X = _load_split(dataset, "test")[0][idx: idx + 1]
    exp = builder.explain_prediction(X, target_node=node,
                                     horizon_step=HORIZON_STEP, timestamp=sc["timestamp"])
    exp["meta"]["city"] = dataset
    with open(own, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


# ===========================================================================
# Mock advisor for --mock-llm: cites exactly the required-change nodes (perfect
# faithful narration), so the harness (search -> narrate -> score -> table) is
# verifiable end-to-end with no Ollama. Never used for paper numbers.
# ===========================================================================
class MockAdvisor:
    def __init__(self, model: str = "mock"):
        self.model = model

    def advise_counterfactual(self, exp: Dict[str, Any],
                              cf: Dict[str, Any]) -> Dict[str, Any]:
        rc = cf["counterfactual"]["required_changes"]
        causes = [{"location": r["node_name"], "resolved_node_id": r["node_id"]}
                  for r in rc]
        adv = {
            "reasoning": "Mock narration: easing {} upstream location(s) by a total "
                         "of {} mph would have prevented the congestion.".format(
                             len(rc), cf["counterfactual"]["total_perturbation_budget"]),
            "cited_causes": causes or [{"location": cf["factual"]["target_name"],
                                        "resolved_node_id": cf["meta"]["target_node"]}],
            "recommendations": [
                {"action": "mock", "location": "mock", "time_window_minutes": 15,
                 "expected_effect": "mock", "grounded_in": ["mock"]}
                for _ in range(3)],
        }
        return {"advisory": adv, "mode": "counterfactual", "model": self.model}


# ===========================================================================
# Per-scenario caching (resumable, like every Phase 10-16 study).
# ===========================================================================
def _cache_path(dataset: str, idx: int, node: int, model: str) -> str:
    d = os.path.join(OUT_DIR, "decisions_cache")
    os.makedirs(d, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return os.path.join(d, "{}_{}_{}_{}.json".format(dataset, idx, node, safe))


# ===========================================================================
# One scenario: search -> (narrate) -> score.
# ===========================================================================
def run_scenario(searcher: CounterfactualSearcher, advisor: Any,
                 table: NodeTable, exp: Dict[str, Any], sc: Dict[str, Any],
                 dataset: str, search_cfg: Dict[str, Any],
                 X: torch.Tensor) -> Dict[str, Any]:
    """Run the counterfactual pipeline for one congested scenario, returning a
    record with the cf, the narration metrics (if narrated), and bookkeeping."""
    result = searcher.search(X, exp["prediction"], exp.get("top_nodes", []))
    cf = build_counterfactual(exp, result, search_cfg,
                              extra_meta={"dataset": dataset,
                                          "sample_index": sc["sample_index"],
                                          "tod_band": sc.get("tod_band", "?"),
                                          "congestion": sc.get("congestion", "?"),
                                          "timestamp": sc.get("timestamp", "")})
    problems = validate_counterfactual(cf)
    if problems:
        print("  [warn] cf schema problems idx {}: {}".format(
            sc["sample_index"], problems))

    ctr = cf["counterfactual"]
    rec: Dict[str, Any] = {
        "sample_index": sc["sample_index"],
        "target_node": sc["target_node"],
        "tod_band": sc.get("tod_band", "?"),
        "congestion": sc.get("congestion", "?"),
        "timestamp": sc.get("timestamp", ""),
        "predicted_speed": cf["factual"]["predicted_speed"],
        "predicted_speed_after": ctr["predicted_speed_after"],
        "flip_achieved": ctr["flip_achieved"],
        "reason": cf["meta"]["reason"],
        "n_critical": cf["meta"]["n_critical"],
        "n_levers": cf["meta"]["n_levers"],
        "n_nodes_changed": len(ctr["required_changes"]),
        # Where the leverage lives (honest reporting of the design decision to
        # include the target as a lever): did the minimal counterfactual need the
        # target bottleneck, upstream nodes, or both?
        "n_upstream_changed": sum(1 for r in ctr["required_changes"]
                                  if not r.get("is_target")),
        "changed_target": any(r.get("is_target") for r in ctr["required_changes"]),
        "perturbation_budget": ctr["total_perturbation_budget"],
        "implied_lead_minutes": cf["meta"]["implied_lead_minutes"],
        "cf": cf,
        # narration metrics (filled only when we actually narrate)
        "narrated": False,
        "faithfulness_f1": None,
        "cause_precision": None,
        "cause_recall": None,
        "hallucination_rate": None,
        "narration": None,
    }

    # Narrate ONLY a valid, non-empty counterfactual: an infeasible/empty one has no
    # required changes for the LLM to be faithful TO (nothing to score).
    if ctr["flip_achieved"] and ctr["required_changes"]:
        res = advisor.advise_counterfactual(exp, cf)
        advisory = res["advisory"]
        ref = counterfactual_reference_explanation(exp, cf)
        metrics = score_advisory(ref, advisory, table)
        rec.update({
            "narrated": True,
            "faithfulness_f1": metrics["faithfulness_f1"],
            "cause_precision": metrics["cause_precision"],
            "cause_recall": metrics["cause_recall"],
            "hallucination_rate": metrics["hallucination_rate"],
            "narration": {"advisory": advisory,
                          "context_used": res.get("context_used", [])},
        })
    return rec


# ===========================================================================
# Aggregation.
# ===========================================================================
def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    valid = [r for r in records if r["flip_achieved"]]
    narrated = [r for r in records if r["narrated"]]

    budget_m, budget_s, _ = mean_std([r["perturbation_budget"] for r in valid])
    nodes_m, nodes_s, _ = mean_std([float(r["n_nodes_changed"]) for r in valid])
    f1_m, f1_s, _ = mean_std([r["faithfulness_f1"] for r in narrated])
    hal_m, hal_s, _ = mean_std([r["hallucination_rate"] for r in narrated])
    prec_m, prec_s, _ = mean_std([r["cause_precision"] for r in narrated])
    rec_m, rec_s, _ = mean_std([r["cause_recall"] for r in narrated])
    lead_m, lead_s, _ = mean_std([r["implied_lead_minutes"] for r in valid
                                  if r["implied_lead_minutes"] is not None])

    # Why flips failed (honest breakdown).
    reasons: Dict[str, int] = {}
    for r in records:
        if not r["flip_achieved"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    # Where leverage lives among the valid counterfactuals.
    n_with_upstream = sum(1 for r in valid if r.get("n_upstream_changed", 0) > 0)
    n_target_only = sum(1 for r in valid
                        if r.get("changed_target") and r.get("n_upstream_changed", 0) == 0)
    up_m, up_s, _ = mean_std([float(r.get("n_upstream_changed", 0)) for r in valid])

    return {
        "n_scenarios": n,
        "n_valid": len(valid),
        "n_narrated": len(narrated),
        "validity_rate": (len(valid) / n) if n else float("nan"),
        "n_valid_with_upstream": n_with_upstream,
        "n_valid_target_only": n_target_only,
        "mean_upstream_nodes_changed": up_m, "std_upstream_nodes_changed": up_s,
        "mean_perturbation_budget": budget_m, "std_perturbation_budget": budget_s,
        "mean_nodes_changed": nodes_m, "std_nodes_changed": nodes_s,
        "mean_implied_lead_minutes": lead_m, "std_implied_lead_minutes": lead_s,
        "mean_faithfulness_f1": f1_m, "std_faithfulness_f1": f1_s,
        "mean_cause_precision": prec_m, "std_cause_precision": prec_s,
        "mean_cause_recall": rec_m, "std_cause_recall": rec_s,
        "mean_hallucination_rate": hal_m, "std_hallucination_rate": hal_s,
        "infeasible_reasons": reasons,
    }


# ===========================================================================
# Outputs.
# ===========================================================================
_CSV_COLS = ["sample_index", "target_node", "tod_band", "congestion",
             "predicted_speed", "predicted_speed_after", "flip_achieved", "reason",
             "n_critical", "n_levers", "n_nodes_changed", "perturbation_budget",
             "implied_lead_minutes", "narrated", "faithfulness_f1",
             "cause_precision", "cause_recall", "hallucination_rate"]


def write_csv(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def write_table(agg: Dict[str, Any], model: str, out_tex: str) -> None:
    """LaTeX booktabs summary table, \\input-ready (matches the Phase-9 style)."""
    def pm(m: float, s: float, nd: int = 2) -> str:
        if m != m:                                            # NaN
            return "n/a"
        return "{:.{nd}f} $\\pm$ {:.{nd}f}".format(m, s, nd=nd)

    lines = [
        "% Auto-generated by evaluation/counterfactual_study.py (Phase 17). Do not edit.",
        "% Counterfactual explanations on METR-LA "
        "(n={} congested scenarios, {} valid, model={}).".format(
            agg["n_scenarios"], agg["n_valid"], model),
        "% Counterfactual = minimum speed uplift on the target bottleneck and/or its "
        "congested critical (top-k) nodes that flips the target's 30-min prediction "
        "above free-flow;",
        "% narrative faithfulness is the Phase-5 metric scored against the "
        "counterfactual's required-change nodes.",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        r"Congested scenarios evaluated & {} \\".format(agg["n_scenarios"]),
        r"Validity rate (flip achieved) & {:.1f}\% \\".format(
            100.0 * agg["validity_rate"]),
        r"Mean perturbation budget (mph) & {} \\".format(
            pm(agg["mean_perturbation_budget"], agg["std_perturbation_budget"])),
        r"Mean nodes changed & {} \\".format(
            pm(agg["mean_nodes_changed"], agg["std_nodes_changed"], nd=2)),
        r"Valid needing upstream node(s) & {}/{} \\".format(
            agg["n_valid_with_upstream"], agg["n_valid"]),
        r"Mean implied lead time (min) & {} \\".format(
            pm(agg["mean_implied_lead_minutes"], agg["std_implied_lead_minutes"], nd=1)),
        r"\midrule",
        r"\multicolumn{2}{l}{\emph{Narrative faithfulness to the counterfactual "
        r"(n=" + str(agg["n_narrated"]) + r")}} \\",
        r"Faithfulness F1 & {} \\".format(
            pm(agg["mean_faithfulness_f1"], agg["std_faithfulness_f1"], nd=3)),
        r"Cause precision & {} \\".format(
            pm(agg["mean_cause_precision"], agg["std_cause_precision"], nd=3)),
        r"Cause recall & {} \\".format(
            pm(agg["mean_cause_recall"], agg["std_cause_recall"], nd=3)),
        r"Hallucination rate & {} \\".format(
            pm(agg["mean_hallucination_rate"], agg["std_hallucination_rate"], nd=3)),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))


def write_example_traces(records: List[Dict[str, Any]], exps: Dict[int, Dict[str, Any]],
                         n_examples: int, out_dir: str) -> str:
    """Dump the first `n_examples` VALID scenarios as full factual->counterfactual->
    narrative traces (both a readable .txt and the raw records as .json)."""
    valid = [r for r in records if r["narrated"]][:n_examples]
    txt_path = os.path.join(out_dir, "example_traces.txt")
    json_path = os.path.join(out_dir, "example_traces.json")
    blocks = []
    for r in valid:
        exp = exps[r["sample_index"]]
        blocks.append(format_full_chain(exp, r["cf"], r["narration"]))
    with open(txt_path, "w") as f:
        f.write("\n\n\n".join(blocks) if blocks else "(no valid narrated scenarios)")
    with open(json_path, "w") as f:
        json.dump([{"cf": r["cf"], "narration": r["narration"]} for r in valid],
                  f, indent=2)
    return txt_path


# ===========================================================================
# Main.
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 17 counterfactual study")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--city", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 congested scenarios; print the full chain for each.")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #congested scenarios (0 = config n_scenarios).")
    ap.add_argument("--model", default=None, help="override the Ollama advisor model.")
    ap.add_argument("--mock-llm", action="store_true",
                    help="no Ollama; a mock advisor that cites the required-change "
                         "nodes (harness verification only).")
    ap.add_argument("--show-chain", action="store_true",
                    help="print the full chain for every evaluated scenario.")
    args = ap.parse_args()

    cfg = load_counterfactual_config()
    dataset = args.dataset or cfg["dataset"]
    city = args.city or cfg["city"]
    checkpoint = args.checkpoint or cfg["checkpoint"]
    search_cfg = cfg["search"]
    n_target = args.limit or cfg["study"]["n_scenarios"]
    if args.smoke:
        n_target = 3
    show_chain = args.show_chain or args.smoke

    device = torch.device("cpu")
    builder = ExplanationBuilder(checkpoint, dataset, device=device)
    searcher = CounterfactualSearcher(builder.model, builder.cfg, builder.scaler,
                                      search_cfg, device)
    table = NodeTable(load_node_meta(dataset))

    # Advisor (mock or real).
    if args.mock_llm:
        advisor: Any = MockAdvisor()
    elif args.model:
        import copy
        from ..models.advisor.advisor import Advisor, load_advisor_config
        adv_cfg = copy.deepcopy(load_advisor_config())
        adv_cfg.setdefault("ollama", {})["model"] = args.model
        advisor = Advisor(city, cfg=adv_cfg)
    else:
        from ..models.advisor.advisor import Advisor
        advisor = Advisor(city)
    model_name = getattr(advisor, "model", "mock")

    # --- pool -> filter to CONGESTED on the LIVE model prediction ----------------
    # We filter on searcher.predict_target_mph (the live model), NOT on the cached
    # explanation's stored speed, so a stale cache can never mis-classify congestion.
    flip_thr = float(search_cfg["flip_threshold_mph"])
    ckpt_mtime = os.path.getmtime(os.path.join(PKG_ROOT, checkpoint))
    Xtest = _load_split(dataset, "test")[0]                   # [S, T, N, C] — loaded ONCE
    pool = sample_scenarios(dataset, builder.scaler, cfg["study"]["per_stratum"])
    print("[phase17] pool of {} stratified scenarios; selecting congested "
          "(live pred <{} mph) targets...".format(len(pool), flip_thr))

    os.makedirs(OUT_DIR, exist_ok=True)
    records: List[Dict[str, Any]] = []
    exps: Dict[int, Dict[str, Any]] = {}
    n_cached = 0
    n_freeflow = 0
    for sc in pool:
        if len(records) >= n_target:
            break
        X = Xtest[sc["sample_index"]: sc["sample_index"] + 1]  # [1, T, N, C]
        if searcher.predict_target_mph(X, sc["target_node"]) >= flip_thr:
            n_freeflow += 1
            continue                                          # free-flow -> no counterfactual needed
        exp = get_explanation(builder, dataset, sc, ckpt_mtime)
        exps[sc["sample_index"]] = exp

        cpath = _cache_path(dataset, sc["sample_index"], sc["target_node"], model_name)
        if os.path.exists(cpath):
            with open(cpath) as f:
                rec = json.load(f)
            n_cached += 1
        else:
            rec = run_scenario(searcher, advisor, table, exp, sc, dataset, search_cfg, X)
            with open(cpath, "w") as f:
                json.dump(rec, f, indent=2)
        records.append(rec)

        flag = "FLIP" if rec["flip_achieved"] else "no-flip({})".format(rec["reason"])
        f1 = rec["faithfulness_f1"]
        print("  [{}/{}] idx={} {}/{}  pred {:.1f} mph  {}  budget={} nodes={}  "
              "F1={}".format(
                  len(records), n_target, sc["sample_index"], sc.get("tod_band", "?"),
                  sc.get("congestion", "?"), rec["predicted_speed"], flag,
                  rec["perturbation_budget"], rec["n_nodes_changed"],
                  "n/a" if f1 is None else "{:.3f}".format(f1)))
        if show_chain:
            print("\n" + format_full_chain(exp, rec["cf"], rec.get("narration")) + "\n")
    if n_cached:
        print("  (reused {} cached scenario records)".format(n_cached))

    if not records:
        print("[phase17] no congested scenarios found — nothing to evaluate.")
        return

    # --- aggregate + write -------------------------------------------------
    agg = aggregate(records)
    summary = {**agg, "dataset": dataset, "model": model_name,
               "flip_threshold_mph": flip_thr,
               "search": search_cfg}
    csv_path = os.path.join(OUT_DIR, "counterfactual_per_scenario.csv")
    json_path = os.path.join(OUT_DIR, "counterfactual_summary.json")
    tex_path = os.path.join(PKG_ROOT, cfg["output_tex"])
    write_csv(records, csv_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    write_table(agg, model_name, tex_path)
    traces_path = write_example_traces(records, exps, cfg["study"]["example_traces"], OUT_DIR)

    # --- console report ----------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 17 — COUNTERFACTUAL EXPLANATIONS (METR-LA, n={}, model={})".format(
        agg["n_scenarios"], model_name))
    print("=" * 72)
    print("  free-flow targets skipped:       {} (no counterfactual needed)".format(
        n_freeflow))
    print("  validity rate (flip achieved):   {:.1f}%  ({}/{})".format(
        100.0 * agg["validity_rate"], agg["n_valid"], agg["n_scenarios"]))
    print("  mean perturbation budget (mph):  {:.2f} +/- {:.2f}".format(
        agg["mean_perturbation_budget"], agg["std_perturbation_budget"]))
    print("  mean nodes changed:              {:.2f} +/- {:.2f}".format(
        agg["mean_nodes_changed"], agg["std_nodes_changed"]))
    print("  valid needing upstream nodes:    {}/{}  (target-only: {})".format(
        agg["n_valid_with_upstream"], agg["n_valid"], agg["n_valid_target_only"]))
    print("  mean implied lead time (min):    {:.1f} +/- {:.1f}".format(
        agg["mean_implied_lead_minutes"], agg["std_implied_lead_minutes"]))
    print("  --- narrative faithfulness to counterfactual (n={}) ---".format(
        agg["n_narrated"]))
    print("  faithfulness F1:                 {:.3f} +/- {:.3f}".format(
        agg["mean_faithfulness_f1"], agg["std_faithfulness_f1"]))
    print("  cause precision / recall:        {:.3f} / {:.3f}".format(
        agg["mean_cause_precision"], agg["mean_cause_recall"]))
    print("  hallucination rate:              {:.3f} +/- {:.3f}".format(
        agg["mean_hallucination_rate"], agg["std_hallucination_rate"]))
    if agg["infeasible_reasons"]:
        print("  infeasible breakdown:            {}".format(agg["infeasible_reasons"]))
    print("\n[phase17] wrote:")
    for p in [csv_path, json_path, tex_path, traces_path]:
        print("  " + p)


if __name__ == "__main__":
    main()
