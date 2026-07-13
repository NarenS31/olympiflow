"""Phase 18 STUDY — uncertainty-aware explanations vs the plain single-run pipeline.

Runs, on 20 stratified METR-LA scenarios, TWO advisory conditions on the SAME
prediction and compares them:

  * A (DETERMINISTIC)  — the standard Phase-5 condition-A advisory on the single-run
                         (seed-0) explanation. This is exactly what a planner gets
                         today, and it is byte-identical to Phase 5's condition A.
  * U (UNCERTAINTY)    — the explainer is run K=10 times; nodes are split into CORE
                         (confident) vs PERIPHERAL (uncertain); the advisor is shown
                         that split with an explicit honesty instruction (assert
                         core, hedge peripheral, omit noise).

The three questions the assignment poses, answered with measurements:
  1. Does the uncertainty framing IMPROVE OR HURT faithfulness F1?
  2. Does it reduce hallucination further below the Phase-5 ~0.5%?
  3. Does the LLM ACTUALLY hedge peripheral causes more than core ones, or treat
     every cited cause as equally confident?  -> hedging_analysis() (uncertain_explainer)

SCORING REFERENCE (documented design decision, FLAGGED)
-------------------------------------------------------
Both A and U are scored by the Phase-5 faithfulness metric against the SAME fixed
ground truth: the DETERMINISTIC single-run top-k (base_exp["top_nodes"], seed 0).
Holding the reference fixed is what makes the A-vs-U F1/hallucination numbers
directly comparable — any difference is attributable purely to the uncertainty
FRAMING, not to a different ground truth. Honest caveat this introduces: the
uncertainty condition drops NOISE-tier nodes it was shown are unstable, so if a
seed-0 top-k node happens to be noise-classified and correctly omitted by U, that
counts against U's recall. That effect is small (core nodes dominate the top-k) and
we ALSO log U's faithfulness against its own shown (core+peripheral) set as a
secondary diagnostic, so both views are visible.

WHY THIS IS CHEAP: reuses the Phase-5 stratified sampler (seed 42) so every base
explanation is already cached; the only new compute per scenario is K=10 explainer
solves for the tier overlay + two LLM calls (A and U).

Run (needs the trained checkpoint + a local Ollama with the advisor model):
  python -m xtraffic.evaluation.uncertainty_study --smoke      # 3 scenarios, full chain shown
  python -m xtraffic.evaluation.uncertainty_study              # the full 20
  python -m xtraffic.evaluation.uncertainty_study --mock-llm   # harness only, no Ollama
  python -m xtraffic.evaluation.uncertainty_study --model mistral:7b   # second LLM

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

from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.uncertain_explainer import (
    UncertaintyAnalyzer, attach_uncertainty, hedging_analysis,
    load_uncertainty_config,
)
from ..models.gnn.loaders import _load_split, load_node_meta
from ..utils.io_utils import PKG_ROOT
from .faithfulness import NodeTable, mean_std, score_advisory
from .run_faithfulness_study import _explanation_cache_path, sample_scenarios

OUT_DIR = os.path.join(PKG_ROOT, "evaluation", "results", "uncertainty")


# ===========================================================================
# Staleness-aware base-explanation loading (same guard as Phase 17): the shared
# Phase-5 cache is trusted ONLY when newer than the checkpoint; otherwise we
# rebuild into OUR OWN cache and never mutate the shared Phase-5 artifacts.
# ===========================================================================
def _own_cache_path(dataset: str, idx: int, node: int) -> str:
    d = os.path.join(OUT_DIR, "explanations_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{}_{}_{}.json".format(dataset, idx, node))


def get_base_explanation(builder: ExplanationBuilder, dataset: str,
                         sc: Dict[str, Any], horizon_step: int,
                         ckpt_mtime: float) -> Dict[str, Any]:
    idx, node = sc["sample_index"], sc["target_node"]
    shared = _explanation_cache_path(dataset, idx, node)
    if os.path.exists(shared) and os.path.getmtime(shared) >= ckpt_mtime:
        with open(shared) as f:
            return json.load(f)
    own = _own_cache_path(dataset, idx, node)
    if os.path.exists(own):
        with open(own) as f:
            return json.load(f)
    print("    (rebuilding stale/missing explanation for idx {} node {} ...)".format(
        idx, node))
    X = _load_split(dataset, "test")[0][idx: idx + 1]
    exp = builder.explain_prediction(X, target_node=node,
                                     horizon_step=horizon_step, timestamp=sc["timestamp"])
    exp["meta"]["city"] = dataset
    with open(own, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


# ===========================================================================
# Scenario selection: a DIVERSE spread of n across the (tod-band x congestion)
# strata via round-robin, so 20 scenarios cover the day/congestion range instead
# of front-truncating to one band (the --limit caveat in the Phase-5 study).
# ===========================================================================
def select_diverse(scenarios: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for sc in scenarios:
        key = "{}|{}".format(sc.get("tod_band", "?"), sc.get("congestion", "?"))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(sc)
    picked: List[Dict[str, Any]] = []
    i = 0
    while len(picked) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop(0))
        i += 1
    return picked


# ===========================================================================
# Reference for U's SECONDARY "vs shown" faithfulness diagnostic: score against
# the core+peripheral nodes the uncertainty prompt actually showed.
# ===========================================================================
def _shown_reference(base_exp: Dict[str, Any], block: Dict[str, Any]) -> Dict[str, Any]:
    ref = dict(base_exp)
    ref["top_nodes"] = [
        {"node_id": int(c["node_id"]), "node_name": c["node_name"],
         "importance": float(c["confidence"]), "current_speed_mph": c["current_speed_mph"]}
        for c in (block.get("core_causes", []) + block.get("peripheral_causes", []))
    ]
    return ref


# ===========================================================================
# Mock advisor for --mock-llm: verifies the WHOLE harness (K-run overlay ->
# two conditions -> scoring -> hedging -> table) with no Ollama.
#   - condition A: cites every top-k node (perfectly faithful).
#   - advise_uncertain: cites core + peripheral, ASSERTS core plainly and HEDGES
#     peripheral (embeds "may/possibly/uncertain" near peripheral region names), so
#     hedging_analysis provably reports peripheral hedged more than core.
# ===========================================================================
class MockAdvisor:
    def __init__(self, model: str = "mock"):
        self.model = model

    def advise_condition(self, exp: Dict[str, Any], condition: str,
                         extra_instruction: Optional[str] = None) -> Dict[str, Any]:
        top = exp.get("top_nodes", [])
        causes = [{"location": n["node_name"], "resolved_node_id": n["node_id"]}
                  for n in top]
        reasoning = " ".join(
            "Congestion at {} is a confirmed contributing cause.".format(
                n["node_name"].split("(")[0].strip()) for n in top)
        return {"advisory": _mock_adv(reasoning, causes), "condition": condition,
                "model": self.model}

    def advise_uncertain(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        u = exp["uncertainty"]
        core = u.get("core_causes", [])
        peri = u.get("peripheral_causes", [])
        causes = [{"location": c["node_name"], "resolved_node_id": c["node_id"]}
                  for c in core + peri]
        sents = ["Congestion at {} is a confirmed cause.".format(
            c["node_name"].split("(")[0].strip()) for c in core]
        sents += ["It may possibly also involve {}, though this is uncertain.".format(
            c["node_name"].split("(")[0].strip()) for c in peri]
        return {"advisory": _mock_adv(" ".join(sents), causes), "mode": "uncertain",
                "model": self.model}


def _mock_adv(reasoning: str, causes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "reasoning": reasoning or "Mock reasoning.",
        "cited_causes": causes or [{"location": "unknown", "resolved_node_id": None}],
        "recommendations": [
            {"action": "mock", "location": "mock", "time_window_minutes": 15,
             "expected_effect": "mock", "grounded_in": ["mock"]} for _ in range(3)],
    }


# ===========================================================================
# One scenario: build overlay -> run A + U -> score both -> hedging analysis.
# ===========================================================================
def run_scenario(builder: ExplanationBuilder, analyzer: UncertaintyAnalyzer,
                 advisor: Any, table: NodeTable, base_exp: Dict[str, Any],
                 sc: Dict[str, Any], X: torch.Tensor) -> Dict[str, Any]:
    block = analyzer.build_uncertainty_block(X, sc["target_node"])
    uncertain_exp = attach_uncertainty(base_exp, block)

    # --- condition A (deterministic single-run) ---
    res_a = advisor.advise_condition(base_exp, "A")
    m_a = score_advisory(base_exp, res_a["advisory"], table)
    hedge_a = hedging_analysis(res_a["advisory"].get("reasoning", ""),
                               block["core_causes"], block["peripheral_causes"])

    # --- condition U (uncertainty-aware) ---
    res_u = advisor.advise_uncertain(uncertain_exp)
    m_u = score_advisory(base_exp, res_u["advisory"], table)          # PRIMARY: vs deterministic top-k
    m_u_shown = score_advisory(_shown_reference(base_exp, block),
                               res_u["advisory"], table)              # SECONDARY: vs shown set
    hedge_u = hedging_analysis(res_u["advisory"].get("reasoning", ""),
                               block["core_causes"], block["peripheral_causes"])

    return {
        "sample_index": sc["sample_index"],
        "target_node": sc["target_node"],
        "tod_band": sc.get("tod_band", "?"),
        "congestion": sc.get("congestion", "?"),
        "timestamp": sc.get("timestamp", ""),
        # uncertainty structure
        "explanation_stability": block["explanation_stability"],
        "n_core": len(block["core_causes"]),
        "n_peripheral": len(block["peripheral_causes"]),
        "n_noise": block["noise_count"],
        # condition A metrics
        "A_f1": m_a["faithfulness_f1"], "A_precision": m_a["cause_precision"],
        "A_recall": m_a["cause_recall"], "A_hallucination": m_a["hallucination_rate"],
        # condition U metrics (primary = vs deterministic top-k)
        "U_f1": m_u["faithfulness_f1"], "U_precision": m_u["cause_precision"],
        "U_recall": m_u["cause_recall"], "U_hallucination": m_u["hallucination_rate"],
        "U_f1_vs_shown": m_u_shown["faithfulness_f1"],
        "U_hallucination_vs_shown": m_u_shown["hallucination_rate"],
        # hedging (the KEY question)
        "A_core_hedge_rate": hedge_a["core_hedge_rate"],
        "A_peripheral_hedge_rate": hedge_a["peripheral_hedge_rate"],
        "A_hedge_gap": hedge_a["hedge_rate_gap"],
        "U_core_hedge_rate": hedge_u["core_hedge_rate"],
        "U_peripheral_hedge_rate": hedge_u["peripheral_hedge_rate"],
        "U_hedge_gap": hedge_u["hedge_rate_gap"],
        # kept for traces / debugging (not in CSV)
        "_block": block,
        "_A_advisory": res_a["advisory"],
        "_U_advisory": res_u["advisory"],
        "_A_hedge": hedge_a, "_U_hedge": hedge_u,
    }


# ===========================================================================
# Per-scenario cache (resumable, keyed by model — like every Phase 10-17 study).
# ===========================================================================
def _cache_path(dataset: str, idx: int, node: int, model: str) -> str:
    d = os.path.join(OUT_DIR, "decisions_cache")
    os.makedirs(d, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return os.path.join(d, "{}_{}_{}_{}.json".format(dataset, idx, node, safe))


# ===========================================================================
# Aggregation.
# ===========================================================================
def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def ms(key: str) -> Dict[str, float]:
        m, s, n = mean_std([r[key] for r in records])
        return {"mean": m, "std": s, "n": n}

    # Hedging: average over scenarios where the gap is defined (both tiers appeared).
    a_gaps = [r["A_hedge_gap"] for r in records if r["A_hedge_gap"] is not None]
    u_gaps = [r["U_hedge_gap"] for r in records if r["U_hedge_gap"] is not None]
    a_core = [r["A_core_hedge_rate"] for r in records if r["A_core_hedge_rate"] is not None]
    a_peri = [r["A_peripheral_hedge_rate"] for r in records if r["A_peripheral_hedge_rate"] is not None]
    u_core = [r["U_core_hedge_rate"] for r in records if r["U_core_hedge_rate"] is not None]
    u_peri = [r["U_peripheral_hedge_rate"] for r in records if r["U_peripheral_hedge_rate"] is not None]

    def _mean(xs: List[float]) -> float:
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    # How often did the LLM hedge peripheral MORE than core (gap > 0)?
    u_hedged_more = sum(1 for g in u_gaps if g > 0)

    return {
        "n_scenarios": len(records),
        "stability": ms("explanation_stability"),
        "mean_core": _mean([float(r["n_core"]) for r in records]),
        "mean_peripheral": _mean([float(r["n_peripheral"]) for r in records]),
        "mean_noise": _mean([float(r["n_noise"]) for r in records]),
        "A": {k: ms("A_" + k) for k in ("f1", "precision", "recall", "hallucination")},
        "U": {k: ms("U_" + k) for k in ("f1", "precision", "recall", "hallucination")},
        "U_vs_shown": {"f1": ms("U_f1_vs_shown"),
                       "hallucination": ms("U_hallucination_vs_shown")},
        "hedging": {
            "A_mean_core_hedge_rate": _mean(a_core),
            "A_mean_peripheral_hedge_rate": _mean(a_peri),
            "A_mean_gap": _mean(a_gaps), "A_n_gap_defined": len(a_gaps),
            "U_mean_core_hedge_rate": _mean(u_core),
            "U_mean_peripheral_hedge_rate": _mean(u_peri),
            "U_mean_gap": _mean(u_gaps), "U_n_gap_defined": len(u_gaps),
            "U_n_hedged_peripheral_more": u_hedged_more,
        },
    }


# ===========================================================================
# Outputs.
# ===========================================================================
_CSV_COLS = ["sample_index", "target_node", "tod_band", "congestion",
             "explanation_stability", "n_core", "n_peripheral", "n_noise",
             "A_f1", "A_precision", "A_recall", "A_hallucination",
             "U_f1", "U_precision", "U_recall", "U_hallucination",
             "U_f1_vs_shown", "U_hallucination_vs_shown",
             "A_core_hedge_rate", "A_peripheral_hedge_rate", "A_hedge_gap",
             "U_core_hedge_rate", "U_peripheral_hedge_rate", "U_hedge_gap"]


def write_csv(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def _pm(d: Dict[str, float], nd: int = 3) -> str:
    m, s = d["mean"], d["std"]
    if m != m:
        return "n/a"
    return "{:.{nd}f} $\\pm$ {:.{nd}f}".format(m, s, nd=nd)


def write_table(agg: Dict[str, Any], model: str, out_tex: str) -> None:
    """LaTeX booktabs table: condition A vs U on faithfulness, plus the hedging
    result. \\input-ready (matches the Phase-9 style)."""
    h = agg["hedging"]

    def rate(x: float) -> str:
        return "n/a" if x != x else "{:.2f}".format(x)

    lines = [
        "% Auto-generated by evaluation/uncertainty_study.py (Phase 18). Do not edit.",
        "% Uncertainty-aware explanations on METR-LA (n={} scenarios, K={} runs, "
        "model={}).".format(agg["n_scenarios"], agg.get("k_runs", "?"), model),
        "% A = deterministic single-run pipeline (Phase-5 condition A); "
        "U = K-run core/peripheral uncertainty framing.",
        "% Faithfulness scored against the SAME deterministic top-k in both "
        "conditions (see uncertainty_study.py).",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & A (deterministic) & U (uncertainty-aware) \\",
        r"\midrule",
        r"Faithfulness F1 & {} & {} \\".format(_pm(agg["A"]["f1"]), _pm(agg["U"]["f1"])),
        r"Cause precision & {} & {} \\".format(
            _pm(agg["A"]["precision"]), _pm(agg["U"]["precision"])),
        r"Cause recall & {} & {} \\".format(
            _pm(agg["A"]["recall"]), _pm(agg["U"]["recall"])),
        r"Hallucination rate & {} & {} \\".format(
            _pm(agg["A"]["hallucination"]), _pm(agg["U"]["hallucination"])),
        r"\midrule",
        r"\multicolumn{3}{l}{\emph{Uncertainty structure (K=" + str(agg.get("k_runs", "?"))
        + r" runs): mean explanation stability " + "{:.3f}".format(agg["stability"]["mean"])
        + r"}} \\",
        r"\multicolumn{3}{l}{mean core / peripheral / noise causes = "
        + "{:.1f} / {:.1f} / {:.1f}".format(
            agg["mean_core"], agg["mean_peripheral"], agg["mean_noise"]) + r"} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\emph{Hedging (fraction of tier-sentences using "
        r"hedge language)}} \\",
        r"Core causes (A / U) & \multicolumn{2}{c}{" + "{} / {}".format(
            rate(h["A_mean_core_hedge_rate"]), rate(h["U_mean_core_hedge_rate"])) + r"} \\",
        r"Peripheral causes (A / U) & \multicolumn{2}{c}{" + "{} / {}".format(
            rate(h["A_mean_peripheral_hedge_rate"]), rate(h["U_mean_peripheral_hedge_rate"]))
        + r"} \\",
        r"Peripheral$-$core hedge gap (U) & \multicolumn{2}{c}{" + "{}".format(
            rate(h["U_mean_gap"])) + r"} \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))


def make_stability_plot(records: List[Dict[str, Any]], path: str) -> Optional[str]:
    """Distribution of the per-scenario explanation-stability score (mean pairwise
    Jaccard across K runs). Shows how much explanation certainty VARIES scenario to
    scenario — the motivation for tiering in the first place. Vector PDF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # pragma: no cover
        print("(matplotlib unavailable, skipping plot: {})".format(e))
        return None
    stab = [r["explanation_stability"] for r in records]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(stab, bins=min(12, max(4, len(stab))), color="#0072B2",
            edgecolor="white")
    mean = sum(stab) / len(stab) if stab else 0.0
    ax.axvline(mean, color="#D55E00", linestyle="--", linewidth=1.5,
               label="mean {:.2f}".format(mean))
    ax.set_xlabel("explanation stability (mean pairwise Jaccard over K runs)")
    ax.set_ylabel("# scenarios")
    ax.set_xlim(0, 1)
    ax.set_title("Phase 18 — per-scenario explanation stability (n={})".format(len(stab)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _region(name: str) -> str:
    return name.split("(")[0].strip()


def format_trace(rec: Dict[str, Any]) -> str:
    """Human-readable core-vs-peripheral treatment example for one scenario: the
    tiers, then how each condition's advisory reasoned + its hedging stats."""
    b = rec["_block"]
    L: List[str] = []
    L.append("=" * 74)
    L.append("SCENARIO idx={} ({}/{})  target node {}".format(
        rec["sample_index"], rec["tod_band"], rec["congestion"], rec["target_node"]))
    L.append("=" * 74)
    L.append("explanation stability = {:.3f} | core={} peripheral={} noise={}".format(
        rec["explanation_stability"], rec["n_core"], rec["n_peripheral"], rec["n_noise"]))
    L.append("\nCORE (assert):")
    for c in b["core_causes"]:
        L.append("  - {:28s} {}".format(_region(c["node_name"]), c["frequency"]))
    L.append("PERIPHERAL (hedge):")
    for c in b["peripheral_causes"]:
        L.append("  - {:28s} {}".format(_region(c["node_name"]), c["frequency"]))

    for cond, akey, hkey in (("A (deterministic)", "_A_advisory", "_A_hedge"),
                             ("U (uncertainty-aware)", "_U_advisory", "_U_hedge")):
        adv = rec[akey]
        hedge = rec[hkey]
        L.append("\n[{}] reasoning:".format(cond))
        L.append("  " + (adv.get("reasoning", "") or "(empty)"))
        L.append("  hedge rate  core={} peripheral={}  gap={}".format(
            _fmt(hedge["core_hedge_rate"]), _fmt(hedge["peripheral_hedge_rate"]),
            _fmt(hedge["hedge_rate_gap"])))
    L.append("\nfaithfulness  A: F1={:.3f} halluc={:.3f}   U: F1={:.3f} halluc={:.3f}".format(
        rec["A_f1"], rec["A_hallucination"], rec["U_f1"], rec["U_hallucination"]))
    return "\n".join(L)


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else "{:.2f}".format(x)


# ===========================================================================
# Main.
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 18 uncertainty-aware study")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--city", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 scenarios; print the full core-vs-peripheral chain for each.")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #scenarios (0 = config n_scenarios).")
    ap.add_argument("--model", default=None, help="override the Ollama advisor model.")
    ap.add_argument("--mock-llm", action="store_true",
                    help="no Ollama; a mock advisor (harness verification only).")
    ap.add_argument("--show-chain", action="store_true",
                    help="print the full chain for every evaluated scenario.")
    args = ap.parse_args()

    cfg = load_uncertainty_config()
    dataset = args.dataset or cfg["dataset"]
    city = args.city or cfg["city"]
    checkpoint = args.checkpoint or cfg["checkpoint"]
    horizon_step = int(cfg["uncertainty"]["horizon_step"])
    n_target = args.limit or cfg["study"]["n_scenarios"]
    if args.smoke:
        n_target = 3
    show_chain = args.show_chain or args.smoke

    device = torch.device("cpu")
    builder = ExplanationBuilder(checkpoint, dataset, device=device,
                                 top_k=int(cfg["uncertainty"]["top_k"]))
    analyzer = UncertaintyAnalyzer(builder, cfg)
    table = NodeTable(load_node_meta(dataset))

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

    ckpt_mtime = os.path.getmtime(os.path.join(PKG_ROOT, checkpoint))
    pool = sample_scenarios(dataset, builder.scaler, cfg["study"]["per_stratum"])
    selected = select_diverse(pool, n_target)
    print("[phase18] {} scenarios selected (diverse spread) | K={} runs | model={}"
          .format(len(selected), analyzer.k_runs, model_name))

    os.makedirs(OUT_DIR, exist_ok=True)
    Xtest = _load_split(dataset, "test")[0]                   # [S, T, N, C] — loaded ONCE
    records: List[Dict[str, Any]] = []
    n_cached = 0
    for i, sc in enumerate(selected):
        base_exp = get_base_explanation(builder, dataset, sc, horizon_step, ckpt_mtime)
        X = Xtest[sc["sample_index"]: sc["sample_index"] + 1]  # [1, T, N, C]
        cpath = _cache_path(dataset, sc["sample_index"], sc["target_node"], model_name)
        if os.path.exists(cpath):
            with open(cpath) as f:
                rec = json.load(f)
            n_cached += 1
        else:
            rec = run_scenario(builder, analyzer, advisor, table, base_exp, sc, X)
            with open(cpath, "w") as f:
                json.dump(rec, f, indent=2)
        records.append(rec)
        print("  [{}/{}] idx={} {}/{}  stab={:.2f} core/peri/noise={}/{}/{}  "
              "F1 A={:.2f} U={:.2f}  halluc A={:.2f} U={:.2f}  hedge_gap U={}".format(
                  i + 1, len(selected), sc["sample_index"], sc.get("tod_band", "?"),
                  sc.get("congestion", "?"), rec["explanation_stability"],
                  rec["n_core"], rec["n_peripheral"], rec["n_noise"],
                  rec["A_f1"], rec["U_f1"], rec["A_hallucination"], rec["U_hallucination"],
                  _fmt(rec["U_hedge_gap"])))
        if show_chain:
            print("\n" + format_trace(rec) + "\n")
    if n_cached:
        print("  (reused {} cached scenario records)".format(n_cached))

    if not records:
        print("[phase18] no scenarios — nothing to aggregate.")
        return

    agg = aggregate(records)
    agg["k_runs"] = analyzer.k_runs
    summary = {**agg, "dataset": dataset, "model": model_name,
               "core_threshold": analyzer.core_threshold,
               "noise_threshold": analyzer.noise_threshold}

    csv_path = os.path.join(OUT_DIR, "uncertainty_per_scenario.csv")
    json_path = os.path.join(OUT_DIR, "uncertainty_summary.json")
    tex_path = os.path.join(PKG_ROOT, cfg["output_tex"])
    fig_path = os.path.join(PKG_ROOT, "evaluation", "paper", "fig_uncertainty_stability.pdf")
    write_csv(records, csv_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    write_table(agg, model_name, tex_path)
    fig = make_stability_plot(records, fig_path)

    # example traces (core-vs-peripheral treatment) for the paper.
    n_ex = cfg["study"]["example_traces"]
    traces_txt = os.path.join(OUT_DIR, "example_traces.txt")
    with open(traces_txt, "w") as f:
        f.write("\n\n\n".join(format_trace(r) for r in records[:n_ex])
                if records else "(none)")

    # --- console report: the three questions --------------------------------
    h = agg["hedging"]
    print("\n" + "=" * 74)
    print("PHASE 18 — UNCERTAINTY-AWARE EXPLANATIONS (METR-LA, n={}, K={}, model={})".format(
        agg["n_scenarios"], analyzer.k_runs, model_name))
    print("=" * 74)
    print("uncertainty structure: mean stability {:.3f} | core/peri/noise "
          "{:.1f}/{:.1f}/{:.1f}".format(
              agg["stability"]["mean"], agg["mean_core"], agg["mean_peripheral"],
              agg["mean_noise"]))
    print("\nQ1/Q2 — faithfulness A (deterministic) vs U (uncertainty):")
    print("  {:16s}{:>16s}{:>16s}".format("metric", "A", "U"))
    for k, lab in (("f1", "F1"), ("precision", "precision"), ("recall", "recall"),
                   ("hallucination", "hallucination")):
        print("  {:16s}{:>16s}{:>16s}".format(
            lab, "{:.3f}+/-{:.3f}".format(agg["A"][k]["mean"], agg["A"][k]["std"]),
            "{:.3f}+/-{:.3f}".format(agg["U"][k]["mean"], agg["U"][k]["std"])))
    print("  (U vs its own shown set: F1 {:.3f}, halluc {:.3f})".format(
        agg["U_vs_shown"]["f1"]["mean"], agg["U_vs_shown"]["hallucination"]["mean"]))
    df1 = agg["U"]["f1"]["mean"] - agg["A"]["f1"]["mean"]
    dhal = agg["U"]["hallucination"]["mean"] - agg["A"]["hallucination"]["mean"]
    print("  -> uncertainty framing changes F1 by {:+.3f} and hallucination by "
          "{:+.3f} vs deterministic.".format(df1, dhal))

    print("\nQ3 — does the LLM actually HEDGE peripheral more than core?")
    print("  condition A:  core hedge rate {:.2f} | peripheral {:.2f} | gap {:.2f}".format(
        h["A_mean_core_hedge_rate"], h["A_mean_peripheral_hedge_rate"], h["A_mean_gap"]))
    print("  condition U:  core hedge rate {:.2f} | peripheral {:.2f} | gap {:.2f}".format(
        h["U_mean_core_hedge_rate"], h["U_mean_peripheral_hedge_rate"], h["U_mean_gap"]))
    print("  U hedged peripheral MORE than core in {}/{} scenarios (gap defined).".format(
        h["U_n_hedged_peripheral_more"], h["U_n_gap_defined"]))
    verdict = ("YES — U hedges uncertain causes more than confident ones (framing works)"
               if (h["U_mean_gap"] == h["U_mean_gap"] and h["U_mean_gap"] > 0.05)
               else "NO/WEAK — the LLM treats cited causes with similar confidence "
                    "(report honestly)")
    print("  ->", verdict)

    print("\n[phase18] wrote:")
    for p in [csv_path, json_path, tex_path, traces_txt] + ([fig] if fig else []):
        print("  " + p)


if __name__ == "__main__":
    main()
