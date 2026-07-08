"""Phase 12 — GNNExplainer vs SHAP faithfulness comparison (the "why not SHAP?"
experiment).

LAYOUT NOTE: the SHAP explainer itself lives in
models/explainer/shap_explainer.py (an explainer artifact). This file is the
EVALUATION study + paper table, which CLAUDE.md pins under /evaluation. Same split
as Phase 5: explainer in models/, study in evaluation/.

THE QUESTION
------------
A reviewer will ask why Layer 2 uses GNNExplainer rather than the popular,
model-agnostic SHAP. We answer with evidence: run the identical LLM pipeline with
each explainer feeding the advisor, and measure which one yields MORE FAITHFUL
natural-language reasoning (does the LLM cite the causes it was actually shown, or
does it wander?).

THREE METHODS (each scored with the Phase-5 faithfulness metric):
  * A  GNNExplainer : advisor gets our GNNExplainer explanation. Score the
                      advisory's cited causes against THAT explanation's top-k.
  * D  SHAP         : advisor gets the SHAP explanation instead (condition-A prompt,
                      SHAP explanation substituted). Score against the SHAP top-k.
  * B  No explainer : advisor gets prediction only (Phase-5 condition B). Score
                      against the GNNExplainer top-k (the reference "true" causes).

WHAT FAITHFULNESS MEASURES HERE (read this before interpreting the table)
------------------------------------------------------------------------
The metric checks whether the LLM's cited causes fall inside the explanation it
was shown. So A and D both ask: "given THIS explainer's output, does the LLM
translate it faithfully or invent extra causes?" If SHAP's node attributions are
less coherent, the LLM is more tempted to confabulate a story around them ->
lower precision / higher hallucination for D. If instead the LLM slavishly cites
whatever it is shown, A and D look similar and SHAP is 'good enough' on
faithfulness -- in which case the case for GNNExplainer rests on its OTHER
advantages (speed, edge/propagation structure, stability), which we also report
as diagnostics (SHAP is ~seconds/prediction and unstable at nsamples=50; see
below). Either outcome is a legitimate, honest answer to the reviewer -- we do
NOT prejudge it.

SPEED / APPROXIMATION (documented): SHAP KernelExplainer samples coalitions; we use
nsamples=50 (fast, but far fewer than the 207 nodes, so the SHAP values are a
sparse approximation and vary run-to-run). That instability is itself part of the
answer. The study is capped at 30 scenarios for this reason.

Run (needs the METR-LA checkpoint + a local Ollama):
  python -m xtraffic.evaluation.shap_comparison --smoke     # 3 scenarios, mock LLM
  python -m xtraffic.evaluation.shap_comparison --limit 3   # 3 scenarios, real LLM
  python -m xtraffic.evaluation.shap_comparison             # full 30, real LLM

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..models.advisor.advisor import Advisor
from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import load_window
from ..models.explainer.shap_explainer import build_shap_explanation_builder
from ..models.gnn.loaders import load_node_meta, load_scaler
from ..utils.io_utils import PKG_ROOT
from .faithfulness import NodeTable, mean_std, score_advisory
from .run_faithfulness_study import _METRIC_KEYS, _aggregate, sample_scenarios

# The three rows of the comparison table, in display order. Each maps a method
# label -> (advisor condition to run, which explanation to show/score against).
#   explanation key: "gnn" = GNNExplainer explanation, "shap" = SHAP explanation.
METHODS = [
    ("A", "GNNExplainer (ours)", "A", "gnn"),
    ("D", "SHAP (KernelExplainer)", "A", "shap"),
    ("B", "No explainer", "B", "gnn"),
]

# The four reported columns (schema keys -> header labels).
COLUMNS = [
    ("faithfulness_f1", "F1"),
    ("hallucination_rate", "Hallucination"),
    ("cause_precision", "Precision"),
    ("cause_recall", "Recall"),
]


# ---------------------------------------------------------------------------
# Explanation caching (build each explainer's explanation once per scenario).
# ---------------------------------------------------------------------------
def _exp_cache_path(results_dir: str, method_key: str, idx: int, node: int) -> str:
    d = os.path.join(results_dir, "explanations_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{}_{}_{}.json".format(method_key, idx, node))


def build_or_load(builder: ExplanationBuilder, method_key: str, dataset: str,
                  sc: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    path = _exp_cache_path(results_dir, method_key, sc["sample_index"], sc["target_node"])
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    X = load_window(dataset, sc["sample_index"])                 # [1, T, N, C]
    exp = builder.explain_prediction(
        X, target_node=sc["target_node"], horizon_step=6, timestamp=sc["timestamp"])
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


def _topk_ids(exp: Dict[str, Any]) -> set:
    return {int(n["node_id"]) for n in exp.get("top_nodes", [])}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


# ---------------------------------------------------------------------------
# Mock advisor (--smoke / --mock-llm) — Ollama-free harness check. Reproduces the
# expected shape: explanation shown (A, D) -> grounded; no explanation (B) ->
# hallucinated. Local copy so this study doesn't depend on the Phase-11 module.
# ---------------------------------------------------------------------------
class MockAdvisor:
    def __init__(self, model: str = "mock"):
        self.model = model

    def advise_condition(self, exp: Dict[str, Any], condition: str) -> Dict[str, Any]:
        top = exp.get("top_nodes", [])
        if condition == "B":                    # no explanation shown -> invent
            causes = [{"location": "Unknown arterial (not in the data)",
                       "resolved_node_id": None}]
            reasoning = "Congestion is likely due to unspecified upstream demand."
        else:                                   # A/D -> cite what was shown
            causes = [{"location": n["node_name"], "resolved_node_id": n["node_id"]}
                      for n in top[:2]]
            reasoning = "Slowdown is driven by the top contributing sensors shown."
        advisory = {"reasoning": reasoning, "cited_causes": causes,
                    "recommendations": [{"action": "mock", "location": "mock",
                        "time_window_minutes": 15, "expected_effect": "mock",
                        "grounded_in": ["mock"]} for _ in range(3)]}
        return {"advisory": advisory, "condition": condition, "model": self.model}


# ---------------------------------------------------------------------------
# Resumable decision cache (one JSONL line per completed (scenario, method)).
# ---------------------------------------------------------------------------
def _decisions_path(results_dir: str, mock: bool) -> str:
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, "decisions{}.jsonl".format("__MOCK" if mock else ""))


def _load_done(path: str) -> Dict[tuple, Dict[str, Any]]:
    done: Dict[tuple, Dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[(r["sample_index"], r["method"])] = r
    return done


# ---------------------------------------------------------------------------
# LaTeX table.
# ---------------------------------------------------------------------------
def _fmt(mean: float, std: float) -> str:
    if mean != mean:                            # NaN
        return "--"
    return "{:.3f} $\\pm$ {:.3f}".format(mean, std)


def build_table(aggs: Dict[str, Dict[str, Any]], n: int, out_tex: str,
                extra_caption: str = "") -> None:
    lines = [
        "% Auto-generated by evaluation/shap_comparison.py (Phase 12). Do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GNNExplainer vs SHAP: faithfulness of the LLM advisory when each "
        r"explainer feeds the pipeline (METR-LA, $n=" + str(n) + r"$ scenarios, "
        r"mean $\pm$ std). Faithfulness is measured against the explanation the "
        r"advisor was shown. `No explainer' is the prediction-only baseline "
        r"(scored against the GNNExplainer causes). SHAP uses KernelExplainer with "
        r"nsamples$=50$ (a deliberate speed/accuracy trade-off)." + extra_caption + "}",
        r"\label{tab:shap_comparison}",
        r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}",
        r"\toprule",
        "Explainer & " + " & ".join(lbl for _, lbl in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for cond, label, _, _ in METHODS:
        a = aggs.get(cond, {})
        cells = [label]
        for key, _ in COLUMNS:
            cells.append(_fmt(a.get(key + "_mean", float("nan")),
                              a.get(key + "_std", float("nan"))))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 12 GNNExplainer vs SHAP comparison")
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--city", default="metr_la")
    ap.add_argument("--checkpoint", default="models/gnn/checkpoints/metr_la_best.pt")
    ap.add_argument("--n-scenarios", type=int, default=30,
                    help="target number of stratified scenarios (SHAP is slow)")
    ap.add_argument("--limit", type=int, default=0, help="hard cap (smoke/debug)")
    ap.add_argument("--nsamples", type=int, default=50,
                    help="SHAP KernelExplainer coalition samples (50 = fast approx)")
    ap.add_argument("--model", default=None, help="override the Ollama advisor model")
    ap.add_argument("--mock-llm", action="store_true", help="Ollama-free harness check")
    ap.add_argument("--smoke", action="store_true",
                    help="3 scenarios + mock LLM, routed to a *_smoke dir")
    args = ap.parse_args()

    device = torch.device("cpu")
    mock = args.mock_llm or args.smoke

    results_dir = os.path.join(PKG_ROOT, "evaluation", "results", "shap_comparison")
    out_tex = os.path.join(PKG_ROOT, "evaluation", "paper", "table_shap_comparison.tex")
    if args.smoke:
        results_dir += "_smoke"
        out_tex = out_tex.replace(".tex", "_smoke.tex")
        if not args.limit:
            args.limit = 3
    os.makedirs(results_dir, exist_ok=True)

    # --- Sample scenarios: 15 strata x per_stratum ~= n_scenarios ------------
    scaler = load_scaler(args.dataset)
    per_stratum = max(1, round(args.n_scenarios / 15))          # 30 -> 2 per cell
    scenarios = sample_scenarios(args.dataset, scaler, per_stratum)
    if args.limit and len(scenarios) > args.limit:
        scenarios = scenarios[:args.limit]
    elif len(scenarios) > args.n_scenarios:
        scenarios = scenarios[:args.n_scenarios]
    print("[shap-cmp] {} scenarios | nsamples={} | model={}".format(
        len(scenarios), args.nsamples, "MOCK" if mock else (args.model or "advisor.yaml")))

    # --- Build both explainers (GNNExplainer + SHAP, separate model loads) ----
    gnn_builder = ExplanationBuilder(args.checkpoint, args.dataset, device=device)
    shap_builder = build_shap_explanation_builder(
        args.checkpoint, args.dataset, device=device, nsamples=args.nsamples)

    if mock:
        advisor: Any = MockAdvisor(args.model or "mock")
    elif args.model:
        import copy

        from ..models.advisor.advisor import load_advisor_config
        cfg = copy.deepcopy(load_advisor_config())
        cfg.setdefault("ollama", {})["model"] = args.model
        advisor = Advisor(args.city, cfg=cfg)
    else:
        advisor = Advisor(args.city)

    table = NodeTable(load_node_meta(args.dataset))
    dec_path = _decisions_path(results_dir, mock)
    done = _load_done(dec_path)

    rows: List[Dict[str, Any]] = []
    overlaps: List[float] = []                  # GNN vs SHAP top-k agreement (diagnostic)

    for i, sc in enumerate(scenarios):
        exp_gnn = build_or_load(gnn_builder, "gnn", args.dataset, sc, results_dir)
        exp_shap = build_or_load(shap_builder, "shap", args.dataset, sc, results_dir)
        exps = {"gnn": exp_gnn, "shap": exp_shap}
        overlaps.append(_jaccard(_topk_ids(exp_gnn), _topk_ids(exp_shap)))

        for method_key, _label, adv_cond, exp_key in METHODS:
            cache_key = (sc["sample_index"], method_key)
            if cache_key in done:
                rows.append(done[cache_key])
                continue
            exp = exps[exp_key]
            res = advisor.advise_condition(exp, adv_cond)
            metrics = score_advisory(exp, res["advisory"], table)
            row = {
                "sample_index": sc["sample_index"], "target_node": sc["target_node"],
                "tod_band": sc["tod_band"], "congestion": sc["congestion"],
                "method": method_key, "advisor_condition": adv_cond,
                # `condition` mirrors `method` so the Phase-5 _aggregate() (which
                # groups rows by r["condition"]) can be reused unchanged.
                "condition": method_key, "explanation": exp_key,
                "advisory_error": res["advisory"].get("_error", ""),
            }
            for k in _METRIC_KEYS + ["n_cited_causes", "n_topk"]:
                row[k] = metrics[k]
            rows.append(row)
            with open(dec_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[cache_key] = row
        print("  [{}/{}] {} cong={} | GNN/SHAP top-k overlap {:.2f}".format(
            i + 1, len(scenarios), sc["tod_band"], sc["congestion"], overlaps[-1]))

    # Keep only rows for the currently-sampled scenarios (robust to a prior larger run).
    wanted = {sc["sample_index"] for sc in scenarios}
    rows = [r for r in rows if r["sample_index"] in wanted]

    # --- Aggregate per method (reuse the Phase-5 aggregator) ------------------
    aggs = {cond: _aggregate(rows, cond) for cond, _, _, _ in METHODS}
    ov_mean, ov_std, _ = mean_std(overlaps)

    # --- Persist CSV + JSON (CLAUDE.md: both) ---------------------------------
    if rows:
        with open(os.path.join(results_dir, "shap_comparison_per_scenario.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summary = {
        "dataset": args.dataset, "n_scenarios": len(scenarios),
        "nsamples_shap": args.nsamples, "model": getattr(advisor, "model", "mock"),
        "methods": {cond: {"label": label} for cond, label, _, _ in METHODS},
        "aggregate": aggs,
        "gnn_vs_shap_topk_overlap": {"mean": ov_mean, "std": ov_std},
    }
    with open(os.path.join(results_dir, "shap_comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- LaTeX table ----------------------------------------------------------
    build_table(aggs, len(scenarios), out_tex,
                extra_caption=" GNN/SHAP top-$k$ overlap (Jaccard) $="
                              + "{:.2f}$.".format(ov_mean))

    # --- Console report + verdict ---------------------------------------------
    print("\n=== SHAP vs GNNExplainer (mean +/- std, n={}) ===".format(len(scenarios)))
    hdr = "{:24s}".format("explainer")
    for _, lbl in COLUMNS:
        hdr += "{:>18s}".format(lbl)
    print(hdr); print("-" * len(hdr))
    for cond, label, _, _ in METHODS:
        line = "{:24s}".format(label)
        for key, _ in COLUMNS:
            m = aggs[cond].get(key + "_mean", float("nan"))
            s = aggs[cond].get(key + "_std", float("nan"))
            line += "{:>18s}".format("nan" if m != m else "{:.3f}+/-{:.3f}".format(m, s))
        print(line)
    a_f1 = aggs["A"]["faithfulness_f1_mean"]; d_f1 = aggs["D"]["faithfulness_f1_mean"]
    a_h = aggs["A"]["hallucination_rate_mean"]; d_h = aggs["D"]["hallucination_rate_mean"]
    print("\nGNN vs SHAP top-k overlap (Jaccard): {:.2f} +/- {:.2f}".format(ov_mean, ov_std))
    if a_f1 == a_f1 and d_f1 == d_f1:
        if a_f1 > d_f1 + 0.02 or a_h < d_h - 0.02:
            verdict = ("GNNExplainer yields MORE faithful advisories than SHAP "
                       "(F1 {:.3f} vs {:.3f}; halluc {:.3f} vs {:.3f}).".format(a_f1, d_f1, a_h, d_h))
        elif d_f1 > a_f1 + 0.02:
            verdict = ("SHAP yields more faithful advisories than GNNExplainer here "
                       "(F1 {:.3f} vs {:.3f}) -- report honestly.".format(d_f1, a_f1))
        else:
            verdict = ("GNNExplainer ~= SHAP on faithfulness (F1 {:.3f} vs {:.3f}); the "
                       "case for GNNExplainer rests on speed/structure/stability.".format(a_f1, d_f1))
        print("VERDICT:", verdict)
    print("\n[shap-cmp] wrote:")
    print("  " + out_tex)
    print("  " + os.path.join(results_dir, "shap_comparison_summary.json"))


if __name__ == "__main__":
    main()
