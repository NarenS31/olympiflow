"""Phase 5 faithfulness STUDY — run the full pipeline on >=100 stratified test
scenarios under conditions A/B/C and report the metric.

This is the experiment that produces the paper's Contribution-#2 table. It:

  1. SAMPLES >=100 scenarios from the METR-LA test split, stratified across
     time-of-day (5 bands) x congestion level (3 terciles) so no single regime
     dominates the average (a metric that only looks good at rush hour would be
     misleading).
  2. For each scenario builds the Layer-2 explanation ONCE (it is condition-
     independent), then runs Layer 3 under all three conditions:
        A full | B no-explanation | C no-city-context
  3. Scores every advisory with evaluation/faithfulness.score_advisory and
     aggregates mean +/- std overall, by time-of-day, and by congestion.
  4. Writes per-scenario CSV, a summary JSON, a formatted A/B/C table, and
     distribution plots (PDF) to evaluation/results/faithfulness/.

The MONEY RESULT is A vs B: strip the mathematical explanation and the LLM must
invent causes, so hallucination should spike and F1 should collapse. If A does
not clearly beat B, the paper's core claim is in question — we report it honestly
either way (CLAUDE.md).

Run (needs the trained checkpoint + a local Ollama with the advisor model):
  python -m xtraffic.evaluation.run_faithfulness_study                # full >=100
  python -m xtraffic.evaluation.run_faithfulness_study --limit 9      # quick check
  python -m xtraffic.evaluation.run_faithfulness_study --per-stratum 7

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.advisor.advisor import Advisor
from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import SPEED_CHANNEL, TOD_CHANNEL, _tod_to_clock
from ..models.gnn.loaders import _load_split, load_node_meta
from ..utils.io_utils import PKG_ROOT
from .bootstrap_ci import bootstrap_from_rows
from .faithfulness import NodeTable, mean_std, score_advisory

SEED = 42  # CLAUDE.md: fixed seed everywhere for reproducibility.

# Time-of-day bands as (label, lo_fraction, hi_fraction) over the clock channel.
# Same band edges the Phase-3 scenarios used, extended to cover the full day so
# every test window lands in exactly one band.
TOD_BANDS: List[Tuple[str, float, float]] = [
    ("night", 0.00, 0.25),        # ~00:00-06:00
    ("am_rush", 0.25, 0.42),      # ~06:00-10:00
    ("midday", 0.42, 0.62),       # ~10:00-15:00
    ("pm_rush", 0.62, 0.83),      # ~15:00-20:00
    ("evening", 0.83, 1.00),      # ~20:00-24:00
]
CONGESTION_LEVELS = ["low", "medium", "high"]  # by mean-speed tercile (high=slow)
CONDITIONS = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Scenario sampling (stratified, deterministic).
# ---------------------------------------------------------------------------
def _window_stats(dataset: str, scaler: Dict[str, float]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per test window return (tod0, mean_real_speed, slowest_valid_node).

    mean speed and validity are computed in REAL mph (missing = raw 0 -> ~0 mph)
    exactly like models/explainer/scenarios.py, because a missing sensor is ~0 in
    mph space but a large NEGATIVE in z-space — testing validity in z-space was a
    real Phase-3 bug that picked missing sensors as targets."""
    X, _ = _load_split(dataset, "test")            # [S, T, N, 2]
    S, T, N, _ = X.shape
    tod0 = X[:, 0, 0, TOD_CHANNEL].numpy()         # clock at window start [S]

    speed_mph = X[..., SPEED_CHANNEL].numpy() * scaler["std"] + scaler["mean"]  # [S,T,N]
    valid = speed_mph > 1.0                        # >1 mph == genuine reading
    flat_valid = valid.reshape(S, -1)
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(
            flat_valid.any(1),
            (speed_mph * valid).reshape(S, -1).sum(1) / np.clip(flat_valid.sum(1), 1, None),
            np.inf)                                # [S]; empty windows -> inf

    # Slowest VALID sensor per window = the target a planner most wants explained
    # (and the Phase-3 stratification target). [S]
    node_mean = speed_mph.mean(1)                  # mean over time per node [S,N]
    node_valid = valid.any(1)                      # nodes with any reading [S,N]
    node_mean_masked = np.where(node_valid, node_mean, np.inf)
    slowest_node = node_mean_masked.argmin(1)      # [S]
    return tod0, mean_speed, slowest_node


def sample_scenarios(dataset: str, scaler: Dict[str, float],
                     per_stratum: int) -> List[Dict[str, Any]]:
    """Deterministically sample up to `per_stratum` windows from each
    (time-of-day band x congestion tercile) cell. Congestion terciles are cut on
    the finite mean-speed distribution so 'high congestion' == slowest third."""
    tod0, mean_speed, slowest_node = _window_stats(dataset, scaler)
    finite = np.isfinite(mean_speed)

    # Congestion terciles by mean speed (lower speed = higher congestion).
    valid_speeds = mean_speed[finite]
    q33, q66 = np.quantile(valid_speeds, [1 / 3, 2 / 3])

    def congestion_of(s: float) -> Optional[str]:
        if not np.isfinite(s):
            return None
        if s <= q33:
            return "high"        # slowest third
        if s <= q66:
            return "medium"
        return "low"

    rng = random.Random(SEED)
    scenarios: List[Dict[str, Any]] = []
    for band, lo, hi in TOD_BANDS:
        for level in CONGESTION_LEVELS:
            cell = [i for i in range(len(mean_speed))
                    if finite[i] and lo <= tod0[i] < hi
                    and congestion_of(mean_speed[i]) == level]
            rng.shuffle(cell)                       # seeded -> reproducible
            for idx in cell[:per_stratum]:
                scenarios.append({
                    "sample_index": int(idx),
                    "target_node": int(slowest_node[idx]),
                    "tod_band": band,
                    "congestion": level,
                    "mean_speed_mph": float(round(mean_speed[idx], 2)),
                    "timestamp": "test#{} @ {}".format(idx, _tod_to_clock(float(tod0[idx]))),
                })
    return scenarios


# ---------------------------------------------------------------------------
# Explanation caching (build once, reuse across conditions + reruns).
# ---------------------------------------------------------------------------
def _explanation_cache_path(dataset: str, sample_index: int, node: int) -> str:
    d = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness",
                     "explanations_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{}_{}_{}.json".format(dataset, sample_index, node))


def build_or_load_explanation(builder: ExplanationBuilder, dataset: str,
                              sc: Dict[str, Any]) -> Dict[str, Any]:
    path = _explanation_cache_path(dataset, sc["sample_index"], sc["target_node"])
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from ..models.explainer.scenarios import load_window
    X = load_window(dataset, sc["sample_index"])            # [1, T, N, C]
    exp = builder.explain_prediction(
        X, target_node=sc["target_node"], horizon_step=6, timestamp=sc["timestamp"])
    exp["meta"]["city"] = dataset
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


# ---------------------------------------------------------------------------
# Decision caching (Phase 15b): make the study RESUMABLE and let the bootstrap
# run "on cached decisions, no full recompute" (the 15b gate). Each (scenario,
# condition, model) advisory+metrics is cached to one JSON, keyed so a rerun with
# a new condition reuses everything already computed and only runs what's missing.
# ---------------------------------------------------------------------------
def _decision_cache_path(out_dir: str, dataset: str, sample: int, node: int,
                         cond: str, model: str) -> str:
    d = os.path.join(out_dir, "decisions_cache")
    os.makedirs(d, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return os.path.join(
        d, "{}_{}_{}_{}_{}.json".format(dataset, sample, node, cond, safe_model))


# ---------------------------------------------------------------------------
# Aggregation + reporting.
# ---------------------------------------------------------------------------
_METRIC_KEYS = ["cause_precision", "cause_recall", "faithfulness_f1",
                "hallucination_rate", "quantitative_fidelity"]

# The per-scenario CSV columns, pinned so the schema stays stable even though the
# in-memory rows may carry extra debug keys (cited_locations, context_used) that
# we keep only in the decision cache / JSON. DictWriter drops the extras.
_CSV_COLS = (["sample_index", "target_node", "tod_band", "congestion",
              "condition", "advisory_error"]
             + _METRIC_KEYS + ["n_cited_causes", "n_topk"])


def _aggregate(rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["condition"] == condition]
    out: Dict[str, Any] = {"condition": condition, "n": len(sub)}
    for k in _METRIC_KEYS:
        m, s, n = mean_std([r[k] for r in sub])
        out[k + "_mean"] = m
        out[k + "_std"] = s
        out[k + "_n"] = n
    return out


def _print_table(aggs: List[Dict[str, Any]]) -> None:
    print("\n=== FAITHFULNESS BY CONDITION (mean +/- std) ===")
    hdr = "{:20s}".format("metric")
    for a in aggs:
        hdr += "{:>22s}".format("{} (n={})".format(a["condition"], a["n"]))
    print(hdr)
    print("-" * len(hdr))
    for k in _METRIC_KEYS:
        line = "{:20s}".format(k)
        for a in aggs:
            m, s = a[k + "_mean"], a[k + "_std"]
            cell = "nan" if m != m else "{:.3f} +/- {:.3f}".format(m, s)
            line += "{:>22s}".format(cell)
        print(line)


def _plots(rows: List[Dict[str, Any]], out_dir: str, conditions: List[str],
           tag: str = "") -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")               # headless: write files, no display
        import matplotlib.pyplot as plt
    except Exception as e:                   # pragma: no cover
        print("(matplotlib unavailable, skipping plots: {})".format(e))
        return None

    fig, axes = plt.subplots(1, 2, figsize=(2 + 2 * len(conditions), 4))
    for ax, key, title in [(axes[0], "faithfulness_f1", "Faithfulness F1"),
                           (axes[1], "hallucination_rate", "Hallucination rate")]:
        data = [[r[key] for r in rows if r["condition"] == c] for c in conditions]
        ax.boxplot(data, tick_labels=conditions, showmeans=True)
        ax.set_title(title)
        ax.set_xlabel("condition")
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("XTraffic faithfulness across conditions "
                 "(A=full, B=no-expl, C=no-context, C_RICH=fuller, D_CONTRA=false ctx)")
    fig.tight_layout()
    pdf = os.path.join(out_dir, "faithfulness_distributions{}.pdf".format(tag))
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--city", default="metr_la")
    ap.add_argument("--checkpoint",
                    default="models/gnn/checkpoints/metr_la_best.pt")
    ap.add_argument("--per-stratum", type=int, default=7,
                    help="windows per (tod-band x congestion) cell; 15 cells x 7 ~= 105")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total scenarios to the FIRST N (0 = no cap). NOTE: "
                         "this truncates the front of the stratum-ordered list, so "
                         "a small --limit is NOT stratified (it will be all one "
                         "band) — use it only for smoke tests, not the real gate.")
    ap.add_argument("--conditions", default="A,B,C")
    ap.add_argument("--model", default=None,
                    help="override the Ollama model in advisor.yaml (Phase 7 "
                         "LLM-model ablation, e.g. qwen2.5:7b). Default = config.")
    ap.add_argument("--out-tag", default="",
                    help="suffix on output filenames so ablation runs (different "
                         "model/dataset) don't overwrite each other.")
    args = ap.parse_args()
    tag = ("_" + args.out_tag) if args.out_tag else ""

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    device = torch.device("cpu")            # explainer is tiny + deterministic
    builder = ExplanationBuilder(args.checkpoint, args.dataset, device=device)
    # LLM-model ablation: deep-copy the advisor config and swap the model name so
    # the SAME pipeline runs against a second local model (shows results aren't an
    # artifact of one LLM). Default path is unchanged.
    if args.model:
        import copy
        from ..models.advisor.advisor import load_advisor_config
        adv_cfg = copy.deepcopy(load_advisor_config())
        adv_cfg.setdefault("ollama", {})["model"] = args.model
        advisor = Advisor(args.city, cfg=adv_cfg)
        print(f"[study] LLM-model ablation: using {args.model}")
    else:
        advisor = Advisor(args.city)
    table = NodeTable(load_node_meta(args.dataset))

    scenarios = sample_scenarios(args.dataset, builder.scaler, args.per_stratum)
    if args.limit and len(scenarios) > args.limit:
        scenarios = scenarios[:args.limit]
    print("Sampled {} scenarios across {} tod-bands x {} congestion levels."
          .format(len(scenarios), len(TOD_BANDS), len(CONGESTION_LEVELS)))

    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness")
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    n_cached = 0
    for i, sc in enumerate(scenarios):
        exp = build_or_load_explanation(builder, args.dataset, sc)
        for cond in conditions:
            cpath = _decision_cache_path(out_dir, args.dataset, sc["sample_index"],
                                         sc["target_node"], cond, advisor.model)
            if os.path.exists(cpath):
                # RESUMABLE: this (scenario, condition, model) was already scored.
                with open(cpath) as f:
                    row = json.load(f)
                n_cached += 1
            else:
                res = advisor.advise_condition(exp, cond)
                metrics = score_advisory(exp, res["advisory"], table)
                row = {
                    "sample_index": sc["sample_index"],
                    "target_node": sc["target_node"],
                    "tod_band": sc["tod_band"],
                    "congestion": sc["congestion"],
                    "condition": cond,
                    "advisory_error": res["advisory"].get("_error", ""),
                    # Debug detail kept in the decision cache (not the CSV): what
                    # the LLM cited + which context it saw — invaluable for the
                    # D_CONTRA analysis (did it cite the decoy corridor?).
                    "cited_locations": [c.get("location") for c in
                                        (res["advisory"].get("cited_causes") or [])
                                        if isinstance(c, dict)],
                    "context_used": res.get("context_used", []),
                }
                for k in _METRIC_KEYS + ["n_cited_causes", "n_topk"]:
                    row[k] = metrics[k]
                with open(cpath, "w") as f:
                    json.dump(row, f, indent=2)
            rows.append(row)
        print("  [{}/{}] {} {} cong={}  {} done".format(
            i + 1, len(scenarios), sc["timestamp"], sc["tod_band"],
            sc["congestion"], "/".join(conditions)))
    if n_cached:
        print("  (reused {} cached decisions)".format(n_cached))

    # --- write per-scenario CSV -------------------------------------------
    csv_path = os.path.join(out_dir, f"faithfulness_per_scenario{tag}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            # Pinned columns; extrasaction="ignore" drops the debug keys so the
            # CSV schema is identical to the committed A/B/C file.
            w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # --- aggregate overall + by time-of-day + by congestion ---------------
    aggs = [_aggregate(rows, c) for c in conditions]
    by_tod = {band: [_aggregate([r for r in rows if r["tod_band"] == band], c)
                     for c in conditions] for band, _, _ in TOD_BANDS}
    by_cong = {lvl: [_aggregate([r for r in rows if r["congestion"] == lvl], c)
                     for c in conditions] for lvl in CONGESTION_LEVELS}

    # Phase 15b: bootstrap 95% CIs on every per-condition metric AND on the
    # A-vs-other paired differences, embedded straight into the summary so the
    # A/B/C table gains CI columns with no separate step.
    bootstrap = bootstrap_from_rows(rows, conditions=conditions)

    summary = {
        "n_scenarios": len(scenarios),
        "conditions": conditions,
        "overall": aggs,
        "by_time_of_day": by_tod,
        "by_congestion": by_cong,
        "model": advisor.model,
        "seed": SEED,
        "bootstrap_ci": bootstrap,
    }
    with open(os.path.join(out_dir, f"faithfulness_summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _print_table(aggs)
    pdf = _plots(rows, out_dir, conditions, tag)

    # --- headline checks (with bootstrap CIs on the gaps) -----------------
    by_cond = {x["condition"]: x for x in aggs}
    pw = bootstrap["pairwise_diff"]

    def _gap(label: str, metric: str) -> str:
        d = pw.get(label, {}).get(metric)
        if not d or d.get("mean_diff") is None:
            return ""
        sig = "excludes 0" if d["excludes_zero"] else "spans 0"
        return "  [95% CI {:+.3f}, {:+.3f}; {}]".format(d["lo"], d["hi"], sig)

    if "A" in by_cond and "B" in by_cond:
        a, b = by_cond["A"], by_cond["B"]
        print("\nHEADLINE (A vs B — the grounding result):")
        print("  F1:            A {:.3f}  vs  B {:.3f}{}".format(
            a["faithfulness_f1_mean"], b["faithfulness_f1_mean"],
            _gap("A-B", "faithfulness_f1")))
        print("  hallucination: A {:.3f}  vs  B {:.3f}{}".format(
            a["hallucination_rate_mean"], b["hallucination_rate_mean"],
            _gap("A-B", "hallucination_rate")))
        verdict = ("A beats B -> grounding proven"
                   if a["faithfulness_f1_mean"] > b["faithfulness_f1_mean"]
                   else "A does NOT beat B -> investigate (see CLAUDE.md gate note)")
        print("  ->", verdict)

    # C_RICH: does a FULLER context block move faithfulness? (Expect: no.)
    if "C_RICH" in by_cond and "A" in by_cond:
        cr, a = by_cond["C_RICH"], by_cond["A"]
        print("\nC_RICH (fuller context) vs A:")
        print("  F1:            A {:.3f}  vs  C_RICH {:.3f}{}".format(
            a["faithfulness_f1_mean"], cr["faithfulness_f1_mean"],
            _gap("A-C_RICH", "faithfulness_f1")))
        print("  hallucination: A {:.3f}  vs  C_RICH {:.3f}{}".format(
            a["hallucination_rate_mean"], cr["hallucination_rate_mean"],
            _gap("A-C_RICH", "hallucination_rate")))
        print("  -> spans 0 => a fuller context block does NOT move faithfulness "
              "(strengthens the 'context is orthogonal' claim).")

    # D_CONTRA: does FALSE context pull the LLM off the math? (Expect: barely.)
    if "D_CONTRA" in by_cond:
        dc = by_cond["D_CONTRA"]
        d_rows = [r for r in rows if r["condition"] == "D_CONTRA"]
        faithful = sum(1 for r in d_rows if (r.get("hallucination_rate") or 0) == 0)
        print("\nD_CONTRA (contradictory context) — grounding stress test:")
        if "A" in by_cond:
            print("  hallucination: A {:.3f}  vs  D_CONTRA {:.3f}{}".format(
                by_cond["A"]["hallucination_rate_mean"],
                dc["hallucination_rate_mean"],
                _gap("A-D_CONTRA", "hallucination_rate")))
        print("  stayed faithful (halluc==0): {}/{} scenarios".format(
            faithful, len(d_rows)))
        print("  -> if hallucination stays ~0, the LLM trusted the math over the "
              "false context (grounding robust); if it spikes, that's a real "
              "limitation to report.")

    print("\nWrote:")
    print("  " + csv_path)
    print("  " + os.path.join(out_dir, f"faithfulness_summary{tag}.json"))
    if pdf:
        print("  " + pdf)


if __name__ == "__main__":
    main()
