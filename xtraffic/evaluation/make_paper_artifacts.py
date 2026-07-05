"""Phase 9 — regenerate EVERY figure and table in the paper from logged results.

This is the paper's reproducibility contract: no figure or table is ever drawn by
hand. Everything below reads a file that some earlier phase wrote into
evaluation/results/ and turns it into a publication artifact under
evaluation/paper/. If you can rerun the phases, you can regenerate the paper.

Run:
    python -m xtraffic.evaluation.make_paper_artifacts

Design rules (called out because a reviewer / my professor will ask):
  * FIGURES are vector PDF (except the architecture diagram, which is SVG),
    colorblind-safe palettes, fonts sized for a two-column IEEE template.
  * TABLES are LaTeX booktabs, ready to \\input{} straight into the manuscript.
  * Every artifact is INDEPENDENT and wrapped so that a missing upstream result
    (e.g. the Colab fusion run that is still OWED) produces a clearly-labelled
    PENDING placeholder instead of crashing the whole script. That way the paper
    skeleton always builds and the holes are honest and visible.

Python 3.9 compatible (typing.Optional/Dict, no `X | Y`).
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from ..utils.io_utils import PKG_ROOT

# ----------------------------------------------------------------------------
# Paths + shared style
# ----------------------------------------------------------------------------
RESULTS = os.path.join(PKG_ROOT, "evaluation", "results")
PAPER = os.path.join(PKG_ROOT, "evaluation", "paper")
FIG_DIR = os.path.join(PAPER, "figures")
TAB_DIR = os.path.join(PAPER, "tables")

# Colorblind-safe qualitative palette (Wong 2011) — used for every categorical
# split so the whole paper reads as one visual system.
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
}
COND_COLOR = {"A": CB["blue"], "B": CB["vermillion"], "C": CB["green"],
              "RAW": CB["grey"], "XAI": CB["orange"], "XTRAFFIC": CB["blue"]}

# IEEE two-column: a single column is ~3.5in wide. Size fonts accordingly.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,  # embed real fonts
})


def _read_json(path: str) -> Optional[dict]:
    """Load a logged result, or None if the phase that writes it hasn't run."""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _read_csv_rows(path: str) -> Optional[List[Dict[str, str]]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _pending_figure(path: str, title: str, reason: str) -> None:
    """Draw a labelled placeholder so the paper skeleton always has a slot and
    the missing result is loud and visible rather than silently absent."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=9, weight="bold")
    ax.text(0.5, 0.38, "PENDING\n" + reason, ha="center", va="center",
            fontsize=7, color=CB["vermillion"])
    fig.savefig(path)
    plt.close(fig)
    print("  [PENDING] {} -> {} ({})".format(title, os.path.basename(path), reason))


def _pending_table(path: str, title: str, reason: str) -> None:
    with open(path, "w") as f:
        f.write("% PENDING: {}\n% {}\n".format(title, reason))
        f.write("% This table regenerates automatically once the upstream result exists.\n")
    print("  [PENDING] {} -> {} ({})".format(title, os.path.basename(path), reason))


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return "{:.{nd}f}".format(x, nd=nd)


# ============================================================================
# FIGURE 1 — System architecture diagram (three layers, data flow), as SVG.
# ============================================================================
def fig1_architecture() -> None:
    """Hand-composed SVG of the three-layer pipeline. SVG (not PDF) because it is
    a schematic, not plotted data — vector text stays crisp at any column width."""
    out = os.path.join(FIG_DIR, "fig1_architecture.svg")
    boxes = [
        ("Heterogeneous inputs\n(sensors + weather\n+ events + transit)", CB["grey"]),
        ("LAYER 1\nST-GNN prediction\n(Graph WaveNet + MOD 1/2/3)", CB["blue"]),
        ("LAYER 2\nGNNExplainer\nmathematical explanation (JSON)", CB["green"]),
        ("LAYER 3\nLLM advisor (local Ollama)\ngrounded recommendations", CB["orange"]),
    ]
    bw, bh, gap, x0, y = 250, 90, 40, 30, 60
    W = x0 * 2 + len(boxes) * bw + (len(boxes) - 1) * gap
    H = 220
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
             'viewBox="0 0 {} {}" font-family="Helvetica,Arial,sans-serif">'.format(W, H, W, H)]
    parts.append('<rect width="{}" height="{}" fill="white"/>'.format(W, H))
    for i, (label, color) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        parts.append('<rect x="{}" y="{}" width="{}" height="{}" rx="10" '
                     'fill="{}" fill-opacity="0.15" stroke="{}" stroke-width="2"/>'
                     .format(x, y, bw, bh, color, color))
        for j, line in enumerate(label.split("\n")):
            weight = "bold" if j == 0 else "normal"
            parts.append('<text x="{}" y="{}" text-anchor="middle" font-size="13" '
                         'font-weight="{}" fill="#222">{}</text>'
                         .format(x + bw / 2, y + 26 + j * 18, weight, line))
        if i < len(boxes) - 1:                       # arrow to the next stage
            ax_ = x + bw
            parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#444" '
                         'stroke-width="2" marker-end="url(#arrow)"/>'
                         .format(ax_, y + bh / 2, ax_ + gap, y + bh / 2))
    parts.append('<defs><marker id="arrow" markerWidth="10" markerHeight="10" '
                 'refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" '
                 'fill="#444"/></marker></defs>')
    parts.append('<text x="{}" y="{}" text-anchor="middle" font-size="12" '
                 'fill="#666">Faithfulness metric aligns Layer 2 (math) with '
                 'Layer 3 (language)</text>'.format(W / 2, H - 20))
    parts.append("</svg>")
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print("  [OK] Figure 1 architecture -> {}".format(os.path.basename(out)))


# ============================================================================
# FIGURE 2 — Explanation visualization (reuse the Phase-3 renderer).
# ============================================================================
def fig2_explanation() -> None:
    out = os.path.join(FIG_DIR, "fig2_explanation.pdf")
    # Pick the most congested committed scenario (best visual example).
    candidates = ["rush_hour_pm", "high_congestion", "rush_hour_am"]
    expl_dir = os.path.join(RESULTS, "explanations")
    chosen = None
    for name in candidates:
        p = os.path.join(expl_dir, name + ".json")
        if os.path.isfile(p):
            chosen = p
            break
    if chosen is None:
        _pending_figure(out, "Figure 2: explanation", "no Phase-3 explanation JSON found")
        return
    try:
        from .visualize_explanation import visualize   # reuse, don't duplicate
        visualize(chosen, dataset="metr_la", out_path=out)
        print("  [OK] Figure 2 explanation -> {} (from {})"
              .format(os.path.basename(out), os.path.basename(chosen)))
    except Exception as e:  # processed data may be absent on a fresh clone
        _pending_figure(out, "Figure 2: explanation",
                        "renderer needs processed metr_la: {}".format(type(e).__name__))


# ============================================================================
# FIGURES 3 & 4 — learned alpha and per-modality gates over training.
# ============================================================================
def _train_curve_rows() -> Optional[List[Dict[str, str]]]:
    return _read_csv_rows(os.path.join(RESULTS, "train_metr_la.csv"))


def fig3_alpha() -> None:
    out = os.path.join(FIG_DIR, "fig3_alpha.pdf")
    rows = _train_curve_rows()
    if not rows or "alpha" not in rows[0]:
        _pending_figure(out, "Figure 3: learned alpha", "train_metr_la.csv missing alpha")
        return
    epochs = [int(r["epoch"]) for r in rows]
    alpha = [float(r["alpha"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(epochs, alpha, marker="o", color=CB["blue"], lw=1.5)
    ax.axhline(0.5, ls="--", color=CB["grey"], lw=0.8, label="init (0.5)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\alpha$ (physical-vs-semantic mix)")
    ax.set_title("Reliance on physical vs. learned graph")
    ax.legend()
    if len(epochs) < 3:                       # honest: this is the placeholder ckpt
        ax.text(0.5, 0.05, "placeholder ckpt ({} epoch(s)) — refresh on full run"
                .format(len(epochs)), transform=ax.transAxes, ha="center",
                fontsize=6, color=CB["vermillion"])
    fig.savefig(out)
    plt.close(fig)
    print("  [OK] Figure 3 alpha -> {}".format(os.path.basename(out)))


def fig4_gates() -> None:
    out = os.path.join(FIG_DIR, "fig4_modality_gates.pdf")
    rows = _train_curve_rows()
    gate_cols = [c for c in (rows[0].keys() if rows else []) if c.startswith("gate_")]
    if not rows or not gate_cols:
        _pending_figure(out, "Figure 4: modality gates",
                        "train_metr_la.csv has no gate_* columns (traffic-only run)")
        return
    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    palette = [CB["blue"], CB["orange"], CB["green"], CB["purple"], CB["sky"]]
    for i, col in enumerate(gate_cols):
        ax.plot(epochs, [float(r[col]) for r in rows], marker="o", lw=1.3,
                color=palette[i % len(palette)], label=col.replace("gate_", ""))
    ax.set_xlabel("epoch")
    ax.set_ylabel("learned gate value")
    ax.set_title("Per-modality fusion gates over training")
    ax.legend(ncol=2)
    if len(epochs) < 3:
        ax.text(0.5, 0.05, "placeholder ckpt — refresh on full fusion run",
                transform=ax.transAxes, ha="center", fontsize=6, color=CB["vermillion"])
    fig.savefig(out)
    plt.close(fig)
    print("  [OK] Figure 4 gates -> {}".format(os.path.basename(out)))


# ============================================================================
# FIGURE 5 — faithfulness distributions across conditions A/B/C.
# ============================================================================
def fig5_faithfulness() -> None:
    out = os.path.join(FIG_DIR, "fig5_faithfulness.pdf")
    per = _read_csv_rows(os.path.join(RESULTS, "faithfulness", "faithfulness_per_scenario.csv"))
    if not per:
        _pending_figure(out, "Figure 5: faithfulness A/B/C",
                        "run_faithfulness_study.py not run")
        return
    # Group F1 by condition. Column names come from faithfulness.py's CSV.
    f1_col = _pick_col(per[0], ["faithfulness_f1", "f1"])
    cond_col = _pick_col(per[0], ["condition", "cond"])
    if not f1_col or not cond_col:
        _pending_figure(out, "Figure 5: faithfulness A/B/C",
                        "per-scenario CSV missing condition/f1 columns")
        return
    conds = ["A", "B", "C"]
    data = [[float(r[f1_col]) for r in per
             if r[cond_col] == c and _isnum(r[f1_col])] for c in conds]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bp = ax.boxplot(data, tick_labels=conds, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], conds):
        patch.set_facecolor(COND_COLOR[c])
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color(CB["black"])
    ax.set_ylabel("faithfulness F1")
    ax.set_xlabel("condition")
    ax.set_title("A=full  B=no-explanation  C=no-city-context")
    ax.set_ylim(-0.05, 1.05)
    n = len(data[0])
    if n < 30:
        ax.text(0.5, 0.02, "smoke sample (n={} per condition)".format(n),
                transform=ax.transAxes, ha="center", fontsize=6, color=CB["vermillion"])
    fig.savefig(out)
    plt.close(fig)
    print("  [OK] Figure 5 faithfulness -> {}".format(os.path.basename(out)))


# ============================================================================
# FIGURE 6 — cross-city transfer bar chart.
# ============================================================================
def fig6_cross_city() -> None:
    out = os.path.join(FIG_DIR, "fig6_cross_city.pdf")
    data = _read_json(os.path.join(RESULTS, "cross_city", "transfer.json"))
    if data is None:
        # cross_city.py names its file after source/target; try a glob-free scan.
        cc_dir = os.path.join(RESULTS, "cross_city")
        if os.path.isdir(cc_dir):
            for fn in sorted(os.listdir(cc_dir)):
                if fn.endswith(".json"):
                    data = _read_json(os.path.join(cc_dir, fn))
                    break
    if not data or "rows" not in data:
        _pending_figure(out, "Figure 6: cross-city transfer",
                        "cross_city.py not run (needs Chicago + source ckpt)")
        return
    rows = data["rows"]
    settings = [r["setting"] for r in rows]
    mae30 = [float(r.get("mae_30min", "nan")) for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(range(len(settings)), mae30, color=CB["blue"], alpha=0.75)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, rotation=15, ha="right")
    ax.set_ylabel("30-min MAE (mph)")
    ax.set_title("Cross-city transfer ({} -> {})"
                 .format(data.get("source", "?"), data.get("target", "?")))
    fig.savefig(out)
    plt.close(fig)
    print("  [OK] Figure 6 cross-city -> {}".format(os.path.basename(out)))


# ============================================================================
# FIGURE 7 — human study results with error bars.
# ============================================================================
def fig7_human_study() -> None:
    out = os.path.join(FIG_DIR, "fig7_human_study.pdf")
    rep = _read_json(os.path.join(RESULTS, "human_study", "analysis_report.json"))
    if not rep or "per_condition" not in rep:
        _pending_figure(out, "Figure 7: human study", "analyze.py not run")
        return
    pc = rep["per_condition"]
    conds = [c for c in ["RAW", "XAI", "XTRAFFIC"] if c in pc]
    metrics = [("correct", "decision accuracy"), ("usefulness", "usefulness (1-7)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6, 2.4))
    for ax, (key, label) in zip(axes, metrics):
        vals, errs = [], []
        for c in conds:
            v = pc[c].get(key)
            n = max(int(pc[c].get("n", 1)), 1)
            vals.append(v if v is not None else float("nan"))
            # crude SE for a proportion / mean; small-n caveat printed below
            errs.append((math.sqrt(max(v, 0) * max(1 - v, 0) / n)
                         if key == "correct" and v is not None else 0.0))
        ax.bar(range(len(conds)), vals, yerr=errs, capsize=3,
               color=[COND_COLOR[c] for c in conds], alpha=0.8)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=15, ha="right")
        ax.set_title(label)
    fig.suptitle("Human decision quality by condition", fontsize=9)
    n_resp = rep.get("n_responses", "?")
    fig.text(0.5, -0.02, "pilot data (n_responses={}) — indicative only"
             .format(n_resp), ha="center", fontsize=6, color=CB["vermillion"])
    fig.savefig(out)
    plt.close(fig)
    print("  [OK] Figure 7 human study -> {}".format(os.path.basename(out)))


# ============================================================================
# TABLE 1 — prediction vs baselines (METR-LA, three horizons).
# ============================================================================
def _overall_and_horizons(d: dict) -> Dict[str, Dict[str, float]]:
    return d.get("per_horizon", {})


def table1_prediction() -> None:
    out = os.path.join(TAB_DIR, "table1_prediction.tex")
    model = _read_json(os.path.join(RESULTS, "eval_metr_la_on_metr_la.json"))
    hist = _read_json(os.path.join(RESULTS, "baseline_historical_average_metr_la.json"))
    lin = _read_json(os.path.join(RESULTS, "baseline_linear_regression_metr_la.json"))
    if not model:
        _pending_table(out, "Table 1 prediction", "evaluate.py not run")
        return
    horizons = ["15min", "30min", "60min"]
    named = [("XTraffic (ours)", model), ("Historical Avg.", hist),
             ("Linear Reg.", lin)]
    lines = [
        "% Table 1 — prediction accuracy vs baselines, METR-LA test.",
        "\\begin{tabular}{l" + "c" * (len(horizons) * 2) + "}",
        "\\toprule",
        "& " + " & ".join("\\multicolumn{2}{c}{%s}" % h for h in horizons) + " \\\\",
        "Model & " + " & ".join("MAE & RMSE" for _ in horizons) + " \\\\",
        "\\midrule",
    ]
    for name, d in named:
        if not d:
            continue
        ph = _overall_and_horizons(d)
        cells = []
        for h in horizons:
            cells.append(_fmt(ph.get(h, {}).get("mae"), 2))
            cells.append(_fmt(ph.get(h, {}).get("rmse"), 2))
        lead = "\\textbf{%s}" % name if "ours" in name else name
        lines.append(lead + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write_lines(out, lines)
    print("  [OK] Table 1 prediction -> {}".format(os.path.basename(out)))


# ============================================================================
# TABLE 2 — explainability metrics vs random baseline.
# ============================================================================
def table2_explainability() -> None:
    out = os.path.join(TAB_DIR, "table2_explainability.tex")
    d = _read_json(os.path.join(RESULTS, "explainer_metrics_metr_la.json"))
    if not d:
        _pending_table(out, "Table 2 explainability", "explainer_metrics.py not run")
        return

    def mean(key: str) -> Optional[float]:
        v = d.get(key)
        return v[0] if isinstance(v, list) and v else None

    rows = [
        ("Fidelity+ ($\\uparrow$)", mean("fidelity_plus"), mean("fidelity_plus_random")),
        ("Fidelity- ($\\downarrow$)", mean("fidelity_minus"), mean("fidelity_minus_random")),
        ("Stability ($\\uparrow$)", mean("stability"), None),
        ("Sparsity", mean("sparsity"), None),
    ]
    lines = [
        "% Table 2 -- explainability metrics vs random top-k baseline "
        "(n={}, k={}).".format(d.get("n_samples", "?"), d.get("top_k", "?")),
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & XTraffic & Random \\\\",
        "\\midrule",
    ]
    for name, ours, rnd in rows:
        lines.append("%s & %s & %s \\\\" % (name, _fmt(ours), _fmt(rnd)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write_lines(out, lines)
    print("  [OK] Table 2 explainability -> {}".format(os.path.basename(out)))


# ============================================================================
# TABLE 3 — faithfulness across conditions (and cities if present).
# ============================================================================
def table3_faithfulness() -> None:
    out = os.path.join(TAB_DIR, "table3_faithfulness.tex")
    d = _read_json(os.path.join(RESULTS, "faithfulness", "faithfulness_summary.json"))
    if not d or "overall" not in d:
        _pending_table(out, "Table 3 faithfulness", "run_faithfulness_study.py not run")
        return
    by_cond = {r["condition"]: r for r in d["overall"]}
    order = [c for c in ["A", "B", "C"] if c in by_cond]
    labels = {"A": "A: full", "B": "B: no-expl.", "C": "C: no-city"}
    lines = [
        "% Table 3 -- faithfulness by condition, METR-LA "
        "(n={}, model={}).".format(d.get("n_scenarios", "?"), d.get("model", "?")),
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Condition & Precision & Recall & F1 & Halluc. \\\\",
        "\\midrule",
    ]
    for c in order:
        r = by_cond[c]
        lines.append("%s & %s & %s & %s & %s \\\\" % (
            labels[c],
            _fmt(r.get("cause_precision_mean")),
            _fmt(r.get("cause_recall_mean")),
            _fmt(r.get("faithfulness_f1_mean")),
            _fmt(r.get("hallucination_rate_mean")),
        ))
    lines += ["\\bottomrule", "\\end{tabular}"]
    if d.get("n_scenarios", 0) and int(d["n_scenarios"]) < 30:
        lines.insert(1, "% NOTE: smoke sample (n={}) -- refresh on the full study."
                     .format(d["n_scenarios"]))
    _write_lines(out, lines)
    print("  [OK] Table 3 faithfulness -> {}".format(os.path.basename(out)))


# ============================================================================
# TABLE 4 — full ablation (reads the Phase-7 harness output if present).
# ============================================================================
def table4_ablation() -> None:
    out = os.path.join(TAB_DIR, "table4_ablation.tex")
    src = os.path.join(RESULTS, "ablations", "model_ablations.tex")
    if os.path.isfile(src):
        with open(src, "r") as f:
            body = f.read()
        _write_lines(out, ["% Table 4 — model ablation (copied from Phase-7 harness).",
                           body.rstrip()])
        print("  [OK] Table 4 ablation -> {} (from harness)".format(os.path.basename(out)))
    else:
        _pending_table(out, "Table 4 ablation",
                       "ablations.py multi-seed run not done (Colab-scale)")


# ============================================================================
# TABLE 5 — human study statistics.
# ============================================================================
def table5_human_study() -> None:
    out = os.path.join(TAB_DIR, "table5_human_study.tex")
    rep = _read_json(os.path.join(RESULTS, "human_study", "analysis_report.json"))
    if not rep or "per_condition" not in rep:
        _pending_table(out, "Table 5 human study", "analyze.py not run")
        return
    pc = rep["per_condition"]
    conds = [c for c in ["RAW", "XAI", "XTRAFFIC"] if c in pc]
    lines = [
        "% Table 5 -- human decision-quality by condition "
        "(pilot, n_responses={}).".format(rep.get("n_responses", "?")),
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Condition & n & Accuracy & Confidence & Usefulness \\\\",
        "\\midrule",
    ]
    for c in conds:
        r = pc[c]
        lines.append("%s & %s & %s & %s & %s \\\\" % (
            c, r.get("n", "?"), _fmt(r.get("correct"), 2),
            _fmt(r.get("confidence"), 2), _fmt(r.get("usefulness"), 2)))
    lines += ["\\midrule"]
    # Paired Wilcoxon headline (XTRAFFIC vs RAW accuracy), if computed.
    pt = rep.get("paired_tests", {}).get("accuracy", {}).get("XTRAFFIC_vs_RAW", {})
    if pt:
        lines.append("\\multicolumn{5}{l}{\\footnotesize XTRAFFIC vs RAW (accuracy): "
                     "p=%s, effect r=%s, n=%s} \\\\" % (
                         _fmt(pt.get("p_value"), 3), _fmt(pt.get("effect_size_r"), 2),
                         pt.get("n_pairs", "?")))
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write_lines(out, lines)
    print("  [OK] Table 5 human study -> {}".format(os.path.basename(out)))


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _isnum(s: str) -> bool:
    try:
        return not math.isnan(float(s))
    except (TypeError, ValueError):
        return False


def _pick_col(row: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row:
            return c
    # tolerate suffixes/prefixes
    for key in row:
        for c in candidates:
            if c in key:
                return key
    return None


# ----------------------------------------------------------------------------
def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    print("Writing paper artifacts into {}".format(PAPER))
    print("-- Figures --")
    for fn in (fig1_architecture, fig2_explanation, fig3_alpha, fig4_gates,
               fig5_faithfulness, fig6_cross_city, fig7_human_study):
        try:
            fn()
        except Exception as e:  # one bad artifact must never sink the rest
            print("  [ERROR] {}: {}: {}".format(fn.__name__, type(e).__name__, e))
    print("-- Tables --")
    for fn in (table1_prediction, table2_explainability, table3_faithfulness,
               table4_ablation, table5_human_study):
        try:
            fn()
        except Exception as e:
            print("  [ERROR] {}: {}: {}".format(fn.__name__, type(e).__name__, e))
    print("\nDone. Artifacts in {}".format(PAPER))


if __name__ == "__main__":
    main()
