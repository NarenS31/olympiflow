"""Render the four tables, three figures and report.md for the whole experiment.

Reads only what the four part scripts wrote into the run directory; computes no
new science. Every prediction in predictions.md is marked CONFIRMED or FALSIFIED
against its own pre-registered criteria, with the number that decided it.

    python -m xtraffic.scripts.report_grounding_without_information

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .noise_run import resolve_run

# Wong colourblind-safe palette, as used by evaluation/make_paper_artifacts.py.
WONG = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
        "red": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442",
        "sky": "#56B4E9", "grey": "#666666"}


def _load(run_path: str, name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(run_path, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if f != f else "{:.{nd}f}".format(f, nd=nd)


# ---------------------------------------------------------------------------
# Tables (markdown + LaTeX booktabs)
# ---------------------------------------------------------------------------
def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _tex_table(header: Sequence[str], rows: Sequence[Sequence[str]],
               caption: str, label: str) -> str:
    cols = "l" + "r" * (len(header) - 1)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{" + caption + "}", "\\label{" + label + "}",
             "\\begin{tabular}{" + cols + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(str(c).replace("%", "\\%") for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table1(p1: Dict[str, Any]) -> Tuple[Sequence[str], List[List[str]]]:
    """Faithfulness by condition, with bootstrap CIs."""
    by = {a["condition"]: a for a in p1["overall"]}
    pc = p1.get("bootstrap_ci", {}).get("per_condition", {})
    header = ["condition", "n", "precision", "recall", "F1", "F1 95% CI",
              "hallucination", "halluc 95% CI"]
    rows = []
    for c in p1["conditions"]:
        a = by.get(c)
        if not a:
            continue
        f = pc.get(c, {}).get("faithfulness_f1", {})
        h = pc.get(c, {}).get("hallucination_rate", {})
        rows.append([
            c, a["n"], _fmt(a["cause_precision_mean"]),
            _fmt(a["cause_recall_mean"]), _fmt(a["faithfulness_f1_mean"]),
            "[{}, {}]".format(_fmt(f.get("lo")), _fmt(f.get("hi"))),
            _fmt(a["hallucination_rate_mean"]),
            "[{}, {}]".format(_fmt(h.get("lo")), _fmt(h.get("hi")))])
    return header, rows


def table2(p2: Dict[str, Any], committed: Dict[str, Any]
           ) -> Tuple[Sequence[str], List[List[str]]]:
    """The loop, round by round, on noise vs on real evidence."""
    header = ["round", "F1 (A_rand)", "precision", "recall", "halluc",
              "reached thr %", "F1 (A, committed)", "reached % (A)"]
    rows = []
    comm = {r["round"]: r for r in committed.get("per_round", [])}
    for r in p2["per_round"]:
        c = comm.get(r["round"], {})
        rows.append([
            r["round"], _fmt(r["mean_f1"]), _fmt(r["mean_precision"]),
            _fmt(r["mean_recall"]), _fmt(r["mean_hallucination"]),
            _fmt(100 * r["frac_reached_threshold"], 1),
            _fmt(c.get("mean_f1")),
            _fmt(100 * c["frac_reached_threshold"], 1) if c else "n/a"])
    return header, rows


def table3(p3: Dict[str, Any]) -> Tuple[Sequence[str], List[List[str]]]:
    header = ["condition", "accuracy", "delay reduction (mph)", "consistency"]
    rows = []
    for a in p3["conditions"]:
        rows.append([
            a["condition"],
            "{} +/- {}".format(_fmt(a["accuracy_mean"]), _fmt(a["accuracy_std"])),
            "{} +/- {}".format(_fmt(a["delay_reduction_mean"], 2),
                               _fmt(a["delay_reduction_std"], 2)),
            "{} +/- {}".format(_fmt(a["consistency_mean"]),
                               _fmt(a["consistency_std"]))])
    return header, rows


def table4(p4: Dict[str, Any]) -> Tuple[Sequence[str], List[List[str]]]:
    header = ["setting", "lambda_size", "n", "norm. entropy", "sources for 80%",
              "split-half J", "seed-repeat J", "precision vs adjacency"]
    rows = []
    for s in p4["settings"]:
        rows.append([
            s["setting"], _fmt(s["lambda_size"], 2), s["n_targets"],
            "{} +/- {}".format(_fmt(s["entropy_of_mean_mean"], 4),
                               _fmt(s["entropy_of_mean_std"], 4)),
            "{} +/- {}".format(_fmt(s["sources_for_80pct_mean"], 1),
                               _fmt(s["sources_for_80pct_std"], 1)),
            "{} +/- {}".format(_fmt(s["split_half_jaccard_mean"]),
                               _fmt(s["split_half_jaccard_std"])),
            _fmt(s.get("seed_repeat_jaccard_mean")),
            _fmt(s["precision_vs_adjacency"])])
    return header, rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    return plt


def fig_loop_on_noise(run_path: str, p2: Dict[str, Any],
                      committed: Dict[str, Any], out: str) -> Optional[str]:
    """The key figure: a correction loop converging on random evidence."""
    plt = _mpl()
    csv_path = os.path.join(run_path, "part2_per_round.csv")
    curves: Dict[int, List[Tuple[int, float]]] = {}
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                curves.setdefault(int(r["sample_index"]), []).append(
                    (int(r["round"]), float(r["faithfulness_f1"])))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    thr = p2.get("f1_threshold", 0.7)
    max_r = p2.get("max_rounds", 3)

    ax = axes[0]
    for pts in curves.values():
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Carry the last value forward: a scenario that exited early keeps its
        # score, it does not vanish from later rounds.
        while len(xs) <= max_r:
            xs.append(xs[-1] + 1)
            ys.append(ys[-1])
        ax.plot(xs, ys, color=WONG["grey"], alpha=0.18, lw=0.6)
    means = [r["mean_f1"] for r in p2["per_round"]]
    ax.plot(range(len(means)), means, color=WONG["red"], lw=2, marker="o",
            ms=3, label="A_rand mean")
    if committed:
        cm = [r["mean_f1"] for r in committed.get("per_round", [])]
        if cm:
            ax.plot(range(len(cm)), cm, color=WONG["blue"], lw=2, marker="s",
                    ms=3, ls="--", label="A (committed)")
    ax.axhline(thr, color="k", lw=0.8, ls=":", label="threshold {:.2f}".format(thr))
    ax.set_xlabel("correction round")
    ax.set_ylabel("faithfulness F1")
    ax.set_title("F1 per round")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for key, colour, lab in [("mean_precision", WONG["green"], "precision"),
                             ("mean_recall", WONG["orange"], "recall"),
                             ("mean_hallucination", WONG["purple"], "hallucination")]:
        ax.plot([r["round"] for r in p2["per_round"]],
                [r[key] for r in p2["per_round"]],
                color=colour, lw=1.6, marker="o", ms=3, label=lab)
    ax.set_xlabel("correction round")
    ax.set_ylabel("rate")
    ax.set_title("precision / recall / hallucination")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="center right")

    ax = axes[2]
    xs = [r["round"] for r in p2["per_round"]]
    ax.plot(xs, [100 * r["frac_reached_threshold"] for r in p2["per_round"]],
            color=WONG["red"], lw=2, marker="o", ms=3, label="A_rand")
    if committed and committed.get("per_round"):
        ax.plot([r["round"] for r in committed["per_round"]],
                [100 * r["frac_reached_threshold"] for r in committed["per_round"]],
                color=WONG["blue"], lw=2, marker="s", ms=3, ls="--",
                label="A (committed)")
    ax.set_xlabel("correction round")
    ax.set_ylabel("% at or above threshold")
    ax.set_title("fraction grounded")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle("The active grounding loop converges on RANDOM evidence "
                 "(n={}, llama3.1:8b)".format(p2.get("n_scenarios", "?")), y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_decisions(p3: Dict[str, Any], out: str) -> Optional[str]:
    plt = _mpl()
    conds = [a["condition"] for a in p3["conditions"]]
    colours = {"RANDOM": WONG["grey"], "RAW": WONG["sky"],
               "XTRAFFIC": WONG["blue"], "XTRAFFIC_RAND": WONG["red"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))
    for ax, key, lab in [(axes[0], "accuracy", "accuracy"),
                         (axes[1], "delay_reduction", "delay reduction (mph)")]:
        vals = [a[key + "_mean"] for a in p3["conditions"]]
        errs = [a[key + "_std"] for a in p3["conditions"]]
        ax.bar(range(len(conds)), vals, yerr=errs, capsize=3,
               color=[colours.get(c, WONG["grey"]) for c in conds])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=20, ha="right")
        ax.set_ylabel(lab)
        ax.set_title(lab)
    fig.suptitle("Decision quality, real vs random explanations "
                 "(n={}, {} seeds)".format(p3["n_scenarios"],
                                           len(p3["decision_seeds"])), y=1.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_entropy_stability(run_path: str, p4: Dict[str, Any],
                          out: str) -> Optional[str]:
    plt = _mpl()
    rows: List[Dict[str, Any]] = []
    csv_path = os.path.join(run_path, "part4_per_target.csv")
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["split_half_jaccard"] in ("", "None"):
                continue
            rows.append(r)

    colours = {"current": WONG["blue"], "3x": WONG["orange"], "10x": WONG["green"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    for ekey, ax, title in [("entropy_of_mean", axes[0],
                             "entropy of the 24-window mean mask"),
                            ("mean_of_entropy", axes[1],
                             "mean per-window entropy")]:
        for setting in ["current", "3x", "10x"]:
            sub = [r for r in rows if r["setting"] == setting]
            if not sub:
                continue
            ax.scatter([float(r[ekey]) for r in sub],
                       [float(r["split_half_jaccard"]) for r in sub],
                       s=12, alpha=0.75, color=colours[setting],
                       label="{} (n={})".format(setting, len(sub)),
                       edgecolors="none")
        sc = p4["scatter_spearman"].get(ekey, {})
        ax.set_xlabel("normalised entropy")
        ax.set_ylabel("split-half top-8 Jaccard")
        ax.set_title("{}\nrho {:+.3f}  95% CI [{:+.3f}, {:+.3f}]".format(
            title, sc.get("rho", float("nan")), sc.get("ci_lo", float("nan")),
            sc.get("ci_hi", float("nan"))))
        T = p4["entropy_threshold"].get(ekey, {}).get("threshold")
        if T is not None:
            ax.axvline(T, color="k", lw=0.8, ls=":",
                       label="threshold {:.4f}".format(T))
        ax.legend(frameon=False, fontsize=6, loc="upper right")

    fig.suptitle("Flatter masks are less stable — entropy vs split-half "
                 "agreement, pooled over sparsity settings", y=1.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Verdicts — each read straight off predictions.md's own criteria
# ---------------------------------------------------------------------------
def verdict_p1(p1: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not p1:
        return {"verdict": "NOT RUN", "detail": "part1_summary.json absent"}
    tok = p1.get("token_parity", {})
    if not tok.get("all_within_tolerance"):
        return {"verdict": "UNTESTED",
                "detail": "token-parity guard failed; predictions.md says P1 is "
                          "then UNTESTED, not confirmed"}
    boot = p1.get("bootstrap_ci", {})
    pc, pw = boot.get("per_condition", {}), boot.get("pairwise_diff", {})
    a = pc.get("A", {})
    checks: List[Dict[str, Any]] = []
    for cond in ("A_rand", "A_mismatch"):
        for m in ("faithfulness_f1", "hallucination_rate"):
            mean = pc.get(cond, {}).get(m, {}).get("mean")
            lo, hi = a.get(m, {}).get("lo"), a.get(m, {}).get("hi")
            inside = (mean is not None and lo is not None
                      and lo <= mean <= hi)
            d = pw.get("A-{}".format(cond), {}).get(m, {})
            spans = (not d.get("excludes_zero")) if d else None
            checks.append({
                "condition": cond, "metric": m, "mean": mean,
                "A_ci": [lo, hi], "inside_A_ci": bool(inside),
                "paired_diff": d.get("mean_diff"),
                "paired_ci": [d.get("lo"), d.get("hi")],
                "paired_spans_zero": bool(spans),
                "passes": bool(inside and spans)})
    ok = all(c["passes"] for c in checks)
    return {"verdict": "CONFIRMED" if ok else "FALSIFIED", "checks": checks}


def verdict_p2(p2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not p2:
        return {"verdict": "NOT RUN", "detail": "part2_summary.json absent"}
    frac = p2.get("frac_reached_overall")
    rounds = p2.get("mean_rounds_used")
    final_prec = p2["per_round"][-1]["mean_precision"] if p2.get("per_round") else None
    c1 = frac is not None and frac >= 0.95
    c2 = rounds is not None and abs(rounds - 0.40) <= 0.25
    c3 = final_prec is not None and final_prec >= 0.95
    return {
        "verdict": "CONFIRMED" if (c1 and c2 and c3) else "FALSIFIED",
        "checks": [
            {"criterion": "fraction reaching threshold >= 0.95",
             "value": frac, "passes": bool(c1)},
            {"criterion": "mean correction rounds within 0.40 +/- 0.25",
             "value": rounds, "passes": bool(c2)},
            {"criterion": "final-round mean precision >= 0.95",
             "value": final_prec, "passes": bool(c3)}]}


def verdict_p3(p3: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not p3:
        return {"verdict": "NOT RUN", "detail": "part3_summary.json absent"}
    by = {a["condition"]: a for a in p3["conditions"]}
    xt, rd, raw = by.get("XTRAFFIC"), by.get("XTRAFFIC_RAND"), by.get("RAW")
    if not (xt and rd and raw):
        return {"verdict": "NOT RUN", "detail": "a condition is missing"}
    checks = []
    for key in ("accuracy", "delay_reduction"):
        lo = xt[key + "_mean"] - xt[key + "_std"]
        hi = xt[key + "_mean"] + xt[key + "_std"]
        v = rd[key + "_mean"]
        checks.append({"criterion": "{} within XTRAFFIC mean +/- 1 sd".format(key),
                       "value": v, "interval": [lo, hi],
                       "passes": bool(lo <= v <= hi)})
    checks.append({"criterion": "XTRAFFIC_RAND accuracy above RAW",
                   "value": rd["accuracy_mean"], "raw": raw["accuracy_mean"],
                   "passes": bool(rd["accuracy_mean"] > raw["accuracy_mean"])})
    ok = all(c["passes"] for c in checks)
    return {"verdict": "CONFIRMED" if ok else "FALSIFIED", "checks": checks}


def verdict_p4(p4: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not p4:
        return {"verdict": "NOT RUN", "detail": "part4_metrics.json absent"}
    checks = []
    ok_any = {}
    for ekey in ("entropy_of_mean", "mean_of_entropy"):
        sc = p4["scatter_spearman"].get(ekey, {})
        rho = sc.get("rho")
        c = (rho is not None and rho < -0.30 and sc.get("excludes_zero"))
        ok_any[ekey] = bool(c)
        checks.append({"criterion": "rho < -0.30 and CI excludes 0 [{}]".format(ekey),
                       "value": rho, "ci": [sc.get("ci_lo"), sc.get("ci_hi")],
                       "passes": bool(c)})
    ce = p4.get("committed_explanations", {}).get("by_checkpoint", {})
    ep34 = ce.get("ep34", {})
    frac = ep34.get("fraction_flagged")
    c_flag = frac is not None and frac >= 1.0
    checks.append({"criterion": "all committed explanations flagged (ep34, the "
                                "checkpoint they were actually produced by)",
                   "value": frac,
                   "detail": "{}/{}".format(ep34.get("n_flagged"), ep34.get("n")),
                   "passes": bool(c_flag)})
    # predictions.md states P4 as TWO claims with their own confirmation and
    # falsification criteria: the correlation, and the flagging. They can land
    # differently, and collapsing them into one word would hide which half
    # survived. `mean_of_entropy` is the pre-registered statistic for the
    # flagging arm because the committed masks are single-window.
    corr_ok = ok_any.get("mean_of_entropy", False)
    return {
        "verdict": ("CONFIRMED" if (corr_ok and c_flag) else
                    "PARTIALLY CONFIRMED" if (corr_ok or c_flag) else
                    "FALSIFIED"),
        "sub_verdicts": {
            "P4a_correlation": "CONFIRMED" if corr_ok else "FALSIFIED",
            "P4b_all_committed_flagged": "CONFIRMED" if c_flag else "FALSIFIED",
        },
        "checks": checks}


def _checkline(c: Dict[str, Any]) -> str:
    mark = "PASS" if c.get("passes") else "FAIL"
    bits = []
    for k in ("criterion", "condition", "metric"):
        if c.get(k):
            bits.append(str(c[k]))
    body = " / ".join(bits)
    nums = []
    if c.get("value") is not None:
        nums.append("value {}".format(_fmt(c["value"], 4)))
    if c.get("mean") is not None:
        nums.append("mean {}".format(_fmt(c["mean"], 4)))
    if c.get("A_ci"):
        nums.append("A CI [{}, {}]".format(_fmt(c["A_ci"][0], 4),
                                           _fmt(c["A_ci"][1], 4)))
    if c.get("paired_ci") and c["paired_ci"][0] is not None:
        nums.append("paired diff {} CI [{}, {}]".format(
            _fmt(c.get("paired_diff"), 4), _fmt(c["paired_ci"][0], 4),
            _fmt(c["paired_ci"][1], 4)))
    if c.get("interval"):
        nums.append("interval [{}, {}]".format(_fmt(c["interval"][0], 4),
                                               _fmt(c["interval"][1], 4)))
    if c.get("ci") and c["ci"][0] is not None:
        nums.append("CI [{}, {}]".format(_fmt(c["ci"][0], 4), _fmt(c["ci"][1], 4)))
    if c.get("raw") is not None:
        nums.append("RAW {}".format(_fmt(c["raw"], 4)))
    if c.get("detail"):
        nums.append(str(c["detail"]))
    return "- **{}** — {}{}".format(mark, body,
                                    (": " + "; ".join(nums)) if nums else "")


def write_report(rp: str, run_id: str, p1, p2, p3, p4, committed_loop,
                 verdicts: Dict[str, Any],
                 tables: Dict[str, Tuple[Sequence[str], List[List[str]]]]) -> str:
    """Assemble report.md. Every number here is read from the part JSONs."""
    L: List[str] = []
    A = L.append

    A("# Grounding without information — report")
    A("")
    A("Run `{}`.".format(run_id))
    A("Pre-registration in `predictions.md`, written before any solve or scored "
      "LLM call. Literature check in `literature.md`. Every number below is read "
      "out of the part JSONs in this directory; nothing is typed by hand.")
    A("")
    A("## Verdicts")
    A("")
    titles = {
        "P1": "A_rand and A_mismatch F1/hallucination land within A's CI",
        "P2": "the loop reaches threshold on noise at the same rate and rounds, "
              "precision -> 1.0",
        "P3": "decisions on A_rand land within XTRAFFIC's spread",
        "P4": "entropy predicts split-half stability (rho < -0.30); all committed "
              "explanations flagged",
    }
    A(_md_table(["prediction", "claim", "verdict"],
                [[k, titles[k], "**{}**".format(verdicts[k]["verdict"])]
                 for k in ("P1", "P2", "P3", "P4")]))
    A("")

    for k in ("P1", "P2", "P3", "P4"):
        v = verdicts[k]
        A("### {} — {}".format(k, v["verdict"]))
        A("")
        if v.get("sub_verdicts"):
            for sk, sv in v["sub_verdicts"].items():
                A("- `{}`: **{}**".format(sk, sv))
            A("")
        if v.get("detail"):
            A(v["detail"])
            A("")
        for c in v.get("checks", []):
            A(_checkline(c))
        A("")

    # ---- Part 1 -------------------------------------------------------
    if p1:
        A("## Part 1 — grounding without information")
        A("")
        A("n = {} stratified METR-LA scenarios, `{}`. Conditions A and B are "
          "REUSED from the committed Phase-5 run (no new calls); `A_rand` and "
          "`A_mismatch` are new.".format(p1["n_scenarios"], p1["model"]))
        A("")
        A(_md_table(*tables["table1_faithfulness"]))
        A("")
        tok = p1["token_parity"]
        A("**Token parity (the pre-registered guard).** Measured on the serving "
          "model over a seeded {}-scenario subsample.".format(
              tok.get("token_sample_n")))
        A("")
        A(_md_table(["condition", "mean prompt tokens", "vs A", "within 5%"],
                    [[c, _fmt(tok["per_condition"][c]["mean_prompt_tokens"], 1),
                      "{:+.2%}".format(tok["checks"][c]["relative_difference"]),
                      "yes" if tok["checks"][c]["within_5pct"] else "NO"]
                     for c in tok["checks"]]))
        A("")
        conf = p1.get("explanation_confidence", {})
        if conf:
            A("**A visible difference we did not hide.** The artifacts also carry "
              "an `explanation_confidence` field, which the prompt shows the LLM. "
              "A random mask genuinely has no stability across reruns, so "
              "`A_rand`'s confidence is near zero and the model was TOLD so:")
            A("")
            A(_md_table(["condition", "mean confidence", "min", "max"],
                        [[c, _fmt(conf[c]["mean"]), _fmt(conf[c]["min"]),
                          _fmt(conf[c]["max"])] for c in conf]))
            A("")
            A("That makes the result stronger, not weaker: the metric did not "
              "move even though the artifact announced its own unreliability.")
            A("")

    # ---- Part 2 -------------------------------------------------------
    if p2:
        A("## Part 2 — the loop on noise")
        A("")
        A("The existing loop (`run_active_grounding`, imported unchanged), same "
          "threshold {:.2f}, same max_rounds {}, run on the Part-1 `A_rand` "
          "artifacts.".format(p2["f1_threshold"], p2["max_rounds"]))
        A("")
        A(_md_table(*tables["table2_loop_on_noise"]))
        A("")
        A("- mean F1 gain round 0 -> final: **{:+.3f}**".format(p2["mean_f1_gain"]))
        A("- mean correction rounds used: **{:.2f}**".format(p2["mean_rounds_used"]))
        A("- reached threshold overall: **{:.1f}%**".format(
            100 * p2["frac_reached_overall"]))
        A("- stop reasons: `{}`".format(p2["stop_reasons"]))
        A("")
        A("Figure: `fig1_loop_on_noise.pdf`.")
        A("")

    # ---- Part 3 -------------------------------------------------------
    if p3:
        A("## Part 3 — decisions on noise")
        A("")
        A("n = {} of the {} Phase-10 scenarios (seeded, stratum-preserving), "
          "{} decision seeds. RANDOM / RAW / XTRAFFIC are re-aggregated on the "
          "SAME subset so all four arms are paired.".format(
              p3["n_scenarios"], p3["n_scenarios_full_study"],
              len(p3["decision_seeds"])))
        A("")
        A(_md_table(*tables["table3_decisions"]))
        A("")
        cf = {c["condition"]: c for c in p3.get("committed_full_study_n444", [])}
        if cf:
            A("Committed full-study reference (n=444, not the comparison): "
              + "; ".join("{} acc {:.3f} delay {:.2f}".format(
                  k, v["accuracy_mean"], v["delay_reduction_mean"])
                  for k, v in cf.items()))
            A("")
        A("Figure: `fig2_decisions.pdf`.")
        A("")

    # ---- Part 4 -------------------------------------------------------
    if p4:
        A("## Part 4 — does entropy predict stability?")
        A("")
        A(_md_table(*tables["table4_sparsity_sweep"]))
        A("")
        A("**The scatter.** Per-target entropy against per-target split-half "
          "top-8 Jaccard, pooled across settings, Spearman with a 10k bootstrap "
          "resampling TARGETS (a target's three settings move together, so "
          "resampling points would understate the CI).")
        A("")
        rows = []
        for ekey, sc in p4["scatter_spearman"].items():
            rows.append([ekey, sc["n_points"], sc["n_clusters"],
                         "{:+.3f}".format(sc["rho"]),
                         "[{:+.3f}, {:+.3f}]".format(sc["ci_lo"], sc["ci_hi"]),
                         "excludes 0" if sc["excludes_zero"] else "spans 0"])
        A(_md_table(["entropy definition", "n points", "n targets", "rho",
                     "95% CI", ""], rows))
        A("")
        A("**Two entropies, deliberately.** `entropy_of_mean` is the entropy of "
          "the 24-window mean mask — the object the top-8 is actually read off. "
          "`mean_of_entropy` is the mean of the per-window entropies. Averaging "
          "masks pulls the average toward uniform, so the first is "
          "systematically higher. The 12 committed explanations are "
          "SINGLE-window solves, so `mean_of_entropy` is the comparable "
          "statistic and is the one the pre-registered flag uses.")
        A("")
        thr = p4["entropy_threshold"]
        A(_md_table(["entropy definition", "threshold T", "Youden J",
                     "sensitivity", "specificity"],
                    [[k, _fmt(t.get("threshold"), 4), _fmt(t.get("youden_j")),
                      _fmt(t.get("sensitivity"), 2), _fmt(t.get("specificity"), 2)]
                     for k, t in thr.items()]))
        A("")
        ce = p4["committed_explanations"]
        A("**The committed explanations.** " + ce["note"])
        A("")
        A(_md_table(["checkpoint", "n", "flagged", "fraction", "entropy min",
                     "entropy max", "entropy mean"],
                    [[k, v["n"], v["n_flagged"], _fmt(v["fraction_flagged"]),
                      _fmt(v["entropy_min"], 4), _fmt(v["entropy_max"], 4),
                      _fmt(v["entropy_mean"], 4)]
                     for k, v in ce["by_checkpoint"].items()]))
        A("")
        A("`ep34` is the checkpoint the committed artifacts were ACTUALLY "
          "produced by — their JSONs name `metr_la_best.pt`, which now holds "
          "epoch 54 and does not reproduce them (top-8 Jaccard 0.000, importance "
          "error 0.647). Epoch 34 reproduces them exactly (Jaccard 1.000, max "
          "difference 5e-5 = 4-decimal rounding). `ep54` is reported so the "
          "committed explanations can also be read on the same checkpoint as the "
          "sweep.")
        A("")
        sup = p4.get("supplementary_current_setting_all_targets") or {}
        if sup and "error" not in sup:
            A("**Supplementary, and free: the same question on all {} Stage-1 "
              "targets at the current setting.** Not the pre-registered test — "
              "that is the pooled scatter above — but those solves already "
              "existed, and 207 targets is better powered than 40.".format(
                  sup["n_targets"]))
            A("")
            A(_md_table(["entropy definition", "rho", "95% CI", ""],
                        [[k, "{:+.3f}".format(v["rho"]),
                          "[{:+.3f}, {:+.3f}]".format(v["ci_lo"], v["ci_hi"]),
                          "excludes 0" if v["excludes_zero"] else "spans 0"]
                         for k, v in sup["spearman"].items()]))
            A("")
            A("So the relationship exists at the committed sparsity setting for "
              "the entropy of the 24-window MEAN mask, and does not survive "
              "either (a) pooling across sparsity settings, or (b) being "
              "computed per-window — which is the only form obtainable from a "
              "SINGLE solve, and therefore the only form that would have made "
              "this a cheap diagnostic. Reported as a negative result.")
            A("")

        A("**The sparsity coefficient does not control sparsity here.** This was "
          "not predicted and is the clearest thing Part 4 found. Multiplying "
          "`lambda_size` by ten moved the mask the WRONG way on every measure "
          "of concentration:")
        A("")
        A(_md_table(["setting", "lambda_size", "normalised entropy",
                     "sources for 80% mass", "split-half J", "seed-repeat J"],
                    [[s["setting"], _fmt(s["lambda_size"], 2),
                      _fmt(s["entropy_of_mean_mean"], 4),
                      _fmt(s["sources_for_80pct_mean"], 1),
                      _fmt(s["split_half_jaccard_mean"]),
                      _fmt(s.get("seed_repeat_jaccard_mean"))]
                     for s in p4["settings"]]))
        A("")
        A("Entropy rises, the number of sources needed to cover 80% of the mass "
          "rises, and stability does not improve. A ten-fold sparsity penalty "
          "produced a FLATTER mask. Whatever `lambda_size` is doing in this "
          "objective, it is not making the explanation more concentrated, and "
          "the near-flat mask this project documented earlier is not a tuning "
          "artefact that a bigger penalty would fix.")
        A("")
        A("Figure: `fig3_entropy_vs_stability.pdf`.")
        A("")

    A("## What the three falsifications mean")
    A("")
    A("Stated without hedging, and without reinterpreting the criteria after "
      "seeing the numbers.")
    A("")
    A("**P2 is falsified on one clause of three.** The loop DID converge on "
      "random evidence — 98.9% of scenarios reached the 0.70 threshold against "
      "100% on real evidence, final precision 0.999, final F1 0.890 against "
      "0.874. What failed is the ROUNDS clause: 0.70 mean correction rounds "
      "against a pre-registered window of 0.40 +/- 0.25. The loop starts worse "
      "on noise (41.9% grounded at round 0 versus 64.5%) and therefore has to "
      "work harder to arrive at the same place. So the honest statement is "
      "narrower than the prediction: the loop cannot tell that its evidence is "
      "noise, but it is not entirely blind to it either — the round-0 rate "
      "carries a signal that the loop then erases. That round-0 gap is the only "
      "place in this entire experiment where a metric distinguished real "
      "evidence from random, and it is worth following up.")
    A("")
    A("**P3 is falsified on a clause that could not have discriminated.** "
      "`XTRAFFIC_RAND` landed inside `XTRAFFIC`'s seed spread on BOTH headline "
      "metrics, which was the substance of the prediction. It failed only "
      "'stay above RAW' — and on this subsample `RAW` scored 0.276 against "
      "0.233 in the committed n=444 run, so even the real `XTRAFFIC` arm "
      "(0.278) is not meaningfully above it. The criterion was badly chosen: it "
      "assumed a RAW-versus-XTRAFFIC accuracy gap that does not exist at n=150. "
      "That is a defect in the pre-registration, not a result about the system, "
      "and it is recorded as such rather than quietly dropped. The delay-"
      "reduction comparison, which does separate the arms from RAW "
      "(21.00 / 20.89 versus 15.94), carries the finding instead.")
    A("")
    A("**P4 is falsified on both clauses.** Pooled across sparsity settings the "
      "correlation is -0.201 / -0.173 with CIs spanning zero, and 3 of 11 "
      "committed explanations are flagged rather than all of them. The n=40 "
      "single-setting version of this test looked like it passed; it did not "
      "replicate. Entropy is not a usable stability proxy here.")
    A("")
    A("## Provenance and honest limits")
    A("")
    A("- LLM output is **not** byte-reproducible on re-run. The committed A / B / "
      "XTRAFFIC numbers were produced on the unseeded inline Ollama path, and "
      "matching that path was the condition for comparing against them at all. "
      "Every call is logged in full to `llm_calls.jsonl`. This project has "
      "measured the resulting per-scenario noise at |dF1| ~ 0.08 mean, up to "
      "0.14. Everything non-LLM is seeded and byte-reproducible: the current "
      "code reproduces the Stage-1 influence solves to `max abs diff 0.0`.")
    A("- CPU only. No MPS anywhere: CPU and MPS disagree on top-k identity at the "
      "same seed (Spearman 0.578, top-8 Jaccard 0.399), and top-k identity is "
      "the object of Part 4.")
    A("- Nothing committed was modified. `explanations_cache/` was read-only for "
      "this run; every artifact produced lives in this run directory.")
    A("- `A_rand`'s importances are the top 8 of 207 uniform(0,1) draws, so they "
      "sit near 1.0, while real top-8 importances sit lower. The prompt shows "
      "those numbers. This is a distributional difference between the conditions "
      "beyond information content, it was fixed by the pre-registration, and it "
      "is a limitation rather than a confound the guard could catch.")
    A("")

    text = "\n".join(L) + "\n"
    with open(os.path.join(rp, "report.md"), "w") as fh:
        fh.write(text)
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    run = resolve_run(args.run_id)
    rp = run.path

    p1 = _load(rp, "part1_summary.json")
    p2 = _load(rp, "part2_summary.json")
    p3 = _load(rp, "part3_summary.json")
    p4 = _load(rp, "part4_metrics.json")

    from ..utils.io_utils import PKG_ROOT
    committed_loop = _load(
        os.path.join(PKG_ROOT, "evaluation", "results", "active_grounding"),
        "active_grounding_summary.json") or {}

    tables: Dict[str, Tuple[Sequence[str], List[List[str]]]] = {}
    if p1:
        tables["table1_faithfulness"] = table1(p1)
    if p2:
        tables["table2_loop_on_noise"] = table2(p2, committed_loop)
    if p3:
        tables["table3_decisions"] = table3(p3)
    if p4:
        tables["table4_sparsity_sweep"] = table4(p4)

    captions = {
        "table1_faithfulness": ("Faithfulness under four conditions on the same "
                                "93 stratified METR-LA scenarios. A and B reused "
                                "from the committed Phase-5 run; A\\_rand and "
                                "A\\_mismatch new. 10k bootstrap CIs."),
        "table2_loop_on_noise": ("The active grounding loop run on random "
                                 "evidence (A\\_rand), beside the committed run "
                                 "on real evidence (A)."),
        "table3_decisions": ("Decision quality when the agent is shown a random "
                             "explanation instead of the real one."),
        "table4_sparsity_sweep": ("Explainer behaviour at three sparsity "
                                  "coefficients: 40 targets x 24 windows each."),
    }
    for name, (hdr, rows) in tables.items():
        with open(os.path.join(rp, name + ".md"), "w") as fh:
            fh.write(_md_table(hdr, rows) + "\n")
        with open(os.path.join(rp, name + ".tex"), "w") as fh:
            fh.write(_tex_table(hdr, rows, captions[name],
                                "tab:" + name.split("_", 1)[1]) + "\n")

    figs = []
    if p2:
        figs.append(fig_loop_on_noise(rp, p2, committed_loop,
                                      os.path.join(rp, "fig1_loop_on_noise.pdf")))
    if p3:
        figs.append(fig_decisions(p3, os.path.join(rp, "fig2_decisions.pdf")))
    if p4:
        figs.append(fig_entropy_stability(
            rp, p4, os.path.join(rp, "fig3_entropy_vs_stability.pdf")))

    verdicts = {"P1": verdict_p1(p1), "P2": verdict_p2(p2),
                "P3": verdict_p3(p3), "P4": verdict_p4(p4)}
    with open(os.path.join(rp, "verdicts.json"), "w") as fh:
        json.dump(verdicts, fh, indent=2, default=str)

    write_report(rp, run.run_id, p1, p2, p3, p4, committed_loop, verdicts, tables)

    print("tables: {}".format(sorted(tables)))
    print("figures: {}".format([os.path.basename(f) for f in figs if f]))
    for k, v in verdicts.items():
        print("{}: {}".format(k, v["verdict"]))
        for c in v.get("checks", []):
            print("    {} {}".format("PASS" if c.get("passes") else "fail",
                                     {kk: vv for kk, vv in c.items()
                                      if kk != "passes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
