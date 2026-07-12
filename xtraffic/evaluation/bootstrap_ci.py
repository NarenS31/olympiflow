"""Phase 15b — nonparametric bootstrap 95% confidence intervals for the
faithfulness study.

WHY THIS EXISTS (reviewer feedback)
-----------------------------------
The Phase-5 study reports mean +/- std per condition. A reviewer's fair pushback:
"mean A > mean B could be sampling noise — show me the uncertainty." This module
turns "A beats B" into "A beats B, and the 95% CI on the GAP excludes 0", which is
the standard bar for a real effect. It adds NO new LLM calls: it resamples the
per-scenario metrics we already logged (evaluation/results/faithfulness/
faithfulness_per_scenario*.csv), so it is instant and fully reproducible.

WHAT A BOOTSTRAP CI IS (plain language — you will defend this)
-------------------------------------------------------------
We have N scenarios, each scored under several conditions. We don't know the TRUE
population mean of, say, faithfulness F1 under condition A — we only have our N
samples. The bootstrap estimates the uncertainty by PRETENDING our sample IS the
population: draw N scenarios WITH REPLACEMENT, recompute the mean, repeat B times
(~10k). The spread of those B means approximates the sampling distribution of the
mean; its 2.5th and 97.5th percentiles are a 95% confidence interval. It is
NONPARAMETRIC — it assumes no bell curve — which is exactly right for bounded,
skewed metrics like a hallucination rate that piles up at 0.

WHY PAIRED RESAMPLING FOR THE A-B / A-C DIFFERENCES
---------------------------------------------------
The study is WITHIN-SCENARIO: the same scenario is scored under A, B, C, ... So to
bound the A-B gap we resample SCENARIOS (not the rows independently) and, for each
drawn scenario, take its A and B value TOGETHER, then average the per-scenario
DIFFERENCES. This respects the pairing — a scenario that is easy (or hard) for both
conditions should not inflate the gap's uncertainty — and is the correct paired
bootstrap. Resampling A-rows and B-rows independently would overstate the variance.

RUN
---
  python -m xtraffic.evaluation.bootstrap_ci                 # on the committed CSV
  python -m xtraffic.evaluation.bootstrap_ci --limit 3       # smoke (machinery only)
  python -m xtraffic.evaluation.bootstrap_ci --csv <path> --out-tag 15b

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.io_utils import PKG_ROOT

SEED = 42               # CLAUDE.md: fixed seed everywhere.
N_ITERS = 10000         # bootstrap resamples; 10k is plenty for a stable 95% CI.
CI = 0.95

# The metrics we bound. Same keys the study logs per scenario.
METRICS = ["cause_precision", "cause_recall", "faithfulness_f1",
           "hallucination_rate", "quantitative_fidelity"]

# A scenario is identified by (sample_index, target_node) — its rows across
# conditions share this key, which is what makes the difference bootstrap paired.
_KEY_COLS = ("sample_index", "target_node")


# ---------------------------------------------------------------------------
# Loading per-scenario rows (from the study's CSV, or reuse in-memory rows).
# ---------------------------------------------------------------------------
def _to_float(cell: Any) -> Optional[float]:
    """CSV cells: '' / 'nan' / 'None' mean 'no value' (e.g. quantitative_fidelity
    is None when the LLM stated no numbers). Everything else parses to float."""
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Read the study's per-scenario CSV into the same row dicts the study builds
    in memory, so both entry points share one bootstrap implementation."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = {
                "sample_index": r.get("sample_index"),
                "target_node": r.get("target_node"),
                "condition": r.get("condition"),
            }
            for m in METRICS:
                if m in r:
                    row[m] = _to_float(r[m])
            rows.append(row)
    return rows


def _scenario_key(row: Dict[str, Any]) -> Tuple[Any, Any]:
    return (str(row.get("sample_index")), str(row.get("target_node")))


def _index_by_scenario(rows: List[Dict[str, Any]]
                       ) -> "Dict[Tuple[Any, Any], Dict[str, Dict[str, Any]]]":
    """key -> {condition -> row}. This is the paired table the bootstrap draws on:
    one row per condition per scenario."""
    table: Dict[Tuple[Any, Any], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        table.setdefault(_scenario_key(row), {})[str(row["condition"])] = row
    return table


# ---------------------------------------------------------------------------
# The bootstrap itself (vectorised over iterations for speed).
# ---------------------------------------------------------------------------
def _mean_ci(values: Sequence[float], rng: "np.random.Generator"
             ) -> Dict[str, Any]:
    """95% bootstrap CI on the MEAN of `values` (one condition, one metric).
    Resample the values with replacement N_ITERS times; take percentiles."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    idx = rng.integers(0, n, size=(N_ITERS, n))          # [B, n] resample indices
    boot_means = arr[idx].mean(axis=1)                   # [B] one mean per resample
    lo, hi = np.percentile(boot_means, [100 * (1 - CI) / 2, 100 * (1 + CI) / 2])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": n}


def _paired_diff_ci(diffs: Sequence[float], rng: "np.random.Generator"
                    ) -> Dict[str, Any]:
    """95% bootstrap CI on the MEAN PAIRED DIFFERENCE. `diffs[i]` is one scenario's
    (cond_a - cond_b) value; we resample SCENARIOS so the pairing is preserved.
    `excludes_zero` True => the gap is significant at the 95% level."""
    arr = np.asarray([float(v) for v in diffs], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean_diff": None, "lo": None, "hi": None,
                "excludes_zero": False, "n": 0}
    idx = rng.integers(0, n, size=(N_ITERS, n))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * (1 - CI) / 2, 100 * (1 + CI) / 2])
    return {"mean_diff": float(arr.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": n}


def _default_pairs(conditions: List[str]) -> List[Tuple[str, str]]:
    """Which condition pairs to bound. Always A vs every other present condition
    (A is the full pipeline, the reference). Plus (C, C_RICH) if both present, to
    directly test whether a FULLER context block moves faithfulness."""
    pairs: List[Tuple[str, str]] = []
    if "A" in conditions:
        for c in conditions:
            if c != "A":
                pairs.append(("A", c))
    if "C" in conditions and "C_RICH" in conditions:
        pairs.append(("C", "C_RICH"))
    return pairs


def bootstrap_from_rows(rows: List[Dict[str, Any]],
                        conditions: Optional[List[str]] = None,
                        metrics: Optional[List[str]] = None,
                        pairs: Optional[List[Tuple[str, str]]] = None,
                        seed: int = SEED) -> Dict[str, Any]:
    """Compute per-condition mean CIs and paired-difference CIs from the study's
    per-scenario rows. Returns a JSON-able block that both the CLI and the study
    embed. One RNG, seeded once, walked in a fixed order => reproducible."""
    table = _index_by_scenario(rows)
    present = []
    for row in rows:                                     # preserve first-seen order
        c = str(row["condition"])
        if c not in present:
            present.append(c)
    conditions = conditions or present
    metrics = metrics or METRICS
    pairs = pairs if pairs is not None else _default_pairs(conditions)

    rng = np.random.default_rng(seed)

    per_condition: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        per_condition[cond] = {}
        for m in metrics:
            vals = [table[k][cond][m] for k in table
                    if cond in table[k] and table[k][cond].get(m) is not None]
            per_condition[cond][m] = _mean_ci(vals, rng)

    pairwise: Dict[str, Dict[str, Any]] = {}
    for a, b in pairs:
        label = "{}-{}".format(a, b)
        pairwise[label] = {}
        for m in metrics:
            # Paired: only scenarios that have BOTH conditions with a value for m.
            diffs = [table[k][a][m] - table[k][b][m] for k in table
                     if a in table[k] and b in table[k]
                     and table[k][a].get(m) is not None
                     and table[k][b].get(m) is not None]
            pairwise[label][m] = _paired_diff_ci(diffs, rng)

    return {
        "n_iters": N_ITERS,
        "seed": seed,
        "ci_level": CI,
        "n_scenarios": len(table),
        "conditions": conditions,
        "metrics": metrics,
        "per_condition": per_condition,
        "pairwise_diff": pairwise,
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _fmt(ci: Dict[str, Any]) -> str:
    if ci.get("mean") is None:
        return "n/a"
    return "{:.3f} [{:.3f}, {:.3f}]".format(ci["mean"], ci["lo"], ci["hi"])


_METRIC_LABELS = {
    "cause_precision": "Precision",
    "cause_recall": "Recall",
    "faithfulness_f1": "F1",
    "hallucination_rate": "Halluc.",
    "quantitative_fidelity": "Quant.\\,fid.",
}


def write_latex(block: Dict[str, Any], path: str) -> None:
    """Emit a booktabs LaTeX table: one row per metric, one column per condition,
    each cell = mean [lo, hi] (the 95% bootstrap CI). This is the paper artifact
    that adds the CI columns the reviewer asked for. \\input-ready."""
    conds = block["conditions"]
    lines: List[str] = []
    lines.append("% Phase 15b — faithfulness with bootstrap 95% CIs "
                 "({} scenarios, {} iters, seed {}).".format(
                     block["n_scenarios"], block["n_iters"], block["seed"]))
    lines.append("\\begin{tabular}{l" + "c" * len(conds) + "}")
    lines.append("\\toprule")
    lines.append("Metric & " + " & ".join(conds) + " \\\\")
    lines.append("\\midrule")
    for m in block["metrics"]:
        cells = []
        for c in conds:
            ci = block["per_condition"][c][m]
            if ci.get("mean") is None:
                cells.append("--")
            else:
                cells.append("{:.3f} [{:.3f}, {:.3f}]".format(
                    ci["mean"], ci["lo"], ci["hi"]))
        lines.append(_METRIC_LABELS.get(m, m) + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # A second small block: the key paired gaps with CI + significance marker.
    if block["pairwise_diff"]:
        lines.append("")
        lines.append("% Paired-difference 95% CIs (mean diff [lo, hi]; "
                     "* = CI excludes 0).")
        lines.append("\\begin{tabular}{ll" + "c" * len(block["metrics"]) + "}")
        lines.append("\\toprule")
        lines.append("Contrast & & " + " & ".join(
            _METRIC_LABELS.get(m, m) for m in block["metrics"]) + " \\\\")
        lines.append("\\midrule")
        for label, per_metric in block["pairwise_diff"].items():
            cells = []
            for m in block["metrics"]:
                d = per_metric[m]
                if d.get("mean_diff") is None:
                    cells.append("--")
                else:
                    star = "*" if d["excludes_zero"] else ""
                    cells.append("{:+.3f}{}".format(d["mean_diff"], star))
            lines.append(label.replace("_", "\\_") + " & & " + " & ".join(cells)
                         + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def print_report(block: Dict[str, Any]) -> None:
    conds = block["conditions"]
    print("\n=== BOOTSTRAP 95% CI — per condition (n_scenarios={}, {} iters) ==="
          .format(block["n_scenarios"], block["n_iters"]))
    hdr = "{:22s}".format("metric")
    for c in conds:
        hdr += "{:>26s}".format(c)
    print(hdr)
    print("-" * len(hdr))
    for m in block["metrics"]:
        line = "{:22s}".format(m)
        for c in conds:
            line += "{:>26s}".format(_fmt(block["per_condition"][c][m]))
        print(line)

    print("\n=== BOOTSTRAP 95% CI — paired differences (excludes 0 => significant) ===")
    for label, per_metric in block["pairwise_diff"].items():
        print("  {}:".format(label))
        for m in block["metrics"]:
            d = per_metric[m]
            if d.get("mean_diff") is None:
                continue
            star = "  *** excludes 0" if d["excludes_zero"] else "     (spans 0)"
            print("    {:22s} {:+.3f} [{:+.3f}, {:+.3f}]{}".format(
                m, d["mean_diff"], d["lo"], d["hi"], star))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    default_csv = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness",
                               "faithfulness_per_scenario.csv")
    ap.add_argument("--csv", nargs="+", default=[default_csv],
                    help="one or more per-scenario faithfulness CSVs to bootstrap. "
                         "Pass several to UNION them, e.g. the committed A/B/C file "
                         "plus a 15b C_RICH/D_CONTRA file for the combined table.")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N distinct scenarios (SMOKE ONLY — a "
                         "3-scenario CI is meaningless, this just exercises the code)")
    ap.add_argument("--out-tag", default="",
                    help="suffix on the output filename so runs don't clobber")
    args = ap.parse_args()

    rows = []
    for path in args.csv:
        if not os.path.exists(path):
            raise SystemExit("CSV not found: {}\nRun the faithfulness study first."
                             .format(path))
        rows.extend(load_rows_from_csv(path))
    if args.limit:
        keep = []
        seen = []
        for r in rows:
            k = _scenario_key(r)
            if k not in seen:
                if len(seen) >= args.limit:
                    continue
                seen.append(k)
            keep.append(r)
        rows = keep
        print("[smoke] limited to first {} scenarios".format(len(seen)))

    block = bootstrap_from_rows(rows)
    print_report(block)

    tag = ("_" + args.out_tag) if args.out_tag else ""
    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "faithfulness_bootstrap_ci{}.json".format(tag))
    with open(out_path, "w") as f:
        json.dump({"csv": [os.path.relpath(p, PKG_ROOT) for p in args.csv],
                   **block}, f, indent=2)

    # LaTeX table (the paper artifact with CI columns) into evaluation/paper.
    paper_dir = os.path.join(PKG_ROOT, "evaluation", "paper")
    os.makedirs(paper_dir, exist_ok=True)
    tex_path = os.path.join(paper_dir, "table_faithfulness_ci{}.tex".format(tag))
    write_latex(block, tex_path)

    print("\nWrote:", out_path)
    print("Wrote:", tex_path)


if __name__ == "__main__":
    main()
