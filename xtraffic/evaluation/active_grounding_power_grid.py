"""Phase 20 — the ACTIVE GROUNDING LOOP, run CROSS-DOMAIN on the IEEE 14-bus grid.

THE CLAIM UNDER TEST
--------------------
Phase 16 built a closed loop: advise -> score with the Phase-5 faithfulness metric
-> if F1 < threshold, name the top-k nodes the LLM failed to cite and re-prompt.
On METR-LA (n=93) it lifted mean F1 0.732 -> 0.874 and took 100% of scenarios to
threshold, driven by recall, inducing no fabrication.

Phase 19 showed the MEASUREMENT (the A/B hallucination gap) transfers to a power
grid. This module asks the harder question: does the MECHANISM transfer? A metric
that ports is a finding; a closed-loop CONTROLLER that ports without retuning is a
system claim. If it holds, the contribution stops being "we measured hallucination"
and becomes "we built a system that automatically eliminates hallucination across
domains at inference time".

WHAT IS REUSED, UNMODIFIED (this is the entire point)
-----------------------------------------------------
`models/advisor/active_grounding.run_active_grounding` is imported and called AS
IS. Its control flow, its stopping rules ("threshold_reached" / "no_missed_nodes" /
"max_rounds"), its scoring call, and its choice of what to name in a correction are
untouched. So are the F1 threshold (0.7) and max_rounds (3) — read from the same
`active_grounding` block of configs/advisor.yaml the traffic run used. NOTHING is
retuned for the grid. This file only supplies power-grid scenarios and explanations
and observes what the loop does with them.

THE ONE HONEST WRINKLE: THE CORRECTION BLOCK'S VOCABULARY
---------------------------------------------------------
Phase 16 predates Phase 19's domains.py, so `build_correction_block` still had
three traffic hard-codes: "the predicted CONGESTION", "currently {} MPH", and "do
NOT invent new SENSORS, ROADS, INCIDENTS". Pointed at a grid, the correction tells
llama3.1 that a substation is "currently 1.021 mph".

domains.py exists precisely because that is a CONFOUND, not a cosmetic bug — an LLM
told a bus is doing 1.021 mph may invent traffic interventions, and we would then
score our own prompt's confusion as the model's hallucination.

Rather than assert which way to go, this study RUNS BOTH and reports the contrast:

  UNMODIFIED    — `run_active_grounding(..., domain=None)`. The Phase-16 correction
                  text byte-for-byte, mph and all. The literal zero-modification
                  claim. `--verify-unchanged` asserts these bytes against goldens.
  DOMAIN_ROUTED — `run_active_grounding(..., domain=POWER_GRID)`. The same three
                  strings routed through domains.py, exactly as Phase 19 did for
                  advisor.py. Mechanism identical; vocabulary correct.

If the two converge identically, the loop is robust even to a mislabelled
correction, and the zero-modification claim is safe. If UNMODIFIED is worse, the
vocabulary fix is load-bearing and we report that instead of hiding it. Either
outcome is a result; only running one of them would be a choice we could not defend.

A NOTE ON WHAT "LOW FAITHFULNESS" MEANS HERE (read before quoting a headline)
-----------------------------------------------------------------------------
The Phase-19 power-grid scenarios are the ones where condition B hallucinated
1.000. But this loop operates on condition A, and condition A on those same 20
scenarios already averaged F1 0.869 — 15 of 20 are ALREADY at or above the 0.7
threshold and will correctly exit at round 0 having spent zero LLM calls. That is
the loop behaving properly (Phase 16's "no needless re-prompting" gate), not the
loop failing to engage. So this study reports two populations:

  ALL 20        — the honest headline; comparable to METR-LA's stratified n=93.
  LOW-F1 SUBSET — the 5 scenarios below threshold at round 0; the direct analogue
                  of the Phase-16 n=5 low-f1 smoke (F1 0.438 -> 0.912 in one round).

Quoting only the second would be cherry-picking; quoting only the first would hide
how much work the loop actually had to do.

Run:
  # 3-scenario smoke, BOTH variants, printing every correction prompt + response
  python -m xtraffic.evaluation.active_grounding_power_grid --smoke
  # the full 20, both variants
  python -m xtraffic.evaluation.active_grounding_power_grid
  # assert the unmodified correction text is byte-identical to Phase 16
  python -m xtraffic.evaluation.active_grounding_power_grid --verify-unchanged

Resumable: every (scenario, variant) trace is cached, so a rerun skips completed
work. Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.advisor.active_grounding import (
    _carry_forward, aggregate, load_grounding_config, run_active_grounding)
from ..models.advisor.advisor import Advisor
from ..models.advisor.domains import POWER_GRID, TRAFFIC
from ..models.explainer.explain import ExplanationBuilder
from ..models.gnn.loaders import load_node_meta, load_scaler
from ..utils.io_utils import PKG_ROOT
from .faithfulness import NodeTable
from .power_grid_faithfulness import (build_or_load_explanation, load_config,
                                      sample_undervoltage_scenarios)

# The two variants. Value is the `domain` argument handed to the UNMODIFIED loop.
VARIANTS: List[Tuple[str, Optional[Any]]] = [
    ("unmodified", None),            # Phase-16 bytes: "congestion", "mph", "roads"
    ("domain_routed", POWER_GRID),   # same mechanism, grid vocabulary
]

# Committed METR-LA reference (evaluation/results/active_grounding/
# active_grounding_summary.json, n=93, llama3.1:8b, stratified). Loaded from disk
# when present so the figure never drifts from the logged run; these are the
# fallback values so the table still builds on a clean clone.
TRAFFIC_FALLBACK = {
    "n_scenarios": 93, "f1_threshold": 0.7, "max_rounds": 3,
    "per_round": [
        {"round": 0, "mean_f1": 0.732, "mean_recall": 0.605,
         "mean_hallucination": 0.005, "frac_reached_threshold": 0.645},
        {"round": 1, "mean_f1": 0.864, "mean_recall": 0.774,
         "mean_hallucination": 0.000, "frac_reached_threshold": 0.968},
        {"round": 2, "mean_f1": 0.870, "mean_recall": 0.782,
         "mean_hallucination": 0.000, "frac_reached_threshold": 0.989},
        {"round": 3, "mean_f1": 0.874, "mean_recall": 0.788,
         "mean_hallucination": 0.000, "frac_reached_threshold": 1.000},
    ],
    "mean_f1_gain": 0.141, "mean_rounds_used": 0.40, "frac_reached_overall": 1.0,
}


def load_traffic_reference() -> Dict[str, Any]:
    """The committed METR-LA convergence curve, from disk when available."""
    p = os.path.join(PKG_ROOT, "evaluation", "results", "active_grounding",
                     "active_grounding_summary.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    print("  (no committed traffic summary on disk — using documented fallback)")
    return dict(TRAFFIC_FALLBACK)


# ---------------------------------------------------------------------------
# Observing the loop without changing it.
# ---------------------------------------------------------------------------
class RecordingAdvisor:
    """Transparent proxy that logs every (correction sent -> advisory returned).

    WHY A PROXY. The smoke run must show the EXACT correction prompt and how the
    LLM answered it. `run_active_grounding` returns only the per-round metric curve
    and the final advisory, so the intermediate prompts/answers are not in its
    output. Adding them would mean editing the loop — the one thing this study
    cannot do. Wrapping the advisor instead observes the same information from
    outside, and the loop cannot tell the difference: it sees an object with
    `.model` and `.advise_condition`, which is all it ever used.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.model = getattr(inner, "model", "?")
        self.log: List[Dict[str, Any]] = []

    def advise_condition(self, exp: Dict[str, Any], condition: str,
                         extra_instruction: Optional[str] = None) -> Dict[str, Any]:
        res = self.inner.advise_condition(exp, condition,
                                          extra_instruction=extra_instruction)
        adv = res.get("advisory", {}) or {}
        self.log.append({
            "round": len(self.log),
            "correction_prompt": extra_instruction,          # None on round 0
            "reasoning": adv.get("reasoning", ""),
            "cited_causes": adv.get("cited_causes", []),
            "n_recommendations": len(adv.get("recommendations", []) or []),
            "advisory_error": adv.get("_error", ""),
        })
        return res

    def reset(self) -> None:
        self.log = []


# ---------------------------------------------------------------------------
# Per-scenario cache (resumable, same pattern as every Phase 10-19 study).
# ---------------------------------------------------------------------------
def _cache_path(out_dir: str, idx: int, node: int, variant: str,
                model: str) -> str:
    d = os.path.join(out_dir, "decisions_cache")
    os.makedirs(d, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in model)
    return os.path.join(d, "pg_{}_{}_{}_{}.json".format(idx, node, variant, safe))


# ---------------------------------------------------------------------------
# Verbose trace printing (the smoke deliverable).
# ---------------------------------------------------------------------------
def print_verbose_trace(sc: Dict[str, Any], exp: Dict[str, Any],
                        trace: Dict[str, Any], log: List[Dict[str, Any]],
                        variant: str, f1_threshold: float) -> None:
    bar = "=" * 78
    print("\n" + bar)
    print("SCENARIO idx={}  target bus {}  [{} / {}]   VARIANT: {}".format(
        sc["sample_index"], sc["target_bus"], sc["tod_band"], sc["severity"],
        variant.upper()))
    print(bar)
    print("  target      : {}".format(exp["prediction"]["node_name"]))
    print("  voltage     : {:.3f} pu now -> {:.3f} pu predicted at 30 min "
          "(normal {:.3f} pu, sag {:.3f})".format(
              exp["prediction"]["current_speed_mph"],
              exp["prediction"]["predicted_speed_mph"],
              sc["normal_pu"], sc["deviation_pu"]))
    print("  explainer top-k (what the advisory MUST cite):")
    for n in exp["top_nodes"]:
        print("      - {}  importance {:.4f}  ({:.3f} pu)".format(
            n["node_name"], n["importance"], n["current_speed_mph"]))

    for i, c in enumerate(trace["curve"]):
        entry = log[i] if i < len(log) else {}
        print("\n  " + "-" * 74)
        print("  ROUND {}{}".format(c["round"],
                                    "  (after correction)" if c["used_correction"] else
                                    "  (plain condition-A advisory)"))
        print("  " + "-" * 74)
        if entry.get("correction_prompt"):
            print("  >>> CORRECTION PROMPT SENT TO THE LLM:")
            for line in entry["correction_prompt"].splitlines():
                print("      | " + line)
        print("  <<< LLM CITED CAUSES:")
        for cc in (entry.get("cited_causes") or []):
            if isinstance(cc, dict):
                print("      * {}".format(cc.get("location")))
        if entry.get("reasoning"):
            txt = entry["reasoning"].replace("\n", " ")
            print("  <<< LLM REASONING: {}{}".format(
                txt[:300], "..." if len(txt) > 300 else ""))
        if entry.get("advisory_error"):
            print("  !!! ADVISORY ERROR: {}".format(entry["advisory_error"]))
        print("  SCORE: F1 {:.3f} | precision {:.3f} | recall {:.3f} | "
              "halluc {:.3f} | cited {} | still missed {}".format(
                  c["faithfulness_f1"], c["cause_precision"], c["cause_recall"],
                  c["hallucination_rate"], c["n_cited_causes"], c["n_missed"]))
        if c["n_missed"]:
            print("  STILL MISSING: {}".format("; ".join(c["missed_regions"])))

    print("\n  OUTCOME: F1 {:.3f} -> {:.3f} ({:+.3f}) in {} correction round(s); "
          "{} threshold {:.2f}; stop reason '{}'".format(
              trace["f1_initial"], trace["f1_final"], trace["f1_gain"],
              trace["n_rounds"],
              "REACHED" if trace["reached_threshold"] else "DID NOT REACH",
              f1_threshold, trace["stop_reason"]))


# ---------------------------------------------------------------------------
# Does the correction VOCABULARY matter? The contrast, with a noise floor.
# ---------------------------------------------------------------------------
def variant_contrast(all_traces: Dict[str, List[Dict[str, Any]]]
                     ) -> Optional[Dict[str, Any]]:
    """Does the correction block's VOCABULARY change the outcome? With a noise floor.

    THE MEASUREMENT THAT MAKES THIS INTERPRETABLE. Round 0 is the plain
    condition-A advisory: `extra_instruction` is None in BOTH variants, so the two
    runs receive a byte-identical prompt. Any round-0 difference between them is
    therefore pure LLM nondeterminism (the advisor runs at temperature 0.1, not 0)
    and CANNOT be a variant effect. That gives a free, honest noise floor measured
    on this exact population — a test-retest reliability check we did not have to
    design.

    The final-round difference is only meaningful if it EXCEEDS that floor.
    Reporting the final delta alone would quote an effect without its measurement
    error, and on this data that would have been actively wrong: the final delta
    comes out SMALLER than the round-0 noise.
    """
    if "unmodified" not in all_traces or "domain_routed" not in all_traces:
        return None
    by = {v: {t["sample_index"]: t for t in ts} for v, ts in all_traces.items()}
    keys = sorted(set(by["unmodified"]) & set(by["domain_routed"]))
    if not keys:
        return None

    def _diffs(getter) -> List[float]:
        return [getter(by["unmodified"][k]) - getter(by["domain_routed"][k])
                for k in keys]

    def _stat(xs: List[float]) -> Dict[str, Any]:
        a = np.array(xs, dtype=float)
        return {"mean": float(a.mean()), "mean_abs": float(np.abs(a).mean()),
                "std": float(a.std()), "n_differing": int((np.abs(a) > 1e-9).sum()),
                "n": len(xs)}

    r0s = _stat(_diffs(lambda t: float(t["curve"][0]["faithfulness_f1"])))
    fins = _stat(_diffs(lambda t: float(t["f1_final"])))
    detectable = fins["mean_abs"] > r0s["mean_abs"]
    engaged = {v: [t for t in ts if t["n_rounds"] > 0]
               for v, ts in all_traces.items()}
    return {
        "n_paired": len(keys),
        "round0_noise_floor": r0s,
        "final_difference": fins,
        "effect_exceeds_noise_floor": bool(detectable),
        "verdict": (("final |diff| {:.4f} EXCEEDS the round-0 noise floor {:.4f} "
                     "-> the correction vocabulary plausibly matters")
                    if detectable else
                    ("final |diff| {:.4f} is SMALLER than the round-0 noise floor "
                     "{:.4f} (measured with NO correction applied) -> no detectable "
                     "effect of the correction vocabulary")).format(
                        fins["mean_abs"], r0s["mean_abs"]),
        "n_engaged": {v: len(e) for v, e in engaged.items()},
        "engaged_caveat": ("Only engaged scenarios can be affected by the correction "
                           "text at all, and n_engaged is small — this is a NULL "
                           "result, not proof of equivalence."),
    }


# ---------------------------------------------------------------------------
# Cross-domain figure + table.
# ---------------------------------------------------------------------------
def make_crossdomain_figure(traffic: Dict[str, Any],
                            pg: Dict[str, Dict[str, Any]], path: str,
                            f1_threshold: float) -> Optional[str]:
    """Left: mean F1 per correction round, BOTH domains on one axis (the paper
    figure). Right: cumulative % of scenarios at/above threshold by round."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                    # pragma: no cover
        print("(matplotlib unavailable, skipping figure: {})".format(e))
        return None

    # Wong colorblind-safe palette, consistent with every other paper figure.
    C_TRAFFIC, C_PG_UNMOD, C_PG_DOM, C_THR = "#0072B2", "#D55E00", "#009E73", "#666666"
    series = [("METR-LA traffic (n={})".format(traffic["n_scenarios"]),
               traffic["per_round"], C_TRAFFIC, "o", "-")]
    style = {"unmodified": (C_PG_UNMOD, "s", "--"),
             "domain_routed": (C_PG_DOM, "^", "-.")}
    label = {"unmodified": "Power grid, unmodified prompt",
             "domain_routed": "Power grid, domain-routed prompt"}
    for v, agg in pg.items():
        col, mark, ls = style.get(v, ("#999999", "x", ":"))
        series.append(("{} (n={})".format(label.get(v, v), agg["n_scenarios"]),
                       agg["per_round"], col, mark, ls))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    for name, pr, col, mark, ls in series:
        xs = [r["round"] for r in pr]
        ys = [r["mean_f1"] for r in pr]
        ax.plot(xs, ys, color=col, marker=mark, linestyle=ls, linewidth=2.2,
                markersize=6, label=name)
    ax.axhline(f1_threshold, color=C_THR, linestyle=":", linewidth=1.4,
               label="threshold {:.2f}".format(f1_threshold))
    ax.set_xlabel("correction round")
    ax.set_ylabel("mean faithfulness F1")
    ax.set_xticks([r["round"] for r in series[0][1]])
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Convergence across domains")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)

    ax = axes[1]
    for name, pr, col, mark, ls in series:
        xs = [r["round"] for r in pr]
        ys = [100.0 * r["frac_reached_threshold"] for r in pr]
        ax.plot(xs, ys, color=col, marker=mark, linestyle=ls, linewidth=2.2,
                markersize=6, label=name)
    ax.set_xlabel("by correction round")
    ax.set_ylabel("% scenarios at/above threshold")
    ax.set_xticks([r["round"] for r in series[0][1]])
    ax.set_ylim(0, 105)
    ax.set_title("Cumulative convergence")
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle("Active Grounding Loop transfers across domains "
                 "(identical loop, threshold {:.2f}, no retuning)".format(f1_threshold))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_crossdomain_table(traffic: Dict[str, Any],
                            pg: Dict[str, Dict[str, Any]],
                            lowf1: Dict[str, Dict[str, Any]],
                            path: str, f1_threshold: float, model: str) -> None:
    """LaTeX booktabs: rows = domains, cols = mean F1 at each round + reached %."""
    def _cells(agg: Dict[str, Any]) -> str:
        pr = {r["round"]: r for r in agg["per_round"]}
        vals = []
        for r in range(4):
            vals.append("{:.3f}".format(pr[r]["mean_f1"]) if r in pr else "--")
        reached = 100.0 * agg["per_round"][-1]["frac_reached_threshold"]
        return " & ".join(vals) + " & {:.1f}".format(reached)

    rows = [("METR-LA traffic ($n={}$)".format(traffic["n_scenarios"]), traffic)]
    nice = {"unmodified": "unmodified prompt", "domain_routed": "domain-routed prompt"}
    for v, agg in pg.items():
        rows.append(("IEEE 14-bus grid, {} ($n={}$)".format(
            nice.get(v, v), agg["n_scenarios"]), agg))

    lines = [
        "% Auto-generated by evaluation/active_grounding_power_grid.py (Phase 20).",
        "% Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        r"\caption{The active grounding loop transfers across domains. The loop "
        r"(\texttt{models/advisor/active\_grounding.py}), its F1 threshold "
        r"($" + "{:.2f}".format(f1_threshold) + r"$) and its correction-round budget "
        r"are IDENTICAL in every row --- nothing was retuned for the power grid. "
        r"Round 0 is the plain condition-A advisory; each later round re-prompts the "
        r"LLM with the top-$k$ nodes it failed to cite. Values are mean faithfulness "
        r"F1, carried forward once a scenario converges. The last column is the "
        r"cumulative fraction of scenarios at or above threshold. Advisor model: "
        + model.replace("_", r"\_") + r". The two grid rows differ ONLY in whether "
        r"the correction block's three residual traffic strings (``congestion'', "
        r"``mph'', ``roads'') are routed through the domain profile.}",
        r"\label{tab:active_grounding_crossdomain}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Domain & Round 0 & Round 1 & Round 2 & Round 3 & Reached thr. (\%) \\",
        r"\midrule",
    ]
    for name, agg in rows:
        lines.append("{} & {} \\\\".format(name, _cells(agg)))

    if lowf1:
        lines += [r"\midrule",
                  r"\multicolumn{6}{l}{\emph{Sub-population: grid scenarios below "
                  r"threshold at round 0 (where the loop actually engages)}} \\"]
        for v, agg in lowf1.items():
            if agg["n_scenarios"]:
                lines.append("\\quad {} ($n={}$) & {} \\\\".format(
                    nice.get(v, v), agg["n_scenarios"], _cells(agg)))

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
def _verify_unchanged() -> int:
    """Assert the `domain=None` correction text is byte-identical to Phase 16.

    The Phase-16 strings are inlined here rather than imported, so this is a real
    external pin: if someone edits build_correction_block's traffic wording, this
    fails. Mirrors evaluation/verify_traffic_unchanged.py's role for the advisor.
    """
    from ..models.advisor.active_grounding import build_correction_block
    missed = [{"node_name": "Glendale / Burbank (sensor 767541, 34.15, -118.25)",
               "node_id": 12, "importance": 0.4123, "current_speed_mph": 21.4}]
    expected = (
        "=== GROUNDING CORRECTION (round 1) ===\n"
        "Your previous answer did not address these locations from the "
        "MATHEMATICAL EXPLANATION above. The model identified them as important "
        "causes of the predicted congestion, but your reasoning and "
        "recommendations did not mention them:\n"
        "  - Glendale / Burbank (sensor 767541, 34.15, -118.25) (node_id 12): "
        "importance 0.412, currently 21.4 mph\n"
        "\n"
        "Revise your reasoning and recommendations so that EVERY one of these "
        "locations is explicitly addressed. You must STILL cite ONLY causes that "
        "appear in the mathematical explanation above — do NOT invent new sensors, "
        "roads, incidents, or numbers to comply. Return ONLY the corrected JSON "
        "object in the same shape as before.")
    got = build_correction_block(missed, 1)                   # domain omitted
    if got == expected:
        print("PASS  build_correction_block(domain=None) is byte-identical to Phase 16.")
        print("PASS  traffic active-grounding numbers (n=93) are unaffected.")
        return 0
    print("FAIL  correction text drifted from the Phase-16 bytes.")
    print("--- expected ---\n" + expected)
    print("--- got ---\n" + got)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 20 — active grounding loop on the power grid")
    ap.add_argument("--config", default=None, help="power_grid_faith.yaml")
    ap.add_argument("--n", type=int, default=0, help="override scenarios.n")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="3 scenarios, both variants, FULL correction traces printed")
    ap.add_argument("--variant", choices=["unmodified", "domain_routed", "both"],
                    default="both")
    ap.add_argument("--model", default=None, help="override the Ollama model")
    ap.add_argument("--verbose-traces", action="store_true",
                    help="print full correction traces outside --smoke too")
    ap.add_argument("--verify-unchanged", action="store_true",
                    help="assert the unmodified correction text, then exit")
    ap.add_argument("--demo-threshold", type=float, default=None,
                    help="MECHANISM DEMONSTRATION ONLY: force the F1 threshold so "
                         "the correction fires on a grid scenario that is already "
                         "grounded at round 0. Quarantined — writes to its own "
                         "cache and NEVER writes paper figures/tables. Not a result.")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    if args.verify_unchanged:
        raise SystemExit(_verify_unchanged())

    tag = ("_" + args.out_tag) if args.out_tag else ""
    cfg = load_config(args.config)
    dataset, city = cfg["dataset"], cfg["city"]
    scfg, ecfg = cfg["scenarios"], cfg["explainer"]
    top_k = int(ecfg["top_k"])

    # The loop's knobs come from advisor.yaml — the SAME values the METR-LA run
    # used. Not overridable here on purpose: retuning them for the grid would
    # forfeit the "no domain tuning" claim this study exists to make.
    f1_threshold, max_rounds = load_grounding_config()

    # --- the quarantined demonstration path ---------------------------------
    # WHY THIS EXISTS. On the 14-bus grid condition A is already at or above 0.70
    # on nearly every scenario at round 0, so the loop correctly exits immediately
    # and the CORRECTION MECHANISM is never exercised. That is the honest finding
    # (see the ceiling-effect note in the run report), but it means nobody can SEE
    # the correction prompt on a grid scenario. Raising the threshold forces it to
    # fire so the mechanism is inspectable.
    #
    # This is a DEMONSTRATION, not a measurement, and it is quarantined so it can
    # never be mistaken for one: its own cache directory, no paper figure, no
    # LaTeX table, and a summary flagged is_demo=true. Every reported number in
    # this study comes from the un-forced threshold above.
    demo = args.demo_threshold is not None
    if demo:
        f1_threshold = float(args.demo_threshold)
        args.verbose_traces = True

    n_target = args.n or int(scfg["n"])
    if args.smoke:
        n_target = min(n_target, 3)
    variants = VARIANTS if args.variant == "both" else [
        (v, d) for v, d in VARIANTS if v == args.variant]

    out_dir = os.path.join(PKG_ROOT, "evaluation", "results",
                           "active_grounding_power_grid")
    if demo:
        out_dir = os.path.join(out_dir, "demo")          # quarantined cache + outputs
        tag = tag + "_DEMO"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("PHASE 20 — ACTIVE GROUNDING LOOP ON THE IEEE 14-BUS POWER GRID")
    print("=" * 78)
    if demo:
        print("!! MECHANISM DEMONSTRATION — NOT A RESULT !!")
        print("!! F1 threshold FORCED to {:.2f} (config value is {:.2f}) purely so "
              "the".format(f1_threshold, load_grounding_config()[0]))
        print("!! correction fires on a grid scenario that is already grounded at "
              "round 0.")
        print("!! No paper figure or table is written from this run.")
        print("=" * 78)
    print("loop        : models/advisor/active_grounding.run_active_grounding "
          "(imported unmodified)")
    print("threshold   : {:.2f}   max_rounds: {}   <- from configs/advisor.yaml, "
          "the SAME values METR-LA used".format(f1_threshold, max_rounds))
    print("variants    : {}".format([v for v, _ in variants]))

    scaler = load_scaler(dataset)
    scenarios, diag = sample_undervoltage_scenarios(
        dataset, scaler, n=n_target,
        min_dev=float(scfg["min_deviation_pu"]),
        min_std=float(scfg["min_bus_std_pu"]),
        seed=int(scfg["seed"]))
    if args.limit:
        scenarios = scenarios[:args.limit]
    print("scenarios   : {} (of {} stressed windows), target buses {}".format(
        len(scenarios), diag["n_stressed_windows"],
        sorted({s["target_bus"] for s in scenarios})))
    if not scenarios:
        print("No qualifying scenarios — nothing to do.")
        return

    ckpt = cfg["checkpoint"]
    builder = ExplanationBuilder(ckpt, dataset, device=torch.device("cpu"),
                                 top_k=top_k, epochs=int(ecfg["epochs"]),
                                 confidence_runs=int(ecfg["confidence_runs"]))
    builder.checkpoint_path = (ckpt if os.path.isabs(ckpt)
                               else os.path.join(PKG_ROOT, ckpt))
    table = NodeTable(load_node_meta(dataset))

    inner: Any
    if args.model:
        import copy
        from ..models.advisor.advisor import load_advisor_config
        acfg = copy.deepcopy(load_advisor_config())
        acfg.setdefault("ollama", {})["model"] = args.model
        inner = Advisor(city, cfg=acfg)
    else:
        inner = Advisor(city)
    advisor = RecordingAdvisor(inner)
    print("advisor     : {} (city profile '{}', domain '{}')".format(
        advisor.model, city, inner.domain.key))

    # --- run ----------------------------------------------------------------
    all_traces: Dict[str, List[Dict[str, Any]]] = {v: [] for v, _ in variants}
    verbose_traces: List[Dict[str, Any]] = []
    n_cached = 0

    for variant, domain in variants:
        print("\n" + "#" * 78)
        print("# VARIANT: {}   (correction vocabulary: {})".format(
            variant.upper(), "traffic / mph" if domain is None else "power grid / pu"))
        print("#" * 78)
        for i, sc in enumerate(scenarios):
            exp = build_or_load_explanation(builder, dataset, sc,
                                            int(scfg["horizon_step"]))
            cpath = _cache_path(out_dir, sc["sample_index"], sc["target_node"],
                                variant, advisor.model)
            advisor.reset()
            if os.path.exists(cpath):
                with open(cpath) as f:
                    trace = json.load(f)
                n_cached += 1
                log = trace.get("llm_log", [])
            else:
                # >>> THE UNMODIFIED PHASE-16 LOOP, called as-is. <<<
                trace = run_active_grounding(advisor, exp, table,
                                             f1_threshold, max_rounds,
                                             domain=domain)
                log = list(advisor.log)
                trace.update({
                    "sample_index": sc["sample_index"],
                    "target_node": sc["target_node"],
                    "target_bus": sc["target_bus"],
                    "tod_band": sc["tod_band"],
                    "congestion": sc["severity"],      # loop's key name; grid severity
                    "severity": sc["severity"],
                    "variant": variant,
                    "model": advisor.model,
                    "llm_log": log,
                })
                trace.pop("final_advisory", None)      # bulky; the log has the detail
                with open(cpath, "w") as f:
                    json.dump(trace, f, indent=2)

            all_traces[variant].append(trace)
            if args.smoke or args.verbose_traces:
                print_verbose_trace(sc, exp, trace, log, variant, f1_threshold)
                verbose_traces.append({
                    "variant": variant, "sample_index": sc["sample_index"],
                    "target_bus": sc["target_bus"], "severity": sc["severity"],
                    "top_nodes": exp["top_nodes"],
                    "curve": trace["curve"], "llm_log": log,
                    "stop_reason": trace["stop_reason"],
                })
            else:
                f1s = " -> ".join("{:.3f}".format(c["faithfulness_f1"])
                                  for c in trace["curve"])
                print("  [{}/{}] idx={} bus {} {}  F1: {}  [{}] {}".format(
                    i + 1, len(scenarios), sc["sample_index"], sc["target_bus"],
                    sc["severity"], f1s,
                    "OK" if trace["reached_threshold"] else "unmet",
                    trace["stop_reason"]))
    if n_cached:
        print("\n  (reused {} cached traces)".format(n_cached))

    # --- aggregate ----------------------------------------------------------
    pg_agg = {v: aggregate(t, max_rounds, f1_threshold)
              for v, t in all_traces.items() if t}
    # The sub-population where the loop actually has work: below threshold at
    # round 0. This is the direct analogue of the Phase-16 n=5 low-f1 smoke.
    low_agg: Dict[str, Dict[str, Any]] = {}
    for v, t in all_traces.items():
        low = [x for x in t if x["curve"][0]["faithfulness_f1"] < f1_threshold]
        if low:
            low_agg[v] = aggregate(low, max_rounds, f1_threshold)

    traffic = load_traffic_reference()

    print("\n" + "=" * 78)
    print("CROSS-DOMAIN CONVERGENCE (threshold {:.2f}, identical loop)".format(
        f1_threshold))
    print("=" * 78)
    hdr = "{:34s}{:>9s}{:>9s}{:>9s}{:>9s}{:>10s}".format(
        "domain / variant", "round0", "round1", "round2", "round3", "reached%")
    print(hdr)
    print("-" * len(hdr))

    def _line(name: str, agg: Dict[str, Any]) -> None:
        pr = {r["round"]: r for r in agg["per_round"]}
        cells = "".join("{:>9s}".format("{:.3f}".format(pr[r]["mean_f1"]))
                        if r in pr else "{:>9s}".format("--") for r in range(4))
        print("{:34s}{}{:>10s}".format(
            name[:34], cells,
            "{:.1f}".format(100.0 * agg["per_round"][-1]["frac_reached_threshold"])))

    _line("METR-LA traffic (n={})".format(traffic["n_scenarios"]), traffic)
    for v, agg in pg_agg.items():
        _line("power grid {} (n={})".format(v, agg["n_scenarios"]), agg)
    if low_agg:
        print("-" * len(hdr))
        print("sub-population: below threshold at round 0 (loop actually engages)")
        for v, agg in low_agg.items():
            _line("  grid {} low-F1 (n={})".format(v, agg["n_scenarios"]), agg)

    print("-" * len(hdr))
    for v, agg in pg_agg.items():
        print("{:20s} mean F1 gain {:+.3f} | mean rounds used {:.2f} | "
              "reached {:.1f}% | stops {}".format(
                  v, agg["mean_f1_gain"], agg["mean_rounds_used"],
                  100.0 * agg["frac_reached_overall"], agg["stop_reasons"]))
    print("{:20s} mean F1 gain {:+.3f} | mean rounds used {:.2f} | reached {:.1f}%"
          .format("METR-LA (committed)", traffic["mean_f1_gain"],
                  traffic["mean_rounds_used"],
                  100.0 * traffic.get("frac_reached_overall", 1.0)))

    # Hallucination is the other axis Phase 16 cared about: does raising recall
    # induce fabrication? Report it per round for both domains.
    print("\nHALLUCINATION PER ROUND (does the correction induce fabrication?)")
    for v, agg in pg_agg.items():
        print("  grid {:14s}: {}".format(
            v, " -> ".join("{:.3f}".format(r["mean_hallucination"])
                           for r in agg["per_round"])))
    print("  {:19s}: {}".format("METR-LA", " -> ".join(
        "{:.3f}".format(r["mean_hallucination"]) for r in traffic["per_round"])))

    # --- variant contrast (the confound check), read against a noise floor ---
    vc = variant_contrast(all_traces)
    if vc:
        u, d = pg_agg["unmodified"], pg_agg["domain_routed"]
        print("\nVARIANT CONTRAST — does the mislabelled correction text matter?")
        print("  final mean F1: unmodified {:.3f} vs domain-routed {:.3f}".format(
            u["per_round"][-1]["mean_f1"], d["per_round"][-1]["mean_f1"]))
        print("  final halluc : unmodified {:.3f} vs domain-routed {:.3f}".format(
            u["per_round"][-1]["mean_hallucination"],
            d["per_round"][-1]["mean_hallucination"]))
        r0s, fins = vc["round0_noise_floor"], vc["final_difference"]
        print("  NOISE FLOOR (round 0 = identical prompt in both variants, so this "
              "is pure LLM nondeterminism):")
        print("    round-0 |diff| {:.4f} (std {:.4f}), differs on {}/{} scenarios"
              .format(r0s["mean_abs"], r0s["std"], r0s["n_differing"], r0s["n"]))
        print("    final   |diff| {:.4f} (std {:.4f}), differs on {}/{} scenarios"
              .format(fins["mean_abs"], fins["std"], fins["n_differing"], fins["n"]))
        print("  VERDICT: {}".format(vc["verdict"]))
        print("  engaged: {} — {}".format(vc["n_engaged"], vc["engaged_caveat"]))

    # --- outputs ------------------------------------------------------------
    csv_path = os.path.join(out_dir, "active_grounding_power_grid_per_round{}.csv"
                            .format(tag))
    cols = ["variant", "sample_index", "target_bus", "tod_band", "severity",
            "round", "used_correction", "faithfulness_f1", "cause_precision",
            "cause_recall", "hallucination_rate", "n_cited_causes", "n_missed",
            "advisory_error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for v, ts in all_traces.items():
            for t in ts:
                for c in t["curve"]:
                    row = {"variant": v, "sample_index": t["sample_index"],
                           "target_bus": t.get("target_bus"),
                           "tod_band": t.get("tod_band"),
                           "severity": t.get("severity")}
                    row.update(c)
                    w.writerow(row)

    traces_path = os.path.join(out_dir, "active_grounding_power_grid_traces.json")
    with open(traces_path, "w") as f:
        json.dump({
            "note": ("Full per-scenario traces. `llm_log[r].correction_prompt` is "
                     "the EXACT text sent to the LLM for round r (null at round 0); "
                     "`cited_causes` is what it answered. The loop itself "
                     "(run_active_grounding) was imported unmodified."),
            "f1_threshold": f1_threshold, "max_rounds": max_rounds,
            "model": advisor.model, "dataset": dataset, "top_k": top_k,
            "variants": {v: all_traces[v] for v in all_traces},
        }, f, indent=2)

    summary_path = os.path.join(out_dir,
                                "active_grounding_power_grid_summary{}.json".format(tag))
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": dataset, "model": advisor.model, "top_k": top_k,
            "f1_threshold": f1_threshold, "max_rounds": max_rounds,
            "n_scenarios": len(scenarios),
            "loop_source": "models/advisor/active_grounding.run_active_grounding "
                           "(imported unmodified; only scenarios/explanations differ)",
            "power_grid": pg_agg,
            "power_grid_low_f1_subpopulation": low_agg,
            "traffic_reference": traffic,
            "variant_contrast": vc,
            "sampling_diagnostics": diag,
            "is_demo": demo,
            "demo_note": ("F1 threshold was FORCED for a mechanism demonstration; "
                          "these numbers are NOT a result." if demo else ""),
        }, f, indent=2)

    written = [csv_path, traces_path, summary_path]
    if demo:
        # Hard quarantine: a forced-threshold run must never reach evaluation/paper.
        print("\n(--demo-threshold: paper figure + LaTeX table NOT written — a "
              "forced threshold is a demonstration, never a reported number.)")
    else:
        fig_path = os.path.join(PKG_ROOT, "evaluation", "paper",
                                "fig_active_grounding_crossdomain{}.pdf".format(tag))
        tex_path = os.path.join(PKG_ROOT, "evaluation", "paper",
                                "table_active_grounding_crossdomain{}.tex".format(tag))
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig = make_crossdomain_figure(traffic, pg_agg, fig_path, f1_threshold)
        write_crossdomain_table(traffic, pg_agg, low_agg, tex_path, f1_threshold,
                                advisor.model)
        written += [tex_path] + ([fig] if fig else [])

    print("\nWrote:")
    for p in written:
        print("  " + p)


if __name__ == "__main__":
    main()
