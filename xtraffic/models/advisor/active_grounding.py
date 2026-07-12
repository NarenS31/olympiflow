"""Phase 16 — the Active Grounding Loop.

WHY THIS EXISTS
---------------
Phase 5 MEASURED that the mathematical explanation makes the LLM's reasoning
faithful (hallucination ~0, high F1). This phase turns that measurement into a
MECHANISM: after the advisor answers, we score its faithfulness automatically and,
if it falls short, we hand the LLM a targeted correction and let it try again.

THE LOOP
--------
  1. Run the Phase-4 advisor on a scenario under CONDITION A -> advisory (round 0).
  2. Score it with the Phase-5 faithfulness metric -> F1 (+ precision/recall/halluc).
  3. If F1 >= threshold (default 0.7): done.
     Else: build a CORRECTION prompt that names the SPECIFIC explainer top-k nodes
     the LLM failed to cite (with their importance + current speed), append it, and
     re-prompt. Repeat up to `max_rounds` (default 3).
  4. Record F1/hallucination/recall at EVERY round = a per-scenario CONVERGENCE
     CURVE. Aggregate across scenarios into "does correction raise F1, and by how
     much per round?" plus "fraction reaching threshold within k rounds".

WHAT IT REUSES (so this file stays small and transparent)
---------------------------------------------------------
  * models/advisor/advisor.py  -> Advisor.advise_condition(exp, "A",
    extra_instruction=...) for both the first prompt and each corrective re-prompt.
    The `extra_instruction` hook is a Phase-16 default-preserving addition (flagged
    in advisor.py) — with it None, the prompt is byte-identical to Phase 4/5.
  * evaluation/faithfulness.py -> score_advisory + NodeTable: the SAME entity
    resolver and F1/hallucination metric the paper's Contribution #2 uses. We do
    NOT invent a second notion of "faithful" here; the loop optimises against the
    exact metric we report.
  * evaluation/run_faithfulness_study.py -> sample_scenarios + build_or_load_
    explanation: identical stratified sampling and cached explanations, so this
    study runs on the SAME population as Phase 5 (comparable, and free when the
    explanations are already cached).

WHY THE CORRECTION TARGETS RECALL
---------------------------------
On METR-LA the condition-A failures are overwhelmingly UNDER-CITATION: the LLM
names the single strongest source and omits the rest of the top-k (precision ~1,
recall collapses, hallucination ~0 — see Phase 13). So "you did not address these
locations" is the right lever: it pushes recall up without inviting the model to
invent causes. If instead a failure is precision-side (the LLM cited something NOT
in the top-k), there are no "missed" nodes to name and this recall-correction
cannot help — we detect that (missed set empty while F1 < threshold) and stop with
reason "no_missed_nodes" rather than pretend to fix it. Honest by construction.

HONEST CAVEAT (state it in the paper)
-------------------------------------
This optimises the advisory TOWARD the explainer's top-k, so it is an ENFORCEMENT /
internal-consistency result, NOT independent evidence that the explanation itself
is correct. Report it that way: "we built a system that ENFORCES grounding in a
closed loop", not "we proved the explanation is right".

CONFIG (no magic numbers in code — CLAUDE.md rule)
--------------------------------------------------
  f1_threshold = 0.7, max_rounds = 3  -> the `active_grounding` block in
  configs/advisor.yaml. Convergence curves + table -> evaluation/results/
  active_grounding/ and evaluation/paper/table_active_grounding.tex.

Run (needs the trained checkpoint + a local Ollama with the advisor model):
  # 5-scenario smoke on the WORST Phase-5 failures (shows the loop actually working)
  python -m xtraffic.models.advisor.active_grounding --select low-f1 --limit 5
  # full study on the same stratified population as Phase 5
  python -m xtraffic.models.advisor.active_grounding --per-stratum 7
  # harness check with no Ollama (mock model that improves recall each round)
  python -m xtraffic.models.advisor.active_grounding --mock-llm --select low-f1 --limit 5

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ...utils.io_utils import PKG_ROOT
from ..explainer.explain import ExplanationBuilder
from .advisor import Advisor, load_advisor_config


# ===========================================================================
# Config: the two knobs (threshold + max rounds) live in advisor.yaml.
# ===========================================================================
# Defaults if the config predates Phase 16 — so an old advisor.yaml still runs.
DEFAULT_F1_THRESHOLD = 0.7
DEFAULT_MAX_ROUNDS = 3


def load_grounding_config(cfg: Optional[Dict[str, Any]] = None
                          ) -> Tuple[float, int]:
    """Return (f1_threshold, max_rounds) from the advisor.yaml `active_grounding`
    block, falling back to the documented defaults if the block is absent."""
    cfg = cfg if cfg is not None else load_advisor_config()
    block = cfg.get("active_grounding", {}) or {}
    f1_threshold = float(block.get("f1_threshold", DEFAULT_F1_THRESHOLD))
    max_rounds = int(block.get("max_rounds", DEFAULT_MAX_ROUNDS))
    return f1_threshold, max_rounds


# ===========================================================================
# The correction: which top-k nodes did the advisory miss, and the prompt block
# that names them.
# ===========================================================================
def missed_topk_nodes(exp: Dict[str, Any],
                      metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The explainer top_nodes NOT covered by any cited cause, in the explanation's
    own order (strongest first).

    `metrics` is a score_advisory() result; each per_cause entry logs `hit_nodes`
    = the top-k node ids that citation covered. The COVERED set is their union;
    the MISSED nodes are the top_nodes whose id is not in it. We return the full
    node dicts (name, importance, current speed) because the correction prompt
    quotes those, so the model is told exactly which real places to address."""
    covered: Set[int] = set()
    for pc in metrics.get("per_cause", []):
        covered |= {int(x) for x in pc.get("hit_nodes", [])}
    return [n for n in exp.get("top_nodes", [])
            if int(n["node_id"]) not in covered]


def build_correction_block(missed: List[Dict[str, Any]], round_num: int) -> str:
    """The targeted correction appended after the TASK for the next round.

    It (a) names the missed top-k locations with their importance and current
    speed, (b) demands EVERY one be addressed, and (c) re-states the hard rule
    against inventing causes — so raising recall never comes at the cost of
    precision. round_num is the round this correction produces (1-based), logged
    in the text for traceability."""
    lines = [
        "=== GROUNDING CORRECTION (round {}) ===".format(round_num),
        "Your previous answer did not address these locations from the "
        "MATHEMATICAL EXPLANATION above. The model identified them as important "
        "causes of the predicted congestion, but your reasoning and "
        "recommendations did not mention them:",
    ]
    for n in missed:
        lines.append(
            "  - {} (node_id {}): importance {:.3f}, currently {} mph".format(
                n["node_name"], n["node_id"], float(n["importance"]),
                n["current_speed_mph"]))
    lines.append("")
    lines.append(
        "Revise your reasoning and recommendations so that EVERY one of these "
        "locations is explicitly addressed. You must STILL cite ONLY causes that "
        "appear in the mathematical explanation above — do NOT invent new sensors, "
        "roads, incidents, or numbers to comply. Return ONLY the corrected JSON "
        "object in the same shape as before.")
    return "\n".join(lines)


# ===========================================================================
# The loop itself.
# ===========================================================================
def _region(name: str) -> str:
    """Strip the '(sensor 123, lat, lon)' tail so logs/figures show region names."""
    return name.split(" (sensor")[0].split(" (segment")[0].strip()


def run_active_grounding(advisor: Any, exp: Dict[str, Any], table: Any,
                         f1_threshold: float, max_rounds: int
                         ) -> Dict[str, Any]:
    """Run the closed grounding loop on ONE (explanation) and return its trace.

    Structure (round 0 = the plain condition-A answer; rounds 1..max are
    corrections):

        extra = None
        for r in 0..max_rounds:
            advisory = advise_condition(exp, "A", extra_instruction=extra)
            score -> f1, missed top-k
            record the round
            if f1 >= threshold: stop "threshold_reached"
            if r == max_rounds:  stop "max_rounds"
            if not missed:       stop "no_missed_nodes"  (recall-correction can't help)
            extra = correction naming the missed nodes   (feeds round r+1)

    Returns a dict with the per-round `curve`, whether/why it stopped, and the
    final advisory (kept for inspection)."""
    # score_advisory is imported lazily so importing this module doesn't pull the
    # evaluation package (mirrors NodeTable's own lazy import of the explainer).
    from ...evaluation.faithfulness import score_advisory

    curve: List[Dict[str, Any]] = []
    extra: Optional[str] = None
    final_advisory: Dict[str, Any] = {}
    stop_reason = "max_rounds"

    for r in range(max_rounds + 1):
        res = advisor.advise_condition(exp, "A", extra_instruction=extra)
        advisory = res["advisory"]
        final_advisory = advisory
        metrics = score_advisory(exp, advisory, table)
        missed = missed_topk_nodes(exp, metrics)

        curve.append({
            "round": r,
            "used_correction": extra is not None,
            "faithfulness_f1": metrics["faithfulness_f1"],
            "cause_precision": metrics["cause_precision"],
            "cause_recall": metrics["cause_recall"],
            "hallucination_rate": metrics["hallucination_rate"],
            "n_cited_causes": metrics["n_cited_causes"],
            "n_topk": metrics["n_topk"],
            "n_missed": len(missed),
            # Region names (no raw ids) of what was still missed AFTER this round —
            # what the NEXT correction will name. Powers "most-missed" aggregation.
            "missed_regions": [_region(n["node_name"]) for n in missed],
            "missed_node_ids": [int(n["node_id"]) for n in missed],
            "advisory_error": advisory.get("_error", ""),
        })

        if metrics["faithfulness_f1"] >= f1_threshold:
            stop_reason = "threshold_reached"
            break
        if r == max_rounds:
            stop_reason = "max_rounds"
            break
        if not missed:
            # F1 is below threshold but nothing is under-cited -> the shortfall is
            # precision-side (a cause outside the top-k). Our recall-correction has
            # no missed node to name, so honestly stop instead of looping uselessly.
            stop_reason = "no_missed_nodes"
            break
        extra = build_correction_block(missed, r + 1)

    f1_0 = curve[0]["faithfulness_f1"]
    f1_f = curve[-1]["faithfulness_f1"]
    return {
        "curve": curve,
        "n_rounds": len(curve) - 1,          # correction rounds actually run
        "reached_threshold": f1_f >= f1_threshold,
        "stop_reason": stop_reason,
        "f1_initial": f1_0,
        "f1_final": f1_f,
        "f1_gain": f1_f - f1_0,
        "final_advisory": final_advisory,
    }


# ===========================================================================
# Mock advisor for --mock-llm: no Ollama, but it IMPROVES recall each round so the
# harness (loop -> aggregate -> figure -> table) is verifiable end-to-end.
# ===========================================================================
class MockAdvisor:
    """Cites the top (1 + correction_round) nodes: round 0 cites 1 (low recall),
    then each correction cites one more, so F1 climbs across rounds exactly like we
    hope the real model does. It reads the round number out of the correction text
    (which build_correction_block stamps) — no hidden state, so it is deterministic
    and resumable-safe."""

    def __init__(self, model: str = "mock"):
        self.model = model

    def advise_condition(self, exp: Dict[str, Any], condition: str,
                         extra_instruction: Optional[str] = None) -> Dict[str, Any]:
        n_cite = 1
        if extra_instruction:
            m = re.search(r"round\s+(\d+)", extra_instruction)
            if m:
                n_cite = 1 + int(m.group(1))
        top = exp.get("top_nodes", [])
        causes = [{"location": n["node_name"], "resolved_node_id": n["node_id"]}
                  for n in top[:n_cite]]
        adv = {
            "reasoning": "Mock reasoning citing the top {} source(s).".format(n_cite),
            "cited_causes": causes,
            "recommendations": [
                {"action": "mock", "location": "mock", "time_window_minutes": 15,
                 "expected_effect": "mock", "grounded_in": ["mock"]}
                for _ in range(3)],
        }
        return {"advisory": adv, "condition": condition, "model": self.model}


# ===========================================================================
# Scenario selection.
# ===========================================================================
def _explanation_exists(dataset: str, idx: int, node: int) -> bool:
    path = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness",
                        "explanations_cache",
                        "{}_{}_{}.json".format(dataset, idx, node))
    return os.path.exists(path)


def select_low_f1(dataset: str, f1_threshold: float, limit: int
                  ) -> List[Dict[str, Any]]:
    """Pick the WORST condition-A scenarios from the Phase-5 per-scenario CSV
    (lowest F1 first) that have a cached explanation. This is the demonstrative
    set: these are real failures, so the correction loop has something to fix.
    Used for the smoke run and any "show me it working" report."""
    csv_path = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness",
                            "faithfulness_per_scenario.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(
            "[phase16] --select low-f1 needs the Phase-5 study CSV ({}). Run "
            "the faithfulness study first, or use --select stratified."
            .format(csv_path))

    def _f(v: str) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    rows = [r for r in csv.DictReader(open(csv_path)) if r["condition"] == "A"]
    rows = [r for r in rows if _f(r["faithfulness_f1"]) == _f(r["faithfulness_f1"])]
    rows.sort(key=lambda r: _f(r["faithfulness_f1"]))
    picked: List[Dict[str, Any]] = []
    for r in rows:
        idx, node = int(r["sample_index"]), int(r["target_node"])
        if not _explanation_exists(dataset, idx, node):
            continue                              # only cached -> no explainer cost
        picked.append({
            "sample_index": idx, "target_node": node,
            "tod_band": r.get("tod_band", "?"), "congestion": r.get("congestion", "?"),
            "phase5_f1": round(_f(r["faithfulness_f1"]), 3),
            "timestamp": "test#{}".format(idx),
        })
        if limit and len(picked) >= limit:
            break
    return picked


# ===========================================================================
# Per-scenario caching (resumable, like every other Phase 10-13 study).
# ===========================================================================
def _decision_cache_path(out_dir: str, dataset: str, idx: int, node: int,
                         model: str) -> str:
    d = os.path.join(out_dir, "decisions_cache")
    os.makedirs(d, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return os.path.join(d, "{}_{}_{}_{}.json".format(dataset, idx, node, safe_model))


# ===========================================================================
# Aggregation.
# ===========================================================================
def _carry_forward(curve: List[Dict[str, Any]], key: str, r: int) -> float:
    """A scenario's `key` value at round r, holding the LAST value once the loop
    stopped early (a scenario that converged at round 1 is treated as staying
    converged at rounds 2, 3). This is the honest "if you stopped at round r"
    view and makes the aggregate curve comparable across scenarios."""
    last = curve[min(r, len(curve) - 1)]
    return float(last[key])


def aggregate(traces: List[Dict[str, Any]], max_rounds: int,
              f1_threshold: float) -> Dict[str, Any]:
    """Build the per-round aggregate curve + cumulative threshold rates."""
    n = len(traces)
    per_round: List[Dict[str, Any]] = []
    for r in range(max_rounds + 1):
        f1s = [_carry_forward(t["curve"], "faithfulness_f1", r) for t in traces]
        rec = [_carry_forward(t["curve"], "cause_recall", r) for t in traces]
        hal = [_carry_forward(t["curve"], "hallucination_rate", r) for t in traces]
        prec = [_carry_forward(t["curve"], "cause_precision", r) for t in traces]
        # Cumulative: fraction whose F1 is at/above threshold BY round r.
        reached = sum(1 for x in f1s if x >= f1_threshold)
        per_round.append({
            "round": r,
            "mean_f1": _mean(f1s), "std_f1": _std(f1s),
            "mean_recall": _mean(rec),
            "mean_precision": _mean(prec),
            "mean_hallucination": _mean(hal),
            "frac_reached_threshold": (reached / n) if n else 0.0,
            "n_reached_threshold": reached,
        })
    return {
        "n_scenarios": n,
        "f1_threshold": f1_threshold,
        "max_rounds": max_rounds,
        "per_round": per_round,
        "mean_f1_gain": _mean([t["f1_gain"] for t in traces]),
        "mean_rounds_used": _mean([float(t["n_rounds"]) for t in traces]),
        "frac_reached_overall": (
            sum(1 for t in traces if t["reached_threshold"]) / n) if n else 0.0,
        "stop_reasons": dict(collections.Counter(t["stop_reason"] for t in traces)),
    }


def most_missed_regions(traces: List[Dict[str, Any]], top: int = 8
                        ) -> List[Tuple[str, int]]:
    """Which regions did the LLM most often fail to cite at round 0 (before any
    correction)? A pointer to where under-citation concentrates."""
    ctr: "collections.Counter[str]" = collections.Counter()
    for t in traces:
        for reg in t["curve"][0].get("missed_regions", []):
            ctr[reg] += 1
    return ctr.most_common(top)


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]                # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# ===========================================================================
# Outputs: CSV, JSONL, convergence figure, LaTeX table.
# ===========================================================================
_CSV_COLS = ["sample_index", "target_node", "tod_band", "congestion", "round",
             "used_correction", "faithfulness_f1", "cause_precision",
             "cause_recall", "hallucination_rate", "n_cited_causes", "n_missed",
             "advisory_error"]


def write_csv(traces: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for t in traces:
            for c in t["curve"]:
                row = {"sample_index": t["sample_index"],
                       "target_node": t["target_node"],
                       "tod_band": t["tod_band"], "congestion": t["congestion"]}
                row.update(c)
                w.writerow(row)


def write_jsonl(traces: List[Dict[str, Any]], path: str) -> None:
    """One compact line per scenario (the full curve, minus the bulky final
    advisory which lives in the per-scenario cache)."""
    with open(path, "w") as f:
        for t in traces:
            rec = {k: v for k, v in t.items() if k != "final_advisory"}
            f.write(json.dumps(rec) + "\n")


def make_figure(agg: Dict[str, Any], traces: List[Dict[str, Any]], path: str
                ) -> Optional[str]:
    """Two panels: (L) per-scenario F1 across rounds (carry-forward spaghetti) with
    the bold mean and the threshold line; (R) cumulative fraction reaching the
    threshold by round. Vector PDF, colorblind-safe."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                        # pragma: no cover
        print("(matplotlib unavailable, skipping figure: {})".format(e))
        return None

    max_rounds = agg["max_rounds"]
    thr = agg["f1_threshold"]
    rounds = list(range(max_rounds + 1))
    # Wong colorblind-safe: blue mean, vermillion threshold, gray spaghetti.
    C_MEAN, C_THR, C_LINE, C_BAR = "#0072B2", "#D55E00", "#999999", "#009E73"

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    for t in traces:                              # faint per-scenario trajectories
        ys = [_carry_forward(t["curve"], "faithfulness_f1", r) for r in rounds]
        ax.plot(rounds, ys, color=C_LINE, alpha=0.35, linewidth=0.8, zorder=1)
    means = [pr["mean_f1"] for pr in agg["per_round"]]
    ax.plot(rounds, means, color=C_MEAN, linewidth=2.5, marker="o",
            label="mean F1", zorder=3)
    ax.axhline(thr, color=C_THR, linestyle="--", linewidth=1.5,
               label="threshold {:.2f}".format(thr), zorder=2)
    ax.set_xlabel("correction round")
    ax.set_ylabel("faithfulness F1")
    ax.set_xticks(rounds)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("F1 across correction rounds (n={})".format(agg["n_scenarios"]))
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    fracs = [100.0 * pr["frac_reached_threshold"] for pr in agg["per_round"]]
    ax.bar(rounds, fracs, color=C_BAR)
    for x, y in zip(rounds, fracs):
        ax.text(x, y + 1.5, "{:.0f}%".format(y), ha="center", fontsize=8)
    ax.set_xlabel("by correction round")
    ax.set_ylabel("% scenarios reaching threshold")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 105)
    ax.set_title("Cumulative convergence")

    fig.suptitle("Phase 16 — Active Grounding Loop (F1 threshold {:.2f})".format(thr))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_table(agg: Dict[str, Any], out_tex: str) -> None:
    """LaTeX booktabs: one row per round, columns mean F1 / recall / hallucination /
    cumulative % reaching threshold. \\input-ready."""
    pr = agg["per_round"]
    lines = [
        "% Auto-generated by models/advisor/active_grounding.py (Phase 16). Do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Active Grounding Loop on METR-LA ($n=" + str(agg["n_scenarios"])
        + r"$ scenarios, F1 threshold $" + "{:.2f}".format(agg["f1_threshold"])
        + r"$, up to " + str(agg["max_rounds"]) + r" correction rounds). Round 0 is "
        r"the plain condition-A advisory; each later round re-prompts the LLM with "
        r"the top-$k$ nodes it failed to cite. Values are carried forward once a "
        r"scenario converges. Faithfulness F1, recall, and hallucination are the "
        r"Phase-5 metrics; the last column is the cumulative fraction of scenarios "
        r"at or above threshold by that round.}",
        r"\label{tab:active_grounding}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Round & Mean F1 & Mean recall & Mean halluc. & Reached thr. (\%) \\",
        r"\midrule",
    ]
    for row in pr:
        lines.append("{} & {:.3f} & {:.3f} & {:.3f} & {:.1f} \\\\".format(
            row["round"], row["mean_f1"], row["mean_recall"],
            row["mean_hallucination"], 100.0 * row["frac_reached_threshold"]))
    gain = agg["mean_f1_gain"]
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\footnotesize Mean F1 gain (round 0 $\rightarrow$ final): "
        + "{:+.3f}".format(gain) + r"; mean correction rounds used: "
        + "{:.2f}".format(agg["mean_rounds_used"]) + r".} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))


# ===========================================================================
# CLI.
# ===========================================================================
def _build_advisor(args: argparse.Namespace) -> Any:
    if args.mock_llm:
        return MockAdvisor()
    if args.model:
        cfg = copy.deepcopy(load_advisor_config())
        cfg.setdefault("ollama", {})["model"] = args.model
        return Advisor(args.city, cfg=cfg)
    return Advisor(args.city)


def _get_scenarios(args: argparse.Namespace, builder: ExplanationBuilder,
                   f1_threshold: float) -> List[Dict[str, Any]]:
    if args.select == "low-f1":
        return select_low_f1(args.dataset, f1_threshold, args.limit)
    # stratified (default): the SAME sampler + population as the Phase-5 study.
    from ...evaluation.run_faithfulness_study import sample_scenarios
    scenarios = sample_scenarios(args.dataset, builder.scaler, args.per_stratum)
    if args.limit and len(scenarios) > args.limit:
        scenarios = scenarios[:args.limit]
    return scenarios


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 16 Active Grounding Loop")
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--city", default="metr_la")
    ap.add_argument("--checkpoint",
                    default="models/gnn/checkpoints/metr_la_best.pt")
    ap.add_argument("--select", choices=["stratified", "low-f1"],
                    default="stratified",
                    help="stratified = Phase-5 sampler (default, full study); "
                         "low-f1 = the worst Phase-5 condition-A failures (smoke / "
                         "demonstration — the loop actually has work to do).")
    ap.add_argument("--per-stratum", type=int, default=7,
                    help="stratified only: windows per (tod x congestion) cell")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total scenarios (0 = no cap for stratified; low-f1 "
                         "defaults to no cap but you usually pass a small N)")
    ap.add_argument("--model", default=None,
                    help="override the Ollama advisor model (e.g. mistral:7b)")
    ap.add_argument("--mock-llm", action="store_true",
                    help="no Ollama — a mock model that raises recall each round, "
                         "to verify the harness end-to-end")
    ap.add_argument("--f1-threshold", type=float, default=None,
                    help="override advisor.yaml active_grounding.f1_threshold")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="override advisor.yaml active_grounding.max_rounds")
    ap.add_argument("--out-tag", default="",
                    help="suffix on output filenames so runs don't clobber")
    args = ap.parse_args()
    tag = ("_" + args.out_tag) if args.out_tag else ""

    # Config: knobs from advisor.yaml, overridable on the CLI for sweeps.
    f1_threshold, max_rounds = load_grounding_config()
    if args.f1_threshold is not None:
        f1_threshold = args.f1_threshold
    if args.max_rounds is not None:
        max_rounds = args.max_rounds

    device = torch.device("cpu")                  # explainer is tiny + deterministic
    builder = ExplanationBuilder(args.checkpoint, args.dataset, device=device)
    advisor = _build_advisor(args)

    # NodeTable (the resolver's universe) — imported here so a --help doesn't load it.
    from ...evaluation.faithfulness import NodeTable
    from ..gnn.loaders import load_node_meta
    from ...evaluation.run_faithfulness_study import build_or_load_explanation
    table = NodeTable(load_node_meta(args.dataset))

    scenarios = _get_scenarios(args, builder, f1_threshold)
    print("[phase16] {} scenarios | select={} | threshold={:.2f} | max_rounds={} "
          "| model={}".format(len(scenarios), args.select, f1_threshold, max_rounds,
                              getattr(advisor, "model", "?")))

    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "active_grounding")
    os.makedirs(out_dir, exist_ok=True)

    traces: List[Dict[str, Any]] = []
    n_cached = 0
    for i, sc in enumerate(scenarios):
        exp = build_or_load_explanation(builder, args.dataset, sc)
        cpath = _decision_cache_path(out_dir, args.dataset, sc["sample_index"],
                                     sc["target_node"], getattr(advisor, "model", "?"))
        if os.path.exists(cpath):
            with open(cpath) as f:
                trace = json.load(f)
            n_cached += 1
        else:
            trace = run_active_grounding(advisor, exp, table, f1_threshold, max_rounds)
            trace.update({"sample_index": sc["sample_index"],
                          "target_node": sc["target_node"],
                          "tod_band": sc.get("tod_band", "?"),
                          "congestion": sc.get("congestion", "?"),
                          "model": getattr(advisor, "model", "?")})
            with open(cpath, "w") as f:
                json.dump(trace, f, indent=2)

        traces.append(trace)
        # Per-scenario iteration trace to the console (this is what the smoke shows).
        f1s = " -> ".join("{:.3f}".format(c["faithfulness_f1"]) for c in trace["curve"])
        print("  [{}/{}] idx={} {}/{}  F1: {}  [{}] {}".format(
            i + 1, len(scenarios), sc["sample_index"], sc.get("tod_band", "?"),
            sc.get("congestion", "?"), f1s,
            "OK" if trace["reached_threshold"] else "unmet", trace["stop_reason"]))
    if n_cached:
        print("  (reused {} cached scenario traces)".format(n_cached))

    if not traces:
        print("[phase16] no scenarios — nothing to aggregate.")
        return

    # Aggregate + write everything.
    agg = aggregate(traces, max_rounds, f1_threshold)
    missed = most_missed_regions(traces)
    summary = {**agg, "select": args.select, "dataset": args.dataset,
               "model": getattr(advisor, "model", "?"),
               "most_missed_regions": missed}

    csv_path = os.path.join(out_dir, "active_grounding_per_round{}.csv".format(tag))
    jsonl_path = os.path.join(out_dir, "active_grounding_decisions{}.jsonl".format(tag))
    json_path = os.path.join(out_dir, "active_grounding_summary{}.json".format(tag))
    fig_path = os.path.join(PKG_ROOT, "evaluation", "paper",
                            "fig_active_grounding{}.pdf".format(tag))
    tex_path = os.path.join(PKG_ROOT, "evaluation", "paper",
                            "table_active_grounding{}.tex".format(tag))

    write_csv(traces, csv_path)
    write_jsonl(traces, jsonl_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    fig = make_figure(agg, traces, fig_path)
    write_table(agg, tex_path)

    # Console report.
    print("\n" + "=" * 72)
    print("ACTIVE GROUNDING — CONVERGENCE (n={}, threshold {:.2f})".format(
        agg["n_scenarios"], f1_threshold))
    print("=" * 72)
    print("{:>6s}{:>12s}{:>12s}{:>14s}{:>16s}".format(
        "round", "mean F1", "mean recall", "mean halluc", "reached thr %"))
    for row in agg["per_round"]:
        print("{:>6d}{:>12.3f}{:>12.3f}{:>14.3f}{:>16.1f}".format(
            row["round"], row["mean_f1"], row["mean_recall"],
            row["mean_hallucination"], 100.0 * row["frac_reached_threshold"]))
    print("-" * 72)
    print("mean F1 gain (round0 -> final): {:+.3f}".format(agg["mean_f1_gain"]))
    print("mean correction rounds used:    {:.2f}".format(agg["mean_rounds_used"]))
    print("reached threshold overall:      {:.1f}%".format(
        100.0 * agg["frac_reached_overall"]))
    print("stop reasons: {}".format(agg["stop_reasons"]))
    if missed:
        print("most-missed regions at round 0: "
              + ", ".join("{} ({})".format(r, c) for r, c in missed[:6]))
    print("\n[phase16] wrote:")
    for p in [csv_path, jsonl_path, json_path, tex_path] + ([fig] if fig else []):
        print("  " + p)


if __name__ == "__main__":
    main()
