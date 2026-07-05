"""Phase 8 — build the human-study scenario set (offline, one time).

Produces everything the evaluation app needs, written to
evaluation/results/human_study/:

  scenarios.json   — 24 incident-response scenarios. Each carries:
                       * the shared candidate intervention list + ground truth
                         (from simulate.InterventionSimulator);
                       * three PRESENTATION BUNDLES (RAW / XAI / XTRAFFIC) — the
                         exact information an evaluator sees under each condition.
  assignment.json  — a balanced Latin-square mapping evaluator -> (scenario ->
                       condition), so every evaluator sees each scenario once and
                       conditions are balanced across evaluators.
  figures/         — one explanation PNG per scenario (used by the XAI/XTRAFFIC
                       conditions).

Why bundles are pre-generated: the GNN + explainer + Ollama are heavy and
non-deterministic in wall-clock; freezing the content once makes each evaluator's
session identical and reproducible, and lets a professor run the app with no ML
dependencies installed (CLAUDE.md: reproducibility).

Run (needs the trained checkpoint + a local Ollama with the advisor model):
  python -m xtraffic.evaluation.human_study.scenarios
  python -m xtraffic.evaluation.human_study.scenarios --no-llm   # skip advisories
  python -m xtraffic.evaluation.human_study.scenarios --evaluators 4

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from ...models.explainer.explain import ExplanationBuilder
from ...models.explainer.scenarios import SPEED_CHANNEL, TOD_CHANNEL, _tod_to_clock, load_window
from ...models.gnn.loaders import _load_split
from ...utils.io_utils import PKG_ROOT
from .simulate import InterventionSimulator

OUT_DIR = os.path.join(PKG_ROOT, "evaluation", "results", "human_study")


def _load_cfg() -> Dict[str, Any]:
    with open(os.path.join(PKG_ROOT, "configs", "human_study.yaml")) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Stratified scenario sampling (deterministic).
# Mirrors run_faithfulness_study._window_stats/sample_scenarios but uses this
# phase's own 4 tod-bands x 3 congestion x per_cell grid. Target node = slowest
# valid sensor in the window (the segment a planner most wants to act on). We
# compute validity/means in REAL mph (missing = ~0 mph) — the Phase-3 z-space bug.
# ---------------------------------------------------------------------------
def _window_stats(dataset: str, scaler: Dict[str, float]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, _ = _load_split(dataset, "test")                 # [S, T, N, 2]
    S, T, N, _ = X.shape
    tod0 = X[:, 0, 0, TOD_CHANNEL].numpy()              # clock at window start [S]

    speed_mph = X[..., SPEED_CHANNEL].numpy() * scaler["std"] + scaler["mean"]  # [S,T,N]
    valid = speed_mph > 1.0
    flat_valid = valid.reshape(S, -1)
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(
            flat_valid.any(1),
            (speed_mph * valid).reshape(S, -1).sum(1) / np.clip(flat_valid.sum(1), 1, None),
            np.inf)                                     # [S]
    node_mean = speed_mph.mean(1)                       # [S, N]
    node_valid = valid.any(1)                           # [S, N]
    slowest_node = np.where(node_valid, node_mean, np.inf).argmin(1)  # [S]
    return tod0, mean_speed, slowest_node


def sample_scenarios(dataset: str, scaler: Dict[str, float],
                     cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sc_cfg = cfg["scenarios"]
    tod0, mean_speed, slowest_node = _window_stats(dataset, scaler)
    finite = np.isfinite(mean_speed)
    q33, q66 = np.quantile(mean_speed[finite], [1 / 3, 2 / 3])

    def congestion_of(s: float) -> Optional[str]:
        if not np.isfinite(s):
            return None
        return "high" if s <= q33 else ("medium" if s <= q66 else "low")

    rng = random.Random(cfg["seed"])
    scenarios: List[Dict[str, Any]] = []
    for band, lo, hi in sc_cfg["tod_bands"]:
        for level in sc_cfg["congestion_levels"]:
            cell = [i for i in range(len(mean_speed))
                    if finite[i] and lo <= tod0[i] < hi and congestion_of(mean_speed[i]) == level]
            rng.shuffle(cell)                           # seeded -> reproducible
            for k, idx in enumerate(cell[:sc_cfg["per_cell"]]):
                scenarios.append({
                    "scenario_id": "s{:02d}".format(len(scenarios)),
                    "sample_index": int(idx),
                    "target_node": int(slowest_node[idx]),
                    "tod_band": band,
                    "congestion": level,
                    "mean_speed_mph": float(round(mean_speed[idx], 2)),
                    "timestamp": "test#{} @ {}".format(idx, _tod_to_clock(float(tod0[idx]))),
                })
    return scenarios


# ---------------------------------------------------------------------------
# Latin-square condition assignment.
# ---------------------------------------------------------------------------
def latin_square_assignment(scenario_ids: List[str], conditions: List[str],
                            n_evaluators: int) -> Dict[str, Dict[str, str]]:
    """Balanced assignment: evaluator -> {scenario_id: condition}.

    Construction: a scenario at position p, for evaluator e, gets condition
    conditions[(p + e) % C]. Rotating by e is a cyclic Latin square, so:
      * each evaluator sees every scenario exactly ONCE (we iterate all scenarios);
      * as e varies, a fixed scenario cycles through all C conditions, so
        conditions are balanced across evaluators;
      * with n_evaluators a multiple of C (=3), every scenario is shown under each
        condition equally often across the panel.
    We recommend n_evaluators in {3, 6} for perfect balance; other counts still
    work but balance is only approximate (reported by print_balance()).
    """
    C = len(conditions)
    return {
        "eval{}".format(e): {
            sid: conditions[(p + e) % C] for p, sid in enumerate(scenario_ids)
        }
        for e in range(n_evaluators)
    }


def print_balance(assignment: Dict[str, Dict[str, str]], conditions: List[str]) -> None:
    counts = {c: 0 for c in conditions}
    for per_scn in assignment.values():
        for cond in per_scn.values():
            counts[cond] += 1
    print("[assignment] condition counts across all evaluator-scenario cells:", counts)


# ---------------------------------------------------------------------------
# Presentation bundles (RAW / XAI / XTRAFFIC).
# ---------------------------------------------------------------------------
def _raw_bundle(exp: Dict[str, Any], candidate_names: List[str]) -> Dict[str, Any]:
    """RAW condition: prediction numbers only, plus the choice list (control)."""
    p = exp["prediction"]
    return {
        "prediction": {
            "location": p["node_name"],
            "current_speed_mph": p["current_speed_mph"],
            "predicted_speed_mph": p["predicted_speed_mph"],
            "horizon_minutes": p["horizon_minutes"],
        },
        "candidates": candidate_names,
    }


def _xai_bundle(raw: Dict[str, Any], exp: Dict[str, Any], figure_rel: str) -> Dict[str, Any]:
    """XAI condition: RAW + the mathematical explanation (importance table,
    propagation path/lag, confidence) + the rendered figure. Current SOTA."""
    return {
        **raw,
        "figure": figure_rel,
        "importance_table": [
            {"location": n["node_name"], "importance": n["importance"],
             "current_speed_mph": n["current_speed_mph"]}
            for n in exp["top_nodes"]
        ],
        "propagation_lag_minutes": exp["propagation_lag_minutes"],
        "explanation_confidence": exp["explanation_confidence"],
    }


def _xtraffic_bundle(xai: Dict[str, Any], advisory: Dict[str, Any]) -> Dict[str, Any]:
    """XTRAFFIC condition: XAI + the LLM advisory (plain-language reasoning +
    concrete, grounded recommendations)."""
    return {
        **xai,
        "advisory": {
            "reasoning": advisory.get("reasoning", ""),
            "recommendations": advisory.get("recommendations", []),
        },
    }


# ---------------------------------------------------------------------------
# Main build.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="skip Ollama; XTRAFFIC bundles reuse XAI + an empty "
                         "advisory (for building scenarios when Ollama is absent).")
    ap.add_argument("--evaluators", type=int, default=6,
                    help="panel size for the Latin square (3 or 6 = perfect balance).")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap scenarios (smoke test only; breaks stratification).")
    args = ap.parse_args()

    cfg = _load_cfg()
    dataset = cfg["dataset"]
    ckpt = cfg["checkpoint"]
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device("cpu")                        # tiny + deterministic
    sim = InterventionSimulator(ckpt, dataset, cfg["simulation"], device=device)
    builder = ExplanationBuilder(ckpt, dataset, device=device)

    advisor = None
    if not args.no_llm:
        from ...models.advisor.advisor import Advisor
        advisor = Advisor(cfg["advisor_city"])

    scenarios = sample_scenarios(dataset, builder.scaler, cfg)
    if args.limit:
        scenarios = scenarios[:args.limit]
    print("Sampled {} scenarios.".format(len(scenarios)))

    # Lazy import here so --no-llm builds don't require matplotlib until needed.
    from ...evaluation.visualize_explanation import visualize

    built: List[Dict[str, Any]] = []
    for sc in scenarios:
        X = load_window(dataset, sc["sample_index"])    # [1, T, N, C]

        # (1) ground truth: simulate every candidate intervention on this window.
        sim_out = sim.score_interventions(X, sc["target_node"])
        candidate_names = [c["intervention"] for c in sim_out["candidates"]]

        # (2) explanation (Layer 2) — shared by XAI + XTRAFFIC.
        exp = builder.explain_prediction(
            X, target_node=sc["target_node"], horizon_step=cfg["simulation"]["horizon_step"],
            timestamp=sc["timestamp"])
        exp["meta"]["city"] = cfg["advisor_city"]
        exp["meta"]["scenario"] = sc["scenario_id"]

        # (3) figure (PNG for the web app) — save the explanation JSON first so the
        # existing renderer can read it, then render.
        exp_path = os.path.join(fig_dir, "{}.json".format(sc["scenario_id"]))
        with open(exp_path, "w") as f:
            json.dump(exp, f, indent=2)
        fig_png = os.path.join(fig_dir, "{}.png".format(sc["scenario_id"]))
        visualize(exp_path, dataset, out_path=fig_png)
        figure_rel = "figures/{}.png".format(sc["scenario_id"])

        # (4) bundles.
        raw = _raw_bundle(exp, candidate_names)
        xai = _xai_bundle(raw, exp, figure_rel)
        if advisor is not None:
            advisory = advisor.advise(exp)["advisory"]
        else:
            advisory = {"reasoning": "", "recommendations": []}
        xtraffic = _xtraffic_bundle(xai, advisory)

        built.append({
            **sc,
            "ground_truth_intervention": sim_out["ground_truth_intervention"],
            "no_action_is_best": sim_out["no_action_is_best"],
            "simulation": sim_out["candidates"],
            "conditions": {"RAW": raw, "XAI": xai, "XTRAFFIC": xtraffic},
        })
        print("  [{}] {} cong={} gt={}".format(
            sc["scenario_id"], sc["tod_band"], sc["congestion"],
            sim_out["ground_truth_intervention"]))

    with open(os.path.join(OUT_DIR, "scenarios.json"), "w") as f:
        json.dump({"dataset": dataset, "conditions": cfg["conditions"],
                   "likert_max": cfg["likert_max"], "scenarios": built}, f, indent=2)

    assignment = latin_square_assignment(
        [s["scenario_id"] for s in built], cfg["conditions"], args.evaluators)
    with open(os.path.join(OUT_DIR, "assignment.json"), "w") as f:
        json.dump(assignment, f, indent=2)
    print_balance(assignment, cfg["conditions"])

    n_degenerate = sum(1 for s in built if s["no_action_is_best"])
    print("\nWrote {} scenarios to {}".format(len(built), OUT_DIR))
    if n_degenerate:
        print("NOTE: {} scenario(s) have no_action as ground truth (no candidate "
              "helps) — consider excluding in analyze.py.".format(n_degenerate))


if __name__ == "__main__":
    main()
