"""Phase 8 — synthetic scenario set for piloting the app WITHOUT the ML stack.

The real scenarios.json is built by scenarios.py and needs the trained checkpoint
+ Ollama. That content is owed pending the Colab run. So a professor (or you) can
still exercise the whole app -> JSONL -> analyze.py loop today, this builds a small
FAKE scenarios.json + assignment.json with the exact same SHAPES the real builder
emits (no figures — the app degrades to the importance table). It is clearly
labelled demo data and must NOT be used for the paper.

Run:
  python -m xtraffic.evaluation.human_study.demo_data          # 6 demo scenarios
  python -m xtraffic.evaluation.human_study.demo_data --n 24

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import random

from .scenarios import OUT_DIR, latin_square_assignment

CONDITIONS = ["RAW", "XAI", "XTRAFFIC"]
INTERVENTIONS = ["no_action", "signal_retiming", "ramp_metering", "reroute", "transit_surge"]
BANDS = ["am_rush", "midday", "pm_rush", "evening"]
LEVELS = ["low", "medium", "high"]


def _bundle(loc: str, cur: float, pred: float):
    raw = {"prediction": {"location": loc, "current_speed_mph": cur,
                          "predicted_speed_mph": pred, "horizon_minutes": 30},
           "candidates": INTERVENTIONS}
    xai = {**raw, "figure": "", "propagation_lag_minutes": 5.0,
           "explanation_confidence": 0.8,
           "importance_table": [
               {"location": "Upstream corridor A", "importance": 0.42, "current_speed_mph": 18.0},
               {"location": "Upstream corridor B", "importance": 0.31, "current_speed_mph": 22.0}]}
    xtraffic = {**xai, "advisory": {
        "reasoning": "Congestion at {} is driven by slowdowns on the upstream "
                     "corridors; acting there 30 min ahead relieves the target.".format(loc),
        "recommendations": [
            {"action": "signal_retiming", "location": "Upstream corridor A",
             "time_window_minutes": 15, "expected_effect": "raise throughput ~8 mph"}]}}
    return {"RAW": raw, "XAI": xai, "XTRAFFIC": xtraffic}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--evaluators", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = []
    for i in range(args.n):
        cur = round(rng.uniform(12, 30), 1)
        pred = round(cur - rng.uniform(2, 8), 1)
        gt = rng.choice(INTERVENTIONS[1:])          # a real action is best
        scenarios.append({
            "scenario_id": "s{:02d}".format(i),
            "sample_index": i, "target_node": i,
            "tod_band": BANDS[i % len(BANDS)], "congestion": LEVELS[i % len(LEVELS)],
            "mean_speed_mph": cur, "timestamp": "demo#{}".format(i),
            "ground_truth_intervention": gt, "no_action_is_best": False,
            "simulation": [{"intervention": iv, "network_delay": round(rng.uniform(50, 200), 1),
                            "target_speed_mph": pred, "n_nodes_affected": 3} for iv in INTERVENTIONS],
            "conditions": _bundle("Demo location {}".format(i), cur, pred),
        })

    with open(os.path.join(OUT_DIR, "scenarios.json"), "w") as f:
        json.dump({"dataset": "DEMO", "conditions": CONDITIONS, "likert_max": 7,
                   "scenarios": scenarios}, f, indent=2)
    assignment = latin_square_assignment(
        [s["scenario_id"] for s in scenarios], CONDITIONS, args.evaluators)
    with open(os.path.join(OUT_DIR, "assignment.json"), "w") as f:
        json.dump(assignment, f, indent=2)
    print("Wrote {} DEMO scenarios + {} evaluators to {}".format(
        len(scenarios), args.evaluators, OUT_DIR))
    print("This is DEMO data (labelled dataset='DEMO') — not for the paper.")


if __name__ == "__main__":
    main()
