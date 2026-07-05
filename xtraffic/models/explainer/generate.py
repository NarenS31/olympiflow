"""Generate explanations for the diverse verification scenarios and save JSON.

Run:
  python -m xtraffic.models.explainer.generate \
      --checkpoint models/gnn/checkpoints/metr_la_best.pt --dataset metr_la

Writes one <scenario>.json per scenario to evaluation/results/explanations/ and
prints a short human summary so you can eyeball whether the explanations are
SEMANTICALLY sane (rush-hour downtown congestion should be explained by upstream
downtown sensors, not random ones across the city — read them, per the gate).

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ...utils.io_utils import PKG_ROOT
from .explain import ExplanationBuilder
from .scenarios import load_window, select_scenarios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/gnn/checkpoints/metr_la_best.pt")
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--epochs", type=int, default=200, help="explainer mask-opt steps")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--conf", type=int, default=5, help="explainer reruns for confidence")
    args = ap.parse_args()

    device = torch.device("cpu")  # explainer is tiny; CPU keeps it deterministic
    builder = ExplanationBuilder(args.checkpoint, args.dataset, device=device,
                                 top_k=args.top_k, epochs=args.epochs,
                                 confidence_runs=args.conf)
    scenarios = select_scenarios(args.dataset, builder.scaler)

    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "explanations")
    os.makedirs(out_dir, exist_ok=True)

    for sc in scenarios:
        X = load_window(args.dataset, sc["sample_index"])          # [1,T,N,C]
        exp = builder.explain_prediction(
            X, target_node=sc["target_node"], horizon_step=sc["horizon_step"],
            timestamp=sc["timestamp"])
        exp["meta"]["scenario"] = sc["name"]
        exp["meta"]["scenario_note"] = sc["note"]

        path = os.path.join(out_dir, f"{sc['name']}.json")
        with open(path, "w") as f:
            json.dump(exp, f, indent=2)

        p = exp["prediction"]
        print(f"\n=== {sc['name']}  ({sc['timestamp']}) ===")
        print(f"  target: {p['node_name']}")
        print(f"  now {p['current_speed_mph']} mph -> predicted "
              f"{p['predicted_speed_mph']} mph at {p['horizon_minutes']} min")
        print(f"  confidence {exp['explanation_confidence']}, "
              f"lag {exp['propagation_lag_minutes']} min, "
              f"path len {len(exp['propagation_path'])}")
        print("  top influencing sensors:")
        for n in exp["top_nodes"][:4]:
            print(f"    - {n['node_name']}  imp={n['importance']}  "
                  f"({n['current_speed_mph']} mph)")
        print(f"  -> {path}")

    print(f"\n[generate] {len(scenarios)} explanations written to {out_dir}")


if __name__ == "__main__":
    main()
