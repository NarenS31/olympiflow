"""Phase 6 — fusion vs traffic-only comparison table (Contribution #1's fusion result).

Run (after both checkpoints exist):
    python -m xtraffic.evaluation.fusion_comparison

Evaluates the two METR-LA checkpoints on the SAME test split and lays them side by
side at 15 / 30 / 60 min:
    * traffic-only : models/gnn/checkpoints/metr_la_best.pt        (Phase 2 config)
    * fusion       : models/gnn/checkpoints/metr_la_fusion_best.pt (weather+events+transit)
Also pulls the learned per-modality GATE values from the last row of the fusion
training log (train_metr_la_fusion.csv) — how much the model ended up trusting each
feed is itself a paper figure. Writes CSV + JSON to evaluation/results/fusion/.

Both checkpoints must have been trained first (train_metr_la.yaml and
train_metr_la_fusion.yaml). Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Optional

from ..models.gnn.evaluate import evaluate
from ..utils.io_utils import PKG_ROOT

HORIZONS = [("15min", 3), ("30min", 6), ("60min", 12)]


def _last_gate_row(run_name: str) -> Optional[Dict[str, float]]:
    """Final-epoch modality gate values from the training CSV, if present."""
    path = os.path.join(PKG_ROOT, "evaluation", "results", f"train_{run_name}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]
    return {k[len("gate_"):]: float(v) for k, v in last.items()
            if k.startswith("gate_") and v not in ("", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--traffic-ckpt", default="models/gnn/checkpoints/metr_la_best.pt")
    ap.add_argument("--fusion-ckpt", default="models/gnn/checkpoints/metr_la_fusion_best.pt")
    args = ap.parse_args()

    for label, ck in (("traffic-only", args.traffic_ckpt), ("fusion", args.fusion_ckpt)):
        full = ck if os.path.isabs(ck) else os.path.join(PKG_ROOT, ck)
        if not os.path.exists(full):
            raise SystemExit(f"[fusion_comparison] {label} checkpoint missing: {full}\n"
                             f"Train it first (see the fusion Colab notebook).")

    print("=== evaluating traffic-only ===")
    traffic = evaluate(args.traffic_ckpt, args.dataset)
    print("=== evaluating fusion ===")
    fusion = evaluate(args.fusion_ckpt, args.dataset)

    # --- Build the side-by-side table (rows = horizons). ---
    rows = []
    for lab, _step in HORIZONS:
        t = traffic["per_horizon"][lab]; f = fusion["per_horizon"][lab]
        rows.append({
            "horizon": lab,
            "traffic_mae": round(t["mae"], 3), "fusion_mae": round(f["mae"], 3),
            "delta_mae": round(f["mae"] - t["mae"], 3),
            "traffic_rmse": round(t["rmse"], 3), "fusion_rmse": round(f["rmse"], 3),
            "traffic_mape": round(t["mape"], 2), "fusion_mape": round(f["mape"], 2),
        })

    gates = _last_gate_row("metr_la_fusion")
    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "fusion")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "comparison.json"), "w") as fp:
        json.dump({"dataset": args.dataset, "rows": rows,
                   "final_modality_gates": gates,
                   "traffic_overall": traffic["overall"],
                   "fusion_overall": fusion["overall"]}, fp, indent=2)

    # --- Print it (delta<0 = fusion is better; MAE is lower-is-better). ---
    print("\n===== FUSION vs TRAFFIC-ONLY (METR-LA test, MAE mph) =====")
    print(f"  {'horizon':>8} {'traffic':>9} {'fusion':>9} {'delta':>8}")
    for r in rows:
        better = "  <- fusion better" if r["delta_mae"] < 0 else ""
        print(f"  {r['horizon']:>8} {r['traffic_mae']:>9} {r['fusion_mae']:>9} "
              f"{r['delta_mae']:>+8}{better}")
    if gates:
        print("\n  final learned modality gates (0=ignored, 1=fully trusted):")
        for name, g in gates.items():
            print(f"    {name:>10}: {g:.3f}")
    print(f"\n[fusion_comparison] saved -> {out_dir}/comparison.csv")


if __name__ == "__main__":
    main()
