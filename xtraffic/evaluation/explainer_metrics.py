"""Standard explainability metrics for the GNNExplainer layer, vs a random baseline.

Run:
  python -m xtraffic.evaluation.explainer_metrics \
      --checkpoint models/gnn/checkpoints/metr_la_best.pt --dataset metr_la --n 100

We measure, over >=100 sampled predictions, the four metrics reviewers expect,
each defined here in words and in the docstring of its function:

  FIDELITY+  : how much the prediction DEGRADES when we OCCLUDE the explanation's
               top-k nodes. A good explanation removes exactly what mattered, so
               masking it should move the prediction a lot. BIGGER is better.
  FIDELITY-  : how well the prediction is PRESERVED when we KEEP ONLY the top-k
               nodes (occlude everything else). A good explanation is sufficient,
               so keeping it should barely change the prediction. SMALLER is better.
  SPARSITY   : fraction of the graph the explanation uses (k / N). SMALLER = more
               concise. (Fixed by top-k here, reported for completeness.)
  STABILITY  : Jaccard overlap of the top-k node set under small input noise. A
               trustworthy explanation shouldn't flip when speeds wiggle by 1%.
               BIGGER is better.

We compare every metric against a RANDOM-EXPLANATION baseline (random top-k
nodes). If our explainer doesn't beat random by a wide margin on fidelity,
something is wrong and we stop and debug (per CLAUDE.md).

Occlusion here = the same masking the explainer uses: push a node's z-scored
speed toward 0 (the dataset mean). Prediction change is measured in mph.

Outputs a CSV + printed table into evaluation/results/. Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..models.explainer.explain import SPEED_CHANNEL, ExplanationBuilder
from ..models.gnn.loaders import _load_split, build_modality_dict
from ..utils.io_utils import PKG_ROOT


def _predict(builder: ExplanationBuilder, X: torch.Tensor, node: int,
             h: int) -> float:
    """Frozen-model prediction (mph) for one node/horizon, no mask."""
    builder.model.physical_adj = builder.explainer._phys_backup
    mods = build_modality_dict(X, builder.explainer.traffic_channels,
                               builder.explainer.modality_names)
    with torch.no_grad():
        z = builder.model(mods)[0, h - 1, node].item()
    return z * builder.scaler["std"] + builder.scaler["mean"]


def _predict_with_kept(builder: ExplanationBuilder, X: torch.Tensor, node: int,
                       h: int, keep_nodes: List[int], invert: bool) -> float:
    """Prediction when we zero the speed of a node SET.

    invert=False -> OCCLUDE keep_nodes (mask them out): FIDELITY+.
    invert=True  -> keep ONLY keep_nodes (mask out everything else): FIDELITY-.
    """
    N = X.shape[2]
    mask = torch.ones(N)
    if invert:
        mask = torch.zeros(N)
        mask[keep_nodes] = 1.0
        mask[node] = 1.0            # always keep the target's own input
    else:
        mask[keep_nodes] = 0.0
    speed = X[..., SPEED_CHANNEL] * mask[None, None, :]     # [1,T,N]
    other = X[..., SPEED_CHANNEL + 1:]
    Xm = torch.cat([speed.unsqueeze(-1), other], dim=-1)
    mods = build_modality_dict(Xm, builder.explainer.traffic_channels,
                               builder.explainer.modality_names)
    builder.model.physical_adj = builder.explainer._phys_backup
    with torch.no_grad():
        z = builder.model(mods)[0, h - 1, node].item()
    return z * builder.scaler["std"] + builder.scaler["mean"]


def _stability(builder: ExplanationBuilder, X: torch.Tensor, node: int, h: int,
               base_topk: List[int], noise_std: float = 0.02) -> float:
    """Jaccard of top-k under small Gaussian noise on the z-scored input."""
    torch.manual_seed(1234)
    Xn = X + torch.randn_like(X) * noise_std
    node_imp, _, _, _ = builder.explainer.explain_target(Xn, node, h, seed=0)
    new_topk = set(builder._topk_nodes(node_imp, node))
    base = set(base_topk)
    union = base | new_topk
    return len(base & new_topk) / len(union) if union else 0.0


def run(checkpoint: str, dataset: str, n: int, epochs: int, top_k: int) -> Dict:
    device = torch.device("cpu")
    builder = ExplanationBuilder(checkpoint, dataset, device=device,
                                 top_k=top_k, epochs=epochs, confidence_runs=1)
    X_all, _ = _load_split(dataset, "test")        # [S,T,N,C]
    S, T, N, _ = X_all.shape

    # Deterministic diverse sample of (sample_index, target_node) pairs.
    rng = np.random.RandomState(42)
    sample_idxs = rng.choice(S, size=min(n, S), replace=False)

    rows = []
    for si in sample_idxs:
        X = X_all[si: si + 1]
        # Validity + congestion in REAL mph (missing == 0 mph sentinel).
        last_mph = (X[0, -1, :, SPEED_CHANNEL] * builder.scaler["std"]
                    + builder.scaler["mean"]).numpy()
        valid = last_mph > 1.0
        cand = np.where(valid)[0]
        if len(cand) == 0:
            continue
        # Explain the SLOWEST valid sensor — the congested prediction that
        # actually has a spatial cause to explain. Explaining a free-flowing
        # 65 mph sensor is vacuous (no propagation), which washes out fidelity
        # if you sample targets uniformly; explainability is meaningfully
        # evaluated only where there is congestion to explain (this also matches
        # the congestion stratification the Phase 5 study uses).
        node = int(cand[np.argmin(last_mph[cand])])
        h = 6                                       # 30-min horizon

        node_imp, _, _, _ = builder.explainer.explain_target(X, node, h, seed=0)
        topk = builder._topk_nodes(node_imp, node)
        rand_topk = [int(x) for x in rng.choice(
            [c for c in cand if c != node] or [node], size=min(top_k, len(cand)),
            replace=False)]

        base = _predict(builder, X, node, h)
        # FIDELITY+ : occlude explanation top-k vs occlude random top-k
        fplus = abs(_predict_with_kept(builder, X, node, h, topk, invert=False) - base)
        fplus_rand = abs(_predict_with_kept(builder, X, node, h, rand_topk, invert=False) - base)
        # FIDELITY- : keep only top-k vs keep only random
        fminus = abs(_predict_with_kept(builder, X, node, h, topk, invert=True) - base)
        fminus_rand = abs(_predict_with_kept(builder, X, node, h, rand_topk, invert=True) - base)

        stab = _stability(builder, X, node, h, topk)
        rows.append({
            "sample": int(si), "node": node,
            "fidelity_plus": fplus, "fidelity_plus_random": fplus_rand,
            "fidelity_minus": fminus, "fidelity_minus_random": fminus_rand,
            "stability": stab, "sparsity": len(topk) / N,
        })

    return _summarize(rows, dataset, top_k)


def _mean(rows: List[Dict], key: str) -> Tuple[float, float]:
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.std())


def _summarize(rows: List[Dict], dataset: str, top_k: int) -> Dict:
    res_dir = os.path.join(PKG_ROOT, "evaluation", "results")
    os.makedirs(res_dir, exist_ok=True)
    # Per-sample CSV (auditable).
    csv_path = os.path.join(res_dir, f"explainer_metrics_{dataset}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "dataset": dataset, "n_samples": len(rows), "top_k": top_k,
        "fidelity_plus": _mean(rows, "fidelity_plus"),
        "fidelity_plus_random": _mean(rows, "fidelity_plus_random"),
        "fidelity_minus": _mean(rows, "fidelity_minus"),
        "fidelity_minus_random": _mean(rows, "fidelity_minus_random"),
        "stability": _mean(rows, "stability"),
        "sparsity": _mean(rows, "sparsity"),
    }
    with open(os.path.join(res_dir, f"explainer_metrics_{dataset}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Explainability metrics ({dataset}, n={len(rows)}, k={top_k}) ===")
    print(f"{'metric':<22}{'ours':>16}{'random':>16}")
    def line(name, ours_k, rand_k=None):
        om, os_ = summary[ours_k]
        if rand_k:
            rm, rs = summary[rand_k]
            print(f"{name:<22}{om:>9.3f}±{os_:<5.2f}{rm:>9.3f}±{rs:<5.2f}")
        else:
            print(f"{name:<22}{om:>9.3f}±{os_:<5.2f}")
    line("Fidelity+ (mph, big=good)", "fidelity_plus", "fidelity_plus_random")
    line("Fidelity- (mph, small=good)", "fidelity_minus", "fidelity_minus_random")
    line("Stability (Jaccard)", "stability")
    line("Sparsity (k/N)", "sparsity")
    fp = summary["fidelity_plus"][0]; fpr = summary["fidelity_plus_random"][0]
    verdict = "PASS: beats random" if fp > 1.5 * max(fpr, 1e-6) else "CHECK: not clearly > random"
    print(f"\nFidelity+ vs random: {verdict} ({fp:.3f} vs {fpr:.3f})")
    print(f"[metrics] CSV -> {csv_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/gnn/checkpoints/metr_la_best.pt")
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=8)
    args = ap.parse_args()
    run(args.checkpoint, args.dataset, args.n, args.epochs, args.top_k)


if __name__ == "__main__":
    main()
