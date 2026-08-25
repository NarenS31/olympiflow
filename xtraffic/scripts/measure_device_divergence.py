"""Does the explainer's answer depend on which device computed it?

    python -m xtraffic.scripts.measure_device_divergence

WHY THIS EXISTS AS AN EXPERIMENT AND NOT A COMMENT
    The influence-graph analysis is entirely about WHICH sensors the model leans
    on. Before running it we noticed MPS was 3x faster than CPU and asked whether
    it could be used. It cannot, and the reason is a measurement worth keeping:
    the two devices agree closely on importance MAGNITUDES and disagree sharply
    on importance RANKS. Reported as evidence about the explainer's optimum --
    a solution that reorders under float-level perturbation is a shallow one --
    rather than filed away as a device note.

    docs/REPOSITORY_AUDIT.md 5.7 already states that cross-device bitwise
    identity is impossible. This quantifies what that costs on the quantity the
    project actually reads off an explanation.

No LLM calls. Writes one run directory. Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np

from ..reproducibility import run_dir, seeds
from ..utils.io_utils import PKG_ROOT, processed_dir

REPO_ROOT = os.path.dirname(PKG_ROOT)
CKPT = "models/gnn/checkpoints/metr_la_best.pt"
EXPERIMENT = "device_divergence"

# Fixed probe set: three (window, target) pairs spanning the test split. Chosen
# before any result was seen, and listed here so the choice is inspectable.
CASES = [(100, 50), (2500, 120), (4200, 7)]


def _topk(v: np.ndarray, target: int, k: int) -> set:
    return set([int(i) for i in np.argsort(-v) if i != target][:k])


def main(argv: Optional[List[str]] = None) -> int:
    import torch

    from ..models.explainer.explain import ExplanationBuilder

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args(argv)

    seed_rec = seeds.set_all_seeds(42)
    X = np.load(os.path.join(processed_dir("metr_la"), "test.npz"))["X"]

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    cfg = {"experiment": EXPERIMENT, "checkpoint": CKPT, "cases": CASES,
           "epochs": args.epochs, "devices": devices, "explainer_seed": 0,
           "horizon_step": 6}

    run = run_dir.RunDir.create(
        REPO_ROOT, EXPERIMENT, cfg,
        artifacts={"checkpoint": os.path.join(PKG_ROOT, CKPT)},
        notes="CPU vs MPS agreement on explainer node importance. No LLM calls.")
    run.write_json("seeds.json", seed_rec)
    run.start()

    builders = {d: ExplanationBuilder(CKPT, "metr_la", device=torch.device(d),
                                      epochs=args.epochs, confidence_runs=0)
                for d in devices}

    rows = []
    for (wi, tgt) in CASES:
        xi = torch.from_numpy(X[wi:wi + 1]).float()
        imp = {}
        for d, b in builders.items():
            v, _, _, _ = b.explainer.explain_target(xi, tgt, 6, seed=0)
            imp[d] = np.asarray(v, dtype=np.float64)
        rec = {"window": wi, "target": tgt}
        if "mps" in imp:
            a, b_ = imp["cpu"], imp["mps"]
            ra = np.argsort(np.argsort(-a))
            rb = np.argsort(np.argsort(-b_))
            rec.update({
                "pearson_magnitude": round(float(np.corrcoef(a, b_)[0, 1]), 4),
                "spearman_rank": round(float(np.corrcoef(ra, rb)[0, 1]), 4),
                "max_abs_diff": float(np.abs(a - b_).max()),
                "top8_jaccard": round(len(_topk(a, tgt, 8) & _topk(b_, tgt, 8))
                                      / len(_topk(a, tgt, 8) | _topk(b_, tgt, 8)), 4),
                "top20_jaccard": round(len(_topk(a, tgt, 20) & _topk(b_, tgt, 20))
                                       / len(_topk(a, tgt, 20) | _topk(b_, tgt, 20)), 4),
            })
        # same-device, same-seed determinism
        for d, b in builders.items():
            v2, _, _, _ = b.explainer.explain_target(xi, tgt, 6, seed=0)
            rec["{}_repeat_identical".format(d)] = bool(
                np.array_equal(np.asarray(v2, dtype=np.float64), imp[d]))
        rows.append(rec)
        print(rec)

    summary = {"cases": rows, "devices": devices}
    if "mps" in devices:
        summary["means"] = {
            k: round(float(np.mean([r[k] for r in rows])), 4)
            for k in ("pearson_magnitude", "spearman_rank", "top8_jaccard",
                      "top20_jaccard")}
    run.write_json("device_divergence.json", summary)
    run.complete(**(summary.get("means") or {}))
    print("\n-> {}".format(run.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
