"""Non-learned reference baselines: Historical Average and Linear Regression.

WHY these exist: a fancy ST-GNN is only interesting if it clearly beats trivial
predictors on the SAME splits with the SAME masked metrics. These two are the
standard traffic-forecasting sanity floors.

  - Historical Average (HA): predict every future step as the mean speed of that
    (node, time-of-day) bucket, learned from TRAIN only. Captures daily rhythm,
    ignores current conditions.
  - Linear Regression (LR): per-node ridge regression from the 12-step input speed
    window to each of the 12 output steps. A cheap learned baseline.

Run:  python -m xtraffic.models.gnn.baselines --config configs/train_metr_la.yaml
Writes a JSON results row per baseline to evaluation/results/. Python 3.9.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from ...utils.io_utils import PKG_ROOT, units_for_dataset
from ...utils.metrics import masked_metrics, per_horizon_metrics
from .loaders import _load_split, load_scaler
from .train import load_config


def _tod_bucket(X_tod: np.ndarray, n_buckets: int = 288) -> np.ndarray:
    """Map the time-of-day channel (0..1) to a discrete daily bucket (5-min => 288/day)."""
    return np.clip((X_tod * n_buckets).astype(int), 0, n_buckets - 1)


def historical_average(dataset: str, scaler: Dict[str, float]):
    """Predict each horizon from the (node, time-of-day) train mean speed."""
    Xtr, Ytr = _load_split(dataset, "train")   # X:[S,12,N,2] Y:[S,12,N]
    Xte, Yte = _load_split(dataset, "test")
    N = Ytr.shape[2]
    n_buckets = 288

    # Build the lookup table: mean z-scored speed per (bucket, node), TRAIN only,
    # masking the missing sentinel so it doesn't pollute the average.
    speed = Xtr[..., 0].numpy()                 # [S,12,N] z-scored speed
    tod = Xtr[..., 1].numpy()                   # [S,12,N] time-of-day
    real = speed * scaler["std"] + scaler["mean"]
    valid = np.abs(real) > 1e-3
    buckets = _tod_bucket(tod)                  # [S,12,N]

    table = np.zeros((n_buckets, N), dtype=np.float64)
    counts = np.zeros((n_buckets, N), dtype=np.float64)
    np.add.at(table, (buckets, np.broadcast_to(np.arange(N), speed.shape)),
              np.where(valid, speed, 0.0))
    np.add.at(counts, (buckets, np.broadcast_to(np.arange(N), speed.shape)),
              valid.astype(np.float64))
    table = np.divide(table, counts, out=np.zeros_like(table), where=counts > 0)

    # Predict test: each output step uses the bucket of the last input step,
    # advanced by the horizon (a step is 5 min; the daily bucket wraps mod 288).
    last_bucket = _tod_bucket(Xte[..., 1].numpy())[:, -1, :]   # [S_te, N]
    T_out = Yte.shape[1]
    pred = np.zeros_like(Yte.numpy())            # [S_te,12,N]
    for h in range(T_out):
        b = (last_bucket + (h + 1)) % n_buckets  # [S_te, N]
        pred[:, h, :] = table[b, np.arange(N)]
    return torch.from_numpy(pred).float(), Yte


def linear_regression(dataset: str, scaler: Dict[str, float], ridge: float = 1.0):
    """Per-node closed-form ridge regression: 12 input speeds -> 12 output speeds."""
    Xtr, Ytr = _load_split(dataset, "train")
    Xte, Yte = _load_split(dataset, "test")
    N = Ytr.shape[2]
    T_in, T_out = Xtr.shape[1], Ytr.shape[1]

    pred = torch.zeros_like(Yte)                 # [S_te,12,N]
    I = np.eye(T_in + 1)                         # +1 for the bias column
    for n in range(N):
        A = Xtr[:, :, n, 0].numpy()              # [S_tr, 12] train input speeds
        A = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)  # bias
        B = Ytr[:, :, n].numpy()                 # [S_tr, 12] targets
        # Ridge closed form: W = (A^T A + ridge I)^-1 A^T B  -> [13, 12]
        W = np.linalg.solve(A.T @ A + ridge * I, A.T @ B)
        Ate = Xte[:, :, n, 0].numpy()
        Ate = np.concatenate([Ate, np.ones((Ate.shape[0], 1))], axis=1)
        pred[:, :, n] = torch.from_numpy(Ate @ W).float()
    return pred, Yte


def persistence(dataset: str, scaler: Dict[str, float]):
    """Predict every future step as the LAST observed value. -> (pred, true).

    ADDED in Phase 19, and it is the baseline that matters most on a smooth signal.
    Historical Average and Linear Regression are the standard TRAFFIC floors, but
    bus voltage moves slowly and continuously, so "whatever it is now, it will
    still be that in 30 minutes" is a genuinely strong predictor. If the ST-GNN
    cannot clearly beat persistence on the power grid, the domain is too easy to
    support any claim, and we would rather find that out here than in review.

    Applies to traffic datasets too, where it is a well-known (and much weaker)
    reference point. Purely additive: no existing baseline changed.
    """
    Xte, Yte = _load_split(dataset, "test")     # X:[S,12,N,2] Y:[S,12,N]
    last = Xte[:, -1, :, 0]                     # [S, N] final z-scored input step
    pred = last.unsqueeze(1).repeat(1, Yte.shape[1], 1)   # [S, 12, N] held flat
    return pred, Yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    scaler = load_scaler(dataset)
    res_dir = os.path.join(PKG_ROOT, cfg["results_dir"])
    os.makedirs(res_dir, exist_ok=True)

    # Phase 19: label + precision follow the dataset's units. On the power grid a
    # per-unit MAE of 0.00209 printed at 3 d.p. reads "0.002", which hides the very
    # differences between baselines we are trying to see. Defaults to mph/3 d.p.
    units = units_for_dataset(dataset)
    prec = 3 if units == "mph" else 5

    for name, fn in [("historical_average", historical_average),
                     ("linear_regression", linear_regression),
                     ("persistence", persistence)]:
        pred, true = fn(dataset, scaler)
        overall = masked_metrics(pred, true, scaler)
        horizon = per_horizon_metrics(pred, true, scaler, tuple(cfg["horizons_steps"]))
        print(f"[baseline] {name:<20}: MAE={overall['mae']:.{prec}f} {units}  "
              f"MAE@30min={horizon['30min']['mae']:.{prec}f} {units}")
        out = {"model": name, "dataset": dataset, "units": units, "overall": overall,
               "per_horizon": horizon}
        with open(os.path.join(res_dir, f"baseline_{name}_{dataset}.json"), "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
