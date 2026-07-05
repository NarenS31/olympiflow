"""Masked traffic-forecasting metrics, computed in REAL mph units.

WHY masking: METR-LA / PEMS-BAY encode a missing sensor reading as the real value
0 mph (a sensor is never genuinely 0 mph on a moving freeway — it means "no data").
After z-scoring, that 0 becomes the sentinel value (0 - mean)/std. If we scored
predictions on those positions we would be training the model to reproduce a
data-collection artifact, not traffic. So every metric ignores positions whose
REAL target is (approximately) 0.

WHY real units: published Graph WaveNet numbers (MAE ~3.0 mph at 30 min on
METR-LA) are in mph. We inverse-transform predictions and targets with the train
scaler before measuring, so our numbers are directly comparable.

All functions take z-scored tensors plus the scaler and return Python floats.
Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Dict

import torch


def _to_real(z: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return z * std + mean  # inverse z-score -> mph


def _mask_from_target(real_target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # 1.0 where the real target is a genuine reading, 0.0 where it is the missing sentinel.
    mask = (real_target.abs() > eps).float()
    # Avoid a divide-by-zero if an entire slice is missing.
    denom = mask.mean().clamp(min=1e-6)
    return mask / denom  # normalised so masked-mean == mean over valid entries


def masked_mae_loss(pred_z: torch.Tensor, true_z: torch.Tensor,
                    scaler: Dict[str, float]) -> torch.Tensor:
    """Masked MAE in mph — this is the TRAINING loss (differentiable, keeps grad)."""
    pred = _to_real(pred_z, scaler["mean"], scaler["std"])
    true = _to_real(true_z, scaler["mean"], scaler["std"])
    mask = _mask_from_target(true)
    loss = torch.abs(pred - true) * mask
    return loss.mean()


@torch.no_grad()
def masked_metrics(pred_z: torch.Tensor, true_z: torch.Tensor,
                   scaler: Dict[str, float]) -> Dict[str, float]:
    """MAE / RMSE / MAPE in mph over valid (non-missing) entries. For reporting."""
    pred = _to_real(pred_z, scaler["mean"], scaler["std"])
    true = _to_real(true_z, scaler["mean"], scaler["std"])
    mask = _mask_from_target(true)

    mae = (torch.abs(pred - true) * mask).mean()
    rmse = torch.sqrt(((pred - true) ** 2 * mask).mean())
    # MAPE only where |true| is meaningfully non-zero (already ensured by mask).
    mape = (torch.abs((pred - true) / true.clamp(min=1e-3)) * mask).mean() * 100.0
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


@torch.no_grad()
def per_horizon_metrics(pred_z: torch.Tensor, true_z: torch.Tensor,
                        scaler: Dict[str, float],
                        horizons_steps=(3, 6, 12)) -> Dict[str, Dict[str, float]]:
    """Metrics at specific forecast horizons. pred/true : [B, T_out, N].

    horizons_steps default (3, 6, 12) = 15 / 30 / 60 minutes at 5-min resolution.
    Returns {"15min": {mae, rmse, mape}, ...}.
    """
    out = {}
    for step in horizons_steps:
        label = f"{step * 5}min"
        # step is 1-indexed horizon; tensor is 0-indexed along time.
        out[label] = masked_metrics(pred_z[:, step - 1], true_z[:, step - 1], scaler)
    return out
