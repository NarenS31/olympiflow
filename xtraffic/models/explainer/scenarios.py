"""Pick diverse, reproducible scenarios from the test split to explain.

The verification gate asks for explanations across rush hour, midday, night,
weekend, and a high-congestion moment. Our input tensor carries only two
channels — speed and TIME-OF-DAY (fraction of day) — so:

  * time-of-day scenarios (rush AM/PM, midday, night) come straight from the
    clock channel;
  * "high congestion" = the window whose mean real speed is lowest;
  * WEEKEND is NOT directly recoverable — day-of-week was never a feature (see
    _csv_common._time_of_day_feature). We therefore use a documented PROXY: a
    weekday-daytime window that is unusually FREE-FLOWING (high speed at a time
    that is normally busy), which is what a weekend looks like to a
    clock-only model. We label it honestly so the paper never overclaims.

Selection is deterministic (fixed test split, argmin/argmax, no randomness) so
the five scenarios are the same on every run — reproducibility (CLAUDE.md).

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..gnn.loaders import _load_split

SPEED_CHANNEL = 0
TOD_CHANNEL = 1


def _tod_to_clock(frac: float) -> str:
    """Fraction-of-day [0,1) -> HH:MM label (for the explanation timestamp)."""
    minutes = int(round(frac * 24 * 60)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def select_scenarios(dataset: str, scaler: Dict[str, float]
                     ) -> List[Dict]:
    """Return a list of scenario dicts:
        {"name","sample_index","target_node","horizon_step","timestamp","note"}
    Each references one window X[sample_index] from the test split.
    """
    X, _ = _load_split(dataset, "test")           # X: [S, T, N, 2]
    S, T, N, _ = X.shape
    tod0 = X[:, 0, 0, TOD_CHANNEL].numpy()        # time-of-day at window start [S]

    # Mean REAL speed per window over valid (non-missing) sensor readings.
    # A missing reading is raw 0 mph -> after inverse z-score it is ~0 mph in
    # REAL space (NOT ~0 in z-space, where it is a large negative number). So we
    # must test validity in mph, matching utils.metrics masking. This was a real
    # bug: testing |z|>eps wrongly treated missing sensors as valid, which made
    # slowest_node pick missing sensors (0 mph) as the target every time.
    speed_z = X[..., SPEED_CHANNEL].numpy()       # [S, T, N]
    speed_mph = speed_z * scaler["std"] + scaler["mean"]
    valid = speed_mph > 1.0                       # >1 mph == a genuine reading
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(valid.reshape(S, -1).any(1),
                              (speed_mph * valid).reshape(S, -1).sum(1)
                              / np.clip(valid.reshape(S, -1).sum(1), 1, None),
                              np.inf)             # [S]

    def in_window(lo: float, hi: float) -> np.ndarray:
        return np.where((tod0 >= lo) & (tod0 < hi))[0]

    def most_congested_in(idxs: np.ndarray) -> int:
        return int(idxs[np.argmin(mean_speed[idxs])]) if len(idxs) else 0

    # Time-of-day bands (fraction of day). AM rush ~7-9:30, PM rush ~16-19.
    am = most_congested_in(in_window(0.29, 0.40))
    pm = most_congested_in(in_window(0.67, 0.79))
    midday = most_congested_in(in_window(0.45, 0.55))
    night = most_congested_in(in_window(0.00, 0.20))
    worst = int(np.argmin(mean_speed))            # global high-congestion moment

    # Weekend proxy: among daytime windows (0.30-0.75), the FASTEST one — a busy
    # clock time that is nonetheless free-flowing (weekend-like to the model).
    day_idx = in_window(0.30, 0.75)
    if len(day_idx):
        weekend = int(day_idx[np.argmax(np.where(np.isfinite(mean_speed[day_idx]),
                                                 mean_speed[day_idx], -np.inf))])
    else:
        weekend = 0

    # Target node per scenario = the SLOWEST valid sensor in that window (the
    # place a planner most wants explained). Horizon 6 = 30 min (headline horizon).
    def slowest_node(sample: int) -> int:
        s = speed_mph[sample].mean(0)             # mean over time per node [N]
        v = valid[sample].any(0)                  # nodes with any valid reading
        s = np.where(v, s, np.inf)
        return int(np.argmin(s))

    specs = [
        ("rush_hour_am", am, "Morning rush (~clock time below)."),
        ("rush_hour_pm", pm, "Evening rush."),
        ("midday", midday, "Midday off-peak."),
        ("night", night, "Overnight, light traffic."),
        ("high_congestion", worst, "Globally most congested window in test set."),
        ("weekend_proxy", weekend,
         "PROXY: free-flowing daytime window (day-of-week is not a model feature)."),
    ]

    scenarios = []
    for name, idx, note in specs:
        node = slowest_node(idx)
        clock = _tod_to_clock(float(tod0[idx]))
        scenarios.append({
            "name": name,
            "sample_index": int(idx),
            "target_node": node,
            "horizon_step": 6,                    # 30-minute horizon
            "timestamp": f"test#{idx} @ {clock}", # derived clock label (see note)
            "note": note,
        })
    return scenarios


def load_window(dataset: str, sample_index: int) -> torch.Tensor:
    """Return X[sample_index] as a batch of 1: [1, T, N, C]."""
    X, _ = _load_split(dataset, "test")
    return X[sample_index: sample_index + 1]
