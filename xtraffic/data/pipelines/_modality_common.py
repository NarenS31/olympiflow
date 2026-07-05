"""Phase 6 — shared machinery for attaching REAL heterogeneous modalities.

Design decision (flagged per CLAUDE.md "never silently change what we built"):
we do NOT rewrite the Phase-1 processed tensors. The traffic X/Y in
processed/<dataset>/{train,val,test}.npz stay exactly as Phase 1 wrote them.
Instead each Phase-6 feed (weather / events / transit) writes an INDEPENDENT
SIDECAR file:

    processed/<dataset>/mod_<name>.npz   -> arrays train/val/test, each [S, 12, N, C]
    processed/<dataset>/mod_<name>.json  -> {channels, per-channel train scaler, stats}

WHY sidecars:
  * A modality is optional. If mod_transit.npz is absent (e.g. Chicago has no GTFS
    wired yet) the loader simply passes None for that modality — exactly the
    "zero the gate, don't crash" contract MOD 3 was designed for.
  * Each feed is built by its own pipeline, on its own schedule, without touching
    the others. Rerunning weather never disturbs events.
  * The window indexing is guaranteed to line up with the traffic tensors because
    we reuse the SAME sliding_windows + chronological_split on a raw feature array
    that shares the traffic series' timestamp index. Same T in -> same S out.

Every sidecar is z-scored with TRAIN-split statistics only (same no-leakage rule
as the speed scaler). Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from xtraffic.utils.graph_utils import chronological_split, sliding_windows
from xtraffic.utils.io_utils import load_data_config, processed_dir, raw_dir


# ---------------------------------------------------------------------------
# Reload the alignment reference (timestamps + node coordinates) for a dataset.
# ---------------------------------------------------------------------------
def load_timestamp_index(dataset: str) -> pd.DatetimeIndex:
    """The 5-min timestamp index of the dataset's speed series, from the raw CSV.

    We only read the first column (the index) so this stays cheap even on the
    ~70 MB METR-LA file. Every modality is aligned to THIS clock before windowing,
    which is what guarantees sidecar sample i corresponds to traffic sample i.
    """
    raw_csv = {
        "metr_la": "METR-LA.csv",
        "pems_bay": "PEMS-BAY.csv",
    }.get(dataset)
    if raw_csv is None:
        raise ValueError(f"No known raw speed CSV for dataset '{dataset}'")
    path = os.path.join(raw_dir(dataset), raw_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — run the Phase-1 pipeline for {dataset} first "
            f"(python -m xtraffic.data.pipelines.{dataset})")
    idx = pd.read_csv(path, usecols=[0], index_col=0).index      # [T]
    return pd.to_datetime(idx)


def load_node_latlon(dataset: str) -> Tuple[List[int], np.ndarray]:
    """(sensor_ids, latlon[N,2]) in canonical node order, from Phase-1 node_meta."""
    with open(os.path.join(processed_dir(dataset), "node_meta.json")) as f:
        meta = json.load(f)
    sensor_ids = [int(s) for s in meta["sensor_ids"]]            # canonical order
    latlon = meta.get("latlon")
    if latlon is None:
        raise ValueError(f"{dataset} node_meta has no latlon — needed for spatial joins")
    # Some sensors may be missing coords (None) -> fill with the dataset centroid so
    # a spatial join still returns *something* rather than crashing.
    arr = np.array([ll if ll is not None else [np.nan, np.nan] for ll in latlon],
                   dtype=np.float64)                             # [N, 2]
    if np.isnan(arr).any():
        centroid = np.nanmean(arr, axis=0)                       # [2]
        arr[np.isnan(arr).any(axis=1)] = centroid
    return sensor_ids, arr                                       # [N], [N,2]


# ---------------------------------------------------------------------------
# The core writer: raw [T,N,C] feature -> windowed, split, train-scaled sidecar.
# ---------------------------------------------------------------------------
def save_modality_sidecar(
    dataset: str,
    name: str,
    raw_feature: np.ndarray,        # [T, N, C], aligned to load_timestamp_index(dataset)
    channels: List[str],
    scale: bool = True,
) -> Dict:
    """Window + chronologically split + train-scale a raw modality, save sidecar.

    raw_feature[t, n, c] is the value of channel c at node n and timestep t. It
    MUST share the traffic series' timestamp index (length T) and node order (N).

    Returns a small stats dict (also written to mod_<name>.json).
    """
    cfg = load_data_config()
    win = cfg["window"]
    split_cfg = cfg["split"]

    T, N, C = raw_feature.shape
    assert C == len(channels), f"{name}: {C} channels but {len(channels)} names"

    # --- Sanitize non-finite values BEFORE anything else. -------------------
    # A single NaN/inf in an input poisons every gradient (train_loss -> nan).
    # Real feeds DO carry gaps: e.g. Open-Meteo's ERA5 archive does not serve the
    # `visibility` variable, so that whole channel arrives NaN. We fill each
    # channel's non-finite entries with that channel's finite MEAN, so after the
    # train z-score they sit at ~0 (neutral) and the MOD-3 gate can ignore them.
    # A channel that is entirely missing collapses to all-0 -> contributes nothing.
    raw_feature = np.asarray(raw_feature, dtype=np.float32).copy()
    for c in range(C):
        chan = raw_feature[..., c]
        finite = np.isfinite(chan)
        if not finite.all():
            fill = float(chan[finite].mean()) if finite.any() else 0.0
            n_bad = int((~finite).sum())
            chan[~finite] = fill
            raw_feature[..., c] = chan
            note = "ENTIRELY missing -> zeroed" if not finite.any() else f"{n_bad} cells"
            print(f"[{name}] WARNING: channel '{channels[c]}' had non-finite values "
                  f"({note}); filled with channel mean {fill:.3f}")

    # Reuse the EXACT Phase-1 windowing. sliding_windows also returns a speed-only
    # Y (channel 0) which is meaningless for a modality -> we discard it and keep X.
    X, _ = sliding_windows(raw_feature.astype(np.float32),
                           win["input_length"], win["output_length"])  # X:[S,12,N,C]
    n_samples = X.shape[0]
    tr, va, te = chronological_split(n_samples, split_cfg["train"], split_cfg["val"])

    # Per-channel z-score, fit on TRAIN windows only (no-leakage rule).
    means = np.zeros(C, dtype=np.float64)
    stds = np.ones(C, dtype=np.float64)
    if scale:
        train_X = X[tr]                                          # [n_tr,12,N,C]
        for c in range(C):
            means[c] = float(train_X[..., c].mean())
            s = float(train_X[..., c].std())
            stds[c] = s if s > 1e-6 else 1.0                     # guard zero-variance
        X = X.copy()
        for c in range(C):
            X[..., c] = (X[..., c] - means[c]) / stds[c]

    d = processed_dir(dataset)
    np.savez_compressed(os.path.join(d, f"mod_{name}.npz"),
                        train=X[tr], val=X[va], test=X[te])
    stats = {
        "dataset": dataset, "modality": name, "channels": channels,
        "n_samples": int(n_samples),
        "shape_per_sample": [int(win["input_length"]), int(N), int(C)],
        "scaled": bool(scale),
        "channel_mean": [round(m, 5) for m in means.tolist()],
        "channel_std": [round(s, 5) for s in stds.tolist()],
    }
    with open(os.path.join(d, f"mod_{name}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{name}] sidecar -> mod_{name}.npz  "
          f"train/val/test = {X[tr].shape[0]}/{X[va].shape[0]}/{X[te].shape[0]}  "
          f"[S,12,{N},{C}]")
    return stats
