"""Shared builder for the two point-sensor datasets (METR-LA, PEMS-BAY).

Both ship as CSV from the Zenodo DCRNN mirror (record 5146275):
  - <NAME>.csv          : index = 5-min timestamps, columns = sensor ids,
                          values = speed (mph), missing encoded as 0.0
  - distances_*.csv     : edge list (from, to, cost) of road-network distances
  - (optional) meta/loc : sensor lat/lon for human-readable node names (Phase 3)

The canonical node ordering is simply the CSV column order. Keeping both cities
in one builder means the LA and Bay pipelines cannot drift apart.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from xtraffic.utils.graph_utils import (
    StandardScaler,
    build_distance_matrix,
    chronological_split,
    gaussian_kernel_adjacency,
    sliding_windows,
)
from xtraffic.utils.io_utils import print_stats_report, save_processed_dataset


def _time_of_day_feature(index: pd.DatetimeIndex, n_nodes: int) -> np.ndarray:
    """Second input channel: fraction of day in [0,1), broadcast to all nodes.

    WHY: speed is strongly periodic over the day; an explicit clock lets the model
    disambiguate 8am from 8pm. Standard Graph WaveNet time-in-day feature. [T,N,1].
    """
    vals = index.values
    tod = ((vals - vals.astype("datetime64[D]")) / np.timedelta64(1, "D")).astype(np.float32)  # [T]
    return np.tile(tod.reshape(-1, 1, 1), (1, n_nodes, 1))    # [T, N, 1]


def build_and_save_from_csv(
    name: str,
    cfg: Dict,
    speed_csv_path: str,
    dist_path: str,
    expected_nodes: int,
    locations_path: Optional[str] = None,
) -> Dict:
    win = cfg["window"]
    split_cfg = cfg["split"]
    kappa = cfg["adjacency"]["normalized_k"]

    # --- Load raw speed frame: [T, N]; index parsed as datetime, cols as sensor ids ---
    df = pd.read_csv(speed_csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)                      # DatetimeIndex [T]
    df.columns = [int(float(c)) for c in df.columns]         # sensor ids as ints
    speed = df.values.astype(np.float32)                     # [T, N]
    n_steps, n_nodes = speed.shape                           # T, N
    assert n_nodes == expected_nodes, f"{name}: expected {expected_nodes} nodes, got {n_nodes}"

    sensor_ids = [int(c) for c in df.columns]                # canonical node ordering
    missing_pct = float((speed == 0.0).mean() * 100.0)       # 0 = missing (DCRNN convention)

    # --- Features [T, N, 2] = (speed, time-of-day) ---
    tod = _time_of_day_feature(df.index, n_nodes)            # [T, N, 1]
    features = np.concatenate([speed.reshape(n_steps, n_nodes, 1), tod], axis=2)  # [T, N, 2]

    # --- Sliding windows BEFORE splitting, then split window indices by time ---
    X, Y = sliding_windows(features, win["input_length"], win["output_length"])
    # X [samples, T_in, N, 2]   Y [samples, T_out, N]
    n_samples = X.shape[0]
    tr, va, te = chronological_split(n_samples, split_cfg["train"], split_cfg["val"])

    # --- Fit scaler on TRAIN speed channel only, apply to all splits ---
    scaler = StandardScaler.fit(X[tr][..., 0])
    splits = {}
    for split_name, sl in (("train", tr), ("val", va), ("test", te)):
        Xs = X[sl].copy()                                    # [n, T_in, N, 2]
        Xs[..., 0] = scaler.transform(Xs[..., 0])            # normalize speed channel only
        splits[split_name] = {"X": Xs, "Y": scaler.transform(Y[sl])}

    # --- Adjacency from the sensor distance edge list ---
    dist_df = pd.read_csv(dist_path)                         # columns: from,to,cost
    dist_mat = build_distance_matrix(sensor_ids, dist_df)    # [N, N]
    adjacency = gaussian_kernel_adjacency(dist_mat, normalized_k=kappa)  # [N, N]
    n_edges = int((adjacency > 0).sum() - n_nodes)           # exclude self-loops

    # --- Node metadata (human-readable names for Phase 3) ---
    node_meta = {"dataset": name, "sensor_ids": sensor_ids, "n_nodes": n_nodes}
    if locations_path and os.path.exists(locations_path):
        loc = pd.read_csv(locations_path)
        cols = {c.strip().lower(): c for c in loc.columns}
        sid_c = cols.get("sensor_id")
        lat_c = cols.get("latitude") or cols.get("lat")
        lon_c = cols.get("longitude") or cols.get("lon")
        if sid_c and lat_c and lon_c:
            latlon = {int(r[sid_c]): [float(r[lat_c]), float(r[lon_c])] for _, r in loc.iterrows()}
            node_meta["latlon"] = [latlon.get(sid) for sid in sensor_ids]

    stats = {
        "dataset": name, "nodes": n_nodes, "edges": n_edges, "timesteps": n_steps,
        "date_range": f"{df.index[0]} -> {df.index[-1]}",
        "missing_data_pct": round(missing_pct, 3), "samples_total": n_samples,
        "samples_train": int(X[tr].shape[0]), "samples_val": int(X[va].shape[0]),
        "samples_test": int(X[te].shape[0]),
        "X_shape": list(splits["train"]["X"].shape), "Y_shape": list(splits["train"]["Y"].shape),
        "scaler_mean": round(scaler.mean, 4), "scaler_std": round(scaler.std, 4),
    }
    save_processed_dataset(name, splits, adjacency, scaler.to_dict(), node_meta, stats)
    print_stats_report(stats)
    return stats
