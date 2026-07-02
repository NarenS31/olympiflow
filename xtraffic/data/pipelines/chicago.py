"""Phase 1 — Chicago Traffic Tracker pipeline (road SEGMENTS as nodes).

Run:  python -m xtraffic.data.pipelines.chicago

Chicago is the CROSS-CITY test domain (train LA, test Chicago). Unlike METR-LA /
PEMS-BAY it publishes segment-level congestion, not point sensors, and ships no
precomputed road-network distance file. So we:
  1. Pull the most-recent window of HISTORICAL congestion estimates (Socrata,
     dataset 4g9f-3jbs). That table already carries per-segment coordinates and
     street names, so it doubles as the metadata source — no snapshot join needed.
  2. Pivot to [T, N] speed on the native 10-min grid, then resample to the 5-min
     grid used by the H5 pipelines; encode missing / "-1 no-estimate" as 0.
  3. Build the graph with SEGMENTS as nodes and an edge between two segments whose
     endpoints are within `segment_adjacency_tol_m` meters (shared intersection).
  4. Emit the EXACT same tensor format as METR-LA so an LA-trained model runs here
     with zero code changes.

Every schema difference from METR-LA is logged in DIFFERENCES.md (paper needs it).
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import requests

from xtraffic.utils.graph_utils import (
    StandardScaler,
    chronological_split,
    haversine_meters,
    sliding_windows,
)
from xtraffic.utils.io_utils import (
    load_data_config,
    print_stats_report,
    processed_dir,
    raw_dir,
    save_processed_dataset,
)


def _fetch_recent_history(cfg: Dict, rdir: str) -> pd.DataFrame:
    """Most-recent page of per-segment estimates (ordered by time DESC).

    We order DESC and take `page_limit` rows so we always get live-ish data even
    if the feed hasn't updated recently — a fixed 'last-14-days' window breaks the
    moment the portal goes stale. No app token needed at this volume.
    """
    ds = cfg["datasets"]["chicago"]
    cache = os.path.join(rdir, "history_recent.csv")
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        print(f"[socrata] cached {os.path.basename(cache)}")
        return pd.read_csv(cache)
    url = f"https://{ds['socrata_domain']}/resource/{ds['historical_dataset_id']}.csv"
    params = {"$limit": ds["page_limit"], "$order": "time DESC"}
    print(f"[socrata] GET {url}  params={params}")
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    with open(cache + ".part", "wb") as f:
        f.write(r.content)
    os.replace(cache + ".part", cache)
    return pd.read_csv(cache)


def _build_segment_adjacency(meta: pd.DataFrame, tol_m: float) -> np.ndarray:
    """Edge between segments sharing an endpoint (within tol meters). [N, N]."""
    n = len(meta)
    endpoints = [
        ((r.start_lat, r.start_lon), (r.end_lat, r.end_lon))
        for r in meta.itertuples(index=False)
    ]
    adj = np.eye(n, dtype=np.float32)                        # self-loops
    for i in range(n):
        for j in range(i + 1, n):
            connected = any(
                haversine_meters(a[0], a[1], b[0], b[1]) <= tol_m
                for a in endpoints[i] for b in endpoints[j]
            )
            if connected:
                adj[i, j] = adj[j, i] = 1.0                  # undirected shared-endpoint edge
    return adj


def main() -> None:
    cfg = load_data_config()
    win, split_cfg = cfg["window"], cfg["split"]
    ds_cfg = cfg["datasets"]["chicago"]
    name = "chicago"
    rdir = raw_dir(name)

    raw = _fetch_recent_history(cfg, rdir)
    raw.columns = [c.strip().lower() for c in raw.columns]

    # Long table -> tidy frame we need. speed = -1 means "no estimate" -> NaN.
    hist = pd.DataFrame({
        "segment_id": raw["segment_id"].astype(int),
        "speed": pd.to_numeric(raw["speed"], errors="coerce"),
        "time": pd.to_datetime(raw["time"], errors="coerce"),
    }).dropna(subset=["time"])
    hist.loc[hist["speed"] < 0, "speed"] = np.nan

    # Per-segment metadata (coords + human-readable name) from the first row seen.
    meta_src = raw.drop_duplicates("segment_id").set_index("segment_id")
    def _meta_col(*names):
        for nm in names:
            if nm in meta_src.columns:
                return meta_src[nm]
        return None

    # Pivot to [T, N]; keep segments that actually have >=1 real speed reading.
    wide = hist.pivot_table(index="time", columns="segment_id", values="speed", aggfunc="mean")
    wide = wide.resample(f"{win['resolution_minutes']}min").mean()
    keep = [s for s in wide.columns if wide[s].notna().any()]
    wide = wide[sorted(keep)]
    seg_order = list(wide.columns)                          # canonical node ordering

    meta = pd.DataFrame({
        "segment_id": seg_order,
        "street": [meta_src.loc[s, "street"] if "street" in meta_src.columns else "" for s in seg_order],
        "from_st": [meta_src.loc[s, "from_street"] if "from_street" in meta_src.columns else "" for s in seg_order],
        "to_st": [meta_src.loc[s, "to_street"] if "to_street" in meta_src.columns else "" for s in seg_order],
        "start_lat": [float(meta_src.loc[s, "start_latitude"]) for s in seg_order],
        "start_lon": [float(meta_src.loc[s, "start_longitude"]) for s in seg_order],
        "end_lat": [float(meta_src.loc[s, "end_latitude"]) for s in seg_order],
        "end_lon": [float(meta_src.loc[s, "end_longitude"]) for s in seg_order],
    })

    speed = wide.to_numpy(dtype=np.float32)                 # [T, N], NaN = missing
    missing_pct = float(np.isnan(speed).mean() * 100.0) if speed.size else 100.0
    speed = np.nan_to_num(speed, nan=0.0)                   # 0 = missing (DCRNN convention)
    n_steps, n_nodes = speed.shape

    # Features [T, N, 2] = (speed, time-of-day) — matches the CSV pipelines exactly.
    idx = wide.index
    tod = ((idx.values - idx.values.astype("datetime64[D]")) / np.timedelta64(1, "D")).astype(np.float32)
    tod = np.tile(tod.reshape(-1, 1, 1), (1, n_nodes, 1))   # [T, N, 1]
    features = np.concatenate([speed.reshape(n_steps, n_nodes, 1), tod], axis=2)  # [T, N, 2]

    if n_steps < win["input_length"] + win["output_length"] + 1:
        raise RuntimeError(
            f"[chicago] only {n_steps} timesteps after resampling — not enough for one "
            f"window. Increase page_limit in configs/data.yaml or clear the cache."
        )

    X, Y = sliding_windows(features, win["input_length"], win["output_length"])
    n_samples = X.shape[0]
    tr, va, te = chronological_split(n_samples, split_cfg["train"], split_cfg["val"])

    scaler = StandardScaler.fit(X[tr][..., 0])
    splits = {}
    for sname, sl in (("train", tr), ("val", va), ("test", te)):
        Xs = X[sl].copy()
        Xs[..., 0] = scaler.transform(Xs[..., 0])
        splits[sname] = {"X": Xs, "Y": scaler.transform(Y[sl])}

    adjacency = _build_segment_adjacency(meta, ds_cfg["segment_adjacency_tol_m"])
    n_edges = int((adjacency > 0).sum() - n_nodes)

    node_meta = {
        "dataset": name, "sensor_ids": [int(s) for s in seg_order], "n_nodes": n_nodes,
        "names": [f"{r.street} ({r.from_st}->{r.to_st})" for r in meta.itertuples(index=False)],
        "latlon": [[float((r.start_lat + r.end_lat) / 2), float((r.start_lon + r.end_lon) / 2)]
                   for r in meta.itertuples(index=False)],
    }
    stats = {
        "dataset": name, "nodes": n_nodes, "edges": n_edges, "timesteps": n_steps,
        "date_range": f"{idx[0]} -> {idx[-1]}", "missing_data_pct": round(missing_pct, 3),
        "samples_total": n_samples, "samples_train": int(X[tr].shape[0]),
        "samples_val": int(X[va].shape[0]), "samples_test": int(X[te].shape[0]),
        "X_shape": list(splits["train"]["X"].shape), "Y_shape": list(splits["train"]["Y"].shape),
        "scaler_mean": round(scaler.mean, 4), "scaler_std": round(scaler.std, 4),
        "note": "segments-as-nodes; recent Socrata window; see DIFFERENCES.md",
    }
    save_processed_dataset(name, splits, adjacency, scaler.to_dict(), node_meta, stats)
    print_stats_report(stats)
    print(f"[chicago] done -> {processed_dir(name)}")


if __name__ == "__main__":
    main()
