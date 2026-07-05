"""Phase 6 — real WEATHER modality (Open-Meteo historical archive, keyless).

Run:  python -m xtraffic.data.pipelines.weather --dataset metr_la

Open-Meteo's archive API (archive-api.open-meteo.com) serves free, keyless,
historical hourly weather for any lat/lon and date range — perfect for the
CLAUDE.md "no API keys, reproducible" constraint.

Pipeline:
  1. Read the dataset's 5-min timestamp index and node lat/lons (Phase 1 outputs).
  2. Cluster nodes onto a coarse lat/lon grid (~0.05deg ≈ 5 km) so weather, which
     varies slowly in space, needs only a handful of API calls instead of one per
     sensor. Each node is assigned its grid cell's weather.
  3. Download hourly temperature / precipitation / visibility per grid cell,
     cache the raw JSON under data/raw/<dataset>/weather/.
  4. Interpolate hourly -> 5-min onto the exact traffic clock, broadcast to nodes.
  5. Hand the [T, N, 3] array to the shared sidecar writer (windows + split +
     train-only z-score), producing processed/<dataset>/mod_weather.npz.

Three channels = temp(°C), precip(mm), visibility(m) — matches modalities.weather:3
in the train config. Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from xtraffic.utils.io_utils import raw_dir
from xtraffic.data.pipelines._modality_common import (
    load_node_latlon, load_timestamp_index, save_modality_sidecar)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["temperature_2m", "precipitation", "visibility"]
GRID_DEG = 0.05   # ~5 km cells: weather is spatially smooth, so this is plenty.


def _grid_key(lat: float, lon: float) -> Tuple[float, float]:
    """Snap a coordinate to the GRID_DEG lattice — nodes in the same cell share
    one weather query. Rounded to a stable string-safe value."""
    return (round(round(lat / GRID_DEG) * GRID_DEG, 3),
            round(round(lon / GRID_DEG) * GRID_DEG, 3))


def _fetch_cell(lat: float, lon: float, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """Hourly weather for one grid cell, cached to disk (idempotent reruns)."""
    fname = f"weather_{lat:.3f}_{lon:.3f}_{start}_{end}.json".replace("-", "m")
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            data = json.load(f)
    else:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "America/Los_Angeles",
        }
        print(f"[weather] fetch cell ({lat:.3f},{lon:.3f}) {start}..{end}")
        r = requests.get(ARCHIVE_URL, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        with open(path, "w") as f:
            json.dump(data, f)
    hourly = data["hourly"]                                       # dict of lists
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")                                  # hourly index


def build_weather(dataset: str) -> Dict:
    ts = load_timestamp_index(dataset)                           # [T] 5-min clock
    sensor_ids, latlon = load_node_latlon(dataset)              # [N], [N,2]
    n_nodes = len(sensor_ids)
    start = ts[0].strftime("%Y-%m-%d")
    end = ts[-1].strftime("%Y-%m-%d")
    cache_dir = os.path.join(raw_dir(dataset), "weather")
    os.makedirs(cache_dir, exist_ok=True)

    # --- Assign each node to a grid cell; fetch weather once per unique cell. ---
    node_cell = [_grid_key(lat, lon) for lat, lon in latlon]     # [N]
    unique_cells = sorted(set(node_cell))
    print(f"[weather] {n_nodes} nodes -> {len(unique_cells)} grid cells "
          f"({start} .. {end})")

    # Per-cell hourly frame, reindexed+interpolated onto the 5-min traffic clock.
    cell_series: Dict[Tuple[float, float], np.ndarray] = {}
    for cell in unique_cells:
        hourly = _fetch_cell(cell[0], cell[1], start, end, cache_dir)  # [H,3]
        hourly = hourly[HOURLY_VARS].astype(np.float32)
        # Reindex the union of hourly + 5-min stamps, time-interpolate, then keep
        # only the 5-min stamps -> smooth hourly->5min without a step function.
        union = hourly.index.union(ts)
        interp = hourly.reindex(union).interpolate(method="time").ffill().bfill()
        cell_series[cell] = interp.loc[ts].to_numpy(dtype=np.float32)  # [T,3]

    # --- Assemble [T, N, 3] by broadcasting each node's cell series. ---
    T = len(ts)
    raw = np.zeros((T, n_nodes, len(HOURLY_VARS)), dtype=np.float32)   # [T,N,3]
    for n, cell in enumerate(node_cell):
        raw[:, n, :] = cell_series[cell]                         # [T,3]

    # Visibility is huge (metres, up to 24000) vs precip (~0-5mm); the shared
    # writer z-scores per channel, so the scales are reconciled automatically.
    return save_modality_sidecar(dataset, "weather", raw,
                                 channels=["temp_c", "precip_mm", "visibility_m"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr_la",
                    help="metr_la or pems_bay (both ship lat/lon)")
    args = ap.parse_args()
    build_weather(args.dataset)


if __name__ == "__main__":
    main()
