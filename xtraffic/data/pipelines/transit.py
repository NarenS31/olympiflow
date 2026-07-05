"""Phase 6 — real TRANSIT modality (GTFS static -> nearby stop count per node).

Run:  python -m xtraffic.data.pipelines.transit --dataset metr_la

GTFS static feeds are public, keyless zips every transit agency publishes; the
stops.txt inside carries every stop's id + lat/lon. We count how many stops sit
within `radius_m` of each traffic sensor. Intuition: a sensor surrounded by many
transit stops sits in a denser, more multi-modal corridor whose congestion
dynamics differ from a car-only freeway segment. It gives the model, and the
cross-city transfer, per-node context the speed series alone lacks.

Transit stops don't move minute-to-minute, so this modality is STATIC in time:
the [T, N, 1] array holds the same per-node count at every timestep. That's fine —
it's node context, not a time signal. After the shared writer z-scores it, the
MOD-3 gate learns how much to weight it (matches modalities.transit:1 in config).

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd

from xtraffic.utils.graph_utils import haversine_meters
from xtraffic.utils.io_utils import download, load_data_config, raw_dir
from xtraffic.data.pipelines._modality_common import (
    load_node_latlon, load_timestamp_index, save_modality_sidecar)


def _load_stops(zip_path: str) -> pd.DataFrame:
    """Read stops.txt (stop_lat, stop_lon) out of a GTFS static zip."""
    with zipfile.ZipFile(zip_path) as z:
        # GTFS mandates stops.txt at the archive root.
        name = next((n for n in z.namelist() if n.endswith("stops.txt")), None)
        if name is None:
            raise ValueError(f"{zip_path} has no stops.txt (not a GTFS feed?)")
        with z.open(name) as fh:
            df = pd.read_csv(io.BytesIO(fh.read()))
    df = df.dropna(subset=["stop_lat", "stop_lon"])
    return df[["stop_lat", "stop_lon"]].astype(float)


def build_transit(dataset: str) -> Dict:
    cfg = load_data_config()
    tcfg = cfg["transit"]
    radius = float(tcfg["radius_m"])
    feeds: List[str] = tcfg["feeds"].get(dataset, [])
    if not feeds:
        raise ValueError(f"No GTFS feeds configured for '{dataset}' (see data.yaml transit.feeds)")

    ts = load_timestamp_index(dataset)                         # [T]
    sensor_ids, latlon = load_node_latlon(dataset)            # [N], [N,2]
    n_nodes = len(sensor_ids)
    T = len(ts)

    # --- Gather all stops from every configured feed for this city. ---
    rdir = os.path.join(raw_dir(dataset), "transit")
    os.makedirs(rdir, exist_ok=True)
    stops = []
    for i, url in enumerate(feeds):
        zpath = download(url, os.path.join(rdir, f"gtfs_{i}.zip"))
        s = _load_stops(zpath)
        stops.append(s)
        print(f"[transit] feed {i}: {len(s)} stops")
    all_stops = pd.concat(stops, ignore_index=True)           # [num_stops, 2]
    stop_lat = all_stops["stop_lat"].to_numpy()               # [S]
    stop_lon = all_stops["stop_lon"].to_numpy()               # [S]

    # --- Count stops within `radius` of each sensor. A cheap bounding-box prefilter
    #     (1 deg lat ~= 111 km) avoids a full haversine to every stop for every node. ---
    deg_pad = radius / 111_000.0 * 1.5                        # generous lat/lon pad
    counts = np.zeros(n_nodes, dtype=np.float32)              # [N]
    for n in range(n_nodes):
        lat_n, lon_n = latlon[n]
        box = ((np.abs(stop_lat - lat_n) < deg_pad) &
               (np.abs(stop_lon - lon_n) < deg_pad))          # bounding-box mask
        cand_lat, cand_lon = stop_lat[box], stop_lon[box]
        c = 0
        for sl, so in zip(cand_lat, cand_lon):
            if haversine_meters(lat_n, lon_n, sl, so) <= radius:
                c += 1
        counts[n] = c
    print(f"[transit] stops/node within {radius:.0f} m: "
          f"min {counts.min():.0f}  mean {counts.mean():.1f}  max {counts.max():.0f}")

    # Broadcast the static per-node count across all timesteps -> [T, N, 1].
    raw = np.tile(counts.reshape(1, n_nodes, 1), (T, 1, 1)).astype(np.float32)
    return save_modality_sidecar(dataset, "transit", raw, channels=["nearby_stop_count"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr_la")
    args = ap.parse_args()
    build_transit(args.dataset)


if __name__ == "__main__":
    main()
