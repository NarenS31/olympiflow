"""Phase 6 — real EVENTS modality (curated major-venue schedule, proximity-decayed).

Run:  python -m xtraffic.data.pipelines.events --dataset metr_la

Big venue events (Dodgers/Lakers games, Hollywood Bowl concerts) dump tens of
thousands of vehicles onto nearby corridors in a short window — a genuine cause
of congestion that the pure-traffic model cannot see. We encode them as a single
proximity-decayed indicator channel (matches modalities.events:1 in the config).

Ground truth is a COMMITTED, editable curated file (events_la_2012.json) — venue
coordinates are exact; the datetimes are a representative set the researcher
verifies/expands against public schedules. This satisfies "reproducible, no manual
downloads": the schedule lives in the repo, not in someone's memory.

Encoding, per node n and 5-min timestep t:
    events[t, n] = sum over events e ACTIVE at t of  exp(-dist(n, venue_e) / DECAY_M)
  * "active" = t within [start - PRE, start + duration + POST] (arrival + departure
    surges bracket the on-site event).
  * exp(-dist/DECAY_M) is a smooth spatial decay: a sensor next to Dodger Stadium
    gets ~1.0, one 5 km away gets a small fraction, one across the metro ~0.
  * overlapping events sum, so a rare double-header night reads hotter.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from xtraffic.utils.graph_utils import haversine_meters
from xtraffic.data.pipelines._modality_common import (
    load_node_latlon, load_timestamp_index, save_modality_sidecar)

# Curated schedule lives next to this file (committed, editable).
EVENTS_FILE = {
    "metr_la": "events_la_2012.json",
}
DECAY_M = 2500.0     # spatial e-fold distance (~2.5 km): venue traffic is local
PRE_MIN = 90         # arrival surge begins ~90 min before first pitch / tip-off
POST_MIN = 90        # departure surge lingers ~90 min after the event ends


def build_events(dataset: str) -> Dict:
    fname = EVENTS_FILE.get(dataset)
    if fname is None:
        raise ValueError(f"No curated events file for '{dataset}' "
                         f"(only METR-LA's 2012 window is curated so far)")
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        spec = json.load(f)

    ts = load_timestamp_index(dataset)                          # [T] 5-min clock
    sensor_ids, latlon = load_node_latlon(dataset)             # [N], [N,2]
    n_nodes = len(sensor_ids)
    T = len(ts)
    tz = spec.get("timezone", "America/Los_Angeles")
    venues = spec["venues"]

    # --- Precompute per-node spatial decay to each venue ONCE (nodes x venues). ---
    venue_names = list(venues.keys())
    decay = np.zeros((n_nodes, len(venue_names)), dtype=np.float32)   # [N, V]
    for vi, vn in enumerate(venue_names):
        vlat, vlon = venues[vn]["lat"], venues[vn]["lon"]
        for n in range(n_nodes):
            dist = haversine_meters(latlon[n, 0], latlon[n, 1], vlat, vlon)
            decay[n, vi] = np.exp(-dist / DECAY_M)             # scalar in (0,1]

    # The traffic clock is tz-naive local time (METR-LA is recorded in LA local
    # time), and our event datetimes are LA local too -> compare directly, no tz math.
    raw = np.zeros((T, n_nodes, 1), dtype=np.float32)          # [T, N, 1]
    n_active_events = 0
    for e in spec["events"]:
        vi = venue_names.index(e["venue"])
        start = pd.Timestamp(f"{e['date']} {e['time']}")
        win_start = start - pd.Timedelta(minutes=PRE_MIN)
        win_end = start + pd.Timedelta(minutes=e["duration_minutes"] + POST_MIN)
        # Boolean mask over the 5-min clock for this event's active window.
        mask = (ts >= win_start) & (ts <= win_end)            # [T]
        if not mask.any():
            continue                                          # event outside data range
        n_active_events += 1
        # Add this venue's spatial decay to every active timestep. Broadcasting:
        # raw[mask, :, 0] is [n_active_steps, N]; decay[:, vi] is [N].
        raw[mask, :, 0] += decay[:, vi][None, :]              # [n_active, N]

    print(f"[events] {len(spec['events'])} curated events, "
          f"{n_active_events} fall inside {ts[0].date()}..{ts[-1].date()}; "
          f"peak intensity {raw.max():.3f}")
    # NB: many timesteps are exactly 0 (no event) — that's correct and expected;
    # the modality gate (MOD 3) will learn how much to trust this sparse signal.
    return save_modality_sidecar(dataset, "events", raw, channels=["event_intensity"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr_la")
    args = ap.parse_args()
    build_events(args.dataset)


if __name__ == "__main__":
    main()
