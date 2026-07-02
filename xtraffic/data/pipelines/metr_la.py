"""Phase 1 — METR-LA data pipeline (207 loop-detector speed sensors, LA, 2012).

Run:  python -m xtraffic.data.pipelines.metr_la

Steps:
  1. Download METR-LA speed CSV (Zenodo DCRNN mirror), sensor-distance CSV, and
     sensor lat/lon locations (DCRNN GitHub).
  2. Build the graph adjacency from sensor distances with a thresholded
     Gaussian kernel (DCRNN Eq. 10) — shared code in utils.graph_utils.
  3. Build sliding-window tensors X [samples,12,207,2], Y [samples,12,207].
  4. Z-score normalize using TRAIN statistics only (no leakage).
  5. Save tensors + scaler + adjacency + node metadata to data/processed/metr_la/.
  6. Print a stats report.
"""
from __future__ import annotations

import os

from xtraffic.utils.io_utils import download, load_data_config, processed_dir, raw_dir
from xtraffic.data.pipelines._csv_common import build_and_save_from_csv


def main() -> None:
    cfg = load_data_config()
    ds = cfg["datasets"]["metr_la"]
    name = "metr_la"

    rdir = raw_dir(name)
    speed_path = download(ds["speed_csv_url"], os.path.join(rdir, "METR-LA.csv"))
    dist_path = download(ds["distances_csv_url"], os.path.join(rdir, "distances_la_2012.csv"))
    loc_path = download(ds["sensor_locations_url"], os.path.join(rdir, "graph_sensor_locations.csv"))

    build_and_save_from_csv(
        name=name,
        cfg=cfg,
        speed_csv_path=speed_path,
        dist_path=dist_path,
        locations_path=loc_path,          # METR-LA ships lat/lon -> human-readable naming (Phase 3)
        expected_nodes=ds["n_nodes"],
    )
    print(f"[metr_la] done -> {processed_dir(name)}")


if __name__ == "__main__":
    main()
