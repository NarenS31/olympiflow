"""Phase 1 — PEMS-BAY data pipeline (325 speed sensors, Bay Area, 2017).

Run:  python -m xtraffic.data.pipelines.pems_bay

Identical treatment to METR-LA (same output format), so a model trained on one
can be evaluated on the other with zero code changes. Node ordering = CSV column
order. Bay ships a metadata CSV with lat/lon for human-readable node names.
"""
from __future__ import annotations

import os

from xtraffic.utils.io_utils import download, load_data_config, processed_dir, raw_dir
from xtraffic.data.pipelines._csv_common import build_and_save_from_csv


def main() -> None:
    cfg = load_data_config()
    ds = cfg["datasets"]["pems_bay"]
    name = "pems_bay"

    rdir = raw_dir(name)
    speed_path = download(ds["speed_csv_url"], os.path.join(rdir, "PEMS-BAY.csv"))
    dist_path = download(ds["distances_csv_url"], os.path.join(rdir, "distances_bay_2017.csv"))
    meta_path = download(ds["sensor_meta_url"], os.path.join(rdir, "PEMS-BAY-META.csv"))

    build_and_save_from_csv(
        name=name,
        cfg=cfg,
        speed_csv_path=speed_path,
        dist_path=dist_path,
        locations_path=meta_path,         # Bay metadata carries lat/lon
        expected_nodes=ds["n_nodes"],
    )
    print(f"[pems_bay] done -> {processed_dir(name)}")


if __name__ == "__main__":
    main()
