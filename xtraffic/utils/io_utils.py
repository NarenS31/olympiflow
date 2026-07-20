"""Filesystem + download + config helpers shared by every pipeline.

Kept separate from graph_utils so that math (graph_utils) and plumbing (io_utils)
don't tangle. Python 3.9 compatible; numpy + pyyaml + requests only.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import requests
import yaml

# Absolute path to the xtraffic package root, so scripts work from any CWD.
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../xtraffic


def load_data_config() -> Dict:
    path = os.path.join(PKG_ROOT, "configs", "data.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def units_for_dataset(dataset: str, default: str = "mph") -> str:
    """Real-world unit of a dataset's TARGET state, from configs/data.yaml.

    Phase 19 added this because the metrics are unit-agnostic but every label was
    hard-coded "mph". On the power grid the target is voltage in per-unit and MAE
    lands around 0.005, so an axis or table caption reading "mph" would be a lie
    about a number that is ~800x smaller.

    Defaults to "mph" when the key (or the whole dataset entry) is absent, so the
    three traffic datasets and any older config behave exactly as before.
    """
    try:
        ds = load_data_config().get("datasets", {}).get(dataset, {})
        return str(ds.get("units", default))
    except Exception:
        # Never let a label lookup break a training run.
        return default


def raw_dir(dataset: str) -> str:
    d = os.path.join(PKG_ROOT, "data", "raw", dataset)
    os.makedirs(d, exist_ok=True)
    return d


def processed_dir(dataset: str) -> str:
    d = os.path.join(PKG_ROOT, "data", "processed", dataset)
    os.makedirs(d, exist_ok=True)
    return d


def download(url: str, dest: str, chunk: int = 1 << 20) -> str:
    """Download `url` to `dest` unless it already exists. Returns dest path.

    Idempotent so re-running a pipeline doesn't re-fetch large files. Streams to
    disk to keep memory flat for the ~100 MB speed files.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[download] cached {os.path.basename(dest)}")
        return dest
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
        os.replace(tmp, dest)
    print(f"[download] saved {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.1f} MB)")
    return dest


def save_processed_dataset(
    dataset: str,
    splits: Dict[str, Dict[str, np.ndarray]],
    adjacency: np.ndarray,
    scaler: Dict[str, float],
    node_meta: Dict,
    stats: Dict,
) -> None:
    """Persist everything a downstream phase needs, in a stable layout."""
    d = processed_dir(dataset)
    for split, arrays in splits.items():
        np.savez_compressed(os.path.join(d, f"{split}.npz"), X=arrays["X"], Y=arrays["Y"])
    np.save(os.path.join(d, "adjacency.npy"), adjacency)
    with open(os.path.join(d, "scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)
    with open(os.path.join(d, "node_meta.json"), "w") as f:
        json.dump(node_meta, f, indent=2)
    with open(os.path.join(d, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def print_stats_report(stats: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"  DATASET STATS: {stats.get('dataset')}")
    print("=" * 60)
    for k, v in stats.items():
        if k == "dataset":
            continue
        print(f"  {k:<26}: {v}")
    print("=" * 60 + "\n")
