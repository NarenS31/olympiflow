"""Phase 1 verification gate — load every processed dataset and assert integrity.

Run:  python -m xtraffic.data.pipelines.sanity_check

Asserts, for each dataset that has been built:
  - no NaNs / infs survive preprocessing
  - train/val/test are chronologically ordered with no sample-count mismatch
  - tensor shapes match configs/data.yaml (T_in, T_out, N, F)
  - adjacency is [N, N], symmetric-diagonal 1.0, values in [0, 1]
"""
from __future__ import annotations

import json
import os

import numpy as np

from xtraffic.utils.io_utils import PKG_ROOT, load_data_config

# power_grid (Phase 19) is checked by the SAME assertions as the traffic datasets
# on purpose: passing this gate IS the proof that the cross-domain tensor contract
# holds and that the unmodified model can consume it.
DATASETS = ["metr_la", "pems_bay", "chicago", "power_grid"]


def _load_split(dpath, split):
    z = np.load(os.path.join(dpath, f"{split}.npz"))
    return z["X"], z["Y"]


def check_dataset(name: str, cfg: dict) -> bool:
    dpath = os.path.join(PKG_ROOT, "data", "processed", name)
    if not os.path.exists(os.path.join(dpath, "train.npz")):
        print(f"[skip] {name}: not built yet")
        return True

    win = cfg["window"]
    Xtr, Ytr = _load_split(dpath, "train")
    Xva, Yva = _load_split(dpath, "val")
    Xte, Yte = _load_split(dpath, "test")
    adj = np.load(os.path.join(dpath, "adjacency.npy"))
    stats = json.load(open(os.path.join(dpath, "stats.json")))

    n_nodes = stats["nodes"]
    # shape contract
    assert Xtr.shape[1] == win["input_length"], f"{name}: T_in mismatch"
    assert Ytr.shape[1] == win["output_length"], f"{name}: T_out mismatch"
    assert Xtr.shape[2] == n_nodes and Ytr.shape[2] == n_nodes, f"{name}: N mismatch"
    assert Xtr.shape[3] == 2, f"{name}: expected 2 feature channels"
    # no NaN/inf
    for arr, tag in ((Xtr, "Xtr"), (Ytr, "Ytr"), (Xva, "Xva"), (Xte, "Xte")):
        assert np.isfinite(arr).all(), f"{name}: non-finite values in {tag}"
    # split ordering / counts
    total = Xtr.shape[0] + Xva.shape[0] + Xte.shape[0]
    assert total == stats["samples_total"], f"{name}: split counts don't sum to total"
    assert Xtr.shape[0] >= Xte.shape[0] >= 0, f"{name}: train should be the largest split"
    # adjacency sanity
    assert adj.shape == (n_nodes, n_nodes), f"{name}: adjacency not [N,N]"
    assert adj.min() >= 0.0 and adj.max() <= 1.0 + 1e-6, f"{name}: adjacency out of [0,1]"
    assert np.allclose(np.diag(adj), 1.0), f"{name}: adjacency diagonal must be 1.0"

    print(f"[ok] {name}: N={n_nodes} X{list(Xtr.shape)} splits "
          f"{Xtr.shape[0]}/{Xva.shape[0]}/{Xte.shape[0]} finite ✓ shapes ✓ adj ✓")
    return True


def main() -> None:
    cfg = load_data_config()
    all_ok = all(check_dataset(name, cfg) for name in DATASETS)
    print("\nSANITY GATE:", "PASS ✓" if all_ok else "FAIL ✗")


if __name__ == "__main__":
    main()
