"""Loading processed tensors into PyTorch, plus the modality-dict adapter.

Keeps all "read Phase-1 outputs -> tensors the model wants" plumbing in one place
so train.py / evaluate.py / the explainer all agree on the data contract:
  processed/<dataset>/{train,val,test}.npz  -> X:[S,12,N,2]  Y:[S,12,N]
  processed/<dataset>/scaler.json           -> {"mean","std"}
  processed/<dataset>/adjacency.npy         -> [N,N]
  processed/<dataset>/node_meta.json        -> {sensor_ids, latlon, ...}
Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...utils.io_utils import processed_dir


def load_scaler(dataset: str) -> Dict[str, float]:
    with open(os.path.join(processed_dir(dataset), "scaler.json")) as f:
        return json.load(f)


def load_adjacency(dataset: str) -> torch.Tensor:
    A = np.load(os.path.join(processed_dir(dataset), "adjacency.npy"))
    return torch.from_numpy(A).float()  # [N, N]


def load_node_meta(dataset: str) -> Dict:
    with open(os.path.join(processed_dir(dataset), "node_meta.json")) as f:
        return json.load(f)


def _load_split(dataset: str, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    d = np.load(os.path.join(processed_dir(dataset), f"{split}.npz"))
    X = torch.from_numpy(d["X"]).float()  # [S, 12, N, 2]
    Y = torch.from_numpy(d["Y"]).float()  # [S, 12, N]
    return X, Y


def make_loaders(dataset: str, batch_size: int, num_workers: int = 0
                 ) -> Dict[str, DataLoader]:
    """Return {'train','val','test'} DataLoaders. Train shuffles *samples* — this is
    NOT temporal leakage: the chronological split was already done in Phase 1, so
    every train sample is strictly older than every val/test sample. Shuffling only
    reorders within the train block for better SGD."""
    loaders = {}
    for split in ("train", "val", "test"):
        X, Y = _load_split(dataset, split)
        ds = TensorDataset(X, Y)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, drop_last=False)
    return loaders


def build_modality_dict(X: torch.Tensor, traffic_channels: List[int],
                        modality_names: List[str],
                        M: Optional[torch.Tensor] = None,
                        layout: Optional[List[Tuple[str, int, int]]] = None
                        ) -> Dict[str, Optional[torch.Tensor]]:
    """Turn a raw X batch [B,12,N,2] into the model's modality dict (MOD 3).

    "traffic" always comes from X's channels. Phase-6 modalities (weather/events/
    transit) come from the concatenated sidecar batch M [B,T,N,Csum] sliced by
    `layout` (a list of (name, start, end)). Any modality with no sidecar present
    -> None, so its MOD-3 gate contributes nothing (no crash). With M/layout left
    as None this is exactly the Phase-2 traffic-only behaviour — fully backward
    compatible for evaluate.py / the explainer.
    """
    slice_by_name = {name: (s, e) for (name, s, e) in (layout or [])}
    mods: Dict[str, Optional[torch.Tensor]] = {}
    for name in modality_names:
        if name == "traffic":
            mods[name] = X[..., traffic_channels]  # [B, T, N, len(channels)]
        elif M is not None and M.shape[-1] > 0 and name in slice_by_name:
            s, e = slice_by_name[name]
            mods[name] = M[..., s:e]               # [B, T, N, C_name]
        else:
            mods[name] = None                      # absent feed -> gated out
    return mods


# ---------------------------------------------------------------------------
# Phase 6 — modality sidecar loading (processed/<dataset>/mod_<name>.npz)
# ---------------------------------------------------------------------------
def modality_layout(dataset: str, modality_names: List[str]
                    ) -> Tuple[List[Tuple[str, int, int]], int]:
    """Channel layout for the sidecars that actually exist on disk.

    Returns (layout, total_channels) where layout = [(name, start, end), ...] in
    the order the modalities appear in the config (traffic excluded — it lives in
    X). A modality whose mod_<name>.npz is missing is simply skipped."""
    d = processed_dir(dataset)
    layout: List[Tuple[str, int, int]] = []
    start = 0
    for name in modality_names:
        if name == "traffic":
            continue
        npz = os.path.join(d, f"mod_{name}.npz")
        meta = os.path.join(d, f"mod_{name}.json")
        if not (os.path.exists(npz) and os.path.exists(meta)):
            continue
        with open(meta) as f:
            c = len(json.load(f)["channels"])
        layout.append((name, start, start + c))
        start += c
    return layout, start


def _load_modalities(dataset: str, split: str,
                     layout: List[Tuple[str, int, int]], total: int
                     ) -> torch.Tensor:
    """Concatenate the present sidecars for one split into [S,12,N,total].

    Always returns a tensor (width 0 if no sidecars) so downstream batching sees a
    fixed 3-tuple (X, Y, M) regardless of how many feeds are wired in."""
    if total == 0:
        # zero-width placeholder aligned to this split's sample count
        d = np.load(os.path.join(processed_dir(dataset), f"{split}.npz"))
        s, t, n = d["X"].shape[0], d["X"].shape[1], d["X"].shape[2]
        return torch.zeros((s, t, n, 0), dtype=torch.float32)
    arrs = []
    for name, _s, _e in layout:
        arrs.append(np.load(os.path.join(processed_dir(dataset), f"mod_{name}.npz"))[split])
    M = np.concatenate(arrs, axis=-1)              # [S,12,N,total]
    return torch.from_numpy(M).float()


def make_fusion_loaders(dataset: str, batch_size: int, modality_names: List[str],
                        num_workers: int = 0
                        ) -> Tuple[Dict[str, DataLoader], List[Tuple[str, int, int]]]:
    """Like make_loaders but each batch is (X, Y, M) with M the concatenated
    Phase-6 sidecar features. Returns (loaders, layout). If no sidecars exist this
    degrades exactly to traffic-only training (M has width 0)."""
    layout, total = modality_layout(dataset, modality_names)
    if layout:
        print(f"[loaders] fusion modalities for {dataset}: "
              + ", ".join(f"{n}[{s}:{e}]" for n, s, e in layout))
    else:
        print(f"[loaders] no sidecars for {dataset} -> traffic-only training")
    loaders = {}
    for split in ("train", "val", "test"):
        X, Y = _load_split(dataset, split)
        M = _load_modalities(dataset, split, layout, total)
        ds = TensorDataset(X, Y, M)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, drop_last=False)
    return loaders, layout
