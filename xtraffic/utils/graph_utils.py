"""Shared graph + time-series helpers for all Phase 1 data pipelines.

Every pipeline (METR-LA, PEMS-BAY, Chicago) imports from here so the adjacency
construction, z-score normalization, and sliding-window logic exist in EXACTLY
one place. If a reviewer questions the method, there is one function to point at.

Kept deliberately dependency-light (numpy only) and Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------
def gaussian_kernel_adjacency(
    dist_matrix: np.ndarray,
    normalized_k: float = 0.1,
) -> np.ndarray:
    """Thresholded Gaussian-kernel adjacency (DCRNN, Li et al. 2018, Eq. 10).

    A_ij = exp(-(dist_ij^2) / sigma^2), then any entry < normalized_k is set to 0
    to keep the graph sparse. sigma is the standard deviation of the finite
    (non-infinite) distances, which makes the kernel self-scaling to each city.

    Parameters
    ----------
    dist_matrix : [N, N] float array of pairwise road-network distances.
                  Unreachable pairs should be np.inf.
    normalized_k : sparsity threshold (kappa in the paper).

    Returns
    -------
    adj : [N, N] float32 weighted adjacency in [0, 1], with 1.0 on the diagonal.
    """
    n = dist_matrix.shape[0]                                    # N sensors
    finite = dist_matrix[~np.isinf(dist_matrix)].flatten()      # [num_finite]
    sigma = finite.std() if finite.size else 1.0               # scalar scale
    adj = np.exp(-np.square(dist_matrix / sigma))              # [N, N] in (0, 1]
    adj[adj < normalized_k] = 0.0                              # sparsify
    np.fill_diagonal(adj, 1.0)                                 # self-loops kept
    return adj.astype(np.float32)                             # [N, N]


def build_distance_matrix(
    sensor_ids: list,
    distances_df,
    default_inf: float = np.inf,
) -> np.ndarray:
    """Turn an edge list (from, to, cost) into a dense [N, N] distance matrix.

    distances_df columns: 'from', 'to', 'cost'. Sensor IDs are mapped to
    contiguous indices in the order given by `sensor_ids` (this ordering is the
    canonical node ordering used everywhere downstream).
    """
    n = len(sensor_ids)                                        # N
    id_to_idx = {int(sid): i for i, sid in enumerate(sensor_ids)}
    dist = np.full((n, n), default_inf, dtype=np.float64)      # [N, N] = inf
    np.fill_diagonal(dist, 0.0)                               # self-distance 0
    # distances_df has columns [from, to, cost] (positional; DCRNN header names vary).
    arr = distances_df.to_numpy()
    for f, t, c in arr:
        f, t = int(f), int(t)
        if f in id_to_idx and t in id_to_idx:
            dist[id_to_idx[f], id_to_idx[t]] = float(c)       # directed cost
    return dist                                               # [N, N]


# ---------------------------------------------------------------------------
# Normalization (train-statistics only)
# ---------------------------------------------------------------------------
class StandardScaler:
    """Z-score scaler fit on TRAIN data only.

    WHY train-only: mean/std are learned parameters of the pipeline. Computing
    them over val/test would leak information about the future distribution into
    training and inflate metrics — the same leakage rule as the chronological
    split. We fit on train, then apply the *same* mean/std to val and test.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "StandardScaler":
        # Fit only on the speed channel of the training split.
        return cls(mean=float(np.mean(train_values)), std=float(np.std(train_values)))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
def sliding_windows(
    series: np.ndarray,
    input_length: int,
    output_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a [T, N, F] time series into supervised (X, Y) windows.

    X[i] = steps [i : i+input_length)                -> shape [T_in, N, F]
    Y[i] = steps [i+input_length : i+in+out)         -> shape [T_out, N] (speed only)

    Returns
    -------
    X : [samples, input_length, N, F]
    Y : [samples, output_length, N]   (channel 0 = speed is the prediction target)
    """
    total_steps, n_nodes, n_features = series.shape           # [T, N, F]
    max_start = total_steps - input_length - output_length + 1
    xs, ys = [], []
    for i in range(max_start):
        xs.append(series[i:i + input_length])                 # [T_in, N, F]
        ys.append(series[i + input_length:i + input_length + output_length, :, 0])  # [T_out, N]
    X = np.stack(xs).astype(np.float32)                       # [samples, T_in, N, F]
    Y = np.stack(ys).astype(np.float32)                       # [samples, T_out, N]
    return X, Y


def chronological_split(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[slice, slice, slice]:
    """Return train/val/test index slices in strict time order (no overlap).

    Windows are built before splitting, so we cut the *window* indices. Note a
    subtle boundary effect: windows straddling the train/val cut share a few raw
    timesteps. We accept this (it is the standard DCRNN protocol) but document it
    in DIFFERENCES.md — it is far milder than shuffling.
    """
    n_train = int(round(n_samples * train_ratio))             # earliest chunk
    n_val = int(round(n_samples * val_ratio))                 # middle chunk
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n_samples)                  # latest chunk
    return train, val, test


def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters between two lat/lon points.

    Used by the Chicago pipeline to decide segment adjacency from endpoint
    coordinates (METR-LA/PEMS-BAY ship a precomputed road-network distance file;
    Chicago does not, so we approximate proximity geometrically).
    """
    r = 6_371_000.0                                           # Earth radius, m
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))
