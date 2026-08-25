"""What does the trained ST-GNN's explainer say about the METR-LA ROAD NETWORK?

Every previous faithfulness study in this repository asks a question about the
LLM. This module asks a question about the GRAPH: which sensors does the model
actually lean on when it forecasts a given sensor, and does that set look like
the adjacency matrix it was handed?

EXPLORATORY. The only external checks available are sensor metadata (lat/lon,
raw-speed pathologies) and the geometry of the adjacency itself. Nothing here
establishes ground truth about LA traffic, and no claim in this module should be
read as one.

--------------------------------------------------------------------------------
THREE FACTS ABOUT THE SETUP THAT CONSTRAIN EVERY INTERPRETATION
--------------------------------------------------------------------------------
(1) THE ADJACENCY IS A DISTANCE THRESHOLD, NOT "THE ROAD NETWORK".
    utils/graph_utils.gaussian_kernel_adjacency builds A_ij = exp(-(d_ij/sigma)^2)
    and zeroes anything below kappa=0.1. With sigma = 2584.5 m that is EXACTLY
    the rule "keep the pair iff directed road distance <= 3921.8 m". Verified by
    rebuilding A from the raw CSV: max edge distance 3921 m, min non-edge finite
    distance 3924 m, rebuilt A matches the committed file to 3e-8.
    => "off-adjacency" in this module means ">3.9 km by road", nothing more.

(2) ROAD DISTANCE IS MOSTLY MISSING OFF-ADJACENCY.
    distances_la_2012.csv covers 26.9% of the 42,642 ordered pairs. Among
    NON-adjacent pairs only 24.4% have a finite road cost. Haversine is recorded
    for the remainder as a SEPARATE column and nothing is derived from it —
    straight-line distance systematically understates road distance by an
    unknown factor, so mixing the two would silently bias any distance summary.

(3) THE MODEL'S SPATIAL RECEPTIVE FIELD IS THE COMPLETE GRAPH IN ONE HOP.
    XTrafficSTGNN diffuses over TWO supports (stgnn.py:289):
        semantic = sigmoid(alpha)*A_phys + (1-sigmoid(alpha))*softmax(relu(E E^T))
        adaptive = softmax(relu(nodevec1 @ nodevec2^T))
    Both are row-softmaxes, so every one of the 42,849 entries is strictly
    positive. gcn_order=2 x n_blocks=4 bounds diffusion at 8 hops over the
    PHYSICAL component only; through either learned support any node reaches any
    other immediately. Hop distance in the adjacency graph is therefore a
    descriptive coordinate, NOT a reachability limit — influence at hop > 8 or at
    an unreachable pair is expected by construction and is not, on its own,
    anomalous. It is reported separately because it was asked for, and because
    "how road-aligned is the learned structure" is still a real question.

--------------------------------------------------------------------------------
DIRECTION CONVENTION (easy to get backwards, so it is pinned here)
--------------------------------------------------------------------------------
GraphConv._nconv is einsum("bcnt,nm->bcmt", x, A): output node m sums over n with
weight A[n, m]. So A[s, t] is the weight with which SOURCE s feeds TARGET t, and
build_distance_matrix writes dist[from, to]. Throughout this module an edge is
the ordered pair (source, target) and is checked against A[source, target].

W is [targets x sources] because that is how it was specified; the edge helpers
convert to (source, target) pairs at the boundary.

Python 3.9 compatible.
"""
from __future__ import annotations

import collections
import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# The kernel constants, recovered from the committed pipeline rather than
# hard-coded: see gaussian_kernel_adjacency. Recomputed at run time in
# load_geometry so a different city or a rebuilt adjacency cannot silently
# invalidate them.
KAPPA = 0.1

# Regime thresholds, justified from the test-split speed distribution in
# docs/EXPLANATION_NETWORK_ANALYSIS.md: over 1,246,275 valid cells the
# percentiles are p10=34.2, p15=46.0, p20=53.1, median 62.9 mph. The jump from
# p10 to p15 is a genuine density gap between the congested tail and the
# free-flow mode, so the boundary is placed inside it. The DEAD BAND between 45
# and 55 is deliberate: windows in the gap are excluded from both regimes rather
# than assigned by a coin-flip's worth of signal. 50 mph is the project's own
# Phase-13 free-flow rule and sits inside the band.
CONGESTED_MAX_MPH = 45.0
FREEFLOW_MIN_MPH = 55.0

# METR-LA encodes missing observations as a raw 0. Anything at or below this in
# REAL mph space is the sentinel, not a measurement. (Phase 3 hit the mirror-image
# bug by testing validity in z-space, where the sentinel is ~-2.8, not ~0.)
MISSING_MAX_MPH = 1.0

REGIME_CONGESTED = "congested"
REGIME_FREEFLOW = "free_flow"
REGIMES = (REGIME_CONGESTED, REGIME_FREEFLOW)


# ---------------------------------------------------------------------------
# Geometry: adjacency, directed road distance, haversine, hop distance
# ---------------------------------------------------------------------------
class Geometry:
    """Everything static about the METR-LA graph, loaded once.

    Attributes
    ----------
    A        : [N,N] committed distance-kernel adjacency (weights in [0,1]).
    adj      : [N,N] bool, A>0 with the diagonal removed — THE edge set.
    road_m   : [N,N] directed road distance in metres, np.inf where absent.
    hav_m    : [N,N] haversine distance in metres, always finite.
    hops     : [N,N] int shortest-path hop count over `adj` (source->target),
               UNREACHABLE where no directed path exists.
    """

    UNREACHABLE = -1

    def __init__(self, A: np.ndarray, road_m: np.ndarray, hav_m: np.ndarray,
                 latlon: np.ndarray, sensor_ids: Sequence[int]):
        self.A = A
        self.n = A.shape[0]
        self.latlon = latlon
        self.sensor_ids = list(sensor_ids)
        self.off = ~np.eye(self.n, dtype=bool)                    # [N,N]
        self.adj = (A > 0) & self.off                             # [N,N] bool
        self.road_m = road_m
        self.hav_m = hav_m
        self.hops = _hop_matrix(self.adj)                         # [N,N] int
        # The distance cutoff the adjacency IS. Recovered, not assumed.
        finite = road_m[np.isfinite(road_m)]
        self.sigma_m = float(finite.std())
        self.cutoff_m = float(self.sigma_m * np.sqrt(-np.log(KAPPA)))

    def out_degree(self) -> np.ndarray:
        """Number of targets each source feeds — the degree used for the
        degree-matched baseline."""
        return self.adj.sum(axis=1).astype(int)                   # [N]

    def in_degree(self) -> np.ndarray:
        return self.adj.sum(axis=0).astype(int)                   # [N]

    def adjacency_pairs(self, targets: Sequence[int]) -> Set[Tuple[int, int]]:
        """The (source, target) edges of the given targets — the reference set."""
        return {(int(s), int(t))
                for t in targets for s in np.where(self.adj[:, t])[0]}


def _hop_matrix(adj: np.ndarray) -> np.ndarray:
    """BFS shortest-path hop count from every source to every target.

    Directed, following the influence direction source -> target. Returns
    Geometry.UNREACHABLE (-1) where no directed path exists. O(N * edges), which
    at N=207 is instant; done once and reused everywhere.
    """
    n = adj.shape[0]
    out = np.full((n, n), Geometry.UNREACHABLE, dtype=np.int32)
    nbrs = [np.where(adj[i])[0].tolist() for i in range(n)]       # successors
    for src in range(n):
        out[src, src] = 0
        frontier = [src]
        d = 0
        seen = {src}
        while frontier:
            d += 1
            nxt: List[int] = []
            for u in frontier:
                for v in nbrs[u]:
                    if v not in seen:
                        seen.add(v)
                        out[src, v] = d
                        nxt.append(v)
            frontier = nxt
    return out


def haversine_matrix(latlon: np.ndarray) -> np.ndarray:
    """[N,N] great-circle metres. Vectorised; only used for REPORTING pairs that
    have no road distance, never for deriving a quantity."""
    lat = np.radians(latlon[:, 0])[:, None]                       # [N,1]
    lon = np.radians(latlon[:, 1])[:, None]                       # [N,1]
    dlat = lat - lat.T                                            # [N,N]
    dlon = lon - lon.T                                            # [N,N]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2)
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_geometry(dataset: str = "metr_la") -> Geometry:
    """Load the committed adjacency + rebuild the directed road-distance matrix."""
    import pandas as pd

    from ..utils.graph_utils import build_distance_matrix
    from ..utils.io_utils import processed_dir, raw_dir

    p, r = processed_dir(dataset), raw_dir(dataset)
    A = np.load(os.path.join(p, "adjacency.npy"))
    with open(os.path.join(p, "node_meta.json")) as fh:
        meta = json.load(fh)
    latlon = np.asarray(meta["latlon"], dtype=float)
    road = build_distance_matrix(meta["sensor_ids"],
                                 pd.read_csv(os.path.join(r, "distances_la_2012.csv")))
    return Geometry(A, road, haversine_matrix(latlon), latlon, meta["sensor_ids"])


# ---------------------------------------------------------------------------
# Window selection — ONE shared set, used for every target
# ---------------------------------------------------------------------------
def select_windows(last_speed_mph: np.ndarray, tod_frac: np.ndarray,
                   n_windows: int = 24, seed: int = 42,
                   n_tod_bands: int = 6, n_congestion_strata: int = 2
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the shared window set, stratified by hour-of-day x NETWORK congestion.

    WHY NETWORK congestion and not the target's: the specification requires the
    SAME windows for every sensor, and "congested" is a property of a (sensor,
    window) pair. A shared set therefore cannot be stratified on a per-target
    regime. We stratify on a network-level statistic — the fraction of valid
    sensors below CONGESTED_MAX_MPH — and label each (target, window) pair by the
    TARGET's own speed afterwards (see regime_labels). That keeps the window set
    representative of the test split while still spanning quiet and busy states.

    The layout is n_tod_bands x n_congestion_strata cells with an equal number of
    replicates per cell, so `split_halves` can take one replicate from each cell
    and produce two halves that are balanced by construction rather than by luck.

    Parameters
    ----------
    last_speed_mph : [S, N] last observed speed of every sensor in every window.
    tod_frac       : [S] time-of-day in [0,1) (channel 1 of X at the last step).

    Returns
    -------
    (window_idx [n_windows], stratum_id [n_windows]) — both sorted by stratum.
    """
    n_cells = n_tod_bands * n_congestion_strata
    if n_windows % n_cells:
        raise ValueError(
            "n_windows={} is not divisible by n_tod_bands*n_congestion_strata={} "
            "— an unbalanced design would make the split-half unbalanced too"
            .format(n_windows, n_cells))
    per_cell = n_windows // n_cells

    valid = last_speed_mph > MISSING_MAX_MPH                      # [S,N]
    cong_frac = np.where(valid, last_speed_mph < CONGESTED_MAX_MPH, False).sum(1) \
        / np.maximum(valid.sum(1), 1)                             # [S]
    band = np.minimum((tod_frac * n_tod_bands).astype(int), n_tod_bands - 1)

    # Congestion strata as quantiles of the network congestion fraction, so the
    # split is defined by the data rather than by an absolute cut.
    edges = np.quantile(cong_frac, np.linspace(0, 1, n_congestion_strata + 1)[1:-1])
    cstr = np.searchsorted(edges, cong_frac, side="right")        # [S] in [0,K)

    rng = np.random.RandomState(seed)
    chosen: List[int] = []
    strata: List[int] = []
    for b in range(n_tod_bands):
        for c in range(n_congestion_strata):
            pool = np.where((band == b) & (cstr == c))[0]
            if len(pool) < per_cell:
                raise ValueError(
                    "stratum (tod={}, congestion={}) has {} windows, need {}"
                    .format(b, c, len(pool), per_cell))
            pick = rng.choice(pool, size=per_cell, replace=False)
            chosen.extend(int(i) for i in np.sort(pick))
            strata.extend([b * n_congestion_strata + c] * per_cell)
    return np.asarray(chosen, dtype=int), np.asarray(strata, dtype=int)


def split_halves(strata: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split window POSITIONS into two disjoint, stratum-balanced halves.

    Returns index arrays into the window list (not window ids), so the caller can
    slice whatever per-window array it holds.
    """
    rng = np.random.RandomState(seed)
    a: List[int] = []
    b: List[int] = []
    for s in np.unique(strata):
        pos = np.where(strata == s)[0]
        pos = pos[rng.permutation(len(pos))]
        a.extend(int(i) for i in pos[: len(pos) // 2])
        b.extend(int(i) for i in pos[len(pos) // 2:])
    return np.asarray(sorted(a), dtype=int), np.asarray(sorted(b), dtype=int)


def regime_labels(target_speed_mph: np.ndarray) -> np.ndarray:
    """Label each window for ONE target by that target's own last observed speed.

    Returns an array of REGIME_CONGESTED / REGIME_FREEFLOW / "" (excluded:
    missing sentinel, or inside the dead band).
    """
    out = np.full(len(target_speed_mph), "", dtype=object)
    valid = target_speed_mph > MISSING_MAX_MPH
    out[valid & (target_speed_mph < CONGESTED_MAX_MPH)] = REGIME_CONGESTED
    out[valid & (target_speed_mph >= FREEFLOW_MIN_MPH)] = REGIME_FREEFLOW
    return out


# ---------------------------------------------------------------------------
# Aggregation: per-solve importance vectors -> the influence matrix W
# ---------------------------------------------------------------------------
def build_W(node_imp: Dict[int, np.ndarray], regimes: Dict[int, np.ndarray],
            regime: str, n_nodes: int, min_windows: int = 4
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean importance per (target, source) over that target's windows in `regime`.

    Parameters
    ----------
    node_imp : target -> [n_windows, N] raw sigmoid masks, one row per window.
    regimes  : target -> [n_windows] labels from regime_labels.
    min_windows : a target's row is only populated if it has at least this many
        windows in the regime. With 24 shared windows and a 14.4% congestion base
        rate the median target sees only a handful of congested windows; a row
        built from one or two is noise wearing a mean's clothing. Targets below
        the floor are returned as all-zero rows and COUNTED, never silently
        dropped — `n_used` is what the report quotes.

    Returns
    -------
    (W [T_all x N] where row t is target t, n_used [T_all] windows behind each row)
    """
    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    n_used = np.zeros(n_nodes, dtype=int)
    for t, imp in node_imp.items():
        sel = regimes[t] == regime
        if sel.sum() < min_windows:
            continue
        W[t] = imp[sel].mean(axis=0)
        n_used[t] = int(sel.sum())
    return W, n_used


def active_targets(n_used: np.ndarray) -> np.ndarray:
    """Targets whose W row is populated."""
    return np.where(n_used > 0)[0]


# ---------------------------------------------------------------------------
# Thresholding W into an effective edge set
# ---------------------------------------------------------------------------
def edges_top_k(W: np.ndarray, targets: Sequence[int], k: int = 8
                ) -> Set[Tuple[int, int]]:
    """Top-k sources per target, target excluded from its own list.

    k=8 matches the project's committed top_k so the effective edge set is
    directly comparable to what every explanation JSON reports.
    """
    out: Set[Tuple[int, int]] = set()
    for t in targets:
        v = W[t].copy()
        v[t] = -np.inf                                            # never self
        order = np.argsort(-v)[:k]
        out.update((int(s), int(t)) for s in order if np.isfinite(v[s]) and v[s] > 0)
    return out


def edges_cumulative_mass(W: np.ndarray, targets: Sequence[int],
                          frac: float = 0.80) -> Set[Tuple[int, int]]:
    """Smallest set of sources per target covering `frac` of that target's
    off-self importance mass.

    Reported alongside top-k because the two disagree in an informative way: a
    flat importance vector needs many sources to reach 80%, a peaked one needs
    few. top-k fixes the count and lets the mass vary; this fixes the mass and
    lets the count vary.
    """
    out: Set[Tuple[int, int]] = set()
    for t in targets:
        v = W[t].copy()
        v[t] = 0.0
        total = v.sum()
        if total <= 0:
            continue
        acc = 0.0
        for s in np.argsort(-v):
            if v[s] <= 0 or acc >= frac * total:
                break
            out.add((int(s), int(t)))
            acc += v[s]
    return out


def self_mass(W: np.ndarray, targets: Sequence[int]) -> np.ndarray:
    """Each target's own diagonal share: W[t,t] / sum_s W[t,s].

    Reported separately because Phase 17 found the 30-min forecast is dominated
    by the target's own recent speed. If the diagonal carries most of the mass,
    every off-diagonal statistic here is describing a small residue and must say
    so.
    """
    out = np.full(len(targets), np.nan)
    for i, t in enumerate(targets):
        tot = W[t].sum()
        out[i] = (W[t, t] / tot) if tot > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Comparison against the adjacency, and against baselines
# ---------------------------------------------------------------------------
def compare_to_adjacency(edges: Set[Tuple[int, int]], geo: Geometry,
                         targets: Sequence[int]) -> Dict[str, float]:
    """Precision / recall / Jaccard of an effective edge set against the
    adjacency edges of the SAME targets (never the whole graph — that would
    charge the effective set with recall it was never asked to cover)."""
    ref = geo.adjacency_pairs(targets)
    tp = len(edges & ref)
    return {
        "n_effective": len(edges),
        "n_reference": len(ref),
        "n_true_positive": tp,
        "precision": tp / len(edges) if edges else float("nan"),
        "recall": tp / len(ref) if ref else float("nan"),
        "jaccard": tp / len(edges | ref) if (edges or ref) else float("nan"),
        "n_off_adjacency": len(edges - ref),
    }


def jaccard(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]]) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def baseline_uniform(geo: Geometry, targets: Sequence[int], k: int = 8,
                     seed: int = 42) -> Set[Tuple[int, int]]:
    """(a) k sources drawn uniformly from the other N-1 nodes, per target."""
    rng = np.random.RandomState(seed)
    out: Set[Tuple[int, int]] = set()
    for t in targets:
        pool = np.array([i for i in range(geo.n) if i != t])
        out.update((int(s), int(t)) for s in rng.choice(pool, size=k, replace=False))
    return out


def baseline_degree_matched(geo: Geometry, targets: Sequence[int], k: int = 8,
                            seed: int = 42) -> Set[Tuple[int, int]]:
    """(b) k sources drawn with probability proportional to adjacency OUT-degree.

    Controls for the boring explanation "the effective edges just pick hubs". If
    the real edge set scores no better than this, its apparent structure is
    degree structure.
    """
    rng = np.random.RandomState(seed)
    deg = geo.out_degree().astype(float)                          # [N]
    out: Set[Tuple[int, int]] = set()
    for t in targets:
        w = deg.copy()
        w[t] = 0.0
        if w.sum() <= 0:
            continue
        p = w / w.sum()
        pick = rng.choice(geo.n, size=min(k, int((p > 0).sum())), replace=False, p=p)
        out.update((int(s), int(t)) for s in pick)
    return out


def baseline_nearest_road(geo: Geometry, targets: Sequence[int], k: int = 8
                          ) -> Tuple[Set[Tuple[int, int]], Dict[str, int]]:
    """(c) the k nearest sources by DIRECTED ROAD distance.

    Deterministic — no seed. Pairs with no road distance are not eligible, and
    haversine is NOT substituted (see module docstring, fact 2). Targets with
    fewer than k road-connected sources contribute what they have; the shortfall
    is returned rather than hidden.
    """
    out: Set[Tuple[int, int]] = set()
    short = 0
    for t in targets:
        d = geo.road_m[:, t].copy()                               # sources -> t
        d[t] = np.inf
        cand = np.where(np.isfinite(d))[0]
        if len(cand) < k:
            short += 1
        order = cand[np.argsort(d[cand])][:k]
        out.update((int(s), int(t)) for s in order)
    return out, {"targets_with_fewer_than_k_road_sources": short}


# ---------------------------------------------------------------------------
# Hop stratification
# ---------------------------------------------------------------------------
def hop_of(edges: Set[Tuple[int, int]], geo: Geometry) -> Dict[Tuple[int, int], int]:
    return {(s, t): int(geo.hops[s, t]) for (s, t) in edges}


def hop_bucket(h: int, k_hops: int = 8) -> str:
    """Bucket label. `k_hops` is gcn_order*n_blocks = the PHYSICAL-path diffusion
    bound; see module docstring fact (3) for why it is not a reachability limit."""
    if h == Geometry.UNREACHABLE:
        return "unreachable"
    if h == 0:
        return "self"
    if h <= k_hops:
        return str(h)
    return ">{}".format(k_hops)


def stratify_by_hop(edges: Set[Tuple[int, int]], geo: Geometry, k_hops: int = 8
                    ) -> "collections.OrderedDict":
    """edge set -> ordered {bucket: set of edges}, buckets in hop order."""
    by: Dict[str, Set[Tuple[int, int]]] = collections.defaultdict(set)
    for (s, t) in edges:
        by[hop_bucket(int(geo.hops[s, t]), k_hops)].add((s, t))
    order = ["self"] + [str(i) for i in range(1, k_hops + 1)] + \
            [">{}".format(k_hops), "unreachable"]
    return collections.OrderedDict((b, by[b]) for b in order if b in by)


# The label for the hop->k / unreachable stratum.
#
# It was originally "beyond-K", which implied a limit of the architecture. It is
# not one: the mixed support the model actually diffuses over is dense (measured
# below), so every pair is reachable in a single hop and nothing in this stratum
# is architecturally surprising.
#
# The replacement names what the set geometrically IS. Note it is NOT simply
# ">3.9 km by road" — that describes every non-adjacent pair, including hop-2
# ones. An adjacency hop is a step of <= the kernel radius, so hop > k means more
# than k CHAINED steps of that size.
BEYOND_K_LABEL = "beyond {k} kernel radii (>{k} chained hops of <= {r:.1f} km), or unreachable"


def beyond_k_label(geo: "Geometry", k_hops: int = 8) -> str:
    return BEYOND_K_LABEL.format(k=k_hops, r=geo.cutoff_m / 1000.0)


def beyond_k_set(edges: Set[Tuple[int, int]], geo: Geometry, k_hops: int = 8
                 ) -> Set[Tuple[int, int]]:
    """Sources more than `k_hops` chained kernel-radius steps from the target in
    the adjacency graph, or with no directed path to it at all.

    NOT an architectural limit — see BEYOND_K_LABEL. Reported because "where does
    influence sit relative to the road graph" is still a real question.
    """
    return {(s, t) for (s, t) in edges
            if geo.hops[s, t] == Geometry.UNREACHABLE or geo.hops[s, t] > k_hops}


def beyond_k_base_rate(geo: Geometry, targets: Sequence[int], k_hops: int = 8
                       ) -> float:
    """Share of all candidate (source, target) pairs for these targets that fall
    in the stratum — the rate an edge set with no spatial preference would show.

    Without this the stratum's share of an edge set is uninterpretable: 22% of
    METR-LA's ordered pairs are already in it.
    """
    n = tot = 0
    for t in targets:
        h = geo.hops[:, t]
        cand = np.ones(geo.n, dtype=bool)
        cand[t] = False
        tot += int(cand.sum())
        n += int(((h == Geometry.UNREACHABLE) | (h > k_hops))[cand].sum())
    return n / tot if tot else float("nan")


# ---------------------------------------------------------------------------
# Edge attribution — the per-edge rows the report must list in full
# ---------------------------------------------------------------------------
def edge_rows(edges: Set[Tuple[int, int]], W: np.ndarray, geo: Geometry,
              regime: str, k_hops: int = 8) -> List[Dict[str, object]]:
    """One row per edge. Road distance and haversine are SEPARATE columns and the
    haversine column is never used to fill a gap in the road column."""
    rows: List[Dict[str, object]] = []
    for (s, t) in sorted(edges, key=lambda p: -W[p[1], p[0]]):
        h = int(geo.hops[s, t])
        road = geo.road_m[s, t]
        rows.append({
            "regime": regime,
            "source": int(s),
            "target": int(t),
            "source_sensor_id": geo.sensor_ids[s],
            "target_sensor_id": geo.sensor_ids[t],
            "importance": round(float(W[t, s]), 6),
            "in_adjacency": bool(geo.adj[s, t]),
            "hops": h,
            "hop_bucket": hop_bucket(h, k_hops),
            "road_distance_m": (round(float(road), 1) if np.isfinite(road) else None),
            "has_road_distance": bool(np.isfinite(road)),
            "haversine_m_reference_only": round(float(geo.hav_m[s, t]), 1),
            "source_lat": round(float(geo.latlon[s, 0]), 5),
            "source_lon": round(float(geo.latlon[s, 1]), 5),
            "target_lat": round(float(geo.latlon[t, 0]), 5),
            "target_lon": round(float(geo.latlon[t, 1]), 5),
        })
    return rows
