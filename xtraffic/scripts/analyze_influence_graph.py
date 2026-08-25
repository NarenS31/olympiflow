"""STAGE 2 — aggregate the explainer solves into the influence graph and report.

    python -m xtraffic.scripts.analyze_influence_graph --run <run_id>
    python -m xtraffic.scripts.analyze_influence_graph --run <run_id> --quick
        (pooled split-half only; safe to run against a PARTIAL Stage-1 run)

Reads a Stage-1 run directory (append-only, gitignored because the raw arrays are
large) and writes derived tables, figures and report.md into
evaluation/results/processed/<run_id>/, which IS committed.

WHAT IS AND IS NOT CLAIMED
    This reports what the model's explanations lean on, whether that replicates
    across two disjoint halves of the window sample, and what the external checks
    (adjacency geometry, sensor metadata, raw-speed pathologies) say about it.
    It does not claim a discovery, and it does not interpret. Every edge above
    threshold is listed; nothing is curated.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ..evaluation import influence_graph as ig
from ..reproducibility import provenance, run_dir
from ..utils.io_utils import PKG_ROOT, processed_dir, raw_dir

REPO_ROOT = os.path.dirname(PKG_ROOT)
TOP_K = 8                 # matches the committed explainer top_k
MASS_FRAC = 0.80
MIN_WINDOWS_FULL = 4      # a regime row needs this many windows to be built
MIN_WINDOWS_HALF = 2      # ... and this many within a split half
K_HOPS = 8                # gcn_order(2) x n_blocks(4): the PHYSICAL-path bound


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_solves(path: str) -> Dict[str, object]:
    sol = os.path.join(path, "solves")
    files = sorted(f for f in os.listdir(sol) if f.startswith("target_"))
    node_imp: Dict[int, np.ndarray] = {}
    tgt_speed: Dict[int, np.ndarray] = {}
    window_idx = strata = None
    for f in files:
        d = np.load(os.path.join(sol, f))
        t = int(d["target"])
        node_imp[t] = d["node_imp"].astype(np.float64)
        tgt_speed[t] = d["target_speed_mph"].astype(np.float64)
        window_idx, strata = d["window_idx"], d["stratum"]
    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    order = []
    wpath = os.path.join(path, "windows.json")
    if os.path.exists(wpath):
        with open(wpath) as fh:
            order = json.load(fh).get("target_order", [])
    return {"node_imp": node_imp, "target_speed": tgt_speed,
            "window_idx": window_idx, "strata": strata, "manifest": man,
            "target_order": order}


def limit_to_prefix(data: Dict[str, object], n: int) -> Dict[str, object]:
    """Keep only the first `n` targets of the run's SEEDED SHUFFLE ORDER.

    Reconstructs what a partial run would have contained, exactly and
    reproducibly. Stage 1 dispatches targets in `windows.json.target_order` — a
    seeded permutation — precisely so that a prefix is an unbiased sample of the
    graph rather than "the first n node indices", which in METR-LA correlate with
    position in the sensor file and therefore with geography.

    Used for sample-size sensitivity: does a conclusion drawn at n=207 also hold
    at n=64? If it does not, the n=207 conclusion is the one to keep, but the
    instability is worth knowing.
    """
    order = (data.get("target_order") or [])[:n]
    if not order:
        raise ValueError("run has no recorded target_order; cannot rebuild a prefix")
    keep = set(int(t) for t in order)
    out = dict(data)
    out["node_imp"] = {t: v for t, v in data["node_imp"].items() if t in keep}
    out["target_speed"] = {t: v for t, v in data["target_speed"].items() if t in keep}
    out["prefix_n"] = len(out["node_imp"])
    return out


def regimes_for(tgt_speed: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {t: ig.regime_labels(v) for t, v in tgt_speed.items()}


# ---------------------------------------------------------------------------
# Step 1 — one regime
# ---------------------------------------------------------------------------
def analyse_regime(regime: str, node_imp, regimes, geo: ig.Geometry,
                   strata: np.ndarray, seed: int = 42) -> Dict[str, object]:
    W, n_used = ig.build_W(node_imp, regimes, regime, geo.n,
                           min_windows=MIN_WINDOWS_FULL)
    tg = ig.active_targets(n_used)
    out: Dict[str, object] = {
        "regime": regime,
        "n_targets_active": int(len(tg)),
        "n_targets_below_window_floor": int(
            sum(1 for t in node_imp if (regimes[t] == regime).sum() < MIN_WINDOWS_FULL)),
        "min_windows_required": MIN_WINDOWS_FULL,
        "windows_per_active_target": {
            "mean": round(float(n_used[tg].mean()), 2) if len(tg) else None,
            "min": int(n_used[tg].min()) if len(tg) else None,
            "max": int(n_used[tg].max()) if len(tg) else None,
        },
    }
    if not len(tg):
        return out

    # --- diagonal: how much of each target's mass is its own? ----------------
    sm = ig.self_mass(W, tg)
    out["self_mass"] = {
        "mean": round(float(np.nanmean(sm)), 4),
        "median": round(float(np.nanmedian(sm)), 4),
        "min": round(float(np.nanmin(sm)), 4),
        "max": round(float(np.nanmax(sm)), 4),
        "chance_1_over_N": round(1.0 / geo.n, 4),
    }

    # --- how flat is the learned mask? ---------------------------------------
    # The headline property of the explainer's output, and the reason the two
    # threshold rules disagree so hard: a peaked mask needs a handful of sources
    # to reach MASS_FRAC, a flat one needs most of the graph. Reported as a
    # DISTRIBUTION, not a mean, because a mean hides whether flatness is uniform
    # across targets or driven by a subset.
    need = []
    for t in tg:
        v = W[t].copy()
        v[t] = 0.0
        tot = v.sum()
        if tot <= 0:
            continue
        c = np.cumsum(np.sort(v)[::-1])
        need.append(int(np.searchsorted(c, MASS_FRAC * tot) + 1))
    if need:
        need_a = np.asarray(need)
        out["sources_for_mass_frac"] = {
            "mass_frac": MASS_FRAC,
            "of_available": geo.n - 1,
            "mean": round(float(need_a.mean()), 1),
            "std": round(float(need_a.std()), 1),
            "min": int(need_a.min()), "max": int(need_a.max()),
            "percentiles": {str(p): int(np.percentile(need_a, p))
                            for p in (5, 25, 50, 75, 95)},
            "per_target": {int(t): int(n) for t, n in zip(tg, need)},
        }

    # --- effective edge sets --------------------------------------------------
    e_top = ig.edges_top_k(W, tg, k=TOP_K)
    e_mass = ig.edges_cumulative_mass(W, tg, frac=MASS_FRAC)
    out["edges"] = {
        "top_k": ig.compare_to_adjacency(e_top, geo, tg),
        "cumulative_mass": ig.compare_to_adjacency(e_mass, geo, tg),
    }
    out["edges"]["cumulative_mass"]["sources_per_target_mean"] = round(
        len(e_mass) / len(tg), 2)

    # --- baselines ------------------------------------------------------------
    b_uni = ig.baseline_uniform(geo, tg, k=TOP_K, seed=seed)
    b_deg = ig.baseline_degree_matched(geo, tg, k=TOP_K, seed=seed)
    b_near, near_info = ig.baseline_nearest_road(geo, tg, k=TOP_K)
    out["baselines"] = {
        "uniform_random": ig.compare_to_adjacency(b_uni, geo, tg),
        "degree_matched_random": ig.compare_to_adjacency(b_deg, geo, tg),
        "nearest_by_road_distance": ig.compare_to_adjacency(b_near, geo, tg),
    }
    out["baselines"]["nearest_by_road_distance"].update(near_info)
    out["baselines"]["overlap_effective_vs_nearest_road"] = round(
        ig.jaccard(e_top, b_near), 4)

    # --- hop stratification ---------------------------------------------------
    by_hop = ig.stratify_by_hop(e_top, geo, K_HOPS)
    ref_hops = collections.Counter(
        ig.hop_bucket(int(geo.hops[s, t]), K_HOPS)
        for (s, t) in geo.adjacency_pairs(tg))
    out["hop_strata"] = [{
        "bucket": b,
        "n_effective_edges": len(es),
        "share_of_effective": round(len(es) / len(e_top), 4) if e_top else None,
        "n_adjacency_edges_same_bucket": int(ref_hops.get(b, 0)),
    } for b, es in by_hop.items()]

    # --- the far stratum: beyond K chained kernel radii, or unreachable -------
    # NOT an architectural limit — the mixed support is dense (Stage 0b). The
    # base rate is what makes the share interpretable: 22% of METR-LA's ordered
    # pairs are already in this stratum.
    bk = ig.beyond_k_set(e_top, geo, K_HOPS)
    base = ig.beyond_k_base_rate(geo, tg, K_HOPS)
    share = len(bk) / len(e_top) if e_top else None
    out["beyond_k"] = {
        "k_hops": K_HOPS,
        "label": ig.beyond_k_label(geo, K_HOPS),
        "n_edges": len(bk),
        "share_of_effective": round(share, 4) if share is not None else None,
        "base_rate_for_these_targets": round(base, 4),
        "enrichment_over_base_rate": round(share / base, 3)
        if (share is not None and base) else None,
        "n_unreachable": int(sum(1 for (s, t) in bk
                                 if geo.hops[s, t] == ig.Geometry.UNREACHABLE)),
    }

    # --- split-half stability -------------------------------------------------
    out["stability"] = split_half(node_imp, regimes, regime, geo, strata, seed)

    return out


# Minimum targets present in BOTH split halves before a regime-level claim is
# made about it. Set by instruction, enforced in code rather than in prose so a
# thin cell cannot quietly acquire a conclusion.
MIN_TARGETS_FOR_REGIME_CLAIM = 40


def split_half(node_imp, regimes, regime: str, geo: ig.Geometry,
               strata: np.ndarray, seed: int = 42) -> Dict[str, object]:
    """Recompute the effective edge set on two disjoint, stratum-balanced halves.

    THE POINT: a per-solve top-k that reorders under float noise (see the
    CPU/MPS divergence in report.md) tells us nothing on its own. What matters is
    whether the AGGREGATE survives being computed on completely different
    windows. If the off-adjacency edges do not replicate, they are noise and the
    report must say so at the top, not in a caveat.
    """
    ha, hb = ig.split_halves(strata, seed=seed)
    res: Dict[str, object] = {"half_a_windows": len(ha), "half_b_windows": len(hb)}
    sets = []
    for half in (ha, hb):
        ni = {t: v[half] for t, v in node_imp.items()}
        rg = {t: v[half] for t, v in regimes.items()}
        W, n_used = ig.build_W(ni, rg, regime, geo.n, min_windows=MIN_WINDOWS_HALF)
        sets.append((W, ig.active_targets(n_used)))
    common = sorted(set(sets[0][1].tolist()) & set(sets[1][1].tolist()))
    res["n_targets_in_both_halves"] = len(common)
    res["min_windows_per_half"] = MIN_WINDOWS_HALF
    if not common:
        res["note"] = "no target had enough windows in both halves"
        return res
    ea = ig.edges_top_k(sets[0][0], common, k=TOP_K)
    eb = ig.edges_top_k(sets[1][0], common, k=TOP_K)
    res["jaccard_all_edges"] = round(ig.jaccard(ea, eb), 4)
    # the same question restricted to the candidate set
    ba, bb = ig.beyond_k_set(ea, geo, K_HOPS), ig.beyond_k_set(eb, geo, K_HOPS)
    res["jaccard_beyond_k"] = round(ig.jaccard(ba, bb), 4) if (ba or bb) else None
    res["n_beyond_k_half_a"], res["n_beyond_k_half_b"] = len(ba), len(bb)
    # and to the off-adjacency subset
    ref = geo.adjacency_pairs(common)
    oa, ob = ea - ref, eb - ref
    res["jaccard_off_adjacency"] = round(ig.jaccard(oa, ob), 4) if (oa or ob) else None
    res["n_off_adjacency_half_a"], res["n_off_adjacency_half_b"] = len(oa), len(ob)
    # a chance reference for the Jaccard itself: two independent uniform draws
    ra = ig.baseline_uniform(geo, common, k=TOP_K, seed=seed + 1)
    rb = ig.baseline_uniform(geo, common, k=TOP_K, seed=seed + 2)
    res["jaccard_two_random_draws_reference"] = round(ig.jaccard(ra, rb), 4)
    return res


# ---------------------------------------------------------------------------
# Step 3 — per-sensor mass concentration + metadata / raw-data cross-check
# ---------------------------------------------------------------------------
def _learned_W(checkpoint: str) -> Tuple[np.ndarray, float]:
    """A_sem as [target, source], plus sigmoid(alpha). Lazy import: the
    learned-graph module imports from this one."""
    import torch

    from .analyze_learned_graph import semantic_graph
    ck = torch.load(os.path.join(PKG_ROOT, checkpoint), map_location="cpu",
                    weights_only=False)
    S, _, alpha = semantic_graph(ck["model_state"])
    return S.T.copy(), alpha


def _learned_rank(WL: np.ndarray, s: int, t: int) -> int:
    """Rank of source s for target t among all non-self sources, 1 = strongest."""
    v = WL[t].copy()
    v[t] = -np.inf
    order = np.argsort(-v)
    return int(np.where(order == s)[0][0]) + 1


def beyond_k_survivors(node_imp, regimes, regime: str, geo: ig.Geometry,
                       strata: np.ndarray, checkpoint: str, seed: int = 42
                       ) -> Dict[str, object]:
    """Do the beyond-K edges that SURVIVE both split halves sit high in the
    model's own learned graph?

    THE QUESTION: an edge that replicates across disjoint windows is the only
    kind worth asking about. If those survivors rank near the top of A_sem, the
    explainer is recovering the model's learned adjacency. If they rank no better
    than one-half-only edges or than random pairs from the same stratum, it is
    not, and the survivors are then just a list of pairs to be described
    geometrically.

    Three groups, all drawn from the same targets so the comparison is matched:
      intersection  - beyond-K in BOTH halves
      one_half_only - beyond-K in exactly one half
      random        - beyond-K pairs sampled uniformly per target, matched in
                      count to the intersection
    """
    ha, hb = ig.split_halves(strata, seed=seed)
    sets, actives = [], []
    for half in (ha, hb):
        ni = {t: v[half] for t, v in node_imp.items()}
        rg = {t: v[half] for t, v in regimes.items()}
        W, n_used = ig.build_W(ni, rg, regime, geo.n, min_windows=MIN_WINDOWS_HALF)
        tg = ig.active_targets(n_used)
        sets.append(ig.beyond_k_set(ig.edges_top_k(W, tg, k=TOP_K), geo, K_HOPS))
        actives.append(set(tg.tolist()))
    common = sorted(actives[0] & actives[1])
    inter = sets[0] & sets[1]
    one_only = (sets[0] | sets[1]) - inter

    out: Dict[str, object] = {
        "regime": regime,
        "n_targets_in_both_halves": len(common),
        "n_beyond_k_half_a": len(sets[0]),
        "n_beyond_k_half_b": len(sets[1]),
        "n_intersection": len(inter),
        "n_one_half_only": len(one_only),
        "survival_rate": round(len(inter) / len(sets[0] | sets[1]), 4)
        if (sets[0] | sets[1]) else None,
    }
    if not inter:
        out["note"] = "no beyond-K edge survived both halves"
        return out

    # matched random draw from the same stratum, same targets, same per-target count
    rng = np.random.RandomState(seed)
    per_t = collections.Counter(t for (_, t) in inter)
    rnd: Set[Tuple[int, int]] = set()
    for t, cnt in per_t.items():
        h = geo.hops[:, t]
        cand = np.where(((h == ig.Geometry.UNREACHABLE) | (h > K_HOPS))
                        & (np.arange(geo.n) != t))[0]
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, size=min(cnt, len(cand)), replace=False)
        rnd.update((int(s), int(t)) for s in pick)

    WL, alpha = _learned_W(checkpoint)
    for name, es in (("intersection", inter), ("one_half_only", one_only),
                     ("random_beyond_k", rnd)):
        if not es:
            out[name] = {"n": 0}
            continue
        w = np.array([WL[t, s] for (s, t) in es])
        r = np.array([_learned_rank(WL, s, t) for (s, t) in es])
        out[name] = {
            "n": len(es),
            "learned_weight": {"mean": float(w.mean()), "median": float(np.median(w))},
            "learned_rank_of_206": {
                "mean": round(float(r.mean()), 1),
                "median": int(np.median(r)),
                "min": int(r.min()), "max": int(r.max()),
                "share_in_top_8": round(float((r <= TOP_K).mean()), 4),
                "share_in_top_20": round(float((r <= 20).mean()), 4),
                "share_in_top_half": round(float((r <= 103).mean()), 4),
            },
        }
    out["chance_rank_reference"] = {
        "uniform_mean_rank": 103.5, "share_in_top_8_if_random": round(TOP_K / 206, 4)}
    out["sigmoid_alpha"] = round(alpha, 4)
    # every surviving edge, with geometry — listed whatever the verdict
    out["intersection_edges"] = sorted(
        [{"source": int(s), "target": int(t),
          "source_sensor_id": geo.sensor_ids[s], "target_sensor_id": geo.sensor_ids[t],
          "hops": (None if geo.hops[s, t] == ig.Geometry.UNREACHABLE
                   else int(geo.hops[s, t])),
          "unreachable": bool(geo.hops[s, t] == ig.Geometry.UNREACHABLE),
          "learned_rank_of_206": _learned_rank(WL, s, t),
          "learned_weight": float(WL[t, s]),
          "road_distance_m": (round(float(geo.road_m[s, t]), 1)
                              if np.isfinite(geo.road_m[s, t]) else None),
          "haversine_m_reference_only": round(float(geo.hav_m[s, t]), 1),
          "source_lat": round(float(geo.latlon[s, 0]), 5),
          "source_lon": round(float(geo.latlon[s, 1]), 5),
          "target_lat": round(float(geo.latlon[t, 0]), 5),
          "target_lon": round(float(geo.latlon[t, 1]), 5)}
         for (s, t) in inter],
        key=lambda d: d["learned_rank_of_206"])
    return out


def compare_to_learned(W: np.ndarray, targets: Sequence[int], geo: ig.Geometry,
                       checkpoint: str) -> Dict[str, object]:
    """Per-target agreement between the EXPLAINER's W and the model's own learned
    semantic graph A_sem.

    These are different objects — one is a behavioural attribution for a specific
    prediction, the other is a static parameter — so agreement is not required and
    disagreement is not an error. It is reported because both were produced by the
    same checkpoint and both are read as "which sensors matter for this one".

    Imported lazily: analyze_learned_graph imports from this module, so a
    top-level import would be circular.
    """
    import torch

    from .analyze_learned_graph import semantic_graph

    ck = torch.load(os.path.join(PKG_ROOT, checkpoint), map_location="cpu",
                    weights_only=False)
    S, _, alpha = semantic_graph(ck["model_state"])
    WL = S.T                                       # [source,target] -> [target,source]
    spear, jac = [], []
    for t in targets:
        a, b = W[t].copy(), WL[t].copy()
        a[t] = b[t] = -np.inf                      # exclude self from both
        ra = np.argsort(np.argsort(-a))
        rb = np.argsort(np.argsort(-b))
        ok = np.isfinite(a) & np.isfinite(b)
        spear.append(float(np.corrcoef(ra[ok], rb[ok])[0, 1]))
        sa = set(int(i) for i in np.argsort(-a)[:TOP_K])
        sb = set(int(i) for i in np.argsort(-b)[:TOP_K])
        jac.append(len(sa & sb) / len(sa | sb))
    sp, jc = np.asarray(spear), np.asarray(jac)
    return {
        "checkpoint": checkpoint,
        "sigmoid_alpha": round(alpha, 4),
        "n_targets": len(targets),
        "spearman": {"mean": round(float(sp.mean()), 4),
                     "median": round(float(np.median(sp)), 4),
                     "min": round(float(sp.min()), 4),
                     "max": round(float(sp.max()), 4)},
        "top_k_jaccard": {"k": TOP_K, "mean": round(float(jc.mean()), 4),
                          "median": round(float(np.median(jc)), 4),
                          "min": round(float(jc.min()), 4),
                          "max": round(float(jc.max()), 4),
                          "n_targets_with_zero_overlap": int((jc == 0).sum())},
        "per_target": {int(t): {"spearman": round(float(s), 4),
                                "top_k_jaccard": round(float(j), 4)}
                       for t, s, j in zip(targets, spear, jac)},
    }


def mass_decomposition(W: np.ndarray, targets: Sequence[int], geo: ig.Geometry
                       ) -> pd.DataFrame:
    """Split each target's OFF-SELF importance mass three ways.

    Adjacency IS "directed road distance <= cutoff", so an adjacency-based split
    and a distance-based split are the same split. The honest decomposition is
    therefore three-way, separating "far by road" from "no road path at all" —
    the latter is an absence of data, not a measurement of distance.
    """
    rows = []
    for t in targets:
        v = W[t].copy()
        self_m = v[t]
        v[t] = 0.0
        tot = v.sum()
        if tot <= 0:
            continue
        adj_m = v[geo.adj[:, t]].sum()
        far = (~geo.adj[:, t]) & np.isfinite(geo.road_m[:, t])
        far[t] = False
        nopath = (~geo.adj[:, t]) & ~np.isfinite(geo.road_m[:, t])
        nopath[t] = False
        rows.append({
            "target": int(t),
            "sensor_id": geo.sensor_ids[t],
            "lat": round(float(geo.latlon[t, 0]), 5),
            "lon": round(float(geo.latlon[t, 1]), 5),
            "in_degree": int(geo.adj[:, t].sum()),
            "self_mass_share": round(float(self_m / (self_m + tot)), 4),
            "mass_on_adjacent": round(float(adj_m / tot), 4),
            "mass_on_far_road": round(float(v[far].sum() / tot), 4),
            "mass_on_no_road_path": round(float(v[nopath].sum() / tot), 4),
            "mass_off_adjacency": round(float((tot - adj_m) / tot), 4),
        })
    return pd.DataFrame(rows).sort_values("mass_off_adjacency", ascending=False)


def raw_pathologies(dataset: str = "metr_la") -> pd.DataFrame:
    """Per-sensor defects visible in the RAW speed matrix, before any windowing.

    Reported as evidence, never as a verdict about a sensor's labelling.
    """
    path = os.path.join(raw_dir(dataset), "METR-LA.csv")
    df = pd.read_csv(path, index_col=0)
    rows = []
    for pos, col in enumerate(df.columns):
        v = df[col].to_numpy(dtype=float)
        zero = v == 0.0
        rows.append({
            "target": pos,
            "sensor_id": int(col),
            "missing_pct": round(100.0 * float(zero.mean()), 3),
            "longest_missing_run_steps": int(_longest_run(zero)),
            "longest_constant_nonzero_run_steps": int(_longest_constant(v[~zero])),
            "max_mph": round(float(v.max()), 1),
            "pct_above_80mph": round(100.0 * float((v > 80).mean()), 4),
            "mean_nonzero_mph": round(float(v[~zero].mean()), 2) if (~zero).any() else None,
        })
    return pd.DataFrame(rows)


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _longest_constant(v: np.ndarray) -> int:
    if not len(v):
        return 0
    best = cur = 1
    for i in range(1, len(v)):
        cur = cur + 1 if v[i] == v[i - 1] else 1
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt


CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "grey": "#999999"}
HOP_COLOR = {"1": CB["green"], "2": CB["sky"], "3": CB["blue"],
             "4": CB["purple"], "5": CB["purple"], "6": CB["orange"],
             "7": CB["orange"], "8": CB["orange"],
             ">8": CB["vermillion"], "unreachable": "#000000"}


def fig_influence_map(edge_sets: Dict[str, Set[Tuple[int, int]]], geo: ig.Geometry,
                      out: str) -> None:
    """Sensor scatter, adjacency edges grey, off-adjacency effective edges
    coloured by hop bucket. One panel per key of `edge_sets` (regimes here; the
    learned-graph analysis passes its own panel names)."""
    plt = _style()
    regs = list(edge_sets.keys())
    fig, axes = plt.subplots(1, max(len(regs), 1), figsize=(3.5 * max(len(regs), 1), 3.4),
                             squeeze=False)
    lat, lon = geo.latlon[:, 0], geo.latlon[:, 1]
    for ax, reg in zip(axes[0], regs):
        ii, jj = np.where(geo.adj)
        for s, t in zip(ii.tolist(), jj.tolist()):
            ax.plot([lon[s], lon[t]], [lat[s], lat[t]], color=CB["grey"],
                    lw=0.25, alpha=0.35, zorder=1)
        off = {(s, t) for (s, t) in edge_sets[reg] if not geo.adj[s, t]}
        for (s, t) in off:
            b = ig.hop_bucket(int(geo.hops[s, t]), K_HOPS)
            ax.plot([lon[s], lon[t]], [lat[s], lat[t]],
                    color=HOP_COLOR.get(b, CB["vermillion"]), lw=0.5, alpha=0.75, zorder=2)
        ax.scatter(lon, lat, s=3, color="#333333", zorder=3, linewidths=0)
        ax.set_title("{}  ({} off-adjacency of {})".format(
            reg.replace("_", " "), len(off), len(edge_sets[reg])))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect("equal", adjustable="datalim")
    handles = [plt.Line2D([], [], color=CB["grey"], lw=1, label="adjacency edge")]
    for b in ("1", "2", "3", ">8", "unreachable"):
        handles.append(plt.Line2D([], [], color=HOP_COLOR[b], lw=1,
                                  label="off-adj, hop {}".format(b)))
    axes[0][0].legend(handles=handles, loc="best", framealpha=0.9)
    fig.savefig(out)
    plt.close(fig)


def fig_hop_strata(per_regime: Dict[str, dict], out: str) -> None:
    """Hop-bucket composition of the effective edge set, per regime, with the
    adjacency's own hop composition as the reference bar."""
    plt = _style()
    regs = [r for r in ig.REGIMES if r in per_regime and "hop_strata" in per_regime[r]]
    fig, axes = plt.subplots(1, max(len(regs), 1), figsize=(3.6 * max(len(regs), 1), 2.6),
                             squeeze=False)
    for ax, reg in zip(axes[0], regs):
        rows = per_regime[reg]["hop_strata"]
        labs = [r["bucket"] for r in rows]
        share = [r["share_of_effective"] or 0 for r in rows]
        x = np.arange(len(labs))
        ax.bar(x, share, color=[HOP_COLOR.get(b, CB["grey"]) for b in labs])
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=45, ha="right")
        ax.set_ylabel("share of effective edges")
        st = per_regime[reg].get("stability", {})
        ax.set_title("{}\nsplit-half J={}  (beyond-K J={})".format(
            reg.replace("_", " "), st.get("jaccard_all_edges"),
            st.get("jaccard_beyond_k")))
    fig.savefig(out)
    plt.close(fig)


def fig_mask_flatness(per_regime: Dict[str, dict], n_nodes: int, out: str) -> None:
    """Distribution of sources-needed-for-MASS_FRAC, per regime. The headline
    property of the explainer's output, shown as a distribution because a mean
    would hide whether flatness is uniform or driven by a subset of targets."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    colors = {ig.REGIME_CONGESTED: CB["vermillion"], ig.REGIME_FREEFLOW: CB["blue"]}
    any_data = False
    for reg in ig.REGIMES:
        sf = (per_regime.get(reg) or {}).get("sources_for_mass_frac")
        if not sf:
            continue
        vals = np.asarray(list(sf["per_target"].values()))
        ax.hist(vals, bins=30, alpha=0.6, color=colors.get(reg, CB["grey"]),
                label="{} (median {})".format(reg.replace("_", " "),
                                              sf["percentiles"]["50"]))
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.axvline(TOP_K, color="#000000", lw=1.0, ls="--",
               label="top-k = {} (what the JSON keeps)".format(TOP_K))
    ax.set_xlim(0, n_nodes)
    ax.set_xlabel("sources needed to cover {:.0%} of off-self mass".format(MASS_FRAC))
    ax.set_ylabel("targets")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(out)
    plt.close(fig)


def fig_sensor_rank(md: pd.DataFrame, out: str) -> None:
    """Step 3: every sensor's off-adjacency mass share, ranked, with the
    no-road-path component separated out."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    d = md.sort_values("mass_off_adjacency", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    ax.bar(x, d["mass_on_far_road"], color=CB["orange"], label="far by road (>cutoff)")
    ax.bar(x, d["mass_on_no_road_path"], bottom=d["mass_on_far_road"],
           color=CB["vermillion"], label="no road path in the distance file")
    ax.plot(x, d["mass_on_adjacent"], color=CB["blue"], lw=0.8, label="on adjacency")
    ax.set_xlabel("sensors, ranked by off-adjacency importance mass")
    ax.set_ylabel("share of off-self mass")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", framealpha=0.9)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(dest: str, res: Dict[str, object], geo: ig.Geometry,
                 md: Dict[str, pd.DataFrame], top_rows: pd.DataFrame) -> None:
    from .report_influence_graph import render
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(render(res, geo, md, top_rows))


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="Stage-1 run id (or absolute path)")
    ap.add_argument("--quick", action="store_true",
                    help="pooled split-half only; no figures, no report, no raw scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sensitivity-prefix", type=int, default=64,
                    help="also compute the survivor analysis on this many targets "
                         "from the seeded shuffle order, as a sample-size check")
    ap.add_argument("--limit-targets", type=int, default=None,
                    help="restrict to the first N targets of the seeded shuffle "
                         "order (sample-size sensitivity; does not re-solve)")
    args = ap.parse_args(argv)

    path = args.run if os.path.isabs(args.run) else os.path.join(
        PKG_ROOT, run_dir.RAW, args.run)
    data = load_solves(path)
    if args.limit_targets:
        data = limit_to_prefix(data, args.limit_targets)
        print("restricted to first {} targets of the seeded shuffle order"
              .format(data["prefix_n"]))
    node_imp, tgt_speed = data["node_imp"], data["target_speed"]
    regs = regimes_for(tgt_speed)
    geo = ig.load_geometry("metr_la")
    strata = np.asarray(data["strata"])

    print("targets loaded : {}".format(len(node_imp)))
    print("windows        : {}".format(len(strata)))

    if args.quick:
        for regime in ig.REGIMES:
            st = split_half(node_imp, regs, regime, geo, strata, args.seed)
            W, n_used = ig.build_W(node_imp, regs, regime, geo.n, MIN_WINDOWS_FULL)
            tg = ig.active_targets(n_used)
            m = (ig.compare_to_adjacency(ig.edges_top_k(W, tg, TOP_K), geo, tg)
                 if len(tg) else {})
            print("\n--- {} (n_targets_active={}) ---".format(regime, len(tg)))
            print("  precision vs adjacency  : {}".format(m.get("precision")))
            print("  off-adjacency edges     : {}".format(m.get("n_off_adjacency")))
            print("  split-half J (all)      : {}".format(st.get("jaccard_all_edges")))
            print("  split-half J (off-adj)  : {}".format(st.get("jaccard_off_adjacency")))
            print("  split-half J (beyond-K) : {}".format(st.get("jaccard_beyond_k")))
            print("  two-random-draws J ref  : {}".format(
                st.get("jaccard_two_random_draws_reference")))
        return 0

    # ------------------------------------------------------------------ full
    out_dir = os.path.join(PKG_ROOT, run_dir.PROCESSED, os.path.basename(path))
    os.makedirs(out_dir, exist_ok=True)

    res: Dict[str, object] = {
        "stage1_run_id": os.path.basename(path),
        "stage1_manifest": data["manifest"].get("provenance", {}),
        "stage1_config": data["manifest"].get("config", {}),
        "n_targets_solved": len(node_imp),
        "n_windows": int(len(strata)),
        "settings": {"top_k": TOP_K, "mass_frac": MASS_FRAC,
                     "min_windows_full": MIN_WINDOWS_FULL,
                     "min_windows_half": MIN_WINDOWS_HALF, "k_hops": K_HOPS},
        "geometry": {
            "n_nodes": geo.n,
            "n_adjacency_edges": int(geo.adj.sum()),
            "adjacency_density": round(float(geo.adj.sum() / geo.off.sum()), 5),
            "sigma_m": round(geo.sigma_m, 1),
            "kappa": ig.KAPPA,
            "cutoff_m": round(geo.cutoff_m, 1),
            "ordered_pairs_with_road_distance": int(
                (np.isfinite(geo.road_m) & geo.off).sum()),
            "share_pairs_with_road_distance": round(float(
                (np.isfinite(geo.road_m) & geo.off).sum() / geo.off.sum()), 4),
            "share_nonadjacent_pairs_with_road_distance": round(float(
                (np.isfinite(geo.road_m) & geo.off & ~geo.adj).sum()
                / (geo.off & ~geo.adj).sum()), 4),
        },
        "per_regime": {},
    }
    cfg_ckpt = (data["manifest"].get("config") or {}).get(
        "checkpoint", "models/gnn/checkpoints/metr_la_best.pt")
    # Measured density of the support the model ACTUALLY diffuses over. Fact (3)
    # in the report depends on this, so it is computed here rather than asserted.
    try:
        import torch

        from .analyze_learned_graph import density_report
        _ck = torch.load(os.path.join(PKG_ROOT, cfg_ckpt), map_location="cpu",
                         weights_only=False)
        res["support_density"] = density_report(_ck["model_state"], geo)
    except Exception as exc:                           # noqa: BLE001
        res["support_density"] = {"error": "{}: {}".format(type(exc).__name__, exc)}

    all_edge_rows: List[Dict[str, object]] = []
    edge_sets: Dict[str, Set[Tuple[int, int]]] = {}
    mass_tables: Dict[str, pd.DataFrame] = {}

    for regime in ig.REGIMES:
        r = analyse_regime(regime, node_imp, regs, geo, strata, args.seed)
        res["per_regime"][regime] = r
        W, n_used = ig.build_W(node_imp, regs, regime, geo.n, MIN_WINDOWS_FULL)
        tg = ig.active_targets(n_used)
        if not len(tg):
            continue
        e = ig.edges_top_k(W, tg, TOP_K)
        edge_sets[regime] = e
        all_edge_rows.extend(ig.edge_rows(e, W, geo, regime, K_HOPS))
        mass_tables[regime] = mass_decomposition(W, tg, geo)
        try:
            r["vs_learned_semantic_graph"] = compare_to_learned(
                W, tg, geo, cfg_ckpt)
            sv = beyond_k_survivors(node_imp, regs, regime, geo, strata,
                                    cfg_ckpt, args.seed)
            sv["regime_level_claims_permitted"] = bool(
                sv["n_targets_in_both_halves"] >= MIN_TARGETS_FOR_REGIME_CLAIM)
            sv["min_targets_for_regime_claim"] = MIN_TARGETS_FOR_REGIME_CLAIM
            r["beyond_k_survivors"] = sv
        except Exception as exc:                       # noqa: BLE001
            r["vs_learned_semantic_graph"] = {"error": "{}: {}".format(
                type(exc).__name__, exc)}
        print("[{}] active targets {}  edges {}  off-adjacency {}  split-half J {}"
              .format(regime, len(tg), len(e),
                      r["edges"]["top_k"]["n_off_adjacency"],
                      r["stability"].get("jaccard_all_edges")))

    # ------------------------------------- sample-size sensitivity on survivors
    # Does the survivor conclusion hold at a smaller n? The prefix is the run's
    # SEEDED SHUFFLE ORDER, so it is an unbiased subsample of the graph and the
    # comparison isolates sample size rather than geography. Reported whatever it
    # shows: a conclusion that only appears at full n is a weaker conclusion.
    if args.sensitivity_prefix and not args.limit_targets \
            and args.sensitivity_prefix < len(node_imp):
        try:
            sub = limit_to_prefix(data, args.sensitivity_prefix)
            sub_rg = regimes_for(sub["target_speed"])
            sens = {"prefix_n_targets": sub["prefix_n"], "full_n_targets": len(node_imp),
                    "per_regime": {}}
            for regime in ig.REGIMES:
                sv = beyond_k_survivors(sub["node_imp"], sub_rg, regime, geo,
                                        strata, cfg_ckpt, args.seed)
                sv["regime_level_claims_permitted"] = bool(
                    sv["n_targets_in_both_halves"] >= MIN_TARGETS_FOR_REGIME_CLAIM)
                sv.pop("intersection_edges", None)     # full list lives at full n
                sens["per_regime"][regime] = sv
            res["survivor_sample_size_sensitivity"] = sens
            for rg, v in sens["per_regime"].items():
                i = (v.get("intersection") or {}).get("learned_rank_of_206") or {}
                print("[sensitivity n={}] {}: survivors {} top-8 {}".format(
                    sub["prefix_n"], rg, v["n_intersection"], i.get("share_in_top_8")))
        except Exception as exc:                       # noqa: BLE001
            res["survivor_sample_size_sensitivity"] = {
                "error": "{}: {}".format(type(exc).__name__, exc)}

    # ------------------------------------------------------------------ tables
    edf = pd.DataFrame(all_edge_rows)
    edf.to_csv(os.path.join(out_dir, "effective_edges.csv"), index=False)
    if len(edf):
        edf[~edf.in_adjacency].to_csv(
            os.path.join(out_dir, "off_adjacency_edges.csv"), index=False)
        edf[edf.hop_bucket.isin([">{}".format(K_HOPS), "unreachable"])].to_csv(
            os.path.join(out_dir, "beyond_k_edges.csv"), index=False)

    for rg, r in res["per_regime"].items():
        sv = r.get("beyond_k_survivors") or {}
        if sv.get("intersection_edges"):
            pd.DataFrame(sv["intersection_edges"]).to_csv(
                os.path.join(out_dir, "beyond_k_survivors_{}.csv".format(rg)),
                index=False)

    hop_rows = [dict(regime=rg, **h)
                for rg, r in res["per_regime"].items()
                for h in r.get("hop_strata", [])]
    pd.DataFrame(hop_rows).to_csv(os.path.join(out_dir, "hop_strata.csv"), index=False)

    base_rows = []
    for rg, r in res["per_regime"].items():
        if "baselines" not in r:
            continue
        base_rows.append(dict(regime=rg, method="effective_top_k", **r["edges"]["top_k"]))
        base_rows.append(dict(regime=rg, method="effective_cumulative_mass",
                              **{k: v for k, v in r["edges"]["cumulative_mass"].items()
                                 if k != "sources_per_target_mean"}))
        for name in ("uniform_random", "degree_matched_random",
                     "nearest_by_road_distance"):
            base_rows.append(dict(regime=rg, method=name,
                                  **{k: v for k, v in r["baselines"][name].items()
                                     if not isinstance(v, str)}))
    pd.DataFrame(base_rows).to_csv(os.path.join(out_dir, "baselines.csv"), index=False)

    md_all = []
    for rg, t in mass_tables.items():
        t = t.copy()
        t.insert(0, "regime", rg)
        md_all.append(t)
    md_df = pd.concat(md_all) if md_all else pd.DataFrame()
    md_df.to_csv(os.path.join(out_dir, "per_target_mass.csv"), index=False)

    # ---------------------------------------------------------------- step 3
    path_df = raw_pathologies("metr_la")
    path_df.to_csv(os.path.join(out_dir, "sensor_raw_pathologies.csv"), index=False)
    if len(md_df):
        step3 = (md_df.groupby(["target", "sensor_id", "lat", "lon", "in_degree"],
                               as_index=False)
                 .agg(mass_off_adjacency=("mass_off_adjacency", "mean"),
                      mass_on_no_road_path=("mass_on_no_road_path", "mean"),
                      mass_on_far_road=("mass_on_far_road", "mean"),
                      self_mass_share=("self_mass_share", "mean"))
                 .merge(path_df.drop(columns=["target"]), on="sensor_id", how="left")
                 .sort_values("mass_off_adjacency", ascending=False))
        step3.to_csv(os.path.join(out_dir, "sensor_anomaly_ranking.csv"), index=False)
    else:
        step3 = pd.DataFrame()

    # ---------------------------------------------------------------- figures
    if edge_sets:
        fig_influence_map(edge_sets, geo, os.path.join(out_dir, "fig1_influence_map.pdf"))
        fig_hop_strata(res["per_regime"], os.path.join(out_dir, "fig2_hop_strata.pdf"))
        fig_mask_flatness(res["per_regime"], geo.n,
                          os.path.join(out_dir, "fig4_mask_flatness.pdf"))
    if len(md_df):
        fig_sensor_rank(md_df.groupby("target", as_index=False).mean(numeric_only=True),
                        os.path.join(out_dir, "fig3_sensor_mass_rank.pdf"))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)

    write_report(os.path.join(out_dir, "report.md"), res, geo, mass_tables, step3)
    print("\nwrote -> {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
