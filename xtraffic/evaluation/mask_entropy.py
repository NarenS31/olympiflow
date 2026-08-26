"""PART 4 — mask entropy, and whether it predicts explainer stability.

THE QUESTION
    A GNNExplainer node mask that is nearly flat has no well-determined top-k:
    the ordering is decided by float-level differences between near-equal
    entries. If that is right, then how flat a mask is should predict how much
    its top-8 survives resampling the windows it was built from. Entropy is the
    natural measure of flatness, and it is computable from a SINGLE solve —
    whereas measuring stability directly costs a second solve. A cheap proxy for
    an expensive property is worth having, if it works.

TWO ENTROPIES, AND WHY THE DISTINCTION MATTERS
    A target's mask is a [n_windows, N] array. There are two different things
    "the entropy of this target's mask" could mean, and they are not
    interchangeable:

      entropy_of_mean : entropy of the mask AVERAGED over the target's windows.
                        This is the object the top-8 is read off, so it is the
                        right partner for a top-8 stability measure.
      mean_of_entropy : the mean of the per-window entropies. Averaging masks
                        pulls the average toward uniform, so entropy_of_mean is
                        systematically HIGHER than mean_of_entropy.

    The 12 committed explanations are SINGLE-WINDOW solves. A threshold learned
    on 24-window averages and applied to single-window masks would be comparing
    two different quantities and would flag things for the wrong reason. So
    `mean_of_entropy` is the one carried to the committed explanations, and both
    are reported everywhere.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

TOP_K = 8
MASS_FRACTION = 0.80


# ---------------------------------------------------------------------------
# Per-vector primitives
# ---------------------------------------------------------------------------
def normalised_entropy(v: np.ndarray, target: int) -> float:
    """Shannon entropy of one mask, self excluded, normalised to [0, 1].

    The mask is a vector of independent sigmoid values, not a distribution, so we
    normalise it to sum 1 first — the same normalisation `sources_for_mass` uses,
    which keeps the two statistics talking about the same object. 1.0 means
    perfectly flat (every source equally important, i.e. no explanation at all);
    0.0 means all the mass sits on one source.
    """
    w = np.delete(np.asarray(v, dtype=np.float64), target)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0 or len(w) < 2:
        return float("nan")
    p = w / total
    nz = p[p > 0]
    h = float(-(nz * np.log(nz)).sum())
    return h / float(np.log(len(w)))


def sources_for_mass(v: np.ndarray, target: int,
                     frac: float = MASS_FRACTION) -> int:
    """How many sources it takes to cover `frac` of the off-self importance mass."""
    w = np.delete(np.asarray(v, dtype=np.float64), target)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return 0
    acc, n = 0.0, 0
    for x in np.sort(w)[::-1]:
        if acc >= frac * total:
            break
        acc += x
        n += 1
    return n


def top_k_sources(v: np.ndarray, target: int, k: int = TOP_K) -> Set[int]:
    """The k most important sources, target excluded — the committed top_k rule."""
    w = np.asarray(v, dtype=np.float64).copy()
    w[target] = -np.inf
    return set(int(i) for i in np.argsort(-w)[:k])


def jaccard(a: Set[int], b: Set[int]) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


# ---------------------------------------------------------------------------
# Per-target summaries over a [n_windows, N] solve block
# ---------------------------------------------------------------------------
def summarise_target(node_imp: np.ndarray, target: int,
                     half_a: Optional[np.ndarray] = None,
                     half_b: Optional[np.ndarray] = None,
                     k: int = TOP_K) -> Dict[str, Any]:
    """Every per-target statistic Part 4 needs, from one target's solve block.

    node_imp : [n_windows, N]
    half_a/b : window POSITIONS for the split-half check (from
               influence_graph.split_halves). None => no split-half reported.
    """
    node_imp = np.asarray(node_imp, dtype=np.float64)
    mean_mask = node_imp.mean(axis=0)

    per_window_H = [normalised_entropy(node_imp[i], target)
                    for i in range(node_imp.shape[0])]
    per_window_mass = [sources_for_mass(node_imp[i], target)
                       for i in range(node_imp.shape[0])]

    out: Dict[str, Any] = {
        "target": int(target),
        "n_windows": int(node_imp.shape[0]),
        "entropy_of_mean": normalised_entropy(mean_mask, target),
        "mean_of_entropy": float(np.nanmean(per_window_H)),
        "std_of_entropy": float(np.nanstd(per_window_H)),
        "sources_for_80pct_of_mean": sources_for_mass(mean_mask, target),
        "mean_sources_for_80pct_per_window": float(np.mean(per_window_mass)),
        "self_importance": float(node_imp[:, target].mean()),
        "top_k": sorted(top_k_sources(mean_mask, target, k)),
        "split_half_jaccard": None,
    }
    if half_a is not None and half_b is not None and len(half_a) and len(half_b):
        ta = top_k_sources(node_imp[half_a].mean(axis=0), target, k)
        tb = top_k_sources(node_imp[half_b].mean(axis=0), target, k)
        out["split_half_jaccard"] = jaccard(ta, tb)
        out["split_half_top_k_a"] = sorted(ta)
        out["split_half_top_k_b"] = sorted(tb)
    return out


# ---------------------------------------------------------------------------
# Threshold selection — on the sweep, never on the thing being flagged
# ---------------------------------------------------------------------------
def youden_threshold(scores: Sequence[float], positive: Sequence[bool]
                     ) -> Dict[str, Any]:
    """Threshold on `scores` maximising Youden's J for predicting `positive`.

    A point is flagged when score > T. Candidate thresholds are the observed
    scores themselves, so the choice is data-driven but has no free parameters
    to tune after the fact. Returns the threshold, its J, and the sensitivity /
    specificity that produced it.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return {"threshold": None, "youden_j": None, "n_positive": n_pos,
                "n_negative": n_neg,
                "note": "one class empty; no threshold is identifiable"}

    best = {"threshold": None, "youden_j": -np.inf}
    for t in np.unique(s):
        pred = s > t
        sens = float((pred & y).sum()) / n_pos
        spec = float((~pred & ~y).sum()) / n_neg
        j = sens + spec - 1.0
        if j > best["youden_j"]:
            best = {"threshold": float(t), "youden_j": float(j),
                    "sensitivity": sens, "specificity": spec}
    best.update({"n_positive": n_pos, "n_negative": n_neg,
                 "rule": "flag when score > threshold"})
    return best


# ---------------------------------------------------------------------------
# Spearman with a cluster bootstrap
# ---------------------------------------------------------------------------
def spearman_clustered_ci(x: Sequence[float], y: Sequence[float],
                          cluster: Sequence[Any], n_boot: int = 10000,
                          seed: int = 42, alpha: float = 0.05
                          ) -> Dict[str, Any]:
    """Spearman rho between x and y, with a CI that respects clustering.

    Each target contributes one point PER SPARSITY SETTING, and those points are
    not independent — they are the same road sensor solved three ways. Resampling
    individual points would treat three views of one target as three targets and
    report a CI that is too narrow. So the bootstrap resamples TARGETS with
    replacement and takes all of a target's rows with it.
    """
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cluster = np.asarray(cluster)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cluster = x[ok], y[ok], cluster[ok]

    rho, p = stats.spearmanr(x, y)
    groups = {c: np.where(cluster == c)[0] for c in np.unique(cluster)}
    keys = list(groups)
    rng = np.random.default_rng(seed)

    boots: List[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(float(r))
    boots_arr = np.asarray(boots)
    lo = float(np.percentile(boots_arr, 100 * alpha / 2)) if len(boots_arr) else float("nan")
    hi = float(np.percentile(boots_arr, 100 * (1 - alpha / 2))) if len(boots_arr) else float("nan")
    return {
        "rho": float(rho), "p_value": float(p), "n_points": int(len(x)),
        "n_clusters": len(keys), "n_boot": n_boot, "n_boot_used": len(boots_arr),
        "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
        "bootstrap_unit": "target (all of a target's settings move together)",
    }
