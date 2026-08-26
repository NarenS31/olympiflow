"""PART 4 analysis — does mask entropy predict explainer stability?

Reads the solves produced by run_sparsity_sweep.py (plus the reused
current-setting solves from the Stage-1 run) and produces, per sparsity setting:

    normalised entropy of node_imp        (per-target distribution)
    sources needed for 80% of the mass
    split-half top-8 Jaccard              (stability under window resampling)
    seed-repeat top-8 Jaccard             (stability under explainer init)
    precision against the kernel adjacency (+ uniform / degree-matched baselines)

then the pooled entropy-vs-stability scatter with a cluster-bootstrapped
Spearman rho, and the pre-registered Youden threshold applied to the 12 committed
explanations.

    python -m xtraffic.scripts.analyze_sparsity_sweep

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..evaluation import influence_graph as ig
from ..evaluation import mask_entropy as me
from ..reproducibility import run_dir
from ..utils.io_utils import PKG_ROOT
from .noise_run import resolve_run
from .run_sparsity_sweep import BASE_RUN, SETTINGS

SPLIT_SEED = 42
N_BOOT = 10000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_setting(run_path: str, label: str, seed: int, targets: List[int],
                 expect_windows: Optional[List[int]] = None
                 ) -> Dict[int, np.ndarray]:
    """{target: [n_windows, N]} for one (setting, seed).

    The current setting at seed 0 is NOT in this run's solve directory — it is
    the Stage-1 run, reused deliberately (run_sparsity_sweep explains why). Every
    other combination is local.

    `expect_windows` is checked, not assumed. A smoke run writes 4-window solves
    under the same filenames as the real 24-window ones, and six of those were
    picked up by this very run before the mismatch was caught. Anything whose
    stored window ids differ from the geometry is DROPPED with a loud line, not
    quietly averaged into the result.
    """
    out: Dict[int, np.ndarray] = {}
    if label == "current" and seed == 0:
        base = os.path.join(PKG_ROOT, run_dir.RAW, BASE_RUN, "solves")
        pattern = os.path.join(base, "target_{}.npz")
    else:
        pattern = os.path.join(run_path, "part4_solves",
                               "{}__seed{}__t{{}}.npz".format(label, seed))

    for t in targets:
        p = pattern.format(t)
        if not os.path.exists(p):
            continue
        with np.load(p) as d:
            imp = d["node_imp"]
            stored = [int(w) for w in d["window_idx"]]
        if expect_windows is not None and stored != [int(w) for w in expect_windows]:
            print("[part4] DROPPED {} — window ids do not match the geometry "
                  "({} rows, expected {}). Stale or smoke artifact."
                  .format(os.path.basename(p), len(stored), len(expect_windows)))
            continue
        out[t] = imp
    return out


def load_committed(run_path: str) -> List[Dict[str, Any]]:
    """The re-solved committed explanations, both checkpoints."""
    sol = os.path.join(run_path, "part4_solves")
    rows: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(sol, "committed_*.npz"))):
        name = os.path.basename(p)[:-len(".npz")]
        m = re.match(r"committed_(ep\d+)__idx(\d+)_node(\d+)$", name)
        if not m:
            continue
        d = np.load(p)
        rows.append({"checkpoint": m.group(1), "sample_index": int(m.group(2)),
                     "target": int(m.group(3)), "node_imp": d["node_imp"]})
    return rows


# ---------------------------------------------------------------------------
# Per-setting analysis
# ---------------------------------------------------------------------------
def analyse_setting(label: str, lam: float, blocks: Dict[int, np.ndarray],
                    strata: np.ndarray, geo: ig.Geometry
                    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    half_a, half_b = ig.split_halves(strata, seed=SPLIT_SEED)
    per_target: List[Dict[str, Any]] = []
    for t in sorted(blocks):
        row = me.summarise_target(blocks[t], t, half_a, half_b)
        row["setting"] = label
        row["lambda_size"] = lam
        per_target.append(row)

    targets = [r["target"] for r in per_target]
    # Effective edge set from the 24-window mean mask, k=8 — the committed rule.
    edges = set()
    for r in per_target:
        edges.update((int(s), int(r["target"])) for s in r["top_k"])
    adj = ig.compare_to_adjacency(edges, geo, targets)
    uni = ig.compare_to_adjacency(
        ig.baseline_uniform(geo, targets, k=me.TOP_K, seed=SPLIT_SEED), geo, targets)
    deg = ig.compare_to_adjacency(
        ig.baseline_degree_matched(geo, targets, k=me.TOP_K, seed=SPLIT_SEED),
        geo, targets)

    def _m(key: str) -> float:
        vals = [r[key] for r in per_target if r.get(key) is not None
                and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    def _s(key: str) -> float:
        vals = [r[key] for r in per_target if r.get(key) is not None
                and r[key] == r[key]]
        return float(np.std(vals)) if vals else float("nan")

    summary = {
        "setting": label,
        "lambda_size": lam,
        "n_targets": len(per_target),
        "n_windows": int(per_target[0]["n_windows"]) if per_target else 0,
        "entropy_of_mean_mean": _m("entropy_of_mean"),
        "entropy_of_mean_std": _s("entropy_of_mean"),
        "mean_of_entropy_mean": _m("mean_of_entropy"),
        "mean_of_entropy_std": _s("mean_of_entropy"),
        "sources_for_80pct_mean": _m("sources_for_80pct_of_mean"),
        "sources_for_80pct_std": _s("sources_for_80pct_of_mean"),
        "sources_for_80pct_per_window_mean": _m("mean_sources_for_80pct_per_window"),
        "self_importance_mean": _m("self_importance"),
        "split_half_jaccard_mean": _m("split_half_jaccard"),
        "split_half_jaccard_std": _s("split_half_jaccard"),
        "precision_vs_adjacency": adj["precision"],
        "recall_vs_adjacency": adj["recall"],
        "n_effective_edges": adj["n_effective"],
        "precision_uniform_baseline": uni["precision"],
        "precision_degree_matched_baseline": deg["precision"],
        "precision_lift_vs_uniform": (adj["precision"] / uni["precision"]
                                      if uni["precision"] else float("nan")),
    }
    return summary, per_target


def seed_repeat(run_path: str, targets: List[int],
                expect_windows: Optional[List[int]] = None
                ) -> List[Dict[str, Any]]:
    """Top-8 Jaccard between explainer seed 0 and seed 1, per (setting, target)."""
    rows: List[Dict[str, Any]] = []
    for label, lam in SETTINGS:
        b0 = load_setting(run_path, label, 0, targets, expect_windows)
        b1 = load_setting(run_path, label, 1, targets, expect_windows)
        for t in sorted(set(b0) & set(b1)):
            t0 = me.top_k_sources(b0[t].mean(axis=0), t)
            t1 = me.top_k_sources(b1[t].mean(axis=0), t)
            rows.append({"setting": label, "lambda_size": lam, "target": t,
                         "seed_repeat_jaccard": me.jaccard(t0, t1),
                         "entropy_seed0": me.normalised_entropy(b0[t].mean(axis=0), t),
                         "entropy_seed1": me.normalised_entropy(b1[t].mean(axis=0), t)})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    run = resolve_run(args.run_id)
    with open(os.path.join(run.path, "part4_geometry.json")) as fh:
        geom = json.load(fh)
    targets = [int(t) for t in geom["targets"]]
    strata = np.asarray(geom["stratum"])
    geo = ig.load_geometry("metr_la")

    # --- per setting ------------------------------------------------------
    summaries: List[Dict[str, Any]] = []
    per_target_all: List[Dict[str, Any]] = []
    for label, lam in SETTINGS:
        blocks = load_setting(run.path, label, 0, targets,
                              expect_windows=[int(w) for w in geom['window_idx']])
        missing = [t for t in targets if t not in blocks]
        if missing:
            print("[part4] setting {}: {} of {} targets missing (still solving?) "
                  "-> analysing the {} present".format(
                      label, len(missing), len(targets), len(blocks)))
        if not blocks:
            print("[part4] setting {}: NO solves found, skipping".format(label))
            continue
        s, pt = analyse_setting(label, lam, blocks, strata, geo)
        s["n_targets_missing"] = len(missing)
        summaries.append(s)
        per_target_all.extend(pt)

    # --- seed repeat ------------------------------------------------------
    sr = seed_repeat(run.path, [int(t) for t in geom["seed_repeat_targets"]],
                     [int(w) for w in geom["window_idx"]])
    sr_by_setting: Dict[str, List[float]] = {}
    for r in sr:
        sr_by_setting.setdefault(r["setting"], []).append(r["seed_repeat_jaccard"])
    for s in summaries:
        vals = sr_by_setting.get(s["setting"], [])
        s["seed_repeat_jaccard_mean"] = float(np.mean(vals)) if vals else float("nan")
        s["seed_repeat_jaccard_n"] = len(vals)

    # --- pooled scatter + Spearman ----------------------------------------
    pooled = [r for r in per_target_all
              if r.get("split_half_jaccard") is not None
              and np.isfinite(r["split_half_jaccard"])]
    scatter = {}
    for ekey in ("entropy_of_mean", "mean_of_entropy"):
        scatter[ekey] = me.spearman_clustered_ci(
            [r[ekey] for r in pooled],
            [r["split_half_jaccard"] for r in pooled],
            [r["target"] for r in pooled], n_boot=N_BOOT, seed=SPLIT_SEED)

    # --- threshold, chosen on the sweep only ------------------------------
    js = np.asarray([r["split_half_jaccard"] for r in pooled], dtype=float)
    median_j = float(np.median(js))
    unstable = [bool(r["split_half_jaccard"] < median_j) for r in pooled]
    thresholds = {
        ekey: me.youden_threshold([r[ekey] for r in pooled], unstable)
        for ekey in ("entropy_of_mean", "mean_of_entropy")
    }
    for ekey in thresholds:
        thresholds[ekey]["unstable_defined_as"] = (
            "split-half top-8 Jaccard < pooled median ({:.4f})".format(median_j))

    # --- the 12 committed explanations ------------------------------------
    committed = load_committed(run.path)
    # Committed masks are SINGLE-window, so `mean_of_entropy` is the comparable
    # statistic and is what the pre-registered flag uses. entropy_of_mean is
    # identical to it at n_windows=1 but is reported so the two are visibly the
    # same object here.
    T = thresholds["mean_of_entropy"]["threshold"]
    committed_rows: List[Dict[str, Any]] = []
    for c in committed:
        s = me.summarise_target(c["node_imp"], c["target"])
        committed_rows.append({
            "checkpoint": c["checkpoint"], "sample_index": c["sample_index"],
            "target": c["target"],
            "entropy": s["mean_of_entropy"],
            "sources_for_80pct": s["sources_for_80pct_of_mean"],
            "self_importance": s["self_importance"],
            "flagged": (bool(s["mean_of_entropy"] > T) if T is not None else None),
        })
    flagged: Dict[str, Any] = {}
    for tag in sorted(set(r["checkpoint"] for r in committed_rows)):
        rows = [r for r in committed_rows if r["checkpoint"] == tag]
        n_flag = sum(1 for r in rows if r["flagged"])
        flagged[tag] = {
            "n": len(rows), "n_flagged": n_flag,
            "fraction_flagged": n_flag / len(rows) if rows else float("nan"),
            "entropy_min": float(min(r["entropy"] for r in rows)),
            "entropy_max": float(max(r["entropy"] for r in rows)),
            "entropy_mean": float(np.mean([r["entropy"] for r in rows])),
        }

    # --- supplementary, and free: the SAME relationship at the current setting
    # --- on all 207 targets, not just the 40 in the sweep.
    #
    # NOT the pre-registered test — that one is the pooled scatter across the
    # three sparsity settings, and it stays exactly as written above. This is the
    # same question asked of the Stage-1 run's full target set, which is already
    # on disk and costs nothing to read. It is reported separately, labelled
    # supplementary, and is the better-powered estimate of the entropy-stability
    # relationship at the committed sparsity setting.
    supplementary: Dict[str, Any] = {}
    try:
        base_w = os.path.join(PKG_ROOT, run_dir.RAW, BASE_RUN, "windows.json")
        with open(base_w) as fh:
            bw = json.load(fh)
        all_targets = [int(t) for t in bw["target_order"]]
        big = load_setting(run.path, "current", 0, all_targets,
                           expect_windows=[int(w) for w in bw["window_idx"]])
        big_strata = np.asarray(bw["stratum"])
        ha, hb = ig.split_halves(big_strata, seed=SPLIT_SEED)
        rows_big = [me.summarise_target(big[t], t, ha, hb) for t in sorted(big)]
        usable = [r for r in rows_big
                  if r["split_half_jaccard"] is not None
                  and np.isfinite(r["split_half_jaccard"])]
        supplementary = {
            "what": ("current setting only, all targets solved by the Stage-1 "
                     "run; supplementary to the pre-registered pooled scatter"),
            "n_targets": len(usable),
            "entropy_of_mean_mean": float(np.mean([r["entropy_of_mean"] for r in usable])),
            "mean_of_entropy_mean": float(np.mean([r["mean_of_entropy"] for r in usable])),
            "sources_for_80pct_mean": float(np.mean(
                [r["sources_for_80pct_of_mean"] for r in usable])),
            "split_half_jaccard_mean": float(np.mean(
                [r["split_half_jaccard"] for r in usable])),
            "spearman": {
                ekey: me.spearman_clustered_ci(
                    [r[ekey] for r in usable],
                    [r["split_half_jaccard"] for r in usable],
                    [r["target"] for r in usable], n_boot=N_BOOT, seed=SPLIT_SEED)
                for ekey in ("entropy_of_mean", "mean_of_entropy")},
        }
    except Exception as exc:                              # noqa: BLE001
        supplementary = {"error": "{}: {}".format(type(exc).__name__, exc)}

    metrics = {
        "part": 4,
        "settings": summaries,
        "supplementary_current_setting_all_targets": supplementary,
        "split_half_seed": SPLIT_SEED,
        "seed_repeat": sr,
        "scatter_spearman": scatter,
        "entropy_threshold": thresholds,
        "committed_explanations": {
            "note": ("12 committed explanation FILES cover 11 distinct "
                     "(target, window) pairs — high_congestion.json and "
                     "rush_hour_pm.json are the same target and window."),
            "per_explanation": committed_rows,
            "by_checkpoint": flagged,
            "checkpoint_ep34_is_the_committed_artifact": True,
        },
        "n_pooled_points": len(pooled),
    }
    with open(os.path.join(run.path, "part4_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    with open(os.path.join(run.path, "part4_per_target.csv"), "w", newline="") as fh:
        cols = ["setting", "lambda_size", "target", "n_windows", "entropy_of_mean",
                "mean_of_entropy", "std_of_entropy", "sources_for_80pct_of_mean",
                "mean_sources_for_80pct_per_window", "self_importance",
                "split_half_jaccard"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_target_all)

    # --- console report ---------------------------------------------------
    print("\n=== PART 4 — SPARSITY SWEEP ===")
    print("{:9s}{:>8s}{:>7s}{:>11s}{:>11s}{:>10s}{:>11s}{:>10s}{:>11s}".format(
        "setting", "lambda", "n", "H(mean)", "meanH", "src80", "splitJ",
        "seedJ", "precision"))
    print("-" * 88)
    for s in summaries:
        print("{:9s}{:>8.2f}{:>7d}{:>11.4f}{:>11.4f}{:>10.1f}{:>11.3f}{:>10.3f}{:>11.3f}"
              .format(s["setting"], s["lambda_size"], s["n_targets"],
                      s["entropy_of_mean_mean"], s["mean_of_entropy_mean"],
                      s["sources_for_80pct_mean"], s["split_half_jaccard_mean"],
                      s["seed_repeat_jaccard_mean"], s["precision_vs_adjacency"]))
    print("\nSCATTER (pooled, n={} points over {} targets):".format(
        len(pooled), scatter["entropy_of_mean"]["n_clusters"]))
    for ekey, sc in scatter.items():
        print("  {:16s} rho {:+.3f}  95% CI [{:+.3f}, {:+.3f}]  {}".format(
            ekey, sc["rho"], sc["ci_lo"], sc["ci_hi"],
            "excludes 0" if sc["excludes_zero"] else "spans 0"))
    print("\nENTROPY THRESHOLD (Youden J on the sweep pool only):")
    for ekey, t in thresholds.items():
        if t["threshold"] is None:
            print("  {:16s} not identifiable ({})".format(ekey, t.get("note")))
        else:
            print("  {:16s} T = {:.4f}  J = {:.3f}  (sens {:.2f} / spec {:.2f})"
                  .format(ekey, t["threshold"], t["youden_j"],
                          t["sensitivity"], t["specificity"]))
    print("\nCOMMITTED EXPLANATIONS FLAGGED (T = {}):".format(
        "{:.4f}".format(T) if T is not None else "n/a"))
    for tag, f in flagged.items():
        print("  {:6s} {:2d}/{:2d} flagged  entropy {:.4f}-{:.4f} (mean {:.4f})"
              .format(tag, f["n_flagged"], f["n"], f["entropy_min"],
                      f["entropy_max"], f["entropy_mean"]))
    if supplementary and "error" not in supplementary:
        print("\nSUPPLEMENTARY (current setting, all {} Stage-1 targets — free, "
              "not the pre-registered test):".format(supplementary["n_targets"]))
        print("  entropy {:.4f} | src80 {:.1f} | split-half J {:.3f}".format(
            supplementary["entropy_of_mean_mean"],
            supplementary["sources_for_80pct_mean"],
            supplementary["split_half_jaccard_mean"]))
        for ekey, sc in supplementary["spearman"].items():
            print("  {:16s} rho {:+.3f}  95% CI [{:+.3f}, {:+.3f}]  {}".format(
                ekey, sc["rho"], sc["ci_lo"], sc["ci_hi"],
                "excludes 0" if sc["excludes_zero"] else "spans 0"))
    elif supplementary:
        print("\nSUPPLEMENTARY skipped: {}".format(supplementary["error"]))

    print("\nwrote {}".format(os.path.join(run.path, "part4_metrics.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
