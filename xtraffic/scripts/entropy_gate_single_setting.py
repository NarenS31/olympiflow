"""Regenerate the SINGLE-SETTING (n=40) entropy-vs-stability correlation.

WHY THIS EXISTS
    The Interpretability-for-Discovery paper reports that a single-setting version
    of the entropy gate appeared to pass at n=40 before failing on the pooled
    120-pair set. That n=40 number was computed inline during analysis and then
    overwritten: `part4_metrics.json` now holds the pooled result, so the figure
    had no artifact backing it. A number in a paper whose only source is "it was
    printed once" is not sourced. This script recomputes it from the solves on
    disk and writes it to its own run directory.

WHAT IT COMPUTES
    Exactly the pre-registered test, restricted to the COMMITTED sparsity setting
    (lambda_size 0.15, explainer seed 0) over the 40 sweep targets: per-target
    normalised mask entropy against per-target split-half top-8 Jaccard, Spearman
    with a 10k bootstrap resampling TARGETS, seed 42. Same functions the pooled
    analysis uses (`evaluation.mask_entropy`), so the two are comparable by
    construction and differ only in which rows enter the pool.

    Solves come from the Stage-1 influence run, which solved all 207 targets at
    this setting; the 40 are its seeded-shuffle prefix, i.e. the same targets the
    sweep used.

    python -m xtraffic.scripts.entropy_gate_single_setting

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from ..evaluation import influence_graph as ig
from ..evaluation import mask_entropy as me
from ..reproducibility import provenance, run_dir, seeds
from ..utils.io_utils import PKG_ROOT

REPO_ROOT = os.path.dirname(PKG_ROOT)
EXPERIMENT = "entropy_gate_single_setting"
BASE_RUN = "20260824T235016Z__influence_graph_solves__73e75966__ee9a7944"
SWEEP_RUN = "20260825T201433Z__grounding_without_information__f193e764__a1b0fc36"
N_TARGETS = 40
LAMBDA_SIZE = 0.15          # the committed setting
EXPLAINER_SEED = 0
SPLIT_SEED = 42
N_BOOT = 10000


def main() -> int:
    seeds.set_all_seeds(SPLIT_SEED)
    base = os.path.join(PKG_ROOT, run_dir.RAW, BASE_RUN)
    with open(os.path.join(base, "windows.json")) as fh:
        w = json.load(fh)
    targets = [int(t) for t in w["target_order"][:N_TARGETS]]
    windows = [int(x) for x in w["window_idx"]]
    strata = np.asarray(w["stratum"][:len(windows)])
    half_a, half_b = ig.split_halves(strata, seed=SPLIT_SEED)

    cfg: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "purpose": ("regenerate the n=40 single-setting entropy-vs-stability "
                    "correlation reported as the pre-pooling result; the inline "
                    "computation was overwritten by the pooled analysis"),
        "source_solves_run": BASE_RUN,
        "compared_against_run": SWEEP_RUN,
        "setting": "current", "lambda_size": LAMBDA_SIZE,
        "explainer_seed": EXPLAINER_SEED, "n_targets": N_TARGETS,
        "n_windows": len(windows), "split_seed": SPLIT_SEED,
        "bootstrap_iters": N_BOOT, "bootstrap_unit": "target",
        "top_k": me.TOP_K,
    }
    run = run_dir.RunDir.create(
        REPO_ROOT, EXPERIMENT, cfg,
        artifacts={"solves_windows": os.path.join(base, "windows.json")},
        notes=("No LLM, no new solves. Reads the Stage-1 masks and recomputes the "
               "single-setting correlation with the same functions the pooled "
               "analysis uses."))
    run.write_json("seeds.json", seeds.describe())

    rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    for t in targets:
        p = os.path.join(base, "solves", "target_{}.npz".format(t))
        if not os.path.exists(p):
            missing.append(t)
            continue
        with np.load(p) as d:
            imp = d["node_imp"]
            stored = [int(x) for x in d["window_idx"]]
        if stored != windows:                     # geometry guard (audit 11.2.x)
            missing.append(t)
            continue
        rows.append(me.summarise_target(imp, t, half_a, half_b))

    usable = [r for r in rows
              if r["split_half_jaccard"] is not None
              and np.isfinite(r["split_half_jaccard"])]
    out: Dict[str, Any] = {
        "n_targets_requested": N_TARGETS,
        "n_targets_used": len(usable),
        "n_targets_missing_or_geometry_mismatch": len(missing),
        "setting": "current", "lambda_size": LAMBDA_SIZE,
        "split_half_jaccard_mean": float(np.mean(
            [r["split_half_jaccard"] for r in usable])),
        "spearman": {
            k: me.spearman_clustered_ci(
                [r[k] for r in usable],
                [r["split_half_jaccard"] for r in usable],
                [r["target"] for r in usable], n_boot=N_BOOT, seed=SPLIT_SEED)
            for k in ("entropy_of_mean", "mean_of_entropy")},
        "pre_registered_criteria": {
            "rho_threshold": -0.30,
            "requires_ci_excluding_zero": True,
        },
    }
    for k, sc in out["spearman"].items():
        out["spearman"][k]["passes_pre_registered_criteria"] = bool(
            sc["rho"] < -0.30 and sc["excludes_zero"])

    run.write_json("entropy_gate_single_setting.json", out)
    run.complete(n_targets_used=len(usable),
                 rho_entropy_of_mean=out["spearman"]["entropy_of_mean"]["rho"],
                 rho_mean_of_entropy=out["spearman"]["mean_of_entropy"]["rho"])

    print("run_id: {}".format(run.run_id))
    print("targets used: {} (requested {}, skipped {})".format(
        len(usable), N_TARGETS, len(missing)))
    print("split-half Jaccard mean: {:.4f}".format(out["split_half_jaccard_mean"]))
    for k, sc in out["spearman"].items():
        print("  {:16s} rho {:+.4f}  95% CI [{:+.4f}, {:+.4f}]  {}  -> {}".format(
            k, sc["rho"], sc["ci_lo"], sc["ci_hi"],
            "excludes 0" if sc["excludes_zero"] else "spans 0",
            "PASSES pre-registered criteria"
            if sc["passes_pre_registered_criteria"] else "fails"))
    print("\nwrote {}".format(
        os.path.join(run.path, "entropy_gate_single_setting.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
