"""STAGE 1 — the expensive half: 207 targets x 24 shared windows of GNNExplainer.

Produces the raw material for scripts/analyze_influence_graph.py: one
[n_windows, N] matrix of node-importance masks per target, plus the per-window
speeds needed to label regimes.

    python -m xtraffic.scripts.run_influence_solves            # the real run
    python -m xtraffic.scripts.run_influence_solves --smoke    # 4 targets, 4 windows
    python -m xtraffic.scripts.run_influence_solves --resume <run_id>

COST MODEL (measured on this machine, Apple M4, 10 cores, before the run)
    1 solve, CPU, 1 thread ......... 39.2 s
    1 solve, CPU, 4 threads ........ 27.3 s
    MPS ............................  9.0 s   -- NOT USED, see below
    207 x 24 = 4,968 solves / 8 workers x 1 thread ~= 6.8 h

WHY ONE SOLVE PER (TARGET, WINDOW) AND NOT SIX
    ExplanationBuilder.explain_prediction runs the explainer 1 + confidence_runs
    times (measured 5.05x) to compute explanation_confidence. This analysis never
    reads that field, so we call GNNExplainer.explain_target directly. Nothing
    about the mask optimisation changes; we simply decline to compute a field we
    do not use. Per-solve stability is recovered instead from the split-half
    design in Stage 2, which is a stronger check than a self-consistency scalar.

WHY CPU AND NOT MPS, DESPITE MPS BEING 3x FASTER
    Measured before this run, same seed, same input, same checkpoint: CPU and MPS
    agree on importance MAGNITUDES (Pearson 0.85-0.98) but disagree on WHICH
    nodes are important (Spearman 0.28-0.87, top-8 Jaccard 0.14-0.60). The
    identity of the top influencers is the entire object of this analysis, so a
    3x speedup that reorders it is not a speedup. CPU also matches every
    committed explanation artifact. The divergence is reported as a finding in
    its own right, not filed as a device note.

APPEND-ONLY, INCREMENTAL
    Workers return their arrays to the parent; the PARENT writes. That keeps all
    file writes in one process (no interleaved-append hazard) and means every
    completed target is on disk the moment it finishes, so a preliminary
    split-half can be computed from a partial run without stopping it.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..evaluation import influence_graph as ig
from ..reproducibility import provenance, run_dir, seeds
from ..utils.io_utils import PKG_ROOT

REPO_ROOT = os.path.dirname(PKG_ROOT)
CKPT = "models/gnn/checkpoints/metr_la_best.pt"
DATASET = "metr_la"
HORIZON_STEP = 6          # 30-minute forecast: the horizon every committed
                          # explanation, faithfulness study and figure uses.
EXPERIMENT = "influence_graph_solves"

# Set once in each worker before torch is used. 8 single-threaded workers beat
# 1 eight-threaded worker on this machine (39.2 s x 1/8 vs 27.6 s x 1/1).
_WORKER_THREADS = 1

_W: Dict[str, object] = {}          # per-worker singletons, built once per process


def _init_worker(ckpt: str, dataset: str, device: str, epochs: int) -> None:
    import torch

    from ..models.explainer.explain import ExplanationBuilder
    torch.set_num_threads(_WORKER_THREADS)
    _W["builder"] = ExplanationBuilder(ckpt, dataset, device=torch.device(device),
                                       top_k=8, epochs=epochs, confidence_runs=0)
    _W["X"] = np.load(os.path.join(_W_processed(dataset), "test.npz"))["X"]


def _W_processed(dataset: str) -> str:
    from ..utils.io_utils import processed_dir
    return processed_dir(dataset)


def _solve_target(job: Tuple[int, List[int]]) -> Dict[str, object]:
    """One target x all its windows. Runs in a worker process."""
    import torch

    target, window_idx = job
    b = _W["builder"]
    X = _W["X"]
    t0 = time.time()
    imp = np.zeros((len(window_idx), b.model.num_nodes), dtype=np.float32)
    for i, w in enumerate(window_idx):
        xi = torch.from_numpy(X[w:w + 1]).float()        # [1,12,N,2]
        node_imp, _, _, _ = b.explainer.explain_target(
            xi, target, HORIZON_STEP, seed=0)            # seed 0 = committed default
        imp[i] = node_imp
    return {"target": int(target), "node_imp": imp,
            "seconds": round(time.time() - t0, 2)}


def _build_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "experiment": EXPERIMENT,
        "dataset": DATASET,
        "checkpoint": CKPT,
        "horizon_step": HORIZON_STEP,
        "horizon_minutes": HORIZON_STEP * 5,
        "n_targets": args.n_targets,
        "n_windows": args.n_windows,
        "n_tod_bands": args.n_tod_bands,
        "n_congestion_strata": args.n_congestion_strata,
        "explainer_epochs": args.epochs,
        "explainer_seed": 0,
        "confidence_runs": 0,
        "sampling_seed": args.seed,
        "device": args.device,
        "workers": args.workers,
        "torch_threads_per_worker": _WORKER_THREADS,
        "congested_max_mph": ig.CONGESTED_MAX_MPH,
        "freeflow_min_mph": ig.FREEFLOW_MIN_MPH,
        "missing_max_mph": ig.MISSING_MAX_MPH,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-targets", type=int, default=207)
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--n-tod-bands", type=int, default=6)
    ap.add_argument("--n-congestion-strata", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200,
                    help="explainer optimisation steps; 200 is the committed value")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="4 targets x 4 windows, for wiring checks")
    ap.add_argument("--resume", default=None, metavar="RUN_ID",
                    help="continue an interrupted run, skipping finished targets")
    args = ap.parse_args(argv)

    if args.smoke:
        args.n_targets, args.n_windows = 4, 4
        args.n_tod_bands, args.n_congestion_strata = 2, 1

    seed_rec = seeds.set_all_seeds(args.seed)

    # ---------------------------------------------------------------- geometry
    from ..utils.io_utils import processed_dir
    pdir = processed_dir(DATASET)
    with open(os.path.join(pdir, "scaler.json")) as fh:
        scaler = json.load(fh)
    X = np.load(os.path.join(pdir, "test.npz"))["X"]                 # [S,12,N,2]
    last_mph = X[:, -1, :, 0] * scaler["std"] + scaler["mean"]       # [S,N]
    tod = X[:, -1, 0, 1]                                             # [S]

    window_idx, strata = ig.select_windows(
        last_mph, tod, n_windows=args.n_windows, seed=args.seed,
        n_tod_bands=args.n_tod_bands, n_congestion_strata=args.n_congestion_strata)

    n_nodes = last_mph.shape[1]
    targets = list(range(min(args.n_targets, n_nodes)))
    # Seeded shuffle: a partial run is then an unbiased sample of the graph, so
    # the preliminary split-half is not "the first 60 node indices" (which in
    # METR-LA correlate with position in the sensor file, hence with geography).
    order = np.random.RandomState(args.seed).permutation(len(targets))
    targets = [targets[i] for i in order]

    cfg = _build_config(args)

    # ------------------------------------------------------------------ run dir
    if args.resume:
        path = os.path.join(REPO_ROOT, "xtraffic", run_dir.RAW, args.resume)
        run = run_dir.RunDir.open(path, REPO_ROOT)
        print("[resume] {}".format(run.run_id))
    else:
        run = run_dir.RunDir.create(
            REPO_ROOT, EXPERIMENT, cfg,
            artifacts={"checkpoint": os.path.join(PKG_ROOT, CKPT),
                       "adjacency": os.path.join(pdir, "adjacency.npy"),
                       "node_meta": os.path.join(pdir, "node_meta.json"),
                       "test_split": os.path.join(pdir, "test.npz")},
            notes=("Explanation-network analysis, Stage 1. No LLM calls. "
                   "Exploratory. CPU only -- MPS reorders the top-k."))
        run.write_json("seeds.json", seed_rec)
        run.write_json("windows.json", {
            "window_idx": window_idx.tolist(),
            "stratum": strata.tolist(),
            "n_tod_bands": args.n_tod_bands,
            "n_congestion_strata": args.n_congestion_strata,
            "target_order": [int(t) for t in targets],
            "last_speed_mph_at_selected_windows": np.round(
                last_mph[window_idx], 3).tolist(),
            "tod_frac": np.round(tod[window_idx], 6).tolist(),
        })
    run.start()

    sol_dir = os.path.join(run.path, "solves")
    os.makedirs(sol_dir, exist_ok=True)
    done = {int(f.split("_")[1].split(".")[0])
            for f in os.listdir(sol_dir) if f.startswith("target_")}
    todo = [t for t in targets if t not in done]

    print("run_id     : {}".format(run.run_id))
    print("targets    : {} ({} already done)".format(len(todo), len(done)))
    print("windows    : {}  strata {}".format(len(window_idx), len(set(strata.tolist()))))
    print("solves     : {}".format(len(todo) * len(window_idx)))
    print("workers    : {} x {} thread  device {}".format(
        args.workers, _WORKER_THREADS, args.device))
    sys.stdout.flush()

    ckpt_abs = os.path.join(PKG_ROOT, CKPT)
    jobs = [(t, window_idx.tolist()) for t in todo]
    t_start = time.time()
    n_done = 0

    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_init_worker,
                      initargs=(ckpt_abs, DATASET, args.device, args.epochs)) as pool:
            for res in pool.imap_unordered(_solve_target, jobs):
                t = res["target"]
                # Parent does every write: one writer, no append interleaving.
                np.savez_compressed(
                    os.path.join(sol_dir, "target_{}.npz".format(t)),
                    node_imp=res["node_imp"], window_idx=window_idx,
                    stratum=strata, target=np.int32(t),
                    target_speed_mph=last_mph[window_idx, t].astype(np.float32))
                run.append_jsonl("progress.jsonl", {
                    "target": t, "seconds": res["seconds"],
                    "n_windows": int(len(window_idx)),
                    "mean_importance": round(float(res["node_imp"].mean()), 6),
                    "self_importance": round(float(res["node_imp"][:, t].mean()), 6),
                })
                n_done += 1
                el = time.time() - t_start
                rate = el / n_done
                print("[{:3d}/{:3d}] target {:3d}  {:6.1f}s   elapsed {:5.1f}m   "
                      "eta {:5.1f}m".format(n_done, len(jobs), t, res["seconds"],
                                            el / 60, rate * (len(jobs) - n_done) / 60))
                sys.stdout.flush()
    except BaseException as exc:                       # noqa: BLE001 - record it
        run.fail("{}: {}".format(type(exc).__name__, exc))
        print("FAILED: {}: {}".format(type(exc).__name__, exc))
        raise

    run.complete(n_targets=len(done) + n_done, n_windows=int(len(window_idx)),
                 n_solves=(len(done) + n_done) * int(len(window_idx)),
                 wall_clock_minutes=round((time.time() - t_start) / 60, 2),
                 checkpoint_sha256=provenance.sha256_file(ckpt_abs))
    print("\nDONE -> {}".format(run.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
