"""PART 4, the expensive half — rerun the explainer at three sparsity coefficients.

WHAT THIS PRODUCES
    node_imp masks for 40 targets x 24 windows at lambda_size in
    {0.15 (current), 0.45 (3x), 1.50 (10x)}, plus a second explainer seed on a
    10-target subset, plus the 12 committed explanations re-solved at the current
    setting. scripts/analyze_sparsity_sweep.py turns these into the entropy /
    stability / adjacency-precision numbers.

WHAT IT DELIBERATELY DOES NOT RECOMPUTE
    The current setting at seed 0 for the 40 sweep targets ALREADY EXISTS: run
    20260824T235016Z__influence_graph_solves solved all 207 targets x the same 24
    windows at lambda_size 0.15, seed 0, on CPU. Those solves are read in place by
    the analysis. Re-solving them would burn 1.6 h to reproduce bytes we have.

WHY THE TARGETS ARE THE FIRST 40 OF THAT RUN'S ORDER
    That order is a seed-42 permutation, so its prefix is an unbiased sample of
    the graph rather than "the first 40 node indices" (which in METR-LA correlate
    with position in the sensor file, hence with geography). Taking the prefix
    also means every sweep target has its current-setting solve already on disk.

COST (measured, 48.28 s/solve/worker on this machine)
    3x  40 x 24 = 960 solves
    10x 40 x 24 = 960 solves
    seed-repeat  3 settings x 10 targets x 24 = 720 solves
    committed    11 distinct (target, window) pairs
                 -------------------------------------
                 2,651 solves / 6 workers ~= 5.9 h

    6 workers, not 8: this runs CONCURRENTLY with the Ollama half of the
    experiment, and the box has 16 GB with a 5.3 GB resident model.

CPU ONLY. MPS reorders the top-k (Spearman 0.578, top-8 Jaccard 0.399 at the same
seed) and top-k identity is the entire object of Part 4.

    python -m xtraffic.scripts.run_sparsity_sweep
    python -m xtraffic.scripts.run_sparsity_sweep --smoke
    python -m xtraffic.scripts.run_sparsity_sweep --resume   # skips finished jobs

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..reproducibility import provenance, run_dir, seeds
from ..utils.io_utils import PKG_ROOT

REPO_ROOT = os.path.dirname(PKG_ROOT)
CKPT = "models/gnn/checkpoints/metr_la_best.pt"

# The checkpoint the 12 COMMITTED explanations were actually produced by.
#
# Their JSONs say `model_checkpoint: "metr_la_best.pt"`, but that path now holds
# the epoch-54 model, and epoch 54 does not reproduce them: re-solving
# metr_la_1194_56 at seed 0 on epoch 54 gives a top-8 with ZERO overlap and an
# importance error of 0.647 on the committed entries. On
# metr_la_best_epoch34_ARCHIVE.pt the same solve reproduces them exactly —
# Jaccard 1.000, max importance difference 5e-5, which is 4-decimal rounding.
# So the committed explanations are epoch-34 artifacts and their masks must be
# recovered from epoch 34. We ALSO solve them on epoch 54 so the entropy of the
# committed explanations can be read on the same checkpoint as the sweep; the
# report gives both and says which is the committed artifact.
CKPT_COMMITTED = "models/gnn/checkpoints/metr_la_best_epoch34_ARCHIVE.pt"
DATASET = "metr_la"
HORIZON_STEP = 6
EXPERIMENT = "grounding_without_information"

# The Stage-1 run whose windows, target order and current-setting solves we reuse.
BASE_RUN = "20260824T235016Z__influence_graph_solves__73e75966__ee9a7944"

# (label, lambda_size). lambda_ent stays at the committed 0.05 throughout: the
# brief says a SPARSITY sweep, and lambda_ent is the binarisation pressure, a
# different knob. Moving both would confound them.
SETTINGS: List[Tuple[str, float]] = [
    ("current", 0.15),
    ("3x", 0.45),
    ("10x", 1.50),
]
LAMBDA_ENT = 0.05

N_TARGETS = 40
N_SEED_REPEAT_TARGETS = 10
SEED_REPEAT_SEEDS = [0, 1]      # seed 0 comes free for 3x/10x from the sweep itself

_WORKER_THREADS = 1
_W: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _init_worker(ckpt: str, dataset: str, epochs: int,
                 needed_windows: Optional[List[int]] = None) -> None:
    import torch

    torch.set_num_threads(_WORKER_THREADS)
    _W["dataset"] = dataset
    _W["epochs"] = epochs
    _W["builders"] = {}
    _W["default_ckpt"] = ckpt
    from ..utils.io_utils import processed_dir
    # Keep ONLY the windows this run will ever ask for. test.npz is
    # [~6850, 12, 207, 2] float32 = 136 MB, and every worker used to hold a
    # private copy of all of it while needing 35 windows (~0.7 MB). At 6-8
    # workers that is ~1 GB of resident memory on a 16 GB box that is already
    # swapping, and the page-faulting showed up as workers sitting at 44% of a
    # core with cores idle. Sliced into a dict keyed by window id so lookups
    # stay by absolute index and callers do not need to know about the packing.
    full = np.load(os.path.join(processed_dir(dataset), "test.npz"))["X"]
    if needed_windows is None:
        _W["X"] = {i: full[i] for i in range(full.shape[0])}
    else:
        _W["X"] = {int(i): np.array(full[int(i)]) for i in sorted(set(needed_windows))}
    del full


def _builder_for(ckpt: str):
    """Lazily build (and cache) one ExplanationBuilder per checkpoint per worker.

    Two checkpoints are in play — see CKPT_COMMITTED — and loading either 2,651
    times would cost more than the solves. Cached per process, built on demand so
    a worker that never touches the archive checkpoint never loads it.
    """
    import torch

    from ..models.explainer.explain import ExplanationBuilder
    if ckpt not in _W["builders"]:
        _W["builders"][ckpt] = ExplanationBuilder(
            ckpt, _W["dataset"], device=torch.device("cpu"), top_k=8,
            epochs=_W["epochs"], confidence_runs=0)
    return _W["builders"][ckpt]


def _solve_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """One (setting, seed, target) x its windows. Runs in a worker process.

    lambda_size is set on the already-constructed explainer rather than passed to
    a new one: GNNExplainer reads self.lambda_size inside the optimisation loop,
    so assigning it is the whole change, and rebuilding the builder per job would
    reload the checkpoint 2,651 times.
    """
    import torch

    b = _builder_for(job.get("checkpoint") or _W["default_ckpt"])
    X = _W["X"]
    b.explainer.lambda_size = float(job["lambda_size"])
    b.explainer.lambda_ent = LAMBDA_ENT

    t0 = time.time()
    windows = job["windows"]
    imp = np.zeros((len(windows), b.model.num_nodes), dtype=np.float32)
    for i, w in enumerate(windows):
        xi = torch.from_numpy(X[int(w)][None, ...]).float()  # [1,12,N,2]
        node_imp, _, _, _ = b.explainer.explain_target(
            xi, job["target"], HORIZON_STEP, seed=job["seed"])
        imp[i] = node_imp
    return {"key": job["key"], "target": int(job["target"]),
            "node_imp": imp, "windows": windows,
            "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------
def _committed_explanation_jobs() -> List[Dict[str, Any]]:
    """The 12 committed explanations, as (target, window) solve jobs.

    The 12 are exactly what evaluation/verify_traffic_unchanged.py pins: the first
    6 of faithfulness/explanations_cache/*.json plus the 6 scenario explanations
    in results/explanations/*.json. Their committed JSONs store only top_nodes
    (8 of 207), so the full node_imp vector an entropy needs is not in them and
    has to be re-solved at the current setting, seed 0, CPU, same checkpoint.
    We solve exactly the window each explanation was built from, recovered from
    the cache filename or from the `test#<idx>` stamp in meta.timestamp.
    """
    seen: Dict[Tuple[int, int], Dict[str, Any]] = {}
    cache = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness",
                         "explanations_cache", "*.json")
    for path in sorted(glob.glob(cache))[:6]:
        base = os.path.basename(path)[:-len(".json")]
        _, idx, node = base.rsplit("_", 2)
        seen.setdefault((int(idx), int(node)), {"sources": []})["sources"].append(base)

    scen = os.path.join(PKG_ROOT, "evaluation", "results", "explanations", "*.json")
    for path in sorted(glob.glob(scen))[:6]:
        with open(path) as fh:
            exp = json.load(fh)
        stamp = str(exp["meta"]["timestamp"])
        idx = int(stamp.split("test#")[1].split()[0].strip())
        node = int(exp["prediction"]["node_id"])
        seen.setdefault((idx, node), {"sources": []})["sources"].append(
            os.path.basename(path)[:-len(".json")])

    jobs = []
    for (idx, node), info in sorted(seen.items()):
        for tag, ckpt in (("ep34", CKPT_COMMITTED), ("ep54", CKPT)):
            jobs.append({
                "kind": "committed",
                "key": "committed_{}__idx{}_node{}".format(tag, idx, node),
                "setting": "current",
                "lambda_size": 0.15,
                "seed": 0,
                "target": node,
                "windows": [idx],
                "checkpoint": os.path.join(PKG_ROOT, ckpt),
                "checkpoint_tag": tag,
                "sources": info["sources"],
            })
    return jobs


def build_jobs(targets: List[int], windows: List[int],
               n_seed_repeat: int) -> List[Dict[str, Any]]:
    """All jobs, ORDERED so that an interrupted run is still analysable.

    Order is not cosmetic here. The pool consumes jobs in order, so whatever is
    at the end is what gets lost if the run is cut short. Two rules:

      1. The committed-explanation jobs go FIRST. There are 22 of them and each
         is a single window, so they cost about two minutes in total and they
         are what the "are the committed explanations flagged?" claim rests on.
         Losing them to a timeout would cost a headline for no compute saving.
      2. The rest are interleaved BY TARGET, not grouped by setting. Grouped by
         setting, a truncated run yields all of 3x and none of 10x — no sweep at
         all. Interleaved, it yields every setting for a prefix of targets,
         which is a smaller sweep rather than a broken one. The prefix is
         already a seeded shuffle of the graph, so it stays unbiased.
    """
    jobs: List[Dict[str, Any]] = _committed_explanation_jobs()
    repeat_targets = set(targets[:n_seed_repeat])

    for t in targets:
        for label, lam in SETTINGS:
            # seed 0 at the current setting already exists in BASE_RUN.
            if label != "current":
                jobs.append({
                    "kind": "sweep", "key": "{}__seed0__t{}".format(label, t),
                    "setting": label, "lambda_size": lam, "seed": 0,
                    "target": int(t), "windows": list(windows),
                })
            # Seed repeat: the SECOND seed only. Seed 0 is the sweep row (or,
            # for the current setting, the base run), so no seed is solved twice.
            if t in repeat_targets:
                for s in SEED_REPEAT_SEEDS:
                    if s == 0:
                        continue
                    jobs.append({
                        "kind": "seed_repeat",
                        "key": "{}__seed{}__t{}".format(label, s, t),
                        "setting": label, "lambda_size": lam, "seed": int(s),
                        "target": int(t), "windows": list(windows),
                    })
    return jobs


def _already_done(sol_dir: str, job: Dict[str, Any]) -> bool:
    """True only if a solve file exists AND matches the job it claims to be.

    Existence alone is not enough. A `--smoke` run writes to the same directory
    under the same keys with 4 windows and a reduced epoch count, and this run
    silently reused six such files as if they were full 24-window, 200-epoch
    solves — the resume logic had no way to tell them apart. So the check is
    against the CONTENT: the stored window ids must be exactly the ones this job
    asks for. A file that does not match is treated as absent and re-solved,
    which is the safe direction to be wrong in.

    (Smoke runs should really write to their own run directory. Until they do,
    this is the guard that makes the mistake impossible rather than unlikely.)
    """
    path = os.path.join(sol_dir, job["key"] + ".npz")
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as d:
            stored = [int(w) for w in d["window_idx"]]
            n_rows = int(d["node_imp"].shape[0])
    except Exception:                                      # noqa: BLE001
        return False                                       # unreadable => redo
    wanted = [int(w) for w in job["windows"]]
    return stored == wanted and n_rows == len(wanted)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None,
                    help="run directory to write into; default = latest "
                         "grounding_without_information run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-targets", type=int, default=N_TARGETS)
    ap.add_argument("--smoke", action="store_true",
                    help="3 targets x 4 windows, wiring check only")
    ap.add_argument("--resume", action="store_true",
                    help="skip jobs whose npz already exists (default behaviour "
                         "is the same; the flag is kept for clarity in logs)")
    args = ap.parse_args(argv)

    seeds.set_all_seeds(42)

    base_path = os.path.join(PKG_ROOT, run_dir.RAW, BASE_RUN)
    with open(os.path.join(base_path, "windows.json")) as fh:
        base_windows = json.load(fh)
    windows = list(base_windows["window_idx"])
    targets = [int(t) for t in base_windows["target_order"][:args.n_targets]]
    n_repeat = N_SEED_REPEAT_TARGETS

    if args.smoke:
        targets, windows, n_repeat = targets[:3], windows[:4], 2

    # ------------------------------------------------------------------ run dir
    from .noise_run import resolve_run
    run = resolve_run(args.run_id)

    sol_dir = os.path.join(run.path, "part4_solves")
    os.makedirs(sol_dir, exist_ok=True)

    jobs = build_jobs(targets, windows, n_repeat)
    todo = [j for j in jobs if not _already_done(sol_dir, j)]

    n_solves = sum(len(j["windows"]) for j in todo)
    print("run_id  : {}".format(run.run_id))
    print("settings: {}".format(SETTINGS))
    print("targets : {}  windows: {}  seed-repeat targets: {}"
          .format(len(targets), len(windows), n_repeat))
    print("jobs    : {} ({} already done)".format(len(todo), len(jobs) - len(todo)))
    print("solves  : {}  ~= {:.1f} h at {} workers"
          .format(n_solves, n_solves * 48.28 / args.workers / 3600, args.workers))
    print("reused  : current@seed0 for {} targets from {}".format(len(targets), BASE_RUN))
    sys.stdout.flush()

    if not todo:
        print("nothing to do")
        return 0

    # Write the sweep's own geometry record once (append-only dir => guard).
    geom = os.path.join(run.path, "part4_geometry.json")
    if not os.path.exists(geom):
        run.write_json("part4_geometry.json", {
            "base_run": BASE_RUN,
            "window_idx": windows,
            "stratum": base_windows["stratum"][:len(windows)],
            "targets": targets,
            "seed_repeat_targets": targets[:n_repeat],
            "settings": [{"label": l, "lambda_size": v} for l, v in SETTINGS],
            "lambda_ent": LAMBDA_ENT,
            "explainer_epochs": args.epochs,
            "horizon_step": HORIZON_STEP,
            "device": "cpu",
            "committed_jobs": [
                {"key": j["key"], "target": j["target"],
                 "window": j["windows"][0], "sources": j["sources"],
                 "checkpoint_tag": j["checkpoint_tag"]}
                for j in jobs if j["kind"] == "committed"],
            "checkpoint_sweep": CKPT,
            "checkpoint_committed_artifacts": CKPT_COMMITTED,
            "checkpoint_note": (
                "the 12 committed explanations are epoch-34 artifacts despite "
                "their JSONs naming metr_la_best.pt; see CKPT_COMMITTED in "
                "run_sparsity_sweep.py"),
        })

    ckpt_abs = os.path.join(PKG_ROOT, CKPT)
    needed = sorted({int(w) for j in todo for w in j["windows"]})
    t_start = time.time()
    n_done = 0
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_init_worker,
                      initargs=(ckpt_abs, DATASET, args.epochs, needed)) as pool:
            for res in pool.imap_unordered(_solve_job, todo):
                np.savez_compressed(
                    os.path.join(sol_dir, res["key"] + ".npz"),
                    node_imp=res["node_imp"],
                    window_idx=np.asarray(res["windows"], dtype=np.int64),
                    target=np.int32(res["target"]))
                run.append_jsonl("part4_progress.jsonl", {
                    "key": res["key"], "target": res["target"],
                    "n_windows": len(res["windows"]),
                    "seconds": res["seconds"],
                    "mean_importance": round(float(res["node_imp"].mean()), 6),
                    "self_importance": round(
                        float(res["node_imp"][:, res["target"]].mean()), 6),
                })
                n_done += 1
                el = time.time() - t_start
                print("[{:4d}/{:4d}] {:28s} {:6.1f}s  elapsed {:5.1f}m  eta {:5.1f}m"
                      .format(n_done, len(todo), res["key"], res["seconds"],
                              el / 60, el / n_done * (len(todo) - n_done) / 60))
                sys.stdout.flush()
    except BaseException as exc:                          # noqa: BLE001
        run.append_jsonl("part4_progress.jsonl",
                         {"error": "{}: {}".format(type(exc).__name__, exc)})
        print("FAILED: {}: {}".format(type(exc).__name__, exc))
        raise

    run.append_jsonl("part4_progress.jsonl", {
        "event": "part4_solves_complete",
        "n_jobs": n_done,
        "n_solves": n_solves,
        "wall_clock_minutes": round((time.time() - t_start) / 60, 2),
        "checkpoint_sha256": provenance.sha256_file(ckpt_abs),
    })
    print("\nDONE -> {}".format(sol_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
