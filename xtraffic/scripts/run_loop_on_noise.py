"""PART 2 — the active grounding loop, pointed at random evidence.

Runs the EXISTING loop (`models.advisor.active_grounding.run_active_grounding`,
imported, not reimplemented) on the Part-1 `A_rand` artifacts, with the same
threshold (0.70), the same max_rounds (3), the same scenarios, the same scorer,
and the same correction builder that produced the committed n=93 run on real
explanations.

The only substitution is the artifact the loop is handed. Everything the loop
does with it — score, find missed top-k, name them in a correction, re-prompt —
is untouched code.

WHY THIS IS THE KEY FIGURE
    The loop's own documented caveat is that it ENFORCES grounding rather than
    verifying it. If that caveat is right, the loop cannot tell that its evidence
    is noise, and it should converge on `A_rand` exactly as it converged on `A`.
    A convergence curve on random evidence is what that caveat looks like when
    you actually measure it.

    python -m xtraffic.scripts.run_loop_on_noise
    python -m xtraffic.scripts.run_loop_on_noise --limit 3

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from ..evaluation.faithfulness import NodeTable
from ..evaluation.run_faithfulness_study import sample_scenarios
from ..models.advisor.active_grounding import (aggregate, load_grounding_config,
                                               most_missed_regions,
                                               run_active_grounding, write_csv,
                                               write_jsonl)
from ..models.advisor.advisor import Advisor, load_advisor_config
from ..models.explainer.explain import ExplanationBuilder
from ..models.gnn.loaders import load_node_meta
from .noise_run import PartCache, logged_call_fn, resolve_run

DATASET = "metr_la"
CITY = "metr_la"
CKPT = "models/gnn/checkpoints/metr_la_best_epoch34_ARCHIVE.pt"


def load_a_rand_artifacts(run_path: str, keys: List[int]) -> Dict[int, Dict[str, Any]]:
    """The A_rand artifacts Part 1 already built and wrote into the run dir.

    Part 2 reads them rather than rebuilding: the loop must operate on the SAME
    artifacts Part 1 scored, or the two parts are describing different objects.
    """
    art = os.path.join(run_path, "part1_artifacts")
    out: Dict[int, Dict[str, Any]] = {}
    missing = []
    for k in keys:
        p = os.path.join(art, "A_rand_{}.json".format(k))
        if not os.path.exists(p):
            missing.append(k)
            continue
        with open(p) as fh:
            out[k] = json.load(fh)
    if missing:
        raise SystemExit(
            "missing {} A_rand artifacts (e.g. {}). Run "
            "`python -m xtraffic.scripts.run_grounding_without_information` "
            "first — Part 2 runs on Part 1's artifacts.".format(
                len(missing), missing[:3]))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--per-stratum", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    run = resolve_run(args.run_id)
    cache = PartCache(run, "part2")
    cfg = load_advisor_config()
    host, model = cfg["ollama"]["host"], cfg["ollama"]["model"]
    f1_threshold, max_rounds = load_grounding_config(cfg)

    device = torch.device("cpu")
    builder = ExplanationBuilder(CKPT, DATASET, device=device)
    table = NodeTable(load_node_meta(DATASET))

    scenarios = sample_scenarios(DATASET, builder.scaler, args.per_stratum)
    if args.limit:
        scenarios = scenarios[:args.limit]
    keys = [sc["sample_index"] for sc in scenarios]
    artifacts = load_a_rand_artifacts(run.path, keys)

    advisor = Advisor(CITY, call_fn=logged_call_fn(
        run, "part2", host, model, float(cfg["ollama"]["temperature"]),
        max(600.0, float(cfg["ollama"]["timeout_seconds"]))))

    print("[part2] {} scenarios | A_rand | threshold {:.2f} | max_rounds {} | {}"
          .format(len(scenarios), f1_threshold, max_rounds, model))

    traces: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, sc in enumerate(scenarios):
        k = sc["sample_index"]
        trace = cache.get("A_rand__{}".format(k))
        if trace is None:
            trace = run_active_grounding(advisor, artifacts[k], table,
                                         f1_threshold, max_rounds)
            trace.update({"sample_index": k, "target_node": sc["target_node"],
                          "tod_band": sc["tod_band"],
                          "congestion": sc["congestion"],
                          "model": model, "condition": "A_rand"})
            cache.put("A_rand__{}".format(k), trace)
        traces.append(trace)
        f1s = " -> ".join("{:.3f}".format(c["faithfulness_f1"])
                          for c in trace["curve"])
        el = time.time() - t0
        print("  [{:3d}/{:3d}] idx={:5d} {:8s}/{:6s}  F1: {:30s} [{}] {}  eta {:5.1f}m"
              .format(i + 1, len(scenarios), k, sc["tod_band"], sc["congestion"],
                      f1s, "OK" if trace["reached_threshold"] else "unmet",
                      trace["stop_reason"],
                      (el / (i + 1)) * (len(scenarios) - i - 1) / 60))
        sys.stdout.flush()

    agg = aggregate(traces, max_rounds, f1_threshold)
    summary = {
        "part": 2,
        "condition": "A_rand",
        "dataset": DATASET,
        "model": model,
        "explanation_checkpoint": CKPT,
        "most_missed_regions": most_missed_regions(traces),
        **agg,
    }
    out_json = os.path.join(run.path, "part2_summary.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    write_csv(traces, os.path.join(run.path, "part2_per_round.csv"))
    write_jsonl(traces, os.path.join(run.path, "part2_traces.jsonl"))

    print("\n" + "=" * 72)
    print("PART 2 — ACTIVE GROUNDING ON RANDOM EVIDENCE (n={}, threshold {:.2f})"
          .format(agg["n_scenarios"], f1_threshold))
    print("=" * 72)
    print("{:>6s}{:>11s}{:>11s}{:>11s}{:>13s}{:>15s}".format(
        "round", "mean F1", "precision", "recall", "halluc", "reached thr %"))
    for row in agg["per_round"]:
        print("{:>6d}{:>11.3f}{:>11.3f}{:>11.3f}{:>13.3f}{:>15.1f}".format(
            row["round"], row["mean_f1"], row["mean_precision"],
            row["mean_recall"], row["mean_hallucination"],
            100.0 * row["frac_reached_threshold"]))
    print("-" * 72)
    print("mean F1 gain:                {:+.3f}".format(agg["mean_f1_gain"]))
    print("mean correction rounds used: {:.2f}".format(agg["mean_rounds_used"]))
    print("reached threshold overall:   {:.1f}%".format(
        100.0 * agg["frac_reached_overall"]))
    print("stop reasons: {}".format(agg["stop_reasons"]))
    print("\nwrote {}".format(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
