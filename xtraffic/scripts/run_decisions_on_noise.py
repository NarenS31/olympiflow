"""PART 3 — the decision simulation, run on random explanations.

Reuses the Phase-10 machinery unchanged: the same scenarios, the same
model-in-the-loop ground truth (already cached and condition-independent), the
same 5-intervention menu, the same decision prompt, the same 3 decision seeds,
the same accuracy / delay-reduction / consistency metrics.

One new condition:

    XTRAFFIC_RAND : the XTRAFFIC agent, shown an `A_rand` explanation and an
                    advisory written from that `A_rand` explanation.

RANDOM / RAW / XTRAFFIC are RE-AGGREGATED from the committed decision log on
exactly the same scenario subset, so all four arms are paired on one population.
The committed n=444 numbers are reported next to them but are not the comparison.

SUBSAMPLE (pre-registered)
    n=150 of the 444, chosen stratum-preserving: each (tod-band x congestion)
    cell keeps round(150 * cell_size / 444) scenarios, taken from a seed-42
    shuffle of that cell. Full n=444 x 3 seeds is 8.6 h of LLM time and did not
    fit the budget; the subsample keeps the stratification the comparison rests on.

    python -m xtraffic.scripts.run_decisions_on_noise
    python -m xtraffic.scripts.run_decisions_on_noise --limit 4

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..evaluation import sim_eval as se
from ..evaluation.noise_conditions import build_a_rand
from ..models.advisor.advisor import Advisor, load_advisor_config
from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import load_window
from ..utils.io_utils import PKG_ROOT
from .noise_run import PartCache, logged_call_fn, resolve_run

DATASET = "metr_la"
CITY = "metr_la"
CKPT = "models/gnn/checkpoints/metr_la_best_epoch34_ARCHIVE.pt"
N_SUBSAMPLE = 150
SUBSAMPLE_SEED = 42
A_RAND_SEED = 20260827
NEW_CONDITION = "XTRAFFIC_RAND"


def stratified_subsample(scenarios: List[Dict[str, Any]], n: int,
                         seed: int) -> List[Dict[str, Any]]:
    """Keep `n` scenarios, preserving the (tod_band x congestion) proportions.

    Largest-remainder allocation so the cell counts sum to exactly n, and a
    seeded shuffle inside each cell so which members survive is reproducible and
    not a function of sampling order.
    """
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for sc in scenarios:
        cells.setdefault((sc["tod_band"], sc["congestion"]), []).append(sc)

    total = len(scenarios)
    exact = {k: n * len(v) / total for k, v in cells.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    # Largest remainder: hand out the leftover slots to the cells that lost the
    # most to truncation, ties broken on the cell key so it stays deterministic.
    leftover = n - sum(alloc.values())
    order = sorted(cells, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in order[:leftover]:
        alloc[k] += 1

    out: List[Dict[str, Any]] = []
    for k in sorted(cells):
        pool = list(cells[k])
        random.Random("{}|{}|{}".format(seed, k[0], k[1])).shuffle(pool)
        out.extend(pool[:alloc[k]])
    out.sort(key=lambda s: s["sample_index"])
    return out


def load_committed_rows(keys: set) -> List[Dict[str, Any]]:
    """RAW / XTRAFFIC per-decision rows from the committed Phase-10 log."""
    path = os.path.join(PKG_ROOT, "evaluation", "results", "sim_eval",
                        "decisions.jsonl")
    rows: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("condition") == "RANDOM":
                continue
            if r.get("dataset") != DATASET or r["sample_index"] not in keys:
                continue
            rows.append(r)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--n", type=int, default=N_SUBSAMPLE)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap scenarios")
    args = ap.parse_args(argv)

    run = resolve_run(args.run_id)
    cache = PartCache(run, "part3")
    cfg = se._load_cfg()
    adv_cfg = load_advisor_config()
    host, model = adv_cfg["ollama"]["host"], adv_cfg["ollama"]["model"]
    llm = {"host": host, "model": model,
           "decision_temperature": cfg["llm"]["decision_temperature"],
           "timeout": max(600.0, float(adv_cfg["ollama"].get("timeout_seconds", 180)))}
    seeds = cfg["decision_seeds"]
    horizon = cfg["simulation"]["horizon_step"]

    device = torch.device("cpu")
    builder = ExplanationBuilder(CKPT, DATASET, device=device)

    all_scenarios = se.sample_scenarios(DATASET, builder.scaler, cfg)
    subset = stratified_subsample(all_scenarios, args.n, SUBSAMPLE_SEED)
    if args.limit:
        subset = subset[:args.limit]
    keys = set(sc["sample_index"] for sc in subset)
    print("[part3] {} of {} scenarios | {} seeds | {}".format(
        len(subset), len(all_scenarios), len(seeds), model))

    # --- A_rand artifacts + their advisories -------------------------------
    advisor = Advisor(CITY, call_fn=logged_call_fn(
        run, "part3", host, model, float(adv_cfg["ollama"]["temperature"]),
        max(600.0, float(adv_cfg["ollama"]["timeout_seconds"]))))

    art_dir = os.path.join(run.path, "part3_artifacts")
    os.makedirs(art_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    sim_records: List[Dict[str, Any]] = []
    t0 = time.time()
    n_done = 0
    total_calls = len(subset) * (1 + len(seeds))

    for i, sc in enumerate(subset):
        idx = sc["sample_index"]
        # Committed sim_eval explanation (cached, condition-independent) — the
        # artifact the committed XTRAFFIC arm saw. A_rand is built from it, so
        # both arms share a prediction block and differ only in evidence.
        exp = se.get_explanation(builder, sc, horizon)
        ap_path = os.path.join(art_dir, "A_rand_{}.json".format(idx))
        if os.path.exists(ap_path):
            with open(ap_path) as fh:
                exp_rand = json.load(fh)
        else:
            exp_rand = build_a_rand(builder, exp, load_window(DATASET, idx),
                                    seed=A_RAND_SEED + i)
            with open(ap_path, "w") as fh:
                json.dump(exp_rand, fh, indent=2)

        # Ground truth: cached from the committed run, condition-independent.
        # It depends on the top EDGE only through the reroute action, and the
        # reroute edge must stay the REAL one — the ground truth is a property of
        # the road network, not of what we showed the agent.
        top_edge = exp["top_edges"][0] if exp.get("top_edges") else None
        sim_out = se.get_sim(None, sc, top_edge,
                             lambda: load_window(DATASET, idx))
        gt = sim_out["ground_truth"]
        sim_records.append({"scenario_type": sc["scenario_type"],
                            "ground_truth": gt,
                            "delay_by_intervention": sim_out["delay_by_intervention"]})

        adv_key = "adv__{}".format(idx)
        cached_adv = cache.get(adv_key)
        if cached_adv is None:
            advisory = advisor.advise(exp_rand)["advisory"]
            cache.put(adv_key, {"advisory": advisory, "sample_index": idx})
            n_done += 1
        else:
            advisory = cached_adv["advisory"]

        for seed in seeds:
            dkey = "dec__{}__{}".format(idx, seed)
            row = cache.get(dkey)
            if row is None:
                chosen = se.decide(NEW_CONDITION.replace("_RAND", ""), exp_rand,
                                   advisory, cfg["interventions"], seed, llm, False)
                row = {
                    "dataset": DATASET, "scenario_id": sc["scenario_id"],
                    "sample_index": idx, "scenario_type": sc["scenario_type"],
                    "tod_band": sc["tod_band"], "congestion": sc["congestion"],
                    "seed": seed, "condition": NEW_CONDITION, "chosen": chosen,
                    "ground_truth": gt, "correct": int(chosen == gt),
                    "delay_reduction": sim_out["delay_by_intervention"].get(chosen, 0.0),
                }
                cache.put(dkey, row)
                n_done += 1
            rows.append(row)

        el = time.time() - t0
        rate = el / max(n_done, 1)
        print("  [{:3d}/{:3d}] idx={:5d} {:8s}/{:6s} gt={:16s} chose={:16s} eta {:5.1f}m"
              .format(i + 1, len(subset), idx, sc["tod_band"], sc["congestion"],
                      gt, rows[-1]["chosen"],
                      rate * (total_calls - n_done) / 60))
        sys.stdout.flush()

    # --- re-aggregate the committed arms on the SAME subset ----------------
    committed = load_committed_rows(keys)
    all_rows = committed + rows
    random_seeds = int(cfg.get("random_report_seeds", 200))

    # se.aggregate loops over the MODULE-LEVEL se.CONDITIONS list, so a condition
    # that is not in it is silently dropped rather than erroring — XTRAFFIC_RAND
    # would have been aggregated into nothing and the summary would have come back
    # missing the only arm this part exists to measure. Widen the list for the
    # call and put it back, so we reuse the committed run's exact metric code
    # (accuracy / delay reduction / consistency, per seed then mean +/- std)
    # rather than reimplementing it beside it.
    orig_conditions = se.CONDITIONS
    se.CONDITIONS = list(orig_conditions) + [NEW_CONDITION]
    try:
        aggs = se.aggregate(all_rows, seeds, sim_records, cfg["interventions"],
                            random_seeds)
    finally:
        se.CONDITIONS = orig_conditions

    summary = {
        "part": 3,
        "n_scenarios": len(subset),
        "n_scenarios_full_study": len(all_scenarios),
        "subsample_seed": SUBSAMPLE_SEED,
        "decision_seeds": seeds,
        "model": model,
        "explanation_checkpoint": CKPT,
        "new_condition": NEW_CONDITION,
        "conditions": aggs,
        "committed_full_study_n444": json.load(open(os.path.join(
            PKG_ROOT, "evaluation", "results", "sim_eval",
            "sim_eval_summary.json")))["conditions"],
    }
    with open(os.path.join(run.path, "part3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    csv_path = os.path.join(run.path, "part3_per_decision.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print("\n=== PART 3 — DECISION QUALITY ON RANDOM EXPLANATIONS (n={}) ==="
          .format(len(subset)))
    print("{:16s}{:>22s}{:>22s}{:>20s}".format(
        "condition", "accuracy", "delay_reduction", "consistency"))
    print("-" * 80)
    for a in aggs:
        print("{:16s}{:>22s}{:>22s}{:>20s}".format(
            a["condition"],
            "{:.3f} +/- {:.3f}".format(a["accuracy_mean"], a["accuracy_std"]),
            "{:.2f} +/- {:.2f}".format(a["delay_reduction_mean"],
                                       a["delay_reduction_std"]),
            "{:.3f} +/- {:.3f}".format(a["consistency_mean"], a["consistency_std"])))
    print("\nwrote {}".format(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
