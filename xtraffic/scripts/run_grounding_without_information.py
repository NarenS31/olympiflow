"""PART 1 — grounding without information.

Runs the Phase-5 faithfulness pipeline on the SAME 93 stratified scenarios, same
prompts, same templates, same metric, under four conditions:

    A           reused from the committed run (no new LLM calls)
    B           reused from the committed run (no new LLM calls)
    A_rand      NEW: importances replaced by a seeded uniform draw
    A_mismatch  NEW: evidence taken from a different, seeded-random target

and reports precision / recall / F1 / hallucination with 10k bootstrap CIs over
the paired scenarios.

The token-parity guard runs FIRST and can veto the whole part: if `A_rand` or
`A_mismatch` prompts are not within 5% of A's mean token count, any difference we
measure is confounded by prompt length, and predictions.md says P1 is then
reported UNTESTED rather than confirmed.

    python -m xtraffic.scripts.run_grounding_without_information
    python -m xtraffic.scripts.run_grounding_without_information --tokens-only
    python -m xtraffic.scripts.run_grounding_without_information --limit 4

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..evaluation.bootstrap_ci import bootstrap_from_rows
from ..evaluation.faithfulness import NodeTable, mean_std, score_advisory
from ..evaluation.noise_conditions import (build_a_mismatch, build_a_rand,
                                           seeded_donor_permutation)
from ..evaluation.run_faithfulness_study import (build_or_load_explanation,
                                                 sample_scenarios)
from ..models.advisor.advisor import (Advisor, build_prompt_condition,
                                      load_advisor_config)
from ..models.advisor.knowledge_base import render_kb_block
from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import load_window
from ..models.gnn.loaders import load_node_meta
from ..utils.io_utils import PKG_ROOT
from .noise_run import PartCache, count_prompt_tokens, logged_call_fn, resolve_run

DATASET = "metr_la"
CITY = "metr_la"
# The prediction block of every A_rand / A_mismatch artifact is copied from the
# committed epoch-34 explanation, so this checkpoint is only ever used for its
# scaler and its graph geometry. Named explicitly anyway: silently loading a
# different model than the one an artifact came from is the exact failure this
# experiment turned up in the committed explanation JSONs.
CKPT = "models/gnn/checkpoints/metr_la_best_epoch34_ARCHIVE.pt"

NEW_CONDITIONS = ["A_rand", "A_mismatch"]
ALL_CONDITIONS = ["A", "B", "A_rand", "A_mismatch"]
METRIC_KEYS = ["cause_precision", "cause_recall", "faithfulness_f1",
               "hallucination_rate"]
TOKEN_TOLERANCE = 0.05
A_RAND_SEED = 20260825
A_MISMATCH_SEED = 20260826


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def build_artifacts(builder: ExplanationBuilder, scenarios: List[Dict[str, Any]]
                    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Return {condition: {sample_index: explanation}} for the two new conditions."""
    committed: Dict[int, Dict[str, Any]] = {}
    windows: Dict[int, torch.Tensor] = {}
    for sc in scenarios:
        committed[sc["sample_index"]] = build_or_load_explanation(builder, DATASET, sc)
        windows[sc["sample_index"]] = load_window(DATASET, sc["sample_index"])

    keys = [sc["sample_index"] for sc in scenarios]
    donors = seeded_donor_permutation(keys, A_MISMATCH_SEED)

    a_rand: Dict[int, Dict[str, Any]] = {}
    a_mismatch: Dict[int, Dict[str, Any]] = {}
    for i, k in enumerate(keys):
        # Per-scenario seed derived from the part seed + position, so adding a
        # scenario never reshuffles the draws of the ones before it.
        a_rand[k] = build_a_rand(builder, committed[k], windows[k],
                                 seed=A_RAND_SEED + i)
        a_mismatch[k] = build_a_mismatch(committed[k], committed[donors[k]],
                                         seed=A_MISMATCH_SEED + i)
    return {"A": committed, "A_rand": a_rand, "A_mismatch": a_mismatch}


# ---------------------------------------------------------------------------
# Token parity
# ---------------------------------------------------------------------------
def prompt_for(advisor: Advisor, exp: Dict[str, Any]) -> str:
    """The exact condition-A prompt for an explanation dict.

    A_rand / A_mismatch ARE condition A — the only thing that differs is the
    artifact — so they go through the identical retrieval + prompt path.
    """
    chunks = advisor.kb.retrieve(exp, top_k=advisor.top_k)
    kb_block = render_kb_block(chunks, advisor.max_context_chars)
    return build_prompt_condition(exp, kb_block, "A", domain=advisor.domain)


def token_parity(advisor: Advisor, artifacts: Dict[str, Dict[int, Dict[str, Any]]],
                 keys: List[int], host: str, model: str,
                 sample: int = 20) -> Dict[str, Any]:
    """Tokenised prompt length per condition, and the 5% guard verdict.

    Counted by the server that will consume them, because a tokeniser we picked
    ourselves could disagree with the one that matters.

    Measured on a SEEDED SUBSAMPLE of `sample` scenarios per condition, not all
    93. A server-side count costs a full prompt eval (~35 s on this box under
    load), so 279 of them would be ~2.7 h spent on a guard. Prompt length is
    tightly clustered — the first five condition-A prompts measured 1177 / 1176 /
    1203 / 1178 / 1184 tokens, sd ~11 on a mean of ~1184, under 1% — so 20 draws
    pin each condition's mean far inside the 5% band we are testing against.
    Character counts, which are free, are reported for ALL 93 as a cross-check,
    and the two new conditions get full-population token counts for free after
    the fact from `prompt_eval_count` on their real advisory calls.
    """
    rng = np.random.RandomState(42)
    pick = sorted(rng.choice(len(keys), size=min(sample, len(keys)),
                             replace=False).tolist())
    sampled = [keys[i] for i in pick]

    out: Dict[str, Any] = {"tolerance": TOKEN_TOLERANCE,
                           "token_sample_n": len(sampled),
                           "token_sample_seed": 42,
                           "token_sample_keys": sampled,
                           "per_condition": {}}
    for cond in ["A"] + NEW_CONDITIONS:
        counts = []
        for k in sampled:
            counts.append(count_prompt_tokens(prompt_for(advisor, artifacts[cond][k]),
                                              host, model))
        counts = [c for c in counts if c is not None]
        chars = [len(prompt_for(advisor, artifacts[cond][k])) for k in keys]
        out["per_condition"][cond] = {
            "n_tokens_measured": len(counts),
            "mean_prompt_tokens": float(np.mean(counts)) if counts else None,
            "std_prompt_tokens": float(np.std(counts)) if counts else None,
            "min_prompt_tokens": int(np.min(counts)) if counts else None,
            "max_prompt_tokens": int(np.max(counts)) if counts else None,
            "n_chars_measured": len(chars),
            "mean_prompt_chars": float(np.mean(chars)),
            "std_prompt_chars": float(np.std(chars)),
        }
    base = out["per_condition"]["A"]["mean_prompt_tokens"]
    base_chars = out["per_condition"]["A"]["mean_prompt_chars"]
    out["reference_condition"] = "A"
    out["checks"] = {}
    ok = True
    for cond in NEW_CONDITIONS:
        m = out["per_condition"][cond]["mean_prompt_tokens"]
        rel = (m - base) / base
        passed = abs(rel) <= TOKEN_TOLERANCE
        ok = ok and passed
        c = out["per_condition"][cond]["mean_prompt_chars"]
        out["checks"][cond] = {
            "mean_tokens": m, "reference_mean": base,
            "relative_difference": rel, "within_5pct": passed,
            "mean_chars": c, "reference_mean_chars": base_chars,
            "relative_difference_chars": (c - base_chars) / base_chars,
        }
    out["all_within_tolerance"] = ok
    return out


def full_population_token_stats(run_path: str) -> Dict[str, Any]:
    """Token counts for every A_rand / A_mismatch prompt actually sent.

    Free: `prompt_eval_count` comes back on every generate call and the caller
    logged it. Only first attempts (retry_index 0) count — a retry appends the
    validator's complaint and is a longer, different prompt.
    """
    path = os.path.join(run_path, "llm_calls.jsonl")
    if not os.path.exists(path):
        return {}
    by_cond: Dict[str, List[int]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("part") != "part1" or r.get("retry_index") != 0:
                continue
            if not r.get("ok") or r.get("prompt_eval_count") is None:
                continue
            label = str(r.get("label") or "")
            cond = label.split("|")[0]
            if cond in NEW_CONDITIONS:
                by_cond.setdefault(cond, []).append(int(r["prompt_eval_count"]))
    return {k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": int(np.min(v)), "max": int(np.max(v))}
            for k, v in by_cond.items() if v}


# ---------------------------------------------------------------------------
# Reused committed rows
# ---------------------------------------------------------------------------
def load_committed_rows(keys: List[int]) -> List[Dict[str, Any]]:
    """Condition A and B per-scenario rows from the committed Phase-5 CSV."""
    path = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness",
                        "faithfulness_per_scenario.csv")
    wanted = set(keys)
    rows: List[Dict[str, Any]] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["condition"] not in ("A", "B"):
                continue
            if int(r["sample_index"]) not in wanted:
                continue
            row: Dict[str, Any] = {
                "sample_index": int(r["sample_index"]),
                "target_node": int(r["target_node"]),
                "tod_band": r["tod_band"],
                "congestion": r["congestion"],
                "condition": r["condition"],
                "source": "committed",
            }
            for k in METRIC_KEYS + ["n_cited_causes", "n_topk"]:
                row[k] = float(r[k]) if r[k] not in ("", None) else None
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(rows: List[Dict[str, Any]], cond: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["condition"] == cond]
    out: Dict[str, Any] = {"condition": cond, "n": len(sub)}
    for k in METRIC_KEYS:
        m, s, n = mean_std([r.get(k) for r in sub])
        out[k + "_mean"] = m
        out[k + "_std"] = s
        out[k + "_n"] = n
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--per-stratum", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tokens-only", action="store_true",
                    help="run the parity guard and stop (no advisories)")
    args = ap.parse_args(argv)

    run = resolve_run(args.run_id)
    cache = PartCache(run, "part1")
    cfg = load_advisor_config()
    host = cfg["ollama"]["host"]
    model = cfg["ollama"]["model"]

    device = torch.device("cpu")
    builder = ExplanationBuilder(CKPT, DATASET, device=device)
    table = NodeTable(load_node_meta(DATASET))

    scenarios = sample_scenarios(DATASET, builder.scaler, args.per_stratum)
    if args.limit:
        scenarios = scenarios[:args.limit]
    keys = [sc["sample_index"] for sc in scenarios]
    by_key = {sc["sample_index"]: sc for sc in scenarios}
    print("[part1] {} scenarios | model {}".format(len(scenarios), model))

    print("[part1] building A_rand / A_mismatch artifacts ...")
    artifacts = build_artifacts(builder, scenarios)
    art_path = os.path.join(run.path, "part1_artifacts")
    os.makedirs(art_path, exist_ok=True)
    for cond in NEW_CONDITIONS:
        for k in keys:
            p = os.path.join(art_path, "{}_{}.json".format(cond, k))
            if not os.path.exists(p):
                with open(p, "w") as fh:
                    json.dump(artifacts[cond][k], fh, indent=2)

    # --- token parity, before any advisory ---------------------------------
    tok_path = os.path.join(run.path, "part1_token_parity.json")
    if os.path.exists(tok_path):
        with open(tok_path) as fh:
            tok = json.load(fh)
        print("[part1] token parity: reusing existing measurement")
    else:
        print("[part1] measuring prompt tokens on the serving model ...")
        tok = token_parity(
            Advisor(CITY), artifacts, keys, host, model)
        run.write_json("part1_token_parity.json", tok)
    for cond, chk in tok["checks"].items():
        print("  {:11s} {:7.1f} tok vs A {:7.1f}  ({:+.2%})  {}".format(
            cond, chk["mean_tokens"], chk["reference_mean"],
            chk["relative_difference"], "OK" if chk["within_5pct"] else "FAIL"))
    if args.tokens_only:
        return 0 if tok["all_within_tolerance"] else 1
    if not tok["all_within_tolerance"]:
        print("[part1] token parity FAILED -> P1 is UNTESTED (predictions.md). "
              "Not running advisories.")
        return 1

    # --- advisories for the two new conditions -----------------------------
    call_fn = logged_call_fn(
        run, "part1", host, model, float(cfg["ollama"]["temperature"]),
        float(cfg["ollama"]["timeout_seconds"]))
    advisor = Advisor(CITY, call_fn=call_fn)

    rows: List[Dict[str, Any]] = load_committed_rows(keys)
    print("[part1] reused {} committed A/B rows".format(len(rows)))

    todo = [(cond, k) for cond in NEW_CONDITIONS for k in keys]
    t0 = time.time()
    for i, (cond, k) in enumerate(todo):
        ck = "{}__{}".format(cond, k)
        row = cache.get(ck)
        if row is None:
            exp = artifacts[cond][k]
            call_fn.label = "{}|{}".format(cond, k)
            res = advisor.advise_condition(exp, "A")
            metrics = score_advisory(exp, res["advisory"], table)
            sc = by_key[k]
            row = {
                "sample_index": k, "target_node": sc["target_node"],
                "tod_band": sc["tod_band"], "congestion": sc["congestion"],
                "condition": cond, "source": "new",
                "advisory_error": res["advisory"].get("_error", ""),
                "cited_locations": [c.get("location") for c in
                                    (res["advisory"].get("cited_causes") or [])
                                    if isinstance(c, dict)],
                "context_used": res.get("context_used", []),
                "explanation_confidence": exp.get("explanation_confidence"),
                "reasoning": res["advisory"].get("reasoning", ""),
            }
            for mk in METRIC_KEYS + ["n_cited_causes", "n_topk",
                                     "quantitative_fidelity"]:
                row[mk] = metrics[mk]
            cache.put(ck, row)
        rows.append(row)
        el = time.time() - t0
        print("  [{:3d}/{:3d}] {:11s} idx={:5d}  F1 {:.3f}  P {:.3f}  halluc {:.3f}"
              "   eta {:5.1f}m".format(
                  i + 1, len(todo), cond, k, row["faithfulness_f1"],
                  row["cause_precision"], row["hallucination_rate"],
                  (el / (i + 1)) * (len(todo) - i - 1) / 60))
        sys.stdout.flush()

    # --- aggregate + bootstrap --------------------------------------------
    aggs = [aggregate(rows, c) for c in ALL_CONDITIONS]
    boot = bootstrap_from_rows(
        rows, conditions=ALL_CONDITIONS, metrics=METRIC_KEYS,
        pairs=[("A", "A_rand"), ("A", "A_mismatch"), ("A", "B"),
               ("A_rand", "A_mismatch")], seed=42)

    summary = {
        "part": 1,
        "n_scenarios": len(scenarios),
        "conditions": ALL_CONDITIONS,
        "reused_conditions": ["A", "B"],
        "new_conditions": NEW_CONDITIONS,
        "model": model,
        "explanation_checkpoint": CKPT,
        "token_parity": tok,
        # Full-population verification of the guard, free from the calls we made.
        "token_full_population": full_population_token_stats(run.path),
        "overall": aggs,
        "bootstrap_ci": boot,
        "explanation_confidence": {
            cond: {
                "mean": float(np.mean([artifacts[cond][k]["explanation_confidence"]
                                       for k in keys])),
                "min": float(np.min([artifacts[cond][k]["explanation_confidence"]
                                     for k in keys])),
                "max": float(np.max([artifacts[cond][k]["explanation_confidence"]
                                     for k in keys])),
            } for cond in ["A"] + NEW_CONDITIONS},
    }
    if not os.path.exists(os.path.join(run.path, "part1_summary.json")):
        run.write_json("part1_summary.json", summary)
    else:
        with open(os.path.join(run.path, "part1_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

    csv_path = os.path.join(run.path, "part1_per_scenario.csv")
    cols = (["sample_index", "target_node", "tod_band", "congestion", "condition",
             "source"] + METRIC_KEYS + ["n_cited_causes", "n_topk"])
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("\n=== PART 1 — FAITHFULNESS BY CONDITION (n={}) ===".format(len(scenarios)))
    hdr = "{:22s}".format("metric")
    for a in aggs:
        hdr += "{:>20s}".format("{} (n={})".format(a["condition"], a["n"]))
    print(hdr)
    print("-" * len(hdr))
    for k in METRIC_KEYS:
        line = "{:22s}".format(k)
        for a in aggs:
            line += "{:>20s}".format("{:.3f} +/- {:.3f}".format(
                a[k + "_mean"], a[k + "_std"]))
        print(line)
    print("\nwrote {}".format(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
