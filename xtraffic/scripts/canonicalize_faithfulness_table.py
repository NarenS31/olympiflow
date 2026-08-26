"""Canonicalise the faithfulness table: one corpus, one epoch, five conditions.

WHAT THIS DOES AND DOES NOT DO
    It assembles A / B / C (from the committed Phase-5 summary) beside
    A_rand / A_mismatch (from the grounding-without-information run), verifies
    they rest on the SAME explanation corpus at the SAME checkpoint epoch, and
    writes claims.csv.

    It does NOT reconcile competing values for the same condition. There are at
    least three condition-A numbers on disk — 0.7246 (Phase-5 summary), 0.7329
    (the 15b run), 0.7232 (the cross-city rebuild) — and they are different LLM
    draws, not disagreeing measurements of one quantity. Averaging them or
    picking the nicest would manufacture a number no run produced.

THE ONE CONFOUND WORTH MEASURING
    The entity resolver was fixed on 2026-07-21 (commit a5ba31d). A / B / C were
    scored on 2026-07-06, BEFORE that fix; A_rand / A_mismatch were scored in
    August, AFTER it. So the five rows are not automatically comparable even
    though they share a corpus.

    I expected to measure this from the decision cache. I could not: that cache
    was introduced with Phase 15b on 2026-07-12, so the 2026-07-06 run predates
    it and its raw `cited_locations` were never persisted. What the cache holds
    for condition A is the 15b draw, not the canonical one — mistaking the two is
    exactly the error this script exists to prevent, and it nearly went in.

    So the canonical rows carry an UNQUANTIFIED resolver caveat, plus a proxy
    bound measured on the 15b-era decisions over the same corpus.

    python -m xtraffic.scripts.canonicalize_faithfulness_table

Python 3.9 compatible.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ..evaluation.faithfulness import NodeTable, mean_std, score_advisory
from ..models.gnn.loaders import load_node_meta
from ..utils.io_utils import PKG_ROOT

FAITH = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness")
CACHE = os.path.join(FAITH, "explanations_cache")
DEC = os.path.join(FAITH, "decisions_cache")
RUN_ID = "20260825T201433Z__grounding_without_information__f193e764__a1b0fc36"
RUN = os.path.join(PKG_ROOT, "evaluation", "results", "raw", RUN_ID)

RESOLVER_FIX_COMMIT = "a5ba31d"
RESOLVER_FIX_DATE = "2026-07-21"
SWAP_DATE = "2026-07-19"          # metr_la_best.pt: epoch 34 -> epoch 54

METRICS = ["faithfulness_f1", "hallucination_rate",
           "cause_precision", "cause_recall"]


def _mtime(p: str) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))


# ---------------------------------------------------------------------------
# 1. Corpus + epoch consistency
# ---------------------------------------------------------------------------
def corpus_check() -> Dict[str, Any]:
    files = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    ts = sorted(os.path.getmtime(f) for f in files)
    swap = time.mktime(time.strptime(SWAP_DATE, "%Y-%m-%d"))
    after = [f for f in files if os.path.getmtime(f) >= swap]

    # Every committed explanation names a checkpoint. Pre-2026-08-26 artifacts
    # carry only a basename (audit 11.2), so the epoch is established by the
    # re-solve evidence recorded there, not by the field itself.
    names = set()
    for f in files:
        with open(f) as fh:
            names.add(json.load(fh)["meta"].get("model_checkpoint"))
    return {
        "n_explanations": len(files),
        "built_from": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts[0])),
        "built_to": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts[-1])),
        "files_written_on_or_after_checkpoint_swap": len(after),
        "meta_model_checkpoint_values": sorted(names),
        "epoch": 34,
        "epoch_evidence": (
            "corpus predates the {} swap entirely (0 of {} files written on or "
            "after it) AND re-solving metr_la_1154_197 / metr_la_1194_56 at seed "
            "0 reproduces the stored top-8 exactly on epoch 34 (Jaccard 1.000) "
            "and not at all on epoch 54 (0.231 / 0.000)".format(
                SWAP_DATE, len(files))),
    }


def run_epoch_check() -> Dict[str, Any]:
    """What the grounding run recorded as ACTUALLY loaded, per part."""
    p = os.path.join(RUN, "checkpoints_used.jsonl")
    if not os.path.exists(p):
        return {"error": "checkpoints_used.jsonl absent"}
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return {r["role"]: {"epoch": r.get("epoch"), "sha256": (r.get("sha256") or "")[:16],
                        "basename": r["basename"]} for r in rows}


# ---------------------------------------------------------------------------
# 2. Resolver delta on the committed A/B/C decisions
# ---------------------------------------------------------------------------
def resolver_delta_proxy() -> Dict[str, Any]:
    """Re-resolve CACHED citations with TODAY's resolver — a PROXY, not the thing.

    IMPORTANT, and the reason this is called a proxy: the canonical A / B / C run
    (2026-07-06) predates decision caching, which was introduced with Phase 15b
    on 2026-07-12. Its raw `cited_locations` were never persisted —
    `faithfulness_per_scenario.csv` stores metrics only, and its column list is
    pinned to exclude the debug keys. So the resolver-version difference between
    the A/B/C rows and the A_rand/A_mismatch rows CANNOT be measured from
    artifacts. It can only be settled by re-running A/B/C.

    What the cache does hold is the 15b-era decisions on the SAME 93-scenario
    corpus with the SAME condition-A prompt: 93 A, 93 C_RICH, 93 D_CONTRA, and a
    3-file C fragment. Re-resolving those bounds how much the resolver fix can
    move this metric on this corpus. It is a different LLM draw, so it bounds the
    effect; it does not correct the canonical rows.

    Nothing is regenerated and no LLM is called, so the only thing that can move
    a number here is the resolver itself.
    """
    table = NodeTable(load_node_meta("metr_la"))
    out: Dict[str, Any] = {}
    per_cond: Dict[str, List[Tuple[Dict[str, float], Dict[str, float]]]] = {}

    for path in sorted(glob.glob(os.path.join(DEC, "*.json"))):
        with open(path) as fh:
            row = json.load(fh)
        cond = row.get("condition")
        if not cond:
            continue
        exp_path = os.path.join(CACHE, "metr_la_{}_{}.json".format(
            row["sample_index"], row["target_node"]))
        if not os.path.exists(exp_path):
            continue
        with open(exp_path) as fh:
            exp = json.load(fh)
        advisory = {
            "reasoning": "",
            "cited_causes": [{"location": loc, "resolved_node_id": None}
                             for loc in (row.get("cited_locations") or [])],
            "recommendations": [],
        }
        now = score_advisory(exp, advisory, table)
        stored = {m: row.get(m) for m in METRICS}
        per_cond.setdefault(cond, []).append(
            ({m: now[m] for m in METRICS}, stored))

    for cond, pairs in sorted(per_cond.items()):
        rec: Dict[str, Any] = {"n": len(pairs)}
        for m in METRICS:
            s_mean, _, _ = mean_std([p[1][m] for p in pairs])
            n_mean, _, _ = mean_std([p[0][m] for p in pairs])
            rec[m] = {"stored": s_mean, "rescored_today": n_mean,
                      "delta": n_mean - s_mean}
        rec["n_scenarios_changed"] = sum(
            1 for now, st in pairs
            if any(abs((now[m] or 0) - (st[m] or 0)) > 1e-9 for m in METRICS))
        out[cond] = rec
    return out


# ---------------------------------------------------------------------------
# 3. Assemble the rows
# ---------------------------------------------------------------------------
def _from_summary(path: str, cond: str) -> Optional[Dict[str, Any]]:
    with open(path) as fh:
        d = json.load(fh)
    for a in d["overall"]:
        if a["condition"] == cond:
            return a
    return None


def _from_part1(cond: str) -> Optional[Dict[str, Any]]:
    with open(os.path.join(RUN, "part1_summary.json")) as fh:
        d = json.load(fh)
    for a in d["overall"]:
        if a["condition"] == cond:
            return a
    return None


def build_rows(delta: Dict[str, Any]) -> List[Dict[str, Any]]:
    sm = os.path.join(FAITH, "faithfulness_summary.json")
    s15 = os.path.join(FAITH, "faithfulness_summary_15b.json")
    p1 = os.path.join(RUN, "part1_summary.json")

    def dnote(_cond: str) -> str:
        """The resolver caveat for the pre-fix rows.

        Deliberately identical for A, B and C: none of their raw citations were
        retained, so nothing condition-specific can honestly be said. The proxy
        bound is quoted so the reader has a magnitude rather than an open worry.
        """
        d = delta.get("A") or {}
        if not d:
            return ("scored before the resolver fix; raw citations not retained, "
                    "so the difference vs the post-fix rows is UNQUANTIFIED")
        return ("scored BEFORE the {} resolver fix while A_rand/A_mismatch were "
                "scored after it. Raw citations from this run were NOT retained "
                "(decision caching began 2026-07-12), so the difference is "
                "UNQUANTIFIED from artifacts and can only be settled by "
                "re-running. PROXY BOUND on the same corpus: re-resolving the "
                "15b-era condition-A citations changes {}/{} scenarios, mean F1 "
                "{:+.4f}".format(RESOLVER_FIX_DATE, d["n_scenarios_changed"],
                                 d["n"], d["faithfulness_f1"]["delta"]))

    CORPUS = ("same 93-scenario epoch-34 corpus (faithfulness/"
              "explanations_cache/, built 2026-07-05..06)")
    rows: List[Dict[str, Any]] = []

    for cond in ("A", "B", "C"):
        a = _from_summary(sm, cond)
        rows.append({
            "condition": cond, "n": a["n"],
            "faithfulness_f1": round(a["faithfulness_f1_mean"], 4),
            "hallucination_rate": round(a["hallucination_rate_mean"], 4),
            "cause_precision": round(a["cause_precision_mean"], 4),
            "cause_recall": round(a["cause_recall_mean"], 4),
            "source_file": "evaluation/results/faithfulness/faithfulness_summary.json",
            "source_mtime": _mtime(sm),
            "checkpoint_epoch": 34,
            "scored_with_resolver": "pre-{} ({})".format(
                RESOLVER_FIX_DATE, RESOLVER_FIX_COMMIT),
            "corpus_consistency_note": "{}; explanations reused, not rebuilt. {}"
                                       .format(CORPUS, dnote(cond)),
        })

    for cond in ("A_rand", "A_mismatch"):
        a = _from_part1(cond)
        detail = {
            "A_rand": "prediction block copied verbatim from the corpus "
                      "artifact; node importances replaced by a seeded "
                      "uniform draw (evidence is synthetic, scenario set and "
                      "prediction are not)",
            "A_mismatch": "prediction block from the corpus artifact; evidence "
                          "fields taken from a DIFFERENT corpus artifact via a "
                          "seeded derangement (both halves are corpus members)",
        }[cond]
        rows.append({
            "condition": cond, "n": a["n"],
            "faithfulness_f1": round(a["faithfulness_f1_mean"], 4),
            "hallucination_rate": round(a["hallucination_rate_mean"], 4),
            "cause_precision": round(a["cause_precision_mean"], 4),
            "cause_recall": round(a["cause_recall_mean"], 4),
            "source_file": "evaluation/results/raw/{}/part1_summary.json".format(RUN_ID),
            "source_mtime": _mtime(p1),
            "checkpoint_epoch": 34,
            "scored_with_resolver": "post-{} (current)".format(RESOLVER_FIX_DATE),
            "corpus_consistency_note": "{}; {}".format(CORPUS, detail),
        })

    # C_RICH / D_CONTRA: real, executed, and sourced from their own run — which
    # carries its OWN condition-A row. Comparing them against the canonical A
    # above would cross two LLM draws, so the note says which A they belong to.
    a15 = _from_summary(s15, "A")
    for cond in ("C_RICH", "D_CONTRA"):
        a = _from_summary(s15, cond)
        if a is None:
            continue
        rows.append({
            "condition": cond, "n": a["n"],
            "faithfulness_f1": round(a["faithfulness_f1_mean"], 4),
            "hallucination_rate": round(a["hallucination_rate_mean"], 4),
            "cause_precision": round(a["cause_precision_mean"], 4),
            "cause_recall": round(a["cause_recall_mean"], 4),
            "source_file": "evaluation/results/faithfulness/faithfulness_summary_15b.json",
            "source_mtime": _mtime(s15),
            "checkpoint_epoch": 34,
            "scored_with_resolver": "pre-{} ({})".format(
                RESOLVER_FIX_DATE, RESOLVER_FIX_COMMIT),
            "corpus_consistency_note": (
                "{}; EXECUTED AND REAL (Phase 15b, n=93, llama3.1:8b). Compare "
                "against the 15b run's OWN condition A (F1 {:.4f}, halluc "
                "{:.4f}), NOT the canonical A row above — they are separate LLM "
                "draws on the same corpus.".format(
                    CORPUS, a15["faithfulness_f1_mean"],
                    a15["hallucination_rate_mean"])),
        })
    return rows


def main() -> int:
    corpus = corpus_check()
    epochs = run_epoch_check()
    delta = resolver_delta_proxy()
    rows = build_rows(delta)

    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "processed",
                           RUN_ID)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["condition", "n", "faithfulness_f1", "hallucination_rate",
            "cause_precision", "cause_recall", "source_file", "source_mtime",
            "checkpoint_epoch", "scored_with_resolver",
            "corpus_consistency_note"]
    csv_path = os.path.join(out_dir, "claims.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    prov = {
        "corpus": corpus,
        "grounding_run_checkpoints_used": epochs,
        "resolver_fix": {
            "commit": RESOLVER_FIX_COMMIT, "date": RESOLVER_FIX_DATE,
            "canonical_ABC_rows_scored": "before the fix (2026-07-06)",
            "A_rand_A_mismatch_rows_scored": "after the fix (2026-08-25/26)",
            "canonical_delta_measurable": False,
            "why_not": ("the 2026-07-06 run predates decision caching "
                        "(introduced 2026-07-12 with Phase 15b); its raw "
                        "cited_locations were never persisted and "
                        "faithfulness_per_scenario.csv stores metrics only"),
            "proxy_on_cached_15b_era_decisions": delta},
        "excluded_deliberately": {
            "cross_city_rebuild": {
                "values": {"A_faithfulness_f1": 0.7232, "B_hallucination": 0.839},
                "reason": ("different corpus, and it spans the {} checkpoint "
                           "swap. Not merged. If cited at all it is a SEPARATE "
                           "result with its own provenance, not a correction to "
                           "the rows in claims.csv.".format(SWAP_DATE)),
            }},
        "not_reconciled": (
            "at least three condition-A values exist on disk (0.7246 Phase-5, "
            "0.7329 Phase-15b, 0.7232 cross-city). They are different LLM draws, "
            "not disagreeing measurements of one quantity, and are left "
            "unreconciled by instruction."),
    }
    with open(os.path.join(out_dir, "claims_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=str)

    print("CORPUS: {} explanations, {} .. {}, {} written on/after the swap"
          .format(corpus["n_explanations"], corpus["built_from"][:10],
                  corpus["built_to"][:10],
                  corpus["files_written_on_or_after_checkpoint_swap"]))
    print("meta.model_checkpoint values in corpus: {}"
          .format(corpus["meta_model_checkpoint_values"]))
    print("\nGROUNDING RUN, checkpoints actually loaded:")
    for k, v in sorted(epochs.items()):
        print("  {:44s} epoch {}  {}".format(k, v["epoch"], v["basename"]))
    print("\nRESOLVER PROXY on CACHED decisions (15b-era; the canonical "
          "2026-07-06 citations were never persisted):")
    for cond, d in sorted(delta.items()):
        print("  {}: {}/{} scenarios changed | F1 {:.4f} -> {:.4f} ({:+.4f}) | "
              "halluc {:.4f} -> {:.4f} ({:+.4f})".format(
                  cond, d["n_scenarios_changed"], d["n"],
                  d["faithfulness_f1"]["stored"], d["faithfulness_f1"]["rescored_today"],
                  d["faithfulness_f1"]["delta"],
                  d["hallucination_rate"]["stored"],
                  d["hallucination_rate"]["rescored_today"],
                  d["hallucination_rate"]["delta"]))
    print("\n=== claims.csv ===")
    print("{:11s}{:>6s}{:>9s}{:>9s}{:>9s}{:>9s}  {}".format(
        "condition", "n", "F1", "halluc", "prec", "recall", "source"))
    for r in rows:
        print("{:11s}{:>6d}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}  {}".format(
            r["condition"], r["n"], r["faithfulness_f1"],
            r["hallucination_rate"], r["cause_precision"], r["cause_recall"],
            os.path.basename(r["source_file"])))
    print("\nwrote {}".format(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
