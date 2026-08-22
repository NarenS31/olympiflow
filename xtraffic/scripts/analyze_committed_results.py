"""Three cheap analyses of the COMMITTED corpus. No new LLM calls.

    python -m xtraffic.scripts.analyze_committed_results

Promised in docs/EXPERIMENT_PLAN.md §5 as the payoff for approving Phases 1+3
before any expensive compute. All three run on artifacts already on disk.

  A. REFUSALS INSIDE CONDITION B.  The committed metric scored an advisory with
     no cited causes as hallucination 1.000 — identical to confidently inventing
     four. So the headline "83.9% ungrounded hallucination rate" may be part
     fabrication and part abstention or parse failure. This separates them from
     `n_cited_causes` and `advisory_error`, which the per-scenario CSVs already
     record.

  B. THE MULTI-TYPE VERIFIER ON SURVIVING FULL-TEXT ADVISORIES.  Only ~100 of
     ~4,000 generations kept their text. Those are the ONLY existing material on
     which the Phase-3 verifier can be run retrospectively. This is the first
     evidence about whether the grounding result survives checking claim types
     the committed metric never examined — edges, directions, paths, lags.

  C. RESOLVER BACKEND SENSITIVITY.  `faithfulness.py` prefers
     `rapidfuzz.token_set_ratio` and silently falls back to
     `difflib.SequenceMatcher` against the SAME threshold of 82. rapidfuzz is
     NOT installed on this machine, so every number here was produced by the
     fallback. This re-resolves every logged citation under both similarity
     functions and counts the disagreements.

     HONESTY: `token_set_ratio` below is a faithful reimplementation of the
     documented algorithm, NOT rapidfuzz itself. It answers "does the choice of
     similarity function move the metric?", which is the scientific question.
     Confirming against the real library needs `pip install rapidfuzz==3.6.1`,
     which this script deliberately does not do to the user's environment.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG = os.path.join(REPO, "xtraffic")
RESULTS = os.path.join(PKG, "evaluation", "results")


# ===========================================================================
# A. Refusals and empty outputs inside the committed condition-B rate
# ===========================================================================
def analysis_a() -> Dict[str, Any]:
    path = os.path.join(RESULTS, "faithfulness", "faithfulness_per_scenario.csv")
    if not os.path.exists(path):
        return {"error": "missing {}".format(path)}

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    out: Dict[str, Any] = {"source": os.path.relpath(path, REPO),
                           "n_rows": len(rows), "conditions": {}}

    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        n = len(sub)

        def as_int(r: Dict[str, str], k: str) -> int:
            try:
                return int(float(r.get(k) or 0))
            except Exception:
                return 0

        def as_float(r: Dict[str, str], k: str) -> Optional[float]:
            try:
                return float(r[k])
            except Exception:
                return None

        no_causes = [r for r in sub if as_int(r, "n_cited_causes") == 0]
        errored = [r for r in sub if (r.get("advisory_error") or "").strip()]
        # Scored hallucination 1.0 == every citation missed, OR none existed.
        h1 = [r for r in sub if (as_float(r, "hallucination_rate") or 0) >= 0.999]
        # The distinction the committed metric could not draw:
        h1_empty = [r for r in h1 if as_int(r, "n_cited_causes") == 0]
        h1_fabricated = [r for r in h1 if as_int(r, "n_cited_causes") > 0]

        halluc = [as_float(r, "hallucination_rate") for r in sub]
        halluc = [x for x in halluc if x is not None]

        # Recompute the rate over ANSWERING advisories only.
        answering = [r for r in sub if as_int(r, "n_cited_causes") > 0]
        ans_h = [as_float(r, "hallucination_rate") for r in answering]
        ans_h = [x for x in ans_h if x is not None]

        out["conditions"][cond] = {
            "n": n,
            "reported_hallucination_mean": (sum(halluc) / len(halluc)
                                            if halluc else None),
            "n_zero_citations": len(no_causes),
            "n_advisory_error": len(errored),
            "n_scored_halluc_1.0": len(h1),
            "n_scored_1.0_because_EMPTY": len(h1_empty),
            "n_scored_1.0_with_REAL_citations": len(h1_fabricated),
            "share_of_max_score_that_is_emptiness":
                (len(h1_empty) / len(h1)) if h1 else None,
            "n_answering": len(answering),
            "hallucination_mean_ANSWERING_ONLY":
                (sum(ans_h) / len(ans_h) if ans_h else None),
        }
    return out


# ===========================================================================
# B. The Phase-3 verifier on surviving full-text advisories
# ===========================================================================
def _find_full_text_advisories() -> List[Tuple[str, Dict[str, Any]]]:
    """Every committed artifact that kept BOTH an explanation and an advisory."""
    found: List[Tuple[str, Dict[str, Any]]] = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "advisories", "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and d.get("advisory") and d.get("explanation"):
                found.append((os.path.relpath(path, REPO), d))
        except Exception:
            continue

    # Phase-13 regenerated advisories for the 7 condition-A failures.
    for path in sorted(glob.glob(os.path.join(
            RESULTS, "failure_modes", "failure_advisories", "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and d.get("advisory"):
                found.append((os.path.relpath(path, REPO), d))
        except Exception:
            continue
    return found


def analysis_b() -> Dict[str, Any]:
    from ..evaluation import claim_metrics
    from ..evaluation.claim_parser import is_empty, is_refusal, parse_advisory
    from ..evaluation.graph_claim_verifier import GraphClaimVerifier

    items = _find_full_text_advisories()
    if not items:
        return {"error": "no full-text advisories found on disk"}

    import numpy as np
    meta_cache: Dict[str, Dict[str, Any]] = {}
    adj_cache: Dict[str, Any] = {}

    def load(ds: str):
        if ds not in meta_cache:
            base = os.path.join(PKG, "data", "processed", ds)
            try:
                with open(os.path.join(base, "node_meta.json"), "r",
                          encoding="utf-8") as fh:
                    meta_cache[ds] = json.load(fh)
            except Exception:
                meta_cache[ds] = {}
            try:
                adj_cache[ds] = np.load(os.path.join(base, "adjacency.npy"))
            except Exception:
                adj_cache[ds] = None
        return meta_cache[ds], adj_cache[ds]

    scored: List[Dict[str, Any]] = []
    per_item: List[Dict[str, Any]] = []

    for rel, d in items:
        exp = d.get("explanation") or {}
        advisory = d.get("advisory") or {}
        ds = (exp.get("meta") or {}).get("city") or "metr_la"
        meta, adj = load(ds)

        rep = parse_advisory(advisory, meta, exp)
        ver = GraphClaimVerifier(exp, meta, adjacency=adj)
        verdicts = [ver.verify(c) for c in rep.claims]
        row = claim_metrics.score_claims(
            verdicts, rep, is_refusal=is_refusal(advisory),
            is_empty=is_empty(advisory))
        scored.append(row)

        # Old metric, same advisory, for a side-by-side on NODE claims only.
        old: Optional[float] = None
        try:
            from ..evaluation.faithfulness import NodeTable, score_advisory
            old = score_advisory(exp, advisory, NodeTable(meta))[
                "hallucination_rate"]
        except Exception:
            old = None

        per_item.append({
            "file": rel, "dataset": ds,
            "n_claims": row["n_claims"],
            "n_verifiable": row["n_verifiable"],
            "new_unsupported_rate": row["unsupported_claim_rate"],
            "new_contradiction_rate": row["contradiction_rate"],
            "old_hallucination_rate": old,
            "unparsed_rate": row.get("unparsed_rate"),
            "by_type": {k: v["unsupported_claim_rate"]
                        for k, v in row["by_claim_type"].items()},
        })

    agg = claim_metrics.aggregate(scored)

    # How much of the claim surface is NEW — i.e. invisible to the old metric?
    n_node = sum(r["by_claim_type"].get("node_existence", {}).get("n", 0)
                 for r in scored)
    n_total = sum(r["n_claims"] for r in scored)

    return {
        "n_advisories": len(items),
        "n_claims_total": n_total,
        "n_node_claims": n_node,
        "n_claims_invisible_to_old_metric": n_total - n_node,
        "share_of_claims_old_metric_never_saw":
            ((n_total - n_node) / n_total) if n_total else None,
        "aggregate": agg,
        "per_item": per_item,
    }


# ===========================================================================
# C. Resolver backend sensitivity
# ===========================================================================
def token_set_ratio(a: str, b: str) -> float:
    """Faithful reimplementation of the documented token_set_ratio algorithm.

    NOT rapidfuzz. Same definition: split into token sets, form
    intersection / a-remainder / b-remainder, build three sorted strings, and
    take the max pairwise SequenceMatcher ratio. The property that matters is
    the one the fallback lacks: order-insensitivity and tolerance of extra
    words, which is exactly why the same threshold of 82 can behave differently
    under the two functions.
    """
    from difflib import SequenceMatcher

    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = sorted(ta & tb)
    ra, rb = sorted(ta - tb), sorted(tb - ta)
    s_inter = " ".join(inter)
    s_a = (s_inter + " " + " ".join(ra)).strip()
    s_b = (s_inter + " " + " ".join(rb)).strip()
    pairs = [(s_inter, s_a), (s_inter, s_b), (s_a, s_b)]
    return max(100.0 * SequenceMatcher(None, x, y).ratio() for x, y in pairs)


def _logged_citations() -> List[Tuple[str, str]]:
    """(dataset, citation text) for every citation persisted anywhere."""
    out: List[Tuple[str, str]] = []
    for path in glob.glob(os.path.join(RESULTS, "**", "*.json"), recursive=True):
        if "decisions_cache" not in path and "decisions" not in path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        locs = d.get("cited_locations")
        if not isinstance(locs, list):
            continue
        ds = d.get("dataset") or d.get("city") or "metr_la"
        for loc in locs:
            if isinstance(loc, str) and loc.strip():
                out.append((ds, loc.strip()))
    for path in glob.glob(os.path.join(RESULTS, "**", "*.jsonl"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    locs = d.get("cited_locations")
                    if isinstance(locs, list):
                        ds = d.get("dataset") or d.get("city") or "metr_la"
                        for loc in locs:
                            if isinstance(loc, str) and loc.strip():
                                out.append((ds, loc.strip()))
        except Exception:
            continue
    return out


def analysis_c(max_citations: int = 20000) -> Dict[str, Any]:
    from ..evaluation import faithfulness as F

    cits = _logged_citations()
    if not cits:
        return {"error": "no logged citations found"}
    cits = cits[:max_citations]

    tables: Dict[str, Any] = {}
    for ds in sorted({d for d, _ in cits}):
        try:
            with open(os.path.join(PKG, "data", "processed", ds,
                                   "node_meta.json"), "r", encoding="utf-8") as fh:
                tables[ds] = F.NodeTable(json.load(fh))
        except Exception:
            tables[ds] = None

    original_ratio = F._ratio
    disagreements: List[Dict[str, Any]] = []
    n_compared = 0
    per_ds: Dict[str, Dict[str, int]] = {}

    try:
        for ds, loc in cits:
            table = tables.get(ds)
            if table is None:
                continue
            F._ratio = original_ratio                       # difflib
            r1 = F.resolve_location(loc, table)
            F._ratio = token_set_ratio                      # token_set_ratio
            r2 = F.resolve_location(loc, table)
            n_compared += 1
            st = per_ds.setdefault(ds, {"n": 0, "differ": 0})
            st["n"] += 1
            if r1.node_ids != r2.node_ids or r1.method != r2.method:
                st["differ"] += 1
                if len(disagreements) < 40:
                    disagreements.append({
                        "dataset": ds, "citation": loc,
                        "difflib": {"method": r1.method,
                                    "n_nodes": len(r1.node_ids),
                                    "matched": r1.matched},
                        "token_set_ratio": {"method": r2.method,
                                            "n_nodes": len(r2.node_ids),
                                            "matched": r2.matched},
                    })
    finally:
        F._ratio = original_ratio

    n_diff = sum(v["differ"] for v in per_ds.values())
    return {
        "installed_backend": F.FUZZ_BACKEND,
        "rapidfuzz_installed": F.FUZZ_BACKEND.startswith("rapidfuzz"),
        "threshold": F.FUZZY_THRESHOLD,
        "n_citations_compared": n_compared,
        "n_disagreements": n_diff,
        "disagreement_rate": (n_diff / n_compared) if n_compared else None,
        "per_dataset": per_ds,
        "examples": disagreements,
        "caveat": ("token_set_ratio here is a faithful reimplementation of the "
                   "documented algorithm, NOT rapidfuzz. It answers whether the "
                   "choice of similarity function moves the metric. Confirming "
                   "against the real library requires "
                   "`pip install rapidfuzz==3.6.1`."),
    }


# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["a", "b", "c"], help="run one analysis")
    ap.add_argument("--out", default=os.path.join(
        RESULTS, "processed", "committed_corpus_analysis.json"))
    args = ap.parse_args(argv)

    results: Dict[str, Any] = {}

    if args.only in (None, "a"):
        print("=" * 74)
        print("A. Refusals and empty outputs inside the committed rates")
        print("=" * 74)
        a = analysis_a()
        results["analysis_a_refusals"] = a
        if "error" in a:
            print("  " + a["error"])
        else:
            print("{:<8} {:>5} {:>10} {:>9} {:>9} {:>11} {:>11}".format(
                "cond", "n", "reported", "zero-cit", "err", "scored1.0",
                "answering"))
            for cond, c in sorted(a["conditions"].items()):
                print("{:<8} {:>5} {:>10} {:>9} {:>9} {:>11} {:>11}".format(
                    cond, c["n"],
                    "{:.3f}".format(c["reported_hallucination_mean"])
                    if c["reported_hallucination_mean"] is not None else "-",
                    c["n_zero_citations"], c["n_advisory_error"],
                    c["n_scored_halluc_1.0"], c["n_answering"]))
            print()
            for cond, c in sorted(a["conditions"].items()):
                if c["n_scored_halluc_1.0"]:
                    print("  {}: of {} advisories scored hallucination 1.000, "
                          "{} had ZERO citations and {} made real citations that "
                          "all missed.".format(
                              cond, c["n_scored_halluc_1.0"],
                              c["n_scored_1.0_because_EMPTY"],
                              c["n_scored_1.0_with_REAL_citations"]))

    if args.only in (None, "b"):
        print()
        print("=" * 74)
        print("B. Phase-3 multi-type verifier on surviving full-text advisories")
        print("=" * 74)
        b = analysis_b()
        results["analysis_b_verifier"] = b
        if "error" in b:
            print("  " + b["error"])
        else:
            print("  advisories with surviving text : {}".format(b["n_advisories"]))
            print("  atomic claims extracted        : {}".format(b["n_claims_total"]))
            print("  node claims (old metric's only : {}".format(b["n_node_claims"]))
            print("    claim type)")
            print("  claims the old metric NEVER saw: {} ({:.1%})".format(
                b["n_claims_invisible_to_old_metric"],
                b["share_of_claims_old_metric_never_saw"] or 0.0))
            agg = b["aggregate"]
            mu = agg["macro"]["unsupported_claim_rate"]
            print("\n  unsupported claim rate  macro {} / micro {}".format(
                "{:.3f}".format(mu["mean"]) if mu["mean"] is not None else "-",
                "{:.3f}".format(agg["micro"]["unsupported_claim_rate"])
                if agg["micro"]["unsupported_claim_rate"] is not None else "-"))
            print("  refusal rate            {:.3f}   unparsed rate {}".format(
                agg["refusal_rate"],
                "{:.3f}".format(agg["macro"]["unparsed_rate"]["mean"])
                if agg["macro"]["unparsed_rate"]["mean"] is not None else "-"))
            print("\n  by claim type (pooled):")
            for t, v in sorted(agg["by_claim_type"].items()):
                print("    {:<22} n={:<4} unsupported={}".format(
                    t, v["n_verifiable"],
                    "{:.3f}".format(v["unsupported_claim_rate"])
                    if v["unsupported_claim_rate"] is not None else "-"))
            for w in agg.get("warnings", []):
                print("\n  WARNING: " + w)

    if args.only in (None, "c"):
        print()
        print("=" * 74)
        print("C. Resolver backend sensitivity")
        print("=" * 74)
        c = analysis_c()
        results["analysis_c_resolver_backend"] = c
        if "error" in c:
            print("  " + c["error"])
        else:
            print("  installed backend   : {}".format(c["installed_backend"]))
            print("  rapidfuzz installed : {}".format(c["rapidfuzz_installed"]))
            print("  citations compared  : {}".format(c["n_citations_compared"]))
            print("  disagreements       : {} ({})".format(
                c["n_disagreements"],
                "{:.2%}".format(c["disagreement_rate"])
                if c["disagreement_rate"] is not None else "-"))
            for ds, st in sorted(c["per_dataset"].items()):
                print("    {:<12} {:>6} citations  {:>4} differ".format(
                    ds, st["n"], st["differ"]))
            if c["examples"]:
                print("\n  examples:")
                for e in c["examples"][:8]:
                    print("    {!r}".format(e["citation"][:56]))
                    print("      difflib          -> {} ({} node(s))".format(
                        e["difflib"]["method"], e["difflib"]["n_nodes"]))
                    print("      token_set_ratio  -> {} ({} node(s))".format(
                        e["token_set_ratio"]["method"],
                        e["token_set_ratio"]["n_nodes"]))
            print("\n  " + c["caveat"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\nsaved -> {}".format(os.path.relpath(args.out, REPO)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
