"""Phase 3 — claim-level metrics, with refusal as its own outcome.

WHY NOT `metrics.py`
--------------------
`xtraffic/utils/metrics.py` already exists (masked MAE/RMSE/MAPE for the ST-GNN).
Naming this `claim_metrics.py` keeps the two unambiguous. The brief asked for
`src/evaluation/metrics.py`; this is that file under the project's existing
layout (Decision 3, docs/EXPERIMENT_PLAN.md §1).

WHAT CHANGES FROM THE COMMITTED METRIC
---------------------------------------
`evaluation/faithfulness.py` computed, per advisory:

    precision     = |citations hitting top-k| / |citations|
    hallucination = 1 - precision,  and  = 1.0 when there are NO citations

Three consequences, all fixed here:

1. A REFUSAL scored hallucination 1.000 — identical to confidently inventing
   four causes. That makes the planned no-answer control (ablation condition N)
   unscoreable, and it conflates "declined" with "fabricated" inside the
   committed condition-B rate. Here, refusals are counted separately, excluded
   from precision, and `refusal_rate` is reported alongside every headline
   number so abstention can never be laundered as accuracy.

2. UNSUPPORTED and CONTRADICTED were merged. Overreach ("cited a real node that
   was not selected") and false structural statements ("asserted the edge runs
   the other way") are different failures, and a mechanism that fixes one but
   not the other is invisible to a metric that adds them together.

3. Only one claim type existed. Per-type rates are reported here, because "the
   unsupported rate fell" means something different if it fell for node claims
   and rose for edge-direction claims.

THE HEADLINE NUMBER
-------------------
`unsupported_claim_rate` = (UNSUPPORTED + CONTRADICTED) / verifiable claims,
where verifiable EXCLUDES UNVERIFIABLE claims (action proposals, terminology,
counterfactuals needing a model rerun). Including them would let a condition
lower its rate by talking about unverifiable things — an obvious and easily
exploited loophole.

Every reported number is accompanied by its denominator. A rate over 3 claims
and a rate over 300 are not the same evidence, and a table that hides which is
which invites exactly the over-reading the project is trying to avoid.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .claim_types import (ADVERSE_VERDICTS, ALIGNMENT, CLAIM_TYPES,
                          CONTRADICTED, PARTIALLY_SUPPORTED, SUPPORTED,
                          UNSUPPORTED, UNVERIFIABLE, VERDICTS, ParseReport,
                          Verdict)


def _safe_div(num: float, den: float) -> Optional[float]:
    """Rate or None. NEVER 0.0 for an empty denominator — a zero would enter a
    mean and silently drag it down, which is how an empty condition can look
    like a good one."""
    return (num / den) if den else None


def score_claims(verdicts: Sequence[Verdict],
                 report: Optional[ParseReport] = None,
                 is_refusal: bool = False,
                 is_empty: bool = False,
                 reference: str = ALIGNMENT) -> Dict[str, Any]:
    """Metrics for ONE advisory.

    Args:
        verdicts:  one per extracted claim.
        report:    the ParseReport, for the unparsed rate.
        is_refusal: the model explicitly declined (claim_parser.is_refusal).
        is_empty:   the advisory had no content at all — distinct from a
                    refusal, because an empty output is usually a PARSE failure
                    and folding it into the abstention rate would hide that.
        reference: ALIGNMENT or GROUND_TRUTH; recorded, never mixed.
    """
    counts: Dict[str, int] = {v: 0 for v in VERDICTS}
    for v in verdicts:
        counts[v.verdict] += 1

    n_total = len(verdicts)
    n_unverifiable = counts[UNVERIFIABLE]
    n_verifiable = n_total - n_unverifiable
    n_adverse = counts[UNSUPPORTED] + counts[CONTRADICTED]
    n_supported = counts[SUPPORTED]

    out: Dict[str, Any] = {
        "reference": reference,
        # --- outcome flags, always reported next to the rates ---------------
        "is_refusal": bool(is_refusal),
        "is_empty": bool(is_empty),
        # --- denominators, never omitted -------------------------------------
        "n_claims": n_total,
        "n_verifiable": n_verifiable,
        "n_unverifiable": n_unverifiable,
        "n_supported": n_supported,
        "n_partially_supported": counts[PARTIALLY_SUPPORTED],
        "n_unsupported": counts[UNSUPPORTED],
        "n_contradicted": counts[CONTRADICTED],
        # --- headline --------------------------------------------------------
        "unsupported_claim_rate": _safe_div(n_adverse, n_verifiable),
        "contradiction_rate": _safe_div(counts[CONTRADICTED], n_verifiable),
        "overreach_rate": _safe_div(counts[UNSUPPORTED], n_verifiable),
        "claim_precision": _safe_div(n_supported, n_verifiable),
        "partial_rate": _safe_div(counts[PARTIALLY_SUPPORTED], n_verifiable),
        "unverifiable_rate": _safe_div(n_unverifiable, n_total),
        "verdict_counts": counts,
    }

    # --- per claim type -----------------------------------------------------
    by_type: Dict[str, Dict[str, Any]] = {}
    for t in CLAIM_TYPES:
        vs = [v for v in verdicts if v.claim.claim_type == t]
        if not vs:
            continue
        ver = [v for v in vs if v.verdict != UNVERIFIABLE]
        adverse = sum(1 for v in ver if v.verdict in ADVERSE_VERDICTS)
        by_type[t] = {
            "n": len(vs), "n_verifiable": len(ver),
            "n_supported": sum(1 for v in ver if v.verdict == SUPPORTED),
            "n_contradicted": sum(1 for v in ver if v.verdict == CONTRADICTED),
            "unsupported_claim_rate": _safe_div(adverse, len(ver)),
        }
    out["by_claim_type"] = by_type

    # --- per source ---------------------------------------------------------
    # The committed metric only ever saw `cited_causes`. Splitting by source is
    # what shows whether prose claims behave differently from declared ones —
    # and prose is where the ablation matrix's prose conditions live.
    by_source: Dict[str, Dict[str, Any]] = {}
    for src in sorted({v.claim.source for v in verdicts}):
        vs = [v for v in verdicts if v.claim.source == src]
        ver = [v for v in vs if v.verdict != UNVERIFIABLE]
        adverse = sum(1 for v in ver if v.verdict in ADVERSE_VERDICTS)
        by_source[src] = {
            "n": len(vs), "n_verifiable": len(ver),
            "unsupported_claim_rate": _safe_div(adverse, len(ver)),
        }
    out["by_source"] = by_source

    # --- parser honesty -----------------------------------------------------
    if report is not None:
        out["n_sentences"] = report.n_sentences
        out["n_unparsed"] = len(report.unparsed)
        out["unparsed_rate"] = report.unparsed_rate

    return out


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-advisory scores over a condition.

    TWO AGGREGATIONS, REPORTED SIDE BY SIDE, because they answer different
    questions and disagree in a revealing way:

      MACRO — mean of per-advisory rates. One advisory, one vote. This is the
              unit the paired bootstrap resamples, and it matches how the
              committed studies aggregated.
      MICRO — pooled over all claims. One claim, one vote. Dominated by verbose
              advisories.

    A large macro/micro gap means claim counts vary sharply across advisories,
    which itself matters: a condition that makes the model say LESS can lower
    its macro rate without becoming more truthful. Reporting only one of the two
    hides that.

    `n_refusals` and `refusal_rate` are always present. A condition's headline
    number is not interpretable without them: driving the unsupported rate to
    zero by refusing everything is a degenerate win, and the brief explicitly
    requires ruling it out.
    """
    n = len(rows)
    if n == 0:
        return {"n_advisories": 0, "note": "no advisories to aggregate"}

    refusals = [r for r in rows if r.get("is_refusal")]
    empties = [r for r in rows if r.get("is_empty")]
    # Refusals and empty outputs make no claims, so including them in the
    # macro mean would score an abstention as a perfect advisory.
    answering = [r for r in rows
                 if not r.get("is_refusal") and not r.get("is_empty")]

    def macro(key: str) -> Dict[str, Any]:
        vals = [r[key] for r in answering
                if r.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / len(vals)
        return {"mean": m, "std": var ** 0.5, "n": len(vals)}

    tot_ver = sum(r.get("n_verifiable", 0) for r in rows)
    tot_adv = sum(r.get("n_unsupported", 0) + r.get("n_contradicted", 0)
                  for r in rows)
    tot_con = sum(r.get("n_contradicted", 0) for r in rows)
    tot_sup = sum(r.get("n_supported", 0) for r in rows)

    out: Dict[str, Any] = {
        "n_advisories": n,
        "n_answering": len(answering),
        "n_refusals": len(refusals),
        "n_empty": len(empties),
        # Denominator is ALL advisories: the refusal rate must be a share of
        # what was asked, not of what was answered.
        "refusal_rate": len(refusals) / n,
        "empty_rate": len(empties) / n,

        "macro": {
            "unsupported_claim_rate": macro("unsupported_claim_rate"),
            "contradiction_rate": macro("contradiction_rate"),
            "overreach_rate": macro("overreach_rate"),
            "claim_precision": macro("claim_precision"),
            "unverifiable_rate": macro("unverifiable_rate"),
            "unparsed_rate": macro("unparsed_rate"),
        },
        "micro": {
            "n_claims": sum(r.get("n_claims", 0) for r in rows),
            "n_verifiable": tot_ver,
            "unsupported_claim_rate": _safe_div(tot_adv, tot_ver),
            "contradiction_rate": _safe_div(tot_con, tot_ver),
            "claim_precision": _safe_div(tot_sup, tot_ver),
        },
    }

    # --- per claim type, pooled --------------------------------------------
    by_type: Dict[str, Dict[str, Any]] = {}
    for t in CLAIM_TYPES:
        nv = sum(r.get("by_claim_type", {}).get(t, {}).get("n_verifiable", 0)
                 for r in rows)
        if not nv:
            continue
        ns = sum(r.get("by_claim_type", {}).get(t, {}).get("n_supported", 0)
                 for r in rows)
        nc = sum(r.get("by_claim_type", {}).get(t, {}).get("n_contradicted", 0)
                 for r in rows)
        by_type[t] = {
            "n_verifiable": nv,
            "unsupported_claim_rate": _safe_div(nv - ns, nv),
            "contradiction_rate": _safe_div(nc, nv),
        }
    out["by_claim_type"] = by_type

    # --- the caveats that must travel with the numbers ---------------------
    warnings: List[str] = []
    if out["refusal_rate"] > 0.0:
        warnings.append(
            "{:.1%} of advisories were refusals and are EXCLUDED from the macro "
            "rates. Compare conditions on refusal_rate as well: a lower "
            "unsupported rate bought by abstaining is not better grounding."
            .format(out["refusal_rate"]))
    mu = out["macro"]["unsupported_claim_rate"]["mean"]
    mi = out["micro"]["unsupported_claim_rate"]
    if mu is not None and mi is not None and abs(mu - mi) > 0.10:
        warnings.append(
            "macro ({:.3f}) and micro ({:.3f}) disagree by more than 0.10: "
            "claim counts vary sharply across advisories, so a condition may be "
            "lowering its macro rate by saying LESS rather than by being more "
            "accurate.".format(mu, mi))
    up = out["macro"]["unparsed_rate"]["mean"]
    if up is not None and up > 0.20:
        warnings.append(
            "unparsed rate is {:.1%}: the parser could not type a large share "
            "of assertive sentences in this condition. If this rate DIFFERS "
            "across conditions it is a confound, not a footnote.".format(up))
    if out["n_answering"] < 20:
        warnings.append(
            "only {} answering advisories: too few for a stable rate; report "
            "the count next to any figure quoted from this condition."
            .format(out["n_answering"]))
    out["warnings"] = warnings
    return out


def compare(a: Dict[str, Any], b: Dict[str, Any],
            label_a: str = "A", label_b: str = "B") -> Dict[str, Any]:
    """Point-estimate difference between two aggregated conditions.

    NO confidence interval is produced here, deliberately. A paired CI needs
    per-SCENARIO values so scenarios can be resampled together
    (`evaluation/bootstrap_ci.py` already does this correctly), and a CI built
    from aggregates would understate uncertainty by ignoring the pairing. This
    function reports differences and flags what still has to be done to them.
    """
    def d(path: str) -> Optional[float]:
        x, y = a, b
        for p in path.split("."):
            x = (x or {}).get(p) if isinstance(x, dict) else None
            y = (y or {}).get(p) if isinstance(y, dict) else None
        return (x - y) if (isinstance(x, (int, float))
                           and isinstance(y, (int, float))) else None

    return {
        "label_a": label_a, "label_b": label_b,
        "n_a": a.get("n_advisories"), "n_b": b.get("n_advisories"),
        "diff_unsupported_claim_rate_macro":
            d("macro.unsupported_claim_rate.mean"),
        "diff_unsupported_claim_rate_micro": d("micro.unsupported_claim_rate"),
        "diff_contradiction_rate_macro": d("macro.contradiction_rate.mean"),
        "diff_claim_precision_macro": d("macro.claim_precision.mean"),
        "diff_refusal_rate": d("refusal_rate"),
        "diff_unparsed_rate": d("macro.unparsed_rate.mean"),
        "caveats": [
            "Point estimates only. Paired bootstrap CIs require per-scenario "
            "values — use evaluation/bootstrap_ci.py, which resamples scenarios "
            "so the pairing is preserved.",
            "No multiple-comparison correction is applied here. The ablation "
            "matrix compares many conditions; corrections belong in the "
            "prespecified analysis plan (Phase 11), not in this function.",
            "A difference in refusal_rate or unparsed_rate between conditions "
            "is a CONFOUND for the headline difference, not a side note.",
        ],
    }
