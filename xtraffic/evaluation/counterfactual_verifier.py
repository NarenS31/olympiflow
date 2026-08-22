"""Phase 3 — verify counterfactual claims by RERUNNING the predictor.

`graph_claim_verifier.py` returns UNVERIFIABLE for every counterfactual claim,
deliberately: it must stay a pure function of stored evidence so it is cheap,
deterministic, and safe to run anywhere. Deciding "if X had been higher, Y would
not have congested" requires actually perturbing the input and running the
frozen model. That is this module's job.

WHAT THE COMMITTED PHASE-17 RESULT DOES AND DOES NOT SHOW
----------------------------------------------------------
`results/counterfactual/counterfactual_summary.json`:

    validity_rate           0.40   (8 of 20 targets flip within a 20 mph budget)
    mean_faithfulness_f1    1.000
    mean_hallucination_rate 0.000

The F1 of 1.000 is computed over the 8 VALID counterfactuals only. The 12
infeasible ones are excluded from that mean, not scored as failures. So
"narrative faithfulness 1.000" and "validity 40%" are not two independent
successes — the first is conditional on the second, and quoting it alone would
be misleading. `CLAUDE.md` states the 40% honestly; this module keeps the four
axes structurally separate so they cannot be conflated again:

  VALIDITY      does the proposed intervention actually flip the prediction when
                the frozen model is rerun? (measured, not asserted)
  FAITHFULNESS  does the LLM's narration match the counterfactual RECORD?
  MINIMALITY    is the intervention the smallest one that works?
  PLAUSIBILITY  is the intervention physically/structurally admissible at all?

They are reported separately and NEVER averaged into one score. An invalid
counterfactual narrated perfectly is not a partial success; it is a faithful
description of something that does not work.

REUSE
-----
`models/explainer/counterfactual.py` already implements the searcher and the
frozen-model forward pass (`CounterfactualSearcher.predict_target_mph`), using
the same perturbation family as the Phase-8/10 intervention simulator. This
module verifies CLAIMS against that machinery rather than reimplementing it.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .claim_types import (CONTRADICTED, PARTIALLY_SUPPORTED, SUPPORTED,
                          UNSUPPORTED, UNVERIFIABLE, Claim, Verdict)

# A claimed prediction change counts as matching the observed one when it is
# within this relative tolerance. Matches faithfulness.NUMERIC_TOLERANCE.
PREDICTION_TOLERANCE = 0.10

# Interventions outside this range are rejected as IMPLAUSIBLE before validity
# is even assessed. A "counterfactual" that requires teleporting a jammed
# segment to 90 mph is not a finding about the model, and scoring it as valid
# would launder an impossible intervention as an actionable one.
MAX_PLAUSIBLE_UPLIFT_MPH = 25.0
FREE_FLOW_CEILING_MPH = 70.0


class CounterfactualVerifier:
    """Verify counterfactual claims against a rerun of the frozen predictor.

    Args:
        searcher:   a `models.explainer.counterfactual.CounterfactualSearcher`,
                    or any object exposing `predict_target_mph(X, node)` and
                    `_apply_uplift(X, {node: mph})`.
        X:          the input window the prediction was made on.
        record:     the counterfactual record produced by the search, when one
                    exists (gives MINIMALITY a reference to compare against).
        free_flow:  the free-flow threshold defining "no longer congested".
    """

    def __init__(self, searcher: Any, X: Any,
                 record: Optional[Dict[str, Any]] = None,
                 flip_threshold_mph: float = 35.0,
                 free_flow_mph: float = FREE_FLOW_CEILING_MPH):
        self.searcher = searcher
        self.X = X
        self.record = record or {}
        self.flip_threshold = float(flip_threshold_mph)
        self.free_flow = float(free_flow_mph)

    # -- plausibility -------------------------------------------------------
    def check_plausibility(self, changes: Dict[int, float],
                           current_speeds: Optional[Dict[int, float]] = None
                           ) -> Dict[str, Any]:
        """Is the proposed intervention admissible at all?

        Checked BEFORE validity, because an impossible intervention that happens
        to flip the prediction is not a valid counterfactual — it is a
        demonstration that the model extrapolates badly.
        """
        reasons: List[str] = []
        for nid, uplift in changes.items():
            if uplift < 0:
                reasons.append(
                    "node {}: negative uplift ({:.1f} mph); the intervention "
                    "family only speeds segments up".format(nid, uplift))
            if uplift > MAX_PLAUSIBLE_UPLIFT_MPH:
                reasons.append(
                    "node {}: uplift {:.1f} mph exceeds the plausible maximum "
                    "{:.0f}".format(nid, uplift, MAX_PLAUSIBLE_UPLIFT_MPH))
            if current_speeds and nid in current_speeds:
                final = current_speeds[nid] + uplift
                if final > self.free_flow:
                    reasons.append(
                        "node {}: would reach {:.1f} mph, above the free-flow "
                        "ceiling {:.0f}".format(nid, final, self.free_flow))
        return {"plausible": not reasons, "reasons": reasons,
                "n_nodes_changed": len(changes),
                "total_budget_mph": sum(changes.values()) if changes else 0.0}

    # -- validity -----------------------------------------------------------
    def check_validity(self, target_node: int,
                       changes: Dict[int, float]) -> Dict[str, Any]:
        """RERUN the model under the intervention and record what happened.

        This is a MEASUREMENT, not an inference. `predicted_after` comes from
        the model, and `flipped` is the observed outcome.
        """
        try:
            before = float(self.searcher.predict_target_mph(self.X, target_node))
        except Exception as exc:                   # pragma: no cover
            return {"measured": False, "error": "baseline rerun failed: {}"
                    .format(exc)}
        try:
            Xp = self.searcher._apply_uplift(self.X, changes)
            after = float(self.searcher.predict_target_mph(Xp, target_node))
        except Exception as exc:                   # pragma: no cover
            return {"measured": False, "error": "perturbed rerun failed: {}"
                    .format(exc)}

        return {
            "measured": True,
            "predicted_before_mph": before,
            "predicted_after_mph": after,
            "observed_delta_mph": after - before,
            "flip_threshold_mph": self.flip_threshold,
            "flipped": after >= self.flip_threshold,
        }

    # -- minimality ---------------------------------------------------------
    def check_minimality(self, target_node: int,
                         changes: Dict[int, float]) -> Dict[str, Any]:
        """Would a SMALLER intervention have worked?

        Halves each node's uplift and reruns. If the target still flips, the
        proposed intervention was not minimal. Cheap (one extra forward pass)
        and sufficient to catch a grossly oversized claim; it does not prove
        strict minimality and does not claim to.
        """
        if not changes:
            return {"assessed": False, "reason": "no changes proposed"}
        half = {k: v / 2.0 for k, v in changes.items()}
        res = self.check_validity(target_node, half)
        if not res.get("measured"):
            return {"assessed": False, "reason": res.get("error")}
        return {
            "assessed": True,
            "half_budget_also_flips": bool(res["flipped"]),
            "minimal": not res["flipped"],
            "half_budget_predicted_mph": res["predicted_after_mph"],
            "note": ("halving the uplift still flips the target, so the "
                     "proposed intervention is larger than necessary"
                     if res["flipped"] else
                     "halving the uplift no longer flips the target, "
                     "consistent with (but not proof of) minimality"),
        }

    # -- the claim ----------------------------------------------------------
    def verify(self, claim: Claim, target_node: int,
               changes: Dict[int, float],
               claimed_delta_mph: Optional[float] = None,
               current_speeds: Optional[Dict[int, float]] = None) -> Verdict:
        """Verify one counterfactual claim on all four axes.

        The verdict reflects VALIDITY + FAITHFULNESS. Plausibility and
        minimality ride along in `evidence` and are reported separately — they
        are properties of the intervention, not of the model's honesty about it.
        """
        plaus = self.check_plausibility(changes, current_speeds)
        ev: Dict[str, Any] = {"plausibility": plaus, "changes": changes,
                              "target_node": target_node}

        if not plaus["plausible"]:
            return Verdict(
                claim, CONTRADICTED,
                "proposed intervention is not physically admissible: {}".format(
                    "; ".join(plaus["reasons"][:2])),
                evidence=ev)

        valid = self.check_validity(target_node, changes)
        ev["validity"] = valid
        if not valid.get("measured"):
            return Verdict(claim, UNVERIFIABLE,
                           "could not rerun the model: {}".format(
                               valid.get("error")), evidence=ev)

        ev["minimality"] = self.check_minimality(target_node, changes)

        if not valid["flipped"]:
            return Verdict(
                claim, CONTRADICTED,
                "rerunning the model under this intervention does NOT produce "
                "the claimed outcome: target moves {:.1f} -> {:.1f} mph, still "
                "below the {:.0f} mph threshold".format(
                    valid["predicted_before_mph"], valid["predicted_after_mph"],
                    self.flip_threshold),
                evidence=ev)

        # The intervention works. Did the model describe its EFFECT correctly?
        if claimed_delta_mph is None:
            return Verdict(
                claim, SUPPORTED,
                "intervention is plausible and the rerun confirms the target "
                "flips ({:.1f} -> {:.1f} mph); no numeric change was claimed, "
                "so magnitude was not checked".format(
                    valid["predicted_before_mph"], valid["predicted_after_mph"]),
                evidence=ev)

        observed = valid["observed_delta_mph"]
        ev["claimed_delta_mph"] = claimed_delta_mph
        if abs(claimed_delta_mph - observed) <= PREDICTION_TOLERANCE * max(
                abs(observed), 1e-9):
            return Verdict(
                claim, SUPPORTED,
                "rerun confirms the flip and the claimed change ({:+.1f} mph) "
                "matches the observed ({:+.1f} mph) within +/-{:.0%}".format(
                    claimed_delta_mph, observed, PREDICTION_TOLERANCE),
                evidence=ev)

        return Verdict(
            claim, PARTIALLY_SUPPORTED,
            "the intervention does flip the target, but the claimed change "
            "({:+.1f} mph) does not match the observed ({:+.1f} mph)".format(
                claimed_delta_mph, observed),
            evidence=ev)


def summarise(verdicts: Sequence[Verdict]) -> Dict[str, Any]:
    """Aggregate counterfactual verdicts, keeping the four axes SEPARATE.

    Deliberately does NOT emit a single combined score. Phase 17's committed
    summary reports `mean_faithfulness_f1 1.000` next to `validity_rate 0.40`,
    where the F1 is conditional on validity — averaging the axes would hide
    exactly that dependency. `faithfulness_among_valid` is labelled so its
    denominator is unmissable.
    """
    n = len(verdicts)
    if n == 0:
        return {"n": 0}

    valid = [v for v in verdicts
             if v.evidence.get("validity", {}).get("flipped")]
    plausible = [v for v in verdicts
                 if v.evidence.get("plausibility", {}).get("plausible")]
    minimal = [v for v in verdicts
               if v.evidence.get("minimality", {}).get("minimal")]
    assessed_min = [v for v in verdicts
                    if v.evidence.get("minimality", {}).get("assessed")]
    faithful_among_valid = [v for v in valid if v.verdict == SUPPORTED]

    return {
        "n": n,
        "validity_rate": len(valid) / n,
        "plausibility_rate": len(plausible) / n,
        "minimality_rate": (len(minimal) / len(assessed_min)
                            if assessed_min else None),
        "n_valid": len(valid),
        "n_assessed_for_minimality": len(assessed_min),
        "faithfulness_among_valid": (len(faithful_among_valid) / len(valid)
                                     if valid else None),
        "faithfulness_denominator": len(valid),
        "note": ("faithfulness_among_valid is CONDITIONAL on validity — its "
                 "denominator is n_valid ({} of {}), not n. Quoting it without "
                 "validity_rate ({:.2f}) beside it would be misleading."
                 .format(len(valid), n, len(valid) / n)),
    }
