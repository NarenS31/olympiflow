"""Phase 3 — deterministic verification of typed claims against graph evidence.

NO LLM. NO HEURISTIC PLAUSIBILITY. Every verdict is a decidable function of the
explanation JSON, the adjacency matrix, and the node table. A claim is never
marked correct because it sounds reasonable.

THE DISTINCTION THE OLD METRIC COULD NOT DRAW
---------------------------------------------
`evaluation/faithfulness.py` computed `hallucination = 1 - precision`, where a
citation "hit" iff its resolved node set intersected the explainer's top-k. That
merges three genuinely different failures:

    cited a real node that was not selected      -> overreach
    cited a place that does not exist            -> fabrication
    asserted an edge that runs the other way     -> a false statement about
                                                    structure it was SHOWN

This module separates them as UNSUPPORTED / CONTRADICTED, because a mechanism
that reduces one but not the others has not been measured by a metric that
merges them.

TWO REFERENCE SETS (DECISION 1, docs/EXPERIMENT_PLAN.md §1)
-----------------------------------------------------------
  ALIGNMENT    — vs the explainer's output = what the LLM was SHOWN. This is the
                 committed metric's reference. It measures transcription
                 fidelity, NOT correctness: Phase 12 found GNNExplainer and SHAP
                 selecting near-disjoint node sets (Jaccard 0.022) while both
                 scored 0.982 precision, which is only possible because neither
                 was ever checked against anything true.
  GROUND_TRUTH — vs the generator's true causal structure. Synthetic only
                 (Phase 2). Pass `ground_truth=` to enable.

They are reported SEPARATELY and never averaged. An advisory can be perfectly
aligned to a wrong explanation; that is a meaningful, reportable state.

A NOTE ON REGION-LEVEL CREDIT
-----------------------------
A citation resolves to the SET of nodes in a region and counts as a hit if that
set intersects top-k. This generosity is inherited from the committed resolver
so the two metrics stay comparable — but it is exactly what produced the
Phase-11 PEMS-BAY artifact, where a control citing only the target's own region
scored F1 0.510 with zero causal content. `evidence.resolved_set_size` is
recorded on every node verdict so the generosity is auditable per claim, and
chance floors remain mandatory alongside any headline number.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .claim_types import (ALIGNMENT, CONFIDENCE, CONTRADICTED, COUNTERFACTUAL,
                          DOMAIN_TERMINOLOGY, EDGE_DIRECTION, EDGE_EXISTENCE,
                          FEATURE_ATTRIBUTION, GROUND_TRUTH, NODE_EXISTENCE,
                          PARTIALLY_SUPPORTED, PATH_EXISTENCE, PREDICTION,
                          SUPPORTED, TEMPORAL, UNSUPPORTED, UNVERIFIABLE,
                          Claim, Verdict)

# Matches faithfulness.NUMERIC_TOLERANCE so the two metrics agree on "close".
NUMERIC_TOLERANCE = 0.10

# Confidence-tier boundaries on explanation_confidence (mean Jaccard of top-k
# over K reruns). Mirrors the Phase-18 CORE/PERIPHERAL/NOISE thresholds so the
# project has one notion of "confident".
CONFIDENCE_HIGH = 0.80
CONFIDENCE_LOW = 0.20


def _within(value: float, ref: float, tol: float = NUMERIC_TOLERANCE) -> bool:
    """|value - ref| <= tol * |ref|, INCLUSIVE at the boundary.

    Inclusive on purpose: 22.0 * 1.10 = 24.2 must count as inside +/-10%, and a
    strict comparison would fail it on floating-point representation alone. A
    tiny epsilon absorbs that.
    """
    return abs(value - ref) <= tol * max(abs(ref), 1e-9) + 1e-9


class GraphClaimVerifier:
    """Verify typed claims against one explanation (+ optional ground truth)."""

    def __init__(self, explanation: Dict[str, Any],
                 node_meta: Optional[Dict[str, Any]] = None,
                 adjacency: Optional[Any] = None,
                 ground_truth: Optional[Dict[str, Any]] = None,
                 reference: str = ALIGNMENT):
        self.exp = explanation or {}
        self.node_meta = node_meta or {}
        self.adj = adjacency
        self.gt = ground_truth
        self.reference = reference

        pred = self.exp.get("prediction", {}) or {}
        self.target: Optional[int] = (int(pred["node_id"])
                                      if pred.get("node_id") is not None else None)

        self.topk: Set[int] = {int(n["node_id"])
                               for n in (self.exp.get("top_nodes") or [])
                               if n.get("node_id") is not None}

        self.edges: Set[Tuple[int, int]] = {
            (int(e["from_id"]), int(e["to_id"]))
            for e in (self.exp.get("top_edges") or [])
            if e.get("from_id") is not None and e.get("to_id") is not None}

        self.path: List[int] = [int(x) for x in
                                (self.exp.get("propagation_path") or [])]

        # Reference quantities for numeric checks, kept SEPARATED by kind.
        # The committed metric pooled every speed and lag into one list and
        # accepted a number matching ANY of them (faithfulness.py:558-563),
        # which with ~11 references over a plausible range accepts a large
        # fraction of arbitrary two-digit numbers. Keeping them apart makes the
        # check mean something.
        self.speeds: Dict[str, float] = {}
        if pred.get("current_speed_mph") is not None:
            self.speeds["target_current"] = float(pred["current_speed_mph"])
        if pred.get("predicted_speed_mph") is not None:
            self.speeds["target_predicted"] = float(pred["predicted_speed_mph"])
        self.node_speeds: Dict[int, float] = {
            int(n["node_id"]): float(n["current_speed_mph"])
            for n in (self.exp.get("top_nodes") or [])
            if n.get("current_speed_mph") is not None}

        self.lag: Optional[float] = (
            float(self.exp["propagation_lag_minutes"])
            if self.exp.get("propagation_lag_minutes") is not None else None)
        self.horizon: Optional[float] = (
            float(pred["horizon_minutes"])
            if pred.get("horizon_minutes") is not None else None)
        self.confidence: Optional[float] = (
            float(self.exp["explanation_confidence"])
            if self.exp.get("explanation_confidence") is not None else None)

        self.n_nodes: int = len(self.node_meta.get("sensor_ids") or []) or (
            int(self.adj.shape[0]) if self.adj is not None else 0)

    # -- graph helpers ------------------------------------------------------
    def _node_exists(self, nid: int) -> bool:
        return 0 <= nid < self.n_nodes if self.n_nodes else False

    def _adj_has(self, i: int, j: int) -> bool:
        """Is (i, j) a real edge? Undirected physical adjacency counts either
        orientation as EXISTING; DIRECTION is judged separately against the
        explanation's directed top_edges."""
        if self.adj is None:
            return False
        try:
            return bool(self.adj[i, j] > 0) or bool(self.adj[j, i] > 0)
        except Exception:
            return False

    # -- dispatch -----------------------------------------------------------
    def verify(self, claim: Claim) -> Verdict:
        fn = {
            NODE_EXISTENCE: self._node,
            EDGE_EXISTENCE: self._edge,
            EDGE_DIRECTION: self._direction,
            PATH_EXISTENCE: self._path,
            PREDICTION: self._numeric,
            TEMPORAL: self._numeric,
            CONFIDENCE: self._confidence,
            FEATURE_ATTRIBUTION: self._feature,
            COUNTERFACTUAL: self._counterfactual,
            DOMAIN_TERMINOLOGY: self._terminology,
        }.get(claim.claim_type)
        if fn is None:                             # pragma: no cover
            return Verdict(claim, UNVERIFIABLE,
                           "no verification procedure for this claim type",
                           self.reference)
        try:
            return fn(claim)
        except Exception as exc:                   # pragma: no cover
            # A verifier that crashes on malformed LLM output cannot score the
            # conditions most likely to produce malformed LLM output.
            return Verdict(claim, UNVERIFIABLE,
                           "verification raised {}: {}".format(
                               type(exc).__name__, exc), self.reference)

    # -- node ---------------------------------------------------------------
    def _node(self, claim: Claim) -> Verdict:
        ids: Set[int] = set(claim.payload.get("node_ids") or [])
        ev = {"resolved_node_ids": sorted(ids), "resolved_set_size": len(ids),
              "topk": sorted(self.topk)}

        # RECOMMENDATION SITES ARE CHECKED FIRST, and are never scored adverse.
        #
        # A recommendation names WHERE TO ACT. That is an action proposal, not a
        # factual assertion that a graph node exists at that address. Real
        # advisories site interventions at street intersections and interchanges
        # ("Sunset Blvd and Santa Monica Blvd", "I-5/SR-134/SR-2 interchange")
        # which are perfectly sensible places to retime a signal and which
        # simply are not METR-LA freeway sensors, so they resolve to nothing.
        #
        # An earlier draft ran the `if not ids` check first and scored both of
        # those CONTRADICTED. That was wrong, and worse, it was BIASED: it would
        # have inflated the unsupported rate for every condition that produces
        # more recommendations, which is a property of the prompt rather than of
        # the model's grounding. Caught by inspecting real advisories rather
        # than by a unit test — which is why §B of
        # scripts/analyze_committed_results.py prints per-claim detail.
        if claim.payload.get("is_recommendation_site"):
            if ids:
                return Verdict(
                    claim, SUPPORTED,
                    "recommendation site resolves to a real node; site choice "
                    "is an action proposal, not a causal claim",
                    self.reference, ev)
            return Verdict(
                claim, UNVERIFIABLE,
                "recommendation site does not resolve to a graph node (street "
                "addresses and interchanges are valid intervention sites but "
                "are not sensors); an action proposal is not a factual claim "
                "about the graph and is not scored for support",
                self.reference, ev)

        if not ids:
            return Verdict(
                claim, CONTRADICTED,
                "cited location resolves to no node in this graph; it names "
                "something that does not exist here",
                self.reference, ev)

        reference_set = self.topk
        if self.reference == GROUND_TRUTH and self.gt:
            reference_set = set(self.gt.get("causal_nodes") or [])
            ev["ground_truth_nodes"] = sorted(reference_set)

        # SELF-ATTRIBUTION, checked FIRST and on one path only.
        #
        # The Phase-13 rule: a citation is self-blame when it resolves to the
        # TARGET ALONE. It is NOT self-blame merely because the target appears
        # in the resolved set, because region-level credit means citing the
        # target's own region can legitimately hit a real top-k node in that
        # region — crediting the neighbour, not blaming the target.
        #
        # This was originally two checks, one inside the `if hit:` branch and
        # one after it. Mutation testing showed the inner one was unreachable in
        # practice: the target is excluded from its own top-k by construction,
        # so `hit` can never contain it and the outer check always fired first.
        # Rather than write a test for a malformed-explanation path that cannot
        # arise, the two were collapsed into this single check.
        if self.target is not None and ids == {self.target}:
            return Verdict(
                claim, CONTRADICTED,
                "claim names the target node as its own cause "
                "(self-attribution); the target is excluded from its own "
                "explanation by construction",
                self.reference, ev)

        hit = ids & reference_set
        if hit:
            return Verdict(claim, SUPPORTED,
                           "resolves to {} node(s), {} in the reference set"
                           .format(len(ids), len(hit)),
                           self.reference, dict(ev, hit_nodes=sorted(hit)))

        # NOTE — there is deliberately NO second self-attribution check here.
        #
        # An earlier draft also flagged self-attribution whenever the target was
        # merely AMONG the resolved nodes. That contradicts the Phase-13 rule:
        # self-blame requires resolving to the target ALONE. A region citation
        # like "Downtown LA" that resolves to {target, neighbour} and hits
        # nothing in top-k is a REGION MISS, not self-blame, and calling it
        # self-attribution would inflate that failure category with ordinary
        # misses — exactly the over-attribution Phase 13 warned about.
        #
        # Found by mutation testing: disabling the check above left the suite
        # green because this redundant branch produced the same verdict for the
        # wrong reason. Removing it makes the check above load-bearing and the
        # semantics match the documented rule.
        if any(self._node_exists(i) for i in ids):
            return Verdict(
                claim, UNSUPPORTED,
                "resolves to real node(s) {} that the explanation did not "
                "select; the evidence is silent about them, it does not deny "
                "them".format(sorted(ids)[:5]),
                self.reference, ev)

        return Verdict(claim, CONTRADICTED,
                       "resolves to node id(s) outside this graph", self.reference, ev)

    # -- edge ---------------------------------------------------------------
    def _endpoints(self, claim: Claim) -> List[Tuple[int, int]]:
        """Candidate (src, dst) pairs. Region-level citations resolve to sets,
        so a single sentence can imply several node pairs; the verdict takes the
        most favourable reading, which is the same generosity the committed
        resolver applies to node citations."""
        if claim.payload.get("from_node") is not None:
            return [(int(claim.payload["from_node"]),
                     int(claim.payload["to_node"]))]
        src = claim.payload.get("from_candidates") or []
        dst = claim.payload.get("to_candidates") or []
        return [(int(a), int(b)) for a in src for b in dst if a != b]

    def _edge(self, claim: Claim) -> Verdict:
        pairs = self._endpoints(claim)
        ev = {"candidate_pairs": pairs[:20], "n_pairs": len(pairs),
              "explanation_edges": sorted(self.edges)}
        if not pairs:
            return Verdict(claim, UNVERIFIABLE,
                           "could not resolve both endpoints to nodes",
                           self.reference, ev)

        if self.reference == GROUND_TRUTH and self.gt:
            gt_edges = {(int(a), int(b))
                        for a, b in (self.gt.get("causal_edges") or [])}
            if any(p in gt_edges for p in pairs):
                return Verdict(claim, SUPPORTED,
                               "edge is in the ground-truth causal structure",
                               self.reference, ev)

        if any(p in self.edges for p in pairs):
            return Verdict(claim, SUPPORTED,
                           "edge appears in the explanation's top_edges",
                           self.reference, ev)

        # Reversed relative to a selected edge: existence holds, direction does
        # not. EDGE_EXISTENCE is about the link, so this is SUPPORTED here and
        # CONTRADICTED by _direction — which is exactly why the two are separate
        # claim types.
        if any((b, a) in self.edges for a, b in pairs):
            return Verdict(claim, SUPPORTED,
                           "the link is in top_edges (orientation is judged "
                           "separately as an edge_direction claim)",
                           self.reference, ev)

        if self.adj is not None and any(self._adj_has(a, b) for a, b in pairs):
            return Verdict(
                claim, UNSUPPORTED,
                "edge exists in the physical adjacency but was NOT selected by "
                "the explainer; true of the graph, absent from the evidence",
                self.reference, ev)

        if self.adj is not None:
            return Verdict(claim, CONTRADICTED,
                           "no such edge exists in the graph's adjacency",
                           self.reference, ev)
        return Verdict(claim, UNSUPPORTED,
                       "edge is not in top_edges and no adjacency was supplied "
                       "to check the graph itself", self.reference, ev)

    def _direction(self, claim: Claim) -> Verdict:
        pairs = self._endpoints(claim)
        ev = {"candidate_pairs": pairs[:20],
              "explanation_edges": sorted(self.edges),
              "propagation_path": self.path}
        if not pairs:
            return Verdict(claim, UNVERIFIABLE,
                           "could not resolve both endpoints to nodes",
                           self.reference, ev)

        if any(p in self.edges for p in pairs):
            return Verdict(claim, SUPPORTED,
                           "direction matches a directed edge in top_edges",
                           self.reference, ev)

        # THE CASE THE COMMITTED METRIC CANNOT SEE.
        reversed_hits = [(a, b) for a, b in pairs if (b, a) in self.edges]
        if reversed_hits:
            return Verdict(
                claim, CONTRADICTED,
                "stated direction is the REVERSE of the explanation's edge: "
                "evidence has {} but the claim asserts {}".format(
                    (reversed_hits[0][1], reversed_hits[0][0]), reversed_hits[0]),
                self.reference, dict(ev, reversed_pairs=reversed_hits))

        # Consistency with the propagation path's ordering.
        if len(self.path) >= 2:
            order = {n: i for i, n in enumerate(self.path)}
            for a, b in pairs:
                if a in order and b in order:
                    if order[a] < order[b]:
                        return Verdict(
                            claim, SUPPORTED,
                            "direction agrees with the propagation path order",
                            self.reference, ev)
                    return Verdict(
                        claim, CONTRADICTED,
                        "direction is the reverse of the propagation path "
                        "order ({} comes after {} in the path)".format(a, b),
                        self.reference, ev)

        if self.adj is not None and any(self._adj_has(a, b) for a, b in pairs):
            return Verdict(claim, UNSUPPORTED,
                           "the link exists in the graph but the explanation "
                           "assigns it no direction", self.reference, ev)

        return Verdict(claim, UNSUPPORTED,
                       "no directed evidence for this pair", self.reference, ev)

    # -- path ---------------------------------------------------------------
    def _path(self, claim: Claim) -> Verdict:
        path = [int(x) for x in (claim.payload.get("path") or [])]
        ev = {"claimed_path": path, "explanation_path": self.path}
        if len(path) < 2:
            return Verdict(claim, UNVERIFIABLE,
                           "a path claim needs at least two nodes",
                           self.reference, ev)

        if self.reference == GROUND_TRUTH and self.gt:
            gt_path = [int(x) for x in (self.gt.get("causal_path") or [])]
            if path == gt_path:
                return Verdict(claim, SUPPORTED,
                               "path matches the ground-truth causal path",
                               self.reference, dict(ev, ground_truth_path=gt_path))

        if path == self.path:
            return Verdict(claim, SUPPORTED,
                           "path matches the explanation's propagation path",
                           self.reference, ev)

        if self.path and _is_contiguous_subsequence(path, self.path):
            return Verdict(
                claim, PARTIALLY_SUPPORTED,
                "path is a contiguous segment of the explanation's propagation "
                "path but not the whole path", self.reference, ev)

        traversable = (self.adj is not None and all(
            self._adj_has(path[i], path[i + 1]) for i in range(len(path) - 1)))
        if traversable:
            return Verdict(
                claim, UNSUPPORTED,
                "every hop exists in the graph, but this is not the "
                "explainer's propagation path", self.reference, ev)

        if self.adj is not None:
            return Verdict(claim, CONTRADICTED,
                           "path is not traversable: at least one hop is not an "
                           "edge in the graph", self.reference, ev)
        return Verdict(claim, UNSUPPORTED,
                       "path does not match the explanation and no adjacency "
                       "was supplied", self.reference, ev)

    # -- numbers ------------------------------------------------------------
    def _numeric(self, claim: Claim) -> Verdict:
        val = claim.payload.get("value")
        qty = claim.payload.get("quantity")
        if val is None:
            return Verdict(claim, UNVERIFIABLE, "no numeric value extracted",
                           self.reference)
        val = float(val)

        refs: Dict[str, float] = {}
        if qty in ("speed", "voltage_pu"):
            refs.update(self.speeds)
            refs.update({"top_node_{}".format(k): v
                         for k, v in self.node_speeds.items()})
        elif qty == "lag_minutes":
            if self.lag is not None:
                refs["propagation_lag"] = self.lag
        elif qty == "horizon_minutes":
            if self.horizon is not None:
                refs["horizon"] = self.horizon
        ev = {"value": val, "quantity": qty, "references": refs}

        if not refs:
            return Verdict(claim, UNVERIFIABLE,
                           "explanation states no {} to compare against"
                           .format(qty), self.reference, ev)

        exact = [k for k, r in refs.items() if _within(val, r)]
        if exact:
            return Verdict(claim, SUPPORTED,
                           "matches {} ({}) within +/-{:.0%}".format(
                               exact[0], refs[exact[0]], NUMERIC_TOLERANCE),
                           self.reference, dict(ev, matched=exact))

        # Near-miss at double tolerance is a DIFFERENT error from an invented
        # number, and merging them loses the distinction.
        near = [k for k, r in refs.items()
                if _within(val, r, NUMERIC_TOLERANCE * 2)]
        if near:
            return Verdict(claim, PARTIALLY_SUPPORTED,
                           "close to {} ({}) but outside the +/-{:.0%} "
                           "tolerance".format(near[0], refs[near[0]],
                                              NUMERIC_TOLERANCE),
                           self.reference, dict(ev, near=near))

        return Verdict(claim, UNSUPPORTED,
                       "no {} in the explanation is within tolerance of {}"
                       .format(qty, val), self.reference, ev)

    # -- confidence ---------------------------------------------------------
    def _confidence(self, claim: Claim) -> Verdict:
        tier = claim.payload.get("asserted_tier")
        ev = {"asserted_tier": tier, "explanation_confidence": self.confidence}
        if self.confidence is None:
            return Verdict(claim, UNVERIFIABLE,
                           "explanation carries no confidence score",
                           self.reference, ev)

        actual = ("high" if self.confidence >= CONFIDENCE_HIGH
                  else "low" if self.confidence < CONFIDENCE_LOW else "medium")
        ev["actual_tier"] = actual
        if tier == actual:
            return Verdict(claim, SUPPORTED,
                           "asserted {} matches explanation_confidence {:.2f}"
                           .format(tier, self.confidence), self.reference, ev)
        if {tier, actual} == {"high", "low"}:
            return Verdict(
                claim, CONTRADICTED,
                "asserts {} confidence but explanation_confidence is {:.2f} "
                "({})".format(tier, self.confidence, actual), self.reference, ev)
        return Verdict(claim, PARTIALLY_SUPPORTED,
                       "asserted {} vs actual {} (confidence {:.2f}) — adjacent "
                       "tiers".format(tier, actual, self.confidence),
                       self.reference, ev)

    # -- feature ------------------------------------------------------------
    def _feature(self, claim: Claim) -> Verdict:
        feat = str(claim.payload.get("feature") or "").lower()
        present = [m.lower() for m in (self.node_meta.get("modalities") or [])]
        fi = self.exp.get("feature_importance") or {}
        ev = {"feature": feat, "available_modalities": present,
              "explanation_feature_importance": fi}

        if fi:
            for k, v in fi.items():
                if feat in str(k).lower():
                    return Verdict(claim, SUPPORTED,
                                   "feature appears in the explanation's "
                                   "feature_importance", self.reference,
                                   dict(ev, importance=v))
            return Verdict(claim, UNSUPPORTED,
                           "feature is not among those the explanation "
                           "attributed", self.reference, ev)

        if present and not any(feat in m or m in feat for m in present):
            return Verdict(claim, CONTRADICTED,
                           "attributes the prediction to a modality this "
                           "dataset does not contain", self.reference, ev)

        # No feature attribution was computed, so there is nothing to check
        # against. UNVERIFIABLE — not a failure of the model.
        return Verdict(claim, UNVERIFIABLE,
                       "explanation carries no feature attribution to check "
                       "against", self.reference, ev)

    # -- counterfactual -----------------------------------------------------
    def _counterfactual(self, claim: Claim) -> Verdict:
        """Deciding a counterfactual requires RERUNNING the model, which this
        class deliberately does not do — it must stay a pure function of stored
        evidence so it is cheap, deterministic, and safe to run anywhere.
        `counterfactual_verifier.py` resolves these by actually rerunning."""
        return Verdict(claim, UNVERIFIABLE,
                       "counterfactual claims require rerunning the predictor; "
                       "see evaluation/counterfactual_verifier.py",
                       self.reference, {"needs_model_rerun": True})

    # -- terminology --------------------------------------------------------
    def _terminology(self, claim: Claim) -> Verdict:
        term = claim.payload.get("term")
        ev = {"term": term,
              "is_action_proposal": bool(claim.payload.get("is_action_proposal"))}
        # An action proposal is not a factual assertion about the graph.
        # Scoring it adversely would penalise the model for doing the task.
        return Verdict(claim, UNVERIFIABLE,
                       "domain terminology / action proposal: not a factual "
                       "claim about the graph, so not scored for support",
                       self.reference, ev)


def _is_contiguous_subsequence(short: Sequence[int],
                               long: Sequence[int]) -> bool:
    n, m = len(short), len(long)
    if n == 0 or n > m:
        return False
    return any(list(long[i:i + n]) == list(short) for i in range(m - n + 1))


def verify_all(claims: Sequence[Claim], verifier: "GraphClaimVerifier"
               ) -> List[Verdict]:
    return [verifier.verify(c) for c in claims]
