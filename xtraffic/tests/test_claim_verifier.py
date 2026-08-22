"""Phase 3 tests — WRITTEN BEFORE THE IMPLEMENTATION.

    python -m unittest xtraffic.tests.test_claim_verifier -v

Uses stdlib `unittest`, not pytest: pytest is not installed and the project
pins Python 3.9, so adding a test dependency to run tests is a worse trade than
using what ships with the interpreter.

WHY THESE PARTICULAR CASES
--------------------------
Each test below encodes a failure the pre-Phase-3 metric could not detect, or a
way this verifier could flatter the system if written carelessly. They are the
specification; the implementation exists to satisfy them.

The adversarial ones matter most:

  * `test_refusal_is_not_hallucination` — the old metric returns hallucination
    1.0 when `cited_causes` is empty (faithfulness.py:540), which makes the
    planned no-answer control unscoreable and conflates "declined" with
    "fabricated" in condition B.
  * `test_reversed_edge_is_contradicted_not_unsupported` — the old metric scores
    both as `1 - precision`. They are different failures.
  * `test_unparsed_sentences_are_counted` — a parser that silently drops what it
    cannot handle manufactures the hypothesis's predicted result out of its own
    limitation, and does so MORE in prose conditions than structured ones.
  * `test_recommendation_is_unverifiable_not_unsupported` — an action proposal
    is not a factual claim about the graph. Scoring it adversely would punish
    the model for performing the task it was given.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict

from ..evaluation import claim_types as CT
from ..evaluation.claim_parser import parse_advisory
from ..evaluation.graph_claim_verifier import GraphClaimVerifier


# ---------------------------------------------------------------------------
# Fixtures — a tiny, fully-known 6-node graph.
# ---------------------------------------------------------------------------
def make_node_meta() -> Dict[str, Any]:
    """Six METR-LA-style sensors across three regions."""
    return {
        "dataset": "metr_la",
        "sensor_ids": [770001, 770002, 770003, 770004, 770005, 770006],
        "latlon": [
            [34.045, -118.240],   # 0 Downtown LA
            [34.048, -118.244],   # 1 Downtown LA
            [34.152, -118.255],   # 2 Glendale / Burbank
            [34.158, -118.260],   # 3 Glendale / Burbank
            [34.190, -118.450],   # 4 San Fernando Valley
            [34.195, -118.460],   # 5 San Fernando Valley
        ],
        "unit": "sensor",
    }


def make_explanation() -> Dict[str, Any]:
    """Target node 0; top-k = {2, 3, 4}; a directed edge 2->0; path 4->2->0.

    Node 5 EXISTS in the graph but is NOT in top-k — that distinction is what
    separates UNSUPPORTED (real node, not selected) from CONTRADICTED (a
    statement the evidence negates).
    """
    return {
        "meta": {"city": "metr_la", "timestamp": "test#1 @ 17:05",
                 "model_checkpoint": "test.pt"},
        "prediction": {"node_id": 0, "node_name": "Downtown LA (sensor 770001)",
                       "predicted_speed_mph": 22.0, "horizon_minutes": 30,
                       "current_speed_mph": 31.0},
        "top_nodes": [
            {"node_id": 2, "node_name": "Glendale / Burbank (sensor 770003)",
             "importance": 0.91, "current_speed_mph": 18.0},
            {"node_id": 3, "node_name": "Glendale / Burbank (sensor 770004)",
             "importance": 0.77, "current_speed_mph": 24.0},
            {"node_id": 4, "node_name": "San Fernando Valley (sensor 770005)",
             "importance": 0.55, "current_speed_mph": 40.0},
        ],
        # Edge 3->1 is in top_edges but NEITHER endpoint is on the propagation
        # path [4,2,0]. That is deliberate: without it, every reversal test can
        # be satisfied by the path-ordering branch instead of the top_edges
        # reversal branch, and mutation testing showed exactly that — sabotaging
        # reversal detection left the whole suite green.
        "top_edges": [
            {"from_id": 2, "to_id": 0, "importance": 0.88},
            {"from_id": 4, "to_id": 2, "importance": 0.61},
            {"from_id": 3, "to_id": 1, "importance": 0.42},
        ],
        "propagation_path": [4, 2, 0],
        "propagation_lag_minutes": 10.0,
        "explanation_confidence": 0.82,
    }


def make_adjacency():
    """Physical adjacency. Contains edge 3->1, which is REAL but NOT in the
    explanation's top_edges — the case that separates "true of the graph" from
    "selected by the explainer"."""
    import numpy as np
    a = np.zeros((6, 6), dtype=float)
    for (i, j) in [(2, 0), (4, 2), (3, 1), (5, 4), (1, 0)]:
        a[i, j] = 1.0
    return a


def adv(reasoning: str = "", causes=None, recs=None) -> Dict[str, Any]:
    return {"reasoning": reasoning, "cited_causes": causes or [],
            "recommendations": recs or []}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.exp = make_explanation()
        self.meta = make_node_meta()
        self.adj = make_adjacency()
        self.v = GraphClaimVerifier(self.exp, self.meta, adjacency=self.adj)

    def verify_advisory(self, advisory: Dict[str, Any]):
        rep = parse_advisory(advisory, self.meta)
        return rep, [self.v.verify(c) for c in rep.claims]

    def only(self, verdicts, claim_type):
        return [v for v in verdicts if v.claim.claim_type == claim_type]


# ---------------------------------------------------------------------------
# Refusal and empty output — DECISION 2
# ---------------------------------------------------------------------------
class TestRefusal(Base):

    def test_empty_advisory_yields_no_claims(self):
        """An empty advisory makes NO claims. It is not a hallucination; there
        is simply nothing to verify. The old metric returned hallucination 1.0
        here (faithfulness.py:540)."""
        rep, verdicts = self.verify_advisory(adv())
        self.assertEqual(rep.claims, [])
        self.assertEqual(verdicts, [])
        self.assertEqual(rep.unparsed_rate, 0.0)

    def test_refusal_is_not_hallucination(self):
        """An explicit refusal is detected as a refusal and yields no adverse
        claims. Without this, ablation condition N is unscoreable."""
        a = adv(reasoning="I cannot determine the cause from the evidence "
                          "provided. There is insufficient information.")
        rep, verdicts = self.verify_advisory(a)
        from ..evaluation.claim_parser import is_refusal
        self.assertTrue(is_refusal(a))
        self.assertEqual([v for v in verdicts if v.is_adverse], [])

    def test_normal_advisory_is_not_a_refusal(self):
        from ..evaluation.claim_parser import is_refusal
        self.assertFalse(is_refusal(adv(
            reasoning="Congestion at Downtown LA is driven by Glendale / Burbank.",
            causes=[{"location": "Glendale / Burbank", "resolved_node_id": 2}])))


# ---------------------------------------------------------------------------
# Node claims — the only type the old metric handled
# ---------------------------------------------------------------------------
class TestNodeClaims(Base):

    def test_cited_topk_node_is_supported(self):
        _, v = self.verify_advisory(adv(
            causes=[{"location": "Glendale / Burbank (sensor 770003)",
                     "resolved_node_id": 2}]))
        n = self.only(v, CT.NODE_EXISTENCE)
        self.assertEqual(len(n), 1)
        self.assertEqual(n[0].verdict, CT.SUPPORTED)

    def test_real_node_outside_topk_is_unsupported_not_contradicted(self):
        """Node 5 exists in the graph but was not selected. The evidence is
        SILENT about it — it does not negate it. UNSUPPORTED, not CONTRADICTED."""
        _, v = self.verify_advisory(adv(
            causes=[{"location": "sensor 770006", "resolved_node_id": 5}]))
        n = self.only(v, CT.NODE_EXISTENCE)
        self.assertEqual(n[0].verdict, CT.UNSUPPORTED)

    def test_nonexistent_location_is_contradicted(self):
        """A location that resolves to NO node in this graph is a claim about
        something that does not exist. That is stronger than unsupported."""
        _, v = self.verify_advisory(adv(
            causes=[{"location": "Brooklyn Bridge", "resolved_node_id": None}]))
        n = self.only(v, CT.NODE_EXISTENCE)
        self.assertEqual(n[0].verdict, CT.CONTRADICTED)

    def test_self_attribution_is_flagged(self):
        """Citing the TARGET as its own cause — the Phase-13 failure mode."""
        _, v = self.verify_advisory(adv(
            causes=[{"location": "Downtown LA (sensor 770001)",
                     "resolved_node_id": 0}]))
        n = self.only(v, CT.NODE_EXISTENCE)
        self.assertNotEqual(n[0].verdict, CT.SUPPORTED)
        self.assertIn("target", n[0].reason.lower())

    def test_self_attribution_vs_region_credit_are_distinguished(self):
        """THE PHASE-13 SUBTLETY, isolated.

        Phase 13 found that self-attribution must require a cause that resolves
        to the target AND misses top-k — NOT merely "target is in the resolved
        set" — because region-level credit means citing the target's own region
        can legitimately hit a real top-k node in that same region. Crediting
        the neighbour is correct; blaming the target is not.

        This fixture puts node 1 (same Downtown LA region as target node 0) into
        top-k, so the two cases separate:
          "Downtown LA"        -> {0, 1}, hits top-k via 1  -> SUPPORTED
          "sensor 770001"      -> {0} exactly, the target    -> CONTRADICTED

        Added after mutation testing: disabling the precise `ids == {target}`
        branch left the suite green, because the base fixture's Downtown LA
        region contains no top-k node and so the coarser fallback caught every
        case. The distinguishing scenario was never tested.
        """
        exp = make_explanation()
        exp["top_nodes"].append(
            {"node_id": 1, "node_name": "Downtown LA (sensor 770002)",
             "importance": 0.40, "current_speed_mph": 27.0})
        ver = GraphClaimVerifier(exp, self.meta, adjacency=self.adj)

        region = ver.verify(CT.Claim(CT.NODE_EXISTENCE, "Downtown LA",
                                     "cited_causes", {"node_ids": [0, 1]}))
        self.assertEqual(region.verdict, CT.SUPPORTED,
                         "region citation hitting a real top-k node must be "
                         "credited, not scored as self-blame")

        exact = ver.verify(CT.Claim(CT.NODE_EXISTENCE, "sensor 770001",
                                    "cited_causes", {"node_ids": [0]}))
        self.assertEqual(exact.verdict, CT.CONTRADICTED)
        self.assertIn("self-attribution", exact.reason.lower())


# ---------------------------------------------------------------------------
# Edge and direction claims — INVISIBLE to the old metric
# ---------------------------------------------------------------------------
class TestEdgeClaims(Base):

    def test_correct_edge_direction_is_supported(self):
        _, v = self.verify_advisory(adv(
            reasoning="Congestion propagates from Glendale / Burbank to "
                      "Downtown LA."))
        e = self.only(v, CT.EDGE_EXISTENCE) + self.only(v, CT.EDGE_DIRECTION)
        self.assertTrue(e, "no edge claim extracted from a directional sentence")
        self.assertTrue(any(x.verdict == CT.SUPPORTED for x in e))

    def test_reversed_edge_is_contradicted_not_unsupported(self):
        """THE CASE THE OLD METRIC CANNOT SEE. The explanation contains 2->0.
        Asserting 0->2 is not merely unsupported — the evidence NEGATES it."""
        _, v = self.verify_advisory(adv(
            reasoning="Congestion propagates from Downtown LA to "
                      "Glendale / Burbank."))
        d = self.only(v, CT.EDGE_DIRECTION)
        self.assertTrue(d, "no direction claim extracted")
        self.assertEqual(d[0].verdict, CT.CONTRADICTED)
        self.assertIn("reverse", d[0].reason.lower())

    def test_reversal_detected_via_top_edges_alone(self):
        """ISOLATES the top_edges reversal branch.

        Nodes 3 and 1 are NOT on the propagation path [4,2,0], so the
        path-ordering fallback cannot fire and only reversal detection can
        return CONTRADICTED. Added after mutation testing showed that stubbing
        out reversal detection left the entire suite green — the original test
        was being satisfied by the path branch, so the reversal logic it claimed
        to cover was never executed.
        """
        v = self.v.verify(CT.Claim(
            CT.EDGE_DIRECTION, "from sensor 770002 to sensor 770004",
            "reasoning", {"from_node": 1, "to_node": 3}))
        self.assertEqual(v.verdict, CT.CONTRADICTED)
        self.assertIn("reverse", v.reason.lower())

    def test_real_graph_edge_not_in_explanation_is_unsupported(self):
        """Edge 5->4 exists in the adjacency but was NOT selected. Distinguishes
        "true of the graph" from "part of the evidence shown"."""
        v = self.v.verify(CT.Claim(
            CT.EDGE_EXISTENCE, "from sensor 770006 to sensor 770005",
            "reasoning", {"from_node": 5, "to_node": 4}))
        self.assertEqual(v.verdict, CT.UNSUPPORTED)
        self.assertIn("adjacency", v.reason.lower())

    def test_edge_between_unconnected_nodes_is_contradicted(self):
        v = self.v.verify(CT.Claim(
            CT.EDGE_EXISTENCE, "from sensor 770001 to sensor 770006",
            "reasoning", {"from_node": 0, "to_node": 5}))
        self.assertEqual(v.verdict, CT.CONTRADICTED)


# ---------------------------------------------------------------------------
# Path claims
# ---------------------------------------------------------------------------
class TestPathClaims(Base):

    def test_correct_path_is_supported(self):
        v = self.v.verify(CT.Claim(CT.PATH_EXISTENCE, "4 -> 2 -> 0",
                                   "reasoning", {"path": [4, 2, 0]}))
        self.assertEqual(v.verdict, CT.SUPPORTED)

    def test_path_subsequence_is_partially_supported(self):
        v = self.v.verify(CT.Claim(CT.PATH_EXISTENCE, "4 -> 2", "reasoning",
                                   {"path": [4, 2]}))
        self.assertEqual(v.verdict, CT.PARTIALLY_SUPPORTED)

    def test_path_valid_in_graph_but_not_explained_is_unsupported(self):
        """5->4 is a real edge; the path is traversable but was not the
        explainer's propagation path."""
        v = self.v.verify(CT.Claim(CT.PATH_EXISTENCE, "5 -> 4", "reasoning",
                                   {"path": [5, 4]}))
        self.assertEqual(v.verdict, CT.UNSUPPORTED)

    def test_untraversable_path_is_contradicted(self):
        v = self.v.verify(CT.Claim(CT.PATH_EXISTENCE, "0 -> 5", "reasoning",
                                   {"path": [0, 5]}))
        self.assertEqual(v.verdict, CT.CONTRADICTED)


# ---------------------------------------------------------------------------
# Numeric tolerance — the boundary must be exact
# ---------------------------------------------------------------------------
class TestNumericClaims(Base):

    def test_exact_speed_is_supported(self):
        v = self.v.verify(CT.Claim(CT.PREDICTION, "22.0 mph", "reasoning",
                                   {"value": 22.0, "quantity": "speed"}))
        self.assertEqual(v.verdict, CT.SUPPORTED)

    def test_speed_just_inside_tolerance_is_supported(self):
        """22.0 * 1.10 = 24.2 — exactly at the +/-10% boundary, inclusive."""
        v = self.v.verify(CT.Claim(CT.PREDICTION, "24.2 mph", "reasoning",
                                   {"value": 24.2, "quantity": "speed"}))
        self.assertEqual(v.verdict, CT.SUPPORTED)

    def test_speed_just_outside_tolerance_is_partially_supported(self):
        """Near-miss is PARTIALLY_SUPPORTED, not UNSUPPORTED: a number close to
        a real quantity is a different error from an invented one, and merging
        them loses the distinction."""
        v = self.v.verify(CT.Claim(CT.PREDICTION, "27.0 mph", "reasoning",
                                   {"value": 27.0, "quantity": "speed"}))
        self.assertEqual(v.verdict, CT.PARTIALLY_SUPPORTED)

    def test_wholly_invented_number_is_unsupported(self):
        v = self.v.verify(CT.Claim(CT.PREDICTION, "153 mph", "reasoning",
                                   {"value": 153.0, "quantity": "speed"}))
        self.assertEqual(v.verdict, CT.UNSUPPORTED)

    def test_correct_lag_is_supported(self):
        v = self.v.verify(CT.Claim(CT.TEMPORAL, "10 minutes", "reasoning",
                                   {"value": 10.0, "quantity": "lag_minutes"}))
        self.assertEqual(v.verdict, CT.SUPPORTED)

    def test_wrong_lag_is_unsupported(self):
        v = self.v.verify(CT.Claim(CT.TEMPORAL, "45 minutes", "reasoning",
                                   {"value": 45.0, "quantity": "lag_minutes"}))
        self.assertEqual(v.verdict, CT.UNSUPPORTED)

    def test_forecast_horizon_is_not_a_lag_error(self):
        """"within the next 30 minutes" is the FORECAST HORIZON, not a claimed
        propagation lag. The Phase-13 taxonomy hit this exact bug and had to add
        a rule for it."""
        rep, v = self.verify_advisory(adv(
            reasoning="Speeds will drop within the next 30 minutes."))
        for x in self.only(v, CT.TEMPORAL):
            self.assertNotEqual(x.verdict, CT.UNSUPPORTED, x.reason)


# ---------------------------------------------------------------------------
# Confidence claims
# ---------------------------------------------------------------------------
class TestConfidenceClaims(Base):

    def test_high_confidence_matching_score_is_supported(self):
        v = self.v.verify(CT.Claim(CT.CONFIDENCE, "high confidence", "reasoning",
                                   {"asserted_tier": "high"}))
        self.assertEqual(v.verdict, CT.SUPPORTED)   # explanation_confidence 0.82

    def test_high_confidence_against_low_score_is_contradicted(self):
        exp = make_explanation()
        exp["explanation_confidence"] = 0.15
        ver = GraphClaimVerifier(exp, make_node_meta(), adjacency=make_adjacency())
        v = ver.verify(CT.Claim(CT.CONFIDENCE, "high confidence", "reasoning",
                                {"asserted_tier": "high"}))
        self.assertEqual(v.verdict, CT.CONTRADICTED)


# ---------------------------------------------------------------------------
# Unverifiable — must not be scored adversely
# ---------------------------------------------------------------------------
class TestUnverifiable(Base):

    def test_recommendation_is_unverifiable_not_unsupported(self):
        """An action proposal is not a factual claim about the graph."""
        _, v = self.verify_advisory(adv(recs=[{
            "action": "Retime signals", "location": "Glendale / Burbank",
            "time_window_minutes": 15, "expected_effect": "reduce delay",
            "grounded_in": ["SR-134 corridor"]}]))
        for x in v:
            self.assertNotEqual(x.verdict, CT.UNSUPPORTED)

    def test_unresolvable_recommendation_site_is_unverifiable(self):
        """A street intersection is a valid place to act and is not a sensor.

        Real committed advisories site interventions at "Sunset Blvd and Santa
        Monica Blvd" and "I-5/SR-134/SR-2 interchange". Neither resolves to a
        METR-LA freeway sensor, and neither is a false claim about the graph.

        An earlier draft scored both CONTRADICTED, which was not merely wrong
        but BIASED: it would inflate the unsupported rate for every condition
        that elicits more recommendations — a property of the prompt, not of
        grounding. Found by inspecting real advisories, not by a unit test.
        """
        for site in ("Sunset Blvd and Santa Monica Blvd",
                     "I-5/SR-134/SR-2 interchange"):
            v = self.v.verify(CT.Claim(
                CT.NODE_EXISTENCE, site, "recommendations",
                {"node_ids": [], "is_recommendation_site": True}))
            self.assertEqual(v.verdict, CT.UNVERIFIABLE, site)
            self.assertFalse(v.is_adverse)

    def test_resolvable_recommendation_site_is_supported(self):
        v = self.v.verify(CT.Claim(
            CT.NODE_EXISTENCE, "Glendale / Burbank", "recommendations",
            {"node_ids": [2, 3], "is_recommendation_site": True}))
        self.assertEqual(v.verdict, CT.SUPPORTED)

    def test_a_CAUSE_that_does_not_resolve_is_still_contradicted(self):
        """The recommendation exemption must NOT leak to causal claims: an
        invented cause is still a false statement about the graph."""
        v = self.v.verify(CT.Claim(CT.NODE_EXISTENCE, "Brooklyn Bridge",
                                   "cited_causes", {"node_ids": []}))
        self.assertEqual(v.verdict, CT.CONTRADICTED)

    def test_domain_terminology_without_a_referent_is_unverifiable(self):
        v = self.v.verify(CT.Claim(CT.DOMAIN_TERMINOLOGY, "ramp metering",
                                   "reasoning", {"term": "ramp metering"}))
        self.assertEqual(v.verdict, CT.UNVERIFIABLE)


# ---------------------------------------------------------------------------
# The parser's honesty mechanism
# ---------------------------------------------------------------------------
class TestParserHonesty(Base):

    def test_unparsed_sentences_are_counted_not_dropped(self):
        """Assertive prose the parser cannot type MUST surface in `unparsed`.

        Silent dropping flatters prose conditions preferentially, which would
        manufacture the hypothesis's predicted result out of a parser
        limitation.

        The sentence is chosen to contain NO location, number, feature word,
        confidence term, or domain term, so it CANNOT yield a claim. The first
        version of this test used prose containing the word "load", which the
        feature regex matched — so a claim was produced, the weaker assertion
        `len(unparsed) + len(claims) > 0` held, and mutation testing showed the
        test stayed green with `unparsed.append` stubbed out entirely.
        """
        rep = parse_advisory(adv(
            reasoning="The situation merits continued attention."), self.meta)
        self.assertEqual(rep.claims, [], "fixture must yield no claims")
        self.assertEqual(len(rep.unparsed), 1,
                         "an assertive, untypeable sentence must be RECORDED")
        self.assertEqual(rep.n_sentences, 1)
        self.assertEqual(rep.unparsed_rate, 1.0)

    def test_every_claim_keeps_its_source_text(self):
        rep = parse_advisory(adv(
            reasoning="Glendale / Burbank is congested at 18 mph.",
            causes=[{"location": "Glendale / Burbank", "resolved_node_id": 2}]),
            self.meta)
        for c in rep.claims:
            self.assertTrue(c.text.strip(), "claim has empty source text")
            self.assertIn(c.source, ("cited_causes", "reasoning",
                                     "recommendations"))

    def test_per_source_breakdown_is_available(self):
        """The old metric only ever saw cited_causes. Per-source rates are what
        show whether prose claims behave differently from declared ones."""
        rep = parse_advisory(adv(
            reasoning="Congestion propagates from Glendale / Burbank to "
                      "Downtown LA at 18 mph.",
            causes=[{"location": "Glendale / Burbank", "resolved_node_id": 2}]),
            self.meta)
        self.assertIn("cited_causes", rep.by_source())
        self.assertGreater(sum(rep.by_source().values()), 1)


# ---------------------------------------------------------------------------
# Robustness — malformed input must never crash the verifier
# ---------------------------------------------------------------------------
class TestRobustness(Base):

    def test_unicode_and_punctuation_do_not_crash(self):
        rep = parse_advisory(adv(
            reasoning="Congestión — “Glendale / Burbank” → Downtown LA … 18 mph.",
            causes=[{"location": "Glendale／Burbank", "resolved_node_id": 2}]),
            self.meta)
        self.assertIsInstance(rep.claims, list)

    def test_missing_fields_do_not_crash(self):
        for bad in ({}, {"reasoning": None}, {"cited_causes": None},
                    {"cited_causes": ["a bare string"]},
                    {"reasoning": "", "cited_causes": [{"no_location": 1}]}):
            rep = parse_advisory(bad, self.meta)
            self.assertIsInstance(rep.claims, list)
            for c in rep.claims:
                self.assertIsInstance(self.v.verify(c), CT.Verdict)

    def test_explanation_without_optional_fields_does_not_crash(self):
        exp = {"prediction": {"node_id": 0, "predicted_speed_mph": 22.0},
               "top_nodes": []}
        ver = GraphClaimVerifier(exp, make_node_meta())
        v = ver.verify(CT.Claim(CT.NODE_EXISTENCE, "Glendale / Burbank",
                                "cited_causes", {"node_ids": [2]}))
        self.assertIsInstance(v, CT.Verdict)

    def test_every_verdict_carries_a_reason(self):
        """A verdict without a justification cannot be audited, debugged, or
        compared against a human annotator — all three of which Phase 4 needs."""
        rep, verdicts = self.verify_advisory(adv(
            reasoning="Congestion propagates from Downtown LA to Glendale / "
                      "Burbank at 99 mph over 45 minutes.",
            causes=[{"location": "Brooklyn Bridge", "resolved_node_id": None},
                    {"location": "Glendale / Burbank", "resolved_node_id": 2}]))
        self.assertTrue(verdicts)
        for v in verdicts:
            self.assertTrue(v.reason and v.reason.strip())
            self.assertIn(v.verdict, CT.VERDICTS)


if __name__ == "__main__":                         # pragma: no cover
    unittest.main(verbosity=2)
