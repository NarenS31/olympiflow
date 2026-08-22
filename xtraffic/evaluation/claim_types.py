"""Phase 3 — the claim vocabulary. Shared by the parser and the verifier.

WHY A SEPARATE MODULE
---------------------
The parser produces claims and the verifier consumes them. Putting the type
definitions in a third module keeps the contract explicit and lets the tests
import it without dragging in either implementation.

THE UNIT OF ANALYSIS, AND WHY IT CHANGES
----------------------------------------
The pre-Phase-3 metric (`evaluation/faithfulness.py`) had exactly one unit: an
entry in the LLM's own `cited_causes[]` array. That is a field the model was
INSTRUCTED to fill, so it is a self-declaration, not an extracted claim. The
free `reasoning` prose was never examined, and the explanation's `top_edges`,
`propagation_path`, and `propagation_lag_minutes` were rendered into the prompt
and never checked against what the model said about them.

An ATOMIC CLAIM here is a single checkable assertion — "node X is a cause",
"there is an edge from X to Y", "the lag is 5 minutes" — extracted from both the
structured fields and the prose, and typed so that each type gets the
verification it actually admits.

THE FIVE VERDICTS
-----------------
  SUPPORTED           the evidence entails the claim
  PARTIALLY_SUPPORTED part of a compound claim holds and part does not, or a
                      quantity is close but outside tolerance
  UNSUPPORTED         the evidence neither entails nor contradicts it; the claim
                      is about something the evidence is silent on
  CONTRADICTED        the evidence entails the NEGATION of the claim
  UNVERIFIABLE        no deterministic procedure can decide it from what we have

UNSUPPORTED vs CONTRADICTED is the distinction the old metric could not draw.
It scored "cited a node outside top-k" and "asserted an edge that runs the other
way" identically, as `1 - precision`. They are different failures: the first is
overreach, the second is a false statement about a structure the model was
shown. A system that reduces one but not the other has not been measured by a
metric that merges them.

UNVERIFIABLE is not a failure and must never be silently counted as one. A
recommendation ("retime the signals at 3rd and Main") is an ACTION PROPOSAL, not
a factual assertion about the graph; scoring it as a hallucination would punish
the model for doing the task. It is counted, reported, and excluded from
precision.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# --- claim types -----------------------------------------------------------
NODE_EXISTENCE = "node_existence"
EDGE_EXISTENCE = "edge_existence"
EDGE_DIRECTION = "edge_direction"
PATH_EXISTENCE = "path_existence"
FEATURE_ATTRIBUTION = "feature_attribution"
TEMPORAL = "temporal"
CONFIDENCE = "confidence"
PREDICTION = "prediction"
COUNTERFACTUAL = "counterfactual"
DOMAIN_TERMINOLOGY = "domain_terminology"

CLAIM_TYPES: List[str] = [
    NODE_EXISTENCE, EDGE_EXISTENCE, EDGE_DIRECTION, PATH_EXISTENCE,
    FEATURE_ATTRIBUTION, TEMPORAL, CONFIDENCE, PREDICTION, COUNTERFACTUAL,
    DOMAIN_TERMINOLOGY,
]

# --- verdicts --------------------------------------------------------------
SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNVERIFIABLE = "UNVERIFIABLE"

VERDICTS: List[str] = [SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED,
                       CONTRADICTED, UNVERIFIABLE]

# Verdicts that count AGAINST the model when computing claim precision.
# PARTIALLY_SUPPORTED is deliberately excluded from both numerator and
# denominator of the strict rate and reported on its own — folding a half-right
# claim into either bucket would hide exactly the cases worth reading.
ADVERSE_VERDICTS = {UNSUPPORTED, CONTRADICTED}

# --- reference sets (DECISION 1 in docs/EXPERIMENT_PLAN.md §1) --------------
# ALIGNMENT    : score against what the LLM was SHOWN (the explainer's output).
#                This is the pre-Phase-3 metric's reference. It measures
#                transcription fidelity, NOT correctness — Phase 12 found two
#                near-disjoint explanations (Jaccard 0.022) both scoring 0.982
#                precision, which is only possible because neither was checked
#                against anything true.
# GROUND_TRUTH : score against the generator's true causal structure. Available
#                on synthetic graphs only (Phase 2).
ALIGNMENT = "alignment"
GROUND_TRUTH = "ground_truth"


class Claim:
    """One atomic, checkable assertion extracted from an explanation.

    Attributes:
        claim_type: one of CLAIM_TYPES.
        text:       the source span, verbatim. Kept so every verdict can be
                    traced back to what the model actually wrote — a verifier
                    whose decisions cannot be inspected is not auditable.
        source:     "cited_causes" | "reasoning" | "recommendations" — WHERE the
                    claim came from. Essential: the old metric only ever saw
                    cited_causes, so per-source rates are what show whether prose
                    claims behave differently from declared ones.
        payload:    type-specific fields the verifier needs (node names, numbers,
                    endpoints, ...).
        span:       (start, end) character offsets into `text`'s source string.
    """

    __slots__ = ("claim_type", "text", "source", "payload", "span")

    def __init__(self, claim_type: str, text: str, source: str,
                 payload: Optional[Dict[str, Any]] = None,
                 span: Optional[Any] = None):
        if claim_type not in CLAIM_TYPES:
            raise ValueError("unknown claim_type: {!r}".format(claim_type))
        self.claim_type = claim_type
        self.text = text
        self.source = source
        self.payload = payload or {}
        self.span = span

    def as_dict(self) -> Dict[str, Any]:
        return {"claim_type": self.claim_type, "text": self.text,
                "source": self.source, "payload": self.payload,
                "span": list(self.span) if self.span else None}

    def __repr__(self) -> str:                     # pragma: no cover
        return "Claim({}, {!r}, src={})".format(
            self.claim_type, self.text[:48], self.source)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Claim):
            return NotImplemented
        return (self.claim_type == other.claim_type and self.text == other.text
                and self.source == other.source and self.payload == other.payload)

    def __hash__(self) -> int:
        return hash((self.claim_type, self.text, self.source))


class Verdict:
    """The outcome of verifying one claim, with the reason.

    `reason` is mandatory and is not decoration. A verifier that returns a label
    without a justification cannot be audited, cannot be debugged, and cannot be
    compared against a human annotator — all three of which Phase 4 requires.
    """

    __slots__ = ("claim", "verdict", "reason", "reference", "evidence")

    def __init__(self, claim: Claim, verdict: str, reason: str,
                 reference: str = ALIGNMENT,
                 evidence: Optional[Dict[str, Any]] = None):
        if verdict not in VERDICTS:
            raise ValueError("unknown verdict: {!r}".format(verdict))
        self.claim = claim
        self.verdict = verdict
        self.reason = reason
        self.reference = reference      # ALIGNMENT | GROUND_TRUTH
        self.evidence = evidence or {}  # what the check actually looked at

    @property
    def is_adverse(self) -> bool:
        return self.verdict in ADVERSE_VERDICTS

    def as_dict(self) -> Dict[str, Any]:
        d = self.claim.as_dict()
        d.update({"verdict": self.verdict, "reason": self.reason,
                  "reference": self.reference, "evidence": self.evidence})
        return d

    def __repr__(self) -> str:                     # pragma: no cover
        return "Verdict({} {} :: {})".format(
            self.claim.claim_type, self.verdict, self.reason[:60])


class ParseReport:
    """What the parser extracted AND what it could not.

    `unparsed` is the honesty mechanism, and it is the single most important
    field in this module.

    A verifier that silently drops sentences it cannot parse FLATTERS every
    condition it scores — and it flatters them unequally. Prose conditions
    (ablation condition B) contain more free text than structured ones
    (condition D), so an unparsed sentence is more likely in exactly the
    condition the hypothesis predicts should do worse. Dropping them would
    manufacture the predicted result out of a parser limitation.

    So: every sentence that looks assertive but yields no claim is recorded, the
    rate is reported per condition, and a DIFFERENCE in unparsed rate across
    conditions is itself a reportable confound.
    """

    __slots__ = ("claims", "unparsed", "n_sentences", "notes")

    def __init__(self, claims: List[Claim], unparsed: List[str],
                 n_sentences: int, notes: Optional[List[str]] = None):
        self.claims = claims
        self.unparsed = unparsed
        self.n_sentences = n_sentences
        self.notes = notes or []

    @property
    def unparsed_rate(self) -> float:
        """Fraction of assertive sentences yielding no claim. Undefined -> 0.0
        when there were no sentences at all (an empty or refused advisory)."""
        return (len(self.unparsed) / self.n_sentences) if self.n_sentences else 0.0

    def by_type(self) -> Dict[str, int]:
        out: Dict[str, int] = {t: 0 for t in CLAIM_TYPES}
        for c in self.claims:
            out[c.claim_type] += 1
        return out

    def by_source(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.claims:
            out[c.source] = out.get(c.source, 0) + 1
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"n_claims": len(self.claims), "n_sentences": self.n_sentences,
                "n_unparsed": len(self.unparsed),
                "unparsed_rate": self.unparsed_rate,
                "unparsed": self.unparsed, "by_type": self.by_type(),
                "by_source": self.by_source(), "notes": self.notes,
                "claims": [c.as_dict() for c in self.claims]}
