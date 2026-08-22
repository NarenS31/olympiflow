"""Phase 3 — decompose an advisory into ATOMIC, TYPED, CHECKABLE claims.

WHAT CHANGED FROM THE PRE-PHASE-3 METRIC
-----------------------------------------
`evaluation/faithfulness.py` had one unit of analysis: an entry in the LLM's own
`cited_causes[]` array — a field the model was INSTRUCTED to fill. The free
`reasoning` prose was never read, so an advisory could invent an edge direction,
a propagation order, or a lag and score hallucination 0.000. The explanation's
`top_edges`, `propagation_path`, and `propagation_lag_minutes` are rendered into
the prompt and were never checked against what the model said about them.

This module reads BOTH the structured fields and the prose, and types each claim
so the verifier can apply the check that type actually admits.

NO LLM IS USED HERE, BY DESIGN
------------------------------
Using a language model to decide whether another language model hallucinated is
circular, and the project brief rules it out. Extraction is regex + the existing
node/region tables + a small set of relational patterns. That buys determinism
and auditability at the cost of coverage — and the cost is MEASURED, not hidden:
every assertive sentence that yields no claim lands in `ParseReport.unparsed`
and the rate is reported.

WHY THE UNPARSED RATE IS THE MOST IMPORTANT NUMBER HERE
-------------------------------------------------------
A parser that silently drops what it cannot handle flatters every condition it
scores — and it flatters them UNEQUALLY. Prose-heavy conditions (the ablation
matrix's condition B) contain more free text than structured ones (condition D),
so an unparsed sentence is more likely in exactly the condition the hypothesis
predicts should perform worse. Dropping them would manufacture the predicted
result out of a parser limitation. A DIFFERENCE in unparsed rate across
conditions is therefore itself a reportable confound, not a footnote.

REUSE
-----
Location text is resolved with `faithfulness.resolve_location` and `NodeTable` —
the SAME five-rung ladder the committed metric uses. That is deliberate: it
keeps one notion of "which node did the model mean", and it makes the Phase-3
gate (new verifier vs old metric on node claims) a meaningful comparison rather
than a comparison of two different resolvers.

Python 3.9 compatible.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .claim_types import (CONFIDENCE, COUNTERFACTUAL, DOMAIN_TERMINOLOGY,
                          EDGE_DIRECTION, EDGE_EXISTENCE, FEATURE_ATTRIBUTION,
                          NODE_EXISTENCE, PATH_EXISTENCE, PREDICTION, TEMPORAL,
                          Claim, ParseReport)

# ---------------------------------------------------------------------------
# Refusal detection (DECISION 2 in docs/EXPERIMENT_PLAN.md §1)
# ---------------------------------------------------------------------------
# Phrases that mark an ABSTENTION rather than an answer. Deliberately narrow:
# a broad list would let an ordinary hedge ("traffic may slow") count as a
# refusal, which would let a condition inflate its score by hedging. Every
# pattern here is an explicit statement of inability or insufficiency.
_REFUSAL_PATTERNS = [
    r"\bi (?:can ?not|cannot|can't|am unable to) (?:determine|identify|say|"
    r"provide|explain|assess|conclude)",
    r"\b(?:there is |i have )?insufficient (?:information|evidence|data|context)",
    r"\bnot enough (?:information|evidence|data|context)\b",
    r"\bno (?:sufficient )?evidence (?:is )?(?:provided|available|given)\b",
    r"\bunable to (?:determine|identify|explain|assess|conclude)\b",
    r"\bcannot be determined (?:from|with)\b",
    r"\bdecline to (?:answer|speculate)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(advisory: Dict[str, Any]) -> bool:
    """True when the advisory ABSTAINS rather than answers.

    Two ways to abstain, both counted:
      1. explicit refusal language in the reasoning, AND no cited causes;
      2. an empty advisory that also carries a structured `_error` (the Phase-4
         retry loop's structured-failure object).

    An empty advisory with no error and no refusal language is NOT a refusal —
    it is an empty answer, which the caller should count separately. Conflating
    the two would hide a parse failure inside an abstention rate.
    """
    if not isinstance(advisory, dict):
        return False
    causes = advisory.get("cited_causes") or []
    reasoning = advisory.get("reasoning") or ""
    if isinstance(reasoning, str) and _REFUSAL_RE.search(reasoning):
        return not causes
    if advisory.get("_error") and not causes and not reasoning:
        return True
    return False


def is_empty(advisory: Dict[str, Any]) -> bool:
    """No reasoning, no causes, no recommendations."""
    if not isinstance(advisory, dict):
        return True
    return not ((advisory.get("reasoning") or "").strip()
                or (advisory.get("cited_causes") or [])
                or (advisory.get("recommendations") or []))


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
# Typographic characters an LLM emits that would otherwise break naive matching.
# NFKC folds most of them (including the fullwidth solidus "／" -> "/"); these
# are the ones it does not.
_PUNCT_MAP = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
    "→": "->", "⇒": "->", "⟶": "->",
}


def normalize_text(text: str) -> str:
    """Fold typographic variants so pattern matching sees one spelling.

    NFKC is applied FIRST because it already handles fullwidth forms; the map
    then fixes quotes, dashes, ellipses, and arrows, which NFKC leaves alone.
    Arrows matter: "A -> B" is a directional claim and must survive.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _PUNCT_MAP.items():
        text = text.replace(k, v)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences. Crude but deterministic and inspectable —
    which is the right trade for a component whose failures must be countable."""
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(normalize_text(text)) if s.strip()]


# A sentence is ASSERTIVE (and therefore something we should be able to type) if
# it makes a statement. Imperatives and questions are not factual claims about
# the graph. This gates what lands in `unparsed`: counting a question as an
# unparsed claim would inflate the rate and make the honesty metric noise.
_NON_ASSERTIVE = re.compile(
    r"^\s*(?:please\b|consider\b|recommend\b|deploy\b|retime\b|implement\b|"
    r"monitor\b|increase\b|reduce\b|adjust\b|activate\b|issue\b|notify\b)",
    re.IGNORECASE)


def is_assertive(sentence: str) -> bool:
    if not sentence or sentence.rstrip().endswith("?"):
        return False
    return not _NON_ASSERTIVE.match(sentence)


# ---------------------------------------------------------------------------
# Relational patterns
# ---------------------------------------------------------------------------
# Directional propagation: "congestion spreads FROM X TO Y", "X -> Y".
# The `from`/`to` capture groups are location TEXT, resolved downstream.
_FROM_TO = re.compile(
    r"\bfrom\s+(?P<src>[^,;.]{2,60}?)\s+(?:to|toward|towards|into|onto)\s+"
    r"(?P<dst>[^,;.]{2,60}?)(?=[,;.]|$)", re.IGNORECASE)

_ARROW = re.compile(r"(?P<src>[^,;.\->]{2,60}?)\s*->\s*(?P<dst>[^,;.\->]{2,60})")

# "X affects/feeds/causes/drives/impacts Y" — directional without from/to.
_VERB_DIRECTIONAL = re.compile(
    r"\b(?P<src>[A-Z][^,;.]{2,60}?)\s+(?:affects|feeds|causes|drives|impacts|"
    r"influences|propagates to|spills into|backs up into)\s+"
    r"(?P<dst>[^,;.]{2,60}?)(?=[,;.]|$)")

# Numbers with units.
_SPEED = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*(?:mph|mi/h|miles per hour)\b",
                    re.IGNORECASE)
_PU = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*(?:pu|p\.u\.)\b", re.IGNORECASE)
_MINUTES = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*(?:minutes?|mins?\b)",
                      re.IGNORECASE)

# THE FORECAST-HORIZON GUARD. "within the next 30 minutes" is the prediction
# horizon, not a claimed propagation lag. The Phase-13 failure taxonomy hit this
# exact bug: its first cut mislabelled "...will reach the target in 30 minutes"
# as a wrong lag. Any minutes figure carrying horizon language is typed as
# PREDICTION (horizon), never TEMPORAL (lag).
_HORIZON_CONTEXT = re.compile(
    r"(?:within|over|in|during|next|coming|forecast|predicted|horizon|"
    r"expected)\s+(?:the\s+)?(?:next\s+)?$", re.IGNORECASE)

# Lead/lag language that DOES indicate a propagation claim.
_LAG_CONTEXT = re.compile(
    r"\b(?:lag|lead|earlier|ahead of|before|precede[sd]?|delay of|"
    r"upstream by|propagation)\b", re.IGNORECASE)

_CONFIDENCE_TERMS = [
    (r"\b(?:high(?:ly)?|strong(?:ly)?|very)\s+(?:confiden\w+|certain|reliable)",
     "high"),
    (r"\bconfiden\w+\s+(?:is\s+)?high\b", "high"),
    (r"\b(?:low|weak|limited|poor)\s+(?:confiden\w+|certainty|reliability)",
     "low"),
    (r"\b(?:uncertain|unclear|tentative|may|might|possibly|could be)\b", "low"),
    (r"\bmoderate(?:ly)?\s+(?:confiden\w+|certain)", "medium"),
]

# Domain vocabulary. Presence of a term is not itself an error — it becomes one
# only if it names infrastructure that does not exist in this domain, which is
# an adversarial condition (Phase 9), not a default check.
_DOMAIN_TERMS = [
    "ramp metering", "signal retiming", "signal timing", "transit surge",
    "reroute", "rerouting", "hov lane", "bus lane", "incident clearance",
    "variable message sign", "load shedding", "reactive power", "var support",
    "shunt capacitor", "tap changer", "n-1 contingency", "undervoltage",
]


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------
class _Resolver:
    """Thin adapter over the committed entity resolver.

    Reusing `faithfulness.NodeTable` + `resolve_location` keeps ONE notion of
    "which node did the model mean" across the old and new metrics, so the
    Phase-3 gate compares verifiers rather than resolvers.
    """

    def __init__(self, node_meta: Dict[str, Any]):
        from .faithfulness import NodeTable, resolve_location
        self._resolve = resolve_location
        try:
            self.table = NodeTable(node_meta)
        except Exception:
            self.table = None
        # Longest-first so "Glendale / Burbank" wins over "Glendale".
        self.names: List[Tuple[str, int]] = []
        if self.table is not None:
            for nid in range(self.table.n_nodes):
                try:
                    self.names.append((self.table.namer.name(nid).lower(), nid))
                except Exception:
                    continue
            self.names.sort(key=lambda kv: -len(kv[0]))

    def resolve(self, text: str) -> Tuple[Set[int], str]:
        """Location text -> (node_ids, method). Empty set == unresolved."""
        if self.table is None or not text:
            return set(), "unresolved"
        try:
            r = self._resolve(text, self.table)
            return set(r.node_ids), r.method
        except Exception:
            return set(), "unresolved"

    def mentions(self, sentence: str) -> List[Tuple[str, Set[int]]]:
        """Location-like spans in a sentence that resolve to nodes.

        Tries region labels and rendered node names first (longest first), then
        explicit "sensor NNNN" / "bus N" ids. Overlapping matches are suppressed
        so one span yields one mention.
        """
        if self.table is None:
            return []
        low = sentence.lower()
        found: List[Tuple[str, Set[int]]] = []
        claimed: List[Tuple[int, int]] = []

        def overlaps(a: int, b: int) -> bool:
            return any(not (b <= s or a >= e) for s, e in claimed)

        candidates: List[str] = sorted(self.table.region_to_nodes.keys(),
                                       key=len, reverse=True)
        for cand in candidates:
            if len(cand) < 3:
                continue
            i = low.find(cand)
            if i >= 0 and not overlaps(i, i + len(cand)):
                ids, _ = self.resolve(cand)
                if ids:
                    found.append((cand, ids))
                    claimed.append((i, i + len(cand)))

        unit = getattr(self.table, "unit", "sensor")
        for m in re.finditer(
                r"\b(?:{}|sensor|segment|bus|node)s?\s*#?\s*(\d{{1,7}})\b"
                .format(re.escape(unit)), low):
            if overlaps(m.start(), m.end()):
                continue
            ids, _ = self.resolve(m.group(0))
            if ids:
                found.append((m.group(0), ids))
                claimed.append((m.start(), m.end()))
        return found


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------
def parse_advisory(advisory: Any, node_meta: Dict[str, Any],
                   explanation: Optional[Dict[str, Any]] = None) -> ParseReport:
    """Decompose one advisory into atomic typed claims.

    Args:
        advisory:    the advisory dict ({reasoning, cited_causes,
                     recommendations}). Malformed input is tolerated — a
                     verifier that crashes on bad LLM output cannot score the
                     conditions most likely to produce bad LLM output.
        node_meta:   the dataset's node metadata (drives resolution).
        explanation: optional; lets the parser tell a restated horizon from a
                     claimed lag using the explanation's own horizon value.

    Returns:
        ParseReport with claims, unparsed sentences, and counts.
    """
    claims: List[Claim] = []
    unparsed: List[str] = []
    notes: List[str] = []

    if not isinstance(advisory, dict):
        return ParseReport([], [], 0, ["advisory was not a dict"])

    res = _Resolver(node_meta)
    if res.table is None:
        notes.append("node table unavailable; location claims cannot be resolved")

    horizon = None
    if explanation:
        try:
            horizon = float(explanation.get("prediction", {})
                            .get("horizon_minutes"))
        except Exception:
            horizon = None

    # --- 1. cited_causes -> node-existence claims --------------------------
    causes = advisory.get("cited_causes") or []
    if isinstance(causes, list):
        for c in causes:
            loc = (c.get("location") if isinstance(c, dict) else str(c)) or ""
            if not str(loc).strip():
                continue
            ids, method = res.resolve(str(loc))
            claims.append(Claim(
                NODE_EXISTENCE, str(loc), "cited_causes",
                {"node_ids": sorted(ids), "resolution_method": method,
                 "llm_self_reported_id": (c.get("resolved_node_id")
                                          if isinstance(c, dict) else None)}))

    # --- 2. reasoning prose -------------------------------------------------
    reasoning = advisory.get("reasoning") or ""
    sentences = split_sentences(reasoning) if isinstance(reasoning, str) else []
    n_assertive = 0

    for sent in sentences:
        if not is_assertive(sent):
            continue
        n_assertive += 1
        before = len(claims)
        claims.extend(_parse_sentence(sent, res, horizon))
        if len(claims) == before:
            # Assertive, but nothing typed. RECORDED, never dropped.
            unparsed.append(sent)

    # --- 3. recommendations -------------------------------------------------
    recs = advisory.get("recommendations") or []
    if isinstance(recs, list):
        for r in recs:
            if not isinstance(r, dict):
                continue
            # An ACTION is a proposal, not a factual assertion about the graph.
            # Its LOCATION, however, is checkable: recommending an intervention
            # at a place that does not exist is a real error.
            loc = str(r.get("location") or "").strip()
            if loc:
                ids, method = res.resolve(loc)
                claims.append(Claim(
                    NODE_EXISTENCE, loc, "recommendations",
                    {"node_ids": sorted(ids), "resolution_method": method,
                     "is_recommendation_site": True}))
            act = str(r.get("action") or "").strip()
            if act:
                term = next((t for t in _DOMAIN_TERMS if t in act.lower()), None)
                claims.append(Claim(
                    DOMAIN_TERMINOLOGY, act, "recommendations",
                    {"term": term, "is_action_proposal": True}))

    return ParseReport(claims, unparsed, n_assertive, notes)


def _parse_sentence(sent: str, res: _Resolver,
                    horizon: Optional[float]) -> List[Claim]:
    """Extract every typed claim from one assertive sentence."""
    out: List[Claim] = []
    low = sent.lower()

    # -- directional relations ---------------------------------------------
    for pat in (_FROM_TO, _ARROW, _VERB_DIRECTIONAL):
        for m in pat.finditer(sent):
            src_txt, dst_txt = m.group("src").strip(), m.group("dst").strip()
            src_ids, _ = res.resolve(src_txt)
            dst_ids, _ = res.resolve(dst_txt)
            if not (src_ids and dst_ids):
                continue
            payload = {"from_candidates": sorted(src_ids),
                       "to_candidates": sorted(dst_ids),
                       "from_text": src_txt, "to_text": dst_txt}
            out.append(Claim(EDGE_EXISTENCE, m.group(0).strip(), "reasoning",
                             dict(payload)))
            out.append(Claim(EDGE_DIRECTION, m.group(0).strip(), "reasoning",
                             dict(payload)))
        if out:
            break     # one directional reading per sentence; avoid double count

    # -- bare location mentions -> node claims ------------------------------
    for text, ids in res.mentions(sent):
        out.append(Claim(NODE_EXISTENCE, text, "reasoning",
                         {"node_ids": sorted(ids), "from_prose": True}))

    # -- speeds -------------------------------------------------------------
    for m in _SPEED.finditer(sent):
        out.append(Claim(PREDICTION, m.group(0), "reasoning",
                         {"value": float(m.group("v")), "quantity": "speed"}))
    for m in _PU.finditer(sent):
        out.append(Claim(PREDICTION, m.group(0), "reasoning",
                         {"value": float(m.group("v")), "quantity": "voltage_pu"}))

    # -- minutes: lag vs forecast horizon -----------------------------------
    for m in _MINUTES.finditer(sent):
        val = float(m.group("v"))
        prefix = sent[:m.start()]
        is_horizon = bool(_HORIZON_CONTEXT.search(prefix)) or (
            horizon is not None and abs(val - horizon) < 1e-9)
        has_lag_language = bool(_LAG_CONTEXT.search(sent))
        if is_horizon and not has_lag_language:
            # The forecast horizon restated, not a propagation-lag claim.
            out.append(Claim(PREDICTION, m.group(0), "reasoning",
                             {"value": val, "quantity": "horizon_minutes"}))
        else:
            out.append(Claim(TEMPORAL, m.group(0), "reasoning",
                             {"value": val, "quantity": "lag_minutes",
                              "has_lag_language": has_lag_language}))

    # -- confidence ---------------------------------------------------------
    for pat, tier in _CONFIDENCE_TERMS:
        m = re.search(pat, sent, re.IGNORECASE)
        if m:
            out.append(Claim(CONFIDENCE, m.group(0), "reasoning",
                             {"asserted_tier": tier}))
            break

    # -- counterfactual -----------------------------------------------------
    if re.search(r"\b(?:if|had|would have|could have|were)\b.{0,80}"
                 r"\b(?:would|could|might)\b", sent, re.IGNORECASE):
        out.append(Claim(COUNTERFACTUAL, sent, "reasoning",
                         {"needs_model_rerun": True}))

    # -- feature attribution ------------------------------------------------
    fm = re.search(r"\b(weather|rain|precipitation|temperature|visibility|"
                   r"event|game|transit|bus|rail|time of day|speed|volume|"
                   r"occupancy|voltage|load|demand)\b", low)
    if fm:
        out.append(Claim(FEATURE_ATTRIBUTION, fm.group(0), "reasoning",
                         {"feature": fm.group(0)}))

    # -- domain terminology -------------------------------------------------
    for term in _DOMAIN_TERMS:
        if term in low:
            out.append(Claim(DOMAIN_TERMINOLOGY, term, "reasoning",
                             {"term": term}))
            break

    return out
