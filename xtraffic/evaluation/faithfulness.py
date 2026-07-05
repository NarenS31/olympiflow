"""The FAITHFULNESS METRIC — Contribution #2 of the paper.

WHAT THIS MEASURES
------------------
Layer 2 (GNNExplainer) produces a *mathematical* explanation: a set of top-k
sensor nodes that drive the prediction. Layer 3 (the LLM advisor) then writes a
*natural-language* explanation citing "causes". Faithfulness asks one question:

    Does the LLM's stated reasoning actually correspond to the math, or is it
    making up plausible-sounding causes?

That is the whole ballgame for an XAI+LLM pipeline. An advisor that sounds
confident but cites causes the model never used is worse than useless — it
launders a hallucination as an explanation. This module quantifies that.

THE PIPELINE OF THIS FILE
-------------------------
1. ENTITY RESOLUTION: map each free-text `cited_causes[i].location` the LLM
   wrote back to a set of sensor node_ids, using an escalating ladder
   (exact region -> exact name/sensor id -> fuzzy -> geographic). We log which
   rung fired for every citation, because a reviewer will ask how robust the
   resolver is (Phase-5 gate: >=90% on a hand-labeled set).
2. METRICS: from the resolved citations and the explanation's top-k node set we
   compute cause precision / recall / F1, quantitative fidelity, and the
   hallucination rate. Each is defined mathematically in its docstring.

WHY WE RESOLVE INDEPENDENTLY OF THE LLM'S OWN `resolved_node_id`
---------------------------------------------------------------
The advisory JSON already contains the LLM's *self-reported* resolved_node_id.
We deliberately DO NOT trust it for scoring — letting the model grade its own
homework would make the metric meaningless. We resolve purely from the location
TEXT against the ground-truth node table, and only report agreement with the
LLM's self-reported id as a side diagnostic.

WHY REGION-LEVEL CREDIT
-----------------------
METR-LA node names are region-tagged ("San Fernando Valley (sensor 772167, ...)")
and the LLM naturally cites at region granularity ("the San Fernando Valley").
So a citation resolves to the SET of nodes in that region, and it counts as
"hitting" the explanation if that set intersects the explanation's top-k. This
is a documented, deliberate generosity appropriate to the naming granularity: if
the explanation surfaced a San Fernando Valley sensor and the LLM said "San
Fernando Valley", that is a faithful translation, not a lucky guess. The failure
modes it still catches: citing a region with NO top-k node in it (resolves,
but misses -> hallucination) and citing something that matches no node at all
(unresolved -> hallucination).

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# --- Fuzzy matcher: prefer rapidfuzz (the playbook's choice), but fall back to
# the stdlib difflib so this module runs on a fresh machine with no extra
# install. Both return a 0-100 similarity so the threshold below is identical
# either way. The active backend is recorded on every resolution so the paper
# can state which was used. ------------------------------------------------
try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore

    def _ratio(a: str, b: str) -> float:
        # token_set_ratio: order-insensitive, ignores extra words ("the ... area")
        return float(_rf_fuzz.token_set_ratio(a, b))

    FUZZ_BACKEND = "rapidfuzz.token_set_ratio"
except Exception:  # pragma: no cover - exercised only when rapidfuzz is absent
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return 100.0 * SequenceMatcher(None, a, b).ratio()

    FUZZ_BACKEND = "difflib.SequenceMatcher"

# Fuzzy acceptance threshold (0-100). 82 is deliberately conservative: high
# enough that "Glendale" -> "Glendale / Burbank" passes but "Glendale" ->
# "Long Beach corridor" does not. Documented here because a reviewer will ask
# where the number came from; it is validated by the hand-labeled resolver test.
FUZZY_THRESHOLD = 82.0

# Quantitative-fidelity tolerance: a number the LLM states (a speed, a lag) is
# "correct" if within +/-10% of the matching quantity in the explanation JSON.
NUMERIC_TOLERANCE = 0.10


# ---------------------------------------------------------------------------
# The ground-truth node table: node_id -> canonical name, region, coordinates.
# Built once from the Phase-1 node_meta (the same source node_names.py uses), so
# the resolver's universe is exactly the model's universe.
# ---------------------------------------------------------------------------
class NodeTable:
    """Everything the resolver needs to map text -> node_ids for one city."""

    def __init__(self, node_meta: Dict[str, Any]):
        # Imported here (not at module top) so importing this file doesn't drag
        # in the explainer package when a caller only wants the metric math.
        from ..models.explainer.node_names import NodeNamer, region_for

        self.namer = NodeNamer(node_meta)
        self.sensor_ids: List[int] = list(node_meta["sensor_ids"])
        self.latlon: List[List[float]] = list(node_meta["latlon"])
        self.n_nodes: int = len(self.sensor_ids)
        # City selects the region table (Phase 6 cross-city); defaults to LA.
        self.city: str = node_meta.get("dataset", "metr_la")

        # Per-node region label (e.g. "san fernando valley"), lowercased once so
        # every comparison downstream is case-insensitive.
        self.region: List[str] = [
            region_for(self.latlon[i][0], self.latlon[i][1], self.city).lower()
            for i in range(self.n_nodes)
        ]
        # region label -> the set of node_ids that fall in it. A region maps to
        # MANY nodes (that's why a citation resolves to a set, not one id).
        self.region_to_nodes: Dict[str, Set[int]] = {}
        for nid, reg in enumerate(self.region):
            self.region_to_nodes.setdefault(reg, set()).add(nid)

        # sensor_id (as string) -> node_id, for "sensor 772167" style citations.
        self.sid_to_node: Dict[str, int] = {
            str(sid): nid for nid, sid in enumerate(self.sensor_ids)
        }
        # Sorted unique region labels, for fuzzy matching against.
        self.regions: List[str] = sorted(self.region_to_nodes.keys())


# ---------------------------------------------------------------------------
# Small geographic gazetteer: free-text directions/landmarks -> region label.
# These are the phrases an LLM uses that are NOT verbatim region names but still
# refer to a real area ("near downtown", "the Westside"). Kept tiny and explicit
# so it is auditable; matched only after exact + fuzzy fail.
# ---------------------------------------------------------------------------
_GEO_ALIASES: List[Tuple[str, str]] = [
    ("downtown", "downtown la"),
    ("dtla", "downtown la"),
    ("city center", "downtown la"),
    ("westside", "santa monica / westside"),
    ("west side", "santa monica / westside"),
    ("santa monica", "santa monica / westside"),
    ("west la", "west la / sawtelle"),
    ("sawtelle", "west la / sawtelle"),
    ("valley", "san fernando valley"),
    ("san fernando", "san fernando valley"),
    ("glendale", "glendale / burbank"),
    ("burbank", "glendale / burbank"),
    ("hollywood", "hollywood"),
    ("koreatown", "koreatown / mid-city"),
    ("mid-city", "koreatown / mid-city"),
    ("mid city", "koreatown / mid-city"),
    ("pasadena", "pasadena / east"),
    ("east la", "east la"),
    ("east los angeles", "east la"),
    ("south la", "south la"),
    ("south los angeles", "south la"),
    ("inglewood", "inglewood / lax"),
    ("lax", "inglewood / lax"),
    ("long beach", "long beach corridor"),
]


def _normalize(text: str) -> str:
    """Lowercase and strip the parenthetical '(sensor 123, lat, lon)' tail plus
    punctuation, so 'Glendale / Burbank (sensor 759772, ...)' and 'glendale
    burbank' compare cleanly."""
    text = text.lower()
    text = re.sub(r"\(.*?\)", " ", text)          # drop parentheticals
    text = re.sub(r"[^a-z0-9/ ]+", " ", text)     # keep '/' (region separators)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Entity resolution: one cited location string -> a set of node_ids + method.
# ---------------------------------------------------------------------------
class Resolution:
    """The outcome of resolving one cited location. `node_ids` is the candidate
    set (empty == unresolved). `method` names the rung of the ladder that fired,
    for the resolver-accuracy report."""

    def __init__(self, node_ids: Set[int], method: str,
                 matched: Optional[str] = None, score: Optional[float] = None):
        self.node_ids = node_ids
        self.method = method            # exact_region|exact_sensor|fuzzy|geographic|unresolved
        self.matched = matched          # what region/name it matched, for logging
        self.score = score              # fuzzy score when method == "fuzzy"

    @property
    def resolved(self) -> bool:
        return len(self.node_ids) > 0


def resolve_location(location: str, table: NodeTable) -> Resolution:
    """Resolve one free-text location to a set of node_ids via the escalating
    ladder. Order matters: cheap/precise rungs first, generous rungs last, so
    the logged `method` reflects the STRONGEST evidence that succeeded."""
    norm = _normalize(location)
    if not norm:
        return Resolution(set(), "unresolved")

    # Rung 1 — explicit sensor id ("sensor 772167" / "node 197" won't appear as a
    # sensor id but a bare sensor number might). Match any 4-6 digit token that
    # is a known sensor id.
    for tok in re.findall(r"\d{4,7}", norm):
        if tok in table.sid_to_node:
            nid = table.sid_to_node[tok]
            return Resolution({nid}, "exact_sensor", matched="sensor " + tok)

    # Rung 2 — exact region name. The LLM cited a region label verbatim.
    if norm in table.region_to_nodes:
        return Resolution(set(table.region_to_nodes[norm]), "exact_region",
                          matched=norm)

    # Rung 3 — fuzzy match against region labels. Handles "glendale burbank" vs
    # "glendale / burbank", "san fernando" vs "san fernando valley", etc.
    best_reg, best_score = None, 0.0
    for reg in table.regions:
        s = _ratio(norm, reg)
        if s > best_score:
            best_reg, best_score = reg, s
    if best_reg is not None and best_score >= FUZZY_THRESHOLD:
        return Resolution(set(table.region_to_nodes[best_reg]), "fuzzy",
                          matched=best_reg, score=best_score)

    # Rung 4 — geographic gazetteer. Free-text directions/landmarks that are not
    # region labels. Longest alias first so "east los angeles" wins over "east".
    for alias, reg in sorted(_GEO_ALIASES, key=lambda kv: -len(kv[0])):
        if alias in norm and reg in table.region_to_nodes:
            return Resolution(set(table.region_to_nodes[reg]), "geographic",
                              matched=alias + "->" + reg)

    return Resolution(set(), "unresolved")


# ---------------------------------------------------------------------------
# The metrics.
# ---------------------------------------------------------------------------
def _topk_node_ids(explanation: Dict[str, Any]) -> Set[int]:
    """The explainer's ground-truth set of causal nodes = the top_nodes ids."""
    return {int(n["node_id"]) for n in explanation.get("top_nodes", [])}


def _extract_numbers(text: str) -> List[float]:
    """Pull decimal numbers out of the LLM's prose for quantitative fidelity.
    We only care about plausible speeds/lags, so keep 0-200."""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]
    return [n for n in nums if 0.0 <= n <= 200.0]


def score_advisory(explanation: Dict[str, Any], advisory: Dict[str, Any],
                   table: NodeTable) -> Dict[str, Any]:
    """Compute all faithfulness metrics for ONE (explanation, advisory) pair.

    DEFINITIONS (let C = the set of causes the LLM cited, resolved to node sets;
    let K = the explainer's top-k node ids):

      cause_precision = |cited causes that hit K| / |cited causes|
          "Of what the LLM claimed, how much was actually in the explanation?"
          A cited cause `c` HITS K iff resolve(c) ∩ K ≠ ∅.

      cause_recall    = |top-k nodes the LLM covered| / |K|
          "Of what the explanation said mattered, how much did the LLM mention?"
          A top-k node is COVERED iff it lies in the union of all resolved sets.

      faithfulness_f1 = harmonic mean(precision, recall)   ← the headline number

      hallucination_rate = |cited causes that miss K or don't resolve| / |cited|
          = 1 - precision (kept as its own field because it is the number the
          paper leads with for condition B).

      quantitative_fidelity = fraction of numbers the LLM stated that match some
          quantity in the explanation (speeds of target/top nodes, the lag)
          within +/-10%. Undefined (None) if the LLM stated no numbers.

    Returns a flat dict of metrics plus a per-cause resolution log.
    """
    topk = _topk_node_ids(explanation)
    causes: List[Dict[str, Any]] = advisory.get("cited_causes", []) or []

    per_cause: List[Dict[str, Any]] = []
    covered: Set[int] = set()          # top-k nodes touched by any citation
    hits = 0                           # citations that intersect top-k

    for c in causes:
        loc = c.get("location", "") if isinstance(c, dict) else str(c)
        res = resolve_location(loc, table)
        inter = res.node_ids & topk
        is_hit = len(inter) > 0
        if is_hit:
            hits += 1
            covered |= inter
        per_cause.append({
            "location": loc,
            "resolved_method": res.method,
            "resolved_matched": res.matched,
            "resolved_score": res.score,
            "resolved_node_ids": sorted(res.node_ids),
            "llm_self_reported_id": (c.get("resolved_node_id")
                                     if isinstance(c, dict) else None),
            "hit_topk": is_hit,
            "hit_nodes": sorted(inter),
        })

    n_causes = len(causes)
    precision = (hits / n_causes) if n_causes else 0.0
    recall = (len(covered) / len(topk)) if topk else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    hallucination = (1.0 - precision) if n_causes else 1.0

    # --- quantitative fidelity ---------------------------------------------
    ref: List[float] = []
    pred = explanation.get("prediction", {})
    if "current_speed_mph" in pred:
        ref.append(float(pred["current_speed_mph"]))
    if "predicted_speed_mph" in pred:
        ref.append(float(pred["predicted_speed_mph"]))
    for n in explanation.get("top_nodes", []):
        ref.append(float(n["current_speed_mph"]))
    if "propagation_lag_minutes" in explanation:
        ref.append(float(explanation["propagation_lag_minutes"]))

    stated = _extract_numbers(advisory.get("reasoning", ""))
    if stated:
        good = 0
        for s in stated:
            # A stated number is "faithful" if within tolerance of ANY reference
            # quantity (we don't know which the LLM meant; being generous here
            # avoids penalising correct-but-ambiguous numbers).
            if any(abs(s - r) <= NUMERIC_TOLERANCE * max(abs(r), 1e-6)
                   for r in ref):
                good += 1
        quant_fidelity: Optional[float] = good / len(stated)
    else:
        quant_fidelity = None

    return {
        "n_cited_causes": n_causes,
        "n_topk": len(topk),
        "cause_precision": precision,
        "cause_recall": recall,
        "faithfulness_f1": f1,
        "hallucination_rate": hallucination,
        "quantitative_fidelity": quant_fidelity,
        "n_numbers_stated": len(stated),
        "per_cause": per_cause,
    }


def mean_std(values: Sequence[Optional[float]]) -> Tuple[float, float, int]:
    """Mean, population std, and count over the non-None values (metrics like
    quantitative_fidelity can be None when the LLM stated no numbers)."""
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, var ** 0.5, n
