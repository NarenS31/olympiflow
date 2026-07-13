"""Phase 18 — Uncertainty-Aware Explanations.

WHY THIS EXISTS
---------------
A single GNNExplainer run gives ONE top-k node set. A reviewer will ask: "how much
does that set depend on the random seed — would you trust one run?" Phase 3 already
answered a SCALAR version of this (explanation_confidence = mean Jaccard of the
top-k over K reruns). Phase 18 promotes that scalar to a PER-NODE uncertainty
distribution and makes the LLM honest about it.

THE METHOD
----------
  1. Run GNNExplainer K=10 times on the SAME prediction with different random mask
     initialisations (seeds 0..K-1). Collect each run's top-k node set.
  2. For each node, count how often it appears in the top-k across the K runs, then
     classify by that appearance FREQUENCY:
         CORE       -> freq/K >= core_threshold (0.8)   -> confident cause, assert it
         PERIPHERAL -> between the two thresholds        -> uncertain, hedge it
         NOISE      -> freq/K <  noise_threshold (0.2)   -> do not assert, omit it
     Also compute the mean pairwise Jaccard of the K top-k sets = the explanation's
     STABILITY score (a set-level cousin of the Phase-3 scalar confidence).
  3. Emit an uncertainty block:
         {"core_causes": [{node_id, node_name, frequency:"9/10", confidence}],
          "peripheral_causes": [...], "explanation_stability": float, "k_runs": K}
     and attach it ADDITIVELY to the Phase-3 explanation dict (an extra top-level
     "uncertainty" key). The base explanation stays schema-valid (schema.py checks
     the required keys are present; it does not forbid extra keys), so the Phase-4
     advisory contract is untouched — this is the additive-field rule from the stub.

WHAT IT REUSES (keeps this file small + transparent)
----------------------------------------------------
  * explain.py -> GNNExplainer.explain_target(seed=k): the SAME solve Phase 3 runs
    for the confidence scalar. We just keep per-node frequencies instead of
    collapsing to one Jaccard number, so the cost model is unchanged (K solves).
  * explain.py -> ExplanationBuilder._topk_nodes / .namer / .scaler: identical
    top-k selection, node naming, and z->mph conversion as the rest of the project.

WHERE THE PROMPT + MEASUREMENT LIVE
-----------------------------------
  * The uncertainty-framed PROMPT (assert core, hedge peripheral, omit noise) is a
    FLAGGED, default-preserving addition to models/advisor/advisor.py
    (advise_uncertain / build_prompt_uncertain), exactly like the Phase-17
    counterfactual mode — so advise()/advise_condition() stay byte-identical.
  * hedging_analysis() below quantifies the KEY QUESTION: does the LLM actually
    use MORE hedge words in sentences about peripheral causes than about core ones,
    or does it treat every cited cause as equally confident?

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import os
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from ...utils.io_utils import PKG_ROOT
from .explain import SPEED_CHANNEL, ExplanationBuilder, _z_to_mph


# ===========================================================================
# Config.
# ===========================================================================
def load_uncertainty_config() -> Dict[str, Any]:
    path = os.path.join(PKG_ROOT, "configs", "uncertainty.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# Tier classification + stability.
# ===========================================================================
def classify_tier(freq: int, k_runs: int, core_threshold: float,
                  noise_threshold: float) -> str:
    """Map a node's top-k appearance count over K runs to a tier.

    Boundary rule (documented so 0.8 / 0.2 are unambiguous):
        core       : freq/K >= core_threshold   (>= is INCLUSIVE, so 8/10 == core)
        noise      : freq/K <  noise_threshold  (<  is EXCLUSIVE, so 2/10 == peripheral)
        peripheral : everything in between
    """
    frac = freq / float(k_runs) if k_runs else 0.0
    if frac >= core_threshold:
        return "core"
    if frac < noise_threshold:
        return "noise"
    return "peripheral"


def mean_pairwise_jaccard(top_sets: List[List[int]]) -> float:
    """Mean Jaccard similarity over every unordered PAIR of the K top-k sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|. Averaging over all C(K,2) pairs gives a
    single [0,1] stability score: 1.0 = every run picked the identical top-k
    (rock-solid explanation), ~0 = the runs barely agree (untrustworthy). This is
    the set-level analogue of the Phase-3 confidence scalar (which compared each
    rerun to run 0 only); the all-pairs version is symmetric and slightly more
    robust to an unlucky run-0."""
    sets = [set(s) for s in top_sets]
    pairs = list(combinations(range(len(sets)), 2))
    if not pairs:
        return 1.0 if sets else 0.0
    sims: List[float] = []
    for i, j in pairs:
        union = sets[i] | sets[j]
        sims.append(len(sets[i] & sets[j]) / len(union) if union else 0.0)
    return float(np.mean(sims))


# ===========================================================================
# The K-run analyzer: wraps an ExplanationBuilder and produces the per-node
# uncertainty distribution for one prediction.
# ===========================================================================
class UncertaintyAnalyzer:
    """Runs the frozen explainer K times on one prediction and tallies per-node
    top-k frequency. Reuses the builder's explainer, top-k selector, node namer,
    and scaler so the uncertainty overlay speaks the exact same units as every
    other layer."""

    def __init__(self, builder: ExplanationBuilder, cfg: Optional[Dict[str, Any]] = None):
        self.builder = builder
        ucfg = (cfg or load_uncertainty_config())["uncertainty"]
        self.k_runs: int = int(ucfg["k_runs"])
        self.horizon_step: int = int(ucfg["horizon_step"])
        self.top_k: int = int(ucfg["top_k"])
        self.core_threshold: float = float(ucfg["core_threshold"])
        self.noise_threshold: float = float(ucfg["noise_threshold"])

    def run_k(self, X: torch.Tensor, target_node: int
              ) -> Tuple[List[List[int]], Dict[int, int], Dict[int, float]]:
        """Run the explainer K times. Returns:
            top_sets : list of K top-k node lists (one per seed)
            freq     : node_id -> how many of the K runs put it in the top-k
            imp_sum  : node_id -> summed importance over the runs it appeared in
                       (used only to break frequency ties when ranking a tier)

        Shapes: each explain_target returns node_imp [N]; _topk_nodes -> a list of
        up to top_k node ids (target excluded), so top_sets[k] has <= top_k ids.
        """
        # Honor the builder's configured top_k for this analysis (the builder was
        # constructed with its own top_k; we temporarily use ours so a run with a
        # different config still asks for the intended set size). Restore after.
        saved_topk = self.builder.top_k
        self.builder.top_k = self.top_k
        top_sets: List[List[int]] = []
        freq: Dict[int, int] = {}
        imp_sum: Dict[int, float] = {}
        try:
            for k in range(self.k_runs):
                node_imp, _, _, _ = self.builder.explainer.explain_target(
                    X, target_node, self.horizon_step, seed=k)     # node_imp [N]
                topk = self.builder._topk_nodes(node_imp, target_node)  # <= top_k ids
                top_sets.append(topk)
                for nid in topk:
                    freq[nid] = freq.get(nid, 0) + 1
                    imp_sum[nid] = imp_sum.get(nid, 0.0) + float(node_imp[nid])
        finally:
            self.builder.top_k = saved_topk
        return top_sets, freq, imp_sum

    def build_uncertainty_block(self, X: torch.Tensor, target_node: int
                                ) -> Dict[str, Any]:
        """Produce the uncertainty block for one prediction (the JSON described in
        the module docstring). Nodes are split into core / peripheral / noise; only
        core + peripheral are reported (noise is deliberately dropped — the LLM must
        never assert it)."""
        top_sets, freq, imp_sum = self.run_k(X, target_node)
        stability = mean_pairwise_jaccard(top_sets)

        namer = self.builder.namer
        scaler = self.builder.scaler
        speeds_z = X[0, -1, :, SPEED_CHANNEL].cpu().numpy()   # last-obs speed per node [N]

        def _node_record(nid: int) -> Dict[str, Any]:
            return {
                "node_id": int(nid),
                "node_name": namer.name(nid),
                "frequency": "{}/{}".format(freq[nid], self.k_runs),
                "confidence": round(freq[nid] / float(self.k_runs), 3),
                "current_speed_mph": round(_z_to_mph(float(speeds_z[nid]), scaler), 2),
            }

        core: List[Dict[str, Any]] = []
        peripheral: List[Dict[str, Any]] = []
        noise_ids: List[int] = []
        for nid, f in freq.items():
            tier = classify_tier(f, self.k_runs, self.core_threshold,
                                 self.noise_threshold)
            if tier == "core":
                core.append(_node_record(nid))
            elif tier == "peripheral":
                peripheral.append(_node_record(nid))
            else:
                noise_ids.append(int(nid))

        # Rank each tier most-frequent first, then by summed importance (a stable,
        # documented tie-break) so the prompt lists the strongest causes at the top.
        def _key(rec: Dict[str, Any]) -> Tuple[float, float]:
            nid = rec["node_id"]
            return (freq[nid], imp_sum.get(nid, 0.0))
        core.sort(key=_key, reverse=True)
        peripheral.sort(key=_key, reverse=True)

        return {
            "k_runs": self.k_runs,
            "core_threshold": self.core_threshold,
            "noise_threshold": self.noise_threshold,
            "explanation_stability": round(float(stability), 4),
            "core_causes": core,
            "peripheral_causes": peripheral,
            "noise_count": len(noise_ids),
            # id lists kept for the study's scoring/logging (not shown to the LLM).
            "core_node_ids": [r["node_id"] for r in core],
            "peripheral_node_ids": [r["node_id"] for r in peripheral],
        }


def attach_uncertainty(exp: Dict[str, Any], block: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of the Phase-3 explanation with the uncertainty block
    added under an extra top-level "uncertainty" key. ADDITIVE ONLY — the required
    schema keys are untouched, so validate_explanation still passes and the Phase-4
    advisory contract is unchanged."""
    out = dict(exp)
    out["uncertainty"] = block
    return out


# ===========================================================================
# Hedging analysis — the KEY QUESTION.
#
# Does the LLM actually HEDGE on peripheral causes (use "may / possibly /
# uncertain" language) while asserting core causes plainly, or does it treat every
# cited cause as equally confident? We measure this by attributing each sentence of
# the advisory's `reasoning` to the tier(s) whose region names it mentions, then
# counting hedge words per sentence.
#
# This is a documented HEURISTIC, not ground truth: sentence attribution is by
# region-name substring and hedge detection is a fixed word list. It is directional
# evidence (peripheral hedged MORE than core => the framing worked), reported with
# its own counts so a reader can judge it.
# ===========================================================================
HEDGE_WORDS = [
    "may", "might", "maybe", "possibly", "possible", "perhaps", "potentially",
    "uncertain", "uncertainty", "unclear", "unconfirmed", "tentative", "tentatively",
    "could", "appears", "appear", "seems", "seem", "likely", "unlikely",
    "presumably", "arguably", "somewhat", "not certain", "less certain",
]
# Precompile a word-boundary matcher for the single-word hedges (multi-word phrases
# like "not certain" are checked by plain substring). Case-insensitive.
_HEDGE_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in HEDGE_WORDS if " " not in w) + r")\b",
    re.IGNORECASE)
_HEDGE_PHRASES = [w for w in HEDGE_WORDS if " " in w]


def region_variants(node_name: str) -> List[str]:
    """The lowercase strings we accept as a mention of a node's region.

    node_name looks like "Glendale / Burbank (sensor 759772, 34.15, -118.30)". We
    take the region label before the parenthesis and its slash-separated parts
    ("glendale", "burbank"), keeping tokens of length > 3 so we don't match on tiny
    fragments. This mirrors the faithfulness resolver's region granularity."""
    label = node_name.split("(")[0].strip().lower()
    variants = {label}
    for part in re.split(r"[/,]", label):
        part = part.strip()
        if len(part) > 3:
            variants.add(part)
    return [v for v in variants if v]


def _split_sentences(text: str) -> List[str]:
    """Cheap sentence split on ., !, ?, ; and newlines. Good enough for the
    advisory's short, declarative reasoning."""
    parts = re.split(r"[.!?;\n]+", text or "")
    return [p.strip() for p in parts if p.strip()]


def _count_hedges(sentence: str) -> int:
    low = sentence.lower()
    n = len(_HEDGE_RE.findall(low))
    for ph in _HEDGE_PHRASES:
        n += low.count(ph)
    return n


def hedging_analysis(reasoning: str, core_causes: List[Dict[str, Any]],
                     peripheral_causes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attribute each reasoning sentence to core/peripheral tiers by region mention
    and tally hedge words. A sentence can be attributed to BOTH tiers if it mentions
    regions from both; that is fine — we count it toward each so a mixed sentence is
    not silently dropped.

    Returns per-tier: number of sentences that mention that tier, how many contain a
    hedge word, the hedge-word total, and the derived rates. The headline number for
    the paper is (peripheral hedge rate - core hedge rate): positive => the LLM
    hedged uncertain causes more than confident ones (the framing worked)."""
    core_vars = [v for c in core_causes for v in region_variants(c["node_name"])]
    peri_vars = [v for c in peripheral_causes for v in region_variants(c["node_name"])]

    sentences = _split_sentences(reasoning)
    stats = {
        "core": {"sentences": 0, "hedged_sentences": 0, "hedge_words": 0},
        "peripheral": {"sentences": 0, "hedged_sentences": 0, "hedge_words": 0},
    }
    for sent in sentences:
        low = sent.lower()
        n_hedge = _count_hedges(sent)
        touches_core = any(v in low for v in core_vars)
        touches_peri = any(v in low for v in peri_vars)
        for tier, touches in (("core", touches_core), ("peripheral", touches_peri)):
            if touches:
                stats[tier]["sentences"] += 1
                stats[tier]["hedge_words"] += n_hedge
                if n_hedge > 0:
                    stats[tier]["hedged_sentences"] += 1

    def _rate(t: str) -> Optional[float]:
        n = stats[t]["sentences"]
        return (stats[t]["hedged_sentences"] / n) if n else None

    core_rate = _rate("core")
    peri_rate = _rate("peripheral")
    return {
        "n_sentences": len(sentences),
        "core": stats["core"],
        "peripheral": stats["peripheral"],
        "core_hedge_rate": core_rate,               # frac of core sentences with >=1 hedge
        "peripheral_hedge_rate": peri_rate,         # frac of peripheral sentences with >=1 hedge
        # Positive => peripheral hedged more than core (the honest behaviour we want).
        # None if either tier had no attributable sentence in this advisory.
        "hedge_rate_gap": (peri_rate - core_rate
                           if (core_rate is not None and peri_rate is not None)
                           else None),
    }


# ===========================================================================
# Single-scenario DEMO — verifies the K-run analysis end-to-end (no LLM).
#   python -m xtraffic.models.explainer.uncertain_explainer
# ===========================================================================
def _demo_render(exp: Dict[str, Any]) -> str:
    u = exp["uncertainty"]
    p = exp["prediction"]
    L: List[str] = []
    L.append("=" * 72)
    L.append("UNCERTAINTY-AWARE EXPLANATION — {}".format(exp["meta"]["timestamp"]))
    L.append("=" * 72)
    L.append("Target: {}".format(p["node_name"]))
    L.append("  current {} mph -> predicted {} mph in {} min".format(
        p["current_speed_mph"], p["predicted_speed_mph"], p["horizon_minutes"]))
    L.append("K = {} explainer runs | explanation stability (mean pairwise "
             "Jaccard) = {:.3f}".format(u["k_runs"], u["explanation_stability"]))
    L.append("")
    L.append("CORE causes (>= {:.0%} of runs — assert):".format(u["core_threshold"]))
    for c in u["core_causes"]:
        L.append("  - {:30s} {}  (conf {:.2f})  {} mph".format(
            c["node_name"].split("(")[0].strip(), c["frequency"],
            c["confidence"], c["current_speed_mph"]))
    if not u["core_causes"]:
        L.append("  (none)")
    L.append("PERIPHERAL causes (uncertain — hedge):")
    for c in u["peripheral_causes"]:
        L.append("  - {:30s} {}  (conf {:.2f})  {} mph".format(
            c["node_name"].split("(")[0].strip(), c["frequency"],
            c["confidence"], c["current_speed_mph"]))
    if not u["peripheral_causes"]:
        L.append("  (none)")
    L.append("NOISE causes dropped: {}".format(u["noise_count"]))
    return "\n".join(L)


def main() -> None:
    import argparse

    from .scenarios import load_window, select_scenarios

    ap = argparse.ArgumentParser(description="Phase 18 uncertainty demo (no LLM)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scenario", default="high_congestion",
                    help="which curated scenario to explain (see scenarios.py)")
    args = ap.parse_args()

    cfg = load_uncertainty_config()
    dataset = args.dataset or cfg["dataset"]
    checkpoint = args.checkpoint or cfg["checkpoint"]
    horizon_step = int(cfg["uncertainty"]["horizon_step"])

    device = torch.device("cpu")
    builder = ExplanationBuilder(checkpoint, dataset, device=device,
                                 top_k=int(cfg["uncertainty"]["top_k"]))
    analyzer = UncertaintyAnalyzer(builder, cfg)

    scenarios = {s["name"]: s for s in select_scenarios(dataset, builder.scaler)}
    sc = scenarios.get(args.scenario) or list(scenarios.values())[0]
    X = load_window(dataset, sc["sample_index"])                  # [1, T, N, C]

    print("[phase18] running {} explainer solves for scenario '{}' (idx {}, node {})..."
          .format(analyzer.k_runs, sc["name"], sc["sample_index"], sc["target_node"]))
    base = builder.explain_prediction(X, target_node=sc["target_node"],
                                      horizon_step=horizon_step,
                                      timestamp=sc["timestamp"])
    block = analyzer.build_uncertainty_block(X, sc["target_node"])
    exp = attach_uncertainty(base, block)
    print(_demo_render(exp))


if __name__ == "__main__":
    main()
