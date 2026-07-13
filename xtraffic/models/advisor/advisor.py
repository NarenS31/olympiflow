"""The LLM advisory module — Layer 3 of XTraffic (Phase 4).

Given a Phase-3 mathematical explanation (the schema.py contract) plus retrieved
city context, this asks a LOCAL Ollama model to:
  1) explain the predicted congestion in plain language, citing ONLY causes that
     appear in the explanation, and
  2) propose 3-5 concrete, infrastructure-aware interventions.

The output is itself structured JSON (validated + retried), because Phase 5's
faithfulness metric compares the LLM's cited causes back against the explanation.
If this output drifts in shape, that metric breaks — so we pin and validate it
here, exactly like schema.py pins the explanation.

Design choices (worth understanding — you'll defend these):
  * LOCAL ONLY via Ollama (CLAUDE.md): no API keys, fully reproducible offline.
  * Low temperature: we want faithful TRANSLATION, not creativity.
  * format="json": Ollama constrains decoding to valid JSON, cutting malformed
    output; we STILL validate + retry, because valid-JSON != right-shape.
  * The system prompt hard-forbids inventing causes. That instruction is the
    behaviour Phase 5 measures — we want the honest baseline to be reasonable.

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from ...utils.io_utils import PKG_ROOT
from .knowledge_base import (
    KnowledgeBase, load_kb_for_city, render_kb_block,
)


# ----------------------------------------------------------------------------
# Advisory output contract (the JSON the LLM must return). Phase 5 depends on it.
# ----------------------------------------------------------------------------
# {
#   "reasoning": str,
#   "cited_causes": [{"location": str, "resolved_node_id": int|null}],
#   "recommendations": [{"action": str, "location": str,
#                        "time_window_minutes": int, "expected_effect": str,
#                        "grounded_in": [str]}]
# }
def validate_advisory(adv: Any) -> List[str]:
    """Return a list of problems (empty == valid). We return rather than raise so
    the retry loop can feed the errors back to the model."""
    errs: List[str] = []
    if not isinstance(adv, dict):
        return ["advisory: expected JSON object, got {}".format(type(adv).__name__)]

    if not isinstance(adv.get("reasoning"), str) or not adv.get("reasoning").strip():
        errs.append("reasoning: expected non-empty string")

    causes = adv.get("cited_causes")
    if not isinstance(causes, list):
        errs.append("cited_causes: expected a list")
    else:
        for i, c in enumerate(causes):
            if not isinstance(c, dict):
                errs.append("cited_causes[{}]: expected object".format(i)); continue
            if not isinstance(c.get("location"), str):
                errs.append("cited_causes[{}].location: expected string".format(i))
            rn = c.get("resolved_node_id", None)
            # int OR null are both allowed; bool is not an int here.
            if not (rn is None or (isinstance(rn, int) and not isinstance(rn, bool))):
                errs.append("cited_causes[{}].resolved_node_id: expected int or null".format(i))

    recs = adv.get("recommendations")
    if not isinstance(recs, list):
        errs.append("recommendations: expected a list")
    elif not (3 <= len(recs) <= 5):
        errs.append("recommendations: expected 3-5 items, got {}".format(len(recs)))
    else:
        for i, r in enumerate(recs):
            if not isinstance(r, dict):
                errs.append("recommendations[{}]: expected object".format(i)); continue
            for key in ("action", "location", "expected_effect"):
                if not isinstance(r.get(key), str):
                    errs.append("recommendations[{}].{}: expected string".format(i, key))
            tw = r.get("time_window_minutes")
            if not (isinstance(tw, int) and not isinstance(tw, bool)):
                errs.append("recommendations[{}].time_window_minutes: expected int".format(i))
            if not isinstance(r.get("grounded_in"), list):
                errs.append("recommendations[{}].grounded_in: expected list".format(i))
    return errs


# ----------------------------------------------------------------------------
# Rendering the explanation as clean structured text for the prompt.
# ----------------------------------------------------------------------------
def render_explanation_text(exp: Dict[str, Any]) -> str:
    """Turn the explanation JSON into human-readable structured text, using node
    NAMES (not indices) so the LLM reasons in places, not numbers."""
    p = exp["prediction"]
    lines: List[str] = []
    lines.append("TARGET PREDICTION:")
    lines.append("  Location: {} (node_id {})".format(p["node_name"], p["node_id"]))
    lines.append("  Current speed: {} mph".format(p["current_speed_mph"]))
    lines.append("  Predicted speed in {} min: {} mph".format(
        p["horizon_minutes"], p["predicted_speed_mph"]))

    lines.append("")
    lines.append("TOP CONTRIBUTING SENSORS (importance = how much this sensor "
                 "drives the prediction; ONLY these are valid causes):")
    for n in exp.get("top_nodes", []):
        lines.append("  - {} (node_id {}): importance {:.3f}, currently {} mph".format(
            n["node_name"], n["node_id"], float(n["importance"]), n["current_speed_mph"]))

    path = exp.get("propagation_path", [])
    lines.append("")
    lines.append("PROPAGATION: congestion appears to travel along a path of {} "
                 "sensors, with an estimated lag of {} minutes between the source "
                 "and the target.".format(len(path), exp.get("propagation_lag_minutes")))
    lines.append("EXPLANATION CONFIDENCE: {:.2f} (0-1; higher = the explainer was "
                 "more stable across reruns).".format(
                     float(exp.get("explanation_confidence", 0.0))))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Prompt assembly (exact order required by Phase 4: SYSTEM, CITY CONTEXT,
# MATHEMATICAL EXPLANATION, TASK).
# ----------------------------------------------------------------------------
_SYSTEM = (
    "You are a traffic-operations advisor. You may ONLY cite causes that are "
    "present in the mathematical explanation below. Do NOT invent sensors, "
    "roads, incidents, or numbers that are not given to you. Prediction is not "
    "your job — translation and recommendation are. Every recommendation must be "
    "physically implementable given the stated city infrastructure."
)

_TASK = (
    "TASK:\n"
    "1) Explain in plain language why this congestion is predicted, referencing "
    "the SPECIFIC locations and quantities in the mathematical explanation "
    "(use the location names and speeds given, not invented ones).\n"
    "2) Provide 3 to 5 interventions implementable within 15 minutes given the "
    "stated infrastructure constraints. Each needs a time window (minutes) and "
    "an expected effect.\n\n"
    "Return ONLY a JSON object with EXACTLY this shape (no prose outside JSON):\n"
    "{\n"
    '  "reasoning": "<plain-language explanation citing the given locations>",\n'
    '  "cited_causes": [\n'
    '    {"location": "<a location name from the explanation>", '
    '"resolved_node_id": <the node_id from the explanation, or null>}\n'
    "  ],\n"
    '  "recommendations": [\n'
    '    {"action": "<what to do>", "location": "<where>", '
    '"time_window_minutes": <int>, "expected_effect": "<result>", '
    '"grounded_in": ["<which cited cause(s) or city-context fact this rests on>"]}\n'
    "  ]\n"
    "}"
)


def build_prompt(exp: Dict[str, Any], kb_block: str) -> str:
    """Assemble the full prompt in the required order."""
    exp_text = render_explanation_text(exp)
    return (
        "{system}\n\n"
        "=== CITY CONTEXT (retrieved knowledge base) ===\n{kb}\n\n"
        "=== MATHEMATICAL EXPLANATION ===\n{exp}\n\n"
        "=== {task}"
    ).format(system=_SYSTEM, kb=kb_block or "(no city context retrieved)",
             exp=exp_text, task=_TASK)


# ----------------------------------------------------------------------------
# Condition variants for the Phase-5 faithfulness ablation (A / B / C).
#
# These are NEW (added in Phase 5) and do NOT change the Phase-4 behavior:
# `advise()` and `build_prompt()` above are untouched and remain exactly
# condition A. We add B and C so run_faithfulness_study.py can prove the paper's
# core claim — that the mathematical EXPLANATION is what keeps the LLM honest.
#
#   A (full)            : SYSTEM + CITY CONTEXT + EXPLANATION + TASK   [= advise()]
#   B (no explanation)  : SYSTEM + CITY CONTEXT + PREDICTION-ONLY + TASK
#                         The LLM is told the predicted congestion but NOT which
#                         sensors caused it, so any "cited cause" is invented ->
#                         hallucination should spike. This is the money result.
#   C (no city context) : SYSTEM + EXPLANATION + TASK  (KB block empty)
#                         Isolates the contribution of the city knowledge base.
#
# Phase 15b adds two MORE conditions (reviewer-driven robustness). They reuse the
# same scoring against the explainer top-k; only the CITY CONTEXT block changes:
#   C_RICH (fuller context)      : like A but a DELIBERATELY FULLER context block
#                                  (more KB chunks + bigger budget). Tests whether
#                                  MORE context — not just any context — moves
#                                  faithfulness. If C_RICH ~= A ~= C, the "context
#                                  is orthogonal to faithfulness" claim is stronger.
#   D_CONTRA (contradictory ctx) : city context that DISAGREES with the math (names
#                                  a DIFFERENT, real corridor as the active
#                                  bottleneck). The sharp grounding stress test: if
#                                  hallucination stays ~0 the LLM trusted the math
#                                  over the false context; if it spikes we found a
#                                  real limitation. (Labelled D_CONTRA, not "D", so
#                                  it never blurs with Phase-12's condition D=SHAP.)
# ----------------------------------------------------------------------------
def render_prediction_only_text(exp: Dict[str, Any]) -> str:
    """Condition B input: the target prediction WITHOUT the explanation's
    top_nodes / propagation. The model sees what is predicted, not why."""
    p = exp["prediction"]
    return (
        "TARGET PREDICTION:\n"
        "  Location: {} (node_id {})\n"
        "  Current speed: {} mph\n"
        "  Predicted speed in {} min: {} mph\n"
        "(No mathematical explanation is provided in this condition.)"
    ).format(p["node_name"], p["node_id"], p["current_speed_mph"],
             p["horizon_minutes"], p["predicted_speed_mph"])


_TASK_NO_EXPLANATION = _TASK.replace(
    "in the mathematical explanation ",
    "").replace(
    "the SPECIFIC locations and quantities in the mathematical explanation "
    "(use the location names and speeds given, not invented ones)",
    "the likely locations and quantities involved")


def build_prompt_condition(exp: Dict[str, Any], kb_block: str,
                           condition: str,
                           extra_instruction: Optional[str] = None) -> str:
    """Assemble the prompt for a condition.

    A, C_RICH, and D_CONTRA are structurally IDENTICAL (SYSTEM + CITY CONTEXT +
    EXPLANATION + TASK) — they differ ONLY in what `kb_block` the caller passes
    (normal / fuller / contradictory). advise_condition() builds the right block;
    this function just places it. C blanks the context; B swaps the explanation
    for a prediction-only view.

    Phase 16 (FLAGGED, default-preserving): `extra_instruction` appends one more
    block AFTER the TASK — used by the Active Grounding Loop to attach a targeted
    correction ("you did not address these top-k nodes ..."). When it is None
    (every Phase-4/5/15b caller), the returned prompt is BYTE-IDENTICAL to before,
    so no earlier behaviour changes. It goes last so the model reads the correction
    as the final, most recent instruction."""
    if condition in ("A", "C_RICH", "D_CONTRA"):
        prompt = build_prompt(exp, kb_block)
    elif condition == "C":
        # Same as A but with the city context deliberately blanked out.
        prompt = build_prompt(exp, "")
    elif condition == "B":
        prompt = (
            "{system}\n\n"
            "=== CITY CONTEXT (retrieved knowledge base) ===\n{kb}\n\n"
            "=== PREDICTION ===\n{pred}\n\n"
            "=== {task}"
        ).format(system=_SYSTEM,
                 kb=kb_block or "(no city context retrieved)",
                 pred=render_prediction_only_text(exp),
                 task=_TASK_NO_EXPLANATION)
    else:
        raise ValueError(
            "condition must be one of A/B/C/C_RICH/D_CONTRA, got {!r}".format(condition))

    if extra_instruction:
        prompt = prompt + "\n\n" + extra_instruction
    return prompt


# ----------------------------------------------------------------------------
# Phase 15b — D_CONTRA: a contradictory CITY CONTEXT block.
# ----------------------------------------------------------------------------
def build_contradictory_context(exp: Dict[str, Any],
                                kb: "KnowledgeBase") -> Tuple[str, str]:
    """Construct a CITY CONTEXT block that DISAGREES with the mathematical
    explanation: it asserts a DIFFERENT, real corridor is the active bottleneck.

    The explanation shown to the model still points at the true top-k nodes; only
    the city context lies. This is a deliberate grounding stress test — if the LLM
    stays faithful to the math, its cited causes still resolve to the true top-k
    and hallucination stays ~0; if the false context pulls it off, it will cite the
    decoy corridor (which is NOT in the top-k) and hallucination spikes.

    Choosing the decoy HONESTLY: we pick the REAL corridor whose regions overlap
    the explanation's regions the LEAST (ideally not at all), so it is genuinely
    elsewhere, not a paraphrase of the truth. We fabricate no sensors or numbers —
    only the framing ("this corridor is the cause") is false, which is exactly the
    confound a planner might carry in from stale operational chatter. Returns
    (block, decoy_title).

    We compare FULL REGION LABELS, not word tokens: the KB region_tags are keyed to
    node_names.py region labels, so an exact-label overlap count is the right,
    noise-free measure. (An earlier token version spuriously matched on generic
    words like 'downtown' or 'i-405' and could pick the explanation's OWN region as
    the decoy — the opposite of what we want.)"""
    def _region_of(name: str) -> str:
        # node_name looks like "San Fernando Valley (sensor 772167, ...)"; the
        # region label is the part before the parenthetical.
        return name.split("(")[0].strip().lower()

    exp_regions = {_region_of(exp["prediction"]["node_name"])}
    for n in exp.get("top_nodes", []):
        exp_regions.add(_region_of(n["node_name"]))

    corridors = [c for c in kb.chunks
                 if str(c.get("id", "")).startswith("corridor_")]
    if not corridors:                        # some cities may not id chunks "corridor_"
        corridors = [c for c in kb.chunks if c.get("region_tags")]

    def _overlap(c: Dict[str, Any]) -> int:
        tags = {str(t).strip().lower() for t in c.get("region_tags", [])}
        return len(tags & exp_regions)

    # Fewest shared regions first; ties break by KB order (stable, reproducible).
    decoy = min(corridors, key=lambda c: (_overlap(c), corridors.index(c))) \
        if corridors else kb.chunks[-1]

    block = (
        "[FIELD-OPERATIONS BULLETIN — reported active bottleneck]\n"
        "Operations reports the dominant cause of the current network slowdown is "
        "the {title}. Prioritise mitigation on this corridor.\n{text}"
    ).format(title=decoy["title"], text=decoy["text"])
    return block, decoy["title"]


# ----------------------------------------------------------------------------
# Phase 17 — COUNTERFACTUAL narration (default-preserving addition, FLAGGED).
#
# Phases 3-5 explain WHY congestion is predicted. Phase 17 finds the MINIMUM speed
# uplift on the explainer's critical nodes that would have FLIPPED the target back
# to free flow (models/explainer/counterfactual.py), and this block lets the LLM
# narrate that counterfactual: "if the Glendale feeder had been eased ~8 min
# earlier, the cascade would not have reached Downtown."
#
# It reuses the SAME advisory output contract (reasoning / cited_causes /
# recommendations) so validate_advisory() and the Phase-5 faithfulness metric both
# apply UNCHANGED — the faithfulness of the narrative is then scored against the
# counterfactual's required-change node set (Phase-17 study). NOTHING here touches
# advise()/advise_condition() or their prompts, so Phase 4/5/15b/16 are byte-identical.
# ----------------------------------------------------------------------------
def render_counterfactual_text(cf: Dict[str, Any]) -> str:
    """Render the counterfactual record (models/explainer/counterfactual.py) as
    clean structured text for the narration prompt. Shows the target being
    prevented and, for each required change, current -> required speed with the
    delta, plus the implied lead time (the explanation's propagation lag)."""
    fac = cf["factual"]
    ctr = cf["counterfactual"]
    lines: List[str] = []
    lines.append("TARGET CONGESTION TO PREVENT:")
    lines.append("  Location: {}".format(fac["target_name"]))
    lines.append("  Predicted speed (factual): {} mph (congested; below the "
                 "{} mph free-flow threshold)".format(
                     fac["predicted_speed"], cf["meta"]["flip_threshold_mph"]))
    lines.append("")
    lines.append("MINIMUM UPSTREAM CHANGE THAT WOULD HAVE PREVENTED IT "
                 "(the counterfactual — ONLY these locations are valid causes):")
    for r in ctr["required_changes"]:
        lines.append(
            "  - {} (node_id {}): raise from {} mph to {} mph (+{} mph)".format(
                r["node_name"], r["node_id"], r["current_speed"],
                r["required_speed"], r["delta_mph"]))
    lines.append("")
    lines.append("  Total speed-uplift budget: {} mph across {} location(s).".format(
        ctr["total_perturbation_budget"], len(ctr["required_changes"])))
    lead = cf["meta"].get("implied_lead_minutes")
    if lead is not None:
        lines.append("  Implied lead time: the change would have had to happen "
                     "~{} minutes before the target congestion (the estimated "
                     "propagation lag from source to target).".format(lead))
    lines.append("  Predicted target speed AFTER this change: {} mph "
                 "(back above free-flow -> congestion prevented).".format(
                     ctr["predicted_speed_after"]))
    return "\n".join(lines)


_TASK_COUNTERFACTUAL = (
    "TASK:\n"
    "The minimum intervention that would have prevented this congestion is listed "
    "above. Explain in plain language what this means for a traffic planner and "
    "what specific actions could achieve this change. Reference ONLY the locations "
    "and quantities in the counterfactual above — do NOT invent sensors, roads, "
    "incidents, or numbers.\n"
    "Provide 3 to 5 concrete recommendations, each a way to achieve the required "
    "speed change at a counterfactual location, with a time window (minutes) and an "
    "expected effect.\n\n"
    "Return ONLY a JSON object with EXACTLY this shape (no prose outside JSON):\n"
    "{\n"
    '  "reasoning": "<plain-language explanation of the minimum intervention and '
    'what it means>",\n'
    '  "cited_causes": [\n'
    '    {"location": "<a location name from the counterfactual>", '
    '"resolved_node_id": <its node_id, or null>}\n'
    "  ],\n"
    '  "recommendations": [\n'
    '    {"action": "<a concrete action that would achieve the required speed '
    'change at this location>", "location": "<where>", '
    '"time_window_minutes": <int>, "expected_effect": "<result>", '
    '"grounded_in": ["<which counterfactual change this rests on>"]}\n'
    "  ]\n"
    "}"
)


def build_prompt_counterfactual(cf: Dict[str, Any], kb_block: str) -> str:
    """Assemble the counterfactual-narration prompt in the required order:
    SYSTEM -> CITY CONTEXT -> COUNTERFACTUAL -> TASK.

    We deliberately show the COUNTERFACTUAL (the required changes), NOT the full
    explanation top_nodes table, so the faithfulness measurement is clean: the only
    causes 'above' are the counterfactual's required-change nodes, so citing
    something else is genuinely off-counterfactual (measured as reduced precision),
    not just echoing a bigger table."""
    return (
        "{system}\n\n"
        "=== CITY CONTEXT (retrieved knowledge base) ===\n{kb}\n\n"
        "=== COUNTERFACTUAL (minimum intervention) ===\n{cf}\n\n"
        "=== {task}"
    ).format(system=_SYSTEM, kb=kb_block or "(no city context retrieved)",
             cf=render_counterfactual_text(cf), task=_TASK_COUNTERFACTUAL)


# ----------------------------------------------------------------------------
# Phase 18 — UNCERTAINTY-AWARE narration (default-preserving addition, FLAGGED).
#
# Phases 3-5 show the LLM a SINGLE explainer run's top-k. Phase 18 runs the
# explainer K times (models/explainer/uncertain_explainer.py) and splits the nodes
# into CORE (confident) vs PERIPHERAL (uncertain) causes. This block renders that
# distribution and instructs the LLM to be epistemically honest — assert core
# causes plainly, HEDGE peripheral ones, and never mention the dropped noise nodes.
#
# It reuses the SAME advisory output contract (reasoning / cited_causes /
# recommendations) so validate_advisory() and the Phase-5 faithfulness metric apply
# UNCHANGED. NOTHING here touches advise()/advise_condition()/their prompts, so
# Phase 4/5/15b/16/17 are byte-identical (verified). The ONLY difference from
# condition A is (a) the SYSTEM prompt gains an honesty clause and (b) the
# explanation is rendered as core/peripheral tiers instead of a flat top-k table.
# ----------------------------------------------------------------------------
_SYSTEM_UNCERTAIN = (
    _SYSTEM + " The explanation below separates CORE causes (high confidence — "
    "confirmed in most explainer runs) from PERIPHERAL causes (uncertain — seen in "
    "only some runs). Reflect this honestly: do NOT present uncertain (peripheral) "
    "causes with the same confidence as core ones. Assert core causes plainly; when "
    "you mention a peripheral cause, use explicit hedging language (e.g. 'may', "
    "'possibly', 'uncertain') so a planner knows it is less certain."
)


def render_uncertain_explanation_text(exp: Dict[str, Any]) -> str:
    """Render the Phase-3 prediction plus the Phase-18 core/peripheral tiers as
    clean structured text. Noise-tier nodes are intentionally omitted — the LLM is
    never shown them, so it cannot assert them."""
    p = exp["prediction"]
    u = exp["uncertainty"]
    lines: List[str] = []
    lines.append("TARGET PREDICTION:")
    lines.append("  Location: {} (node_id {})".format(p["node_name"], p["node_id"]))
    lines.append("  Current speed: {} mph".format(p["current_speed_mph"]))
    lines.append("  Predicted speed in {} min: {} mph".format(
        p["horizon_minutes"], p["predicted_speed_mph"]))
    lines.append("")
    lines.append("The explainer was run {} times. A cause's 'frequency' is how many "
                 "of those runs identified it as important.".format(u["k_runs"]))
    lines.append("")
    lines.append("CORE CAUSES (HIGH CONFIDENCE — confirmed in most runs; state these "
                 "plainly; ONLY these and the peripheral causes below are valid causes):")
    if u.get("core_causes"):
        for n in u["core_causes"]:
            lines.append("  - {} (node_id {}): confirmed in {} runs, currently "
                         "{} mph".format(n["node_name"], n["node_id"],
                                         n["frequency"], n["current_speed_mph"]))
    else:
        lines.append("  (none reached the high-confidence threshold)")
    lines.append("")
    lines.append("PERIPHERAL CAUSES (UNCERTAIN — appeared in only some runs; HEDGE "
                 "these, do not assert them with full confidence):")
    if u.get("peripheral_causes"):
        for n in u["peripheral_causes"]:
            lines.append("  - {} (node_id {}): appeared in only {} runs, currently "
                         "{} mph".format(n["node_name"], n["node_id"],
                                         n["frequency"], n["current_speed_mph"]))
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("EXPLANATION STABILITY: {:.2f} (0-1; mean agreement of the top "
                 "causes across the {} runs — higher = more trustworthy)."
                 .format(float(u["explanation_stability"]), u["k_runs"]))
    return "\n".join(lines)


_TASK_UNCERTAIN = (
    "TASK:\n"
    "1) Explain in plain language why this congestion is predicted, referencing the "
    "SPECIFIC locations in the explanation above. ASSERT the core (high-confidence) "
    "causes plainly. If you mention a peripheral (uncertain) cause, HEDGE it with "
    "explicit uncertainty language ('may', 'possibly', 'uncertain') and do not give "
    "it the same weight as a core cause. Do NOT invent causes or mention any location "
    "not listed above.\n"
    "2) Provide 3 to 5 interventions implementable within 15 minutes given the stated "
    "infrastructure constraints. Each needs a time window (minutes) and an expected "
    "effect. Prioritise acting on the CORE causes; treat peripheral causes as "
    "contingencies.\n\n"
    "Return ONLY a JSON object with EXACTLY this shape (no prose outside JSON):\n"
    "{\n"
    '  "reasoning": "<plain-language explanation; assert core causes, hedge '
    'peripheral ones>",\n'
    '  "cited_causes": [\n'
    '    {"location": "<a location name from the explanation>", '
    '"resolved_node_id": <the node_id from the explanation, or null>}\n'
    "  ],\n"
    '  "recommendations": [\n'
    '    {"action": "<what to do>", "location": "<where>", '
    '"time_window_minutes": <int>, "expected_effect": "<result>", '
    '"grounded_in": ["<which cited cause(s) or city-context fact this rests on>"]}\n'
    "  ]\n"
    "}"
)


def build_prompt_uncertain(exp: Dict[str, Any], kb_block: str) -> str:
    """Assemble the uncertainty-aware prompt in the required order:
    SYSTEM(+honesty clause) -> CITY CONTEXT -> UNCERTAINTY-AWARE EXPLANATION -> TASK."""
    return (
        "{system}\n\n"
        "=== CITY CONTEXT (retrieved knowledge base) ===\n{kb}\n\n"
        "=== MATHEMATICAL EXPLANATION (with per-cause confidence) ===\n{exp}\n\n"
        "=== {task}"
    ).format(system=_SYSTEM_UNCERTAIN, kb=kb_block or "(no city context retrieved)",
             exp=render_uncertain_explanation_text(exp), task=_TASK_UNCERTAIN)


# ----------------------------------------------------------------------------
# Config loading.
# ----------------------------------------------------------------------------
def load_advisor_config() -> Dict[str, Any]:
    path = os.path.join(PKG_ROOT, "configs", "advisor.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# The advisor itself.
# ----------------------------------------------------------------------------
class Advisor:
    """Wraps a local Ollama model. Reusable across many explanations (loads the
    KB + config once)."""

    def __init__(self, city: str, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg if cfg is not None else load_advisor_config()
        self.city = city
        self.kb: KnowledgeBase = load_kb_for_city(city, self.cfg)

        oll = self.cfg["ollama"]
        self.host: str = oll["host"].rstrip("/")
        self.model: str = oll["model"]
        self.temperature: float = float(oll.get("temperature", 0.1))
        self.timeout: float = float(oll.get("timeout_seconds", 180))
        self.force_json: bool = bool(oll.get("force_json", True))

        self.top_k: int = int(self.cfg["retrieval"]["top_k"])
        self.max_context_chars: int = int(self.cfg["retrieval"]["max_context_chars"])
        self.max_retries: int = int(self.cfg["retry"]["max_retries"])
        # Phase 15b — the C_RICH condition's fuller-context knobs. Default to
        # doubling the normal budgets if the config predates 15b, so an old
        # advisor.yaml still works.
        self.rich_top_k: int = int(
            self.cfg["retrieval"].get("rich_top_k", self.top_k * 2))
        self.rich_max_context_chars: int = int(
            self.cfg["retrieval"].get("rich_max_context_chars",
                                      self.max_context_chars * 2))

    # --- the raw Ollama call ------------------------------------------------
    def _call_ollama(self, prompt: str) -> str:
        """POST to Ollama /api/generate and return the raw response text.

        Raises RuntimeError with a friendly message if Ollama is unreachable or
        the model isn't pulled — those are the two things that actually go wrong
        on a fresh machine, and a clear message beats a stack trace."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.force_json:
            payload["format"] = "json"  # Ollama constrains decoding to valid JSON
        try:
            resp = requests.post("{}/api/generate".format(self.host),
                                 json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama at {}. Start it with `ollama serve` and "
                "pull the model with `ollama pull {}`.".format(self.host, self.model))
        if resp.status_code == 404:
            raise RuntimeError(
                "Model '{}' not found in Ollama. Pull it: `ollama pull {}`."
                .format(self.model, self.model))
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    # --- parse + retry ------------------------------------------------------
    def _generate_validated(self, prompt: str
                            ) -> Tuple[Dict[str, Any], List[str]]:
        """Call the model, parse JSON, validate against the advisory contract.
        On failure, re-ask up to max_retries times with the error appended — the
        model usually fixes its own mistake when told what was wrong. Returns
        (advisory_dict, raw_responses) — raw kept for the Phase-7 ablation log."""
        raws: List[str] = []
        current = prompt
        last_problems: List[str] = []
        for attempt in range(self.max_retries + 1):
            raw = self._call_ollama(current)
            raws.append(raw)
            try:
                adv = json.loads(raw)
            except json.JSONDecodeError as e:
                last_problems = ["response was not valid JSON: {}".format(e)]
                adv = None
            if adv is not None:
                last_problems = validate_advisory(adv)
                if not last_problems:
                    return adv, raws
            # Not valid — append the problem and try again (if retries remain).
            current = (
                prompt
                + "\n\nYOUR PREVIOUS ANSWER WAS INVALID:\n"
                + "\n".join("- " + p for p in last_problems)
                + "\nReturn ONLY the corrected JSON object, nothing else."
            )
        # Exhausted retries: surface a structured failure rather than crashing the
        # whole pipeline, so a batch run (Phase 5) can record it and move on.
        failed = {
            "reasoning": "",
            "cited_causes": [],
            "recommendations": [],
            "_error": "advisory failed validation after {} attempts: {}".format(
                self.max_retries + 1, "; ".join(last_problems)),
        }
        return failed, raws

    # --- public API ---------------------------------------------------------
    def advise(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Full Layer-3 step: retrieve city context -> build prompt -> LLM ->
        validated advisory. Returns a dict with the advisory plus provenance
        (which KB chunks and model were used) for the ablation/faithfulness logs."""
        chunks = self.kb.retrieve(exp, top_k=self.top_k)
        kb_block = render_kb_block(chunks, self.max_context_chars)
        prompt = build_prompt(exp, kb_block)
        advisory, raws = self._generate_validated(prompt)
        return {
            "advisory": advisory,
            "context_used": [c["title"] for c in chunks],
            "model": self.model,
            "raw_responses": raws,   # Phase 7 ablation study logs these
        }

    def advise_condition(self, exp: Dict[str, Any],
                         condition: str,
                         extra_instruction: Optional[str] = None) -> Dict[str, Any]:
        """Run one advisory under a Phase-5/15b condition. Same output shape as
        advise() plus the condition tag, so the faithfulness study can score every
        condition identically.

        The ONLY thing that varies across A / C_RICH / D_CONTRA is the CITY CONTEXT
        block (normal / fuller / contradictory); B swaps the explanation for a
        prediction-only view; C blanks the context. We build the right context
        here, then hand a single kb_block to build_prompt_condition.

          A        : normal retrieval against the explanation.
          B        : retrieve against a PREDICTION-ONLY view so the explanation's
                     top_nodes cannot leak into the prompt via the KB — B must be
                     blind to the explanation.
          C        : no city context at all.
          C_RICH   : fuller retrieval (rich_top_k, padded past the relevant chunks)
                     rendered with a bigger character budget.
          D_CONTRA : a contradictory context block naming a different real corridor
                     as the bottleneck (build_contradictory_context).

        Phase 16 (FLAGGED, default-preserving): `extra_instruction`, when given, is
        appended after the TASK by build_prompt_condition — the Active Grounding
        Loop uses it to attach a correction naming the missed top-k nodes. Default
        None reproduces the exact Phase-4/5/15b prompt, so callers that don't pass
        it are unaffected."""
        context_used: List[str] = []
        if condition == "A":
            chunks = self.kb.retrieve(exp, top_k=self.top_k)
            kb_block = render_kb_block(chunks, self.max_context_chars)
            context_used = [c["title"] for c in chunks]
        elif condition == "B":
            stripped = {"prediction": exp["prediction"], "top_nodes": []}
            chunks = self.kb.retrieve(stripped, top_k=self.top_k)
            kb_block = render_kb_block(chunks, self.max_context_chars)
            context_used = [c["title"] for c in chunks]
        elif condition == "C":
            kb_block = ""                    # no city context in condition C
        elif condition == "C_RICH":          # Phase 15b — fuller context
            chunks = self.kb.retrieve(exp, top_k=self.rich_top_k, pad_to_k=True)
            kb_block = render_kb_block(chunks, self.rich_max_context_chars)
            context_used = [c["title"] for c in chunks]
        elif condition == "D_CONTRA":        # Phase 15b — contradictory context
            kb_block, decoy_title = build_contradictory_context(exp, self.kb)
            context_used = ["CONTRADICTORY:" + decoy_title]
        else:
            raise ValueError(
                "condition must be one of A/B/C/C_RICH/D_CONTRA, got {!r}"
                .format(condition))

        prompt = build_prompt_condition(exp, kb_block, condition,
                                        extra_instruction=extra_instruction)
        advisory, raws = self._generate_validated(prompt)
        return {
            "advisory": advisory,
            "condition": condition,
            "context_used": context_used,
            "model": self.model,
            "raw_responses": raws,
        }

    def advise_counterfactual(self, exp: Dict[str, Any],
                              cf: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 17: narrate a COUNTERFACTUAL. Given the Phase-3 explanation `exp`
        (used only to retrieve relevant city context) and a counterfactual record
        `cf` (models/explainer/counterfactual.py), ask the LLM to explain the
        minimum intervention in plain language and propose concrete actions.

        Same output shape as advise()/advise_condition() so validate_advisory() and
        the Phase-5 faithfulness metric apply unchanged — the study then scores the
        narrative's cited causes against the counterfactual's required-change nodes."""
        chunks = self.kb.retrieve(exp, top_k=self.top_k)
        kb_block = render_kb_block(chunks, self.max_context_chars)
        prompt = build_prompt_counterfactual(cf, kb_block)
        advisory, raws = self._generate_validated(prompt)
        return {
            "advisory": advisory,
            "mode": "counterfactual",
            "context_used": [c["title"] for c in chunks],
            "model": self.model,
            "raw_responses": raws,
        }

    def advise_uncertain(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 18: run the advisory with UNCERTAINTY framing. `exp` must carry an
        `uncertainty` block (from uncertain_explainer.attach_uncertainty) with
        core_causes / peripheral_causes. The LLM is told which causes are confident
        (core) vs uncertain (peripheral) and instructed to hedge the latter.

        Same output shape as advise()/advise_condition() so validate_advisory() and
        the Phase-5 faithfulness metric apply unchanged. City context is retrieved
        exactly as in condition A (against the same explanation), so the ONLY
        difference from A is the honesty framing + the tiered rendering — which is
        precisely the variable the Phase-18 study isolates."""
        chunks = self.kb.retrieve(exp, top_k=self.top_k)
        kb_block = render_kb_block(chunks, self.max_context_chars)
        prompt = build_prompt_uncertain(exp, kb_block)
        advisory, raws = self._generate_validated(prompt)
        return {
            "advisory": advisory,
            "mode": "uncertain",
            "context_used": [c["title"] for c in chunks],
            "model": self.model,
            "raw_responses": raws,
        }
