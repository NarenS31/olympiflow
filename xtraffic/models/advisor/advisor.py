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
                           condition: str) -> str:
    """Assemble the prompt for condition 'A', 'B', or 'C'."""
    if condition == "A":
        return build_prompt(exp, kb_block)
    if condition == "C":
        # Same as A but with the city context deliberately blanked out.
        return build_prompt(exp, "")
    if condition == "B":
        return (
            "{system}\n\n"
            "=== CITY CONTEXT (retrieved knowledge base) ===\n{kb}\n\n"
            "=== PREDICTION ===\n{pred}\n\n"
            "=== {task}"
        ).format(system=_SYSTEM,
                 kb=kb_block or "(no city context retrieved)",
                 pred=render_prediction_only_text(exp),
                 task=_TASK_NO_EXPLANATION)
    raise ValueError("condition must be 'A', 'B', or 'C', got {!r}".format(condition))


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
                         condition: str) -> Dict[str, Any]:
        """Run one advisory under Phase-5 condition A/B/C. Same output shape as
        advise() plus the condition tag, so the faithfulness study can score all
        three identically.

        City context is retrieved for A and B (both include CITY CONTEXT). For B
        we retrieve against a PREDICTION-ONLY view so the explanation's top_nodes
        cannot leak into the prompt through the KB retrieval — B must be blind to
        the explanation. C gets no city context at all."""
        if condition == "A":
            chunks = self.kb.retrieve(exp, top_k=self.top_k)
        elif condition == "B":
            stripped = {"prediction": exp["prediction"], "top_nodes": []}
            chunks = self.kb.retrieve(stripped, top_k=self.top_k)
        elif condition == "C":
            chunks = []                      # no city context in condition C
        else:
            raise ValueError("condition must be 'A', 'B', or 'C'")

        kb_block = render_kb_block(chunks, self.max_context_chars)
        prompt = build_prompt_condition(exp, kb_block, condition)
        advisory, raws = self._generate_validated(prompt)
        return {
            "advisory": advisory,
            "condition": condition,
            "context_used": [c["title"] for c in chunks],
            "model": self.model,
            "raw_responses": raws,
        }
