"""The explanation JSON schema — a CONTRACT the Phase 4 LLM layer depends on.

WHY a hard schema: Layer 3 (the LLM advisor) is only allowed to cite causes that
appear in the mathematical explanation. If the shape of that explanation drifts,
the LLM prompt breaks and the faithfulness metric (Phase 5) becomes meaningless.
So we pin the exact keys here and validate every explanation against them before
it ever leaves the explainer. Treat this file as the interface, not an
implementation detail.

The schema (all fields required):
{
  "meta": {"city": str, "timestamp": str, "model_checkpoint": str},
  "prediction": {"node_id": int, "node_name": str,
                 "predicted_speed_mph": float, "horizon_minutes": int,
                 "current_speed_mph": float},
  "top_nodes": [{"node_id": int, "node_name": str, "importance": float,
                 "current_speed_mph": float}],
  "top_edges": [{"from_id": int, "to_id": int, "importance": float}],
  "propagation_path": [int],
  "propagation_lag_minutes": float,
  "explanation_confidence": float
}

Python 3.9 compatible (typing.Dict/List, no `X | Y`).
"""
from __future__ import annotations

from typing import Any, Dict, List

# Each top-level key -> the Python type(s) it must hold. Lists are validated
# element-by-element against the *_ITEM specs below.
_TOP_LEVEL = {
    "meta": dict,
    "prediction": dict,
    "top_nodes": list,
    "top_edges": list,
    "propagation_path": list,
    "propagation_lag_minutes": (int, float),
    "explanation_confidence": (int, float),
}

_META_KEYS = {"city": str, "timestamp": str, "model_checkpoint": str}

_PREDICTION_KEYS = {
    "node_id": int, "node_name": str, "predicted_speed_mph": (int, float),
    "horizon_minutes": int, "current_speed_mph": (int, float),
}

_TOP_NODE_KEYS = {
    "node_id": int, "node_name": str, "importance": (int, float),
    "current_speed_mph": (int, float),
}

_TOP_EDGE_KEYS = {
    "from_id": int, "to_id": int, "importance": (int, float),
}


def _check_dict(name: str, d: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems (empty == valid)."""
    errs: List[str] = []
    if not isinstance(d, dict):
        return [f"{name}: expected dict, got {type(d).__name__}"]
    for key, typ in spec.items():
        if key not in d:
            errs.append(f"{name}.{key}: missing")
        elif not isinstance(d[key], typ) or isinstance(d[key], bool):
            # bool is a subclass of int in Python — reject it explicitly so a
            # stray True/False can't masquerade as a node_id.
            got = type(d[key]).__name__
            errs.append(f"{name}.{key}: expected {typ}, got {got}")
    return errs


def validate_explanation(exp: Dict[str, Any]) -> List[str]:
    """Validate an explanation dict against the contract. Returns a list of
    problems; an empty list means the explanation is well-formed. We RETURN the
    problems (instead of raising) so callers can log them and decide — the
    explainer wants to warn-and-continue, tests want to assert emptiness."""
    errs: List[str] = []
    for key, typ in _TOP_LEVEL.items():
        if key not in exp:
            errs.append(f"top-level.{key}: missing")
        elif not isinstance(exp[key], typ):
            errs.append(f"top-level.{key}: expected {typ}, got {type(exp[key]).__name__}")
    if errs:
        return errs  # shape is broken; deeper checks would just be noise

    errs += _check_dict("meta", exp["meta"], _META_KEYS)
    errs += _check_dict("prediction", exp["prediction"], _PREDICTION_KEYS)
    for i, node in enumerate(exp["top_nodes"]):
        errs += _check_dict(f"top_nodes[{i}]", node, _TOP_NODE_KEYS)
    for i, edge in enumerate(exp["top_edges"]):
        errs += _check_dict(f"top_edges[{i}]", edge, _TOP_EDGE_KEYS)
    for i, nid in enumerate(exp["propagation_path"]):
        if not isinstance(nid, int) or isinstance(nid, bool):
            errs.append(f"propagation_path[{i}]: expected int node_id")
    return errs


def assert_valid(exp: Dict[str, Any]) -> None:
    """Raise ValueError if the explanation violates the contract (used in tests)."""
    problems = validate_explanation(exp)
    if problems:
        raise ValueError("Explanation failed schema validation:\n  - "
                         + "\n  - ".join(problems))
