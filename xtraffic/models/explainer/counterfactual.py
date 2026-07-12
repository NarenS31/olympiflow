"""Phase 17 — Counterfactual Explanations (Layer 2, actionable variant).

WHY THIS EXISTS
---------------
Phases 3-5 explain WHY congestion is predicted. A planner's real question is the
next one: "what could I have done differently?" This phase answers it. Instead of
attributing the prediction, we find the MINIMUM speed uplift on the explainer's
critical (top-k) UPSTREAM nodes that FLIPS the target's 30-min prediction from
congested back to free flow — then let the LLM narrate that counterfactual:

    "If signal timing on the Glendale feeder had been eased ~5 minutes earlier,
     the cascade would not have reached NE Downtown."

THE SEARCH (gradient-free — the METHOD, documented because reviewers WILL ask
exactly how "minimum" is defined and searched)
-----------------------------------------------------------------------------
Candidate levers = the explainer's top-k nodes that are CONGESTED (1 < current
speed < free-flow). We do NOT perturb already free-flowing critical nodes — you
cannot "ease" a road that is already at free-flow, and clipping would make it a
no-op anyway. If no critical node is congested, there is nothing to ease and the
counterfactual is reported infeasible (the free-flow-cause regime from Phase 13).

  1. FEASIBILITY SWEEP. Raise every lever by a single uniform uplift
     u = step, 2*step, ... up to max_uplift, re-running the frozen GNN after each
     step, until the target's predicted speed crosses the free-flow threshold. The
     first u that flips is the coarse feasible perturbation. If none flips by
     max_uplift, the counterfactual is INFEASIBLE (flip_achieved=False) — an
     honest, reportable outcome for a deeply-congested target.
  2. PER-NODE MINIMISATION (greedy coordinate relaxation). From that uniform level,
     visit the levers LEAST-IMPORTANT FIRST and relax each one's uplift back toward
     0 (in `step` decrements), keeping the SMALLEST value that still keeps the
     target flipped given the others. This turns a blanket uplift into the genuinely
     MINIMAL per-node change — "you only needed to ease THESE nodes by THESE
     amounts" — which is what makes the counterfactual actionable. It is greedy, so
     it finds a locally-minimal (not provably global-minimal) perturbation; we say
     so in the paper.

MECHANISM REUSE (one notion of "what an action does" across the project)
------------------------------------------------------------------------
The perturbation is the SAME family as the Phase-8/10 intervention simulator: a
speed uplift added to the last `apply_steps` input timesteps of a node's speed
channel, clipped at free-flow. The uplift is applied to the INPUT window and read
out 30 min later through the frozen model, so easing an UPSTREAM source can relieve
the DOWNSTREAM target exactly the way congestion propagates.

OUTPUT (the schema the assignment asked for, plus a small `meta` block the LLM
narration and the study use)
------------------------------------------------------------------------------
  {
    "factual":        {"predicted_speed", "target_name", "critical_nodes":[...]},
    "counterfactual": {"required_changes":[{node_id,node_name,current_speed,
                        required_speed,delta_mph}], "predicted_speed_after",
                        "total_perturbation_budget", "flip_achieved"},
    "meta":           {flip_threshold_mph, free_flow_mph, implied_lead_minutes,
                        reason, dataset, sample_index, target_node, ...}
  }

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import yaml

from ...utils.io_utils import PKG_ROOT
from ..gnn.loaders import build_modality_dict
from .explain import SPEED_CHANNEL


# ===========================================================================
# Config.
# ===========================================================================
def load_counterfactual_config() -> Dict[str, Any]:
    with open(os.path.join(PKG_ROOT, "configs", "counterfactual.yaml")) as f:
        return yaml.safe_load(f)


# ===========================================================================
# The searcher — reuses an already-loaded frozen model/scaler/cfg (e.g. from an
# ExplanationBuilder) so we never load the checkpoint twice.
# ===========================================================================
class CounterfactualSearcher:
    """Find the minimum flip-to-free-flow perturbation for one prediction.

    The model is FROZEN — we only perturb the INPUT window and read the output.
    All knobs come from configs/counterfactual.yaml's `search` block (no magic
    numbers in code, CLAUDE.md)."""

    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any],
                 scaler: Dict[str, float], search_cfg: Dict[str, Any],
                 device: torch.device):
        self.model = model.eval()
        self.device = device
        self.mean = float(scaler["mean"])
        self.std = float(scaler["std"])
        self.traffic_channels = cfg["traffic_channels"]
        self.modality_names = list(cfg["modalities"].keys())

        self.horizon_step = int(search_cfg["horizon_step"])   # 1-based
        self.flip_threshold = float(search_cfg["flip_threshold_mph"])
        self.free_flow = float(search_cfg["free_flow_mph"])
        self.step = float(search_cfg["step_mph"])
        self.max_uplift = float(search_cfg["max_uplift_mph"])
        self.apply_steps = int(search_cfg["apply_steps"])
        self.top_k = int(search_cfg["top_k"])

    # -- frozen forward -----------------------------------------------------
    def predict_target_mph(self, X: torch.Tensor, target_node: int) -> float:
        """Target's predicted speed (real mph) at the scored horizon on window X.

        This is the LIVE model's prediction — the single source of truth for "is
        the target congested". Callers filter congestion on THIS, not on a cached
        explanation's stored speed, so a stale cache can never drive the search."""
        h = self.horizon_step - 1                              # 0-based time index
        with torch.no_grad():
            mods = build_modality_dict(X.to(self.device), self.traffic_channels,
                                       self.modality_names)
            pred_z = float(self.model(mods)[0, h, target_node].item())     # z-scored
        return pred_z * self.std + self.mean                   # -> real mph

    # -- perturbation (same family as the Phase-10 simulator) ---------------
    def _apply_uplift(self, X: torch.Tensor,
                      node_deltas: Dict[int, float]) -> torch.Tensor:
        """Return a COPY of X with each node raised by its uplift (mph) on the last
        `apply_steps` timesteps of the speed channel, clipped at free-flow and never
        lowered (a positive uplift can only speed a segment up, up to free-flow)."""
        Xm = X.clone()                                         # [1, T, N, C]
        if not node_deltas:
            return Xm
        t0 = X.shape[1] - self.apply_steps                     # first touched step
        for nid, delta in node_deltas.items():
            if delta <= 0:
                continue
            seg_z = Xm[0, t0:, nid, SPEED_CHANNEL]             # [apply_steps] z-scored
            seg_mph = seg_z * self.std + self.mean             # -> real mph
            raised = torch.clamp(seg_mph + delta, max=self.free_flow)   # add + cap
            raised = torch.maximum(raised, seg_mph)            # never lower a value
            Xm[0, t0:, nid, SPEED_CHANNEL] = (raised - self.mean) / self.std
        return Xm

    # -- lever selection ----------------------------------------------------
    def _levers(self, target: Dict[str, Any],
                critical: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The candidate nodes the search may ease. = congested top-k critical
        nodes PLUS the target bottleneck itself.

        WHY THE TARGET IS INCLUDED (documented design decision — CLAUDE.md):
        empirically (see the Phase-17 calibration in the lab notebook, and the
        Phase-3 Fidelity- result) this ST-GNN's 30-min forecast is dominated by the
        TARGET's own recent speed — easing only the upstream top-k barely moves the
        prediction, so an upstream-only counterfactual is almost never feasible. A
        minimum-intervention counterfactual must be allowed to act AT the bottleneck
        (the standard, most actionable lever: signal retiming / incident clearance =
        the Phase-8/10 target-scope action). Upstream critical nodes STAY in the
        candidate set, so where propagation genuinely carries leverage the greedy
        search still uses them; the study reports how often upstream actually
        contributed vs the target alone. Only GENUINE congested readings
        (1 < speed < free-flow) are levers — you cannot ease a free-flow or a
        missing-sensor node."""
        levers = [dict(c, is_target=False) for c in critical[: self.top_k]
                  if 1.0 < float(c["current_speed_mph"]) < self.free_flow]
        tgt_cur = float(target["current_speed_mph"])
        if 1.0 < tgt_cur < self.free_flow:
            # importance 1.0 -> relaxed LAST in the greedy pass (it is the strongest
            # lever, so it is the one most likely to be kept).
            levers.append({"node_id": int(target["node_id"]),
                           "node_name": target["node_name"],
                           "current_speed_mph": tgt_cur,
                           "importance": 1.0, "is_target": True})
        return levers

    # -- public API ---------------------------------------------------------
    def search(self, X: torch.Tensor, target: Dict[str, Any],
               critical: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Search for the minimum perturbation that flips the target to free flow.

        X        : [1, T, N, C] z-scored input window for this scenario.
        target   : the congested target dict {node_id, node_name, current_speed_mph}
                   (from the explanation's `prediction` block).
        critical : the explainer's top-k nodes, each a dict with at least
                   {node_id, node_name, current_speed_mph, importance}.

        Returns the search result (consumed by build_counterfactual)."""
        target_node = int(target["node_id"])
        pred_before = self.predict_target_mph(X, target_node)

        levers = self._levers(target, critical)
        base = {"predicted_speed_before": pred_before, "levers": levers}

        # Guard: the LIVE model does not predict congestion here -> no counterfactual
        # needed (a free-flow target has nothing to flip). This also prevents a
        # degenerate "flip with 0 budget" if a caller passes a non-congested target.
        if pred_before >= self.flip_threshold:
            return {**base, "flip_achieved": False, "reason": "target_not_congested",
                    "deltas": {}, "predicted_speed_after": pred_before}

        if not levers:
            return {**base, "flip_achieved": False, "reason": "no_congested_causes",
                    "deltas": {}, "predicted_speed_after": pred_before}

        node_ids = [int(c["node_id"]) for c in levers]

        # --- Stage 1: uniform feasibility sweep --------------------------------
        flip_u: Optional[float] = None
        pred_at_max = pred_before
        u = self.step
        while u <= self.max_uplift + 1e-9:
            pred = self.predict_target_mph(
                self._apply_uplift(X, {nid: u for nid in node_ids}), target_node)
            pred_at_max = pred                                 # last (largest) uplift tried
            if pred >= self.flip_threshold:
                flip_u = u
                break
            u += self.step

        if flip_u is None:
            # Infeasible within budget: report the best (max-uplift) attempt honestly.
            return {**base, "flip_achieved": False, "reason": "infeasible_within_budget",
                    "deltas": {}, "predicted_speed_after": pred_at_max}

        # --- Stage 2: greedy per-node minimisation -----------------------------
        # Start all levers at the uniform flip level, then relax each toward 0
        # (least-important first — the cheapest contribution to give up).
        deltas: Dict[int, float] = {nid: flip_u for nid in node_ids}
        for c in sorted(levers, key=lambda c: float(c["importance"])):
            nid = int(c["node_id"])
            chosen = deltas[nid]                               # keep current if none smaller works
            v = 0.0
            while v <= deltas[nid] + 1e-9:                     # try smallest first
                trial = dict(deltas)
                trial[nid] = v
                pred = self.predict_target_mph(self._apply_uplift(X, trial), target_node)
                if pred >= self.flip_threshold:
                    chosen = v
                    break
                v += self.step
            deltas[nid] = chosen

        pred_after = self.predict_target_mph(self._apply_uplift(X, deltas), target_node)
        return {**base, "flip_achieved": True, "reason": "flip",
                "deltas": deltas, "predicted_speed_after": pred_after}


# ===========================================================================
# Assemble the counterfactual record (the output schema).
# ===========================================================================
def _round(x: float, n: int = 2) -> float:
    return round(float(x), n)


def build_counterfactual(exp: Dict[str, Any], result: Dict[str, Any],
                         search_cfg: Dict[str, Any],
                         extra_meta: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """Turn a search result into the counterfactual JSON record.

    `exp` is the Phase-3 explanation (for names, top-k, and the propagation lag =
    the implied lead time). `result` comes from CounterfactualSearcher.search."""
    pred = exp["prediction"]
    top_nodes = exp.get("top_nodes", [])[: int(search_cfg["top_k"])]

    # factual.critical_nodes = the full top-k (the causal frontier the search ranged
    # over), as the assignment's schema asks.
    critical_nodes = [{
        "node_id": int(n["node_id"]),
        "node_name": n["node_name"],
        "current_speed": _round(n["current_speed_mph"]),
    } for n in top_nodes]

    # counterfactual.required_changes = the levers the minimisation left non-zero.
    # Metadata comes from the search's OWN lever list (which includes the target
    # bottleneck, not just the top-k), so a required change at the target resolves.
    deltas: Dict[int, float] = result.get("deltas", {}) or {}
    by_id = {int(n["node_id"]): n for n in result.get("levers", [])}
    required_changes: List[Dict[str, Any]] = []
    for nid, u in deltas.items():
        if u <= 1e-9:
            continue                                          # relaxed away -> not required
        node = by_id.get(int(nid))
        if node is None:
            continue
        cur = float(node["current_speed_mph"])
        req = min(cur + float(u), float(search_cfg["free_flow_mph"]))
        required_changes.append({
            "node_id": int(nid),
            "node_name": node["node_name"],
            "current_speed": _round(cur),
            "required_speed": _round(req),
            "delta_mph": _round(req - cur),
            "is_target": bool(node.get("is_target", False)),
        })
    # Sort strongest-lever first (largest delta) so the narrative leads with it.
    required_changes.sort(key=lambda r: -r["delta_mph"])
    budget = _round(sum(r["delta_mph"] for r in required_changes))

    cf = {
        "factual": {
            "predicted_speed": _round(result["predicted_speed_before"]),
            "target_name": pred["node_name"],
            "critical_nodes": critical_nodes,
        },
        "counterfactual": {
            "required_changes": required_changes,
            "predicted_speed_after": _round(result["predicted_speed_after"]),
            "total_perturbation_budget": budget,
            "flip_achieved": bool(result["flip_achieved"]),
        },
        "meta": {
            "flip_threshold_mph": float(search_cfg["flip_threshold_mph"]),
            "free_flow_mph": float(search_cfg["free_flow_mph"]),
            "max_uplift_mph": float(search_cfg["max_uplift_mph"]),
            "step_mph": float(search_cfg["step_mph"]),
            "implied_lead_minutes": exp.get("propagation_lag_minutes"),
            "reason": result.get("reason", ""),
            "target_node": int(pred["node_id"]),
            "n_critical": len(critical_nodes),
            "n_levers": len(result.get("levers", [])),
        },
    }
    if extra_meta:
        cf["meta"].update(extra_meta)
    return cf


def validate_counterfactual(cf: Dict[str, Any]) -> List[str]:
    """Light structural check on the counterfactual record (the core schema the
    assignment specified). Returns a list of problems (empty == valid)."""
    errs: List[str] = []
    if not isinstance(cf, dict):
        return ["counterfactual: expected object"]
    fac, ctr = cf.get("factual"), cf.get("counterfactual")
    if not isinstance(fac, dict):
        errs.append("factual: expected object")
    else:
        if not isinstance(fac.get("predicted_speed"), (int, float)):
            errs.append("factual.predicted_speed: expected number")
        if not isinstance(fac.get("critical_nodes"), list):
            errs.append("factual.critical_nodes: expected list")
    if not isinstance(ctr, dict):
        errs.append("counterfactual: expected object")
    else:
        if not isinstance(ctr.get("flip_achieved"), bool):
            errs.append("counterfactual.flip_achieved: expected bool")
        if not isinstance(ctr.get("predicted_speed_after"), (int, float)):
            errs.append("counterfactual.predicted_speed_after: expected number")
        if not isinstance(ctr.get("total_perturbation_budget"), (int, float)):
            errs.append("counterfactual.total_perturbation_budget: expected number")
        rc = ctr.get("required_changes")
        if not isinstance(rc, list):
            errs.append("counterfactual.required_changes: expected list")
        else:
            for i, r in enumerate(rc):
                for key in ("node_id", "current_speed", "required_speed", "delta_mph"):
                    if key not in r:
                        errs.append("required_changes[{}].{}: missing".format(i, key))
    return errs


# ===========================================================================
# Faithfulness reference: score the LLM narration against the COUNTERFACTUAL's
# required-change nodes using the Phase-5 metric UNCHANGED. We do this by handing
# score_advisory an explanation-shaped dict whose top_nodes ARE the required
# changes — the metric reads top_nodes ids for its reference set, so no new notion
# of "faithful" is invented (same as Phase 16 reuses the exact metric).
# ===========================================================================
def counterfactual_reference_explanation(exp: Dict[str, Any],
                                         cf: Dict[str, Any]) -> Dict[str, Any]:
    """A shallow copy of `exp` with top_nodes replaced by the counterfactual's
    required-change nodes, so evaluation.faithfulness.score_advisory scores the
    narrative against exactly the nodes the counterfactual says must change."""
    ref = dict(exp)
    ref["top_nodes"] = [{
        "node_id": int(r["node_id"]),
        "node_name": r["node_name"],
        "importance": 1.0,                                    # unused by the metric
        "current_speed_mph": float(r["current_speed"]),
    } for r in cf["counterfactual"]["required_changes"]]
    return ref


# ===========================================================================
# Pretty-print the full chain (prediction -> explanation -> counterfactual ->
# LLM narrative) for a single scenario — used by the demo + the study smoke.
# ===========================================================================
def _region(name: str) -> str:
    return name.split(" (sensor")[0].split(" (segment")[0].strip()


def format_full_chain(exp: Dict[str, Any], cf: Dict[str, Any],
                      narration: Optional[Dict[str, Any]]) -> str:
    """Human-readable dump of the whole chain for one scenario."""
    p = exp["prediction"]
    L: List[str] = []
    L.append("=" * 74)
    L.append("FULL CHAIN — {}".format(exp["meta"]["timestamp"]))
    L.append("=" * 74)

    L.append("\n[1] PREDICTION (Layer 1 ST-GNN)")
    L.append("    Target: {}".format(p["node_name"]))
    L.append("    Current {} mph -> predicted {} mph in {} min  "
             "({})".format(p["current_speed_mph"], p["predicted_speed_mph"],
                           p["horizon_minutes"],
                           "CONGESTED" if p["predicted_speed_mph"]
                           < cf["meta"]["flip_threshold_mph"] else "free-flow"))

    L.append("\n[2] EXPLANATION (Layer 2 GNNExplainer) — top critical sources")
    for n in exp.get("top_nodes", [])[:6]:
        L.append("    - {:28s} importance {:.3f}  currently {} mph".format(
            _region(n["node_name"]), float(n["importance"]), n["current_speed_mph"]))
    L.append("    propagation lag (source->target): {} min | confidence {:.2f}".format(
        exp["propagation_lag_minutes"], exp["explanation_confidence"]))

    L.append("\n[3] COUNTERFACTUAL (Layer 2 — minimum flip-to-free-flow perturbation)")
    ctr = cf["counterfactual"]
    if not ctr["flip_achieved"]:
        L.append("    flip_achieved = False  ({})".format(cf["meta"]["reason"]))
        L.append("    -> no small upstream easing flips this target within budget.")
    else:
        L.append("    Minimum required upstream changes:")
        for r in ctr["required_changes"]:
            L.append("    - {:28s} {} -> {} mph  (+{} mph)".format(
                _region(r["node_name"]), r["current_speed"],
                r["required_speed"], r["delta_mph"]))
        L.append("    total budget {} mph over {} node(s); implied lead ~{} min".format(
            ctr["total_perturbation_budget"], len(ctr["required_changes"]),
            cf["meta"]["implied_lead_minutes"]))
        L.append("    target prediction AFTER change: {} -> {} mph "
                 "(flipped above {} mph).".format(
                     cf["factual"]["predicted_speed"], ctr["predicted_speed_after"],
                     cf["meta"]["flip_threshold_mph"]))

    L.append("\n[4] LLM NARRATIVE (Layer 3 advisory — counterfactual mode)")
    if narration is None:
        L.append("    (not generated)")
    else:
        adv = narration.get("advisory", narration)
        if adv.get("_error"):
            L.append("    ERROR: {}".format(adv["_error"]))
        L.append("    reasoning: {}".format(adv.get("reasoning", "")))
        cites = adv.get("cited_causes", []) or []
        L.append("    cited causes: {}".format(
            ", ".join(_region(c.get("location", "")) for c in cites) or "(none)"))
        for rec in (adv.get("recommendations", []) or [])[:3]:
            L.append("    rec: {} @ {} ({} min) -> {}".format(
                rec.get("action", ""), rec.get("location", ""),
                rec.get("time_window_minutes", "?"), rec.get("expected_effect", "")))
    return "\n".join(L)


# ===========================================================================
# Single-scenario demo (the "show me the full chain for one scenario" request).
#   python -m xtraffic.models.explainer.counterfactual              # first congested cached scenario
#   python -m xtraffic.models.explainer.counterfactual --no-llm     # skip the narration
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 17 counterfactual — one-scenario demo")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--city", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default=None, help="override the Ollama advisor model")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM narration")
    args = ap.parse_args()

    cfg = load_counterfactual_config()
    dataset = args.dataset or cfg["dataset"]
    city = args.city or cfg["city"]
    checkpoint = args.checkpoint or cfg["checkpoint"]
    search_cfg = cfg["search"]

    device = torch.device("cpu")
    # Lazy imports (mirrors the rest of the package): keep module import light.
    from .explain import ExplanationBuilder
    from .scenarios import load_window
    from ...evaluation.run_faithfulness_study import (
        build_or_load_explanation, sample_scenarios)

    builder = ExplanationBuilder(checkpoint, dataset, device=device)
    searcher = CounterfactualSearcher(builder.model, builder.cfg, builder.scaler,
                                      search_cfg, device)

    # Pick a scenario that is CONGESTED on the LIVE model (not the cached speed —
    # a stale cache can disagree) and that actually FLIPS, so the demo shows the whole
    # working chain. Fall back to the first congested-by-live one (honest infeasible)
    # if none flip in the scanned pool.
    scenarios = sample_scenarios(dataset, builder.scaler, cfg["study"]["per_stratum"])
    flip_thr = float(search_cfg["flip_threshold_mph"])
    chosen = None            # (sc, exp, X, result)
    fallback = None
    for sc in scenarios:
        X = load_window(dataset, sc["sample_index"])
        if searcher.predict_target_mph(X, sc["target_node"]) >= flip_thr:
            continue                                          # free-flow -> no counterfactual
        exp = build_or_load_explanation(builder, dataset, sc)
        result = searcher.search(X, exp["prediction"], exp.get("top_nodes", []))
        if fallback is None:
            fallback = (sc, exp, X, result)
        if result["flip_achieved"]:
            chosen = (sc, exp, X, result)
            break
    if chosen is None:
        chosen = fallback
    if chosen is None:
        print("No congested scenario found in the sampled pool.")
        return
    sc, exp, X, result = chosen
    cf = build_counterfactual(exp, result, search_cfg,
                              extra_meta={"dataset": dataset,
                                          "sample_index": sc["sample_index"]})
    problems = validate_counterfactual(cf)
    if problems:
        print("[counterfactual] WARNING schema problems:", problems)

    narration = None
    if not args.no_llm and cf["counterfactual"]["flip_achieved"]:
        from ..advisor.advisor import Advisor, load_advisor_config
        adv_cfg = load_advisor_config()
        if args.model:
            import copy
            adv_cfg = copy.deepcopy(adv_cfg)
            adv_cfg.setdefault("ollama", {})["model"] = args.model
        advisor = Advisor(city, cfg=adv_cfg)
        narration = advisor.advise_counterfactual(exp, cf)

    print(format_full_chain(exp, cf, narration))
    print("\nCounterfactual record:")
    print(json.dumps(cf, indent=2))


if __name__ == "__main__":
    main()
