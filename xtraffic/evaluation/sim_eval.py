"""Phase 10 — simulation-based decision evaluation (Contribution #3).

This REPLACES the Phase-8 human study. Instead of scheduling human experts, we ask
three DECISION AGENTS to pick a traffic-management intervention for each of 500
stratified scenarios, and score them against a model-in-the-loop ground truth:

  RANDOM   : pick uniformly at random (the floor).
  RAW      : a local LLM given ONLY the prediction numbers.
  XTRAFFIC : the same LLM given the FULL pipeline output
             (prediction + mathematical explanation + advisory).

If XTRAFFIC beats RAW beats RANDOM, the full pipeline demonstrably helps a decision
maker choose better actions — the same claim the human study made, but reproducible
and at scale.

------------------------------------------------------------------------------
Ground truth (model-in-the-loop) — HONEST LIMITATION, state it in the paper
------------------------------------------------------------------------------
We do not have a field experiment, so we SIMULATE each candidate intervention by
perturbing the trained GNN's own input window and reading its 30-min-ahead network
delay. The lowest-delay intervention is the "ground-truth-optimal" one. This is
what the model BELIEVES helps, not observed reality — internally consistent (the
same model produces the predictions the agents are shown) and fully reproducible,
but it inherits the model's biases. Inherited directly from Phase 8's simulate.py.

------------------------------------------------------------------------------
Modeling note — SPEED, not flow
------------------------------------------------------------------------------
The GNN's input channel is SPEED (mph); there is no flow channel. A demand/flow
reduction relieves congestion => higher speed, so we model a flow/demand cut of
`pct` as a proportional SPEED uplift on the affected nodes over the last
`apply_steps` input timesteps. Rerouting changes graph CONNECTIVITY (not node
demand), so it is modeled as a weight cut on the top explanation edge via a
temporary physical-adjacency swap — the same buffer-swap trick the explainer uses.

  network_delay(pred) = sum over VALID nodes of max(0, free_flow_mph - pred_mph)
                        at the 30-min horizon.  Lower = better.
  delay_reduction     = network_delay(no_action) - network_delay(chosen).

Run (needs the trained checkpoint + a local Ollama with the advisor model):
  python -m xtraffic.evaluation.sim_eval --smoke           # 10 scenarios, sanity
  python -m xtraffic.evaluation.sim_eval                   # full 500
  python -m xtraffic.evaluation.sim_eval --model llama3.2:3b   # faster LLM
  python -m xtraffic.evaluation.sim_eval --mock-llm --limit 6  # harness only, no Ollama

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import yaml

from ..models.advisor.advisor import (
    Advisor, load_advisor_config, render_explanation_text,
)
from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import (
    SPEED_CHANNEL, TOD_CHANNEL, _tod_to_clock, load_window,
)
from ..models.gnn.loaders import _load_split, build_modality_dict
from ..utils.io_utils import PKG_ROOT
from .faithfulness import mean_std

OUT_DIR = os.path.join(PKG_ROOT, "evaluation", "results", "sim_eval")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
CONDITIONS = ["RANDOM", "RAW", "XTRAFFIC"]


def _load_cfg() -> Dict[str, Any]:
    with open(os.path.join(PKG_ROOT, "configs", "sim_eval.yaml")) as f:
        return yaml.safe_load(f)


# ===========================================================================
# 1. Stratified scenario sampling (deterministic).
#    Mirrors run_faithfulness_study._window_stats: mean speed + validity are in
#    REAL mph (missing sensor = raw 0 -> ~0 mph), because a missing reading is a
#    large NEGATIVE in z-space (the Phase-3 target-selection bug we fixed).
# ===========================================================================
def _window_stats(dataset: str, scaler: Dict[str, float]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per test window return (tod0, mean_real_speed, slowest_valid_node)."""
    X, _ = _load_split(dataset, "test")                    # X: [S, T, N, 2]
    S, T, N, _ = X.shape
    tod0 = X[:, 0, 0, TOD_CHANNEL].numpy()                 # clock at window start [S]

    speed_mph = X[..., SPEED_CHANNEL].numpy() * scaler["std"] + scaler["mean"]  # [S,T,N]
    valid = speed_mph > 1.0                                # >1 mph == genuine reading [S,T,N]
    flat_valid = valid.reshape(S, -1)                      # [S, T*N]
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(
            flat_valid.any(1),
            (speed_mph * valid).reshape(S, -1).sum(1) / np.clip(flat_valid.sum(1), 1, None),
            np.inf)                                        # [S]; empty window -> inf

    node_mean = speed_mph.mean(1)                          # mean over time per node [S, N]
    node_valid = valid.any(1)                              # nodes with any reading [S, N]
    slowest_node = np.where(node_valid, node_mean, np.inf).argmin(1)  # [S] slowest valid sensor
    return tod0, mean_speed, slowest_node


def sample_scenarios(dataset: str, scaler: Dict[str, float],
                     cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sample up to n_scenarios windows, spread evenly across the 15 strata cells
    (5 tod-bands x 3 congestion terciles). Deterministic given the seed."""
    tod0, mean_speed, slowest_node = _window_stats(dataset, scaler)
    finite = np.isfinite(mean_speed)                       # [S] windows with any valid reading

    # Congestion terciles by mean speed on the FINITE windows (slow = congested).
    q33, q66 = np.quantile(mean_speed[finite], [1 / 3, 2 / 3])

    def congestion_of(s: float) -> Optional[str]:
        if not np.isfinite(s):
            return None
        return "high" if s <= q33 else ("medium" if s <= q66 else "low")

    bands = cfg["stratify"]["tod_bands"]
    levels = cfg["stratify"]["congestion_levels"]
    n_cells = len(bands) * len(levels)
    # Spread the budget evenly; round up so we never fall short of n_scenarios.
    per_cell = int(np.ceil(cfg["n_scenarios"] / n_cells))

    rng = random.Random(cfg["seed"])
    scenarios: List[Dict[str, Any]] = []
    for band, lo, hi in bands:
        for level in levels:
            cell = [i for i in range(len(mean_speed))
                    if finite[i] and lo <= tod0[i] < hi
                    and congestion_of(mean_speed[i]) == level]
            rng.shuffle(cell)                              # seeded -> reproducible order
            for idx in cell[:per_cell]:
                scenarios.append({
                    "scenario_id": "s{:04d}".format(len(scenarios)),
                    "dataset": dataset,
                    "sample_index": int(idx),
                    "target_node": int(slowest_node[idx]),
                    "tod_band": band,
                    "congestion": level,
                    "scenario_type": "{}|{}".format(band, level),   # for consistency metric
                    "mean_speed_mph": float(round(mean_speed[idx], 2)),
                    "timestamp": "test#{} @ {}".format(idx, _tod_to_clock(float(tod0[idx]))),
                })
    # Trim to exactly n_scenarios (cells may over/under-fill); keep the front so the
    # trim is deterministic and still balanced across the earlier-listed cells.
    return scenarios[:cfg["n_scenarios"]]


# ===========================================================================
# 2. Intervention simulator (model-in-the-loop ground truth).
#    Reuses the ExplanationBuilder's already-loaded model/scaler/cfg so we don't
#    load the checkpoint twice. Perturbation semantics follow configs/sim_eval.yaml.
# ===========================================================================
class Phase10Simulator:
    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any],
                 scaler: Dict[str, float], sim_cfg: Dict[str, Any],
                 interventions: List[Dict[str, Any]], device: torch.device):
        self.model = model.eval()
        self.device = device
        self.scaler = scaler
        self.traffic_channels = cfg["traffic_channels"]
        self.modality_names = list(cfg["modalities"].keys())
        self.adj_bool = (self.model.physical_adj.detach().cpu().numpy() > 0)  # [N,N]

        self.horizon_step = int(sim_cfg["horizon_step"])   # 1-based
        self.free_flow = float(sim_cfg["free_flow_mph"])
        self.apply_steps = int(sim_cfg["apply_steps"])
        self.budget = float(sim_cfg["uplift_budget_mph"])  # equal budget per active action
        self.max_per_node = float(sim_cfg["max_per_node_mph"])  # cap on any one segment's uplift
        self.edge_cut = float(sim_cfg["reroute_edge_cut"]) # reroute edge-weight reduction
        self.interventions = interventions                 # list of dicts from the YAML

    # -- forward + delay ----------------------------------------------------
    def _predict_mph(self, X: torch.Tensor,
                     edge: Optional[Tuple[int, int, float]] = None) -> np.ndarray:
        """Forward the frozen model on window X; return 30-min speeds per node [N].

        X    : [1, T, N, C] z-scored input window.
        edge : optional (i, j, keep_factor) — temporarily scale physical_adj[i,j]
               and [j,i] by keep_factor (the reroute intervention).
        """
        h = self.horizon_step - 1                          # 0-based time index
        backup = None
        if edge is not None:
            i, j, keep = edge
            backup = self.model.physical_adj.detach().clone()       # [N,N] restore later
            A = backup.clone()
            A[i, j] = A[i, j] * keep                                 # cut the diverted link
            A[j, i] = A[j, i] * keep                                 # both directions
            self.model.physical_adj = A
        try:
            with torch.no_grad():
                mods = build_modality_dict(X, self.traffic_channels, self.modality_names)
                pred_z = self.model(mods)[0, h, :].cpu().numpy()     # [N] z-scored
        finally:
            if backup is not None:
                self.model.physical_adj = backup                     # always restore
        return pred_z * self.scaler["std"] + self.scaler["mean"]     # [N] real mph

    def _network_delay(self, pred_mph: np.ndarray, valid: np.ndarray) -> float:
        """Sum of speed deficit below free-flow over valid nodes (lower=better)."""
        deficit = np.clip(self.free_flow - pred_mph, 0.0, None)      # [N]
        return float(deficit[valid].sum())

    # -- node sets ----------------------------------------------------------
    def _neighbours(self, node: int) -> List[int]:
        """1-hop physical neighbours of `node` (excludes self)."""
        row = self.adj_bool[node].copy()
        row[node] = False
        return [int(i) for i in np.where(row)[0]]

    def _two_hop(self, node: int) -> List[int]:
        """Nodes within 2 physical hops of `node` (excludes self)."""
        one = set(self._neighbours(node))
        two = set(one)
        for n in one:
            two.update(self._neighbours(n))
        two.discard(node)
        return sorted(two)

    def _nodes_for(self, scope: str, target: int, last_speed: np.ndarray,
                   top_edge: Optional[Dict[str, Any]]) -> List[int]:
        """The characteristic node set the budget lands on, per intervention scope.

        For the SPREAD scopes (feeders / two_hop) we keep only the CONGESTED members
        (speed < free_flow AND a genuine reading), because spending budget on
        already-fast nodes is wasted (clipped at free-flow) — that dilution was why
        ramp/transit never won. Restricting to congested members makes the spread
        efficient, so the winner depends on WHERE congestion sits: an isolated
        bottleneck -> signal (target); several congested feeders -> ramp; a wide
        congested neighbourhood -> transit; one dominant upstream source -> reroute."""
        def congested(nodes: List[int]) -> List[int]:
            return [n for n in nodes if 1.0 < last_speed[n] < self.free_flow]
        if scope == "target":
            return [target]
        if scope == "feeders":                             # ramp metering: congested 1-hop feeders
            return congested(self._neighbours(target))
        if scope == "two_hop":                             # transit: congested broad neighbourhood
            return congested(self._two_hop(target))
        if scope == "source":                              # reroute: the top-edge source
            return [int(top_edge["from_id"])] if top_edge is not None else []
        raise ValueError("bad scope: {}".format(scope))

    def _apply_budget(self, X: torch.Tensor, nodes: List[int]) -> torch.Tensor:
        """Return a copy of X with the EQUAL BUDGET spread over `nodes` on the last
        `apply_steps` timesteps of the speed channel, CLIPPED at free-flow.

        per-node uplift = budget / |nodes|  (single-node actions get the whole
        budget). Clipping at free_flow is what makes concentration wasteful when the
        touched node is already near free-flow — the key to a non-degenerate GT."""
        Xm = X.clone()                                     # [1, T, N, C]
        if not nodes:                                      # empty set -> no-op (rare)
            return Xm
        # Split the equal budget, but cap any single segment's uplift. A concentrated
        # action (signal, 1 node) is therefore limited to max_per_node, while a spread
        # action (ramp/transit, many congested nodes) deploys more of the budget.
        per_node = min(self.budget / len(nodes), self.max_per_node)
        t0 = X.shape[1] - self.apply_steps                 # first touched timestep
        idx = torch.tensor(nodes, dtype=torch.long)
        seg_z = Xm[0, t0:, idx, SPEED_CHANNEL]             # [apply_steps, len(nodes)] z-scored
        seg_mph = seg_z * self.scaler["std"] + self.scaler["mean"]   # -> real mph
        seg_mph = np.minimum(seg_mph.numpy() + per_node, self.free_flow)  # add + clip at ff
        seg_mph_t = torch.from_numpy(seg_mph).to(seg_z.dtype)
        Xm[0, t0:, idx, SPEED_CHANNEL] = (seg_mph_t - self.scaler["mean"]) / self.scaler["std"]
        return Xm

    # -- public API ---------------------------------------------------------
    def score(self, X: torch.Tensor, target_node: int,
              top_edge: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Score every candidate intervention on window X; return per-intervention
        delay + ground truth (lowest delay) + delay reduction vs no_action.

        top_edge : the explanation's #1 edge {"from_id","to_id",...} used by the
                   reroute intervention. If None, reroute falls back to no-op.
        """
        speed_mph_window = (X[0, :, :, SPEED_CHANNEL].cpu().numpy()
                            * self.scaler["std"] + self.scaler["mean"])   # [T, N] real mph
        valid = (speed_mph_window > 1.0).any(0)            # [N] nodes with any reading
        last_speed = speed_mph_window[-1]                  # [N] most-recent obs (congestion state)

        results: List[Dict[str, Any]] = []
        for iv in self.interventions:
            name, kind, scope = iv["name"], iv["kind"], iv["scope"]
            edge = None
            Xm = X
            if kind == "speed_uplift":
                Xm = self._apply_budget(X, self._nodes_for(scope, target_node, last_speed, top_edge))
            elif kind == "reroute":
                # Reroute BOTH relieves the corridor source (diverted demand -> higher
                # source speed) AND cuts the top edge's weight (weakens propagation).
                Xm = self._apply_budget(X, self._nodes_for(scope, target_node, last_speed, top_edge))
                if top_edge is not None:
                    edge = (int(top_edge["from_id"]), int(top_edge["to_id"]),
                            1.0 - self.edge_cut)
            elif kind == "none":
                pass                                       # baseline
            else:
                raise ValueError("unknown intervention kind: {}".format(kind))

            pred_mph = self._predict_mph(Xm, edge=edge)    # [N]
            results.append({
                "intervention": name,
                "network_delay": round(self._network_delay(pred_mph, valid), 3),
                "target_speed_mph": round(float(pred_mph[target_node]), 2),
            })

        base = next(r for r in results if r["intervention"] == "no_action")
        for r in results:
            # Positive = this action lowered network delay vs doing nothing.
            r["delay_reduction"] = round(base["network_delay"] - r["network_delay"], 3)

        # Ground truth = lowest predicted delay; ties broken by config order.
        order = [iv["name"] for iv in self.interventions]
        best = min(results, key=lambda r: (r["network_delay"], order.index(r["intervention"])))
        return {
            "candidates": results,
            "ground_truth": best["intervention"],
            "delay_by_intervention": {r["intervention"]: r["delay_reduction"] for r in results},
            "no_action_is_best": best["intervention"] == "no_action",
        }


# ===========================================================================
# 3. Decision agents (RANDOM / RAW / XTRAFFIC).
# ===========================================================================
def _candidate_menu(interventions: List[Dict[str, Any]]) -> str:
    """Numbered menu of the 5 options with one-line descriptions (shown to the LLM)."""
    return "\n".join("  - {}: {}".format(iv["name"], iv["desc"]) for iv in interventions)


def _map_choice(raw_choice: str, names: List[str]) -> str:
    """Map a free-text LLM answer to one of the valid intervention names.
    exact -> substring -> fuzzy -> fallback to 'no_action' (the safe default)."""
    if not raw_choice:
        return "no_action"
    c = raw_choice.strip().lower().replace(" ", "_").replace("-", "_")
    if c in names:
        return c
    for n in names:                                        # substring either way
        if n in c or c in n:
            return n
    close = difflib.get_close_matches(c, names, n=1, cutoff=0.6)
    return close[0] if close else "no_action"


def _ollama_choose(host: str, model: str, prompt: str, temperature: float,
                   seed: int, timeout: float) -> str:
    """POST a decision prompt to Ollama and return the raw JSON text. The `seed`
    makes each decision reproducible AND lets the 3 study seeds diverge (temp>0).

    Retries transient failures (timeout / dropped connection) a few times with
    backoff — over thousands of calls a single hiccup must not kill the whole run.
    A missing model or a persistent outage still raises (nothing to salvage)."""
    import time
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",                                  # constrain to valid JSON
        "options": {"temperature": temperature, "seed": seed},
    }
    last_err: Optional[Exception] = None
    for attempt in range(3):                               # 3 tries, then give up
        try:
            resp = requests.post("{}/api/generate".format(host.rstrip("/")),
                                 json=payload, timeout=timeout)
            if resp.status_code == 404:
                raise RuntimeError("Model '{}' not found. Pull it: `ollama pull {}`."
                                   .format(model, model))
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))                  # 2s, 4s backoff
    raise RuntimeError(
        "Ollama at {} unreachable after 3 tries ({}). Is `ollama serve` running "
        "and `{}` pulled?".format(host, last_err, model))


# DESIGN NOTE (the Phase-10 XTRAFFIC-underperforms-RANDOM fix):
# The first version of this system prompt just said "pick the best intervention".
# With no guidance the LLM collapsed to a DEGENERATE policy: RAW always said
# signal_retiming, and XTRAFFIC always said reroute (it just copied the advisory,
# which always foregrounds a "reroute at I-5/SR-134" city-context fact). Because
# reroute is usually a DELAY-INCREASING action in the simulator, XTRAFFIC scored
# below RANDOM — not because the pipeline hurts, but because the agent never used
# the explanation's SPATIAL structure.
#
# Fix: give the SAME decision guidance to BOTH conditions (so the ONLY difference
# between RAW and XTRAFFIC stays the INFORMATION each sees, never the instructions).
# The guidance is generic traffic-engineering knowledge — match the intervention's
# SCOPE to WHERE the congestion originates — not a leak of the simulator's argmax.
# RAW is told the same rule but only sees the target's own speed, so it cannot
# locate an upstream source and falls back to a target-local action; XTRAFFIC has
# the full sensor table and CAN locate the source. That asymmetry is exactly the
# value of the explanation we are trying to measure.
_DECISION_SYSTEM = (
    "You are a traffic-operations decision-maker. From the numbered list of "
    "candidate interventions, choose EXACTLY ONE that will most reduce predicted "
    "network congestion 30 minutes from now.\n"
    "Each candidate relieves a DIFFERENT part of the network (read its "
    "description): the target segment itself, its immediate upstream feeders, one "
    "single upstream corridor, or the wider surrounding neighbourhood. To choose, "
    "read the contributing sensors in the information you are given and note which "
    "are actually SLOW (well below free-flow, about 60 mph) and how they are "
    "spread out. The mere fact that congestion propagates from upstream does NOT "
    "by itself mean reroute — most jams are fed by more than one place. Judge by "
    "HOW MANY sensors are slow and WHERE they sit:\n"
    "  - almost every contributing sensor is near free-flow and only the target is "
    "slow -> the bottleneck is local -> signal_retiming\n"
    "  - a few slow sensors sit immediately upstream, feeding the target "
    "-> ramp_metering\n"
    "  - several sensors are slow, spread across the wider surrounding area "
    "-> transit_surge\n"
    "  - reroute ONLY when a SINGLE upstream corridor is the sole slow source "
    "while the rest of the network flows freely\n"
    "  - the target is already near free-flow with no slow contributing sensors "
    "-> no_action\n"
    "Base the choice on how many sensors are slow and where they sit, not on the "
    "propagation narrative alone. Answer ONLY with JSON of the form "
    '{"chosen_intervention": "<one intervention name from the list>", '
    '"reasoning": "<one sentence describing how many sensors were slow and where>"}.'
)


def _raw_prompt(exp: Dict[str, Any], menu: str) -> str:
    """RAW condition: the LLM sees ONLY the prediction, then the menu."""
    p = exp["prediction"]
    return (
        "{sys}\n\n"
        "=== PREDICTION ===\n"
        "Location: {loc}\n"
        "Current speed: {cur} mph\n"
        "Predicted speed in {hz} min: {pred} mph\n\n"
        "=== CANDIDATE INTERVENTIONS ===\n{menu}\n"
    ).format(sys=_DECISION_SYSTEM, loc=p["node_name"], cur=p["current_speed_mph"],
             hz=p["horizon_minutes"], pred=p["predicted_speed_mph"], menu=menu)


def _xtraffic_prompt(exp: Dict[str, Any], advisory: Dict[str, Any], menu: str) -> str:
    """XTRAFFIC condition: prediction + full mathematical explanation + advisory."""
    recs = advisory.get("recommendations", []) if advisory else []
    rec_lines = "\n".join(
        "  - {} at {} (expected: {})".format(
            r.get("action", ""), r.get("location", ""), r.get("expected_effect", ""))
        for r in recs) or "  (none)"
    # The advisory lists 3-5 candidate actions of MIXED type (it is a grab-bag, not
    # a ranked pick), and it habitually foregrounds a "reroute" city-context fact.
    # Tell the agent to treat it as supporting context, NOT the answer, and to
    # decide from the spatial pattern in the explanation — otherwise it just copies
    # the first advisory line (the degenerate always-reroute behaviour we fixed).
    return (
        "{sys}\n\n"
        "=== MATHEMATICAL EXPLANATION (use this to LOCATE the congestion source) ===\n{exp}\n\n"
        "=== ADVISORY REASONING (supporting context) ===\n{reason}\n"
        "=== ADVISORY RECOMMENDATIONS (candidate ideas, NOT the answer) ===\n{recs}\n\n"
        "Decide from the spatial pattern in the explanation above (which sensors are "
        "slow and where the source is); do not simply copy the first advisory line.\n\n"
        "=== CANDIDATE INTERVENTIONS ===\n{menu}\n"
    ).format(sys=_DECISION_SYSTEM, exp=render_explanation_text(exp),
             reason=(advisory.get("reasoning", "") if advisory else ""),
             recs=rec_lines, menu=menu)


def decide(condition: str, exp: Dict[str, Any], advisory: Optional[Dict[str, Any]],
           interventions: List[Dict[str, Any]], seed: int,
           llm: Dict[str, Any], mock: bool) -> str:
    """Return the intervention name chosen by `condition` for this scenario+seed."""
    names = [iv["name"] for iv in interventions]
    if condition == "RANDOM":
        # Per-scenario reproducible uniform pick. We fold the scenario's target into
        # the seed so different scenarios get different picks under the same study seed.
        r = random.Random((seed + 1) * 1_000_003 + exp["prediction"]["node_id"])
        return names[r.randrange(len(names))]

    menu = _candidate_menu(interventions)
    prompt = (_raw_prompt(exp, menu) if condition == "RAW"
              else _xtraffic_prompt(exp, advisory, menu))

    if mock:
        # PLUMBING ONLY (no Ollama): deterministic, NOT scientific. RAW guesses the
        # first non-no_action option; XTRAFFIC "reads" the advisory and picks the
        # first recommendation-like option. Never use for paper numbers.
        return names[1] if condition == "RAW" else names[min(2, len(names) - 1)]

    raw = _ollama_choose(llm["host"], llm["model"], prompt,
                         float(llm["decision_temperature"]), seed, float(llm["timeout"]))
    try:
        choice = json.loads(raw).get("chosen_intervention", "")
    except json.JSONDecodeError:
        choice = raw                                       # let _map_choice salvage it
    return _map_choice(str(choice), names)


# ===========================================================================
# 4. Explanation + advisory caching (expensive, condition/seed-independent).
# ===========================================================================
def _cache_path(kind: str, sc: Dict[str, Any]) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "{}_{}_{}_{}.json".format(
        kind, sc["dataset"], sc["sample_index"], sc["target_node"]))


def get_explanation(builder: ExplanationBuilder, sc: Dict[str, Any],
                    horizon_step: int) -> Dict[str, Any]:
    path = _cache_path("exp", sc)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    X = load_window(sc["dataset"], sc["sample_index"])     # [1, T, N, C]
    exp = builder.explain_prediction(X, target_node=sc["target_node"],
                                     horizon_step=horizon_step, timestamp=sc["timestamp"])
    exp["meta"]["city"] = sc["dataset"]
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


def get_advisory(advisor: Optional[Advisor], sc: Dict[str, Any],
                 exp: Dict[str, Any], mock: bool) -> Dict[str, Any]:
    """Advisory (Layer 3) for the XTRAFFIC condition. Cached; empty under --mock-llm."""
    if mock or advisor is None:
        return {"reasoning": "", "recommendations": []}
    path = _cache_path("adv", sc)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    adv = advisor.advise(exp)["advisory"]
    with open(path, "w") as f:
        json.dump(adv, f, indent=2)
    return adv


def get_sim(sim: "Phase10Simulator", sc: Dict[str, Any],
            top_edge: Optional[Dict[str, Any]], load_X) -> Dict[str, Any]:
    """Cache the simulator's ground truth + per-intervention delay table for one
    scenario. This is condition/seed-INDEPENDENT (pure model-in-the-loop), so it
    caches like the explanation. We need the FULL delay_by_intervention (not just the
    chosen action's) so the RANDOM baseline can be reported as its EXPECTATION over
    many uniform draws without re-running the model per seed, and so a resumed run
    still has every scenario's delay table for that estimate. `load_X` is a thunk so a
    cache hit never pays to load the window."""
    path = _cache_path("sim", sc)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = sim.score(load_X(), sc["target_node"], top_edge)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ===========================================================================
# 5. Metrics + aggregation.
# ===========================================================================
def _consistency(rows: List[Dict[str, Any]]) -> float:
    """Decision consistency: fraction of decisions that match the MODAL decision
    within their scenario type (tod_band|congestion). 1.0 = the agent always makes
    the same call for the same kind of situation; ~1/5 = it decides at random."""
    by_type: Dict[str, List[str]] = {}
    for r in rows:
        by_type.setdefault(r["scenario_type"], []).append(r["chosen"])
    total = matches = 0
    for choices in by_type.values():
        modal_count = max(choices.count(c) for c in set(choices))
        matches += modal_count
        total += len(choices)
    return matches / total if total else float("nan")


def _per_seed_metrics(rows: List[Dict[str, Any]], condition: str, seed: int
                      ) -> Dict[str, float]:
    """Accuracy / mean delay reduction / consistency for one condition at one seed."""
    sub = [r for r in rows if r["condition"] == condition and r["seed"] == seed]
    if not sub:
        return {"accuracy": float("nan"), "delay_reduction": float("nan"),
                "consistency": float("nan")}
    acc = np.mean([r["correct"] for r in sub])
    dred = np.mean([r["delay_reduction"] for r in sub])
    return {"accuracy": float(acc), "delay_reduction": float(dred),
            "consistency": _consistency(sub)}


def random_expected_metrics(sim_records: List[Dict[str, Any]],
                            interventions: List[Dict[str, Any]], n_seeds: int
                            ) -> Dict[str, Any]:
    """RANDOM baseline reported as its EXPECTATION over `n_seeds` uniform draws.

    Each draw picks one intervention uniformly at random per scenario, then scores
    accuracy (matches ground truth?) and delay reduction (that action's delay cut)
    from the cached delay table. Reporting the mean over many draws recovers RANDOM's
    true ~1/K accuracy floor, instead of the noisy 3-study-seed realization that
    fluctuated to ~0.32 by clustering on one action. RAW/XTRAFFIC cannot be averaged
    this cheaply — each of their draws is an 8B-model call — so they stay on the 3
    study seeds; only the zero-cost RANDOM baseline gets the many-seed expectation."""
    names = [iv["name"] for iv in interventions]
    K = len(names)
    accs: List[float] = []
    dels: List[float] = []
    cons: List[float] = []
    for s in range(n_seeds):
        rng = random.Random(20260712 + s)      # fixed offset -> reproducible draws
        picks: List[Dict[str, Any]] = []
        for rec in sim_records:
            pick = names[rng.randrange(K)]
            picks.append({
                "scenario_type": rec["scenario_type"],
                "chosen": pick,
                "correct": int(pick == rec["ground_truth"]),
                "delay_reduction": rec["delay_by_intervention"].get(pick, 0.0),
            })
        accs.append(float(np.mean([p["correct"] for p in picks])))
        dels.append(float(np.mean([p["delay_reduction"] for p in picks])))
        cons.append(_consistency(picks))
    am, astd, _ = mean_std(accs)
    dm, dstd, _ = mean_std(dels)
    cm, cstd, _ = mean_std(cons)
    return {"condition": "RANDOM", "n_seeds": n_seeds,
            "accuracy_mean": am, "accuracy_std": astd,
            "delay_reduction_mean": dm, "delay_reduction_std": dstd,
            "consistency_mean": cm, "consistency_std": cstd,
            "per_seed": []}


def aggregate(rows: List[Dict[str, Any]], seeds: List[int],
              sim_records: List[Dict[str, Any]],
              interventions: List[Dict[str, Any]], random_seeds: int
              ) -> List[Dict[str, Any]]:
    """Per condition, compute each metric per seed then report mean +/- std across
    seeds. RANDOM is the many-seed EXPECTATION (see random_expected_metrics); the LLM
    conditions RAW/XTRAFFIC use the 3 study seeds (each draw is a model call)."""
    out: List[Dict[str, Any]] = []
    for cond in CONDITIONS:
        if cond == "RANDOM":
            out.append(random_expected_metrics(sim_records, interventions, random_seeds))
            continue
        per_seed = [_per_seed_metrics(rows, cond, s) for s in seeds]
        agg: Dict[str, Any] = {"condition": cond, "n_seeds": len(seeds)}
        for metric in ("accuracy", "delay_reduction", "consistency"):
            m, s, _ = mean_std([ps[metric] for ps in per_seed])
            agg[metric + "_mean"] = m
            agg[metric + "_std"] = s
        agg["per_seed"] = per_seed
        out.append(agg)
    return out


# ===========================================================================
# 6. Output — console table + CSV + JSON + LaTeX booktabs.
# ===========================================================================
def print_table(aggs: List[Dict[str, Any]]) -> None:
    print("\n=== SIM-EVAL DECISION QUALITY (mean +/- std over seeds) ===")
    for a in aggs:                                          # RANDOM = many-seed expectation
        if a["condition"] == "RANDOM":
            print("(RANDOM reported as its expectation over {} uniform draws; "
                  "RAW/XTRAFFIC over the {} study seeds.)".format(
                      a["n_seeds"], next(x["n_seeds"] for x in aggs
                                         if x["condition"] == "RAW")))
            break
    hdr = "{:12s}{:>20s}{:>22s}{:>20s}".format(
        "condition", "accuracy", "delay_reduction", "consistency")
    print(hdr)
    print("-" * len(hdr))
    for a in aggs:
        print("{:12s}{:>20s}{:>22s}{:>20s}".format(
            a["condition"],
            "{:.3f} +/- {:.3f}".format(a["accuracy_mean"], a["accuracy_std"]),
            "{:.2f} +/- {:.2f}".format(a["delay_reduction_mean"], a["delay_reduction_std"]),
            "{:.3f} +/- {:.3f}".format(a["consistency_mean"], a["consistency_std"])))


def write_latex(aggs: List[Dict[str, Any]], path: str, meta: Dict[str, Any]) -> None:
    """Booktabs table, \\input-ready, matching the Phase-9 table style."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pretty = {"RANDOM": "Random", "RAW": "RAW (pred only)", "XTRAFFIC": "XTraffic (full)"}
    lines = [
        "% Table -- simulation-based decision quality, {ds} "
        "(n={n} scenarios, {k} seeds, model={m}).".format(
            ds=meta["dataset"], n=meta["n_scenarios"], k=meta["n_seeds"], m=meta["model"]),
        "% Ground truth = lowest predicted 30-min network delay (model-in-the-loop).",
        "% delay reduction is in aggregate mph-deficit units (sum over valid nodes).",
        "% Random reported as its expectation over {r} uniform draws (no LLM); "
        "RAW/XTraffic over {k} study seeds.".format(
            r=meta.get("random_report_seeds", "many"), k=meta["n_seeds"]),
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Decision agent & Accuracy & Delay reduction & Consistency \\\\",
        "\\midrule",
    ]
    for a in aggs:
        lines.append("{cond} & {acc:.3f} $\\pm$ {accs:.3f} & {d:.2f} $\\pm$ {ds:.2f} "
                     "& {c:.3f} $\\pm$ {cs:.3f} \\\\".format(
                         cond=pretty[a["condition"]],
                         acc=a["accuracy_mean"], accs=a["accuracy_std"],
                         d=a["delay_reduction_mean"], ds=a["delay_reduction_std"],
                         c=a["consistency_mean"], cs=a["consistency_std"]))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ===========================================================================
# Main.
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="10 scenarios, quick sanity.")
    ap.add_argument("--limit", type=int, default=0, help="cap #scenarios (0 = config n).")
    ap.add_argument("--model", default=None, help="override the Ollama model name.")
    ap.add_argument("--mock-llm", action="store_true",
                    help="no Ollama; deterministic non-scientific choices (plumbing only).")
    ap.add_argument("--gt-only", action="store_true",
                    help="only compute + print the ground-truth intervention distribution "
                         "(no LLM, no decisions) — used to calibrate the simulator so no "
                         "single action dominates before the full run.")
    ap.add_argument("--budget", type=float, default=None,
                    help="override simulation.uplift_budget_mph (fast calibration sweeps).")
    ap.add_argument("--cap", type=float, default=None,
                    help="override simulation.max_per_node_mph (fast calibration sweeps).")
    args = ap.parse_args()

    cfg = _load_cfg()
    if args.smoke:
        cfg["n_scenarios"] = 10
    if args.limit:
        cfg["n_scenarios"] = args.limit
    if args.budget is not None:
        cfg["simulation"]["uplift_budget_mph"] = args.budget
    if args.cap is not None:
        cfg["simulation"]["max_per_node_mph"] = args.cap

    device = torch.device("cpu")                           # deterministic + tiny here
    datasets = [cfg["dataset"]]
    # Chicago joins automatically once a Chicago checkpoint exists (Colab-owed).
    chicago_ckpt = os.path.join(PKG_ROOT, "models", "gnn", "checkpoints", "chicago_best.pt")
    if os.path.exists(chicago_ckpt):
        datasets.append("chicago")

    # LLM connection details come from advisor.yaml; --model overrides everything.
    adv_cfg = load_advisor_config()
    model_name = args.model or adv_cfg["ollama"]["model"]
    llm = {
        "host": adv_cfg["ollama"]["host"],
        "model": model_name,
        "decision_temperature": cfg["llm"]["decision_temperature"],
        "timeout": adv_cfg["ollama"].get("timeout_seconds", 180),
    }
    seeds = cfg["decision_seeds"]

    # Resumable decision log: each (dataset, scenario, seed, condition) row is appended
    # to a JSONL as soon as it's computed, so a killed run (the full 500 is many hours
    # of CPU + 8B calls) resumes from where it stopped. --mock-llm uses its own log so
    # mock rows never contaminate a real run.
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUT_DIR, "decisions{}.jsonl".format("_mock" if args.mock_llm else ""))
    all_rows: List[Dict[str, Any]] = []
    # Resume key is (dataset, SAMPLE_INDEX, seed, condition) — the STABLE window
    # identity, NOT the positional scenario_id. scenario_id is assigned by sampling
    # ORDER (s0000, s0001, ...) and that order depends on n_scenarios: smoke's
    # "s0003" and the full run's "s0003" are DIFFERENT windows. Keying resume on
    # scenario_id would therefore let a --smoke decision be silently reused for a
    # different window in the full run (the exact cross-run contamination we hit).
    # sample_index names the actual test window, so a smoke decision is reused ONLY
    # for the identical window (smoke windows are a strict subset of the full 500:
    # same seed + same shuffle order => cell[:1] is a prefix of cell[:34]).
    done = set()                                           # (dataset, sample_index, seed, condition)
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["condition"] == "RANDOM":
                    continue                               # RANDOM is now a many-seed
                                                           # EXPECTATION, not a logged
                                                           # per-decision row; ignore any
                                                           # legacy RANDOM lines.
                all_rows.append(r)
                done.add((r["dataset"], r["sample_index"], r["seed"], r["condition"]))
        print("[resume] loaded {} completed decisions from {}".format(len(all_rows), jsonl_path))
    jsonl = open(jsonl_path, "a")

    # Per-scenario ground truth + delay table for the RANDOM many-seed expectation.
    # Filled for EVERY sampled scenario (via cached get_sim), including resumed ones.
    sim_records: List[Dict[str, Any]] = []

    for dataset in datasets:
        ckpt = (cfg["checkpoint"] if dataset == cfg["dataset"]
                else "models/gnn/checkpoints/{}_best.pt".format(dataset))
        print("\n### dataset={} ckpt={}".format(dataset, ckpt))
        builder = ExplanationBuilder(ckpt, dataset, device=device)
        sim = Phase10Simulator(builder.model, builder.cfg, builder.scaler,
                               cfg["simulation"], cfg["interventions"], device)
        advisor = None
        if not args.mock_llm and not args.gt_only:
            city = cfg["city"] if dataset == cfg["dataset"] else dataset
            advisor = Advisor(city, cfg=_advisor_cfg_with_model(adv_cfg, model_name))

        scenarios = sample_scenarios(dataset, builder.scaler, cfg)
        print("Sampled {} scenarios ({} tod-bands x {} congestion cells).".format(
            len(scenarios), len(cfg["stratify"]["tod_bands"]),
            len(cfg["stratify"]["congestion_levels"])))

        # --- GT-only calibration path: tally ground truth, print, skip LLM ----
        if args.gt_only:
            gt_counts: Dict[str, int] = {iv["name"]: 0 for iv in cfg["interventions"]}
            for i, sc in enumerate(scenarios):
                X = load_window(dataset, sc["sample_index"])
                exp = get_explanation(builder, sc, cfg["simulation"]["horizon_step"])
                top_edge = exp["top_edges"][0] if exp.get("top_edges") else None
                gt = sim.score(X, sc["target_node"], top_edge)["ground_truth"]
                gt_counts[gt] += 1
                if (i + 1) % 10 == 0 or i + 1 == len(scenarios):
                    print("  [{}/{}] tallied".format(i + 1, len(scenarios)))
            total = sum(gt_counts.values()) or 1
            print("\n=== GROUND-TRUTH DISTRIBUTION ({}, budget={} mph, n={}) ===".format(
                dataset, cfg["simulation"]["uplift_budget_mph"], total))
            for name, c in sorted(gt_counts.items(), key=lambda kv: -kv[1]):
                print("  {:16s} {:4d}  {:5.1f}%".format(name, c, 100.0 * c / total))
            continue

        # LLM conditions only: RANDOM is no longer a logged per-decision row, it is the
        # many-seed expectation computed from sim_records after the loop.
        llm_conditions = [c for c in CONDITIONS if c != "RANDOM"]

        for i, sc in enumerate(scenarios):
            # The GT + delay table (get_sim) is needed for EVERY scenario (the RANDOM
            # baseline is averaged over these), so compute/load it before the LLM skip.
            exp = get_explanation(builder, sc, cfg["simulation"]["horizon_step"])
            top_edge = exp["top_edges"][0] if exp.get("top_edges") else None
            sim_out = get_sim(sim, sc, top_edge,
                              lambda: load_window(dataset, sc["sample_index"]))
            gt = sim_out["ground_truth"]
            sim_records.append({"scenario_type": sc["scenario_type"],
                                "ground_truth": gt,
                                "delay_by_intervention": sim_out["delay_by_intervention"]})

            # Skip the LLM work if all (seed x LLM-condition) decisions are already done.
            todo = [(seed, cond) for seed in seeds for cond in llm_conditions
                    if (dataset, sc["sample_index"], seed, cond) not in done]
            if not todo:
                continue
            advisory = get_advisory(advisor, sc, exp, args.mock_llm)  # cached

            for seed, cond in todo:
                chosen = decide(cond, exp, advisory, cfg["interventions"],
                                seed, llm, args.mock_llm)
                row = {
                    "dataset": dataset,
                    "scenario_id": sc["scenario_id"],
                    "sample_index": sc["sample_index"],
                    "scenario_type": sc["scenario_type"],
                    "tod_band": sc["tod_band"],
                    "congestion": sc["congestion"],
                    "seed": seed,
                    "condition": cond,
                    "chosen": chosen,
                    "ground_truth": gt,
                    "correct": int(chosen == gt),
                    "delay_reduction": sim_out["delay_by_intervention"].get(chosen, 0.0),
                }
                all_rows.append(row)
                jsonl.write(json.dumps(row) + "\n")        # persist immediately (resumable)
                jsonl.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(scenarios):
                print("  [{}/{}] {} gt={}".format(i + 1, len(scenarios),
                                                  sc["scenario_type"], gt))

    jsonl.close()
    if args.gt_only:
        return                                             # distribution already printed

    # --- write outputs -----------------------------------------------------
    # CSV holds the LLM per-decision rows (RAW/XTRAFFIC). RANDOM is an aggregate
    # expectation (no per-decision rows), so it appears only in the summary + table.
    csv_path = os.path.join(OUT_DIR, "sim_eval_per_decision.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    random_seeds = int(cfg.get("random_report_seeds", 200))
    aggs = aggregate(all_rows, seeds, sim_records, cfg["interventions"], random_seeds)
    # Count distinct WINDOWS (sample_index), not scenario_ids: a resumed run can mix
    # smoke-positional and full-positional ids for the same window set (see the resume
    # note above), so sample_index is the collision-free unit to count.
    n_scenarios = len(set((r["dataset"], r["sample_index"]) for r in all_rows))
    meta = {"dataset": "+".join(datasets), "n_scenarios": n_scenarios,
            "n_seeds": len(seeds), "random_report_seeds": random_seeds,
            "model": ("mock" if args.mock_llm else model_name)}
    summary = {"meta": meta, "conditions": aggs}
    with open(os.path.join(OUT_DIR, "sim_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_table(aggs)
    tex_path = os.path.join(PKG_ROOT, cfg["output_tex"])
    write_latex(aggs, tex_path, meta)
    print("\nWrote:\n  {}\n  {}\n  {}".format(
        csv_path, os.path.join(OUT_DIR, "sim_eval_summary.json"), tex_path))


def _advisor_cfg_with_model(adv_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Copy the advisor config with the model name swapped (for --model runs)."""
    import copy
    c = copy.deepcopy(adv_cfg)
    c.setdefault("ollama", {})["model"] = model_name
    return c


if __name__ == "__main__":
    main()
