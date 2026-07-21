"""Phase 11 — CROSS-CITY x CROSS-MODEL faithfulness study.

WHY THIS EXISTS
---------------
Phase 5 proved the paper's Contribution #2 on ONE city (METR-LA) with ONE LLM
(llama3.1:8b): giving the advisor the mathematical explanation drives the
hallucination rate to ~0 (condition A), while withholding it makes the LLM
invent causes (condition B, hallucination ~0.83). A reviewer's obvious question
is: "Is that a METR-LA quirk, or a llama3.1 quirk?" This phase answers it by
re-running the SAME A/B/C study across:

  * THREE cities  -- METR-LA (in-domain), PEMS-BAY and Chicago (both ZERO-SHOT
    transfers of the METR-LA model), and
  * TWO LLMs      -- the primary (llama3.1:8b) and a second local Ollama model
    (the LLM-model ablation).

If "A has near-zero hallucination and a large positive F1 gap over B" holds in
EVERY city x model cell, the result generalizes and the claim is robust.

TWO THINGS THAT MAKE THIS HONEST (both stated in the paper):
  1. ZERO-SHOT TRANSFER. PEMS-BAY (325 nodes) and Chicago (1020 segments) do not
     match METR-LA's 207 nodes, so the node-SPECIFIC parameters (semantic
     embeddings, adaptive-graph node vectors, physical adjacency) cannot be
     copied -- they are re-initialised at the target node count, exactly as
     evaluation/cross_city.py documents. Only the node-AGNOSTIC weights
     (temporal/graph convs, modality encoders, readout) transfer. Prediction
     accuracy is therefore degraded off-domain. BUT faithfulness is a
     PROMPT-STRUCTURAL property: condition A shows the LLM the explainer's top-k
     and forbids inventing causes; condition B hides them. Whether the LLM obeys
     is independent of how ACCURATE the underlying prediction is -- which is
     precisely why a possibly-inaccurate transferred model is still a valid probe
     of the grounding effect.
  2. CHICAGO POOLING. Chicago's test split has only 15 windows (< our 20 floor),
     so we pool train+val+test (76 windows). Because the model is zero-shot on
     Chicago (never trained on it), pooling splits introduces NO train/test
     leakage for a faithfulness (not accuracy) measurement. Flagged in-code.

REUSE, NOT REINVENT
-------------------
Everything heavy is imported from the Phase-3/4/5 code:
  * ExplanationBuilder (Layer 2)            -- models/explainer/explain.py
  * Advisor.advise_condition (Layer 3, A/B/C) -- models/advisor/advisor.py
  * score_advisory + NodeTable (the metric) -- evaluation/faithfulness.py
  * sample_scenarios + _aggregate + grid    -- evaluation/run_faithfulness_study.py
  * transfer_weights + save_transferred     -- evaluation/cross_city.py
This file only adds the ORCHESTRATION: build a transferred checkpoint per target
city, pick a second model, loop cities x models x conditions (resumably), and
emit the master table.

OUTPUT: one LaTeX booktabs table (rows = cities, cols = models; each cell =
A-halluc / B-halluc / F1-gap) to evaluation/paper/table_cross_city_faith.tex,
plus CSV + JSON + per-cell summaries under
evaluation/results/cross_city_faithfulness/.

Run (needs the METR-LA checkpoint + a local Ollama):
  python -m xtraffic.evaluation.cross_city_faithfulness              # full study
  python -m xtraffic.evaluation.cross_city_faithfulness --smoke      # mock-LLM smoke
  python -m xtraffic.evaluation.cross_city_faithfulness --table-only # rebuild table
  python -m xtraffic.evaluation.cross_city_faithfulness --limit 4    # few scenarios/cell

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import yaml

from ..models.advisor.advisor import Advisor, load_advisor_config
from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import TOD_CHANNEL, _tod_to_clock
from ..models.gnn.loaders import _load_split, load_node_meta, load_scaler
from ..utils.io_utils import PKG_ROOT, processed_dir
from .cross_city import (build_model_for, save_transferred, set_seed,
                         transfer_weights)
from .faithfulness import NodeTable, score_advisory
from .run_faithfulness_study import (CONGESTION_LEVELS, SEED, TOD_BANDS,
                                     _METRIC_KEYS, _aggregate,
                                     sample_scenarios)

SPEED_CHANNEL = 0  # channel 0 of X is z-scored speed (channel 1 is time-of-day)

# Human-readable city labels for the table.
CITY_LABELS = {"metr_la": "METR-LA", "pems_bay": "PEMS-BAY", "chicago": "Chicago"}


# ===========================================================================
# Config loading
# ===========================================================================
def load_cfg(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or os.path.join(PKG_ROOT, "configs", "cross_city_faith.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# Ollama model discovery + second-model selection
# ===========================================================================
def installed_ollama_models(host: str) -> List[str]:
    """Return the list of model names installed in the local Ollama, or [] if the
    server is unreachable. We use /api/tags (the endpoint `ollama list` calls)."""
    try:
        resp = requests.get("{}/api/tags".format(host.rstrip("/")), timeout=10)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception as e:                       # server down / not installed
        print("[phase11] could not query Ollama at {} ({}).".format(host, e))
        return []


def resolve_second_model(host: str, primary: str, ladder: List[str]) -> Optional[str]:
    """Pick the first model in `ladder` that is installed and differs from the
    primary. Returns None if none are available (the cross-model column is then
    written as PENDING -- an honest hole, not a crash)."""
    installed = installed_ollama_models(host)
    # Ollama tags are like "mistral:7b"; a bare "mistral" should also match.
    inst_set = set(installed) | {m.split(":")[0] for m in installed}
    for cand in ladder:
        if cand == primary:
            continue
        if cand in inst_set or cand.split(":")[0] in inst_set:
            # Report if the playbook's first choices were absent, so it is on the
            # record WHY we used this model instead of qwen2.5/tinyllama.
            missing = [m for m in ladder[:ladder.index(cand)] if m != primary]
            if missing:
                print("[phase11] second-model ladder: {} not installed -> using '{}'."
                      .format(", ".join(missing), cand))
            return cand
    print("[phase11] no second model from the ladder is installed "
          "(installed: {}). Cross-model column -> PENDING.".format(installed or "none"))
    return None


# ===========================================================================
# Zero-shot transfer checkpoints (reuse cross_city.py's proven machinery)
# ===========================================================================
def _abs_ckpt(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PKG_ROOT, path)


def ensure_transfer_checkpoint(source_ckpt: str, source_city: str, target_ds: str,
                               device: torch.device, force: bool = False) -> str:
    """Return a checkpoint the ExplanationBuilder can load for `target_ds`.

    * If the target IS the source city, no transfer is needed -> return source.
    * Otherwise build a ZERO-SHOT checkpoint: copy every node-agnostic weight from
      the source into a fresh model sized to the target's node count + physical
      adjacency, leave node-specific params at their init, and save it in the
      checkpoint format load_model() expects. This is exactly cross_city.py's
      zero_shot path, factored out so the faithfulness study can consume it.

    WHY a saved checkpoint (not an in-memory model): ExplanationBuilder.load_model
    rebuilds from a checkpoint and would CRASH trying to load the 207-sized
    node-specific tensors into a 325/1020-node model (a shape mismatch raises even
    with strict=False). Pre-building the transferred checkpoint at the target N
    makes every shape line up, so the loader path stays byte-identical to Phase 5.
    """
    source_ckpt = _abs_ckpt(source_ckpt)
    if target_ds == source_city:
        return source_ckpt

    ckpt_dir = os.path.join(PKG_ROOT, "models", "gnn", "checkpoints")
    out_path = os.path.join(ckpt_dir, "{}_zero_shot.pt".format(target_ds))
    if os.path.exists(out_path) and not force:
        print("[phase11] reusing transferred checkpoint {}".format(out_path))
        return out_path

    print("[phase11] building ZERO-SHOT {}->{} checkpoint...".format(source_city, target_ds))
    ck = torch.load(source_ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    set_seed(cfg.get("seed", 42))                # deterministic re-init of node params
    modality_dims = cfg["modalities"]            # {'traffic':2,'weather':3,...}
    model = build_model_for(target_ds, modality_dims, cfg["model"], device)
    copied, skipped = transfer_weights(ck["model_state"], model)
    print("[phase11]   copied {} node-agnostic tensors, re-initialised {} "
          "node-specific tensors".format(copied, skipped))
    # scaler_ds = target_ds so the saved checkpoint carries the TARGET scaler
    # (load_model reloads the scaler by dataset anyway; kept consistent).
    path = save_transferred(model, target_ds, "zero_shot", cfg, target_ds)
    return path


# ===========================================================================
# Scenario sampling (metr_la/pems_bay reuse Phase 5; Chicago pools splits)
# ===========================================================================
# Cache of the pooled Chicago window tensor so we load it from disk only once.
_POOLED_X: Dict[str, torch.Tensor] = {}


def _pooled_X(dataset: str) -> torch.Tensor:
    """Concatenate train+val+test windows for a sparse dataset -> [S, T, N, C]."""
    if dataset not in _POOLED_X:
        parts = [_load_split(dataset, sp)[0] for sp in ("train", "val", "test")]
        _POOLED_X[dataset] = torch.cat(parts, dim=0)     # [S_total, T, N, C]
    return _POOLED_X[dataset]


def _window_stats_from_X(X: torch.Tensor, scaler: Dict[str, float]
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per window: (tod0, mean_real_speed, slowest_valid_node).

    Same rule as run_faithfulness_study._window_stats but on a pre-loaded tensor
    (so Chicago can pool splits). Validity is tested in REAL mph (missing = ~0
    mph), never in z-space -- that was a real Phase-3 bug."""
    Xn = X.numpy()
    S, T, N, _ = Xn.shape
    tod0 = Xn[:, 0, 0, TOD_CHANNEL]                        # clock at window start [S]
    speed_mph = Xn[..., SPEED_CHANNEL] * scaler["std"] + scaler["mean"]  # [S,T,N]
    valid = speed_mph > 1.0                                # >1 mph == genuine reading
    flat_valid = valid.reshape(S, -1)
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(
            flat_valid.any(1),
            (speed_mph * valid).reshape(S, -1).sum(1) / np.clip(flat_valid.sum(1), 1, None),
            np.inf)                                        # [S]; empty -> inf
    node_mean = speed_mph.mean(1)                          # [S,N]
    node_valid = valid.any(1)                              # [S,N]
    node_mean_masked = np.where(node_valid, node_mean, np.inf)
    slowest_node = node_mean_masked.argmin(1)              # [S]
    return tod0, mean_speed, slowest_node


def sample_scenarios_chicago(scaler: Dict[str, float], per_stratum: int,
                             min_scenarios: int) -> List[Dict[str, Any]]:
    """Stratified sample over Chicago's POOLED windows, with a non-stratified
    fallback if the sparse data can't fill the strata to `min_scenarios`."""
    X = _pooled_X("chicago")                               # [S, T, N, C]
    tod0, mean_speed, slowest_node = _window_stats_from_X(X, scaler)
    finite = np.isfinite(mean_speed)
    valid_speeds = mean_speed[finite]
    q33, q66 = np.quantile(valid_speeds, [1 / 3, 2 / 3])

    def congestion_of(s: float) -> Optional[str]:
        if not np.isfinite(s):
            return None
        if s <= q33:
            return "high"
        if s <= q66:
            return "medium"
        return "low"

    rng = random.Random(SEED)
    scenarios: List[Dict[str, Any]] = []
    for band, lo, hi in TOD_BANDS:
        for level in CONGESTION_LEVELS:
            cell = [i for i in range(len(mean_speed))
                    if finite[i] and lo <= tod0[i] < hi
                    and congestion_of(mean_speed[i]) == level]
            rng.shuffle(cell)
            for idx in cell[:per_stratum]:
                scenarios.append(_chic_scenario(idx, tod0, mean_speed,
                                                 slowest_node, band, level))

    if len(scenarios) < min_scenarios:
        # FALLBACK: Chicago is too sparse to fill the strata. Take ALL valid
        # windows (slowest first -- the most explainable), tagged with their real
        # tod/congestion, so we still reach the floor. Logged loudly so the paper
        # states Chicago's sample is small and not perfectly balanced.
        print("[phase11] Chicago stratified sample only {} (< {}); falling back to "
              "ALL {} valid pooled windows.".format(
                  len(scenarios), min_scenarios, int(finite.sum())))
        order = [i for i in np.argsort(mean_speed) if finite[i]]   # slowest first
        scenarios = []
        for idx in order:
            band = next((b for b, lo, hi in TOD_BANDS if lo <= tod0[idx] < hi), "night")
            level = congestion_of(mean_speed[idx]) or "high"
            scenarios.append(_chic_scenario(int(idx), tod0, mean_speed,
                                            slowest_node, band, level))
    return scenarios


def _chic_scenario(idx: int, tod0, mean_speed, slowest_node, band, level
                   ) -> Dict[str, Any]:
    return {
        "sample_index": int(idx),
        "target_node": int(slowest_node[idx]),
        "tod_band": band,
        "congestion": level,
        "mean_speed_mph": float(round(mean_speed[idx], 2)),
        "timestamp": "chicago-pooled#{} @ {}".format(idx, _tod_to_clock(float(tod0[idx]))),
    }


def sample_scenarios_for(city: str, dataset: str, scaler: Dict[str, float],
                         per_stratum: int, chicago_min: int) -> List[Dict[str, Any]]:
    """Dispatch: Chicago pools splits (sparse); everyone else uses the Phase-5
    test-split stratified sampler unchanged."""
    if city == "chicago":
        return sample_scenarios_chicago(scaler, per_stratum, chicago_min)
    return sample_scenarios(dataset, scaler, per_stratum)


def load_window_for(city: str, dataset: str, sc: Dict[str, Any]) -> torch.Tensor:
    """Return the single input window [1, T, N, C] for a scenario. Chicago indexes
    the pooled tensor; the others index the test split (same as Phase 5)."""
    if city == "chicago":
        i = sc["sample_index"]
        return _pooled_X("chicago")[i: i + 1]
    X, _ = _load_split(dataset, "test")
    return X[sc["sample_index"]: sc["sample_index"] + 1]


# ===========================================================================
# Explanation caching (per city+scenario -- LLM-independent, model-of-GNN fixed)
# ===========================================================================
def _exp_cache_path(results_dir: str, city: str, sample_index: int, node: int) -> str:
    d = os.path.join(results_dir, "explanations_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{}_{}_{}.json".format(city, sample_index, node))


def build_or_load_explanation(builder: ExplanationBuilder, city: str, dataset: str,
                              sc: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    """Explanation depends only on (city checkpoint, scenario), NOT on the LLM, so
    we build it once per city+scenario and reuse it across models and conditions."""
    path = _exp_cache_path(results_dir, city, sc["sample_index"], sc["target_node"])
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    X = load_window_for(city, dataset, sc)                  # [1, T, N, C]
    exp = builder.explain_prediction(
        X, target_node=sc["target_node"], horizon_step=6, timestamp=sc["timestamp"])
    exp["meta"]["city"] = city
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


# ===========================================================================
# Advisor construction (real Ollama, empty-KB for KB-less cities, or mock)
# ===========================================================================
class _EmptyKB:
    """A knowledge base that retrieves nothing -- used for cities with no KB
    (e.g. PEMS-BAY). Phase 5 already showed city context is ORTHOGONAL to
    faithfulness (C ~= A), so an empty context does not distort the A/B contrast;
    it just means condition A's city-context block is blank for that city."""

    def retrieve(self, exp: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
        return []


def make_advisor(city: str, model: str) -> Advisor:
    """Build an Advisor for (city, model). If the city has a registered KB
    (metr_la, chicago) use it; otherwise fall back to an empty KB so KB-less
    cities (pems_bay) run without fabricating out-of-region facts."""
    cfg = copy.deepcopy(load_advisor_config())
    cfg.setdefault("ollama", {})["model"] = model
    registered = cfg.get("cities", {})
    if city in registered:
        return Advisor(city, cfg)
    # No KB for this city: construct against a registered city (to satisfy
    # __init__), then blank the KB and correct the city label.
    base_city = next(iter(registered))
    adv = Advisor(base_city, cfg)
    adv.kb = _EmptyKB()
    adv.city = city
    print("[phase11] no KB registered for '{}' -> empty city context "
          "(Phase-5: city context is orthogonal to faithfulness).".format(city))
    return adv


class MockAdvisor:
    """A deterministic, Ollama-free stand-in for smoke tests (--smoke / --mock-llm).

    It mimics the REAL finding so the harness + aggregation + table can be
    verified end-to-end without a live LLM:
      * conditions A and C (explanation shown) -> cite two of the explainer's
        top-node NAMES verbatim  -> high F1, ~0 hallucination.
      * condition B (no explanation) -> cite a made-up location -> resolves to
        nothing -> hallucination ~1.0.
    This exercises exactly the A/B contrast the study is designed to detect."""

    def __init__(self, model: str = "mock"):
        self.model = model

    def advise_condition(self, exp: Dict[str, Any], condition: str) -> Dict[str, Any]:
        top = exp.get("top_nodes", [])
        if condition == "B":
            causes = [{"location": "Unknown arterial (not in the data)",
                       "resolved_node_id": None}]
            reasoning = "Congestion is likely due to unspecified upstream demand."
        else:  # A or C: cite real top-node names -> grounded
            causes = [{"location": n["node_name"], "resolved_node_id": n["node_id"]}
                      for n in top[:2]]
            reasoning = ("Predicted slowdown is driven by the top contributing "
                         "sensors listed in the explanation.")
        advisory = {
            "reasoning": reasoning,
            "cited_causes": causes,
            "recommendations": [
                {"action": "mock", "location": "mock", "time_window_minutes": 15,
                 "expected_effect": "mock", "grounded_in": ["mock"]}
                for _ in range(3)
            ],
        }
        return {"advisory": advisory, "condition": condition,
                "context_used": [], "model": self.model, "raw_responses": []}


# ===========================================================================
# Running one (city, model) cell -- resumable
# ===========================================================================
def chance_metrics(exp: Dict[str, Any], table: NodeTable, n_causes: float,
                   vocabulary: str, draws: int, rng: random.Random
                   ) -> Dict[str, float]:
    """Score `draws` RANDOM advisories against this explanation's top-k.

    Phase-11 correction. This is the traffic analogue of Phase 19's
    chance_metrics (RANDOM_bus / RANDOM_zone); it is duplicated rather than
    imported so the committed power-grid numbers cannot be perturbed by an edit
    made for the traffic side. The only difference is the vocabulary:
      "node"   -> the full rendered sensor names, e.g.
                  "Downtown San Jose (sensor 400664, 37.303, -121.878)"
                  = a PRECISE citation, resolves to exactly one node.
      "region" -> the region labels, e.g. "Downtown San Jose"
                  = a COARSE citation, resolves to that region's whole node set.

    WHY THIS ROW IS NOT OPTIONAL (learned the hard way). The metric grants
    region-level credit, so a coarse citation "hits" the top-k whenever the
    region contains ANY top-k node. Under the old compass-fallback naming, PEMS-BAY's
    largest region held 37.5% of the graph, and a citer that simply echoed the
    target's own sector scored precision 0.894 / F1 0.510 / hallucination 0.106
    with zero causal information — which is essentially the number the
    PEMS-BAY x mistral condition-B cell reported (0.888 / 0.508 / 0.112). Without
    a chance row on the table there was nothing to reveal that.

    Everything downstream — resolution, hit test, precision/recall/F1 — goes
    through the SAME score_advisory the LLM conditions use, so the comparison is
    exact rather than approximate.
    """
    if vocabulary == "node":
        pool = [table.namer.name(i) for i in range(table.n_nodes)]
    elif vocabulary == "region":
        pool = sorted(table.region_to_nodes.keys())
    elif vocabulary == "target_region":
        # THE TIGHT FLOOR, and the one that actually matters for condition B.
        #
        # "node" and "region" draw UNIFORMLY, so they are denied something
        # condition B is given: the prompt SHOWS the LLM the target's location.
        # A citer that knows only that, and simply names the target's own region,
        # is therefore the honest lower bound — and because causes are spatially
        # local, it hits top-k far more often than a uniform draw. Measured gap on
        # Chicago: uniform region F1 0.094 vs target-echo 0.565, a 6x difference
        # on the same city. Judging condition A against the uniform row alone
        # would flatter every cell.
        #
        # This is also the control that REPRODUCED the original PEMS-BAY x mistral
        # outlier almost exactly (0.894/0.385/0.510/0.106 vs the reported
        # 0.888/0.385/0.508/0.112), so it is the control with a track record.
        #
        # It is deterministic — there is exactly one such citation per scenario —
        # so the `draws` loop below simply repeats an identical advisory.
        tgt = (exp.get("prediction") or {}).get("node_id")
        if tgt is None or not (0 <= int(tgt) < table.n_nodes):
            return {k: float("nan") for k in
                    ("cause_precision", "cause_recall",
                     "faithfulness_f1", "hallucination_rate")}
        pool = [table.region[int(tgt)]]
        n_causes, draws = 1, 1          # deterministic: one citation, one draw
    else:
        raise ValueError("unknown vocabulary {!r}".format(vocabulary))

    n = max(1, int(round(n_causes)))
    acc: Dict[str, List[float]] = {"cause_precision": [], "cause_recall": [],
                                   "faithfulness_f1": [], "hallucination_rate": []}
    for _ in range(draws):
        advisory = {
            "reasoning": "",
            # with replacement: an LLM can repeat a location
            "cited_causes": [{"location": rng.choice(pool), "resolved_node_id": None}
                             for _ in range(n)],
            "recommendations": [],
        }
        m = score_advisory(exp, advisory, table)
        for k in acc:
            acc[k].append(float(m[k]))
    return {k: float(sum(v) / len(v)) for k, v in acc.items()}


def _sanitize(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _decisions_path(results_dir: str, city: str, model: str, mock: bool) -> str:
    d = os.path.join(results_dir, "decisions")
    os.makedirs(d, exist_ok=True)
    suffix = "__MOCK" if mock else ""
    return os.path.join(d, "{}__{}{}.jsonl".format(city, _sanitize(model), suffix))


def _load_done(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Load already-computed decisions keyed by (sample_index, condition) so a
    rerun of a multi-hour cell skips finished work (like Phase 10's resumability)."""
    done: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[(r["sample_index"], r["condition"])] = r
    return done


def run_cell(city: str, dataset: str, checkpoint: str, model: str,
             conditions: List[str], scenarios: List[Dict[str, Any]],
             table: NodeTable, advisor: Any, results_dir: str,
             explainer_cfg: Dict[str, Any], device: torch.device,
             mock: bool, chance_cfg: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """Run every scenario x condition for one (city, model) cell and return the
    aggregated A/B/C metrics. Explanations are built once per scenario (shared
    across conditions); advisories + metrics are cached per decision (resumable)."""
    print("\n=== CELL  city={}  model={}  n_scenarios={}  conditions={} ==="
          .format(city, model, len(scenarios), ",".join(conditions)))
    builder = ExplanationBuilder(
        checkpoint, dataset, device=device,
        top_k=explainer_cfg["top_k"], epochs=explainer_cfg["epochs"],
        confidence_runs=explainer_cfg["confidence_runs"])

    dec_path = _decisions_path(results_dir, city, model, mock)
    done = _load_done(dec_path)
    rows: List[Dict[str, Any]] = []

    exps: List[Dict[str, Any]] = []          # kept for the chance baselines below
    for i, sc in enumerate(scenarios):
        exp = build_or_load_explanation(builder, city, dataset, sc, results_dir)
        exps.append(exp)
        for cond in conditions:
            key = (sc["sample_index"], cond)
            if key in done:
                rows.append(done[key])
                continue
            res = advisor.advise_condition(exp, cond)
            metrics = score_advisory(exp, res["advisory"], table)
            row = {
                "city": city, "model": model,
                "sample_index": sc["sample_index"], "target_node": sc["target_node"],
                "tod_band": sc["tod_band"], "congestion": sc["congestion"],
                "condition": cond,
                "advisory_error": res["advisory"].get("_error", ""),
                # Phase-11 correction: LOG THE CITATION TEXT AND THE RESOLVER RUNG.
                # This study previously persisted metrics ONLY, so diagnosing the
                # PEMS-BAY condition-B outlier required re-running the LLM against
                # the cached explanations just to see what it had actually cited —
                # and an LLM at temp 0.1 does not reproduce its own output, so the
                # re-run could only ever be circumstantial. Phase 19 already logs
                # both fields; this backports that to the traffic study. With these
                # two columns the same investigation is a grep.
                "cited_locations": [c.get("location") for c in
                                    (res["advisory"].get("cited_causes") or [])
                                    if isinstance(c, dict)],
                "resolution_methods": [p["resolved_method"]
                                       for p in metrics["per_cause"]],
                "resolved_set_sizes": [len(p["resolved_node_ids"])
                                       for p in metrics["per_cause"]],
            }
            for k in _METRIC_KEYS + ["n_cited_causes", "n_topk"]:
                row[k] = metrics[k]
            rows.append(row)
            with open(dec_path, "a") as f:       # append-on-complete -> resumable
                f.write(json.dumps(row) + "\n")
            done[key] = row
        if (i + 1) % 10 == 0 or (i + 1) == len(scenarios):
            print("  [{}/{}] {} cong={} done".format(
                i + 1, len(scenarios), sc["tod_band"], sc["congestion"]))

    # Keep only rows for the CURRENTLY-sampled scenarios (a smaller --limit must
    # not aggregate stale cache rows from a bigger previous run).
    wanted = {sc["sample_index"] for sc in scenarios}
    rows = [r for r in rows if r["sample_index"] in wanted]

    # --- chance baselines (Phase-11 correction; no LLM, so these are cheap) ----
    # Sized to condition A's mean citation count so the random citer is given the
    # SAME budget of guesses the real model used — otherwise a chance row with
    # more citations would score a higher recall for free.
    all_conditions = list(conditions)
    if chance_cfg:
        rng = random.Random(SEED)
        a_rows = [r for r in rows if r["condition"] == "A"]
        mean_cited = (float(np.mean([r["n_cited_causes"] for r in a_rows]))
                      if a_rows else 3.0)
        draws = int(chance_cfg.get("random_draws", 200))
        for vocab in chance_cfg.get("random_baselines", []):
            cond_name = "RANDOM_" + vocab
            for sc, exp in zip(scenarios, exps):
                row = {
                    "city": city, "model": model,
                    "sample_index": sc["sample_index"], "target_node": sc["target_node"],
                    "tod_band": sc["tod_band"], "congestion": sc["congestion"],
                    "condition": cond_name, "advisory_error": "",
                    "n_cited_causes": int(round(mean_cited)),
                    "n_topk": len(exp.get("top_nodes", [])),
                    # A random citer writes no prose, so it states no numbers and
                    # quantitative fidelity is UNDEFINED — the same None
                    # score_advisory returns for a numberless advisory, which
                    # _aggregate already filters out of its mean.
                    "quantitative_fidelity": None,
                    # Same columns as the LLM rows so the per-scenario CSV has one
                    # consistent schema. A chance draw has no prose to log.
                    "cited_locations": [], "resolution_methods": [],
                    "resolved_set_sizes": [],
                }
                row.update(chance_metrics(exp, table, mean_cited, vocab, draws, rng))
                # NOT written to decisions.jsonl: these are recomputed
                # deterministically from (seed, explanation) and carry no LLM cost,
                # so caching them would only risk staleness.
                rows.append(row)
            all_conditions.append(cond_name)
        print("  chance baselines: {} ({} draws x {:.1f} citations)".format(
            ", ".join(chance_cfg.get("random_baselines", [])), draws, mean_cited))

    aggs = {c: _aggregate(rows, c) for c in all_conditions}
    summary = {
        "city": city, "model": model, "n_scenarios": len(scenarios),
        "conditions": all_conditions, "aggregate": aggs, "seed": SEED,
        "zero_shot": (city != "metr_la"),
    }
    # Persist per-cell CSV + JSON (CLAUDE.md: both formats).
    tag = "{}__{}{}".format(city, _sanitize(model), "__MOCK" if mock else "")
    with open(os.path.join(results_dir, "summary_{}.json".format(tag)), "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        csv_path = os.path.join(results_dir, "per_scenario_{}.csv".format(tag))
        # Rows can legitimately carry DIFFERENT key sets. This study is resumable,
        # so decisions replayed from a cache written by an EARLIER version lack any
        # column added since (here: cited_locations / resolution_methods /
        # resolved_set_sizes), while rows computed fresh in this process have them.
        # Keying the writer off rows[0] alone made a resumed run crash on its own
        # cache. Take the UNION of keys in first-seen order and fill absentees with
        # "", so schema evolution degrades to blank cells instead of a hard failure.
        fieldnames: List[str] = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            w.writeheader()
            w.writerows(rows)

    a, b = aggs.get("A"), aggs.get("B")
    if a and b:
        print("  -> A halluc {:.3f} | B halluc {:.3f} | F1 gap {:+.3f}".format(
            a["hallucination_rate_mean"], b["hallucination_rate_mean"],
            a["faithfulness_f1_mean"] - b["faithfulness_f1_mean"]))
    return summary


# ===========================================================================
# Reusing the committed Phase-5 METR-LA / llama3.1 summary
# ===========================================================================
def load_phase5_metr_la(model: str) -> Optional[Dict[str, Any]]:
    """Return an aggregate dict compatible with run_cell()'s output built from the
    committed Phase-5 summary, IF it exists and used the same model. Saves us from
    recomputing ~93 scenarios we already have on record."""
    path = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness",
                        "faithfulness_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        s = json.load(f)
    if s.get("model") != model:
        return None
    aggs = {a["condition"]: a for a in s.get("overall", [])}
    print("[phase11] reusing committed Phase-5 METR-LA summary for model '{}' "
          "(n={}).".format(model, s.get("n_scenarios")))
    return {"city": "metr_la", "model": model,
            "n_scenarios": s.get("n_scenarios"), "conditions": list(aggs.keys()),
            "aggregate": aggs, "seed": s.get("seed", SEED), "zero_shot": False,
            "source": "phase5-reused"}


# ===========================================================================
# Master table (rows = cities, cols = models; cell = A/B halluc + F1 gap)
# ===========================================================================
def _cell_values(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Extract (A halluc, B halluc, F1 gap) from a cell summary, or None."""
    if summary is None:
        return None
    aggs = summary.get("aggregate", {})
    a, b = aggs.get("A"), aggs.get("B")
    if not a or not b:
        return None
    return {
        "a_halluc": a["hallucination_rate_mean"],
        "b_halluc": b["hallucination_rate_mean"],
        "a_f1": a["faithfulness_f1_mean"],
        "b_f1": b["faithfulness_f1_mean"],
        "f1_gap": a["faithfulness_f1_mean"] - b["faithfulness_f1_mean"],
        "n": summary.get("n_scenarios", a.get("n")),
    }


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def build_master_table(results: Dict[Tuple[str, str], Optional[Dict[str, Any]]],
                       cities: List[str], models: List[str],
                       out_tex: str, results_dir: str,
                       table_cities: Optional[List[str]] = None) -> Dict[str, Any]:
    """Write the LaTeX booktabs master table + CSV + JSON, and return a verdict
    dict. A cell with no data is written as PENDING (honest hole, never a crash).
    Each populated cell shows A-halluc / B-halluc / F1-gap stacked."""
    if table_cities is None:
        table_cities = list(cities)
    # --- LaTeX ------------------------------------------------------------
    # `table_cities` is the FAITHFULNESS subset (config: faithfulness_table_cities).
    # It is not necessarily `cities`: Chicago is still run and still reported for
    # prediction transfer, but is withheld here because its chance floor is too
    # close to condition A to report honestly (see the caption and the config).
    col_spec = "l" + "c" * len(models)
    header = " & ".join(["City"] + [_tex_escape(m) for m in models]) + r" \\"
    excluded = [c for c in cities if c not in table_cities]
    excl_note = ""
    if excluded:
        excl_note = (
            r" \textbf{" + ", ".join(CITY_LABELS.get(c, c) for c in excluded) +
            r"} is excluded from faithfulness reporting: its spatial density makes "
            r"the chance floor (target-region $\mathrm{F1}=0.565$) too close to "
            r"condition~A ($\mathrm{F1}=0.629$) to report honestly. It is retained "
            r"for prediction-transfer results.")
    lines = [
        "% Auto-generated by evaluation/cross_city_faithfulness.py (Phase 11).",
        "% Do NOT edit by hand -- rerun the study to refresh.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-city, cross-model faithfulness. Each cell reports "
        r"condition-A hallucination / condition-B hallucination / faithfulness-F1 "
        r"gap ($\mathrm{F1}_A-\mathrm{F1}_B$). Near-zero A-hallucination and a large "
        r"positive F1 gap in every cell show the mathematical \emph{explanation}---"
        r"not the city or the LLM---is what keeps the advisor grounded. The "
        r"\textsc{chance} rows are citers with no model and no explanation: "
        r"\textsc{random region} draws a region uniformly, and \textsc{target "
        r"region} names the target's own region---the tight floor, since condition~B "
        r"is always shown the target's location. Condition~A must be read against "
        r"the latter. PEMS-BAY uses zero-shot transfer of the METR-LA model."
        + excl_note + "}",
        r"\label{tab:cross_city_faith}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for ci, city in enumerate(table_cities):
        cells = [CITY_LABELS.get(city, city)]
        for model in models:
            v = _cell_values(results.get((city, model)))
            if v is None:
                cells.append(r"\emph{PENDING}")
            else:
                # A's F1 is shown explicitly: the chance rows beneath are F1s, and
                # without it a reader cannot make the comparison the caption asks
                # for (condition A against the target-region floor).
                cells.append(
                    r"\shortstack{{A F1 {:.3f}\\A hal. {:.3f}\\B hal. {:.3f}\\"
                    r"$\Delta$F1 {:+.3f}}}".format(
                        v["a_f1"], v["a_halluc"], v["b_halluc"], v["f1_gap"]))
        lines.append(" & ".join(cells) + r" \\")
        # The two chance rows for THIS city, indented directly beneath its cell so
        # a reader cannot read a condition-A number without its floor.
        for vocab, label in (("RANDOM_region", r"\quad \textsc{chance}: random region"),
                             ("RANDOM_target_region",
                              r"\quad \textsc{chance}: target region")):
            row = [label]
            for model in models:
                s = results.get((city, model))
                f1 = None
                if s:
                    agg = s.get("aggregate", {}).get(vocab)
                    if agg:
                        f1 = agg.get("faithfulness_f1_mean")
                row.append(r"\emph{--}" if f1 is None else "F1 {:.3f}".format(f1))
            lines.append(" & ".join(row) + r" \\")
        if ci != len(table_cities) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(_abs_ckpt(out_tex)), exist_ok=True)
    with open(_abs_ckpt(out_tex), "w") as f:
        f.write("\n".join(lines))

    # --- CSV + JSON master + verdict --------------------------------------
    flat: List[Dict[str, Any]] = []
    holds = True
    any_cell = False
    for city in cities:
        for model in models:
            v = _cell_values(results.get((city, model)))
            rec: Dict[str, Any] = {"city": city, "model": model,
                                   # False => run and logged, but withheld from the
                                   # faithfulness table (see faithfulness_table_cities)
                                   "in_faithfulness_table": city in table_cities}
            if v is None:
                rec["status"] = "PENDING"
            else:
                any_cell = True
                rec["status"] = "ok"
                rec.update({k: round(v[k], 4) if isinstance(v[k], float) else v[k]
                            for k in v})
                # "Gap holds" = A much less hallucination than B AND positive F1 gap.
                cell_holds = (v["b_halluc"] - v["a_halluc"] > 0.30) and (v["f1_gap"] > 0.20)
                rec["gap_holds"] = cell_holds
                # Only cells we actually REPORT may support the verdict.
                if city in table_cities:
                    holds = holds and cell_holds
            flat.append(rec)

    with open(os.path.join(results_dir, "cross_city_faithfulness_master.csv"),
              "w", newline="") as f:
        fields = sorted({k for r in flat for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)

    n_ok = sum(1 for r in flat
               if r.get("status") == "ok" and r.get("in_faithfulness_table"))
    n_pending = sum(1 for r in flat if r.get("status") == "PENDING")
    if n_ok == 0:
        note = ("no cells computed yet ({} PENDING) -- run the full study to "
                "populate the table".format(n_pending))
    elif holds:
        note = ("A vs B hallucination gap holds across ALL {} populated city x "
                "model cells{}".format(n_ok,
                    " ({} still PENDING)".format(n_pending) if n_pending else ""))
    else:
        note = ("gap does NOT hold in some populated cell -- inspect the failing "
                "cells (gap_holds=false)")
    verdict = {
        "cities": cities, "models": models, "cells": flat,
        "n_computed": n_ok, "n_pending": n_pending,
        "hallucination_gap_holds_everywhere": bool(any_cell and holds),
        "note": note,
    }
    with open(os.path.join(results_dir, "cross_city_faithfulness_master.json"),
              "w") as f:
        json.dump(verdict, f, indent=2)
    return verdict


def print_master(results, cities, models, verdict) -> None:
    print("\n================ CROSS-CITY x CROSS-MODEL FAITHFULNESS ================")
    print("(each cell: A-halluc / B-halluc / F1-gap)")
    hdr = "{:12s}".format("city")
    for m in models:
        hdr += "{:>26s}".format(m)
    print(hdr)
    print("-" * len(hdr))
    for city in cities:
        line = "{:12s}".format(CITY_LABELS.get(city, city))
        for m in models:
            v = _cell_values(results.get((city, m)))
            cell = ("PENDING" if v is None else
                    "{:.3f} / {:.3f} / {:+.3f}".format(v["a_halluc"], v["b_halluc"], v["f1_gap"]))
            line += "{:>26s}".format(cell)
        print(line)
    print("-" * len(hdr))
    print("VERDICT:", verdict["note"])


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 11 cross-city x cross-model faithfulness")
    ap.add_argument("--config", default=None)
    ap.add_argument("--primary-model", default=None, help="override the primary LLM")
    ap.add_argument("--second-model", default=None,
                    help="force the second LLM (else auto-detect from the ladder)")
    ap.add_argument("--cities", default=None, help="comma list overriding the config")
    ap.add_argument("--per-stratum", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap scenarios per cell (smoke only; truncates the front of "
                         "the stratum-ordered list, so a small --limit is not stratified)")
    ap.add_argument("--mock-llm", action="store_true",
                    help="use a deterministic mock advisor (no Ollama) to verify the harness")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end check: mock-llm + few scenarios + short explainer "
                         "+ outputs routed to a *_smoke dir (never clobbers real results)")
    ap.add_argument("--table-only", action="store_true",
                    help="rebuild the master table from existing per-cell summaries; no runs")
    ap.add_argument("--force-transfer", action="store_true",
                    help="rebuild the zero-shot transfer checkpoints even if cached")
    ap.add_argument("--explainer-epochs", type=int, default=None)
    ap.add_argument("--confidence-runs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cpu")                 # explainer is tiny + deterministic

    cities = ([c.strip() for c in args.cities.split(",")] if args.cities
              else list(cfg["cities"]))
    per_stratum = args.per_stratum or cfg["per_stratum"]
    chicago_min = cfg["chicago_min_scenarios"]

    # Models -------------------------------------------------------------
    adv_cfg = load_advisor_config()
    host = adv_cfg["ollama"]["host"]
    primary = args.primary_model or cfg["models"].get("primary") or adv_cfg["ollama"]["model"]

    explainer_cfg = {
        "epochs": args.explainer_epochs or cfg["explainer"]["epochs"],
        "confidence_runs": args.confidence_runs or cfg["explainer"]["confidence_runs"],
        "top_k": cfg["explainer"]["top_k"],
    }

    results_dir = _abs_ckpt(cfg["output"]["results_dir"])
    out_tex = cfg["output"]["tex"]
    mock = args.mock_llm or args.smoke

    if args.smoke:                               # fast, self-contained, isolated
        results_dir = results_dir + "_smoke"
        out_tex = out_tex.replace(".tex", "_smoke.tex")
        explainer_cfg["epochs"] = args.explainer_epochs or 15
        explainer_cfg["confidence_runs"] = args.confidence_runs or 1
        if not args.limit:
            args.limit = 3
    os.makedirs(results_dir, exist_ok=True)

    # Second model: forced, auto-detected, or mock ('mock2' when --mock-llm).
    if mock:
        second: Optional[str] = "mock2"
    elif args.second_model:
        second = args.second_model
    else:
        second = resolve_second_model(host, primary, cfg["models"]["second_ladder"])

    models = [primary] + ([second] if second else [])
    print("[phase11] primary model: {} | second model: {}".format(
        primary, second or "(none available -> PENDING column)"))

    # Faithfulness-table subset, resolved ONCE so every emit path agrees. Chicago
    # is run and logged but withheld here (see faithfulness_table_cities in the
    # config for the full reasoning).
    table_cities = [c for c in cfg.get("faithfulness_table_cities", cities)
                    if c in cities]
    withheld = [c for c in cities if c not in table_cities]

    def _emit(results):
        verdict = build_master_table(results, cities, models, out_tex, results_dir,
                                     table_cities=table_cities)
        if withheld:
            print("\n[phase11] WITHHELD from the faithfulness table (still run + "
                  "logged, reported for prediction transfer only): {}"
                  .format(", ".join(withheld)))
        print_master(results, table_cities, models, verdict)
        return verdict

    # --- table-only: rebuild from whatever per-cell summaries exist on disk ---
    if args.table_only:
        results = {}
        for city in cities:
            for model in models:
                tag = "{}__{}".format(city, _sanitize(model))
                p = os.path.join(results_dir, "summary_{}.json".format(tag))
                results[(city, model)] = (json.load(open(p)) if os.path.exists(p)
                                          else None)
        _emit(results)
        return

    conditions_primary = cfg["conditions"]["primary"]
    conditions_second = cfg["conditions"]["second"]
    reuse_p5 = cfg["output"].get("reuse_phase5_metr_la", True) and not mock

    results: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

    for city in cities:
        dataset = city                            # dataset name == city name here
        if not os.path.exists(os.path.join(processed_dir(dataset), "test.npz")):
            print("[phase11] {} not processed -> skipping (cells PENDING).".format(dataset))
            for model in models:
                results[(city, model)] = None
            continue

        # One transferred checkpoint per city (in-domain for metr_la).
        checkpoint = ensure_transfer_checkpoint(
            cfg["source_checkpoint"], cfg["source_city"], dataset, device,
            force=args.force_transfer)
        scaler = load_scaler(dataset)
        table = NodeTable(load_node_meta(dataset))
        scenarios = sample_scenarios_for(city, dataset, scaler, per_stratum, chicago_min)
        if args.limit and len(scenarios) > args.limit:
            scenarios = scenarios[:args.limit]
        print("[phase11] {}: {} scenarios".format(city, len(scenarios)))

        for model in models:
            conditions = conditions_primary if model == primary else conditions_second

            # Reuse the committed Phase-5 numbers for the METR-LA/primary cell.
            if reuse_p5 and city == "metr_la" and model == primary:
                reused = load_phase5_metr_la(model)
                if reused is not None:
                    results[(city, model)] = reused
                    continue

            advisor = (MockAdvisor(model) if mock else make_advisor(city, model))
            results[(city, model)] = run_cell(
                city, dataset, checkpoint, model, conditions, scenarios, table,
                advisor, results_dir, explainer_cfg, device, mock,
                chance_cfg={"random_baselines": cfg.get("random_baselines", []),
                            "random_draws": cfg.get("random_draws", 200)})

    verdict = _emit(results)
    print("\n[phase11] wrote:")
    print("  " + _abs_ckpt(out_tex))
    print("  " + os.path.join(results_dir, "cross_city_faithfulness_master.{csv,json}"))


if __name__ == "__main__":
    main()
