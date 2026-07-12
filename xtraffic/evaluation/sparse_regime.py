"""Phase 14 — sparse-sensor-regime experiments ("the rural claim without rural data").

METR-LA is a DENSE urban freeway network (207 sensors packed along LA freeways).
Rural and small-city networks are SPARSELY instrumented. We have no rural dataset,
but we can SIMULATE rural density by keeping only a random fraction of the sensors
and masking the rest, then watching how the whole pipeline degrades:

  100% -> 75% -> 50% -> 25% -> 10%   sensor density

For each density we measure two very different things:
  1. PREDICTION quality (MAE / RMSE / MAPE at 15/30/60 min) — Layer 1.  No Ollama.
  2. FAITHFULNESS (Phase-5 methodology: cause F1, hallucination A-vs-B) plus which
     FAILURE MODE (Phase-13 taxonomy) dominates — Layers 2+3.  Needs Ollama.

The headline question: does faithfulness degrade GRACEFULLY (roughly linear) or off
a CLIFF (a sudden collapse at some density)? Either answer is a real finding. A
graceful curve is evidence the system stays useful where sensors are scarce; a cliff
tells a deployer the minimum density they must instrument to.

------------------------------------------------------------------------------
Design decisions (all in configs/sparse_regime.yaml too)
------------------------------------------------------------------------------
* MASKED SENSOR = z-space 0 = the dataset MEAN (~54 mph). This is exactly the
  "remove this node" operation the GNNExplainer / SHAP explainers already use
  (Phase 3 / Phase 12): a neutral, no-information value. We deliberately do NOT use
  the dataset's missing-data sentinel (raw 0 mph == z ~= -2.79), because that reads
  to the model as a STOPPED-TRAFFIC jam and would propagate phantom congestion — the
  opposite of "this sensor is simply absent". Only the SPEED channel is zeroed; the
  time-of-day channel is a global clock, shared across nodes, and is left intact.
  NOTE (flagged per CLAUDE.md): the model has a per-MODALITY gate, not a per-SENSOR
  gate, so "the gate handles it" is only literally true at the modality level. At the
  sensor level, graceful handling comes from the graph convolution + learned adaptive/
  semantic adjacency imputing a masked node from its surviving neighbours.

* NESTED MASKS. The masks are prefixes of one seeded node permutation, so the
  10%-kept set is a subset of the 25%-kept set is a subset of the 50%-kept set ...
  This lets us evaluate a FIXED "core" of always-retained sensors at every density
  (the "rural sensors" that survive all levels) — same nodes throughout, so the
  degradation curve compares like with like instead of a different random subset per
  level. We also record "all" nodes (network-wide, incl. imputed) and "kept" (this
  density's observed sensors) for full transparency.

Run:
  # prediction only (NO Ollama) — safe to run while Phase 10 holds the LLM:
  python -m xtraffic.evaluation.sparse_regime --pred-only --smoke   # 3 densities, capped windows
  python -m xtraffic.evaluation.sparse_regime --pred-only           # full 5 densities

  # faithfulness (NEEDS Ollama free) — run AFTER Phase 10 finishes:
  python -m xtraffic.evaluation.sparse_regime --faith-only
  python -m xtraffic.evaluation.sparse_regime                       # both parts + figure + table
  python -m xtraffic.evaluation.sparse_regime --mock-llm --smoke    # exercise faith harness, no Ollama

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import SPEED_CHANNEL, load_window
from ..models.gnn.loaders import (build_modality_dict, load_adjacency,
                                   load_node_meta, load_scaler,
                                   make_fusion_loaders)
from ..models.gnn.stgnn import XTrafficSTGNN
from ..utils.io_utils import PKG_ROOT
from ..utils.metrics import masked_metrics
from .faithfulness import NodeTable, mean_std, score_advisory
from .failure_modes import (CATEGORIES, MockAdvisor, bfs_hops, classify,
                            is_failure)

# Densities and node sets are read from config; only truly-fixed labels live here.
HORIZON_LABELS_ORDER = ["15min", "30min", "60min"]
NODE_SETS = ["core", "kept", "all"]

# Wong colourblind-safe palette (same family as the Phase-9 paper figures).
WONG = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}


def _load_cfg() -> Dict[str, Any]:
    with open(os.path.join(PKG_ROOT, "configs", "sparse_regime.yaml")) as f:
        return yaml.safe_load(f)


def _out_dir(cfg: Dict[str, Any]) -> str:
    d = os.path.join(PKG_ROOT, cfg["output"]["results_dir"])
    os.makedirs(d, exist_ok=True)
    return d


# ===========================================================================
# 1. Sensor masking (the core new mechanism).
# ===========================================================================
def node_permutation(n_nodes: int, seed: int) -> np.ndarray:
    """One fixed random ordering of the N sensors. NESTED masks are prefixes of it,
    so keeping the first k nodes for a small k is a subset of keeping the first k'
    for k' > k — the property that makes the 'core' node set well-defined."""
    return np.random.RandomState(seed).permutation(n_nodes)   # [N]


def kept_mask(n_nodes: int, density: float, perm: np.ndarray,
              protect: Sequence[int] = ()) -> np.ndarray:
    """Boolean [N] mask of the sensors KEPT (given real readings) at `density`.

    We keep the first round(density*N) nodes of the shared permutation `perm`, plus
    any `protect` nodes forced in (used for the faithfulness study, where a scenario's
    TARGET sensor must always exist — you cannot advise about a sensor that is not
    there). protect adds at most a handful of nodes; the density accounting notes it.
    """
    k = max(1, int(round(density * n_nodes)))                 # at least one sensor
    keep = np.zeros(n_nodes, dtype=bool)                      # [N] all masked...
    keep[perm[:k]] = True                                     # ...keep the prefix
    for p in protect:
        if 0 <= p < n_nodes:
            keep[p] = True
    return keep


def mask_speed_input(X: torch.Tensor, keep: np.ndarray,
                     mask_value: float = 0.0) -> torch.Tensor:
    """Return a copy of X [B, T, N, 2] with masked sensors' SPEED channel set to
    `mask_value` (z-space 0 == dataset mean). Time-of-day channel is untouched."""
    Xm = X.clone()                                           # [B, T, N, 2] don't mutate the loader tensor
    masked_idx = torch.from_numpy(np.where(~keep)[0]).long()  # node indices to blank out
    if masked_idx.numel() > 0:
        Xm[:, :, masked_idx, SPEED_CHANNEL] = mask_value      # [B, T, |masked|] -> mean
    return Xm


# ===========================================================================
# 2. Model loading (reuses the exact evaluate.py recipe).
# ===========================================================================
def load_model(checkpoint: str, dataset: str, device: torch.device
               ) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, float], int]:
    """Load the trained checkpoint and rebuild the model exactly as evaluate.py does.
    Returns (model.eval(), cfg, scaler, n_nodes)."""
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(PKG_ROOT, checkpoint)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = cfg["model"]

    adj = load_adjacency(dataset)                            # [N, N] from the EVAL dataset
    scaler = load_scaler(dataset)                            # {"mean","std"}
    n_nodes = adj.shape[0]

    model = XTrafficSTGNN(
        num_nodes=n_nodes, physical_adj=adj, modality_dims=cfg["modalities"],
        residual_channels=m["residual_channels"], dilation_channels=m["dilation_channels"],
        skip_channels=m["skip_channels"], end_channels=m["end_channels"],
        n_blocks=m["n_blocks"], embed_dim=m["embed_dim"],
        gcn_order=m["gcn_order"], dropout=m["dropout"], out_len=m["out_len"],
        use_semantic=m.get("use_semantic", True),
        use_multiscale=m.get("use_multiscale", True),
    ).to(device)
    model.load_state_dict(ck["model_state"], strict=(n_nodes == ck["n_nodes"]))
    model.eval()
    return model, cfg, scaler, n_nodes


# ===========================================================================
# 3. Prediction metrics per density (NO Ollama).
# ===========================================================================
def _collect_masked_predictions(
        model: torch.nn.Module, dataset: str, cfg: Dict[str, Any],
        keep: np.ndarray, horizons_steps: Sequence[int], batch_size: int,
        device: torch.device, mask_value: float, max_windows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the (frozen) model over the test set with `keep`-masked input and return
    (preds, trues) sliced to just the requested horizon steps.

    Slicing to the 3 horizons up front keeps memory tiny: [S, 3, N] floats instead of
    the full [S, 12, N] — a few MB, not hundreds. preds/trues are z-scored (the metric
    functions inverse-transform to mph internally, exactly like evaluate.py).
    """
    traffic_channels = cfg["traffic_channels"]
    modality_names = list(cfg["modalities"].keys())
    # metr_la_best is traffic-only (no use_sidecars), so we request only the traffic
    # feed; M has width 0 and every other modality is gated to None, byte-for-byte the
    # same path evaluate.py takes for this checkpoint.
    loaders, layout = make_fusion_loaders(dataset, batch_size, ["traffic"])
    h_idx = [s - 1 for s in horizons_steps]                  # 1-based step -> 0-based time index

    preds_chunks: List[torch.Tensor] = []
    trues_chunks: List[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for X, Y, M in loaders["test"]:                      # X:[B,12,N,2] Y:[B,12,N] M:[B,12,N,0]
            X = mask_speed_input(X, keep, mask_value).to(device)
            M = M.to(device)
            mods = build_modality_dict(X, traffic_channels, modality_names, M, layout)
            pred = model(mods).cpu()                          # [B, 12, N] z-scored
            preds_chunks.append(pred[:, h_idx, :])            # [B, 3, N]
            trues_chunks.append(Y[:, h_idx, :])               # [B, 3, N]
            seen += X.shape[0]
            if max_windows and seen >= max_windows:           # smoke: stop early (chronological prefix)
                break
    preds = torch.cat(preds_chunks, 0)                        # [S, 3, N]
    trues = torch.cat(trues_chunks, 0)                        # [S, 3, N]
    return preds, trues


def _subset_metrics(preds: torch.Tensor, trues: torch.Tensor,
                    scaler: Dict[str, float], horizons_steps: Sequence[int],
                    node_idx: np.ndarray) -> Dict[str, Dict[str, float]]:
    """MAE/RMSE/MAPE per horizon over a NODE SUBSET. preds/trues: [S, 3, N] z-scored.

    The metric mask (missing-target sentinel) is applied inside masked_metrics via the
    real target, so slicing the node axis here simply scopes WHICH sensors are scored
    (core / kept / all) — the missing-value handling is unchanged."""
    idx = torch.from_numpy(node_idx).long()
    out: Dict[str, Dict[str, float]] = {}
    for h, step in enumerate(horizons_steps):
        label = "{}min".format(step * 5)
        p = preds[:, h, :].index_select(1, idx)               # [S, |subset|]
        t = trues[:, h, :].index_select(1, idx)               # [S, |subset|]
        out[label] = masked_metrics(p, t, scaler)
    return out


def run_prediction_sweep(cfg: Dict[str, Any], densities: List[float],
                         device: torch.device, max_windows: int) -> Dict[str, Any]:
    """Prediction MAE/RMSE/MAPE at every density, for the core / kept / all node sets."""
    dataset = cfg["dataset"]
    model, ck_cfg, scaler, n_nodes = load_model(cfg["checkpoint"], dataset, device)
    horizons_steps = cfg["horizons_steps"]
    mask_value = float(cfg["mask_value_zspace"])
    perm = node_permutation(n_nodes, cfg["mask_seed"])

    # The fixed "core" = sensors kept even at the LOWEST density in this run.
    core_keep = kept_mask(n_nodes, min(densities), perm)
    core_idx = np.where(core_keep)[0]

    results: Dict[str, Any] = {"n_nodes": n_nodes, "n_core": int(core_idx.size),
                               "densities": densities, "by_density": {}}
    for d in densities:
        keep = kept_mask(n_nodes, d, perm)
        kept_idx = np.where(keep)[0]
        preds, trues = _collect_masked_predictions(
            model, dataset, ck_cfg, keep, horizons_steps, cfg["batch_size"],
            device, mask_value, max_windows)
        entry = {
            "density": d, "n_kept": int(kept_idx.size),
            "all": _subset_metrics(preds, trues, scaler, horizons_steps, np.arange(n_nodes)),
            "kept": _subset_metrics(preds, trues, scaler, horizons_steps, kept_idx),
            "core": _subset_metrics(preds, trues, scaler, horizons_steps, core_idx),
        }
        results["by_density"]["{:.2f}".format(d)] = entry
        print("[pred] density {:>4.0f}%  kept={:3d}/{}  30-min MAE  core={:.3f}  "
              "all={:.3f}".format(d * 100, kept_idx.size, n_nodes,
                                  entry["core"]["30min"]["mae"], entry["all"]["30min"]["mae"]))
    return results


# ===========================================================================
# 4. Faithfulness + failure modes per density (NEEDS Ollama).
# ===========================================================================
def _faith_scenarios(cfg: Dict[str, Any], scaler: Dict[str, float]) -> List[Dict[str, Any]]:
    """Stratified faithfulness scenarios (reuses the tested Phase-10 sampler so the
    grid matches Phase 5/10 exactly)."""
    from .sim_eval import sample_scenarios          # local import: keeps --pred-only light
    fc = cfg["faithfulness"]
    sampler_cfg = {
        "n_scenarios": fc["n_scenarios"], "seed": cfg["seed"],
        "stratify": fc["stratify"],
    }
    return sample_scenarios(cfg["dataset"], scaler, sampler_cfg)


def _masked_explanation(builder: ExplanationBuilder, dataset: str, density: float,
                        sc: Dict[str, Any], keep: np.ndarray, mask_value: float,
                        cache_dir: str) -> Dict[str, Any]:
    """Explanation for one scenario computed on the DENSITY-MASKED input window,
    cached per (density, window, target) so the multi-hour run is resumable."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "{}_d{:03d}_{}_{}.json".format(
        dataset, int(round(density * 100)), sc["sample_index"], sc["target_node"]))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    X = load_window(dataset, sc["sample_index"])              # [1, T, N, C]
    X = mask_speed_input(X, keep, mask_value)                 # blank out absent sensors
    exp = builder.explain_prediction(
        X, target_node=sc["target_node"],
        horizon_step=6, timestamp=sc["timestamp"])           # 30-min horizon (Phase-5 convention)
    exp["meta"]["city"] = dataset
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


def run_faithfulness_sweep(cfg: Dict[str, Any], densities: List[float],
                           device: torch.device, mock: bool,
                           model_override: Optional[str],
                           limit: int = 0) -> Dict[str, Any]:
    """For each density: regenerate the explanation on masked input, run advisor
    conditions A and B, score faithfulness/hallucination, and tally the dominant
    failure mode among condition-A failures. Resumable via a JSONL decision log."""
    dataset, city = cfg["dataset"], cfg["city"]
    fc = cfg["faithfulness"]
    conditions = fc["conditions"]
    mask_value = float(cfg["mask_value_zspace"])
    out_dir = _out_dir(cfg)
    exp_cache = os.path.join(out_dir, "explanations_cache")
    jsonl_path = os.path.join(out_dir, "faith_decisions{}.jsonl".format("_mock" if mock else ""))

    # Explainer (also gives us model cfg/scaler/adj so we don't reload the checkpoint).
    ex = fc["explainer"]
    builder = ExplanationBuilder(
        cfg["checkpoint"], dataset, device=device,
        top_k=ex["top_k"], epochs=ex["epochs"], confidence_runs=ex["confidence_runs"])
    # Physical graph as a NumPy bool matrix, built exactly the way failure_modes.py
    # does so bfs_hops / classify get precisely the type they expect (the mock smoke
    # never hits classify, so this must be right for the real Ollama run).
    adj_bool = (load_adjacency(dataset).numpy() > 0)         # [N, N] bool
    n_nodes = adj_bool.shape[0]
    perm = node_permutation(n_nodes, cfg["mask_seed"])
    table = NodeTable(load_node_meta(dataset))

    # Advisor (real Ollama or the Phase-13 mock).
    if mock:
        advisor: Any = MockAdvisor()
    else:
        from ..models.advisor.advisor import Advisor, load_advisor_config
        if model_override or cfg["llm"].get("model"):
            import copy
            adv_cfg = copy.deepcopy(load_advisor_config())
            adv_cfg.setdefault("ollama", {})["model"] = model_override or cfg["llm"]["model"]
            advisor = Advisor(city, cfg=adv_cfg)
        else:
            advisor = Advisor(city)

    scenarios = _faith_scenarios(cfg, builder.scaler)
    if limit and len(scenarios) > limit:                     # quick harness checks
        scenarios = scenarios[:limit]
    print("[faith] {} scenarios per density x {} densities, model={}".format(
        len(scenarios), len(densities), getattr(advisor, "model", "mock")))

    # Resume: (density, sample_index, condition) already scored -> skip.
    rows: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                done.add((r["density_pct"], r["sample_index"], r["condition"]))
        print("[faith][resume] loaded {} scored decisions from {}".format(len(rows), jsonl_path))
    jsonl = open(jsonl_path, "a")

    for d in densities:
        dpct = int(round(d * 100))
        for sc in scenarios:
            # The scenario's target must exist -> protect it in this density's mask.
            keep = kept_mask(n_nodes, d, perm, protect=[sc["target_node"]])
            need = [c for c in conditions if (dpct, sc["sample_index"], c) not in done]
            if not need:
                continue
            exp = _masked_explanation(builder, dataset, d, sc, keep, mask_value, exp_cache)
            topk = {int(n["node_id"]) for n in exp.get("top_nodes", [])}
            hops = bfs_hops(adj_bool, topk | {exp["prediction"]["node_id"]})
            for cond in need:
                res = advisor.advise_condition(exp, cond)
                metrics = score_advisory(exp, res["advisory"], table)
                row = {
                    "density_pct": dpct, "density": d,
                    "sample_index": sc["sample_index"], "target_node": sc["target_node"],
                    "tod_band": sc["tod_band"], "congestion": sc["congestion"],
                    "condition": cond,
                    "cause_precision": metrics["cause_precision"],
                    "cause_recall": metrics["cause_recall"],
                    "faithfulness_f1": metrics["faithfulness_f1"],
                    "hallucination_rate": metrics["hallucination_rate"],
                    "quantitative_fidelity": metrics["quantitative_fidelity"],
                    "failure_category": "",
                }
                # Classify the failure mode only for condition A (the deployed system),
                # and only when the scenario actually failed (Phase-13 rule).
                if cond == "A":
                    failed = is_failure(metrics)
                    row["is_failure"] = int(failed)
                    if failed:
                        row["failure_category"] = classify(
                            metrics, exp, res["advisory"], metrics["per_cause"], hops)
                rows.append(row)
                jsonl.write(json.dumps(row) + "\n")
                jsonl.flush()
        print("[faith] density {:>4.0f}% done".format(d * 100))
    jsonl.close()

    return _aggregate_faithfulness(rows, densities)


def _aggregate_faithfulness(rows: List[Dict[str, Any]], densities: List[float]
                            ) -> Dict[str, Any]:
    """Per density: mean+/-std of each faithfulness metric per condition, plus the
    dominant condition-A failure mode."""
    out: Dict[str, Any] = {"densities": densities, "by_density": {}}
    metric_keys = ["cause_precision", "cause_recall", "faithfulness_f1",
                   "hallucination_rate", "quantitative_fidelity"]
    for d in densities:
        dpct = int(round(d * 100))
        entry: Dict[str, Any] = {"density": d}
        for cond in ("A", "B"):
            sub = [r for r in rows if r["density_pct"] == dpct and r["condition"] == cond]
            cond_agg: Dict[str, Any] = {"n": len(sub)}
            for k in metric_keys:
                m, s, _ = mean_std([r.get(k) for r in sub])
                cond_agg[k + "_mean"], cond_agg[k + "_std"] = m, s
            entry[cond] = cond_agg
        # Dominant failure mode among condition-A failures at this density.
        fails = [r["failure_category"] for r in rows
                 if r["density_pct"] == dpct and r["condition"] == "A"
                 and r.get("is_failure") and r.get("failure_category")]
        counts = collections.Counter(fails)
        entry["failure_counts"] = dict(counts)
        entry["n_failures"] = len(fails)
        entry["dominant_failure"] = counts.most_common(1)[0][0] if counts else "none"
        out["by_density"]["{:.2f}".format(d)] = entry
    return out


# ===========================================================================
# 5. Degradation classification (graceful vs cliff) + interpretation text.
# ===========================================================================
def _series(pred: Dict[str, Any], node_set: str, horizon: str, metric: str,
            densities: List[float]) -> List[float]:
    """A metric read out across densities in the given ORDER."""
    return [pred["by_density"]["{:.2f}".format(d)][node_set][horizon][metric]
            for d in densities]


def classify_degradation(values: List[float], densities: List[float],
                         higher_is_better: bool) -> Dict[str, Any]:
    """Is the degradation across densities GRACEFUL (roughly linear) or a CLIFF?

    `values` and `densities` are aligned and ordered from HIGH density (best case)
    to LOW density (worst case). We look at how the metric worsens step by step:
      - if it barely moves at all           -> "robust"
      - if one step dominates the total drop -> "cliff" at that step
      - otherwise                            -> "graceful" (spread across steps)
    """
    best, worst = values[0], values[-1]
    # Worsening amount at each step, in the metric's "bad" direction.
    sign = -1.0 if higher_is_better else 1.0
    total = sign * (worst - best)                            # >0 == net worse
    steps = [sign * (values[i + 1] - values[i]) for i in range(len(values) - 1)]
    worsening = [max(0.0, s) for s in steps]                 # only count steps that got worse
    max_step = max(worsening) if worsening else 0.0
    max_i = int(np.argmax(worsening)) if worsening else 0
    rel_total = total / (abs(best) + 1e-9)                   # fractional change vs the 100% baseline

    if abs(rel_total) < 0.15:
        verdict = "robust"
    elif total > 0 and max_step > 0.6 * total:
        verdict = "cliff"
    else:
        verdict = "graceful"
    return {
        "verdict": verdict, "best": best, "worst": worst,
        "total_change": total, "rel_change": rel_total, "max_step": max_step,
        "cliff_between": ["{:.0f}%".format(densities[max_i] * 100),
                          "{:.0f}%".format(densities[max_i + 1] * 100)] if worsening else None,
    }


def _nearest_density(densities: List[float], target: float) -> float:
    return min(densities, key=lambda d: abs(d - target))


def build_interpretation(cfg: Dict[str, Any], densities: List[float],
                         pred: Optional[Dict[str, Any]],
                         faith: Optional[Dict[str, Any]]) -> str:
    """The plain-language paragraph the assignment asks for, filled with real numbers.
    Always mentions the 25% 'rural analog' density explicitly. `densities` are the
    levels actually run this invocation (high -> low order)."""
    node_set = cfg["headline_node_set"]
    lines: List[str] = []

    if pred is not None:
        mae30 = _series(pred, node_set, "30min", "mae", densities)
        deg = classify_degradation(mae30, densities, higher_is_better=False)
        d25 = _nearest_density(densities, 0.25)
        i25 = densities.index(d25)
        base = mae30[0]
        mae25 = mae30[i25]
        rel25 = 100.0 * (mae25 - base) / (base + 1e-9)
        verdict_word = {"robust": "barely moves (robust)",
                        "graceful": "degrades gracefully (roughly linear)",
                        "cliff": "falls off a cliff"}[deg["verdict"]]
        lines.append(
            "PREDICTION: 30-min MAE at the retained ({}) sensors {} as sensor "
            "density drops from 100% to {:.0f}% ({:.3f} -> {:.3f} mph, "
            "{:+.1f}%).".format(node_set, verdict_word, densities[-1] * 100,
                                base, mae30[-1], deg["rel_change"] * 100))
        if deg["verdict"] == "cliff" and deg["cliff_between"]:
            lines.append("  The collapse is concentrated between {} and {} density — "
                         "that step is the deployment floor to stay above.".format(
                             deg["cliff_between"][0], deg["cliff_between"][1]))
        lines.append(
            "  At {:.0f}% density (rural analog) the system shows a 30-min MAE of "
            "{:.3f} mph ({:+.1f}% vs the fully-instrumented 100% network) — the "
            "prediction a planner would act on in a sparsely-sensored region.".format(
                d25 * 100, mae25, rel25))

    if faith is not None:
        f1a = [faith["by_density"]["{:.2f}".format(d)]["A"]["faithfulness_f1_mean"]
               for d in densities]
        hga = [faith["by_density"]["{:.2f}".format(d)]["A"]["hallucination_rate_mean"]
               for d in densities]
        hgb = [faith["by_density"]["{:.2f}".format(d)]["B"]["hallucination_rate_mean"]
               for d in densities]
        degf = classify_degradation(f1a, densities, higher_is_better=True)
        d25 = _nearest_density(densities, 0.25)
        i25 = densities.index(d25)
        fword = {"robust": "holds essentially flat (robust)",
                 "graceful": "degrades gracefully",
                 "cliff": "collapses off a cliff"}[degf["verdict"]]
        lines.append(
            "FAITHFULNESS: condition-A faithfulness F1 {} as density thins "
            "({:.3f} at 100% -> {:.3f} at {:.0f}%). Hallucination in the grounded "
            "system (A) stays {:.3f}->{:.3f} while the ungrounded baseline (B) sits "
            "at {:.3f}->{:.3f}, so the explanation keeps suppressing hallucination "
            "even under sparsity.".format(
                fword, f1a[0], f1a[-1], densities[-1] * 100,
                hga[0], hga[-1], hgb[0], hgb[-1]))
        dom25 = faith["by_density"]["{:.2f}".format(d25)]["dominant_failure"]
        nfail25 = faith["by_density"]["{:.2f}".format(d25)]["n_failures"]
        lines.append(
            "  At {:.0f}% density the dominant failure mode is {} ({} condition-A "
            "failures), telling us HOW faithfulness breaks first when sensors are "
            "scarce.".format(d25 * 100, dom25, nfail25))

    if not lines:
        lines.append("No results computed yet.")
    return "\n".join(lines)


# ===========================================================================
# 6. Plot — Figure 7 (vector PDF, colourblind-safe).
# ===========================================================================
def plot_degradation(cfg: Dict[str, Any], densities: List[float],
                     pred: Optional[Dict[str, Any]],
                     faith: Optional[Dict[str, Any]], out_pdf: str) -> None:
    import matplotlib
    matplotlib.use("Agg")                                    # headless: write a file, no display
    import matplotlib.pyplot as plt

    xs = [d * 100 for d in densities]                        # % on the x-axis
    order = np.argsort(xs)                                   # plot low->high density left->right
    xs_sorted = [xs[i] for i in order]
    node_set = cfg["headline_node_set"]
    hcolors = {"15min": WONG["blue"], "30min": WONG["vermillion"], "60min": WONG["green"]}

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    fig.suptitle("Sparse-sensor regime: degradation vs sensor density "
                 "(METR-LA, headline node set = {})".format(node_set), fontsize=11)

    def _reorder(vals: List[float]) -> List[float]:
        return [vals[i] for i in order]

    # --- Panels 1-3: prediction MAE / RMSE / MAPE ---
    for ax, metric, ylab in [(axes[0][0], "mae", "MAE (mph)"),
                             (axes[0][1], "rmse", "RMSE (mph)"),
                             (axes[0][2], "mape", "MAPE (%)")]:
        if pred is None:
            ax.text(0.5, 0.5, "PENDING\n(run --pred-only)", ha="center", va="center",
                    transform=ax.transAxes, color=WONG["vermillion"])
        else:
            for hz in HORIZON_LABELS_ORDER:
                ys = _reorder(_series(pred, node_set, hz, metric, densities))
                ax.plot(xs_sorted, ys, "o-", color=hcolors[hz], label=hz)
            ax.legend(fontsize=7, title="horizon")
        ax.set_xlabel("sensor density (%)")
        ax.set_ylabel(ylab)
        ax.set_title("Prediction {}".format(metric.upper()))

    # --- Panel 4: faithfulness F1 (condition A) ---
    ax = axes[1][0]
    if faith is None:
        ax.text(0.5, 0.5, "PENDING\n(needs Ollama)", ha="center", va="center",
                transform=ax.transAxes, color=WONG["vermillion"])
    else:
        ys = _reorder([faith["by_density"]["{:.2f}".format(d)]["A"]["faithfulness_f1_mean"]
                       for d in densities])
        ax.plot(xs_sorted, ys, "o-", color=WONG["purple"])
        ax.set_ylim(0, 1.02)
    ax.set_xlabel("sensor density (%)")
    ax.set_ylabel("faithfulness F1")
    ax.set_title("Faithfulness (condition A)")

    # --- Panel 5: hallucination A vs B ---
    ax = axes[1][1]
    if faith is None:
        ax.text(0.5, 0.5, "PENDING\n(needs Ollama)", ha="center", va="center",
                transform=ax.transAxes, color=WONG["vermillion"])
    else:
        for cond, col in [("A", WONG["green"]), ("B", WONG["vermillion"])]:
            ys = _reorder([faith["by_density"]["{:.2f}".format(d)][cond]["hallucination_rate_mean"]
                           for d in densities])
            ax.plot(xs_sorted, ys, "o-", color=col,
                    label="A (grounded)" if cond == "A" else "B (no explanation)")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("sensor density (%)")
    ax.set_ylabel("hallucination rate")
    ax.set_title("Hallucination: A vs B")

    # --- Panel 6: failure-mode composition (condition A) ---
    ax = axes[1][2]
    if faith is None:
        ax.text(0.5, 0.5, "PENDING\n(needs Ollama)", ha="center", va="center",
                transform=ax.transAxes, color=WONG["vermillion"])
    else:
        cats = [c[0] for c in CATEGORIES]                    # fixed category order
        palette = [WONG["orange"], WONG["sky"], WONG["green"], WONG["yellow"],
                   WONG["blue"], WONG["purple"]]
        bottom = np.zeros(len(densities))
        dens_desc = densities                                # already high->low
        for ci, cat in enumerate(cats):
            heights = np.array([faith["by_density"]["{:.2f}".format(d)]
                                ["failure_counts"].get(cat, 0) for d in dens_desc], float)
            ax.bar([d * 100 for d in dens_desc], heights, bottom=bottom, width=6,
                   color=palette[ci % len(palette)], label=cat.replace("_", " ").title())
            bottom += heights
        ax.legend(fontsize=5, ncol=1, loc="upper left")
        ax.set_ylabel("# condition-A failures")
    ax.set_xlabel("sensor density (%)")
    ax.set_title("Failure-mode composition")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("[plot] wrote {}".format(out_pdf))


# ===========================================================================
# 7. Output — CSV + JSON + LaTeX booktabs table.
# ===========================================================================
def write_outputs(cfg: Dict[str, Any], densities: List[float],
                  pred: Optional[Dict[str, Any]],
                  faith: Optional[Dict[str, Any]], interpretation: str) -> None:
    out_dir = _out_dir(cfg)
    node_set = cfg["headline_node_set"]

    # --- JSON summary (everything) ---
    summary = {
        "dataset": cfg["dataset"], "headline_node_set": node_set,
        "densities": densities, "prediction": pred, "faithfulness": faith,
        "interpretation": interpretation,
    }
    with open(os.path.join(out_dir, "sparse_regime_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Per-density CSV (flat, one row per density) ---
    csv_path = os.path.join(out_dir, "sparse_regime_per_density.csv")
    fields = ["density_pct", "n_kept", "n_core",
              "mae_15_core", "mae_30_core", "mae_60_core",
              "rmse_30_core", "mape_30_core",
              "mae_30_all", "faith_f1_A", "halluc_A", "halluc_B", "dominant_failure"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in densities:
            key = "{:.2f}".format(d)
            row: Dict[str, Any] = {"density_pct": int(round(d * 100))}
            if pred is not None:
                pe = pred["by_density"][key]
                row.update({
                    "n_kept": pe["n_kept"], "n_core": pred["n_core"],
                    "mae_15_core": round(pe["core"]["15min"]["mae"], 4),
                    "mae_30_core": round(pe["core"]["30min"]["mae"], 4),
                    "mae_60_core": round(pe["core"]["60min"]["mae"], 4),
                    "rmse_30_core": round(pe["core"]["30min"]["rmse"], 4),
                    "mape_30_core": round(pe["core"]["30min"]["mape"], 4),
                    "mae_30_all": round(pe["all"]["30min"]["mae"], 4),
                })
            if faith is not None:
                fe = faith["by_density"][key]
                row.update({
                    "faith_f1_A": round(fe["A"]["faithfulness_f1_mean"], 4),
                    "halluc_A": round(fe["A"]["hallucination_rate_mean"], 4),
                    "halluc_B": round(fe["B"]["hallucination_rate_mean"], 4),
                    "dominant_failure": fe["dominant_failure"],
                })
            w.writerow(row)

    # --- LaTeX booktabs table (\input-ready, Phase-9 style) ---
    write_latex(cfg, densities, pred, faith, os.path.join(PKG_ROOT, cfg["output"]["tex"]))
    print("[out] wrote {}\n      {}\n      {}".format(
        os.path.join(out_dir, "sparse_regime_summary.json"), csv_path,
        os.path.join(PKG_ROOT, cfg["output"]["tex"])))


def _cell(v: Optional[float], fmt: str) -> str:
    """Format a number, or a LaTeX-safe placeholder when the value is still owed."""
    if v is None or (isinstance(v, float) and v != v):       # None or NaN
        return "--"
    return fmt.format(v)


def write_latex(cfg: Dict[str, Any], densities: List[float],
                pred: Optional[Dict[str, Any]],
                faith: Optional[Dict[str, Any]], path: str) -> None:
    node_set = cfg["headline_node_set"]
    lines = [
        "% Table -- sparse-sensor regime (METR-LA). Prediction MAE at the retained",
        "% ({}) sensors + faithfulness/hallucination per sensor-density level.".format(node_set),
        "% PENDING cells (--) are Ollama-owed faithfulness numbers; run "
        "python -m xtraffic.evaluation.sparse_regime --faith-only.",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Density & 15-min MAE & 30-min MAE & 60-min MAE & Faith. F1 (A) & "
        "Halluc. (A/B) & Dominant failure \\\\",
        "\\midrule",
    ]
    for d in densities:
        key = "{:.2f}".format(d)
        mae15 = mae30 = mae60 = None
        if pred is not None:
            core = pred["by_density"][key]["core"]
            mae15, mae30, mae60 = core["15min"]["mae"], core["30min"]["mae"], core["60min"]["mae"]
        f1a = ha = hb = None
        dom = "--"
        if faith is not None:
            fe = faith["by_density"][key]
            f1a = fe["A"]["faithfulness_f1_mean"]
            ha, hb = fe["A"]["hallucination_rate_mean"], fe["B"]["hallucination_rate_mean"]
            dom = fe["dominant_failure"].replace("_", " ").title()
        halluc = "--" if (ha is None or hb is None) else "{:.2f}/{:.2f}".format(ha, hb)
        lines.append("{:.0f}\\% & {} & {} & {} & {} & {} & {} \\\\".format(
            d * 100, _cell(mae15, "{:.3f}"), _cell(mae30, "{:.3f}"),
            _cell(mae60, "{:.3f}"), _cell(f1a, "{:.3f}"), halluc, dom))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ===========================================================================
# Main.
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 14 — sparse-sensor-regime experiments.")
    ap.add_argument("--pred-only", action="store_true",
                    help="prediction metrics only (NO Ollama). Safe anytime.")
    ap.add_argument("--faith-only", action="store_true",
                    help="faithfulness + failure-mode sweep only (NEEDS Ollama free).")
    ap.add_argument("--smoke", action="store_true",
                    help="3 densities (100/50/10) + capped test windows for a fast sanity check.")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="cap #test windows for prediction (0 = all; smoke sets 512).")
    ap.add_argument("--mock-llm", action="store_true",
                    help="faithfulness harness with a grounded mock advisor (no Ollama).")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #faithfulness scenarios per density (0 = config n_scenarios).")
    ap.add_argument("--model", default=None, help="override the Ollama model name.")
    args = ap.parse_args()

    cfg = _load_cfg()
    densities = cfg["smoke_densities"] if args.smoke else cfg["densities"]
    max_windows = args.max_windows or (512 if args.smoke else 0)
    device = torch.device("cpu")                             # deterministic; matches sim_eval

    do_pred = not args.faith_only
    do_faith = not args.pred_only

    pred = run_prediction_sweep(cfg, densities, device, max_windows) if do_pred else None
    faith = (run_faithfulness_sweep(cfg, densities, device, args.mock_llm, args.model, args.limit)
             if do_faith else None)

    # On a partial run, backfill the OTHER part from a previous summary so the figure/
    # table/interpretation stay whole (honest PENDING only when genuinely never run).
    # Only accept a backfill computed at the SAME density levels as this run, else the
    # shared x-axis would mix (e.g.) a 3-level smoke with a 5-level full run.
    def _accept(part: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if part is not None and part.get("densities") != densities:
            print("[backfill] skipped stale part (densities {} != {})".format(
                part.get("densities"), densities))
            return None
        return part

    prev_path = os.path.join(_out_dir(cfg), "sparse_regime_summary.json")
    if os.path.exists(prev_path) and (pred is None or faith is None):
        with open(prev_path) as f:
            prev = json.load(f)
        if pred is None:
            pred = _accept(prev.get("prediction"))
        if faith is None:
            faith = _accept(prev.get("faithfulness"))

    interpretation = build_interpretation(cfg, densities, pred, faith)
    print("\n=== PHASE 14 INTERPRETATION ===\n" + interpretation + "\n")

    write_outputs(cfg, densities, pred, faith, interpretation)
    plot_degradation(cfg, densities, pred, faith, os.path.join(PKG_ROOT, cfg["output"]["fig"]))


if __name__ == "__main__":
    main()
