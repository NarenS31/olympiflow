"""Phase 19 Step 3 — CROSS-DOMAIN faithfulness: does the grounding result hold on
a POWER GRID?

THE CLAIM UNDER TEST
--------------------
Phase 5 showed, on METR-LA, that the mathematical explanation is what stops the
LLM inventing causes: condition A (full pipeline) hallucinated 0.005 of its cited
causes, condition B (prediction only, no explanation) hallucinated 0.828. Phase 11
showed that survives swapping the LLM family (mistral) and the city (PEMS-BAY,
zero-shot). Every one of those is still a road network.

This module runs the SAME A/B protocol on the IEEE 14-bus power grid. If the gap
holds, the claim stops being "explanation grounding works for traffic" and becomes
"mathematical GNN-explanation grounding is a DOMAIN-AGNOSTIC mechanism for
eliminating LLM hallucination" — the strongest available framing of Contribution
#2, and the reason Phase 19 exists.

WHAT PORTED AND WHAT DID NOT (be precise about this; it IS the result)
---------------------------------------------------------------------
Ported with ZERO changes: XTrafficSTGNN, GNNExplainer (models/explainer/explain.py),
the explanation schema, the advisory JSON contract, the faithfulness metric math
(score_advisory), and the prompt STRUCTURE including the grounding instruction.
Those carry the science.

Needed adaptation, all of it labels or sampling, none of it mechanism:
  1. VOCABULARY — models/advisor/domains.py. The Phase-4 prompt is hard-coded
     traffic; pointed at a grid it says "Current speed: 0.88 mph" for a bus
     voltage. DIFFERENCES_POWER_GRID.md §8.2 predicted this. Left unfixed it is a
     CONFOUND, not a cosmetic bug: an LLM told a substation is doing 0.88 mph will
     invent traffic interventions, and we would then score our own prompt's
     confusion as the model's hallucination.
  2. ENTITY RESOLUTION — faithfulness.NodeTable grows a non-geographic ladder
     (bus id -> electrical zone -> fuzzy zone). A circuit has no lat/lon, so the
     coordinate path cannot run. Structurally the same ladder, different notion of
     "group". Both changes are default-preserving and pinned by
     evaluation/verify_traffic_unchanged.py.
  3. SCENARIO SAMPLING — "congested" does not port. See sample_undervoltage_
     scenarios below; this is the subtlest part of the phase.
  4. top_k 8 -> 4, because k/N differs by 15x between the two graphs. See the
     config comment and the CHANCE BASELINES below — we measure this rather than
     asserting it away.

THE CHANCE-BASELINE POINT (read before quoting any number from this study)
-------------------------------------------------------------------------
METR-LA has 207 nodes and marks 8 as causal: a random citation hits the top-k
about 4% of the time, so condition B's 0.83 hallucination rate is close to the
0.96 a pure guesser would score. The grid has 14 buses and marks 4: a random BUS
citation hits ~31% of the time, and a random ZONE citation far more, because one
zone spans up to 7 buses. A hallucination rate of 0.5 on the grid is therefore NOT
comparable to 0.5 on METR-LA.

So this study computes the chance rates directly (no LLM: draw citations uniformly
from the same vocabulary and score them with the same metric) and reports every
condition against them. Without that control the cross-domain comparison would be
an artifact of graph size, and a reviewer would be right to say so.

Run:
  python -m xtraffic.evaluation.power_grid_faithfulness --smoke     # 3 scenarios
  python -m xtraffic.evaluation.power_grid_faithfulness             # the full n=20
  python -m xtraffic.evaluation.power_grid_faithfulness --explain-only
  python -m xtraffic.evaluation.power_grid_faithfulness --mock-llm  # no Ollama

Resumable: every (scenario, condition) decision is cached to disk, so a rerun
skips completed work. Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from ..models.advisor.advisor import Advisor
from ..models.explainer.explain import ExplanationBuilder
from ..models.explainer.scenarios import TOD_CHANNEL, _tod_to_clock
from ..models.gnn.loaders import _load_split, load_node_meta, load_scaler
from ..utils.io_utils import PKG_ROOT
from .faithfulness import NodeTable, mean_std, score_advisory

SPEED_CHANNEL = 0        # channel 0 is the z-scored state = VOLTAGE here

# Same time-of-day bands as the Phase-5 traffic study. A power grid has a daily
# demand cycle too (DIFFERENCES §5.1: peaks at 08:00 / 13:00 / 19:00), so the
# bands are meaningful, and reusing them keeps the two studies comparable.
TOD_BANDS: List[Tuple[str, float, float]] = [
    ("night", 0.00, 0.25),
    ("am_rush", 0.25, 0.42),
    ("midday", 0.42, 0.62),
    ("pm_rush", 0.62, 0.83),
    ("evening", 0.83, 1.00),
]
SEVERITY_LEVELS = ["mild", "moderate", "severe"]


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or os.path.join(PKG_ROOT, "configs", "power_grid_faith.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Scenario sampling — the traffic rule does NOT port, so this is built from the
# domain up rather than copied.
# ---------------------------------------------------------------------------
def bus_reference_voltages(dataset: str, scaler: Dict[str, float]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Each bus's NORMAL voltage and its variability, from the TRAIN split only.

    Train-only for the same reason Phase 1 z-scores on train statistics: using the
    test split to define what "normal" means would leak test information into
    scenario selection. Returns (mean_pu[N], std_pu[N]).
    """
    X, _ = _load_split(dataset, "train")                  # [S, T, N, C]
    v = X[..., SPEED_CHANNEL].numpy() * scaler["std"] + scaler["mean"]  # [S,T,N] pu
    flat = v.reshape(-1, v.shape[2])                      # [S*T, N]
    return flat.mean(0), flat.std(0)


def sample_undervoltage_scenarios(dataset: str, scaler: Dict[str, float],
                                  n: int, min_dev: float, min_std: float,
                                  seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pick `n` windows where some bus is genuinely sagging, stratified.

    WHY NOT THE TRAFFIC RULE. Phase 3/5 pick the SLOWEST VALID SENSOR in the
    window, because on a road network every sensor shares roughly one free-flow
    speed, so "slowest" means "most congested". A grid has no such shared
    reference: bus 3 normally sits at 1.008 pu and bus 8 at 1.089 pu, so the
    lowest-voltage bus is bus 3 in 1733 of 1995 test windows regardless of whether
    anything is wrong. Copying the traffic rule would have produced a 20-scenario
    "study" of one bus, and the flatness would have looked like a finding.

    WHAT WE DO INSTEAD. Deviation of each bus from ITS OWN train-mean voltage:
        dev[i] = ref[i] - v_last[i]        (positive = below its own normal)
    the target is the bus with the largest deviation, and the window qualifies only
    if that deviation clears `min_dev`. This is the real analogue of "this road is
    below ITS free-flow speed", and it yields targets spread across buses
    3, 5, 9, 10, 12, 13, 14 instead of one.

    Buses with train std below `min_std` are ineligible as targets: that is bus 1,
    the slack bus, which by construction holds the reference voltage (std 2.0e-4).
    Its deviation is pure numerical noise, but it is still an argmax about half the
    time when nothing else is sagging, which would silently hijack the sample.

    Stratified over (time-of-day band x severity tercile) so the study is not all
    evening-peak, seeded for reproducibility. Also returns a diagnostics dict —
    what was available and what was actually taken — because a stratified sample
    that quietly failed to fill its cells is a silent coverage claim.
    """
    X, _ = _load_split(dataset, "test")                   # [S, T, N, C]
    tod0 = X[:, 0, 0, TOD_CHANNEL].numpy()                # clock at window start [S]
    v = X[..., SPEED_CHANNEL].numpy() * scaler["std"] + scaler["mean"]  # [S,T,N] pu
    ref, sd = bus_reference_voltages(dataset, scaler)     # [N], [N]

    eligible = sd >= min_std                              # [N] bool
    last = v[:, -1, :]                                    # last OBSERVED step [S,N]
    dev = (ref[None, :] - last) * eligible[None, :]       # [S,N] sag below own normal
    target = dev.argmax(1)                                # [S]
    max_dev = dev.max(1)                                  # [S]

    stressed = np.where(max_dev >= min_dev)[0]
    diag: Dict[str, Any] = {
        "test_windows": int(len(max_dev)),
        "ineligible_buses": [int(i) + 1 for i in range(len(sd)) if not eligible[i]],
        "min_deviation_pu": min_dev,
        "n_stressed_windows": int(len(stressed)),
    }
    if len(stressed) == 0:
        return [], diag

    # Severity terciles cut on the STRESSED windows only (cutting on all windows
    # would put every stressed window in the top tercile and defeat the point).
    q33, q66 = np.quantile(max_dev[stressed], [1 / 3, 2 / 3])
    diag["severity_cuts_pu"] = [float(round(q33, 5)), float(round(q66, 5))]

    def severity_of(d: float) -> str:
        if d <= q33:
            return "mild"
        if d <= q66:
            return "moderate"
        return "severe"

    def band_of(t: float) -> str:
        for name, lo, hi in TOD_BANDS:
            if lo <= t < hi:
                return name
        return TOD_BANDS[-1][0]

    # Build the (band x severity) cells, then ROUND-ROBIN across them. Round-robin
    # rather than a fixed per-cell quota because the grid's stress is concentrated
    # in the evening peak (206 of 344 stressed windows are pm_rush) — a fixed quota
    # would either under-fill or refuse to reach n. Round-robin takes one from each
    # non-empty cell in turn, so coverage is as even as the data allows and the
    # shortfall is visible in the diagnostics rather than hidden.
    cells: Dict[Tuple[str, str], List[int]] = {}
    for i in stressed.tolist():
        cells.setdefault((band_of(float(tod0[i])), severity_of(float(max_dev[i]))),
                         []).append(i)
    rng = random.Random(seed)
    for lst in cells.values():
        rng.shuffle(lst)

    order = sorted(cells.keys())
    picked: List[int] = []
    depth = 0
    while len(picked) < n:
        added = False
        for key in order:
            if depth < len(cells[key]):
                picked.append(cells[key][depth])
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break                                          # every cell exhausted
        depth += 1

    diag["cells_available"] = {"{}|{}".format(b, s): len(v_)
                               for (b, s), v_ in sorted(cells.items())}
    diag["n_requested"] = n
    diag["n_sampled"] = len(picked)

    scenarios: List[Dict[str, Any]] = []
    for idx in picked:
        tnode = int(target[idx])
        scenarios.append({
            "sample_index": int(idx),
            "target_node": tnode,
            "target_bus": tnode + 1,               # 1-indexed, as the literature numbers buses
            "tod_band": band_of(float(tod0[idx])),
            "severity": severity_of(float(max_dev[idx])),
            "deviation_pu": float(round(max_dev[idx], 5)),
            "voltage_pu": float(round(last[idx, tnode], 5)),
            "normal_pu": float(round(ref[tnode], 5)),
            "timestamp": "test#{} @ {}".format(idx, _tod_to_clock(float(tod0[idx]))),
        })
    return scenarios, diag


# ---------------------------------------------------------------------------
# Explanations (Step 2 of the phase) — cached to their own directory.
# ---------------------------------------------------------------------------
def explanation_dir() -> str:
    d = os.path.join(PKG_ROOT, "evaluation", "results", "power_grid_explanations")
    os.makedirs(d, exist_ok=True)
    return d


def build_or_load_explanation(builder: ExplanationBuilder, dataset: str,
                              sc: Dict[str, Any], horizon_step: int,
                              force: bool = False) -> Dict[str, Any]:
    """Explain one scenario, caching to results/power_grid_explanations/.

    STALENESS (the Phase-17 lesson): a cache keyed only on filename silently
    serves explanations built by an OLDER checkpoint — Phase 17 found 6 stale ones
    that way. So we also compare mtimes against the checkpoint and rebuild if the
    checkpoint is newer.
    """
    path = os.path.join(explanation_dir(),
                        "{}_{}_{}.json".format(dataset, sc["sample_index"],
                                               sc["target_node"]))
    ckpt = builder.checkpoint_path if hasattr(builder, "checkpoint_path") else None
    if os.path.exists(path) and not force:
        fresh = True
        if ckpt and os.path.exists(ckpt):
            fresh = os.path.getmtime(path) >= os.path.getmtime(ckpt)
        if fresh:
            with open(path) as f:
                return json.load(f)
    X, _ = _load_split(dataset, "test")
    exp = builder.explain_prediction(
        X[sc["sample_index"]: sc["sample_index"] + 1],
        target_node=sc["target_node"], horizon_step=horizon_step,
        timestamp=sc["timestamp"])
    exp["meta"]["city"] = dataset
    # Additive provenance so a reader of the JSON is never misled by the schema's
    # traffic-era key names. schema.validate_explanation checks required keys are
    # PRESENT and does not forbid extras (the same property Phase 18 relied on).
    exp["meta"]["units"] = "pu"
    exp["meta"]["note"] = ("Power grid: schema keys named *_speed_mph carry VOLTAGE "
                           "MAGNITUDE in per-unit. Key names are frozen by the "
                           "Phase-3 contract; the values are voltages.")
    exp["scenario"] = {k: sc[k] for k in
                       ("target_bus", "tod_band", "severity", "deviation_pu",
                        "voltage_pu", "normal_pu")}
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
    return exp


# ---------------------------------------------------------------------------
# Chance baselines — what does a RANDOM citer score under this exact metric?
# ---------------------------------------------------------------------------
def chance_metrics(exp: Dict[str, Any], table: NodeTable, n_causes: int,
                   vocabulary: str, draws: int, rng: random.Random
                   ) -> Dict[str, float]:
    """Score `draws` random advisories against this explanation's top-k.

    A "random advisory" cites `n_causes` locations drawn uniformly (with
    replacement, since an LLM can repeat a location) from one vocabulary:
      "bus"  -> the 14 full bus names, i.e. precise citations
      "zone" -> the 3 electrical zone labels, i.e. coarse citations
    Everything else — resolution, hit test, precision/recall/F1 — goes through the
    SAME score_advisory the LLM conditions use, so the comparison is exact.

    This is the number that makes the cross-domain claim honest. On a 14-bus graph
    with k=4, guessing is far more successful than on a 207-sensor graph with k=8,
    and any hallucination rate must be read against it.
    """
    if vocabulary == "bus":
        pool = [table.namer.name(i) for i in range(table.n_nodes)]
    elif vocabulary == "zone":
        pool = sorted(table.region_to_nodes.keys())
    else:
        raise ValueError("unknown vocabulary {!r}".format(vocabulary))

    n_causes = max(1, int(n_causes))
    acc: Dict[str, List[float]] = {"cause_precision": [], "cause_recall": [],
                                   "faithfulness_f1": [], "hallucination_rate": []}
    for _ in range(draws):
        advisory = {
            "reasoning": "",
            "cited_causes": [{"location": rng.choice(pool), "resolved_node_id": None}
                             for _ in range(n_causes)],
            "recommendations": [],
        }
        m = score_advisory(exp, advisory, table)
        for k in acc:
            acc[k].append(float(m[k]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# A mock LLM so the whole harness is testable with no Ollama running.
# ---------------------------------------------------------------------------
class _MockAdvisor:
    """Deterministic stand-in for Advisor. Condition A cites the explanation's
    top-k (a perfectly faithful model); condition B, having no explanation, cites
    plausible-sounding buses at random (an inventing model). This reproduces the
    SHAPE of the expected result so the plumbing can be verified without an LLM —
    it is NOT evidence of anything and never appears in a reported number."""

    def __init__(self, table: NodeTable, seed: int = 0):
        self.table = table
        self.model = "mock"
        self.rng = random.Random(seed)

    def advise_condition(self, exp: Dict[str, Any], condition: str) -> Dict[str, Any]:
        if condition == "A":
            locs = [n["node_name"] for n in exp.get("top_nodes", [])]
        else:
            locs = [self.table.namer.name(self.rng.randrange(self.table.n_nodes))
                    for _ in range(3)]
        return {
            "advisory": {
                "reasoning": "mock",
                "cited_causes": [{"location": l, "resolved_node_id": None} for l in locs],
                "recommendations": [],
            },
            "condition": condition, "context_used": [], "model": "mock",
        }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
_METRIC_KEYS = ["cause_precision", "cause_recall", "faithfulness_f1",
                "hallucination_rate"]


def _aggregate(rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["condition"] == condition]
    out: Dict[str, Any] = {"condition": condition, "n": len(sub)}
    for k in _METRIC_KEYS:
        m, s, cnt = mean_std([r.get(k) for r in sub])
        out[k + "_mean"], out[k + "_std"], out[k + "_n"] = m, s, cnt
    return out


def citation_granularity(rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    """How PRECISELY did this condition cite? Bus-level vs zone-level vs nothing.

    This matters more here than on METR-LA and it is not a nicety. Region-level
    credit (faithfulness.py's documented generosity) means a citation counts as a
    hit if the group it resolves to INTERSECTS the top-k. On METR-LA a region holds
    a handful of 207 sensors, so that is a mild concession. On this grid a zone
    holds up to 7 of 14 buses, so a zone-level citation is close to a free hit —
    which is exactly why RANDOM_zone scores an F1 near 0.7 with no model at all.

    So a condition's F1 is only as impressive as its citations are precise. Logging
    the mix lets the paper say which it was, instead of quietly banking the
    generosity. Percentages are over all citations made under the condition.
    """
    methods: List[str] = []
    for r in rows:
        if r["condition"] == condition:
            methods.extend(r.get("resolution_methods") or [])
    n = len(methods)
    if not n:
        return {"n_citations": 0}
    bus = sum(1 for m in methods if m.startswith("exact_bus"))
    zone = sum(1 for m in methods if m in ("exact_region", "fuzzy"))
    unres = sum(1 for m in methods if m == "unresolved")
    return {"n_citations": n,
            "bus_level_frac": round(bus / n, 3),
            "zone_level_frac": round(zone / n, 3),
            "unresolved_frac": round(unres / n, 3)}


def paired_bootstrap(rows: List[Dict[str, Any]], cond_a: str, cond_b: str,
                     metric: str, iters: int = 10000, seed: int = 42
                     ) -> Optional[Dict[str, Any]]:
    """Nonparametric bootstrap 95% CI on the PER-SCENARIO difference cond_a - cond_b.

    Paired by scenario (the same window is scored under both conditions), which is
    the right pairing here for the same reason Phase 15b paired its A/B/C bootstrap:
    scenarios differ a lot in how much signal they carry, and pairing removes that
    variance instead of letting it swamp the contrast.

    This is what turns "A beats chance" into "A beats chance, 95% CI on the gap
    excludes 0" — the difference between an anecdote and a claim, and cheap because
    every per-scenario metric is already logged. Returns None if the conditions
    don't share scenarios.
    """
    by: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["sample_index"]] = r
    if cond_a not in by or cond_b not in by:
        return None
    keys = sorted(set(by[cond_a]) & set(by[cond_b]))
    if not keys:
        return None
    diffs = [float(by[cond_a][k][metric]) - float(by[cond_b][k][metric]) for k in keys]
    n = len(diffs)
    rng = random.Random(seed)
    boots = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)]
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {"comparison": "{} - {}".format(cond_a, cond_b), "metric": metric,
            "n_paired": n, "mean_diff": float(np.mean(diffs)),
            "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "iters": iters, "seed": seed}


def _print_table(aggs: Sequence[Dict[str, Any]], label: str) -> None:
    print("\n=== {} ===".format(label))
    hdr = "{:22s}".format("metric")
    for a in aggs:
        hdr += "{:>22s}".format("{} (n={})".format(a["condition"], a["n"]))
    print(hdr)
    print("-" * len(hdr))
    for k in _METRIC_KEYS:
        line = "{:22s}".format(k)
        for a in aggs:
            m, s = a[k + "_mean"], a[k + "_std"]
            line += "{:>22s}".format("nan" if m != m else "{:.3f} +/- {:.3f}".format(m, s))
        print(line)


def write_latex(aggs: Sequence[Dict[str, Any]], path: str, n: int,
                model: str, top_k: int) -> None:
    """LaTeX booktabs table, \\input-ready, matching the other paper tables."""
    label = {"A": r"XTraffic full pipeline (explanation shown)",
             "B": r"No explanation (prediction only)",
             "RANDOM_bus": r"\emph{Chance}: random bus citation",
             "RANDOM_zone": r"\emph{Chance}: random zone citation"}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cross-domain faithfulness on the IEEE 14-bus power grid "
        r"($n=%d$ undervoltage scenarios, %s, top-$k$=%d). The same pipeline, "
        r"explainer and metric as the METR-LA study; only the prompt vocabulary "
        r"and the entity resolver's notion of a location group were adapted. "
        r"Chance rows cite uniformly at random from the same vocabulary and are "
        r"scored identically --- they are the reference the hallucination rates "
        r"must be read against, because $k/N$ is far larger here (4/14) than on "
        r"METR-LA (8/207).}" % (n, model.replace("_", r"\_"), top_k),
        r"\label{tab:power_grid_faith}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Condition & Precision & Recall & F1 & Hallucination \\", r"\midrule",
    ]
    for a in aggs:
        lines.append("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
            label.get(a["condition"], a["condition"]),
            a["cause_precision_mean"], a["cause_recall_mean"],
            a["faithfulness_f1_mean"], a["hallucination_rate_mean"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=0, help="override scenarios.n")
    ap.add_argument("--limit", type=int, default=0, help="cap scenarios (smoke)")
    ap.add_argument("--smoke", action="store_true", help="3 scenarios, real LLM")
    ap.add_argument("--mock-llm", action="store_true", help="no Ollama; verify plumbing")
    ap.add_argument("--explain-only", action="store_true",
                    help="Step 2 only: build + save explanations, no LLM")
    ap.add_argument("--model", default=None, help="override the Ollama model")
    ap.add_argument("--top-k", type=int, default=0, help="override explainer.top_k")
    ap.add_argument("--force-explain", action="store_true", help="ignore explanation cache")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset, city = cfg["dataset"], cfg["city"]
    scfg, ecfg, stcfg = cfg["scenarios"], cfg["explainer"], cfg["study"]
    top_k = args.top_k or int(ecfg["top_k"])
    n_target = args.n or int(scfg["n"])
    if args.smoke:
        n_target = min(n_target, 3)
    tag = ("_" + args.out_tag) if args.out_tag else ""

    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "power_grid_faithfulness")
    os.makedirs(out_dir, exist_ok=True)

    scaler = load_scaler(dataset)
    scenarios, diag = sample_undervoltage_scenarios(
        dataset, scaler, n=n_target,
        min_dev=float(scfg["min_deviation_pu"]),
        min_std=float(scfg["min_bus_std_pu"]),
        seed=int(scfg["seed"]))
    if args.limit:
        scenarios = scenarios[:args.limit]

    print("SCENARIO SAMPLING")
    print("  test windows            : {}".format(diag["test_windows"]))
    print("  ineligible (near-const) : bus {}".format(diag.get("ineligible_buses")))
    print("  stressed (>= {:.3f} pu)  : {}".format(
        diag["min_deviation_pu"], diag["n_stressed_windows"]))
    print("  severity cuts (pu)      : {}".format(diag.get("severity_cuts_pu")))
    print("  sampled                 : {} of {} requested".format(
        len(scenarios), diag.get("n_requested")))
    if len(scenarios) < diag.get("n_requested", 0):
        print("  NOTE: fewer scenarios than requested — the stratified cells were "
              "exhausted. Reported n is the real n.")
    tally: Dict[str, int] = {}
    for s in scenarios:
        tally[s["severity"]] = tally.get(s["severity"], 0) + 1
    print("  severity mix            : {}".format(tally))
    print("  target buses            : {}".format(
        sorted({s["target_bus"] for s in scenarios})))

    if not scenarios:
        print("No qualifying undervoltage scenarios — nothing to do.")
        return

    # --- Step 2: explanations -------------------------------------------------
    ckpt = cfg["checkpoint"]
    builder = ExplanationBuilder(ckpt, dataset, device=torch.device("cpu"),
                                 top_k=top_k, epochs=int(ecfg["epochs"]),
                                 confidence_runs=int(ecfg["confidence_runs"]))
    builder.checkpoint_path = (ckpt if os.path.isabs(ckpt)
                               else os.path.join(PKG_ROOT, ckpt))
    table = NodeTable(load_node_meta(dataset))

    print("\nBUILDING EXPLANATIONS (top_k={}) -> {}".format(top_k, explanation_dir()))
    exps: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenarios):
        exp = build_or_load_explanation(builder, dataset, sc,
                                        int(scfg["horizon_step"]),
                                        force=args.force_explain)
        exps.append(exp)
        top = ", ".join("bus {}".format(int(table.sensor_ids[n["node_id"]]))
                        for n in exp["top_nodes"])
        print("  [{}/{}] {} target bus {} ({:.3f} pu, {} sag {:.3f}) <- {} | conf {:.2f}"
              .format(i + 1, len(scenarios), sc["timestamp"], sc["target_bus"],
                      sc["voltage_pu"], sc["severity"], sc["deviation_pu"], top,
                      exp["explanation_confidence"]))
    conf = [e["explanation_confidence"] for e in exps]
    print("  mean explanation confidence: {:.3f} (min {:.3f}, max {:.3f})".format(
        float(np.mean(conf)), float(np.min(conf)), float(np.max(conf))))

    if args.explain_only:
        print("\n--explain-only: {} explanations written. Stopping before the LLM."
              .format(len(exps)))
        return

    # --- Step 3: A/B conditions ----------------------------------------------
    if args.mock_llm:
        advisor: Any = _MockAdvisor(table, seed=int(scfg["seed"]))
    elif args.model:
        import copy
        from ..models.advisor.advisor import load_advisor_config
        acfg = copy.deepcopy(load_advisor_config())
        acfg.setdefault("ollama", {})["model"] = args.model
        advisor = Advisor(city, cfg=acfg)
    else:
        advisor = Advisor(city)
    model_name = advisor.model
    conditions = list(stcfg["conditions"])
    print("\nRUNNING CONDITIONS {} with model '{}'".format(conditions, model_name))

    cache_dir = os.path.join(out_dir, "decisions_cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model_name)

    rows: List[Dict[str, Any]] = []
    n_cached = 0
    for i, (sc, exp) in enumerate(zip(scenarios, exps)):
        for cond in conditions:
            cpath = os.path.join(cache_dir, "{}_{}_{}_{}_k{}.json".format(
                sc["sample_index"], sc["target_node"], cond, safe_model, top_k))
            if os.path.exists(cpath):
                with open(cpath) as f:
                    row = json.load(f)
                n_cached += 1
            else:
                res = advisor.advise_condition(exp, cond)
                adv = res["advisory"]
                m = score_advisory(exp, adv, table)
                row = {
                    "sample_index": sc["sample_index"], "target_bus": sc["target_bus"],
                    "tod_band": sc["tod_band"], "severity": sc["severity"],
                    "condition": cond,
                    "advisory_error": adv.get("_error", ""),
                    "n_cited_causes": m["n_cited_causes"], "n_topk": m["n_topk"],
                    "cited_locations": [c.get("location") for c in
                                        (adv.get("cited_causes") or [])
                                        if isinstance(c, dict)],
                    "resolution_methods": [p["resolved_method"] for p in m["per_cause"]],
                }
                for k in _METRIC_KEYS:
                    row[k] = m[k]
                with open(cpath, "w") as f:
                    json.dump(row, f, indent=2)
            rows.append(row)
        print("  [{}/{}] bus {} {} -> {}".format(
            i + 1, len(scenarios), sc["target_bus"], sc["severity"],
            " ".join("{} F1 {:.2f}/H {:.2f}".format(
                r["condition"], r["faithfulness_f1"], r["hallucination_rate"])
                for r in rows[-len(conditions):])))
    if n_cached:
        print("  (reused {} cached decisions)".format(n_cached))

    # --- chance baselines (no LLM) -------------------------------------------
    rng = random.Random(int(scfg["seed"]))
    draws = int(stcfg["random_draws"])
    a_rows = [r for r in rows if r["condition"] == "A"]
    mean_cited = (float(np.mean([r["n_cited_causes"] for r in a_rows]))
                  if a_rows else 3.0)
    print("\nCHANCE BASELINES ({} draws/scenario, {:.1f} citations each = "
          "condition A's mean)".format(draws, mean_cited))
    for vocab in stcfg["random_baselines"]:
        for sc, exp in zip(scenarios, exps):
            cm = chance_metrics(exp, table, mean_cited, vocab, draws, rng)
            row = {"sample_index": sc["sample_index"], "target_bus": sc["target_bus"],
                   "tod_band": sc["tod_band"], "severity": sc["severity"],
                   "condition": "RANDOM_" + vocab, "advisory_error": "",
                   "n_cited_causes": int(round(mean_cited)),
                   "n_topk": len(exp["top_nodes"]),
                   "cited_locations": [], "resolution_methods": []}
            row.update(cm)
            rows.append(row)

    all_conditions = list(conditions) + ["RANDOM_" + v for v in stcfg["random_baselines"]]
    aggs = [_aggregate(rows, c) for c in all_conditions]
    _print_table(aggs, "POWER GRID FAITHFULNESS (mean +/- std)")

    # --- headline ------------------------------------------------------------
    by = {a["condition"]: a for a in aggs}
    print("\nHEADLINE — does the grounding gap hold on a power grid?")
    if "A" in by and "B" in by:
        ah, bh = by["A"]["hallucination_rate_mean"], by["B"]["hallucination_rate_mean"]
        af, bf = by["A"]["faithfulness_f1_mean"], by["B"]["faithfulness_f1_mean"]
        print("  hallucination : A {:.3f}  vs  B {:.3f}   (gap {:+.3f})".format(ah, bh, ah - bh))
        print("  F1            : A {:.3f}  vs  B {:.3f}   (gap {:+.3f})".format(af, bf, af - bf))
        for v in stcfg["random_baselines"]:
            key = "RANDOM_" + v
            if key in by:
                print("  chance ({:4s}) : hallucination {:.3f}  F1 {:.3f}".format(
                    v, by[key]["hallucination_rate_mean"],
                    by[key]["faithfulness_f1_mean"]))
        print("  -> B must beat CHANCE to count as 'inventing plausibly'; A must "
              "beat B by a margin that is not explained by chance.")

    # Paired bootstrap: A against B and against BOTH chance baselines. The chance
    # comparisons are the load-bearing ones — beating B only shows the explanation
    # helps, whereas beating RANDOM_zone shows the result is not an artifact of a
    # small graph plus generous zone-level credit.
    boot: List[Dict[str, Any]] = []
    for other in [c for c in all_conditions if c != "A"]:
        for metric in ("hallucination_rate", "faithfulness_f1", "cause_precision"):
            b = paired_bootstrap(rows, "A", other, metric, seed=int(scfg["seed"]))
            if b:
                boot.append(b)
    if boot:
        print("\nPAIRED BOOTSTRAP 95% CI on per-scenario differences "
              "(n={}, {} iters, seed {})".format(boot[0]["n_paired"],
                                                 boot[0]["iters"], boot[0]["seed"]))
        for b in boot:
            print("  {:18s} {:20s} {:+.3f}  [{:+.3f}, {:+.3f}]  {}".format(
                b["comparison"], b["metric"], b["mean_diff"], b["lo"], b["hi"],
                "EXCLUDES 0" if b["excludes_zero"] else "spans 0"))

    gran = {c: citation_granularity(rows, c) for c in conditions}
    print("\nCITATION GRANULARITY (how precise were the citations?)")
    for c in conditions:
        g = gran[c]
        if g.get("n_citations"):
            print("  {}: {} citations — bus-level {:.0%} | zone-level {:.0%} | "
                  "unresolved {:.0%}".format(c, g["n_citations"],
                                             g["bus_level_frac"],
                                             g["zone_level_frac"],
                                             g["unresolved_frac"]))
    print("  -> zone-level citations get generous credit (a zone spans up to 7 of "
          "14 buses); read F1 against RANDOM_zone in proportion to this fraction.")

    # --- outputs -------------------------------------------------------------
    csv_path = os.path.join(out_dir, "power_grid_faith_per_scenario{}.csv".format(tag))
    cols = (["sample_index", "target_bus", "tod_band", "severity", "condition",
             "advisory_error"] + _METRIC_KEYS + ["n_cited_causes", "n_topk"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "dataset": dataset, "model": model_name, "top_k": top_k,
        "n_scenarios": len(scenarios), "seed": int(scfg["seed"]),
        "sampling_diagnostics": diag,
        "conditions": all_conditions,
        "overall": aggs,
        "by_severity": {lvl: [_aggregate([r for r in rows if r["severity"] == lvl], c)
                              for c in all_conditions] for lvl in SEVERITY_LEVELS},
        "citation_granularity": {c: citation_granularity(rows, c) for c in conditions},
        "paired_bootstrap": boot,
        "mean_explanation_confidence": float(np.mean(conf)),
        "mock_llm": bool(args.mock_llm),
    }
    with open(os.path.join(out_dir, "power_grid_faith_summary{}.json".format(tag)), "w") as f:
        json.dump(summary, f, indent=2)

    tex = os.path.join(PKG_ROOT, "evaluation", "paper",
                       "table_power_grid_faith{}.tex".format(tag))
    os.makedirs(os.path.dirname(tex), exist_ok=True)
    if not args.mock_llm:
        write_latex(aggs, tex, len(scenarios), model_name, top_k)
    else:
        print("\n(--mock-llm: LaTeX table NOT written — mock numbers never reach "
              "the paper directory.)")

    print("\nWrote:")
    print("  " + csv_path)
    print("  " + os.path.join(out_dir, "power_grid_faith_summary{}.json".format(tag)))
    if not args.mock_llm:
        print("  " + tex)
    print("  " + explanation_dir() + "/  ({} explanations)".format(len(exps)))


if __name__ == "__main__":
    main()
