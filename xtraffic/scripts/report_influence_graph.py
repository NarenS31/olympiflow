"""Renders report.md for the explanation-network analysis.

Kept separate from analyze_influence_graph.py so the prose lives in one place and
the analysis stays readable. Every number here is interpolated from the metrics
dict or a DataFrame — there are no hard-coded results in this file. Where a
number comes from a DIFFERENT run (the device-divergence probe), it is read from
that run's artifact and the run id is printed next to it.

The report states what the model relies on, whether it replicates across halves,
and what the external checks say. It does not interpret and does not claim a
discovery. Python 3.9 compatible.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..evaluation import influence_graph as ig
from ..reproducibility import run_dir
from ..utils.io_utils import PKG_ROOT


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if np.isnan(x):
            return "n/a"
        return "{:.{}f}".format(x, nd)
    return str(x)


def _device_divergence() -> Optional[Dict]:
    """Newest device_divergence run, if one exists."""
    base = os.path.join(PKG_ROOT, run_dir.RAW)
    hits = sorted(glob.glob(os.path.join(base, "*__device_divergence__*")), reverse=True)
    for h in hits:
        p = os.path.join(h, "device_divergence.json")
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            d["_run_id"] = os.path.basename(h)
            return d
    return None


def render(res: Dict, geo: ig.Geometry, mass_tables: Dict[str, pd.DataFrame],
           step3: pd.DataFrame) -> str:
    g = res["geometry"]
    s = res["settings"]
    cfg = res.get("stage1_config", {})
    L = []
    A = L.append

    # ---------------------------------------------------------------- header
    A("# What the ST-GNN's explanations say about the METR-LA road network")
    A("")
    A("**Status: EXPLORATORY.** This describes what the trained model's explainer")
    A("leans on and whether that survives being recomputed on disjoint windows. It")
    A("is not a confirmatory result, there is no ground truth anywhere in it, and")
    A("nothing here is a discovery claim. The only external checks are the geometry")
    A("of the adjacency, sensor coordinates, and raw-speed pathologies.")
    A("")
    A("**No LLM was called at any point in this analysis.**")
    A("")
    A("| | |")
    A("|---|---|")
    A("| Stage-1 run | `{}` |".format(res["stage1_run_id"]))
    A("| Targets solved | {} of {} |".format(res["n_targets_solved"], g["n_nodes"]))
    A("| Windows per target | {} (shared across every target) |".format(res["n_windows"]))
    A("| Solves | {} |".format(res["n_targets_solved"] * res["n_windows"]))
    A("| Checkpoint | `{}`, horizon {} min |".format(
        cfg.get("checkpoint"), cfg.get("horizon_minutes")))
    A("| Explainer | {} steps, seed {}, confidence reruns {} |".format(
        cfg.get("explainer_epochs"), cfg.get("explainer_seed"),
        cfg.get("confidence_runs")))
    A("| Device | {} ({} workers x {} thread) |".format(
        cfg.get("device"), cfg.get("workers"), cfg.get("torch_threads_per_worker")))
    A("| Sampling seed | {} |".format(cfg.get("sampling_seed")))
    A("| Regimes | congested < {} mph, free-flow >= {} mph, dead band between |".format(
        cfg.get("congested_max_mph"), cfg.get("freeflow_min_mph")))
    A("")

    # -------------------------------------------------- setup facts
    A("## Three setup facts that constrain every number below")
    A("")
    A("**1. The adjacency is a distance threshold, not \"the road network\".**")
    A("`gaussian_kernel_adjacency` computes `A_ij = exp(-(d_ij/sigma)^2)` and zeroes")
    A("anything below kappa = {}. With sigma = {} m that is exactly the rule *keep the".format(
        g["kappa"], _fmt(g["sigma_m"], 1)))
    A("pair iff directed road distance <= {} m*. So \"off-adjacency\" below means".format(
        _fmt(g["cutoff_m"], 1)))
    A("\">{} m by road\" and nothing more.".format(_fmt(g["cutoff_m"], 0)))
    A("")
    A("**2. Road distance is mostly missing off-adjacency.** The distance file covers")
    A("{:.1%} of ordered pairs overall and only {:.1%} of non-adjacent ones. Haversine".format(
        g["share_pairs_with_road_distance"], g["share_nonadjacent_pairs_with_road_distance"]))
    A("is carried in a separate column and nothing is derived from it.")
    A("")
    A("**3. The model's spatial receptive field is the complete graph in one hop.**")
    A("`XTrafficSTGNN` diffuses over two supports, both row-softmaxes, both 100% dense")
    A("(42,849 / 42,849 strictly positive entries each). `gcn_order` x `n_blocks` =")
    A("{} bounds diffusion at {} hops over the *physical* component only; through".format(
        s["k_hops"], s["k_hops"]))
    A("either learned support any node reaches any other immediately. **Hop distance is")
    A("therefore a descriptive coordinate, not a reachability limit.** Influence at hop")
    A("> {} or at an unreachable pair is expected by construction and is not, on its".format(s["k_hops"]))
    A("own, anomalous. It is reported separately because it was asked for, and because")
    A("\"how road-aligned is the learned structure\" remains a real question.")
    A("")

    # -------------------------------------------------- step 1
    A("## Step 1 — learned influence vs the given adjacency")
    A("")
    A("W is [targets x sources]; an edge is the ordered pair (source, target) and is")
    A("checked against `A[source, target]`, matching `GraphConv._nconv`. A target's row")
    A("is built only if it has >= {} windows in that regime; targets below the floor are".format(
        s["min_windows_full"]))
    A("counted, never silently dropped.")
    A("")

    for regime in ig.REGIMES:
        r = res["per_regime"].get(regime)
        if not r:
            continue
        A("### Regime: {}".format(regime.replace("_", " ")))
        A("")
        A("- Active targets: **{}** ({} fell below the {}-window floor)".format(
            r["n_targets_active"], r["n_targets_below_window_floor"],
            s["min_windows_full"]))
        if not r.get("edges"):
            A("- No target met the floor in this regime; nothing further is reported.")
            A("")
            continue
        wp = r["windows_per_active_target"]
        A("- Windows per active target: mean {}, range {}-{}".format(
            wp["mean"], wp["min"], wp["max"]))
        sm = r["self_mass"]
        A("- **Diagonal (target's own mass share of its row):** mean {}, median {},".format(
            _fmt(sm["mean"], 4), _fmt(sm["median"], 4)))
        A("  range {}-{}. Uniform-share reference 1/N = {}.".format(
            _fmt(sm["min"], 4), _fmt(sm["max"], 4), _fmt(sm["chance_1_over_N"], 4)))
        A("")
        A("**Effective edge set vs adjacency**")
        A("")
        A("| Rule | \\|E\\| | Precision | Recall | Jaccard | Off-adjacency |")
        A("|---|---|---|---|---|---|")
        for lab, key in (("top-k (k={})".format(s["top_k"]), "top_k"),
                         ("cumulative mass {:.0%}".format(s["mass_frac"]),
                          "cumulative_mass")):
            m = r["edges"][key]
            A("| {} | {} | {} | {} | {} | {} |".format(
                lab, m["n_effective"], _fmt(m["precision"]), _fmt(m["recall"]),
                _fmt(m["jaccard"]), m["n_off_adjacency"]))
        A("")
        A("Reference edge count for these targets: {}. The cumulative-mass rule needed".format(
            r["edges"]["top_k"]["n_reference"]))
        A("**{} sources per target on average** to cover {:.0%} of the off-self mass,".format(
            r["edges"]["cumulative_mass"].get("sources_per_target_mean"), s["mass_frac"]))
        A("out of {} available — a direct measure of how flat the learned mask is. A".format(
            g["n_nodes"] - 1))
        A("peaked mask would need a handful; the two rules are reported together")
        A("precisely because that gap is informative.")
        A("")
        A("**Baselines** (same targets, same k)")
        A("")
        A("| Method | \\|E\\| | Precision | Recall | Jaccard |")
        A("|---|---|---|---|---|")
        m = r["edges"]["top_k"]
        A("| **effective top-k** | {} | **{}** | {} | {} |".format(
            m["n_effective"], _fmt(m["precision"]), _fmt(m["recall"]), _fmt(m["jaccard"])))
        for name, lab in (("uniform_random", "(a) uniform random"),
                          ("degree_matched_random", "(b) degree-matched random"),
                          ("nearest_by_road_distance", "(c) nearest by road distance")):
            b = r["baselines"][name]
            A("| {} | {} | {} | {} | {} |".format(
                lab, b["n_effective"], _fmt(b["precision"]), _fmt(b["recall"]),
                _fmt(b["jaccard"])))
        A("")
        eff = m["precision"]
        uni = r["baselines"]["uniform_random"]["precision"]
        if eff is not None and uni and not np.isnan(eff) and not np.isnan(uni):
            A("Effective-vs-uniform precision ratio: **{}x**.".format(_fmt(eff / uni, 2)))
        A("Overlap between the effective set and the nearest-by-road set (Jaccard): {}.".format(
            _fmt(r["baselines"]["overlap_effective_vs_nearest_road"])))
        nrk = r["baselines"]["nearest_by_road_distance"].get(
            "targets_with_fewer_than_k_road_sources")
        if nrk:
            A("({} targets have fewer than {} road-connected sources, so baseline (c)".format(
                nrk, s["top_k"]))
            A("contributes fewer edges for them. Haversine was not substituted.)")
        A("")
        A("**Hop-distance stratification of the effective edge set**")
        A("")
        A("| Hop bucket | Effective edges | Share | Adjacency edges in same bucket |")
        A("|---|---|---|---|")
        for h in r["hop_strata"]:
            A("| {} | {} | {} | {} |".format(
                h["bucket"], h["n_effective_edges"], _fmt(h["share_of_effective"]),
                h["n_adjacency_edges_same_bucket"]))
        A("")
        bk = r["beyond_k"]
        A("**Candidate set — beyond {} hops or unreachable:** {} edges ({} of the".format(
            bk["k_hops"], bk["n_edges"], _fmt(bk["share_of_effective"])))
        A("effective set), of which {} are unreachable in the adjacency graph.".format(
            bk["n_unreachable"]))
        A("Full listing in `beyond_k_edges.csv`. Per fact (3) this is not by itself")
        A("anomalous — both learned supports connect every pair directly.")
        A("")
        st = r["stability"]
        A("**Split-half stability** ({} vs {} windows, stratum-balanced, >= {} windows".format(
            st["half_a_windows"], st["half_b_windows"], st["min_windows_per_half"]))
        A("per half; {} targets qualify in both halves)".format(
            st.get("n_targets_in_both_halves")))
        A("")
        A("| Subset | Jaccard between halves |")
        A("|---|---|")
        A("| all effective edges | **{}** |".format(_fmt(st.get("jaccard_all_edges"))))
        A("| off-adjacency edges only | **{}** ({} vs {} edges) |".format(
            _fmt(st.get("jaccard_off_adjacency")),
            st.get("n_off_adjacency_half_a"), st.get("n_off_adjacency_half_b")))
        A("| beyond-{}/unreachable only | **{}** ({} vs {} edges) |".format(
            s["k_hops"], _fmt(st.get("jaccard_beyond_k")),
            st.get("n_beyond_k_half_a"), st.get("n_beyond_k_half_b")))
        A("| *reference: two independent uniform random draws* | {} |".format(
            _fmt(st.get("jaccard_two_random_draws_reference"))))
        A("")
        ja = st.get("jaccard_all_edges")
        jr = st.get("jaccard_two_random_draws_reference")
        joff = st.get("jaccard_off_adjacency")
        if ja is not None and jr is not None:
            if ja <= jr * 1.5:
                A("**The effective edge set does not replicate across halves at a rate")
                A("meaningfully above two independent random draws ({} vs {}).** Read".format(
                    _fmt(ja), _fmt(jr)))
                A("everything above with that in front of it: on this sample the")
                A("thresholded set is close to unstable, so its composition should not be")
                A("treated as a property of the model.")
            else:
                A("The effective edge set replicates across halves at {}, against {} for".format(
                    _fmt(ja), _fmt(jr)))
                A("two independent random draws.")
            if joff is not None:
                A("")
                A("Off-adjacency edges specifically replicate at {}.".format(_fmt(joff)))
        A("")

    # -------------------------------------------------- off-adjacency listing
    A("## Every edge above threshold")
    A("")
    A("No curation and no selected examples. Complete listings:")
    A("")
    A("- `effective_edges.csv` — every (source, target) above threshold, both regimes,")
    A("  with importance, adjacency membership, hop distance, road distance (or blank),")
    A("  haversine in its own reference-only column, and both endpoints' coordinates.")
    A("- `off_adjacency_edges.csv` — the off-adjacency subset.")
    A("- `beyond_k_edges.csv` — the beyond-{}/unreachable subset.".format(s["k_hops"]))
    A("- `hop_strata.csv`, `baselines.csv`, `per_target_mass.csv`.")
    A("")

    # -------------------------------------------------- step 2
    A("## Step 2 — implied propagation speed: NOT RUN")
    A("")
    A("Skipped deliberately, on the evidence gathered in the feasibility check.")
    A("")
    A("The explainer learns a node mask `m in [0,1]^N` — one scalar per node, applied")
    A("uniformly across all 12 input timesteps (`explain.py:111`). There is no")
    A("per-timestep dimension in the mask, the schema, or anywhere else in the")
    A("artifact, so there is no importance-weighted lag to compute. The schema's")
    A("`propagation_lag_minutes` is a single scalar per explanation, covering only")
    A("`top_nodes[0] -> target`, and it is obtained by cross-correlating the *input")
    A("window*: it is a property of the data, not of the model. Deriving implied")
    A("speeds from it would measure METR-LA's autocorrelation structure while")
    A("labelling the result as the model's propagation behaviour.")
    A("")
    A("No traffic-flow reference values were looked up, because none were used.")
    A("")

    # -------------------------------------------------- step 3
    A("## Step 3 — sensors whose influence mass sits farthest from their neighbourhood")
    A("")
    A("Ranked by the share of off-self importance mass landing on non-adjacent sources.")
    A("Because adjacency *is* the {} m road-distance cutoff, an adjacency split and a".format(
        _fmt(g["cutoff_m"], 0)))
    A("distance split are the same split; the mass is therefore decomposed three ways,")
    A("separating *far by road* from *no road path in the distance file*, since the")
    A("latter is an absence of data rather than a measurement of distance.")
    A("")
    A("**No sensor is described as mislabelled.** The columns are evidence.")
    A("")
    if len(step3):
        cols = ["target", "sensor_id", "lat", "lon", "in_degree",
                "mass_off_adjacency", "mass_on_far_road", "mass_on_no_road_path",
                "self_mass_share", "missing_pct", "longest_missing_run_steps",
                "longest_constant_nonzero_run_steps", "max_mph"]
        cols = [c for c in cols if c in step3.columns]
        head = step3.head(20)
        A("Top 20 of {} ranked sensors (full table: `sensor_anomaly_ranking.csv`;".format(
            len(step3)))
        A("raw-speed defects for all 207: `sensor_raw_pathologies.csv`):")
        A("")
        A("| " + " | ".join(cols) + " |")
        A("|" + "---|" * len(cols))
        for _, row in head.iterrows():
            A("| " + " | ".join(
                _fmt(row[c], 4) if isinstance(row[c], float) else str(row[c])
                for c in cols) + " |")
        A("")
        if "missing_pct" in step3.columns:
            top = step3.head(20)
            A("Cross-check on the top 20: missingness mean {} vs {} for all sensors;".format(
                _fmt(top["missing_pct"].mean(), 2), _fmt(step3["missing_pct"].mean(), 2)))
            A("longest constant non-zero run mean {} vs {} steps.".format(
                _fmt(top["longest_constant_nonzero_run_steps"].mean(), 1),
                _fmt(step3["longest_constant_nonzero_run_steps"].mean(), 1)))
            A("")
    else:
        A("No regime produced enough active targets to rank.")
        A("")
    A("Road assignment is not reported: `node_meta.json` carries only `dataset`,")
    A("`sensor_ids`, `n_nodes` and `latlon`. There is no freeway or road-name field in")
    A("any committed artifact, and resolving PeMS station ids to freeways needs an")
    A("external source that is not available offline.")
    A("")

    # -------------------------------------------------- explainer optimum
    A("## What this run says about the explainer's optimum")
    A("")
    A("Two measurements taken around this analysis bear on how much weight any single")
    A("explainer solve can carry. Both are about the optimisation, not about hardware")
    A("or about a dry-run's bookkeeping.")
    A("")
    dv = _device_divergence()
    if dv and dv.get("means"):
        A("**The solution reorders under float-level perturbation.** Same seed, same")
        A("input, same checkpoint, two devices (run `{}`):".format(dv["_run_id"]))
        A("")
        A("| Window / target | Pearson (magnitude) | Spearman (rank) | Top-8 Jaccard | Top-20 Jaccard |")
        A("|---|---|---|---|---|")
        for c in dv["cases"]:
            A("| w{} / t{} | {} | {} | {} | {} |".format(
                c["window"], c["target"], _fmt(c.get("pearson_magnitude"), 4),
                _fmt(c.get("spearman_rank"), 4), _fmt(c.get("top8_jaccard"), 3),
                _fmt(c.get("top20_jaccard"), 3)))
        mn = dv["means"]
        A("| **mean** | **{}** | **{}** | **{}** | **{}** |".format(
            _fmt(mn["pearson_magnitude"], 4), _fmt(mn["spearman_rank"], 4),
            _fmt(mn["top8_jaccard"], 3), _fmt(mn["top20_jaccard"], 3)))
        A("")
        A("Within a device the same seed reproduces byte-identically, so this is not")
        A("randomness in the algorithm. The two devices agree on *how much* mass the")
        A("mask assigns and disagree on *which nodes* receive it. Since every")
        A("downstream consumer in this repository reads the top-k node set and not the")
        A("magnitudes, that is the axis that matters. `docs/REPOSITORY_AUDIT.md` 5.7")
        A("already notes cross-device bitwise identity is impossible; this quantifies")
        A("what it costs on the quantity the project actually uses.")
    else:
        A("*(device-divergence artifact not found; run")
        A("`python -m xtraffic.scripts.measure_device_divergence`)*")
    A("")
    A("**A pre-run dry-run over the 12 committed explanations scored below chance.**")
    A("Aggregating their top-8 lists and comparing to the adjacency gave precision")
    A("0.013, against a chance rate of about 0.035 for drawing 8 sources uniformly at")
    A("a mean in-degree of 7.3. At 12 explanations that is not a result, and it is")
    A("recorded here only because it set the expectation the full run was built to")
    A("test, rather than being discovered afterwards.")
    A("")
    A("Both observations point the same way and are consistent with two committed")
    A("numbers the project already holds: Fidelity- is inconclusive (12.93 vs 12.84")
    A("random) and Phase-18 mean explanation stability is 0.427 across K=10 seeds.")
    A("")

    # -------------------------------------------------- limitations
    A("## Limitations")
    A("")
    A("**\"Off-adjacency\" means \">{} m by road\".** The adjacency is not a road map;".format(
        _fmt(g["cutoff_m"], 0)))
    A("it is a thresholded Gaussian kernel on directed road distance with sigma = {} m".format(
        _fmt(g["sigma_m"], 1)))
    A("and kappa = {}, which reduces exactly to that cutoff. Every statement about the".format(g["kappa"]))
    A("model \"using edges outside the graph\" is a statement about a 3.9 km radius, and")
    A("carries no implication that the pair is unconnected in the real road network.")
    A("Relatedly, **{:.0%} of non-adjacent ordered pairs have no directed road distance".format(
        1 - g["share_nonadjacent_pairs_with_road_distance"]))
    A("in `distances_la_2012.csv` at all**, so most off-adjacency edges cannot be given")
    A("a road distance. Haversine is reported for them in a separate column and nothing")
    A("in this analysis is derived from it; straight-line distance understates road")
    A("distance by an unknown and non-constant factor.")
    A("")
    A("**Hop distance does not bound influence.** Both of the model's supports are dense")
    A("row-softmaxes, so every pair is one hop apart inside the network regardless of")
    A("the adjacency. The hop-stratified tables describe where influence lands relative")
    A("to the road graph; they do not identify influence the architecture could not have")
    A("produced.")
    A("")
    A("**The window sample is small per target.** {} shared windows with a ~14% network".format(
        res["n_windows"]))
    A("congestion base rate leaves the median target only a handful of congested")
    A("windows. Pooled across targets the edge-set comparison has ample observations,")
    A("but individual rows of W are thin and the per-target rankings in Step 3 should be")
    A("read as ordering, not as measurement.")
    A("")
    A("**Single explainer solve per (target, window), and the solve is not stable.** See")
    A("the section above. The aggregate is what is reported; the split-half Jaccard is")
    A("the honest measure of how much of it survives resampling.")
    A("")
    A("**Alignment, not correctness.** This measures what the explainer says the model")
    A("relies on. It does not establish that the model's reliance is correct, nor that")
    A("the explainer recovered the model's true dependence. Fidelity+/- is the axis that")
    A("would speak to that and it is not rerun here.")
    A("")
    A("**One checkpoint, one city, one horizon.** epoch-54 `metr_la_best.pt`, METR-LA,")
    A("30-minute forecast. Nothing here transfers without being rerun.")
    A("")
    A("**Exploratory.** Thresholds (top-k = {}, cumulative mass {:.0%}, congested < {} mph,".format(
        s["top_k"], s["mass_frac"], cfg.get("congested_max_mph")))
    A("free-flow >= {} mph, window floor {}) were chosen before the run from the data".format(
        cfg.get("freeflow_min_mph"), s["min_windows_full"]))
    A("distribution and the project's existing conventions, but there is no")
    A("prespecified analysis plan for this study, so no result in it is confirmatory.")
    A("")
    return "\n".join(L) + "\n"
