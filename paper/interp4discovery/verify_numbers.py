"""Self-check for the Interpretability-for-Discovery paper.

Two jobs, both run before any resubmission:

  1. NUMBERS. Re-read every source file and assert the literal string is present
     in main.tex, so an edit that changes a number without changing its source is
     caught rather than shipped. Numbers reused from the JUDGe paper are checked
     against the SAME files that paper cites, so the two cannot drift apart.

  2. LANGUAGE. The withdrawn survivor result must never be described with
     discovery language. "discover", "reveal", "finding" and friends are banned
     within its section; the words are reserved for what the paper actually
     establishes (the instrumentation, and the geometry baseline's positive
     result). A paper that hedges in the abstract and then calls a withdrawn
     effect a "finding" three pages later has un-withdrawn it.

    python3 paper/interp4discovery/verify_numbers.py

Exit status 0 = every check passed.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "xtraffic", "evaluation", "results")
PROC = os.path.join(RES, "processed")
IG = os.path.join(PROC, "20260824T235016Z__influence_graph_solves__73e75966__ee9a7944")
LG = os.path.join(PROC, "20260825T093132Z__learned_semantic_graph__4715a20b__b3c55183")
GWI_ID = "20260825T201433Z__grounding_without_information__f193e764__a1b0fc36"
GWI = os.path.join(PROC, GWI_ID)
DEV = os.path.join(RES, "raw",
                   "20260824T235643Z__device_divergence__58efae64__ee9a7944")
TEX = os.path.join(HERE, "main.tex")

# Section that must stay free of discovery language, and the banned stems.
WITHDRAWN_SECTION = r"\section{Result 5: an apparent effect, withdrawn}"
BANNED = ("discover", "reveal", "finding", "we find that", "establishes that",
          "demonstrates that", "shows that the explainer agrees")


def main() -> int:
    tex = open(TEX).read()
    fail = []

    def need(label, s):
        if s not in tex:
            fail.append("MISSING {}: {!r}".format(label, s))

    # ---------------- numbers ------------------------------------------
    ig = json.load(open(os.path.join(IG, "metrics.json")))
    geo = ig["geometry"]
    need("sigma", str(geo["sigma_m"]))
    need("cutoff", "{:.1f}".format(geo["cutoff_m"]))
    need("n adjacency edges", "{:,}".format(geo["n_adjacency_edges"]).replace(",", "{,}"))
    need("adjacency density", "{:.4f}".format(geo["adjacency_density"]))
    need("road coverage", "{:.1f}".format(100 * geo["share_pairs_with_road_distance"]))
    need("road coverage nonadj",
         "{:.1f}".format(100 * geo["share_nonadjacent_pairs_with_road_distance"]))

    for reg, _lbl in (("congested", "congested"), ("free_flow", "free-flow")):
        r = ig["per_regime"][reg]
        m = r["sources_for_mass_frac"]
        need("mass mean " + reg, "{:.1f}".format(m["mean"]))
        need("mass sd " + reg, "{:.1f}".format(m["std"]))
        need("explainer precision " + reg, "{:.4f}".format(r["edges"]["top_k"]["precision"]))
        for b in ("uniform_random", "degree_matched_random", "nearest_by_road_distance"):
            need("baseline {} {}".format(b, reg),
                 "{:.4f}".format(r["baselines"][b]["precision"]))
        st = r["stability"]
        need("split-half " + reg, "{:.4f}".format(st["jaccard_all_edges"]))
        need("random ref " + reg, "{:.4f}".format(st["jaccard_two_random_draws_reference"]))
        need("beyond-k enrichment " + reg, "{:.3f}".format(r["beyond_k"]["enrichment_over_base_rate"]))
        vs = r["vs_learned_semantic_graph"]["top_k_jaccard"]
        need("W vs Asem " + reg, "{:.4f}".format(vs["mean"]))
        need("W vs Asem zero-overlap " + reg, str(vs["n_targets_with_zero_overlap"]))

    surv = ig["per_regime"]["free_flow"]["beyond_k_survivors"]
    need("survivors n", str(surv["n_intersection"]))
    need("survivors top8", "{:.4f}".format(surv["intersection"]["learned_rank_of_206"]["share_in_top_8"]))
    need("one-half-only top8", "{:.4f}".format(surv["one_half_only"]["learned_rank_of_206"]["share_in_top_8"]))
    need("survival rate", "{:.4f}".format(surv["survival_rate"]))

    sens = ig["survivor_sample_size_sensitivity"]
    pre = sens["per_regime"]["free_flow"]
    need("prefix n", str(sens["prefix_n_targets"]))
    need("prefix survivors", str(pre["n_intersection"]))
    need("prefix top8", "{:.4f}".format(pre["intersection"]["learned_rank_of_206"]["share_in_top_8"]))
    need("prefix one-half-only", "{:.4f}".format(pre["one_half_only"]["learned_rank_of_206"]["share_in_top_8"]))

    lg = json.load(open(os.path.join(LG, "learned_graph.json")))
    ck = lg["per_checkpoint"]["metr_la_best.pt"]
    need("alpha learned", "{:.4f}".format(ck["alpha"]["weight_on_learned_semantic"]))
    need("alpha given", "{:.4f}".format(ck["alpha"]["weight_on_given_adjacency"]))
    need("learned mass", "{:.1f}".format(ck["sources_for_mass_frac"]["mean"]))
    need("learned road overlap", "{:.4f}".format(ck["overlap_with_nearest_road"]))
    ra = ck["road_alignment_semantic"]
    need("learned road mass", "{:.4f}".format(ra["mass_on_adjacency_edges"]))
    need("learned road lift", "{:.3f}".format(ra["lift_over_chance"]))
    for pair in lg["cross_checkpoint_stability"]["pairwise_topk_jaccard"]:
        need("cross-ckpt J", "{:.4f}".format(pair["topk_jaccard"]))
    need("cross-ckpt ref",
         "{:.4f}".format(lg["cross_checkpoint_stability"]["two_random_draws_reference"]))

    dev = json.load(open(os.path.join(DEV, "device_divergence.json")))["means"]
    for k in ("pearson_magnitude", "spearman_rank", "top8_jaccard", "top20_jaccard"):
        need("device " + k, "{:.4f}".format(dev[k]))

    p4 = json.load(open(os.path.join(GWI, "part4_metrics.json")))
    for st in p4["settings"]:
        need("per-target split-half " + st["setting"],
             "{:.4f}".format(st["split_half_jaccard_mean"]))
        need("per-target sd " + st["setting"],
             "{:.4f}".format(st["split_half_jaccard_std"]))
        need("seed-repeat " + st["setting"],
             "{:.4f}".format(st["seed_repeat_jaccard_mean"]))

    ver = json.load(open(os.path.join(GWI, "verdicts.json")))["P4"]
    for c in ver["checks"]:
        if c.get("ci") and c["ci"][0] is not None:
            need("P4 rho", "{:.4f}".format(c["value"]))
            need("P4 ci lo", "{:.4f}".format(c["ci"][0]))
            need("P4 ci hi", "{:+.4f}".format(c["ci"][1]))
    if ver["sub_verdicts"]["P4a_correlation"] != "FALSIFIED":
        fail.append("CLAIM BROKEN: P4a is no longer FALSIFIED upstream")

    shap = json.load(open(os.path.join(RES, "shap_comparison",
                                       "shap_comparison_summary.json")))
    need("shap overlap", "{:.4f}".format(shap["gnn_vs_shap_topk_overlap"]["mean"]))
    need("shap precision", "{:.4f}".format(shap["aggregate"]["A"]["cause_precision_mean"]))
    need("shap halluc", "{:.4f}".format(shap["aggregate"]["A"]["hallucination_rate_mean"]))
    need("shap n", str(shap["n_scenarios"]))

    # Reused-from-JUDGe consistency: the same source files must still say the same
    # thing, so the two papers cannot quietly disagree.
    judge = os.path.join(HERE, "..", "judge", "main.tex")
    if os.path.exists(judge):
        jt = open(judge).read()
        for shared in ("0.2256", "0.1399", "0.0165", "0.0182", "0.2323", "0.3480",
                       "0.9193", "0.3991", "0.0221", "0.9821", "0.0179",
                       "-0.2008", "-0.1732"):
            if shared in jt and shared not in tex:
                fail.append("DRIFT vs JUDGe paper: {!r} is in that paper but not "
                            "here, though both cite the same source".format(shared))

    # ---------------- language -----------------------------------------
    if WITHDRAWN_SECTION not in tex:
        fail.append("MISSING the withdrawn-result section header; the language "
                    "guard below cannot be scoped without it")
    else:
        start = tex.index(WITHDRAWN_SECTION)
        end = tex.index(r"\section{Result 6", start)
        body = tex[start:end].lower()
        for w in BANNED:
            if w in body:
                fail.append("BANNED LANGUAGE in the withdrawn-result section: "
                            "{!r}. That effect did not replicate; discovery words "
                            "are reserved for the instrumentation and the geometry "
                            "baseline.".format(w))
        for required in ("withdraw", "did not reproduce", "not been shown to replicate"):
            if required not in body:
                fail.append("MISSING withdrawal language {!r} in Result 5".format(required))

    if r"\section{Responsible use statement}" not in tex:
        fail.append("MISSING Responsible Use Statement -- this venue desk-rejects "
                    "submissions without one")

    # n=40 single-setting entropy figure: must come from its own run_dir, not
    # from an overwritten inline computation.
    n40 = sorted(glob.glob(os.path.join(
        RES, "raw", "*__entropy_gate_single_setting__*",
        "entropy_gate_single_setting.json")))
    if not n40:
        fail.append("MISSING the n=40 entropy run_dir; the single-setting rho must "
                    "be sourced from an artifact, not from session memory")
    else:
        d40 = json.load(open(n40[-1]))
        sc = d40["spearman"]["entropy_of_mean"]
        need("n=40 rho", "{:.4f}".format(sc["rho"]))
        need("n=40 ci lo", "{:.4f}".format(sc["ci_lo"]))
        need("n=40 ci hi", "{:.4f}".format(sc["ci_hi"]))
        if not sc["passes_pre_registered_criteria"]:
            fail.append("CLAIM BROKEN: the n=40 run no longer passes the "
                        "pre-registered criteria the paper says it passed")
        if d40["n_targets_used"] != 40:
            fail.append("n=40 run used {} targets".format(d40["n_targets_used"]))

    # SHAP is a CROSS-CHECKPOINT comparison (epoch 34 vs the epoch-54 results
    # elsewhere). The caveat must stay, or Result 4 silently pools two models.
    for token in ("Checkpoint caveat", "epoch-34 checkpoint"):
        if token not in tex:
            fail.append("MISSING SHAP checkpoint caveat {!r}: that comparison ran "
                        "on epoch 34 while Results 1-3 are epoch 54".format(token))

    for f in fail:
        print("  " + f)
    print("\n{} failure(s)".format(len(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
