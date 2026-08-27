"""Self-check: every number in main.tex traces to a file on disk.

Run this before any resubmission. It re-reads the sources and asserts the
literal strings are present in main.tex, so a hand-edit that changes a number
without changing its source is caught rather than shipped.

It also asserts the EXCLUSIONS hold: the cross-city rebuild's values must not
appear anywhere, and no cross-model claim may be made, because neither is
supported by claims.csv / claims_provenance.json.

    python3 paper/judge/verify_numbers.py

Exit status 0 = every check passed.
"""
from __future__ import annotations

import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RID = "20260825T201433Z__grounding_without_information__f193e764__a1b0fc36"
PROC = os.path.join(REPO, "xtraffic", "evaluation", "results", "processed", RID)
RAW = os.path.join(REPO, "xtraffic", "evaluation", "results", "raw", RID)
RES = os.path.join(REPO, "xtraffic", "evaluation", "results")
IG = os.path.join(REPO, "xtraffic", "evaluation", "results", "processed",
                  "20260824T235016Z__influence_graph_solves__73e75966__ee9a7944")
TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.tex")


def main() -> int:
    tex = open(TEX).read()
    rows = {r["condition"]: r for r in csv.DictReader(
        open(os.path.join(PROC, "claims.csv")))}
    failures = []

    def need(label, s):
        if s not in tex:
            failures.append("MISSING {}: {!r}".format(label, s))

    def forbid(label, s):
        if s in tex:
            failures.append("FORBIDDEN {}: {!r} appears in main.tex".format(label, s))

    # Table 1 must match claims.csv exactly, to 4 dp.
    for cond in ("A", "B", "C", "A_rand", "A_mismatch"):
        r = rows[cond]
        for col in ("cause_precision", "cause_recall", "faithfulness_f1",
                    "hallucination_rate"):
            need("{}.{}".format(cond, col), "%.4f" % float(r[col]))

    # Conditions we must NOT put in Table 1 as if they were the same draw.
    # (They may be named in prose; we only check the numbers are not smuggled in
    # as table rows, which the table-row check above already pins.)

    # Deliberate exclusions from claims_provenance.json.
    prov = json.load(open(os.path.join(PROC, "claims_provenance.json")))
    cx = prov["excluded_deliberately"]["cross_city_rebuild"]["values"]
    forbid("cross-city F1", "%.4f" % cx["A_faithfulness_f1"])
    for s in ("0.839", "0.8387", "0.7232"):
        forbid("cross-city value", s)
    if "mistral" in tex.lower():
        failures.append("FORBIDDEN: mistral is named; claims_provenance.json "
                        "does not support any cross-model claim for A_rand")

    # Flat-mask figures.
    ig = json.load(open(os.path.join(IG, "metrics.json")))
    for reg in ("congested", "free_flow"):
        m = ig["per_regime"][reg]["sources_for_mass_frac"]
        need("80%% mass mean {}".format(reg), "%.1f" % m["mean"])
        need("80%% mass sd {}".format(reg), "%.1f" % m["std"])
    need("of_available", str(ig["per_regime"]["congested"]
                             ["sources_for_mass_frac"]["of_available"]))

    # Loop numbers, real vs random.
    ag = json.load(open(os.path.join(RES, "active_grounding",
                                     "active_grounding_summary.json")))
    p2 = json.load(open(os.path.join(RAW, "part2_summary.json")))
    need("round0 real", "%.1f" % (100 * ag["per_round"][0]["frac_reached_threshold"]))
    need("round0 random", "%.1f" % (100 * p2["per_round"][0]["frac_reached_threshold"]))
    need("round0 F1 real", "%.4f" % ag["per_round"][0]["mean_f1"])
    need("round0 F1 random", "%.4f" % p2["per_round"][0]["mean_f1"])
    need("final F1 real", "%.4f" % ag["per_round"][-1]["mean_f1"])
    need("final F1 random", "%.4f" % p2["per_round"][-1]["mean_f1"])
    need("final reach random", "%.1f" % (100 * p2["per_round"][-1]["frac_reached_threshold"]))

    # Resolver gate.
    res = json.load(open(os.path.join(RES, "resolver_accuracy.json")))
    need("resolver n", "{}/{}".format(res["n_pass"], res["n_cases"]))
    need("resolver pct", "%.1f" % (100 * res["accuracy"]))

    # Model actually used for the new conditions.
    p1 = json.load(open(os.path.join(RAW, "part1_summary.json")))
    need("model", p1["model"])
    b = p1["bootstrap_ci"]
    need("bootstrap iters", "{:,}".format(b["n_iters"]).replace(",", "{,}"))
    for c in ("A", "A_rand", "A_mismatch"):
        pc = b["per_condition"][c]["faithfulness_f1"]
        need("{} F1 CI lo".format(c), "%.4f" % pc["lo"])
        need("{} F1 CI hi".format(c), "%.4f" % pc["hi"])

    # The identity claim the paper leads with.
    if rows["A"]["cause_precision"] != rows["A_rand"]["cause_precision"]:
        failures.append("CLAIM BROKEN: A and A_rand precision are no longer equal")
    if rows["A"]["hallucination_rate"] != rows["A_rand"]["hallucination_rate"]:
        failures.append("CLAIM BROKEN: A and A_rand hallucination are no longer equal")


    # ---- numbers added for the 4-page Short Papers version ----------------
    IG_M = json.load(open(os.path.join(IG, "metrics.json")))
    for reg in ("congested", "free_flow"):
        st = IG_M["per_regime"][reg]["stability"]
        need("split-half {}".format(reg), "%.4f" % st["jaccard_all_edges"])
        need("random ref {}".format(reg),
             "%.4f" % st["jaccard_two_random_draws_reference"])

    p4 = json.load(open(os.path.join(PROC, "part4_metrics.json")))
    for st in p4["settings"]:
        need("split-half {}".format(st["setting"]),
             "%.4f" % st["split_half_jaccard_mean"])
        need("split-half sd {}".format(st["setting"]),
             "%.4f" % st["split_half_jaccard_std"])
        need("seed-repeat {}".format(st["setting"]),
             "%.4f" % st["seed_repeat_jaccard_mean"])

    dev = json.load(open(os.path.join(
        RAW.replace(RID, "20260824T235643Z__device_divergence__58efae64__ee9a7944"),
        "device_divergence.json")))["means"]
    for k in ("pearson_magnitude", "spearman_rank", "top8_jaccard", "top20_jaccard"):
        need("device {}".format(k), "%.4f" % dev[k])

    shap = json.load(open(os.path.join(RES, "shap_comparison",
                                       "shap_comparison_summary.json")))
    need("shap overlap", "%.4f" % shap["gnn_vs_shap_topk_overlap"]["mean"])
    for m in ("A", "D"):
        need("shap {} F1".format(m),
             "%.4f" % shap["aggregate"][m]["faithfulness_f1_mean"])
    need("shap precision", "%.4f" % shap["aggregate"]["A"]["cause_precision_mean"])
    need("shap halluc", "%.4f" % shap["aggregate"]["A"]["hallucination_rate_mean"])
    need("shap n", str(shap["n_scenarios"]))

    pab = json.load(open(os.path.join(PROC, "paired_decision_ab.json")))
    for metric, fmt in (("accuracy", "%+.4f"), ("delay_reduction", "%+.4f")):
        d = pab[metric]
        need("paired {} diff".format(metric), (fmt % d["paired_mean_diff"]))
        need("paired {} ci lo".format(metric), (fmt % d["ci_lo"]))
        need("paired {} ci hi".format(metric), (fmt % d["ci_hi"]))
        if d["excludes_zero"]:
            failures.append("CLAIM BROKEN: paired {} CI no longer spans zero; the "
                            "paper says both intervals span zero".format(metric))
    for k in ("n_scenarios_real_better", "n_scenarios_random_better",
              "n_scenarios_tied"):
        need("paired accuracy {}".format(k), str(pab["accuracy"][k]))
        need("paired delay {}".format(k), str(pab["delay_reduction"][k]))
    ri = pab["raw_baseline_instability"]
    need("baseline n150", "%.4f" % ri["raw_accuracy_on_this_subset_n150"])
    need("baseline n444", "%.4f" % ri["raw_accuracy_full_study_n444"])

    ver = json.load(open(os.path.join(PROC, "verdicts.json")))["P4"]
    for c in ver["checks"]:
        if c.get("ci") and c["ci"][0] is not None:
            need("P4 rho", "%.4f" % c["value"])
            need("P4 ci lo", "%.4f" % c["ci"][0])
            need("P4 ci hi", "%+.4f" % c["ci"][1])
    if ver["sub_verdicts"]["P4a_correlation"] != "FALSIFIED":
        failures.append("CLAIM BROKEN: P4a is no longer FALSIFIED")
    if ver["sub_verdicts"]["P4b_all_committed_flagged"] != "FALSIFIED":
        failures.append("CLAIM BROKEN: P4b is no longer FALSIFIED")

    for f in failures:
        print("  " + f)
    print("\n{} failure(s)".format(len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
