"""Direct paired comparison: XTRAFFIC (real explanation) vs XTRAFFIC_RAND (noise).

WHY PAIRED, AND WHY NOT AGAINST RAW
    The 2-page draft compared each arm's mean against RAW. That comparison turned
    out to be unstable across sample sizes -- RAW scored 0.2756 on this 150-scenario
    subset against 0.2333 on the full 444 -- so an "is it above RAW" test says more
    about which scenarios were sampled than about the explanation.

    The question that does not depend on RAW is whether showing the agent a REAL
    explanation beats showing it a RANDOM one, on the SAME scenario, at the SAME
    seed. Both arms are present for all 150 scenarios x 3 seeds, so this is a
    within-scenario paired contrast and RAW never enters it.

BOOTSTRAP UNIT
    Scenarios, not decisions. Each scenario contributes three decisions (one per
    seed) that share a window, a ground-truth action and an explanation; resampling
    decisions would treat those three as independent and report a CI that is too
    narrow. Resampling scenarios carries all of a scenario's seeds together.

    python -m xtraffic.scripts.paired_decision_ab

Python 3.9 compatible.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from ..utils.io_utils import PKG_ROOT

RID = "20260825T201433Z__grounding_without_information__f193e764__a1b0fc36"
RUN = os.path.join(PKG_ROOT, "evaluation", "results", "raw", RID)
OUT = os.path.join(PKG_ROOT, "evaluation", "results", "processed", RID)
N_BOOT = 10000
SEED = 42
REAL, RAND = "XTRAFFIC", "XTRAFFIC_RAND"


def load() -> List[Dict[str, str]]:
    with open(os.path.join(RUN, "part3_per_decision.csv")) as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    rows = load()
    # (sample_index, seed) -> {condition: row}
    cell: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = {}
    for r in rows:
        cell.setdefault((r["sample_index"], r["seed"]), {})[r["condition"]] = r

    paired = [(k, v) for k, v in cell.items() if REAL in v and RAND in v]
    scenarios = sorted({k[0] for k, _ in paired})

    # Per-scenario mean over its seeds, so the bootstrap unit is one number per
    # scenario per metric per arm.
    by_scen: Dict[str, Dict[str, List[float]]] = {}
    for (sample, _seed), v in paired:
        d = by_scen.setdefault(sample, {"acc_real": [], "acc_rand": [],
                                        "dly_real": [], "dly_rand": []})
        d["acc_real"].append(float(v[REAL]["correct"]))
        d["acc_rand"].append(float(v[RAND]["correct"]))
        d["dly_real"].append(float(v[REAL]["delay_reduction"]))
        d["dly_rand"].append(float(v[RAND]["delay_reduction"]))

    acc_r = np.array([np.mean(by_scen[s]["acc_real"]) for s in scenarios])
    acc_n = np.array([np.mean(by_scen[s]["acc_rand"]) for s in scenarios])
    dly_r = np.array([np.mean(by_scen[s]["dly_real"]) for s in scenarios])
    dly_n = np.array([np.mean(by_scen[s]["dly_rand"]) for s in scenarios])

    rng = np.random.default_rng(SEED)
    out: Dict[str, object] = {
        "n_scenarios": len(scenarios),
        "n_paired_decisions": len(paired),
        "seeds_per_scenario": len(paired) // max(len(scenarios), 1),
        "bootstrap": {"n_iters": N_BOOT, "seed": SEED, "unit": "scenario"},
        "comparison": "XTRAFFIC (real explanation) minus XTRAFFIC_RAND (random), "
                      "paired within scenario and seed; RAW not involved",
    }

    for name, real, rand in (("accuracy", acc_r, acc_n),
                             ("delay_reduction", dly_r, dly_n)):
        diff = real - rand
        idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
        boots = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        n_better = int((diff > 0).sum())
        n_worse = int((diff < 0).sum())
        out[name] = {
            "real_mean": float(real.mean()),
            "random_mean": float(rand.mean()),
            "paired_mean_diff": float(diff.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_scenarios_real_better": n_better,
            "n_scenarios_random_better": n_worse,
            "n_scenarios_tied": int(len(diff) - n_better - n_worse),
        }

    # The RAW instability, recorded so the footnote in the paper has a source.
    summ = json.load(open(os.path.join(RUN, "part3_summary.json")))
    sub = {c["condition"]: c for c in summ["conditions"]}
    full = {c["condition"]: c for c in summ["committed_full_study_n444"]}
    out["raw_baseline_instability"] = {
        "raw_accuracy_on_this_subset_n150": sub["RAW"]["accuracy_mean"],
        "raw_accuracy_full_study_n444": full["RAW"]["accuracy_mean"],
        "xtraffic_accuracy_on_this_subset_n150": sub["XTRAFFIC"]["accuracy_mean"],
        "xtraffic_accuracy_full_study_n444": full["XTRAFFIC"]["accuracy_mean"],
        "note": ("RAW moves by more between sample sizes than XTRAFFIC exceeds it "
                 "by on either, so an above-RAW test is not informative here; the "
                 "paired contrast above avoids RAW entirely"),
    }

    with open(os.path.join(OUT, "paired_decision_ab.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("PAIRED  XTRAFFIC minus XTRAFFIC_RAND  (n={} scenarios x {} seeds, "
          "bootstrap {} over scenarios)".format(
              out["n_scenarios"], out["seeds_per_scenario"], N_BOOT))
    for name in ("accuracy", "delay_reduction"):
        d = out[name]
        print("  {:16s} real {:.4f}  random {:.4f}  diff {:+.4f}  "
              "95% CI [{:+.4f}, {:+.4f}]  {}".format(
                  name, d["real_mean"], d["random_mean"], d["paired_mean_diff"],
                  d["ci_lo"], d["ci_hi"],
                  "EXCLUDES 0" if d["excludes_zero"] else "spans 0"))
        print("  {:16s} scenarios real better {} / random better {} / tied {}"
              .format("", d["n_scenarios_real_better"],
                      d["n_scenarios_random_better"], d["n_scenarios_tied"]))
    ri = out["raw_baseline_instability"]
    print("  RAW accuracy: {:.4f} at n=150 vs {:.4f} at n=444".format(
        ri["raw_accuracy_on_this_subset_n150"], ri["raw_accuracy_full_study_n444"]))
    print("\nwrote {}".format(os.path.join(OUT, "paired_decision_ab.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
