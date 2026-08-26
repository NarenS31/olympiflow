"""Create the ONE run directory for the grounding-without-information experiment
and write its pre-registration.

This script is the FIRST thing that runs. It must complete before any explainer
solve and before any scored LLM call, because predictions.md is a pre-registration
and a pre-registration written after the data is in is not one.

    python -m xtraffic.scripts.init_grounding_noise_run

Python 3.9 compatible.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from ..reproducibility import run_dir, seeds
from ..utils.io_utils import PKG_ROOT

REPO_ROOT = os.path.dirname(PKG_ROOT)
EXPERIMENT = "grounding_without_information"
CKPT = "models/gnn/checkpoints/metr_la_best.pt"
DATASET = "metr_la"

# --- The four parts' knobs. Everything a reader needs to know what was fixed in
# --- advance lives in this dict and is hashed into the run id.
CONFIG: Dict[str, Any] = {
    "experiment": EXPERIMENT,
    "dataset": DATASET,
    "checkpoint": CKPT,
    "horizon_step": 6,
    "seed": 42,
    "device": "cpu",
    "llm": {
        "model": "llama3.1:8b",
        "temperature": 0.1,
        "call_path": "advisor_inline_unseeded",
        "why": ("the committed A / B / XTRAFFIC numbers were produced on this "
                "path; the seeded logging path pins top_p/top_k/num_ctx and "
                "therefore changes generated text (REPOSITORY_AUDIT 5.2), which "
                "would break the comparison these four parts exist to make"),
    },
    "part1": {
        "n_scenarios": 93,
        "per_stratum": 7,
        "conditions_reused": ["A", "B"],
        "conditions_new": ["A_rand", "A_mismatch"],
        "a_rand_seed": 20260825,
        "a_mismatch_seed": 20260826,
        "token_parity_tolerance": 0.05,
        "bootstrap_iters": 10000,
        "bootstrap_seed": 42,
    },
    "part2": {
        "f1_threshold": 0.70,
        "max_rounds": 3,
        "source_condition": "A_rand",
        "baseline": "committed active_grounding run on A (n=93)",
    },
    "part3": {
        "n_scenarios": 150,
        "subsample_of": 444,
        "subsample_seed": 42,
        "subsample_rule": "stratum-preserving proportional, seeded shuffle per cell",
        "decision_seeds": [0, 1, 2],
        "new_condition": "XTRAFFIC_RAND",
        "reaggregated_on_same_subsample": ["RANDOM", "RAW", "XTRAFFIC"],
    },
    "part4": {
        "lambda_size_settings": [0.15, 0.45, 1.50],
        "lambda_size_labels": ["current", "3x", "10x"],
        "lambda_ent": 0.05,
        "n_targets": 40,
        "n_windows": 24,
        "target_order_source": ("windows.json target_order of run "
                                "20260824T235016Z__influence_graph_solves"),
        "reuse_existing_solves_for": "current",
        "seed_repeat_targets": 10,
        "seed_repeat_seeds": [0, 1],
        "top_k": 8,
        "explainer_epochs": 200,
        "entropy": "normalised Shannon entropy of off-self node_imp / log(N-1)",
        "threshold_rule": "Youden-J on the sweep pool only; see predictions.md P4",
        "spearman_bootstrap_iters": 10000,
        "spearman_bootstrap_seed": 42,
        "spearman_bootstrap_unit": "target (clustered: a target carries all its settings)",
    },
    "budget_hours": 14,
}


PREDICTIONS = """# Pre-registration — grounding without information

Written **before** any explainer solve and **before** any scored LLM call in this
experiment. Run id is in the manifest next to this file. Nothing below is edited
after the fact; the report records confirmed/falsified against exactly these
words.

## Disclosure: LLM calls made before this file existed

Nine (9) calls to `llama3.1:8b` were made before this pre-registration, solely to
measure throughput so the compute budget could be estimated:

  * 1 cold + 4 warm condition-A advisories on committed cached explanations
  * 4 concurrent condition-A advisories (an Ollama-parallelism test)

None were scored, none entered any condition, none are read by any part of this
experiment. They are recorded here rather than omitted. Measured constants that
came out of them and that set the budget: condition-A advisory 34.9 s/call,
sim-eval decision 11.2 s/call, Ollama gives no throughput gain from concurrency.

## Scope changes agreed before pre-registration

1. **Part 3 runs at n=150, not n=444.** Full n=444 x 3 seeds is 8.6 h of the
   ~13 h critical path and would leave no margin. The 150 are a seeded,
   stratum-preserving subsample; `RANDOM` / `RAW` / `XTRAFFIC` are re-aggregated
   on the SAME 150 so all four arms are paired on one population. The committed
   n=444 numbers are quoted alongside but are NOT the comparison.
2. **`A_mismatch` keeps the target and swaps the sources.** Scenario i's
   prediction block is kept; `top_nodes` / `top_edges` / `propagation_path` /
   `propagation_lag_minutes` / `explanation_confidence` come from the real
   committed explanation of a different, seeded-random target. The alternative
   (swap the whole artifact) is condition A re-run on another window and is
   tautological.
3. **The LLM path is the unseeded inline path**, matching the committed runs.
   Seeding it would change generated text and break comparability
   (REPOSITORY_AUDIT 5.2). Every call is logged in full to `llm_calls.jsonl`, so
   the run is auditable and re-derivable from the log. **LLM output is therefore
   NOT byte-reproducible on re-run**; this project has measured that at
   |dF1| ~ 0.08 mean, up to 0.14 on single scenarios. Everything non-LLM
   (explainer solves, artifact construction, sampling, subsampling, bootstrap)
   IS seeded and byte-reproducible.

---

## P1 — `A_rand` and `A_mismatch` F1 / hallucination land within condition A's CI

**Claim.** Replacing the explainer's node importances with seeded uniform noise
(`A_rand`), or with a real explanation belonging to a different target
(`A_mismatch`), does not move the faithfulness metric. The metric scores citation
*compliance*, not evidential *content*, so an artifact with no information in it
should score like one with information in it.

**Confirmation.** The `A_rand` and `A_mismatch` means for `faithfulness_f1` and
`hallucination_rate` each fall inside condition A's 95% bootstrap CI, AND the
paired-difference CI (A minus the noise condition, over the 93 paired scenarios,
10k resamples) spans zero for both metrics.

**Falsification.** Either noise condition's mean falls outside A's CI, or the
paired-difference CI excludes zero, for F1 or hallucination. A *drop* under noise
would mean the metric carries some signal about evidence quality. A *rise* would
mean the metric rewards incoherent evidence, which is worse and must be reported
as such.

**Pre-committed guard.** Prompt token counts for `A_rand` and `A_mismatch` must
sit within 5% of condition A's mean. If they do not, any difference is
confounded by prompt length and P1 is reported as UNTESTED, not confirmed.

## P2 — the loop reaches threshold on noise at the same rate and in the same number of rounds as on A, and precision goes to 1.0

**Claim.** The active grounding loop is an enforcement mechanism, not a
verification mechanism. Pointed at random evidence it should converge exactly as
it did on real evidence, because the thing it optimises — citing the artifact's
top-k — is equally achievable whatever the artifact says.

**Baseline (committed, n=93, threshold 0.70, llama3.1:8b).** round0 F1 0.732 /
recall 0.605 / halluc 0.005 / 64.5% reached -> round1 0.864 / 0.774 / 0.000 /
96.8% -> round2 0.870 / 0.782 / 0.000 / 98.9% -> round3 0.874 / 0.788 / 0.000 /
100.0%. Mean F1 gain +0.141, mean 0.40 correction rounds, 100% reached, stop
reasons {threshold_reached: 93}.

**Confirmation.** On `A_rand`: final fraction reaching threshold within 3 rounds
is >= 0.95; mean correction rounds used is within +/-0.25 of 0.40; mean
precision at the final round is >= 0.95.

**Falsification.** Fraction reaching threshold < 0.95, or mean rounds used
outside 0.40 +/- 0.25, or final-round precision < 0.95. Any of these means the
loop is doing something the random artifact cannot support — which would be
evidence FOR the loop and against this prediction.

**What confirmation would mean.** A correction loop converging on random
evidence. The loop's own caveat ("enforcement, not proof") would stop being a
caveat and become the finding.

## P3 — decisions made on `A_rand` explanations land inside XTRAFFIC's CI

**Claim.** If the decision agent's advantage over RAW came from the explanation's
spatial content, randomising that content should cost it. If the advantage came
from the *presence* of a structured, confident-looking evidence block, it should
not.

**Confirmation.** `XTRAFFIC_RAND` accuracy and delay reduction both fall within
the seed-spread of `XTRAFFIC` re-aggregated on the same 150 scenarios (mean
+/- 1 sd over the 3 decision seeds), and `XTRAFFIC_RAND` remains above `RAW`.

**Falsification.** `XTRAFFIC_RAND` falls outside that interval on either metric.
Falling to or below `RAW` would mean the explanation's content is load-bearing
for decision quality — a positive result for the pipeline and against this
prediction.

**Committed reference (n=444, 3 seeds).** RANDOM 0.200 +/- 0.019 acc /
13.27 +/- 0.67 delay; RAW 0.233 +/- 0.001 / 15.72 +/- 0.02; XTRAFFIC
0.266 +/- 0.012 / 19.73 +/- 0.22.

## P4 — mask entropy predicts split-half stability (rho < -0.30), and every committed explanation is flagged

**Claim.** A flat importance mask has nothing to be stable about. Normalised
entropy of the node mask should therefore predict, negatively, how much a
target's top-8 survives resampling the windows it was built from.

**Definitions, fixed now.**
  * `node_imp` is the raw sigmoid mask over N=207 nodes. Drop the self entry,
    normalise the remaining 206 to sum 1, take Shannon entropy, divide by
    log(206). 1.0 = perfectly flat.
  * split-half Jaccard = top-8 sets built from the two seeded halves of that
    target's 24 windows, per `influence_graph.split_halves`.
  * The scatter pools all (target, sparsity-setting) pairs with both quantities
    defined. Spearman rho with a 10k bootstrap CI resampling **targets**, not
    pairs, so a target's three settings stay together.

**Confirmation.** Pooled Spearman rho < -0.30 and its 95% CI excludes zero.

**Falsification.** rho >= -0.30, or the CI spans zero. A rho near zero means
entropy is not a usable reliability proxy and the "flag high-entropy
explanations" idea does not survive.

**The threshold, chosen on the sweep and never on the 12.** Pool the sweep's
(target, setting) pairs. Label a pair *unstable* if its split-half Jaccard is
below the pooled median. Choose T as the normalised-entropy threshold maximising
Youden's J (sensitivity + specificity - 1) for predicting *unstable*. T is then
applied unchanged to the 12 committed explanations, whose masks are re-solved at
the current setting (seed 0, CPU, same checkpoint and window) purely to obtain
the full node_imp vector the committed JSONs do not store.

**Confirmation.** 12/12 committed explanations have normalised entropy > T.

**Falsification.** Fewer than 12 of 12. Reported as the exact fraction either
way; "most of them" is not a result.

---

## Standing rules for this run

  * No hedging after the fact. Each prediction gets CONFIRMED or FALSIFIED
    against the criteria above, with the number that decided it.
  * A falsified prediction is a result, not a failure, and is reported at the
    same prominence as a confirmed one.
  * Nothing committed is modified. `explanations_cache/` is read-only for this
    experiment; every artifact this run creates lives in the run directory.
  * No MPS anywhere: CPU and MPS disagree on top-k identity (Spearman 0.578,
    top-8 Jaccard 0.399 at the same seed), and top-k identity is the object of
    Part 4.
"""


def main() -> int:
    seed_rec = seeds.set_all_seeds(int(CONFIG["seed"]))
    pdir = os.path.join(PKG_ROOT, "data", "processed", DATASET)
    run = run_dir.RunDir.create(
        REPO_ROOT, EXPERIMENT, CONFIG,
        artifacts={"checkpoint": os.path.join(PKG_ROOT, CKPT),
                   "adjacency": os.path.join(pdir, "adjacency.npy"),
                   "node_meta": os.path.join(pdir, "node_meta.json"),
                   "test_split": os.path.join(pdir, "test.npz")},
        ollama_host="http://localhost:11434",
        ollama_model=str(CONFIG["llm"]["model"]),
        notes=("One experiment, four parts, pre-registered. Does the faithfulness "
               "metric / the grounding loop / the decision agent distinguish real "
               "explainer evidence from noise, and does mask entropy predict "
               "explainer stability? CPU only, no MPS."))
    run.write_json("seeds.json", seed_rec)
    run.write_text("predictions.md", PREDICTIONS)
    run.start()
    print(run.run_id)
    print(run.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
