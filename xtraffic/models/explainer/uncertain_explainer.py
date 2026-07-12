"""Phase 18 — Uncertainty-Aware Explanations (STUB — not built yet).

WHY THIS EXISTS
---------------
A single GNNExplainer run gives ONE top-k node set. A reviewer will ask: "how much
does that depend on the random seed — would you trust one run?" Phase 3 already
answered a scalar version of this (explanation_confidence = mean Jaccard over K
reruns). This phase promotes that scalar to a PER-NODE uncertainty distribution and
makes the LLM honest about it.

THE METHOD (what we will build here)
------------------------------------
  * Run GNNExplainer K=10 times with different random initialisations.
  * For each node, count how often it appears in the top-k, then classify:
        CORE       -> in top-k in >80% of runs   (confident cause)
        PERIPHERAL -> 20%–80% of runs            (uncertain)
        NOISE      -> <20% of runs               (do not assert)
  * Inject this distribution into the advisor prompt so the reasoning is
    epistemically honest:
        "East LA corridor confirmed in 9/10 runs. The I-10 connector is
         uncertain (4/10 runs)."

WHAT IT REUSES
--------------
  * models/explainer/explain.py -> GNNExplainer + ExplanationBuilder. The K reruns
    are the SAME solves Phase 3 already does for the confidence scalar, so the cost
    model is unchanged; we just keep per-node frequencies instead of collapsing to
    one Jaccard number.
  * schema.py -> extend ADDITIVELY (e.g. a per-node `runs_in_topk` / `stability_tier`
    field). Do NOT break the Phase-4 contract — additive fields only, flagged in
    schema.py.

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# TODO(Phase 18): run the K=10 explainer solves and tally per-node top-k frequency.
# TODO(Phase 18): classify_nodes(freqs, *, core=0.8, noise=0.2) -> tiers.
# TODO(Phase 18): additive schema field for the tier/frequency; advisor prompt
#   rendering that asserts CORE, hedges PERIPHERAL, and omits NOISE.
# TODO(Phase 18): thresholds (K, core/noise cutoffs) come from config, not literals.


def explain_with_uncertainty(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """STUB. Will produce a per-node core/peripheral/noise distribution over K runs.

    Planned signature (subject to Phase-18 design):
        explain_with_uncertainty(model, window, target_node, *, k_runs=10,
                                 core=0.8, noise=0.2) -> explanation_with_tiers
    """
    raise NotImplementedError(
        "Phase 18 (Uncertainty-Aware Explanations) is not built yet — this is a stub."
    )


if __name__ == "__main__":
    print("STUB — Phase 18 (Uncertainty-Aware Explanations) is not implemented yet.")
