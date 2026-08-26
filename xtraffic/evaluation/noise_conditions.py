"""PART 1 — the two noise conditions: `A_rand` and `A_mismatch`.

WHAT THESE ARE FOR
------------------
Condition A shows the LLM a mathematical explanation and the faithfulness metric
asks whether the LLM's prose cites what that explanation said. The metric never
asks whether the explanation was *right*, or even whether it contained
information. These two conditions remove the information and keep everything
else, so we can see what the metric is actually measuring.

  A_rand      : the committed artifact with its node importances replaced by a
                seeded uniform draw. Same target, same predicted speed, same
                prompt skeleton — evidence that means nothing.
  A_mismatch  : the committed artifact's prediction block, with the evidence
                (top_nodes / top_edges / propagation / lag / confidence) taken
                from the real committed explanation of a DIFFERENT, seeded-random
                target. Real structure, attached to the wrong place.

HOW `A_rand` IS BUILT — through the real builder, not by hand
------------------------------------------------------------
`ExplanationBuilder.explain_prediction` is what produced every committed
explanation: it picks the top-k, selects the top edges that touch them, runs BFS
for the propagation path, cross-correlates for the lag, reruns the explainer for
the confidence, and validates against schema.py. Hand-assembling a dict that
merely looks like its output would let the two drift apart silently.

So we do not hand-assemble. We temporarily swap the ONE function that produces
the masks — `GNNExplainer.explain_target` — for a seeded random draw, and let
`explain_prediction` run untouched on top of it. Every derived field is then
derived the way it always is, from noise. That is what makes the artifact
internally consistent rather than merely random-looking: its propagation path
really is the BFS path to its own top source, its top edges really do touch its
own top nodes, and its confidence really is the stability of its own mask across
reruns (which, for noise, is near zero — an honest property of the artifact, and
one we report rather than hide, because the LLM sees that number in the prompt).

WHY THE PREDICTION BLOCK IS TAKEN FROM THE COMMITTED ARTIFACT
------------------------------------------------------------
The committed explanations were produced on the EPOCH-34 checkpoint (their JSONs
name `metr_la_best.pt`, but that path now holds epoch 54, which does not
reproduce them — see run_sparsity_sweep.CKPT_COMMITTED). Recomputing the
prediction here would silently swap the model underneath the comparison. Reusing
the committed prediction block keeps `A_rand`'s prompt identical to `A`'s
everywhere except the evidence, which is the whole point of the condition.

Python 3.9 compatible.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

SPEED_CHANNEL = 0

# Fields that carry the EVIDENCE. `A_mismatch` swaps exactly these and nothing
# else; `meta` and `prediction` stay with the scenario.
EVIDENCE_FIELDS = ("top_nodes", "top_edges", "propagation_path",
                   "propagation_lag_minutes", "explanation_confidence")


def _mph_to_z(mph: float, scaler: Dict[str, float]) -> float:
    return (float(mph) - scaler["mean"]) / scaler["std"]


class _RandomMaskPatch:
    """Context manager swapping `explain_target` for a seeded uniform draw.

    The replacement keeps the real signature and returns the committed
    prediction's (pred_z, cur_z) so `explain_prediction` fills the prediction
    block with the numbers condition A actually showed the LLM. Each call
    advances the RNG, so the confidence reruns (seeds 1..K) get fresh draws and
    the resulting confidence is the genuine stability of a random mask.
    """

    def __init__(self, builder: Any, rng: np.random.RandomState,
                 pred_z: float, cur_z: float):
        self.explainer = builder.explainer
        self.rng = rng
        self.pred_z = pred_z
        self.cur_z = cur_z
        self._orig: Optional[Callable] = None
        self.n_calls = 0

    def __enter__(self) -> "_RandomMaskPatch":
        n = self.explainer.n_nodes
        edge_index = self.explainer._edge_index.cpu().numpy()

        def _fake(X: torch.Tensor, target_node: int, horizon_step: int,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float, float]:
            self.n_calls += 1
            node_imp = self.rng.uniform(0.0, 1.0, size=n).astype(np.float32)
            edge_imp = (self.rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
                        * edge_index)
            return node_imp, edge_imp, self.pred_z, self.cur_z

        self._orig = self.explainer.explain_target
        self.explainer.explain_target = _fake        # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        # ALWAYS restore, even on error: a builder left patched would quietly
        # produce random explanations for every later caller in the process.
        if self._orig is not None:
            self.explainer.explain_target = self._orig   # type: ignore[assignment]


def build_a_rand(builder: Any, committed: Dict[str, Any], X: torch.Tensor,
                 seed: int) -> Dict[str, Any]:
    """`A_rand` for one scenario: the committed artifact, evidence replaced by noise.

    builder   : a live ExplanationBuilder (any checkpoint — the model is never
                consulted, since the patch supplies the masks AND the prediction).
    committed : the committed condition-A explanation dict for this scenario.
    X         : [1, T, N, C] the scenario's own window (used for the lag and for
                every node's current speed, both of which are properties of the
                DATA and so are kept real).
    seed      : per-scenario seed; recorded in the artifact.
    """
    pred = committed["prediction"]
    rng = np.random.RandomState(seed)
    pred_z = _mph_to_z(pred["predicted_speed_mph"], builder.scaler)
    cur_z = _mph_to_z(pred["current_speed_mph"], builder.scaler)

    with _RandomMaskPatch(builder, rng, pred_z, cur_z) as patch:
        exp = builder.explain_prediction(
            X, target_node=int(pred["node_id"]), horizon_step=6,
            timestamp=committed["meta"]["timestamp"])
        n_calls = patch.n_calls

    exp["meta"] = copy.deepcopy(committed["meta"])
    exp["_noise"] = {
        "condition": "A_rand",
        "node_imp_distribution": "uniform(0,1), numpy RandomState",
        "seed": int(seed),
        "explainer_calls_drawn": int(n_calls),
        "prediction_block_source": "committed condition-A artifact (epoch-34)",
    }
    return exp


def build_a_mismatch(committed: Dict[str, Any], donor: Dict[str, Any],
                     seed: int) -> Dict[str, Any]:
    """`A_mismatch` for one scenario: this target's prediction, another's evidence.

    No solve and no model: both halves are already-committed real artifacts. The
    result is internally consistent in the only sense that matters here — every
    evidence field comes from ONE real explanation, so the importances, the
    edges, the path and the lag all agree with each other. They just do not
    belong to the target named in the prediction block.
    """
    exp = copy.deepcopy(committed)
    for field in EVIDENCE_FIELDS:
        exp[field] = copy.deepcopy(donor[field])
    exp["_noise"] = {
        "condition": "A_mismatch",
        "seed": int(seed),
        "donor_target_node": int(donor["prediction"]["node_id"]),
        "donor_timestamp": donor["meta"]["timestamp"],
        "prediction_block_source": "committed condition-A artifact (epoch-34)",
        "evidence_source": "committed condition-A artifact of the donor target",
    }
    return exp


def seeded_donor_permutation(keys: List[Any], seed: int) -> Dict[Any, Any]:
    """A seeded DERANGEMENT of `keys`: every scenario gets a donor, and no
    scenario ever gets itself.

    A plain shuffle would leave a few fixed points, and a fixed point is silently
    condition A — which would dilute the mismatch condition with a handful of
    ordinary A runs and pull its mean toward A for the wrong reason.
    """
    rng = np.random.RandomState(seed)
    n = len(keys)
    if n < 2:
        raise ValueError("need at least 2 scenarios to derange")
    while True:
        perm = rng.permutation(n)
        if all(perm[i] != i for i in range(n)):
            return {keys[i]: keys[int(perm[i])] for i in range(n)}
