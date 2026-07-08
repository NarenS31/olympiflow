"""Phase 12 — a SHAP baseline explainer for XTraffic.

WHY THIS EXISTS
---------------
A reviewer will ask: "Your Layer 2 is GNNExplainer. Why not just use SHAP?"
The honest answer is not an assertion, it is an EXPERIMENT: build a SHAP explainer
that plugs into the exact same pipeline, feed its output to the same LLM advisor,
and measure whether the LLM's reasoning stays as faithful as it does with
GNNExplainer. This file is the SHAP explainer half; evaluation/shap_comparison.py
runs the A/B/D study and builds the comparison table.

WHAT SHAP DOES HERE (and how it maps onto our model)
----------------------------------------------------
SHAP (Lundberg & Lee, NeurIPS 2017) attributes a model's output to its input
features by averaging each feature's marginal contribution over feature
coalitions (the Shapley value from cooperative game theory). KernelExplainer is
the model-agnostic variant: it perturbs the input, watches the output, and fits a
weighted linear model whose coefficients ARE the Shapley values.

Our "features" are the N sensor NODES -- the same unit GNNExplainer scores -- so
the two explainers are directly comparable. We define a coalition as a binary
mask m in {0,1}^N:
  * m[i] = 1  -> node i keeps its real (z-scored) speed series.
  * m[i] = 0  -> node i's speed is pushed to 0 in z-space == the dataset MEAN
                 speed. This is EXACTLY GNNExplainer's "remove this node"
                 operation (see explain.py), so both explainers share one notion
                 of "absence" and the comparison is fair.
The background (all-masked) row is the baseline f(mean-speed everywhere); the
instance (all-present) is the real prediction. KernelExplainer decomposes the gap
between them across the N nodes. |SHAP value| ranks node importance; the product
of two endpoint SHAP values approximates an edge's importance (an interaction
proxy -- SHAP scores nodes, not edges, so we build edge scores from node scores).

THE OUTPUT IS THE SAME CONTRACT
-------------------------------
We reuse ExplanationBuilder wholesale: it already turns a (node_importance,
edge_importance, prediction, current_speed) tuple into the schema.py JSON the LLM
depends on (naming, top-k, propagation path/lag, confidence, validation). We only
swap the *core attribution step* -- ExplanationBuilder calls
`self.explainer.explain_target(...)`, so a SHAP object with that same method
signature drops straight in. Result: a byte-compatible explanation JSON, produced
by SHAP instead of GNNExplainer.

SPEED NOTE (documented on purpose)
----------------------------------
KernelExplainer is slow: it re-runs the model on many perturbations per
prediction. With N=207 nodes an exact solution needs O(2^N) coalitions, so we
SAMPLE. We default to nsamples=50 (the assignment's choice) for a focused
comparison -- this makes the SHAP values an APPROXIMATION (fewer samples than
features, so a sparse L1 solve selects the most influential nodes). More samples
= more faithful SHAP but linearly more model calls. We batch all coalition rows
through the model in ONE forward pass, so a single explanation is seconds, not
minutes, on METR-LA -- but the study is still deliberately capped at 30 scenarios.

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from ..gnn.loaders import build_modality_dict
from .explain import SPEED_CHANNEL, ExplanationBuilder

# shap is a Phase-12 dependency (pinned in requirements.txt). Import lazily-ish
# with a clear message so a machine without it gets a fix, not a stack trace.
try:
    import shap  # type: ignore
    _HAVE_SHAP = True
except Exception:  # pragma: no cover
    _HAVE_SHAP = False


class ShapExplainer:
    """SHAP (KernelExplainer) attribution over sensor nodes, exposing the SAME
    `explain_target` interface as GNNExplainer so it drops into ExplanationBuilder.

    The model is FROZEN and used only for forward passes -- SHAP is
    perturbation-based, so (unlike GNNExplainer) we never need gradients.
    """

    # l1_reg MUST be of the "num_features(k)" family at low nsamples. We sample far
    # fewer coalitions than we have nodes (50 << 207), so the SHAP linear solve is
    # underdetermined and needs a sparse regulariser to pick the most influential
    # nodes. The information-criterion options ("aic"/"bic") are NOT usable here:
    # LassoLarsIC cannot estimate the noise variance when samples < features and
    # raises. num_features(k) selects k nodes via a LARS path, which is exactly the
    # documented approximation that nsamples=50 buys us.
    def __init__(self, model, cfg, device: torch.device, nsamples: int = 50,
                 l1_reg: str = "num_features(20)"):
        if not _HAVE_SHAP:
            raise ImportError(
                "The 'shap' package is required for the SHAP baseline. Install it "
                "with `pip install shap==0.44.1` (see xtraffic/requirements.txt).")
        self.model = model
        self.cfg = cfg
        self.device = device
        self.nsamples = nsamples            # coalitions sampled per prediction
        self.l1_reg = l1_reg                # sparse solve when nsamples < N
        self.n_nodes = model.num_nodes
        self.traffic_channels = cfg["traffic_channels"]
        self.modality_names = list(cfg["modalities"].keys())
        # Which (i, j) are real physical edges -- the only edges we score/report,
        # exactly like GNNExplainer (planners think in roads, not dense tensors).
        self._edge_index = (model.physical_adj.detach().cpu().numpy() > 0)  # [N,N] bool

    @torch.no_grad()
    def _masked_pred_batch(self, X: torch.Tensor, target_node: int, h: int,
                           Z: np.ndarray) -> np.ndarray:
        """Run the frozen model on a BATCH of node-mask coalitions.

        X : [1, T, N, C] z-scored input for one window.
        Z : [n, N] coalition matrix in {0,1} from SHAP (1=keep node, 0=mean it out).
        Returns the target prediction for each coalition: [n].
        """
        n = Z.shape[0]
        mask = torch.as_tensor(Z, dtype=torch.float32, device=self.device)  # [n, N]
        base_speed = X[0, :, :, SPEED_CHANNEL]                    # [T, N] real speeds
        # Broadcast the mask over time: masked node -> speed 0 (z-space mean). [n,T,N]
        speed = base_speed.unsqueeze(0) * mask.unsqueeze(1)
        # Keep every non-speed channel (e.g. time-of-day) untouched, per coalition.
        other = X[0, :, :, SPEED_CHANNEL + 1:].unsqueeze(0).expand(n, -1, -1, -1)  # [n,T,N,C-1]
        Xm = torch.cat([speed.unsqueeze(-1), other], dim=-1)     # [n, T, N, C]
        mods = build_modality_dict(Xm, self.traffic_channels, self.modality_names)
        preds = self.model(mods)                                 # [n, T_out, N]
        return preds[:, h, target_node].detach().cpu().numpy()   # [n]

    def explain_target(self, X: torch.Tensor, target_node: int, horizon_step: int,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Explain one (node, horizon) prediction with SHAP.

        Signature matches GNNExplainer.explain_target so ExplanationBuilder can use
        either. `seed` re-seeds the coalition sampler, which is what makes the
        confidence reruns (stability under re-explanation) meaningful.

        Returns (node_importance[N], edge_importance[N,N], pred_orig_z, cur_z).
        node_importance is |SHAP|, normalised to [0,1] so its scale matches
        GNNExplainer's [0,1] mask values (keeps the LLM prompt comparable).
        """
        X = X.to(self.device)
        h = horizon_step - 1                                     # 0-based time index
        N = self.n_nodes

        # Reference prediction with ALL nodes present (the SHAP "instance").
        pred_orig_val = float(self._masked_pred_batch(X, target_node, h, np.ones((1, N)))[0])
        cur_z = float(X[0, -1, target_node, SPEED_CHANNEL].item())  # last observed speed

        # The model-as-a-function SHAP will probe: coalition matrix -> predictions.
        f = lambda Z: self._masked_pred_batch(X, target_node, h, np.asarray(Z))

        background = np.zeros((1, N))                            # all nodes meaned out
        instance = np.ones((1, N))                              # all nodes present
        # KernelExplainer samples coalitions with numpy's RNG; seed it so a given
        # seed is reproducible AND different seeds explore differently (stability).
        np.random.seed(seed)
        explainer = shap.KernelExplainer(f, background)
        # The LARS sparse solve emits "degenerate active set" convergence warnings
        # when nsamples < N -- expected and harmless here (it's the low-sample
        # approximation, not a bug), so we silence just this call to keep batch
        # logs readable. The sparsity itself is real and reported in the writeup.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(
                instance, nsamples=self.nsamples, l1_reg=self.l1_reg, silent=True)
        sv = np.asarray(shap_values).reshape(-1)[:N]            # [N] Shapley values

        node_imp = np.abs(sv)                                    # |SHAP| = importance
        mx = float(node_imp.max())
        node_imp = node_imp / mx if mx > 0 else node_imp        # -> [0,1], like GNNExpl.
        # Edge importance = product of endpoint node importances, on real edges only.
        # SHAP scores nodes; this is the standard interaction proxy for turning node
        # attributions into edge attributions (both endpoints matter -> product).
        edge_imp = np.outer(node_imp, node_imp) * self._edge_index   # [N,N]
        return node_imp, edge_imp, pred_orig_val, cur_z


def build_shap_explanation_builder(checkpoint: str, dataset: str,
                                   device: Optional[torch.device] = None,
                                   top_k: int = 8, nsamples: int = 50,
                                   confidence_runs: int = 2,
                                   l1_reg: str = "num_features(40)"
                                   ) -> ExplanationBuilder:
    """Return an ExplanationBuilder whose attribution engine is SHAP, not
    GNNExplainer. Callers then use `.explain_prediction(...)` exactly as before and
    get a schema-valid explanation JSON produced by SHAP.

    We construct a normal ExplanationBuilder (which loads the model/config/scaler,
    namer, and adjacency once) and then SWAP its `.explainer` for a ShapExplainer.
    This reuses every downstream step -- top-k, naming, propagation path/lag,
    confidence reruns, schema validation -- so the ONLY thing that changes between
    the two explainers is the node/edge importances themselves. That is precisely
    the controlled comparison the reviewer question needs.

    confidence_runs defaults to 2 (not GNNExplainer's 5) because each rerun is a
    full extra SHAP solve -- we keep the schema's confidence field meaningful
    without paying 5x the SHAP cost.
    """
    device = device or torch.device("cpu")
    builder = ExplanationBuilder(checkpoint, dataset, device=device, top_k=top_k,
                                 epochs=1, confidence_runs=confidence_runs)
    builder.explainer = ShapExplainer(builder.model, builder.cfg, device,
                                      nsamples=nsamples, l1_reg=l1_reg)
    return builder


# ---------------------------------------------------------------------------
# Tiny self-check: build a couple of SHAP explanations and validate the schema.
# Run: python -m xtraffic.models.explainer.shap_explainer --n 2
# ---------------------------------------------------------------------------
def _demo() -> None:
    import argparse

    from ..gnn.loaders import load_scaler
    from .scenarios import load_window
    from .schema import validate_explanation

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/gnn/checkpoints/metr_la_best.pt")
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--n", type=int, default=2, help="how many test windows to explain")
    ap.add_argument("--nsamples", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cpu")
    builder = build_shap_explanation_builder(
        args.checkpoint, args.dataset, device=device, nsamples=args.nsamples)
    scaler = load_scaler(args.dataset)
    for idx in range(args.n):
        X = load_window(args.dataset, idx)                      # [1, T, N, C]
        # slowest valid sensor in the window = the target worth explaining (same
        # REAL-mph validity rule as Phase 3/5).
        spd = X[0, :, :, SPEED_CHANNEL].numpy() * scaler["std"] + scaler["mean"]  # [T,N]
        valid = (spd > 1.0).any(0)                              # [N]
        node_mean = np.where(valid, spd.mean(0), np.inf)        # [N]
        target = int(node_mean.argmin())
        exp = builder.explain_prediction(X, target_node=target, horizon_step=6,
                                         timestamp="demo#{}".format(idx))
        problems = validate_explanation(exp)
        print("\n=== SHAP explanation for test window {} (target node {}) ===".format(idx, target))
        print("  predicted {} mph; top nodes: {}".format(
            exp["prediction"]["predicted_speed_mph"],
            [n["node_id"] for n in exp["top_nodes"]]))
        print("  schema valid:", not problems, problems or "")


if __name__ == "__main__":
    _demo()
