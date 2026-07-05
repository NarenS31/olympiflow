"""Layer 2 — GNNExplainer for XTraffic, a faithful pure-PyTorch reimplementation.

WHAT GNNExplainer does (Ying, Bourtsoulatze, Zitnik, Leskovec — NeurIPS 2019):
given a trained GNN and one prediction, it learns a small soft MASK over the
input (which nodes / which edges) such that keeping only the masked part still
reproduces the prediction, while the mask is pushed to be small and near-binary.
The nodes/edges that survive that pressure are, by construction, the ones the
model actually relied on. It optimises

    min_M   L(f(x), f(x ⊙ σ(M)))  +  λ_size·mean(σ(M))  +  λ_ent·H(σ(M))

where f is our frozen model, σ is the sigmoid, L is a prediction-preservation
loss, and H is the element-wise entropy that encourages σ(M) toward 0/1.

WHY we reimplement instead of importing torch_geometric.explain:
Our XTrafficSTGNN takes a *modality dict* and a *dense adjacency* — not the
(x, edge_index) signature torch_geometric's Explainer harness expects. Wrapping
it would mean converting our dense graph to edge lists and writing a shim model
anyway. Reimplementing the (small, well-documented) algorithm against our own
forward() is less code, fully transparent, and keeps our Phase-2 "no
torch-geometric dependency" decision intact. Cite as a faithful reimplementation.

We learn two masks:
  * a NODE feature mask m ∈ [0,1]^N applied to the input speed signal. Masking a
    z-scored value toward 0 pushes that node's input toward the dataset MEAN
    speed — the natural "remove this node's information" baseline.
  * an EDGE mask over the physical adjacency, so we can report which physical
    road connections carry the influence (planners think in roads, not tensors).

Python 3.9 compatible. Shapes annotated on every line they change.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...utils.io_utils import PKG_ROOT
from ..gnn.loaders import (build_modality_dict, load_adjacency, load_node_meta,
                           load_scaler)
from ..gnn.stgnn import XTrafficSTGNN
from .node_names import NodeNamer
from .schema import validate_explanation

SPEED_CHANNEL = 0  # channel 0 of X is z-scored speed; channel 1 is time-of-day


# ---------------------------------------------------------------------------
# Loading a trained model exactly as it was trained (mirrors evaluate.py)
# ---------------------------------------------------------------------------
def load_model(checkpoint: str, dataset: str, device: torch.device
               ) -> Tuple[XTrafficSTGNN, Dict, Dict]:
    """Rebuild the model from a checkpoint. Returns (model, config, scaler)."""
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(PKG_ROOT, checkpoint)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = cfg["model"]
    adj = load_adjacency(dataset)                 # [N, N]
    model = XTrafficSTGNN(
        num_nodes=adj.shape[0], physical_adj=adj, modality_dims=cfg["modalities"],
        residual_channels=m["residual_channels"], dilation_channels=m["dilation_channels"],
        skip_channels=m["skip_channels"], end_channels=m["end_channels"],
        n_blocks=m["n_blocks"], embed_dim=m["embed_dim"],
        gcn_order=m["gcn_order"], dropout=m["dropout"], out_len=m["out_len"],
    ).to(device)
    model.load_state_dict(ck["model_state"], strict=(adj.shape[0] == ck["n_nodes"]))
    model.eval()                                  # freeze BN / dropout — we never train f
    return model, cfg, load_scaler(dataset)


class GNNExplainer:
    """Learns node + edge masks explaining a single (node, horizon) prediction.

    The model is FROZEN: we only ever optimise the two mask tensors, never the
    network weights. That is the whole point — we are asking "what in the INPUT
    drove this fixed function's output", not retraining anything.
    """

    def __init__(self, model: XTrafficSTGNN, cfg: Dict, scaler: Dict[str, float],
                 device: torch.device, epochs: int = 200, lr: float = 0.01,
                 lambda_size: float = 0.15, lambda_ent: float = 0.05):
        self.model = model
        self.cfg = cfg
        self.scaler = scaler
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.lambda_size = lambda_size          # sparsity pressure on the masks
        self.lambda_ent = lambda_ent            # push masks toward 0/1
        self.n_nodes = model.num_nodes
        self.traffic_channels = cfg["traffic_channels"]
        self.modality_names = list(cfg["modalities"].keys())
        # A frozen copy of the physical adjacency (unmasked) to restore after each
        # explanation, since we temporarily overwrite the model's buffer below.
        self._phys_backup = model.physical_adj.detach().clone()
        # Which (i, j) pairs are real physical edges — the only edges we mask/report.
        self._edge_index = (self._phys_backup > 0)  # [N, N] bool

    def _forward_masked(self, X: torch.Tensor, node_mask_logit: torch.Tensor,
                        edge_mask_logit: torch.Tensor) -> torch.Tensor:
        """Run the frozen model with both masks applied. Returns preds [1,T_out,N]."""
        # --- node feature mask: scale the speed channel of every node's input. ---
        # Build a fresh tensor (no in-place edits — those break autograd through
        # the mask). We multiply the speed channel by the mask and leave the
        # time-of-day channel untouched, then re-stack along the channel dim.
        node_mask = torch.sigmoid(node_mask_logit)          # [N] in (0,1)
        speed = X[..., SPEED_CHANNEL] * node_mask[None, None, :]   # [1, T, N]
        other = X[..., SPEED_CHANNEL + 1:]                   # [1, T, N, C-1]
        Xm = torch.cat([speed.unsqueeze(-1), other], dim=-1)  # [1, T, N, C]

        # --- edge mask: multiply the physical adjacency, only on real edges. ---
        edge_mask = torch.sigmoid(edge_mask_logit)           # [N, N] in (0,1)
        masked_phys = self._phys_backup * edge_mask          # [N, N]
        # Temporarily install the masked adjacency so the model's internal
        # _semantic_support() uses it. Restored by explain_target's finally.
        self.model.physical_adj = masked_phys

        mods = build_modality_dict(Xm, self.traffic_channels, self.modality_names)
        return self.model(mods)                              # [1, T_out, N]

    def explain_target(self, X: torch.Tensor, target_node: int, horizon_step: int,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Explain one prediction.

        X            : [1, T, N, C] z-scored input window for a single sample.
        target_node  : node index whose forecast we explain.
        horizon_step : 1-based forecast step (3=15min, 6=30min, 12=60min).
        seed         : RNG seed for mask init (used for the confidence reruns).

        Returns (node_importance[N], edge_importance[N,N], pred_orig_z, cur_z).
        Importances are the learned sigmoid mask values in [0,1].
        """
        torch.manual_seed(seed)
        X = X.to(self.device)
        h = horizon_step - 1                                 # 0-based time index

        # Reference prediction with NO mask (all information present).
        with torch.no_grad():
            self.model.physical_adj = self._phys_backup
            mods = build_modality_dict(X, self.traffic_channels, self.modality_names)
            pred_orig = self.model(mods)[0, h, target_node]  # scalar (z-scored)
        pred_orig_val = float(pred_orig.item())

        # Learnable masks. Init near 0 -> sigmoid ~0.5 (neutral half-open) with a
        # tiny random jitter so repeated seeds explore slightly different optima
        # (this is what makes the confidence reruns meaningful).
        node_mask_logit = torch.randn(self.n_nodes, device=self.device) * 0.1
        node_mask_logit.requires_grad_(True)
        edge_mask_logit = (torch.randn(self.n_nodes, self.n_nodes,
                                       device=self.device) * 0.1)
        edge_mask_logit.requires_grad_(True)
        opt = torch.optim.Adam([node_mask_logit, edge_mask_logit], lr=self.lr)

        try:
            for _ in range(self.epochs):
                opt.zero_grad()
                preds = self._forward_masked(X, node_mask_logit, edge_mask_logit)
                pred_masked = preds[0, h, target_node]       # scalar
                # (1) preservation: masked prediction should match the original.
                loss = (pred_masked - pred_orig_val) ** 2
                # (2) sparsity: keep masks small so only what matters survives.
                node_m = torch.sigmoid(node_mask_logit)
                edge_m = torch.sigmoid(edge_mask_logit)[self._edge_index]
                loss = loss + self.lambda_size * node_m.mean()
                loss = loss + self.lambda_size * edge_m.mean()
                # (3) entropy: push mask values toward crisp 0/1 decisions.
                loss = loss + self.lambda_ent * _entropy(node_m).mean()
                loss = loss + self.lambda_ent * _entropy(edge_m).mean()
                loss.backward()
                opt.step()
        finally:
            # ALWAYS restore the real adjacency, even if optimisation errored,
            # so the model is never left in a masked state for the next call.
            self.model.physical_adj = self._phys_backup

        node_imp = torch.sigmoid(node_mask_logit).detach().cpu().numpy()   # [N]
        edge_imp = (torch.sigmoid(edge_mask_logit).detach().cpu().numpy()
                    * self._edge_index.cpu().numpy())                       # [N,N]
        cur_z = float(X[0, -1, target_node, SPEED_CHANNEL].item())          # last obs
        return node_imp, edge_imp, pred_orig_val, cur_z


def _entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Element-wise binary entropy H(p) = -p·log p - (1-p)·log(1-p), min at 0/1."""
    p = p.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


# ---------------------------------------------------------------------------
# Turning raw masks into the contract JSON (the object Phase 4 consumes)
# ---------------------------------------------------------------------------
def _z_to_mph(z: float, scaler: Dict[str, float]) -> float:
    return z * scaler["std"] + scaler["mean"]


def _shortest_path(adj_bool: np.ndarray, src: int, dst: int) -> List[int]:
    """BFS shortest path over the (undirected) physical graph, src -> dst.

    We report the physical road path along which congestion propagates from the
    most important source sensor to the target. If they are disconnected we
    return just [src, dst] and let the lag/importance speak instead."""
    if src == dst:
        return [src]
    n = adj_bool.shape[0]
    prev = {src: -1}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        if cur == dst:
            break
        for nxt in np.where(adj_bool[cur])[0]:
            if nxt not in prev:
                prev[int(nxt)] = cur
                queue.append(int(nxt))
    if dst not in prev:
        return [src, dst]
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return list(reversed(path))


def _propagation_lag(X: torch.Tensor, src: int, dst: int) -> float:
    """Estimate the lag (minutes) by which the source series LEADS the target.

    METHOD: cross-correlate the two z-scored speed series over the input window
    and take the lag that maximises correlation. A positive lag means the source
    node's speed changes first and the target follows `lag` steps later — the
    signature of congestion propagating from source to target. At 5-min
    resolution we multiply the best integer lag by 5. Documented because a
    reviewer will (rightly) ask how we got a causal-sounding number from
    correlation: we are estimating *temporal precedence within the window*, not
    claiming proof of causation — the explainer's importance is what claims
    relevance; the lag only times it.
    """
    s = X[0, :, src, SPEED_CHANNEL].cpu().numpy()   # [T]
    d = X[0, :, dst, SPEED_CHANNEL].cpu().numpy()    # [T]
    s = s - s.mean()
    d = d - d.mean()
    if np.allclose(s, 0) or np.allclose(d, 0):
        return 0.0
    T = len(s)
    best_lag, best_corr = 0, -np.inf
    for lag in range(0, T):                          # source leading by `lag` steps
        if lag == 0:
            corr = float(np.dot(s, d))
        else:
            corr = float(np.dot(s[:-lag], d[lag:]))
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return float(best_lag * 5)                        # steps -> minutes


class ExplanationBuilder:
    """High-level: run the explainer and emit a schema-valid explanation dict."""

    def __init__(self, checkpoint: str, dataset: str,
                 device: Optional[torch.device] = None,
                 top_k: int = 8, epochs: int = 200, confidence_runs: int = 5):
        self.device = device or torch.device("cpu")
        self.model, self.cfg, self.scaler = load_model(checkpoint, dataset, self.device)
        self.dataset = dataset
        self.checkpoint = os.path.basename(checkpoint)
        self.node_meta = load_node_meta(dataset)
        self.namer = NodeNamer(self.node_meta)
        self.adj_bool = (load_adjacency(dataset).numpy() > 0)   # [N,N]
        self.top_k = top_k
        self.confidence_runs = confidence_runs
        self.explainer = GNNExplainer(self.model, self.cfg, self.scaler,
                                      self.device, epochs=epochs)

    def _topk_nodes(self, node_imp: np.ndarray, target_node: int) -> List[int]:
        # Exclude the target itself; take the k most important other nodes.
        order = np.argsort(-node_imp)
        return [int(i) for i in order if i != target_node][: self.top_k]

    def explain_prediction(self, X: torch.Tensor, target_node: int,
                           horizon_step: int, timestamp: str) -> Dict:
        """Produce one explanation dict validated against schema.py.

        X : [1, T, N, C] single z-scored input window. horizon_step is 1-based.
        """
        node_imp, edge_imp, pred_z, cur_z = self.explainer.explain_target(
            X, target_node, horizon_step, seed=0)

        top_nodes_idx = self._topk_nodes(node_imp, target_node)
        speeds_z = X[0, -1, :, SPEED_CHANNEL].cpu().numpy()   # last-obs speed per node

        top_nodes = [{
            "node_id": nid,
            "node_name": self.namer.name(nid),
            "importance": round(float(node_imp[nid]), 4),
            "current_speed_mph": round(_z_to_mph(float(speeds_z[nid]), self.scaler), 2),
        } for nid in top_nodes_idx]

        # Top edges: rank real physical edges by learned edge importance, keep the
        # k strongest that touch either the target or a top node (keeps them
        # relevant to the explanation rather than globally-strong-but-unrelated).
        relevant = set(top_nodes_idx) | {target_node}
        edges: List[Tuple[int, int, float]] = []
        ii, jj = np.where(self.adj_bool)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i in relevant or j in relevant:
                edges.append((i, j, float(edge_imp[i, j])))
        edges.sort(key=lambda e: -e[2])
        top_edges = [{"from_id": i, "to_id": j, "importance": round(imp, 4)}
                     for i, j, imp in edges[: self.top_k]]

        # Propagation path + lag from the single most important source node.
        src = top_nodes_idx[0] if top_nodes_idx else target_node
        path = _shortest_path(self.adj_bool, src, target_node)
        lag = _propagation_lag(X, src, target_node)

        confidence = self._confidence(X, target_node, horizon_step, top_nodes_idx)

        exp = {
            "meta": {"city": self.dataset, "timestamp": timestamp,
                     "model_checkpoint": self.checkpoint},
            "prediction": {
                "node_id": int(target_node),
                "node_name": self.namer.name(target_node),
                "predicted_speed_mph": round(_z_to_mph(pred_z, self.scaler), 2),
                "horizon_minutes": int(horizon_step * 5),
                "current_speed_mph": round(_z_to_mph(cur_z, self.scaler), 2),
            },
            "top_nodes": top_nodes,
            "top_edges": top_edges,
            "propagation_path": [int(n) for n in path],
            "propagation_lag_minutes": round(float(lag), 1),
            "explanation_confidence": round(float(confidence), 4),
        }
        problems = validate_explanation(exp)
        if problems:  # should never fire; loud if the contract ever drifts
            print("[explainer] WARNING schema problems:", problems)
        return exp

    def _confidence(self, X: torch.Tensor, target_node: int, horizon_step: int,
                    base_topk: List[int]) -> float:
        """Rerun the explainer K times with different seeds; confidence = mean
        Jaccard similarity of the top-k node SET vs the base run. High = the
        explanation is stable across random inits (we trust it more)."""
        base = set(base_topk)
        if not base:
            return 0.0
        sims = []
        for k in range(1, self.confidence_runs + 1):
            node_imp, _, _, _ = self.explainer.explain_target(
                X, target_node, horizon_step, seed=k)
            other = set(self._topk_nodes(node_imp, target_node))
            union = base | other
            sims.append(len(base & other) / len(union) if union else 0.0)
        return float(np.mean(sims)) if sims else 0.0
