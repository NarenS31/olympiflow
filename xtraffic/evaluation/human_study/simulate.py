"""Phase 8 — simulation-based ground truth for the decision-quality study.

For each incident scenario we must know which intervention is BEST, so we can
score whether a human evaluator chose it. We don't have a field experiment, so we
SIMULATE: perturb the trained GNN's own input window to mimic each candidate
intervention, run the model forward, and read the 30-min-ahead network delay. The
intervention that yields the lowest predicted delay is the ground-truth-optimal
one.

HONEST LIMITATION (must be stated in the paper, see configs/human_study.yaml):
this is a *model-in-the-loop* proxy — "ground truth" is what the trained model
believes helps, not observed reality. It is internally consistent (the same model
the RAW/XAI/XTRAFFIC conditions are built from) and fully reproducible, but it
inherits the model's biases. We do NOT claim it is the true causal effect.

How an intervention is simulated
--------------------------------
Every action is modelled as a throughput gain: we ADD `uplift_mph` to the recent
speed (the last `apply_steps` of the 12-step input window) on a set of affected
nodes, then let the GNN propagate that forward to the 30-min forecast. The node
set encodes what the action physically touches:

  no_action        : nothing changes (baseline delay).
  signal_retiming  : the target segment + its 1-hop physical neighbours (retiming
                     a corridor's signals speeds the immediate area).
  ramp_metering    : the UPSTREAM 1-hop neighbours that are currently slower than
                     the target (metering the on-ramps feeding the jam).
  reroute          : the target segment only (diverting demand off it raises its
                     own speed without directly helping neighbours).
  transit_surge    : a broad 2-hop neighbourhood at HALF uplift (a mode shift
                     removes a little demand across a wide area).

Network delay
-------------
delay(pred) = sum over VALID nodes of max(0, free_flow_mph - pred_speed_mph)
at the scored horizon. Lower = better (less city-wide speed deficit). We sum over
nodes that had a genuine reading in the input window (missing sensors, ~0 mph,
would otherwise dominate the sum with fake delay).

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...models.gnn.loaders import build_modality_dict
from ...models.explainer.explain import load_model
from ...models.explainer.scenarios import SPEED_CHANNEL


class InterventionSimulator:
    """Loads the trained GNN once, then scores interventions on any window.

    Reuses the exact model + scaler the explainer/advisor use, so the study's
    ground truth is consistent with the predictions the evaluators are shown.
    """

    def __init__(self, checkpoint: str, dataset: str,
                 sim_cfg: Dict, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        # load_model returns (model, cfg, scaler); model is frozen (eval) here.
        self.model, self.cfg, self.scaler = load_model(checkpoint, dataset, self.device)
        self.model.eval()
        self.dataset = dataset
        self.traffic_channels = self.cfg["traffic_channels"]
        self.modality_names = list(self.cfg["modalities"].keys())

        # Physical adjacency (bool) for the 1-/2-hop neighbourhood rules. [N, N]
        self.adj_bool = (self.model.physical_adj.detach().cpu().numpy() > 0)

        # Simulation knobs (all from configs/human_study.yaml — no magic numbers).
        self.horizon_step = int(sim_cfg["horizon_step"])          # 1-based
        self.free_flow = float(sim_cfg["free_flow_mph"])
        self.uplift_mph = float(sim_cfg["uplift_mph"])
        self.apply_steps = int(sim_cfg["apply_steps"])
        self.interventions = list(sim_cfg["interventions"])

    # -- prediction ---------------------------------------------------------
    def _predict_speed_mph(self, X: torch.Tensor) -> np.ndarray:
        """Forward the frozen model on one window; return 30-min speeds per node.

        X : [1, T, N, C] z-scored input window.
        returns : [N] predicted speed in REAL mph at the scored horizon.
        """
        h = self.horizon_step - 1                                 # 0-based time idx
        with torch.no_grad():
            mods = build_modality_dict(X, self.traffic_channels, self.modality_names)
            pred_z = self.model(mods)[0, h, :].cpu().numpy()      # [N] z-scored
        return pred_z * self.scaler["std"] + self.scaler["mean"]  # [N] mph

    def _network_delay(self, pred_mph: np.ndarray, valid: np.ndarray) -> float:
        """Sum of speed deficit below free-flow over valid nodes (lower=better)."""
        deficit = np.clip(self.free_flow - pred_mph, 0.0, None)   # [N]
        return float(deficit[valid].sum())

    # -- intervention node sets --------------------------------------------
    def _neighbours(self, node: int) -> List[int]:
        """1-hop physical neighbours of `node` (excludes self)."""
        row = self.adj_bool[node].copy()
        row[node] = False
        return [int(i) for i in np.where(row)[0]]

    def _two_hop(self, node: int) -> List[int]:
        """Nodes within 2 physical hops of `node` (excludes self)."""
        one = set(self._neighbours(node))
        two = set(one)
        for n in one:
            two.update(self._neighbours(n))
        two.discard(node)
        return sorted(two)

    def _affected_nodes(self, name: str, target: int,
                        last_speed_mph: np.ndarray) -> Tuple[List[int], float]:
        """Return (node_ids to uplift, uplift magnitude in mph) for an action."""
        if name == "no_action":
            return [], 0.0
        if name == "signal_retiming":
            return [target] + self._neighbours(target), self.uplift_mph
        if name == "ramp_metering":
            # Upstream = 1-hop neighbours that are currently SLOWER than the target
            # (they are the congested feeders we meter).
            nbrs = self._neighbours(target)
            up = [n for n in nbrs if last_speed_mph[n] < last_speed_mph[target]]
            return up, self.uplift_mph
        if name == "reroute":
            return [target], self.uplift_mph
        if name == "transit_surge":
            return self._two_hop(target), self.uplift_mph * 0.5   # broad, gentle
        raise ValueError("unknown intervention: {}".format(name))

    def _apply(self, X: torch.Tensor, nodes: List[int], uplift_mph: float
               ) -> torch.Tensor:
        """Return a copy of X with +uplift_mph added to the last `apply_steps`
        input timesteps of the speed channel on `nodes` (all in z-space)."""
        Xm = X.clone()                                            # [1, T, N, C]
        if not nodes or uplift_mph == 0.0:
            return Xm
        uplift_z = uplift_mph / self.scaler["std"]                # mph -> z-units
        t0 = X.shape[1] - self.apply_steps                        # first touched step
        idx = torch.tensor(nodes, dtype=torch.long)
        Xm[0, t0:, idx, SPEED_CHANNEL] += uplift_z
        return Xm

    # -- public API ---------------------------------------------------------
    def score_interventions(self, X: torch.Tensor, target_node: int
                            ) -> Dict[str, object]:
        """Score every configured intervention on window X for `target_node`.

        Returns a dict with per-intervention predicted delay + predicted target
        speed, plus the ground-truth (lowest-delay) intervention. Deterministic.
        """
        # Nodes that had a genuine reading in the window (mph > 1 at any step).
        speed_mph_window = (X[0, :, :, SPEED_CHANNEL].cpu().numpy()
                            * self.scaler["std"] + self.scaler["mean"])   # [T, N]
        valid = (speed_mph_window > 1.0).any(0)                   # [N] bool
        last_speed_mph = speed_mph_window[-1]                     # [N] most-recent obs

        results: List[Dict[str, object]] = []
        for name in self.interventions:
            nodes, uplift = self._affected_nodes(name, target_node, last_speed_mph)
            Xm = self._apply(X, nodes, uplift)
            pred_mph = self._predict_speed_mph(Xm)               # [N]
            results.append({
                "intervention": name,
                "n_nodes_affected": len(nodes),
                "network_delay": round(self._network_delay(pred_mph, valid), 3),
                "target_speed_mph": round(float(pred_mph[target_node]), 2),
            })

        # Ground truth = lowest predicted network delay. Ties broken by earliest in
        # the configured order (deterministic); if that is "no_action" the scenario
        # is degenerate (no action helps) and we flag it for optional exclusion.
        best = min(results, key=lambda r: (r["network_delay"],
                                           self.interventions.index(r["intervention"])))
        return {
            "target_node": int(target_node),
            "candidates": results,
            "ground_truth_intervention": best["intervention"],
            "no_action_is_best": best["intervention"] == "no_action",
        }
