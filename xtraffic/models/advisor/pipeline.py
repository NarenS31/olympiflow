"""The whole XTraffic system in one function (Phase 4).

    advise(city, node_id, ...) -> {prediction, explanation, advisory}

This chains all three layers:
    Layer 1  ST-GNN prediction            (models/gnn/stgnn.py)
    Layer 2  GNNExplainer explanation      (models/explainer/explain.py)
    Layer 3  LLM advisory translation      (models/advisor/advisor.py)

"This function is the paper" — everything downstream (faithfulness study,
ablations, human eval) calls into this chain.

Two ways in:
  * advise_live(...)          — run the GNN + explainer fresh on a test window,
                                then the advisor. This is the true end-to-end
                                path (needs the trained checkpoint + Ollama).
  * advise_from_explanation() — feed an already-produced explanation JSON (e.g.
                                the 6 committed Phase-3 scenarios) straight into
                                Layer 3. Same Layer-3 output, no GNN reload — the
                                fast path for the demo and for Phase 5 batches.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..explainer.explain import ExplanationBuilder
from ..explainer.scenarios import load_window
from .advisor import Advisor


def advise_from_explanation(exp: Dict[str, Any],
                            advisor: Advisor) -> Dict[str, Any]:
    """Layer 3 only: given a Phase-3 explanation dict + a ready Advisor, return
    the combined {prediction, explanation, advisory} result."""
    result = advisor.advise(exp)
    return {
        "prediction": exp["prediction"],
        "explanation": exp,
        "advisory": result["advisory"],
        "context_used": result["context_used"],
        "model": result["model"],
        "raw_responses": result["raw_responses"],
    }


class Pipeline:
    """Holds the (heavy) GNN+explainer and the (light) advisor so a batch run
    loads the checkpoint once. Lazily builds the explainer — the fast path
    (saved explanations) never pays for it."""

    def __init__(self, city: str, checkpoint: str, dataset: Optional[str] = None):
        self.city = city
        self.dataset = dataset if dataset is not None else city
        self.checkpoint = checkpoint
        self.advisor = Advisor(city)
        self._builder: Optional[ExplanationBuilder] = None

    @property
    def builder(self) -> ExplanationBuilder:
        if self._builder is None:
            # CPU: the explainer is tiny and this keeps it deterministic (matches
            # models/explainer/generate.py).
            self._builder = ExplanationBuilder(
                self.checkpoint, self.dataset, device=torch.device("cpu"))
        return self._builder

    def advise_live(self, node_id: int, sample_index: int,
                    horizon_step: int = 6, timestamp: str = "live") -> Dict[str, Any]:
        """True end-to-end: run GNN + explainer on test window `sample_index`
        for `node_id`, then the advisor. horizon_step is 1-based (6 == 30 min at
        5-min resolution)."""
        X = load_window(self.dataset, sample_index)          # [1, T, N, C]
        exp = self.builder.explain_prediction(
            X, target_node=node_id, horizon_step=horizon_step, timestamp=timestamp)
        exp["meta"]["city"] = self.city
        return advise_from_explanation(exp, self.advisor)

    def advise_from_explanation(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Fast path: skip the GNN, feed an existing explanation to Layer 3."""
        return advise_from_explanation(exp, self.advisor)


def advise(city: str, node_id: int, sample_index: int,
           checkpoint: str, horizon_step: int = 6,
           dataset: Optional[str] = None) -> Dict[str, Any]:
    """One-shot convenience wrapper: build a Pipeline and run the full chain for
    a single prediction. For many predictions, build a Pipeline once and reuse
    it (avoids reloading the checkpoint each call)."""
    pipe = Pipeline(city, checkpoint, dataset=dataset)
    return pipe.advise_live(node_id, sample_index, horizon_step=horizon_step)
