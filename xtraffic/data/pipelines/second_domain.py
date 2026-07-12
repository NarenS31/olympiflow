"""Phase 19 — Cross-Modal Grounding Transfer (STUB — not built yet).

*** DISCUSS WITH THE PROFESSOR BEFORE BUILDING. ***
This phase needs faculty guidance on which second domain to use and how far to
scope it. Start with DATA EXPLORATION ONLY — do NOT train anything until the
domain is agreed.

WHY THIS EXISTS
---------------
The whole paper's cleanest claim is Contribution #2: the mathematical GNN
explanation eliminates LLM hallucination. So far that's shown on TRAFFIC (METR-LA,
PEMS-BAY, Chicago). This phase asks the strongest possible question: does the SAME
result hold in a COMPLETELY DIFFERENT domain, with NO architecture changes? If yes,
the claim generalises from "works on traffic" to:

    "mathematical GNN-explanation grounding is a DOMAIN-AGNOSTIC mechanism for
     eliminating LLM hallucination."

THE PLAN (staged — exploration first)
-------------------------------------
  STEP 1 (this file, first): EXPLORATION ONLY. Identify which second-domain graph
  dataset is most accessible and defensible. Candidates:
      * IEEE 14-bus power grid   (small, clean, well-understood graph)
      * MIMIC-III patient flow   (access-controlled — check credentialing first)
      * SupplyGraph supply chain (supply-chain graph benchmark)
  Deliverable: a short DIFFERENCES-style note per candidate (how to access it, its
  licence, how the graph is defined, node/edge semantics, and whether it fits our
  X[.,T,N,C] / Y[.,T,N] tensor contract) so the professor can choose with full
  information.

  STEP 2 (only after sign-off): window the chosen dataset into the SAME tensor
  contract, train the SAME XTrafficSTGNN (no architecture edits — that's the point),
  run the Phase-5 A/B conditions, and report the HALLUCINATION GAP next to the
  METR-LA numbers.

NOTES
-----
  * Reuse the entire downstream stack unchanged (explainer -> advisor ->
    faithfulness). The cross-CITY pipelines already prove the tensor contract is
    portable; this pushes it cross-DOMAIN.
  * A domain with no natural knowledge base -> empty city context. Phase 5 already
    showed context is orthogonal to faithfulness, so A/B is unaffected and we
    fabricate NO domain facts.

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Candidate second-domain datasets to evaluate in STEP 1 (exploration).
CANDIDATE_DOMAINS = ("ieee_14_bus", "mimic_iii_patient_flow", "supplygraph")


# TODO(Phase 19, STEP 1): explore_candidates() -> a DIFFERENCES note per candidate
#   (access, licence, graph definition, tensor-contract fit). NO training here.
# TODO(Phase 19, STEP 2, after professor sign-off): load the chosen domain into the
#   shared X[.,T,N,C] / Y[.,T,N] contract (same windowing/normalisation helpers),
#   so the existing XTrafficSTGNN trains on it with zero architecture changes.
# TODO(Phase 19, STEP 2): run Phase-5 A/B and report the hallucination gap.


def explore_candidates(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """STUB (STEP 1). Will summarise access/licence/graph/tensor-fit per candidate.

    Deliberately does NOT download or train — exploration is a professor-gated
    decision step (see module docstring).
    """
    raise NotImplementedError(
        "Phase 19 STEP 1 (second-domain exploration) is not built yet — this is a "
        "stub. Discuss the domain choice with the professor before building."
    )


def build_second_domain(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """STUB (STEP 2). Will window the chosen domain into the shared tensor contract.

    Gated on Phase-19 STEP 1 and professor sign-off — do not implement first.
    """
    raise NotImplementedError(
        "Phase 19 STEP 2 (second-domain build) is not built yet — this is a stub."
    )


if __name__ == "__main__":
    print("STUB — Phase 19 (Cross-Modal Grounding Transfer) is not implemented yet.")
    print("STEP 1 is exploration only. Discuss the domain choice with the professor.")
