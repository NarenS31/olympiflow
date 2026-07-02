# XTraffic — Lab Notebook

Running log of what was built, the numbers, and what surprised us. 3–5 lines per
session. This becomes the paper's experiments section almost for free.

---

## 2026-07-01 — Phase 0 + Phase 1 (data pipelines)
- Established project memory (Phase 0) by prepending the XTraffic context to
  `CLAUDE.md`; kept the full build playbook below it.
- Built the canonical `/xtraffic` scaffold on top of the existing OlympiFlow repo
  (OlympiFlow kept as a base + Ollama-RAG reference for Phase 4).
- Wrote reproducible pipelines: `metr_la.py`, `pems_bay.py` (shared `_h5_common.py`),
  and `chicago.py` (segment-graph, Socrata API). All emit one tensor contract:
  `X [samples,12,N,2]`, `Y [samples,12,N]`, plus adjacency/scaler/node_meta/stats.
- Key design decisions: chronological (never shuffled) 70/10/20 split to avoid
  temporal leakage; z-score scaler fit on TRAIN speeds only; Gaussian-kernel
  adjacency (DCRNN Eq. 10) for LA/Bay, shared-endpoint binary graph for Chicago.
- Schema divergences documented in `data/pipelines/DIFFERENCES.md` for the paper's
  generalization section.
- **Gate PASSED.** Numbers: METR-LA 207 nodes / 34,272 steps / 8.11% missing;
  PEMS-BAY 325 nodes / 52,116 steps / 0.003% missing; Chicago 1,020 segment-nodes
  / 99 steps / 66% missing (portal feed was stale to ~2026-04-30, so only a few
  hours came back — enough to prove the tensor contract, not to train on).
- **Surprise:** the DCRNN `.h5` files 404'd on Zenodo; the record actually hosts
  CSV. Switched to CSV, which also let us drop the pytables dependency entirely.
- **TODO next session (Phase 2):** `pip install -r xtraffic/requirements.txt`
  (torch + torch-geometric), then build the ST-GNN. Also widen the Chicago
  Socrata window before it's usable for cross-city training (Phase 6).
