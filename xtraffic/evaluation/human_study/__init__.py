"""Phase 8 — expert decision-quality study (XTraffic Contribution #3).

Modules:
  simulate.py   — model-in-the-loop intervention ground truth
  scenarios.py  — build the 24-scenario set + RAW/XAI/XTRAFFIC bundles + Latin square
  demo_data.py  — synthetic scenario set (no ML) so the app can be piloted anywhere
  app.py        — minimal FastAPI evaluation web app (one command, JSONL log)
  analyze.py    — per-condition accuracy/confidence/usefulness + paired Wilcoxon
"""
