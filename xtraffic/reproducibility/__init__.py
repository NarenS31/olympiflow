"""Phase 1 — reproducibility infrastructure.

Four small modules, each with one job:

  provenance.py  WHAT produced a result (git SHA, package versions, checkpoint
                 hash, data hash, Ollama model digest, prompt-template hash).
  seeds.py       Make a run as deterministic as the hardware allows, and be
                 HONEST in the record about the parts that cannot be.
  run_dir.py     Versioned, append-only run directories. Refuses to overwrite.
  llm_log.py     Every LLM call persisted in full: prompt, raw response, parsed
                 object, sampling parameters, model digest, timings.

WHY THIS EXISTS (see docs/REPOSITORY_AUDIT.md §5)
-------------------------------------------------
The pre-Phase-1 codebase recorded a checkpoint FILENAME and nothing else. That
filename was overwritten three times (3-epoch placeholder -> epoch 34 -> epoch
54), so no committed artifact can say which model produced it. Separately, the
Ollama call passed no seed, and no study persisted the LLM's raw output, so
~3,900 of ~4,000 generations exist only as summary metrics and can never be
re-scored under a new metric.

Nothing here changes any existing behaviour. These modules are additive: old
scripts that never import them run exactly as before.

Python 3.9 compatible (typing.Optional/Union, no `X | Y`).
"""
from __future__ import annotations

__all__ = ["provenance", "seeds", "run_dir", "llm_log"]
