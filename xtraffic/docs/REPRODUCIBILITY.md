# XTraffic — Reproducibility (Phase 1)

**Status:** implemented and gate-passed 2026-08-22.
**Scope:** infrastructure only. No scientific result changes as a consequence of
this phase; existing studies run exactly as before unless they opt in.

---

## 1. What was wrong

From `docs/REPOSITORY_AUDIT.md` §5. Four defects made committed results
untraceable:

| # | Defect | Consequence |
|---|---|---|
| 5.2 | `advisor.py` sent `options={"temperature": 0.1}` — no seed, no token cap | every LLM number was one unreplicable draw |
| 5.3 | studies persisted metrics, not text | ~3,900 of ~4,000 generations cannot be re-scored |
| 5.5 | artifacts recorded a checkpoint *filename* | `metr_la_best.pt` was three different models; no artifact says which |
| 5.8 | results written to fixed paths | a rerun silently overwrites a committed number |

---

## 2. The measured before/after

`python -m xtraffic.scripts.measure_llm_determinism --n-draws 4`

Real committed advisory prompt (2,750 chars / 385 words), `llama3.1:8b`, same
host, same session, back-to-back:

| | LEGACY (pre-Phase-1 payload) | PHASE 1 (seeded payload) |
|---|---|---|
| seed sent | none | 42 |
| distinct outputs / draws | **4 / 4** | **1 / 4** |
| reproducibility rate | 0.250 | 1.000 |
| byte-identical | False | **True** |
| output length range | 1529–1665 chars | 1567–1567 |

**The audit's §5.2 claim is confirmed by measurement**, not argument: four
identical requests under the committed payload produced four different
explanations, varying by ~9% in length. Every committed LLM number is one
arbitrary draw from that distribution.

The seed is verified to be doing real work, not merely being accepted:

| sampling | result |
|---|---|
| temp 1.0, seed 777, ×4 | 1 distinct output |
| temp 1.0, seeds 1001–1005 | **5 distinct outputs** |
| temp 1.0, no seed key, ×4 | 4 distinct outputs |

**Scope of the determinism claim — read this before quoting it.** Byte-identity
was demonstrated for the *same Ollama server version, same model digest, same
session*. Cross-session, cross-version, and cross-hardware reproducibility is
**not** established by this test and is **not** claimed. `analysis/statistical_plan.md`
therefore treats DRAW as a nested random factor regardless.

> **A development note kept deliberately.** The first version of this check was
> wrong and looked like a *positive* result. It expressed "unseeded" as
> `sampling={"seed": None}`, but `build_options` treats `None` as "use the
> default", so it silently sent `seed=42` and the legacy path appeared perfectly
> deterministic. It was caught only because a follow-up asked whether the seed
> did anything at all, and found "unseeded" output matching `seed=42` exactly —
> too tidy to be real. `build_options(..., omit=["seed"])` now exists so the
> legacy regime can be reproduced honestly.

---

## 3. What now happens on every run

```
xtraffic/reproducibility/
  provenance.py   git SHA + dirty flag + diff hash · Python · package-vs-pins
                  mismatches · platform/device · sha256 of every input file ·
                  Ollama model DIGEST · prompt-template hashes · resolver backend
  seeds.py        Python · NumPy · torch · CUDA · cuDNN · CUBLAS workspace ·
                  DataLoader workers + generator; records what is NOT enforceable
  run_dir.py      append-only versioned run directories; refuses to overwrite
  llm_log.py      every LLM call in full: prompt, raw response, parsed object,
                  sampling options, digest, tokens, latency, truncation flag
```

Run directory layout:

```
evaluation/results/raw/<UTC>__<experiment>__<cfg-hash8>__<git-sha8>/
    manifest.json        config + full provenance + status + per-file sha256
    llm_calls.jsonl      one record per call, including every retry
    <study outputs>
```

`RunDir.create()` raises `FileExistsError` on collision and there is **no force
flag**. To replace a run, create a new one and call `mark_superseded(by, reason)`
on the old — which writes the retraction **into the artifact**, fixing audit
§5.4, where the superseded Phase-5 summary carries no marker and
`make_paper_artifacts.py:424` reads it without complaint.

Failed runs keep their partial outputs and record why they stopped. A run left
in `running` crashed. Neither is deleted: what was tried is part of the record.

---

## 4. Environment truth

The audit assumed `requirements.txt` described the environment. It does not.
Measured on this machine:

```
packages : 10 mismatch(es) vs requirements.txt
resolver : difflib.SequenceMatcher
```

| package | pinned | installed |
|---|---|---|
| torch | 2.2.2 | **2.8.0** |
| torch-geometric | 2.5.3 | **absent** |
| rapidfuzz | 3.6.1 | **absent** |
| pandas | 2.1.4 | 2.2.3 |
| matplotlib | 3.8.2 | 3.9.4 |
| requests | 2.31.0 | 2.32.5 |
| fastapi | 0.110.0 | 0.115.6 |

**`rapidfuzz`'s absence is not cosmetic.** `evaluation/faithfulness.py:60-74`
prefers `rapidfuzz.token_set_ratio` and silently falls back to
`difflib.SequenceMatcher` — a *different similarity function* compared against
the *same* threshold of 82. Every faithfulness number depends on which is
installed, and nothing recorded it. It is now in `provenance.metric` on every
run, and a change to it is flagged **CRITICAL** by `reproduce_run`.

Quantifying the actual impact on committed numbers is a Phase-3 deliverable
(§7 below). `torch-geometric` is declared but never imported — Phase 3 of the
original build reimplemented GNNExplainer in pure PyTorch — so its absence is
harmless, but the pin misrepresents the dependency surface.

---

## 5. Commands

```bash
# Exercise the whole harness without running a study
python -m xtraffic.scripts.run_experiment --config configs/base.yaml \
       --experiment smoke --dry-run --check-determinism

# Resolved config, no side effects
python -m xtraffic.scripts.run_experiment --print-config

# Override anything
python -m xtraffic.scripts.run_experiment --set llm.model=mistral:7b \
       --set seeds.master=43

# Measure LLM reproducibility (legacy vs seeded)
python -m xtraffic.scripts.measure_llm_determinism --n-draws 4

# List runs / diff a past run against the current environment
python -m xtraffic.scripts.reproduce_run --list
python -m xtraffic.scripts.reproduce_run --run-id 20260822T113900Z
```

`reproduce_run` emits a **field-by-field diff**, not a pass/fail, because
"reproduces except the checkpoint hash changed" is a diagnosis and "FAIL" is
not. Verdicts: `REPRODUCIBLE` · `REPRODUCIBLE_WITH_CAVEATS` · `NOT_REPRODUCIBLE`.

---

## 6. Opting a study in

Existing studies are untouched. To log and seed one:

```python
from xtraffic.reproducibility import llm_log
from xtraffic.models.advisor.advisor import Advisor

logger = llm_log.LLMLogger(run.file("llm_calls.jsonl"), run.run_id)
call   = llm_log.make_logged_caller(host, model, logger,
                                    sampling=cfg["llm"]["sampling"])
advisor = Advisor("metr_la", call_fn=call)     # <- the only change
```

`advisor.py` gained two optional parameters (`call_fn`, `sampling`). With both
omitted the original inline request path runs and the payload is
`{"temperature": self.temperature}` exactly as before.

**Verified default-preserving:** `python -m xtraffic.evaluation.verify_traffic_unchanged`
→ `PASS` (12 explanations × 5 rendered artifacts, the 3 prompt constants, and
the resolver on 2 cities, all byte-identical to pre-Phase-19).

**Honest caveat.** Prompt bytes are unchanged; **generated text is not**. The
logged path adds a seed and pins `top_p`/`top_k`/`num_predict`/`num_ctx`, which
the old path left to server defaults. So committed pre-Phase-1 numbers are **not
comparable** to logged-path numbers, and `verify_traffic_unchanged` must be
re-baselined with a documented reason before the logged path is used for a
headline study.

---

## 7. What Phase 1 does *not* fix

| Problem | Status |
|---|---|
| Chicago data cannot be re-downloaded (audit §5.1) | **open** — needs a date-bounded Socrata query + snapshot checksum, or Chicago drops out of confirmatory claims |
| ~3,900 existing generations have no stored text | **unfixable retrospectively.** Phase 1 prevents recurrence; it cannot recover what was never written |
| Committed artifacts carry no provenance | **unfixable retrospectively.** Which checkpoint produced which committed number is not recoverable |
| Paper Table 3 built from superseded numbers (§5.4) | **open** — `mark_superseded` exists; repointing `make_paper_artifacts.py` is a separate change |
| rapidfuzz-vs-difflib impact on committed numbers | **open** — quantified in Phase 3 |
| Cross-session / cross-version LLM determinism | **not established.** Only same-session byte-identity was measured |
| MPS ↔ CUDA ↔ CPU bitwise identity | **not achievable.** Recorded in `seeds.not_enforceable`, never claimed |

---

## 8. The rule

**An LLM call that is not fully logged did not happen.**

Every call writes prompt, raw response, parsed object, sampling parameters,
model digest, retry index, token counts, latency, and a truncation flag. That
record is sufficient to re-score the call under any future metric without
touching the LLM again — which is precisely what the existing corpus cannot do,
and why the Phase-11 outlier investigation had to re-run the model to diagnose
itself.
