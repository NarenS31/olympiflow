# XTraffic — AI Usage Log

Record of AI assistance in the XTraffic project. Maintained so that any reader —
professor, reviewer, or competition judge — can see exactly where AI contributed
and where the human author's scientific judgement was applied.

**Principle:** every material AI contribution is logged, whether or not it made it
into the final work, including AI-assisted work that turned out to be wrong.

**Human author:** Naren Sara. Retains responsibility for the research question,
scientific judgement, interpretation of results, final writing, and all
application/submission responses. No AI system wrote submission prose.

---

## How to read the columns

| Column | Meaning |
|---|---|
| **Date** | When the assistance occurred |
| **Tool** | Which AI system and interface |
| **Contribution type** | Code · Analysis · Documentation · Review · Design proposal |
| **What the AI produced** | Concrete artifact |
| **Human verification** | What the human author did to check it |
| **Accepted?** | Whether it entered the project, was revised, or was rejected |

---

## 1. Retrospective note on Phases 0–20 (pre-2026-08-21) — INCOMPLETE, ACTION REQUIRED

**This is a gap that must be closed before submission, and it should be closed by
the human author, not reconstructed by an AI.**

The XTraffic system was built following `CLAUDE.md`, which is explicitly an
*"XTraffic — Claude Code Build Playbook"*: a phase-by-phase sequence of prompts
intended to be pasted into Claude Code. The status log in that file records
AI-assisted implementation across Phases 0–20, including code, debugging, and
experiment harnesses.

**No per-session AI usage log was kept during that period.** The evidence that
survives is:

- `CLAUDE.md` — the playbook itself plus a dated, detailed status log
- `LAB_NOTEBOOK.md` (64 KB) — session-by-session narrative
- Git history — 20+ commits, phase-tagged, with dated messages
- In-code comments, many of which record *why* a design decision was made and
  flag deviations

**Recommended action for the human author:** reconstruct a coarse-grained
retrospective entry per phase from `LAB_NOTEBOOK.md` and the git log — dates,
which parts were AI-drafted vs. human-written, and what was verified. A
phase-level record is honest and sufficient; a fabricated turn-by-turn record
would not be. **This is deliberately left for the human author because it is a
factual claim about their own working process, which an AI cannot verify.**

What can be stated accurately right now, and should be:

> The XTraffic implementation was developed with substantial AI assistance
> (Anthropic Claude, via Claude Code) following a written phase playbook. AI
> assistance covered code implementation, debugging, experiment harness
> construction, and documentation drafting. Experimental design decisions,
> dataset and domain selection, interpretation of results, and all scientific
> claims were made by the human author. Several AI-assisted implementations
> contained real bugs that were subsequently found and fixed — these are
> documented in `CLAUDE.md` (e.g. the Phase-11 entity-resolver defects, the
> Phase-3 z-space vs mph-space target-selection bug, the Phase-19 N-1 screening
> bug, the Phase-6 all-NaN visibility channel).

That last sentence is worth keeping. A record showing AI-generated code was
audited and found defective is stronger evidence of scientific care than a record
implying it was accepted as-is.

---

## 2. Session log

### 2026-08-21 — Phase 0 repository audit

| Field | Detail |
|---|---|
| **Tool** | Claude Opus 5 (`claude-opus-5[1m]`) via Claude Code, VS Code extension |
| **Contribution type** | Analysis · Documentation · Design proposal |
| **Scope given by human** | Audit the existing repository read-only; produce a repository map, dependency map, experiment inventory, reproducibility-problem list, supported/unsupported claim lists, and a proposed implementation plan. Explicit instruction: do not modify files, do not run experiments, stop after the audit. |

**What the AI did**

Read-only inspection of the repository: directory structure, all 13 config files,
`requirements.txt`, git log and branch state, and source of the data pipelines,
model, explainer, advisor, and evaluation modules (with `evaluation/faithfulness.py`
read in full, 591 lines). Enumerated committed result artifacts and read the
summary JSON/CSV for each completed study. Queried the local Ollama server for its
installed model list. Ran no experiment, trained nothing, generated nothing.

**What the AI produced**

- `docs/REPOSITORY_AUDIT.md` — repository map (60 Python files, 18,275 lines);
  dependency map with five identified risks; inventory of 20 experiments with
  n and artifact paths; ten ranked reproducibility problems; 19 claims currently
  supported by artifacts; 9 areas not supported; verified / needs-rerun / informal
  classification.
- `docs/EXPERIMENT_PLAN.md` — three scientific decisions escalated to the human
  author rather than decided unilaterally; a revised phase ordering with
  rationale; per-phase deliverables, gates, and costs; a compute budget flagging
  which phases need approval.
- `AI_USAGE_LOG.md` — this file.

**Substantive findings the AI reported** *(each traceable to a file and line)*

1. The Ollama call passes no seed and no token limit
   (`models/advisor/advisor.py:650-657`), so no committed LLM result is
   reproducible. The project had already measured this itself (F1 0.857 vs 0.333
   on a repeated scenario).
2. Raw LLM output is not persisted at scale — roughly 100 of ~4,000 generations
   survive as text. Any new metric therefore requires regeneration.
3. `data/pipelines/chicago.py:57` requests the most recent rows from a live API
   with no date bound, so Chicago data cannot be re-downloaded as it was.
4. Paper Table 3 is built from the superseded pre-correction Phase-5 numbers;
   `make_paper_artifacts.py:424` reads the stale summary file.
5. The faithfulness metric scores a refusal as hallucination 1.0
   (`evaluation/faithfulness.py:540`), which makes the planned no-answer control
   unscoreable as written.
6. Zero automated tests exist across 18,275 lines.
7. **The brief's headline "active grounding 0.438 → 0.912" is the n=5 smoke run
   selected on low F1, not the committed n=93 result of 0.732 → 0.874.** Reported
   as selection on the dependent variable.
8. **The brief's "82.8%" ungrounded rate is the superseded value; the corrected
   figure is 83.9%.**

**Human verification** *(to be completed by the human author)*

- [ ] Spot-check the audit's file/line citations against the source
- [ ] Confirm findings 7 and 8 against `active_grounding_summary.json` and
      `cross_city_faithfulness_master.json`
- [ ] Decide Decisions 1–3 in `docs/EXPERIMENT_PLAN.md`
- [ ] Approve or revise the proposed phase ordering

**Accepted?** Pending human review. No code was changed; the three documents
above are the only files created.

**Files created:** `docs/REPOSITORY_AUDIT.md`, `docs/EXPERIMENT_PLAN.md`,
`AI_USAGE_LOG.md`
**Files modified:** none
**Experiments run:** none

---

### 2026-08-22 — Phases 1 and 3 (reproducibility infrastructure + claim verifier)

| Field | Detail |
|---|---|
| **Tool** | Claude Opus 5 (`claude-opus-5[1m]`) via Claude Code, VS Code extension |
| **Contribution type** | Code · Analysis · Documentation |
| **Scope given by human** | "alright go ahead" — approval following the Phase-0 audit, whose recommendation was to approve Phases 1 + 3 only (~5 days engineering, near-zero compute, both additive). |
| **Decisions taken under that approval** | The three escalated decisions were resolved using the recommended options and stated explicitly to the user: 1C (report alignment and ground-truth references separately), 2B (refusal as its own outcome category), 3A (keep the existing directory layout). All three are config-flagged or additive and reversible. |

**Files created** — Phase 1: `xtraffic/reproducibility/{__init__,provenance,seeds,run_dir,llm_log}.py`, `configs/base.yaml`, `scripts/{run_experiment,reproduce_run,measure_llm_determinism}.py`, `docs/REPRODUCIBILITY.md`. Phase 3: `evaluation/{claim_types,claim_parser,graph_claim_verifier,counterfactual_verifier,claim_metrics}.py`, `tests/{test_claim_verifier,mutation_check}.py`, `scripts/analyze_committed_results.py`, `docs/CLAIM_VERIFICATION.md`.

**Files modified** — `models/advisor/advisor.py` (two optional parameters, flagged in-file, default-preserving; `verify_traffic_unchanged` PASSES), `.gitignore`.

**Experiments run** — `measure_llm_determinism` (2 regimes × 4 draws, real prompt), `analyze_committed_results` (3 analyses, no new LLM calls), unit tests, mutation check, traffic regression gate, 4 resolver gates. No training, no study rerun.

**Substantive results reported**

1. **Measured, not asserted:** the pre-Phase-1 Ollama payload produced **4 distinct outputs from 4 identical requests**; the seeded payload is byte-identical. Confirms audit §5.2. Claim scoped to same server + digest + session.
2. `requirements.txt` does not describe the environment — **10 mismatches**; `torch-geometric` and `rapidfuzz` absent.
3. **Analysis A refuted a concern the AI itself had raised.** The audit suggested the 82.8%/83.9% ungrounded rate might partly be abstention. It is not: all 72 condition-B advisories scoring 1.000 made real citations that all missed; none was empty. Reported as a correction in the project's favour.
4. **Analysis C likewise came back negative:** the resolver backend changes the resolved node set in **0 of 1,212** logged citations.
5. Analysis B: 38.7% of the claim surface was never examined by the old metric; on the 6 surviving full-text advisories the multi-type rate (0.068) is comparable to the old (~0.083). Heavily caveated at n=6.

**Errors the AI made and corrected, on the record**

- **A determinism test that produced a false positive.** The first version expressed "unseeded" as `sampling={"seed": None}`, but `build_options` treats `None` as "use the default", so it silently sent `seed=42` and the legacy path looked deterministic. Caught by a follow-up asking whether the seed did anything at all. `omit=` was added; the episode is documented in `measure_llm_determinism.py` and `docs/REPRODUCIBILITY.md`.
- **A biased verdict rule in the new verifier.** Recommendation sites that resolve to no sensor ("Sunset Blvd and Santa Monica Blvd") were scored `CONTRADICTED`, which would inflate the unsupported rate for any condition eliciting more recommendations. Fixed to `UNVERIFIABLE`; the headline moved 0.220 → 0.068.
- **A test suite that passed for the wrong reasons.** Mutation testing showed the reversal-detection test was satisfied by a fallback branch, the unparsed test was vacuous, and — chasing a third survivor — that the self-attribution rule contradicted the documented Phase-13 definition. All three fixed; 9/9 mutants now killed.

**Human verification** *(to be completed)*

- [ ] Confirm the determinism before/after by rerunning `measure_llm_determinism`
- [ ] Spot-check analysis A against `faithfulness_per_scenario.csv`
- [ ] Review the three Phase-3 decisions as implemented
- [ ] Decide whether to `pip install rapidfuzz==3.6.1` and re-confirm analysis C

**Accepted?** Pending human review. Committed to branch
`phase-1-3-reproducibility-and-verifier`, not merged.

---

## 3. Standing boundaries for AI assistance on this project

Agreed at the start of the Phase-0 session and in force for all subsequent work.

**The AI may:** implement code, tests, experiment pipelines, statistical analysis
code, documentation, and reproducibility infrastructure; audit existing work;
propose experimental designs with tradeoffs.

**The AI may not:** invent data, results, citations, or literature claims; report
an experiment as successful if it was not run; write the human author's STS
application answers, personal essays, or research-report prose intended for
submission; declare a hypothesis proven; silently change the research question;
delete existing work; conceal failed experiments; act as the sole judge of whether
another LLM hallucinated; or generalise beyond the domains actually tested.

**Escalation rule:** where an implementation requires a scientific decision, the
AI presents alternatives with tradeoffs and a recommendation, and the human author
decides.

---

## 4. Template for future entries

```markdown
### YYYY-MM-DD — <phase / task>

| Field | Detail |
|---|---|
| **Tool** | <model + interface> |
| **Contribution type** | Code / Analysis / Documentation / Review / Design |
| **Scope given by human** | <the instruction> |

**What the AI produced:** <artifacts, with paths>
**Experiments run:** <none, or exact commands + run ids>
**Human verification:** <what was checked, and how>
**Accepted?** <accepted / revised / rejected — and why>
**Files created:** <paths>
**Files modified:** <paths>
```
