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

### 2026-08-24/25 — Explanation-network analysis (what the explanations say about the road network)

| Field | Detail |
|---|---|
| **Tool** | Claude Opus 5 (`claude-opus-5[1m]`) via Claude Code, VS Code extension |
| **Contribution type** | Code · Analysis · Documentation |
| **Scope given by human** | Analyse what the trained ST-GNN's explanations say about the METR-LA road network, not about the LLM. Three steps, in order, all exploratory. Explicit constraints: no LLM calls anywhere, no Ollama; route every output through the Phase-1 run_dir infrastructure; seed everything; do not modify the checkpoint, the 12 committed explanations, or anything the verifier reads; report every edge above threshold, not a curated subset; do not claim a discovery. A feasibility gate (Step 0) was required before any run, with a stop-and-report. |
| **Decisions escalated, not taken** | Step 0 reported that the full 207x400 grid was 809 h and stopped. The human chose sample A (207x24), CPU, no confidence reruns, and Step 2 skipped entirely. Mid-run the AI reported a throughput overrun against its own estimate and offered four options rather than silently continuing or silently cutting scope; the human chose to let it run. |

**Files created** — `evaluation/influence_graph.py`, `scripts/{run_influence_solves,analyze_influence_graph,report_influence_graph,analyze_learned_graph,report_learned_graph,measure_device_divergence}.py`, `tests/test_influence_graph.py`.

**Files modified** — `CLAUDE.md` (dated alpha correction + status block), `LAB_NOTEBOOK.md`, `docs/REPOSITORY_AUDIT.md` (addendum §11.1). No existing code path was changed; one figure helper was generalised from regime keys to arbitrary panel keys.

**Experiments run** — 4,968 GNNExplainer solves (207 targets x 24 windows, CPU, 8.4 h); a static read of three N=207 checkpoints; a 12-solve CPU-vs-MPS probe. **Zero LLM calls.**

**Substantive results reported**

1. **The explainer's mask is nearly flat.** 147-154 of 206 sources are needed to cover 80% of a target's off-self importance mass (sd 4.7 / 2.9). The committed explanation JSONs keep `top_nodes` = 8 of 207, so every downstream consumer in this repository — including the entire faithfulness line of work — has been reading the first 8 entries of a nearly flat ranking. This is the most consequential finding and it constrains the interpretation of prior phases.
2. **The aggregate is nevertheless well above chance** (precision 0.286 / 0.140 vs 0.033 / 0.037 uniform), while a pure nearest-by-road-distance rule scores 0.756-0.809 — as it must, since the adjacency *is* a 3.9 km road-distance threshold.
3. **The far stratum is depleted, not enriched** (0.57x base rate free-flow), contradicting the AI's own prior expectation, which is stated as such in report.md.
4. **"Which nodes are important" is not a well-determined object in this model.** Three independent probes converge: CPU vs MPS at one seed, epoch-34 vs epoch-54, and split-half over windows. Ranks are stable; top-k identity is not.
5. **Surviving far edges do not recover the learned graph** (15.4% in A_sem's top 8 vs 3.9% chance) — reported as a negative answer to the human's own hypothesis, with all 52 survivors listed with geometry rather than summarised.
6. **alpha correction:** CLAUDE.md carried only 0.49->0.39 from a deleted 3-epoch checkpoint. Measured values are 0.3366 (ep34) and 0.2968 (ep54). Appended as a dated correction; the original line was left as written.

**Errors the AI made and corrected, on the record**

- **A cost estimate that was wrong by 2.5x.** The 6.8 h projection assumed 8 homogeneous cores; this M4 is 4 performance + 6 efficiency cores, so the benchmark (taken on an idle machine) came off a performance core. Reported to the human mid-run with options rather than absorbed silently. Actual: 8.4 h, helped by an unrelated job of the human's finishing.
- **An `UnboundLocalError` shipped into a results file.** `cfg_ckpt` was referenced above its definition, so the first full analysis wrote `support_density: {"error": ...}` instead of the measurement. Caught by reading the output rather than trusting the exit code; fixed and the analysis rerun.
- **A relabel instruction that could not be followed literally.** The human asked for the "beyond-K" stratum to be relabelled ">3.9 km". That describes *every* non-adjacent pair, whereas the stratum is >8 *chained* 3.9 km hops. The AI flagged the imprecision and used an accurate label instead of silently applying a wrong one.

**Human verification** *(to be completed)*

- [ ] Confirm the flatness result against `metrics.json` -> `sources_for_mass_frac.per_target`
- [ ] Spot-check any survivor row in `beyond_k_survivors_free_flow.csv` against the coordinates
- [ ] Decide whether the flat-mask finding requires a caveat in the Phase-5/11/12 faithfulness write-ups
- [ ] Re-run `analyze_learned_graph` (seconds, no solves) to confirm the alpha correction

**Accepted?** Pending human review. Committed to branch `explanation-network-analysis`, not merged.

---

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

---

### 2026-08-25/26 — Grounding without information (pre-registered, four parts)

| Field | Detail |
|---|---|
| **Tool** | Claude Opus 5 (`claude-opus-5[1m]`) via Claude Code, VS Code extension |
| **Contribution type** | Code · Analysis · Documentation |
| **Scope given by human** | One experiment, four parts, pre-registered, on branch `explanation-network-analysis`. Budget ~14 h compute; estimate each part first and say if it does not fit. Part 1: faithfulness under A / B / A_rand / A_mismatch on the same 93 scenarios, token counts within 5% of A. Part 2: the existing active grounding loop on A_rand. Part 3: the existing decision simulation on A_rand. Part 4: sparsity sweep, entropy vs stability, threshold applied to the 12 committed explanations. Write predictions.md before any LLM call or solve. Literature check before report.md. Seeded, byte-reproducible, run_dir only, no MPS, nothing written into explanations_cache. |

**What the AI produced**

- `evaluation/noise_conditions.py` — `A_rand` / `A_mismatch` artifact builders.
  `A_rand` patches `GNNExplainer.explain_target` for a seeded uniform draw and
  lets the real `explain_prediction` run on top, so every derived field is
  internally consistent. `A_mismatch` uses a seeded derangement.
- `evaluation/mask_entropy.py` — normalised entropy, 80%-mass count, top-k
  Jaccard, Youden threshold, Spearman with a target-clustered bootstrap.
- `scripts/`: `init_grounding_noise_run.py`, `noise_run.py`,
  `run_grounding_without_information.py`, `run_loop_on_noise.py`,
  `run_decisions_on_noise.py`, `run_sparsity_sweep.py`,
  `analyze_sparsity_sweep.py`, `report_grounding_without_information.py`.
- Run dir `20260825T201433Z__grounding_without_information__f193e764__a1b0fc36`
  with `predictions.md`, `literature.md`, `report.md`, `verdicts.json`, 4 tables
  (md + LaTeX), 3 figures, per-scenario CSVs. Committed copy under
  `evaluation/results/processed/<run_id>/`.

**Experiments run:** 498 logged LLM calls (llama3.1:8b) across Parts 1–3, plus
2,662 CPU explainer solves for Part 4. 11.6 h wall clock. Full command list and
per-call log in the run dir.

**Budget estimate given BEFORE starting, as instructed:** measured 34.9 s/advisory,
11.2 s/decision, 6.04 s/solve, and reported that the brief did not fit in 14 h
(16.4–18.7 h serial). The human chose the reductions: Part 3 subsampled to n=150,
`A_mismatch` defined as sources-swapped, and the unseeded LLM path kept for
comparability. Those three decisions are the human's, not the AI's.

**Substantive findings reported**

1. **P1 CONFIRMED.** `A_rand` (uniform-random importances) scores F1 0.701 /
   precision 0.995 / hallucination 0.005 against real A's 0.725 / 0.995 / 0.005;
   `A_mismatch` scores 0.727. All paired-difference CIs span zero. Condition B
   (explanation removed) still collapses to 0.090 / 0.828 — so the metric is not
   broken, it is measuring citation compliance rather than evidential content.
2. **P2, P3, P4 FALSIFIED**, each on stated criteria, with the failing clause
   named and not reinterpreted. P3's failing clause was a badly chosen criterion
   (it assumed a RAW-vs-XTRAFFIC accuracy gap that does not exist at n=150) and
   is recorded as an error in the pre-registration, not a result.
3. **The sparsity coefficient does not control sparsity** — `lambda_size` x10
   made the mask flatter (entropy 0.9976 -> 0.9988). Not predicted.
4. **Provenance bug in committed artifacts**: the 12 committed explanation JSONs
   name a checkpoint that no longer reproduces them; they are epoch-34 objects
   and the schema stores a filename rather than a hash.

**Human verification:** OWED. The human author has not yet reviewed this run.
Specifically owed: (a) the novelty claim in `literature.md`, which is flagged for
the human's decision and must not be used until reviewed; (b) the "LOGIC" paper
named in the brief, which the AI could not identify and recorded as NOT CHECKED
rather than silently dropping; (c) two near-neighbour papers assessed from
abstracts only and labelled as such.

**Accepted?** PENDING REVIEW. Committed so the numbers are on the record, not
because they have been accepted.

**Files created:** the modules and run dir listed above.
**Files modified:** `LAB_NOTEBOOK.md`, `CLAUDE.md`, this file — all append-only.
No committed code, config, result or explanation artifact was modified;
`git diff` over tracked files was empty before staging.
