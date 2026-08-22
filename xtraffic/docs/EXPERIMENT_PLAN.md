# XTraffic — Proposed Experiment Plan (Phase 0 deliverable)

**Date:** 2026-08-21
**Status:** PROPOSAL. Nothing here has been implemented. Awaiting approval.
**Companion:** `docs/REPOSITORY_AUDIT.md` (the evidence this plan is built on).

---

## 0. The hypothesis, and what would falsify it

**Central hypothesis (as stated in the brief):**

> Structured, machine-checkable graph evidence reduces unsupported LLM claims
> when an LLM explains graph-neural-network predictions more effectively than
> additional textual context, generic self-correction, or retrieval-only
> grounding.

**What would falsify it** — the plan must be able to produce each of these:

| Falsifier | The condition that produces it |
|---|---|
| Prose carrying the same facts does as well as structured evidence | B ≈ D |
| Flat JSON (no relations) does as well as relational graph evidence | C ≈ D |
| Structurally *wrong* evidence works as well as correct evidence | M ≈ D → the model is following *form*, not content |
| Shuffling edges/labels doesn't hurt | E ≈ D or F ≈ D → same conclusion |
| A generic retry closes most of the gap | J ≈ L |
| Self-critique with no new evidence closes the gap | K ≈ L |
| Retrieval alone closes the gap | I ≈ D |
| The gain is the model refusing more often | N reveals D's advantage is abstention |
| The effect vanishes on held-out domains | Phase 7 |
| The effect is an explainer artifact | Phase 6 |
| The effect vanishes when the verifier checks claim types beyond node membership | Phase 3 |
| The effect vanishes when scored against *ground truth* rather than *what was shown* | Phase 2 |

The last two are the ones current XTraffic cannot test at all, and they are the
ones most likely to change the answer. **They are therefore the highest priority.**

---

## 1. Three scientific decisions I am NOT making unilaterally

Per the implementation rules, these change what the project *is* and are yours.

### DECISION 1 — What "unsupported" is measured against

The current metric scores claims against **the explainer's top-k, i.e. what the
LLM was shown**. The hypothesis says "unsupported claims." These are different
things, and the project's own SHAP result proves it: two near-disjoint
explanations (Jaccard 0.022) both score as perfectly faithful.

| Option | Reference set | Answers | Cost | Risk |
|---|---|---|---|---|
| **1A. Keep alignment only** | explainer output | "Does the LLM describe what it was shown?" | zero | Hypothesis becomes about *transcription fidelity*, not unsupported claims. A reviewer will say so. |
| **1B. Ground truth only** (synthetic) | generator's true subgraph | "Does the LLM state true things?" | Phase 2 required | Only available on synthetic data; real-world claims become unverifiable |
| **1C. Both, reported separately** ← *recommended* | both | "Is it faithful to the evidence?" **and** "Is the evidence true?" | Phase 2 + a two-column metric | More columns, more nuance to communicate |

**Recommendation: 1C.** It is the only option that keeps the existing real-data
studies meaningful (as *alignment* results) while letting the synthetic benchmark
answer the correctness question. It also makes the SHAP finding a *result*
("alignment is explainer-invariant, correctness is not") rather than an
embarrassment.

**This changes what the paper claims.** Under 1C the headline is
*"structured evidence improves machine-verifiable explanation fidelity, and on
synthetic graphs with known truth it also improves correctness"* — two claims, one
strictly weaker than the current implicit one.

### DECISION 2 — How refusals are scored

Current code: `hallucination = 1.0 if no cited causes`. A refusal is scored
identically to four fabrications. Condition N (no-answer control) is
unimplementable until this changes.

| Option | Treatment | Consequence |
|---|---|---|
| **2A. Refusal = hallucination** (status quo) | punished maximally | Condition N is meaningless; also inflates current condition-B numbers by conflating "declined" with "fabricated" |
| **2B. Refusal excluded from precision, reported as its own rate** ← *recommended* | precision computed over *attempted* claims; `refusal_rate` reported alongside | Standard in the QA-abstention literature. Makes "did structure just teach it to shut up?" directly answerable |
| **2C. Refusal = fully correct** | rewarded | Degenerate: a model that always refuses wins |

**Recommendation: 2B**, with the explicit rule that **any condition's headline
number must be reported next to its refusal rate**, so abstention can never be
laundered as accuracy.

**Note:** this retroactively affects the interpretation of committed condition-B
numbers. How many of the 93 B-condition scenarios were empty vs. fabricated is
recoverable from `advisory_error` and `n_cited_causes` in the existing CSV — a
cheap, no-LLM re-analysis worth doing early.

### DECISION 3 — Directory layout

The brief asks for `src/`, `scripts/`, `results/`, `tests/`. `CLAUDE.md` declares
a canonical layout (`models/`, `evaluation/`, `data/`, `utils/`, `configs/`) and
says *"do not deviate."* 2,497 tracked files and every `python -m xtraffic.*`
invocation, plus `REPRODUCE.md` and both runner scripts, assume the current one.

| Option | Cost | Risk |
|---|---|---|
| **3A. Keep current layout; add new dirs alongside** ← *recommended* | none | Layout differs from the brief's literal text |
| **3B. Migrate to `src/`** | rewrite every import, both runners, REPRODUCE.md, all module paths | High — silently breaks caches keyed by path; invalidates the regression gate |
| **3C. `src/` as a shim re-exporting current modules** | small | Two names for everything; confusing |

**Recommendation: 3A.** Map the brief's paths onto the existing layout:
`src/reproducibility/` → `xtraffic/reproducibility/`;
`src/synthetic_graphs/` → `xtraffic/synthetic_graphs/`;
`src/evaluation/*` → `xtraffic/evaluation/*` (already there);
`src/explainers/` → `xtraffic/models/explainer/` (already there);
`src/adversarial/`, `src/counterfactuals/` → `xtraffic/adversarial/`,
`xtraffic/models/explainer/counterfactual.py` (exists).
`scripts/`, `tests/`, `annotation/`, `analysis/`, `docs/` are genuinely new and go
in at `xtraffic/` root. **Say the word if you'd rather match the brief literally.**

---

## 2. Ordering, and why it differs from the brief's numbering

The brief lists Phases 1→15. I propose executing them in a different order,
because three of them are **prerequisites that invalidate later work if deferred**:

```
  1  Reproducibility           ─┐ nothing downstream is trustworthy without this
  3  Deterministic verifier    ─┤ test-first; every later metric depends on it
  2  Synthetic ground truth    ─┘ the only source of truth; unblocks 6, 7, 9, 10
        ↓
  5  Matched ablation matrix     ← the hypothesis test itself
  11 Statistical plan (WRITE)    ← must be written BEFORE confirmatory runs
        ↓
  6  Explainer robustness   ┐
  7  Cross-domain           ├ can run in parallel once 1/2/3/5 land
  8  Cross-model            │
  9  Adversarial            │
  10 Counterfactual         ┘
        ↓
  4  Human annotation          ← needs real generated text from 5 to annotate
  12 Claim-evidence matrix
  13 Sim-decision audit        ← independent; can start any time (analysis only)
  14 Freeze + confirmatory run
  15 Final outputs
```

**Why Phase 3 before Phase 2:** the verifier is the contract the generator must
satisfy. Writing the generator first risks a ground-truth format the verifier
can't consume.

**Why Phase 4 (human) late:** there is currently almost nothing to annotate
(§5.3 of the audit — ~100 of ~4,000 generations survive as text). Annotation
should run on Phase-5 output, which will be persisted properly.

**Why Phase 11 is split:** *write* the analysis plan before Phase 5's confirmatory
run; *execute* the analysis after. Writing it afterwards makes every result
exploratory by definition.

---

## 3. Phase-by-phase plan

Each phase lists: deliverables · gate (what must be true to proceed) · cost ·
what could go wrong.

### PHASE 1 — Reproducibility infrastructure

*No experiments. Pure infrastructure. Nothing existing is deleted or rewritten.*

**Build**
- `xtraffic/reproducibility/provenance.py` — captures git SHA + dirty flag, Python
  version, `pip freeze`, platform/device, checkpoint **sha256**, data **sha256**,
  prompt-template hash, resolver fuzzy backend, Ollama model **digest** (from
  `/api/tags`), wall-clock. One call, one dict.
- `xtraffic/reproducibility/seeds.py` — `set_all_seeds(seed)` covering Python,
  NumPy, torch, CUDA, cuDNN determinism flags, `CUBLAS_WORKSPACE_CONFIG`,
  DataLoader `worker_init_fn` + `generator`. Documents honestly that MPS↔CUDA
  bitwise identity is **not** achievable.
- `xtraffic/reproducibility/run_dir.py` — `results/raw/<run_id>/` where
  `run_id = <UTC timestamp>_<config hash>_<git short SHA>`. **Refuses to write into
  an existing run dir.** Emits `manifest.json`.
- **`xtraffic/reproducibility/llm_log.py`** — the single most important file in
  this phase. Every Ollama call appends one JSONL record:
  `{run_id, scenario_id, condition, model, model_digest, seed, options{...},
    prompt_sha256, prompt, raw_response, parsed, parse_errors, retry_index,
    prompt_tokens, completion_tokens, latency_ms}`.
- **Change to `advisor.py`** (flagged, default-preserving): pass `seed`,
  `num_predict`, `top_p`, `top_k`, `num_ctx`, `repeat_penalty` from config; route
  every call through `llm_log`. Defaults chosen so that with the current
  `advisor.yaml` the *prompt bytes* are unchanged — but note that adding a seed
  **will** change generated text, so this is not byte-identical on output and
  `verify_traffic_unchanged.py` will need re-baselining with an explicit,
  documented reason.
- `configs/` additions: `base.yaml` (seeds, LLM sampling, evidence budget, token
  budget, retries, eval split) + per-experiment overlays.
- `scripts/run_experiment.py` (config → run dir → manifest) and
  `scripts/reproduce_run.py` (run dir → re-execute → **diff report**, not a silent
  pass).
- `docs/REPRODUCIBILITY.md`.

**Gate**
1. Two runs of the same config with the same seed produce **byte-identical**
   parsed claims for ≥95% of scenarios *(if not, report the actual rate — Ollama
   may not be fully seed-deterministic across model reloads, and that is itself a
   finding worth documenting rather than hiding)*.
2. `reproduce_run.py` on an existing run dir reports a clean diff.
3. Writing into an existing run dir fails loudly.
4. A deliberately corrupted checkpoint hash is detected.

**Cost:** ~2 days engineering. Trivial compute (one ~20-scenario smoke).

**Risk:** Ollama seeding may not give exact determinism. If so, gate 1 becomes a
*measured reproducibility rate*, and the k-draws design in Phase 11 carries the
load instead. **Do not fake determinism that isn't there.**

---

### PHASE 3 — Deterministic claim verifier *(before Phase 2, test-first)*

**Build**
- `xtraffic/evaluation/claim_parser.py` — decomposes an advisory into **atomic
  claims**, from *both* the structured fields and the `reasoning` prose. Claim
  types: node-existence · edge-existence · edge-direction · path-existence ·
  feature-attribution · temporal · confidence · prediction · counterfactual ·
  domain-terminology.
- `xtraffic/evaluation/graph_claim_verifier.py` — per-type deterministic checks
  against the explanation JSON + adjacency + node table. Verdicts:
  `SUPPORTED · UNSUPPORTED · CONTRADICTED · UNVERIFIABLE · PARTIALLY_SUPPORTED`.
  **`CONTRADICTED` is new and important** — the current metric cannot distinguish
  "cited something not in top-k" from "asserted the reverse of a real edge."
- `xtraffic/evaluation/counterfactual_verifier.py` — reruns the frozen model to
  check claimed prediction changes. Reuses `counterfactual.predict_target_mph`,
  which already does exactly this.
- `xtraffic/evaluation/metrics.py` — claim-level precision / recall / F1 /
  unsupported rate / contradiction rate / **refusal rate** / coverage, with the
  Decision-2 refusal semantics and a per-claim-type breakdown.
- `tests/test_claim_verifier.py` — **written first.** Must include: empty
  `cited_causes`; refusal text; reversed edge direction; a path that exists in the
  graph but not in the explanation; a plausible-but-absent node name; a number
  just inside and just outside ±10%; a claim about a node in the graph but not in
  top-k; unicode/format edge cases.
- `tests/test_faithfulness_resolver.py` — retrospective tests pinning the
  *current* resolver behaviour (all four hand-labeled sets, plus the `n_causes==0`
  branch and the rapidfuzz/difflib divergence).
- `docs/CLAIM_VERIFICATION.md`.

**Design commitment — no LLM judges.** Parsing is rule-based: regex + the existing
node/region tables + a small typed grammar for relational statements. Where a
sentence cannot be parsed into an atomic claim, it is logged as
`UNPARSED` and counted — **never silently dropped**, and the unparsed rate is a
reported number. If the unparsed rate is high, that is a finding about prose
explanations, and it is reported as one.

**Gate**
1. All unit tests pass, including every adversarial case above.
2. On the ~100 surviving full-text advisories, the new verifier's node-claim
   verdicts agree with the old `score_advisory` on ≥95% of claims; **every
   disagreement is inspected by hand and explained in writing.**
3. Unparsed-claim rate is reported, not hidden.
4. The verifier runs with no network and no LLM.

**Cost:** ~3 days. Zero LLM compute (runs on existing artifacts).

**Risk — the real one:** prose parsing is genuinely hard and the failure mode is a
verifier that quietly under-counts claims it can't parse, which would *flatter*
every condition. Mitigation: report unparsed rate per condition; if it differs
across conditions, that is a confound and must be reported as one.

---

### PHASE 2 — Synthetic graph benchmark with known truth

**The single highest-value addition in this plan.** It is the only way to score
claims against something *true* rather than something *shown*.

**Build**
- `xtraffic/synthetic_graphs/generators.py` — parameterised generator emitting the
  existing `X[S,T,N,C] / Y[S,T,N]` contract (so the whole downstream stack runs
  unchanged, exactly as the power-grid pipeline proved).
  Rule families: single-hop · multi-hop · competing paths · delayed effects ·
  shared-parent confounding · spurious correlation · edge-sign reversal ·
  time-varying causal edges · node interventions.
  Nuisance structure: distractor nodes/edges · confounded pairs · irrelevant
  high-degree hubs · missing features · noise · train→test structural shift.
  Sweeps: graph size · sparsity · temporal horizon.
- `xtraffic/synthetic_graphs/ground_truth.py` — emits causal nodes, causal edges,
  directed paths, relevant features, **and a ground-truth explanation JSON valid
  against the existing `schema.py`** so it drops into the current prompt renderer
  with no changes.
- `tests/test_synthetic_ground_truth.py` — the generator must be verified, not
  assumed: intervening on a declared causal node changes the target; intervening
  on a distractor does not; declared paths exist in the adjacency; declared lags
  match the realised cross-correlation; a model trained on family A and tested on
  family B actually degrades.
- `configs/synthetic_benchmark.yaml`, `docs/SYNTHETIC_BENCHMARK.md`.

**Split discipline (explicit, because it's easy to get wrong):** train / val / test
are **separate graph families** — different topology draws *and* different rule
mixes — not different time windows of near-identical graphs. Held-out families are
never touched during prompt or threshold tuning.

**Gate**
1. All generator self-verification tests pass.
2. A trained `XTrafficSTGNN` beats persistence and historical-average on the
   synthetic task by a clear margin — **otherwise there is no signal to explain
   and the benchmark is degenerate.** (This is exactly the check the power-grid
   pipeline's neighbour-R² diagnostic caught: v1 had a 1.17× neighbour ratio and
   would have been unexplainable.)
3. Explainer ground-truth recovery is **above chance but below ceiling** — if
   GNNExplainer recovers 100%, the benchmark is too easy to discriminate methods;
   if ~chance, either the model or the generator is broken.
4. The generated explanation JSON validates against `schema.py` unchanged.

**Cost:** ~4–5 days engineering + a few hours GPU/CPU training per family.

**Risk:** synthetic benchmarks are easy to make trivially easy or accidentally
impossible. Gates 2 and 3 exist to catch both. Budget for two iterations.

**Honest limitation to state in the paper:** results on synthetic graphs bound
what we can claim about *real* graphs. The benchmark tells us whether the
mechanism works when truth is knowable; it does not prove the real-data
explanations are correct.

---

### PHASE 5 — Matched ablation matrix *(the hypothesis test)*

14 conditions A–N as specified in the brief. Notes on the ones that are hard:

- **B (prose evidence)** must carry *exactly* the same facts as D, generated from
  the same explanation JSON by a deterministic templater — never hand-written, or
  the contrast confounds content with authorship.
- **C (flat JSON)** = same node/feature facts, **relations removed**. This is the
  cleanest single test of "does *structure* matter", because C and D differ only
  in whether relationships are expressed.
- **E/F (shuffled edges / labels)** are the sharpest falsifiers. If E ≈ D the model
  is following *format*, not *content* — and the whole hypothesis is in trouble.
  These must be run and reported whatever they show.
- **M (wrong evidence)** must be *plausible* wrong, not obviously wrong.
- **N (no-answer)** requires Decision 2 resolved first.
- **J/K (retry / self-critique)** are the *retry-count* controls for L. They must
  use the **same number of LLM calls** as L actually consumed, per scenario —
  matching the mean is not enough, because L's engaged subpopulation is what does
  the work (see the audit's §7.1).

**Matching protocol (enforced in code, logged per call, and *verified* not assumed)**
model · digest · temperature · seed set · `num_predict` · `num_ctx` · number of
attempts · number of evidence items · scenario set · **prompt token count within a
declared tolerance**. `scripts/run_ablation_matrix.py` refuses to run if any
declared match constraint is violated, and the achieved match is reported in a
table so a reviewer can check it.

**Design**: k ≥ 5 LLM draws per (scenario × condition), all persisted.
Scenarios drawn from synthetic families (ground truth available) **and** METR-LA
(real data, alignment only).

**Gate:** smoke at n=5 scenarios × 14 conditions × 2 draws first; verify the match
report is clean and every condition's prompt renders correctly, **before** the
full run.

**Cost:** the expensive one. 14 conditions × ~90 scenarios × 5 draws ≈ 6,300 LLM
calls per model per domain. At observed local throughput this is on the order of
**tens of hours per model-domain cell.** Needs explicit approval and a queue plan.

---

### PHASE 11 (write) — Prespecified analysis plan

**Written and committed before the confirmatory run.** `analysis/statistical_plan.md`
fixes: primary endpoint (proposal: **claim-level unsupported rate**, D vs the
strongest non-structural competitor among B/I/J/K) · secondary endpoints ·
minimum meaningful effect · **unit of analysis** · power justification ·
multiple-comparison correction (proposal: Holm within a declared family) ·
bootstrap procedure · **hierarchical model** · seed handling · missing-data rule ·
exclusion criteria · stopping rule.

**The pseudoreplication problem, stated concretely.** Observations nest:

```
draw ⊂ scenario ⊂ domain × model × explainer × condition
```

93 scenarios from one METR-LA split sharing one graph and one checkpoint are not
93 independent observations. The current bootstrap clusters by scenario only,
which handles the *within-scenario* correlation but not domain or model
dependence. Proposal: mixed-effects logistic regression on claim-level outcomes
with random intercepts for scenario and domain, condition as fixed effect;
cluster bootstrap at the **domain** level for cross-domain claims; report the
per-scenario bootstrap as a secondary, clearly-labelled analysis.

**No compute.** ~2 days of writing, and it is the difference between a
confirmatory result and an exploratory one.

---

### PHASES 6–10 *(parallel after 1/2/3/5)*

| Phase | Scope | Notable design point | Compute |
|---|---|---|---|
| **6 Explainer robustness** | GNNExplainer, SHAP, Integrated Gradients, saliency, perturbation, **random-subgraph**, **uniform-feature** | Existing SHAP result (Jaccard 0.022, identical outcomes) suggests the current metric may be **insensitive to explanation content**. The random-subgraph and uniform-feature baselines are the diagnostic: if they also score ~1.0 precision, that is the finding, and it is a negative one about the metric. Must match evidence *quantity, confidence, and formatting* across explainers, or the comparison is confounded. | Medium |
| **7 Cross-domain** | Leave-one-domain-out, protocols A–D | Held-out domain touched for **nothing** — not prompts, thresholds, failure categories, examples, or which results to highlight. Enforced by a config flag that makes held-out data unreadable until the confirmatory run. Language: *"transferred across the tested graph-based dynamical systems."* Never "domain-agnostic." | High |
| **8 Cross-model** | llama3.1:8b, mistral:7b, **llama3.2:3b** (smaller), **gemma2:9b** or **phi3:medium** (stronger), + a **deterministic template baseline** | All five are already installed locally (verified). `qwen2.5:7b` in `configs/ablations.yaml` is **not** — fix or drop that reference. The template baseline is the floor: a non-LLM renderer of the same evidence, which by construction has unsupported rate 0. It bounds how much of D's benefit is "the LLM" vs "the evidence." | High |
| **9 Adversarial** | 14 conditions from the brief | Overlaps E/F/M in Phase 5 — build once, reuse. `D_CONTRA` already covers "correct evidence + misleading prose" and found **no** degradation (halluc 0.000, 93/93); the untested direction is corrupting the **evidence** rather than the context. | Medium |
| **10 Counterfactual** | Rerun-verified CF evaluation | Report **validity, faithfulness, minimality, plausibility separately.** Current n=20 reports narration F1 1.000 *on the 8 valid cases only* — the 12 infeasible ones are excluded from that mean. Reporting "F1 1.000" without "validity 40%" adjacent would be misleading, and the brief is right to flag it. Add an **intervention-validity filter** so physically impossible interventions are rejected before scoring. | Medium |

---

### PHASE 4 — Blinded human annotation

**Purpose:** the *only* independent check on the deterministic verifier. Without
it, "unsupported" means whatever the code says it means.

- `annotation/guidelines.md` — operational definitions of the 8 categories, each
  with ≥2 positive and ≥2 negative worked examples drawn from **real** outputs.
- `annotation/annotation_form.json`, `annotation/annotator_examples.json`.
- `scripts/compute_interrater_agreement.py` — Cohen's κ (2 annotators) and
  Krippendorff's α (≥3, or missing data), per-category agreement, disagreement
  dump, adjudication log, **human-vs-verifier agreement**.
- `docs/HUMAN_EVALUATION.md`.

**Blinding:** condition, model, and "is this the proposed method" are stripped;
items are shuffled with a seeded permutation; the mapping is held in a separate
file the annotators cannot see. Annotators include at least one person not
involved in building the system.

**Realistic scoping:** ≥2 annotators × ~150 explanations is a real time ask. Target
a stratified subset covering every condition, not a random sample — otherwise rare
conditions get 2 items each and per-category agreement is uncomputable.

**Gate:** κ or α ≥ 0.6 on the pilot (~30 items) **before** the main annotation
round. If agreement is below that, the guidelines are not yet operational —
revise and re-pilot rather than proceeding. **Report the achieved agreement
whatever it is.**

---

### PHASE 13 — Simulated-decision audit *(can start immediately; analysis only)*

Not a new experiment — an audit of the existing n=444 result, answering each
question in the brief from the artifacts. Two findings are already visible and
should be confirmed:

1. **RAW consistency = 0.998** — the prediction-only condition picks essentially
   one action always. Part of XTRAFFIC's margin may be "breaks a degenerate
   constant policy" rather than "explanation helps."
2. **The simulator was calibrated on the scenario pool** (`budget=45`, `cap=24`
   chosen by sweeps until the ground-truth distribution looked non-degenerate).
   The mechanism is physical, which mitigates but does not eliminate the concern.
   The honest statement is that ground truth and calibration share a pool.

Missing comparison arms the brief asks for and that don't exist yet: **random
explanation**, **unstructured explanation**, **oracle explanation**. Adding at
least random-explanation is cheap and is the control that separates "explanation
content helps" from "any text in the prompt helps."

**Claim discipline:** this is a *simulated internal-consistency* result. It cannot
support any real-world traffic-improvement claim, and the delay-reduction units
are model-internal mph-deficit, not minutes.

---

### PHASE 12, 14, 15 — Matrix, freeze, outputs

- **12** `docs/CLAIM_EVIDENCE_MATRIX.md` — every claim → the experiment, dataset,
  model, explainer, n, seeds, method, effect size, CI, limitations, and
  **exploratory vs confirmatory**. Everything currently in the repo enters this
  matrix as **exploratory**, because no analysis plan preceded it. That is not a
  demotion; it is the accurate label, and having an explicit confirmatory tier is
  worth more than pretending the earlier work was one.
- **14** Freeze splits, prompts, model digests, evaluation code, endpoints →
  `configs/final_frozen_experiment.yaml` + `results/final_manifest.json`. Run
  `scripts/run_final_experiment.py` **without protocol changes.** Exploratory and
  confirmatory results live in separate directories and are never merged in a
  table without a column distinguishing them.
- **15** `docs/FINAL_LIMITATIONS.md`, `docs/CLAIMS_NOT_SUPPORTED.md`,
  `docs/REPRODUCTION_GUIDE.md`, `docs/AI_USAGE_LOG.md`, plus the final reporting
  table in the brief's column format.

---

## 4. Compute budget — approval needed before anything expensive

| Phase | LLM calls | Wall-clock (local Ollama, observed rates) | Needs approval? |
|---|---|---|---|
| 1 | ~50 | < 1 h | No |
| 3 | **0** | < 1 h | No |
| 2 | 0 LLM; ~4 training runs | Hours | No |
| 5 smoke | ~140 | ~1 h | No |
| **5 full** | **~6,300 per model-domain** | **tens of hours per cell** | **YES** |
| 6 | ~2,000 + explainer solves (SHAP is slow) | High | **YES** |
| 7 | ~4,000 | High | **YES** |
| 8 | ~6,000 across 5 models | High | **YES** |
| 9 | ~2,000 | Medium | **YES** |
| 10 | ~500 + model reruns | Medium | Probably |
| 4 | 0 (human time) | Days of human time | **YES** (people) |
| 13 | 0–500 | Low | No for the audit |
| 14 | Full confirmatory rerun | Highest | **YES** |

**Commitment:** I will run smoke tests freely and will not launch anything in the
"YES" column without explicit approval, per the implementation rules.

---

## 5. What I recommend you approve first

**Phases 1 + 3 only.** Together they are ~5 days of engineering, near-zero
compute, cannot invalidate anything existing (both are additive; Phase 3 runs on
artifacts already on disk), and they unblock everything else. They also produce
three cheap, genuinely informative analyses along the way, with no new LLM calls:

1. **How many condition-B "hallucinations" were actually empty outputs?**
   Recoverable from `advisory_error` and `n_cited_causes` in the committed CSV.
   This directly affects the headline 83.9% figure.
2. **What does the new multi-type verifier say about the ~100 surviving
   full-text advisories?** First evidence on whether the effect survives checking
   edges, paths, and temporal claims — the claim types nothing currently checks.
3. **How much does the rapidfuzz→difflib fallback move the numbers?** A one-line
   environment change, rerun the resolver gates, report the delta. If it is
   material, `rapidfuzz` must become a hard pin rather than an optional import.

If any of those three changes the picture materially, it is far better to know
before spending tens of hours of LLM compute on the ablation matrix.

---

## 6. Standing commitments

- No experiment is reported as run unless it ran, and no number is written that
  did not come out of an artifact.
- Failed and null results are reported with the same prominence as positive ones.
- No LLM is used as the sole judge of whether another LLM hallucinated.
- Existing work is not deleted. Superseded artifacts are **marked** superseded,
  in the artifact, not just in prose.
- Scientific decisions get options and tradeoffs, not a silent choice.
- Every phase gets its own branch and commit; every AI-assisted contribution is
  recorded in `AI_USAGE_LOG.md`.
- Scope and interpretation stay yours.
