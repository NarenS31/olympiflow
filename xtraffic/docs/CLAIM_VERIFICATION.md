# XTraffic — Deterministic Claim Verification (Phase 3)

**Status:** implemented and gate-passed 2026-08-22.
**No LLM is used to judge whether another LLM hallucinated.** Every verdict is a
decidable function of the explanation JSON, the adjacency matrix, and the node
table.

---

## 1. What the committed metric measured

`evaluation/faithfulness.py`, per advisory:

```
precision     = |citations whose resolved node set intersects top-k| / |citations|
hallucination = 1 - precision          and = 1.0 when there are NO citations
```

One unit of analysis: an entry in the LLM's own `cited_causes[]` array — a field
the model was *instructed* to fill. Three consequences:

1. **The prose was never read.** `reasoning` could assert a reversed edge, an
   invented propagation order, or a wrong lag and score hallucination 0.000.
2. **Overreach and falsehood were merged.** "Cited a real node that wasn't
   selected" and "asserted the edge runs the other way" both scored as
   `1 - precision`.
3. **A refusal scored 1.000** — identical to confidently inventing four causes.

The explanation JSON contains `top_edges`, `propagation_path`, and
`propagation_lag_minutes`. All three are rendered into the prompt. **None was
ever checked.**

---

## 2. What Phase 3 adds

```
evaluation/claim_types.py             the vocabulary: 10 claim types, 5 verdicts
evaluation/claim_parser.py            advisory -> atomic typed claims (+ unparsed)
evaluation/graph_claim_verifier.py    deterministic per-type verification
evaluation/counterfactual_verifier.py claims needing a model RERUN
evaluation/claim_metrics.py           rates, with refusal as its own outcome
tests/test_claim_verifier.py          38 tests
tests/mutation_check.py               do those tests have teeth?
```

**Claim types:** node existence · edge existence · edge direction · path
existence · feature attribution · temporal · confidence · prediction ·
counterfactual · domain terminology.

**Verdicts:**

| verdict | meaning |
|---|---|
| `SUPPORTED` | the evidence entails the claim |
| `PARTIALLY_SUPPORTED` | part of a compound claim holds; or a quantity is close but outside tolerance |
| `UNSUPPORTED` | the evidence is **silent** — it neither entails nor denies |
| `CONTRADICTED` | the evidence entails the **negation** |
| `UNVERIFIABLE` | no deterministic procedure decides it from what we have |

`UNSUPPORTED` vs `CONTRADICTED` is the distinction the committed metric could not
draw. `UNVERIFIABLE` is **not a failure** and is excluded from precision: an
action proposal ("retime signals at 3rd and Main") is not a factual assertion
about the graph, and scoring it adversely would punish the model for doing the
task it was given.

**Headline:** `unsupported_claim_rate = (UNSUPPORTED + CONTRADICTED) / verifiable`,
where verifiable excludes `UNVERIFIABLE` — otherwise a condition could lower its
rate by talking about unverifiable things.

---

## 3. Two references, never averaged (Decision 1)

| reference | scored against | answers |
|---|---|---|
| `ALIGNMENT` | the explainer's output = **what the LLM was shown** | is it a faithful transcription? |
| `GROUND_TRUTH` | the generator's true causal structure (synthetic, Phase 2) | is it **true**? |

Alignment is the committed metric's reference. It measures transcription
fidelity, not correctness — and the project's own Phase-12 result proves the gap
is real: GNNExplainer and SHAP selected **near-disjoint** node sets (Jaccard
0.022) while both scored precision 0.982. Two mutually contradictory
explanations were both "perfectly faithful", which is only possible because
neither was checked against anything true.

An advisory can be perfectly aligned to a wrong explanation. That is a
meaningful, reportable state, so the two references are reported separately and
never combined.

---

## 4. Refusal semantics (Decision 2)

`hallucination = 1.0 if no cited causes` made the planned no-answer control
unscoreable. Now:

- refusals are detected (`claim_parser.is_refusal`), counted, and **excluded
  from the macro rates** — including them would score an abstention as a
  perfect advisory;
- `refusal_rate` is reported **alongside every headline number**, with a
  denominator of all advisories;
- **empty** output is tracked separately from **refusal**, because an empty
  advisory is usually a parse failure and folding it into abstention would hide
  that;
- `aggregate()` emits a warning whenever refusals are non-zero: *a lower
  unsupported rate bought by abstaining is not better grounding.*

---

## 5. The parser's honesty mechanism

Extraction is regex + the existing node/region tables. That buys determinism and
auditability at the cost of coverage, and **the cost is measured**: every
assertive sentence yielding no claim lands in `ParseReport.unparsed` and the
rate is reported.

This is not a footnote. A parser that silently drops what it cannot handle
flatters every condition **unequally** — prose-heavy conditions contain more free
text than structured ones, so an unparsed sentence is likelier in exactly the
condition the hypothesis predicts should do worse. Dropping them would
manufacture the predicted result out of a parser limitation. **A difference in
unparsed rate across conditions is a confound, and `aggregate()` warns above
20%.**

Reuse: locations resolve through `faithfulness.NodeTable` + `resolve_location` —
the same five-rung ladder the committed metric uses — so the Phase-3 gate
compares *verifiers*, not two different resolvers.

---

## 6. Testing, and whether the tests mean anything

```
python -m unittest xtraffic.tests.test_claim_verifier    # 38 tests, all pass
python -m xtraffic.tests.mutation_check                  # killed 9 / 9
```

A green suite proves nothing on its own. `mutation_check.py` sabotages one
critical branch at a time and checks the suite notices. **It earned its keep
immediately, three times:**

| finding | what happened |
|---|---|
| **Reversal detection was never exercised.** | Stubbing out the top-`edges` reversal check — the single capability the committed metric lacks, and the headline justification for this module — left all 33 tests green. The reversal test was being satisfied by the propagation-path ordering *fallback*. Fixed by adding an edge (3→1) with neither endpoint on the path, isolating the branch. |
| **The unparsed test was vacuous.** | Its fixture prose contained the word "load", which the feature regex matched, so a claim was produced and the weak assertion held even with `unparsed.append` stubbed out. Replaced with prose that cannot yield a claim. |
| **A genuine correctness bug.** | Chasing a third survivor showed the code flagged self-attribution whenever the target was merely *among* the resolved nodes. That contradicts the Phase-13 rule — self-blame requires resolving to the target *alone*; a region citation resolving to {target, neighbour} that misses top-k is a region miss. The redundant branch was removed. |

Mutation testing is a **diagnostic, not a gate**: some mutants are legitimately
equivalent, and the right response to a survivor is to look at it.

---

## 7. First evidence, and a bug it exposed

`python -m xtraffic.scripts.analyze_committed_results` — three analyses, no new
LLM calls.

### A. Are the committed "hallucinations" actually refusals? **No.**

| condition | n | reported halluc. | zero-citation | scored 1.000 | of those, empty |
|---|---|---|---|---|---|
| A | 93 | 0.005 | 0 | 0 | — |
| **B** | 93 | **0.828** | **0** | **72** | **0** |
| C | 93 | 0.005 | 0 | 0 | — |

All 72 condition-B advisories scoring hallucination 1.000 made **real citations
that all missed**. None was empty; none errored. **The concern raised in the
audit does not apply to this corpus** — the ungrounded rate is genuine
fabrication, not abstention. The refusal fix remains necessary for the *planned*
no-answer control, but it does not revise the committed number.

### B. Does the result survive checking more claim types? **Tentatively yes, n=6.**

Only 6 advisories retain full text, so this is indicative, not conclusive.

```
atomic claims extracted            93
node claims (the old metric's only 57
  claim type)
claims the old metric NEVER saw    36  (38.7%)

unsupported claim rate   macro 0.068 / micro 0.050
refusal rate             0.000     unparsed rate 0.042
```

Against the old metric's mean hallucination of ~0.083 on the same six, the
multi-type rate of 0.068 is comparable. **38.7% of the claim surface was
previously unexamined**, and checking it did not overturn the picture — on six
advisories. This is the strongest statement the surviving corpus supports.

One genuine detection the old metric missed, in `night.json`: the prose asserts
top contributors are in **"Downtown LA"** (nodes {3,4,5,6,15}, none in top-k
{11,28,55,66,79,96,108,183}), while the declared `cited_causes` say "N of
Downtown LA" / "NE of Downtown LA". The prose claims something the declared
causes do not. The old metric never read prose.

> **A bug this analysis exposed in the new verifier.** The first run reported
> macro 0.220. Inspecting the per-claim detail showed recommendation *sites* —
> "Sunset Blvd and Santa Monica Blvd", "I-5/SR-134/SR-2 interchange" — scored
> `CONTRADICTED` because they resolve to no sensor. They are perfectly sensible
> places to retime a signal and simply are not METR-LA freeway nodes. That was
> not merely wrong but **biased**: it would inflate the unsupported rate for any
> condition eliciting more recommendations, a property of the prompt rather than
> of grounding. Unresolvable recommendation sites are now `UNVERIFIABLE`; the
> rate fell 0.220 → 0.068. Found by reading real advisories, not by a unit test
> — which is why analysis B prints per-claim detail.

### C. Does the resolver backend matter? **Not materially, on this corpus.**

`rapidfuzz` is **not installed**, so every committed number was produced by the
`difflib` fallback — a different similarity function against the same threshold
of 82.

- Resolver gates reproduce **exactly** under difflib: METR-LA 29/30, PEMS-BAY
  32/32, Chicago 32/32, power grid 22/22 — identical to the committed values.
- Re-resolving **1,212 logged citations** under both functions: **11
  disagreements (0.91%)**, and in **all 11 the resolved node set is identical** —
  only the recorded *method label* changes (`fuzzy` → `geographic`), e.g.
  "Glendale/Burbank" → 49 nodes either way.

**Conclusion: the backend does not move the metric on this corpus.** The audit
flagged this as a live risk; measurement shows it is not one here. It still must
be recorded per run, because that conclusion is corpus-specific.

*Honesty:* `token_set_ratio` in that comparison is a faithful reimplementation of
the documented algorithm, **not rapidfuzz**. Confirming against the real library
needs `pip install rapidfuzz==3.6.1`, which the script deliberately does not do
to the environment.

---

## 8. Counterfactuals — four axes, never averaged

`counterfactual_verifier.py` **reruns the frozen model**:

| axis | question |
|---|---|
| VALIDITY | does the intervention actually flip the prediction on rerun? |
| FAITHFULNESS | does the narration match the counterfactual record? |
| MINIMALITY | would half the intervention have worked? |
| PLAUSIBILITY | is the intervention admissible at all? (rejected before validity) |

Phase 17 reports `mean_faithfulness_f1 1.000` next to `validity_rate 0.40`. The
F1 is computed over the **8 valid cases only** — it is *conditional on* validity,
not independent of it. `summarise()` therefore names the field
`faithfulness_among_valid`, always emits `faithfulness_denominator`, and attaches
a note stating that quoting it without `validity_rate` beside it would be
misleading. A smoke test reproduces the pathology exactly:
`faith_among_valid=1.0, denom=1`.

---

## 9. Limits — what this does **not** do

| limit | consequence |
|---|---|
| **Cannot be applied retrospectively at scale.** | ~3,900 of ~4,000 generations kept no text. Analysis B ran on **6**. Everything else needs regeneration under Phase-1 logging. |
| **Coverage is bounded by regex.** | Measured and reported as `unparsed_rate`, not hidden — but a claim the parser cannot type is a claim it cannot check. |
| **Alignment ≠ correctness.** | Until Phase 2 supplies ground truth, `GROUND_TRUTH` scoring has nothing to score against. |
| **Region-level credit is inherited.** | Kept for comparability with the committed metric, but it is what produced the Phase-11 PEMS-BAY artifact (a target's-own-region control scored F1 0.510 with zero causal content). `resolved_set_size` is recorded per verdict; **chance floors remain mandatory**. |
| **No human validation yet.** | Nothing establishes that these verdicts match human judgement of "unsupported". That is Phase 4, and it is the only independent check on this module. |
