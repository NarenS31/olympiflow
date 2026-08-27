# Literature check

Done before `report.md` was written. Four areas, as scoped. For each entry: what
it did, and what it did **not** do — the second column is what the novelty claim
has to survive.

Sources were read as abstracts/landing pages; where a claim below depends on a
detail I could not verify from that level, it says so rather than guessing.

---

## 1. Irrelevant or random context, and citation evaluation

**Shi, Chen, Misra, Scales, Dohan, Chi, Schärli, Zhou (2023), "Large Language
Models Can Be Easily Distracted by Irrelevant Context", ICML 2023.**
*Did:* introduced GSM-IC, grade-school arithmetic problems with an irrelevant
sentence added, and showed that a single irrelevant sentence substantially
degrades accuracy on problems the model otherwise solves. Showed mitigations —
self-consistency decoding, and an explicit "ignore irrelevant information"
instruction.
*Did not:* touch attribution or citation. Irrelevant context there is a
DISTRACTOR beside a correct answer, and the measured outcome is task ACCURACY.
Nothing in it asks what happens when the context the model is instructed to cite
is itself the noise, nor whether a grounding metric can tell.

**Gao, Yen, Yu, Chen (2023), "Enabling Large Language Models to Generate Text
with Citations" (ALCE), EMNLP 2023.**
*Did:* built the first reproducible benchmark for citation generation (ASQA,
QAMPARI, ELI5) and automatic metrics along three axes — fluency, correctness,
and citation quality (citation precision and recall, judged by an NLI model).
Established that correctness and citation quality are SEPARATE axes and reported
them separately; found strong systems still leave ~50% of ELI5 statements
unsupported.
*Did not:* run a condition in which the retrieved passages are randomised or
mismatched while the citation metric is still applied. Its correctness axis is
anchored to gold answers, so ALCE can already SEE a wrong-but-well-cited answer.
Our setting has no gold "correct explanation" to anchor to — the explainer output
IS the reference — which is exactly the configuration ALCE's design avoids.

**Rashkin, Nikolaev, Lamm, Aroyo, Collins, Das, Petrov, Tomar, Turc, Reitter
(2023), "Measuring Attribution in Natural Language Generation Models" (AIS),
Computational Linguistics 49(4).**
*Did:* gave a formal definition of attribution — "According to [source],
[statement]" — with explicatures for context-dependence, plus a two-stage human
annotation pipeline, instantiated on conversational QA, summarisation, and
table-to-text.
*Did not:* claim attribution implies correctness — it explicitly separates them,
and says so. That separation is stated as a definitional property. It is not
accompanied by an experiment showing a system scoring high on attribution while
its source carries no information, which is the empirical form of the same point
and is what Part 1 supplies in this domain.

**Read-across.** The attribution-vs-correctness distinction is well established
in text QA and is not ours. What the area has not done, as far as this check
found, is instrument it on an XAI→LLM pipeline where the cited source is a
learned model artifact rather than a retrieved document, and where there is no
independent gold reference to fall back on.

---

## 2. LLM narration of GNN explanations

**Cedro & Martens (2024), "GraphXAIN: Narratives to Explain Graph Neural
Networks", arXiv 2411.02540 (later in an xAI-2025 volume).**
*Did:* model- and explainer-agnostic pipeline turning an explanatory subgraph
plus feature-importance scores into a natural-language narrative via an LLM.
Evaluated by a survey of ML researchers/practitioners on understandability,
satisfaction, convincingness and suitability; 95% found it a valuable addition.
*Did not:* measure whether the narrative is faithful to the subgraph by any
automatic metric, and did not test the narrator against a corrupted or random
subgraph. Its evaluation is human preference, which is precisely the axis a
fluent narration of noise would be expected to pass.

**"Natural Language Counterfactual Explanations for Graphs Using Large Language
Models" (arXiv 2410.09295, 2024)** and **"Explaining Graph Neural Networks with
Large Language Models: A Counterfactual Perspective for Molecular Property
Prediction" (arXiv 2410.15165, 2024).**
*Did:* used LLMs to render graph counterfactuals in natural language; the
molecular one reports more faithful counterfactuals with consistently lower
proximity than baselines.
*Did not:* score the LLM's narration against the explainer's own attribution set,
and did not include a randomised-explanation control.

**"From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and
Graph Context" (arXiv 2508.07117, 2025)** and **"How good is my story? Towards
quantitative metrics for evaluating LLM-generated XAI narratives" (arXiv
2412.10220, 2024).**
*Did:* the latter is the closest thing found to an automatic quality metric for
LLM-generated XAI narratives, and it explicitly discusses GraphXAIN.
*Did not:* (flagged as unverified — judged from abstract only) appear to include
a noise/placebo explanation arm. I did not read the full paper, so treat this as
"not found" rather than "does not exist".

**NOTE on "LOGIC".** The brief named LOGIC as a paper to check. I could not
identify a paper by that name in this area from the searches run; the acronym
collides with a great deal of unrelated work. Recorded as **not checked** rather
than silently dropped. If you have the citation, it should be added before
submission — it is the single most likely place a prior noise-control could be
hiding.

---

## 3. Explainer instability and evaluation

**Sanchez-Lengeling, Wei, Lee, Reif, Wang, Qian, McCloskey, Colwell, Wiltschko
(2020), "Evaluating Attribution for Graph Neural Networks", NeurIPS 2020.**
*Did:* built synthetic graph benchmarks with computable ground-truth
attributions and used them to benchmark attribution methods quantitatively.
*Did not:* address the case where no ground-truth attribution exists — which is
every real traffic network, including ours.

**Faber, Moghaddam, Wattenhofer (2021), "When Comparing to Ground Truth is
Wrong: On Evaluating GNN Explanation Methods", KDD 2021.**
*Did:* showed the standard evaluate-against-ground-truth pipeline is flawed
because the trained GNN need not use the ground-truth edges; separated the
RESPONSIBLE motif, the CAUSING motif (what the model actually used), and the
EXPLANATORY motif (what the explainer found).
*Did not:* extend the argument downstream to a consumer of the explanation. Their
mismatch is explainer-vs-model; ours is a metric that cannot distinguish an
explanation from noise, one layer further along.

**Agarwal, Zitnik, Lakkaraju (2022), "Probing GNN Explainers: A Rigorous
Theoretical and Empirical Analysis of GNN Explanation Methods", AISTATS 2022;
and Agarwal et al. (2023), "Evaluating explainability for graph neural
networks", Scientific Data 10.**
*Did:* first theoretical analysis of GNN-explainer reliability — upper bounds on
violations of faithfulness, stability and fairness-preservation via data
processing inequalities, total variation distance and Lipschitz continuity —
plus large empirical studies and a benchmark dataset suite.
*Did not:* propose a per-explanation, computable-at-inference diagnostic for
whether a particular explanation is stable. Their stability is measured by
perturb-and-recompute, which costs a second explainer solve; that cost is the
gap Part 4 is aimed at.

**Also relevant, found in passing:** Zorro (arXiv 2105.08621) explicitly targets
"valid, sparse and stable" GNN explanations and criticises soft masks; Funke et
al. and the GNNExplainer "introduced evidence" critique make the same point about
soft masks not being discrete. These support our flatness observation but do not
turn it into a diagnostic.

---

## 4. Mask entropy as a reliability proxy

**Ying, Bourtsoulatze, Zitnik, Leskovec (2019), GNNExplainer, NeurIPS 2019.**
*Did:* used element-wise entropy of the mask as a REGULARISER in the objective,
to push mask values toward 0/1, alongside a size penalty. Entropy is a term in
the loss.
*Did not:* read entropy back off the SOLVED mask as a diagnostic, and did not
relate it to the stability of the resulting top-k. In the original method,
entropy is something you minimise during optimisation, not something you measure
afterwards to decide whether to trust the result.

**PEEK / entropy-centric XAI (e.g. "Entropy-Centric Explainable AI for Remote
Sensing Image Segmentation", arXiv 2608.11064; "A Comparative Analysis of XAI
Techniques", Entropy 28(5)).**
*Did:* computed Shannon entropy over CNN feature maps as an explanation-quality
evaluation metric, in vision.
*Did not:* apply to graph explainers or to node masks, and did not validate
entropy against a resampling-based stability measure of the same explanation.

**Max-Sensitivity and the stability-metric line (Yeh et al.; the ASTESJ and
LWDA-24 stability-metric papers found in search).**
*Did:* measured explanation stability directly, as the maximum change in the
explanation under a small input perturbation.
*Did not:* offer a single-solve proxy — they all require recomputing the
explanation, which is the expense we are trying to avoid.

**"The Query Channel: Information-Theoretic Limits of Masking-Based
Explanations" (arXiv 2604.16689).**
*Did:* an information-theoretic treatment of masking-based explanation, trading
query budget, sparsity and resolution, with mutual-information estimates for when
reliable explanation is possible at all.
*Did not:* (unverified — abstract only) provide a per-instance entropy threshold
calibrated against measured stability. This is the nearest theoretical neighbour
found and should be read in full before any claim is submitted.

**Verdict for this area.** No prior work was found that reads the entropy of a
solved GNNExplainer node mask and validates it as a predictor of that mask's own
top-k stability. The ingredients all exist separately; the combination was not
found. Caveat: two of the nearest neighbours were assessed from abstracts only.

---

## Novelty claim, written as narrowly as the list permits

REVISED after the results were in. The first draft of this sentence also claimed
the entropy proxy worked. It does not, at the sample size that matters, so the
clause has been removed rather than softened — see point 3.

> On a spatial-temporal GNN traffic pipeline, we show that an
> explanation-to-narration faithfulness metric, a closed-loop grounding
> corrector, and a downstream decision agent are all insensitive to whether the
> node-importance mask they consume is the real one or a seeded-random
> substitute, while the same pipeline's no-explanation control collapses.

**Flagged for your decision.** Four things to weigh before this goes in a paper:

1. **The attribution-vs-correctness point is not new** — AIS states it
   definitionally and ALCE separates the axes by construction. What appears to be
   new is the *empirical demonstration on an XAI→LLM pipeline with no gold
   reference*, plus the loop and decision arms. The sentence above is written to
   claim only that. If you want a stronger claim, it needs the "LOGIC" citation
   resolved and the two abstract-only papers read in full.
2. **"Insensitive" is a null result across three arms.** It is evidence that
   these metrics do not discriminate, not proof that no metric could. Phrasing it
   as "the metric measures citation compliance, not evidential content" is
   defensible; phrasing it as "explanations are useless" is not, and the numbers
   do not support it. The B control matters here and should be reported beside
   it: the same metric DOES collapse when the explanation block is removed
   entirely (F1 0.725 -> 0.090), so it is not simply broken.
3. **The entropy proxy is dropped from the claim.** It was the more clearly novel
   half and it did not survive. Two statistics are in play: the entropy of the
   24-window MEAN mask, and the mean of the per-window entropies. Only the first
   predicts split-half stability at n=207 (rho -0.305, CI [-0.434, -0.171]); the
   per-window one — the only one computable from a SINGLE solve, and therefore
   the only one that would make this a cheap diagnostic — falls to rho -0.120
   with a CI spanning zero. A proxy that needs 24 solves to predict the stability
   of a 24-solve average is not a proxy. Report it as a negative result.
4. **Everything here is one model, one dataset, one explainer, one LLM.** The
   Phase-11 cross-model axis was not repeated, so "the metric" means this
   metric as implemented here.
