# What the ST-GNN's explanations say about the METR-LA road network

**Status: EXPLORATORY.** This describes what the trained model's explainer
leans on and whether that survives being recomputed on disjoint windows. It
is not a confirmatory result, there is no ground truth anywhere in it, and
nothing here is a discovery claim. The only external checks are the geometry
of the adjacency, sensor coordinates, and raw-speed pathologies.

**No LLM was called at any point in this analysis.**

| | |
|---|---|
| Stage-1 run | `20260824T234653Z__influence_graph_solves__83787796__d01bffd8` |
| Targets solved | 4 of 207 |
| Windows per target | 4 (shared across every target) |
| Solves | 16 |
| Checkpoint | `models/gnn/checkpoints/metr_la_best.pt`, horizon 30 min |
| Explainer | 200 steps, seed 0, confidence reruns 0 |
| Device | cpu (4 workers x 1 thread) |
| Sampling seed | 42 |
| Regimes | congested < 45.0 mph, free-flow >= 55.0 mph, dead band between |

## Three setup facts that constrain every number below

**1. The adjacency is a distance threshold, not "the road network".**
`gaussian_kernel_adjacency` computes `A_ij = exp(-(d_ij/sigma)^2)` and zeroes
anything below kappa = 0.1. With sigma = 2584.5 m that is exactly the rule *keep the
pair iff directed road distance <= 3921.7 m*. So "off-adjacency" below means
">3922 m by road" and nothing more.

**2. Road distance is mostly missing off-adjacency.** The distance file covers
27.1% of ordered pairs overall and only 24.4% of non-adjacent ones. Haversine
is carried in a separate column and nothing is derived from it.

**3. The model's spatial receptive field is the complete graph in one hop.**
`XTrafficSTGNN` diffuses over two supports, both row-softmaxes, both 100% dense
(42,849 / 42,849 strictly positive entries each). `gcn_order` x `n_blocks` =
8 bounds diffusion at 8 hops over the *physical* component only; through
either learned support any node reaches any other immediately. **Hop distance is
therefore a descriptive coordinate, not a reachability limit.** Influence at hop
> 8 or at an unreachable pair is expected by construction and is not, on its
own, anomalous. It is reported separately because it was asked for, and because
"how road-aligned is the learned structure" remains a real question.

## Headline: the explainer's mask is flat, not peaked

Before any comparison to the road network, the most consequential property of
the explainer's output is how much of the graph it needs in order to account for
a prediction. Sources required to cover 80% of a target's off-self importance
mass, out of 206 available:

**free flow** — mean **148.3** +/- 2.9, median **148**, range 145-152;
percentiles p5 145 / p25 146 / p50 148 / p75 150 / p95 151.

A peaked explanation would need a handful of sources. This one needs most of
the network, and the spread across targets is narrow — flatness is a property
of the method here, not a few pathological targets. It is why the two
threshold rules below disagree so sharply: `top-k` imposes a peak the mask
does not have, while the cumulative-mass rule reports how little structure
there is to find. The committed explanation JSONs keep `top_nodes` = 8 of
207, so every downstream consumer in this repository has been reading the
first 8 entries of a nearly flat ranking.

Per-target values are in `metrics.json` under
`per_regime.<regime>.sources_for_mass_frac.per_target`; the distribution is
plotted in `fig4_mask_flatness.pdf`.

## Step 1 — learned influence vs the given adjacency

W is [targets x sources]; an edge is the ordered pair (source, target) and is
checked against `A[source, target]`, matching `GraphConv._nconv`. A target's row
is built only if it has >= 4 windows in that regime; targets below the floor are
counted, never silently dropped.

### Regime: congested

- Active targets: **0** (4 fell below the 4-window floor)
- No target met the floor in this regime; nothing further is reported.

### Regime: free flow

- Active targets: **3** (1 fell below the 4-window floor)
- Windows per active target: mean 4.0, range 4-4
- **Diagonal (target's own mass share of its row):** mean 0.0125, median 0.0147,
  range 0.0079-0.0148. Uniform-share reference 1/N = 0.0048.

**Effective edge set vs adjacency**

| Rule | \|E\| | Precision | Recall | Jaccard | Off-adjacency |
|---|---|---|---|---|---|
| top-k (k=8) | 24 | 0.250 | 0.222 | 0.133 | 18 |
| cumulative mass 80% | 445 | 0.058 | 0.963 | 0.058 | 419 |

Reference edge count for these targets: 27. The cumulative-mass rule needed
**148.33 sources per target on average** to cover 80% of the off-self mass,
out of 206 available — a direct measure of how flat the learned mask is. A
peaked mask would need a handful; the two rules are reported together
precisely because that gap is informative.

**Baselines** (same targets, same k)

| Method | \|E\| | Precision | Recall | Jaccard |
|---|---|---|---|---|
| **effective top-k** | 24 | **0.250** | 0.222 | 0.133 |
| (a) uniform random | 24 | 0.000 | 0.000 | 0.000 |
| (b) degree-matched random | 24 | 0.000 | 0.000 | 0.000 |
| (c) nearest by road distance | 24 | 1.000 | 0.889 | 0.889 |

Overlap between the effective set and the nearest-by-road set (Jaccard): 0.143.

**Hop-distance stratification of the effective edge set**

| Hop bucket | Effective edges | Share | Adjacency edges in same bucket |
|---|---|---|---|
| 1 | 6 | 0.250 | 27 |
| 2 | 2 | 0.083 | 0 |
| 3 | 4 | 0.167 | 0 |
| 4 | 3 | 0.125 | 0 |
| 5 | 2 | 0.083 | 0 |
| 6 | 2 | 0.083 | 0 |
| 7 | 1 | 0.042 | 0 |
| >8 | 3 | 0.125 | 0 |
| unreachable | 1 | 0.042 | 0 |

**Candidate set — beyond 8 hops or unreachable:** 4 edges (0.167 of the
effective set), of which 1 are unreachable in the adjacency graph.
Full listing in `beyond_k_edges.csv`. Per fact (3) this is not by itself
anomalous — both learned supports connect every pair directly.

**Split-half stability** (2 vs 2 windows, stratum-balanced, >= 2 windows
per half; 3 targets qualify in both halves)

| Subset | Jaccard between halves |
|---|---|
| all effective edges | **0.091** |
| off-adjacency edges only | **0.051** (20 vs 21 edges) |
| beyond-8/unreachable only | **0.000** (3 vs 3 edges) |
| *reference: two independent uniform random draws* | 0.043 |

The effective edge set replicates across halves at 0.091, against 0.043 for
two independent random draws.

Off-adjacency edges specifically replicate at 0.051.

## Every edge above threshold

No curation and no selected examples. Complete listings:

- `effective_edges.csv` — every (source, target) above threshold, both regimes,
  with importance, adjacency membership, hop distance, road distance (or blank),
  haversine in its own reference-only column, and both endpoints' coordinates.
- `off_adjacency_edges.csv` — the off-adjacency subset.
- `beyond_k_edges.csv` — the beyond-8/unreachable subset.
- `hop_strata.csv`, `baselines.csv`, `per_target_mass.csv`.

## The explainer's W against the model's own learned graph

Both objects were produced by the same checkpoint and both get read as "which
sensors matter here", but they are not the same kind of thing: `W` is a
behavioural attribution for specific predictions, `A_sem` is a static parameter
that enters only through the `(1 - alpha)` term of one of two supports.
Agreement is not required and disagreement is not an error.

| Regime | targets | Spearman mean | Spearman median | top-8 J mean | top-8 J median | targets with zero top-8 overlap |
|---|---|---|---|---|---|---|
| free flow | 3 | 0.0007 | -0.0735 | 0.0444 | 0.0667 | 1 |

Per-target values in `metrics.json` under
`per_regime.<regime>.vs_learned_semantic_graph.per_target`. The learned
graph's own properties are reported separately in the Stage-0b run
(`learned_semantic_graph`), which needed no solves.

## Step 2 — implied propagation speed: NOT RUN

Skipped deliberately, on the evidence gathered in the feasibility check.

The explainer learns a node mask `m in [0,1]^N` — one scalar per node, applied
uniformly across all 12 input timesteps (`explain.py:111`). There is no
per-timestep dimension in the mask, the schema, or anywhere else in the
artifact, so there is no importance-weighted lag to compute. The schema's
`propagation_lag_minutes` is a single scalar per explanation, covering only
`top_nodes[0] -> target`, and it is obtained by cross-correlating the *input
window*: it is a property of the data, not of the model. Deriving implied
speeds from it would measure METR-LA's autocorrelation structure while
labelling the result as the model's propagation behaviour.

No traffic-flow reference values were looked up, because none were used.

## Step 3 — sensors whose influence mass sits farthest from their neighbourhood

Ranked by the share of off-self importance mass landing on non-adjacent sources.
Because adjacency *is* the 3922 m road-distance cutoff, an adjacency split and a
distance split are the same split; the mass is therefore decomposed three ways,
separating *far by road* from *no road path in the distance file*, since the
latter is an absence of data rather than a measurement of distance.

**No sensor is described as mislabelled.** The columns are evidence.

Top 20 of 3 ranked sensors (full table: `sensor_anomaly_ranking.csv`;
raw-speed defects for all 207: `sensor_raw_pathologies.csv`):

| target | sensor_id | lat | lon | in_degree | mass_off_adjacency | mass_on_far_road | mass_on_no_road_path | self_mass_share | missing_pct | longest_missing_run_steps | longest_constant_nonzero_run_steps | max_mph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0000 | 767541.0000 | 34.1162 | -118.2380 | 10.0000 | 0.9467 | 0.4548 | 0.4919 | 0.0148 | 6.2700 | 381.0000 | 3.0000 | 70.0000 |
| 2.0000 | 767542.0000 | 34.1164 | -118.2382 | 8.0000 | 0.9405 | 0.3347 | 0.6058 | 0.0079 | 6.2700 | 381.0000 | 3.0000 | 70.0000 |
| 0.0000 | 773869.0000 | 34.1550 | -118.3183 | 9.0000 | 0.9396 | 0.2925 | 0.6472 | 0.0147 | 11.2130 | 1118.0000 | 3.0000 | 70.0000 |

Cross-check on the top 20: missingness mean 7.92 vs 7.92 for all sensors;
longest constant non-zero run mean 3.0 vs 3.0 steps.

Road assignment is not reported: `node_meta.json` carries only `dataset`,
`sensor_ids`, `n_nodes` and `latlon`. There is no freeway or road-name field in
any committed artifact, and resolving PeMS station ids to freeways needs an
external source that is not available offline.

## What this run says about the explainer's optimum

Two measurements taken around this analysis bear on how much weight any single
explainer solve can carry. Both are about the optimisation, not about hardware
or about a dry-run's bookkeeping.

**The solution reorders under float-level perturbation.** Same seed, same
input, same checkpoint, two devices (run `20260824T235643Z__device_divergence__58efae64__ee9a7944`):

| Window / target | Pearson (magnitude) | Spearman (rank) | Top-8 Jaccard | Top-20 Jaccard |
|---|---|---|---|---|
| w100 / t50 | 0.9805 | 0.5886 | 0.455 | 0.379 |
| w2500 / t120 | 0.8533 | 0.2790 | 0.143 | 0.212 |
| w4200 / t7 | 0.9240 | 0.8658 | 0.600 | 0.481 |
| **mean** | **0.9193** | **0.5778** | **0.399** | **0.358** |

Within a device the same seed reproduces byte-identically, so this is not
randomness in the algorithm. The two devices agree on *how much* mass the
mask assigns and disagree on *which nodes* receive it. Since every
downstream consumer in this repository reads the top-k node set and not the
magnitudes, that is the axis that matters. `docs/REPOSITORY_AUDIT.md` 5.7
already notes cross-device bitwise identity is impossible; this quantifies
what it costs on the quantity the project actually uses.

**A pre-run dry-run over the 12 committed explanations scored below chance.**
Aggregating their top-8 lists and comparing to the adjacency gave precision
0.013, against a chance rate of about 0.035 for drawing 8 sources uniformly at
a mean in-degree of 7.3. At 12 explanations that is not a result, and it is
recorded here only because it set the expectation the full run was built to
test, rather than being discovered afterwards.

Both observations point the same way and are consistent with two committed
numbers the project already holds: Fidelity- is inconclusive (12.93 vs 12.84
random) and Phase-18 mean explanation stability is 0.427 across K=10 seeds.

## Limitations

**"Off-adjacency" means ">3922 m by road".** The adjacency is not a road map;
it is a thresholded Gaussian kernel on directed road distance with sigma = 2584.5 m
and kappa = 0.1, which reduces exactly to that cutoff. Every statement about the
model "using edges outside the graph" is a statement about a 3.9 km radius, and
carries no implication that the pair is unconnected in the real road network.
Relatedly, **76% of non-adjacent ordered pairs have no directed road distance
in `distances_la_2012.csv` at all**, so most off-adjacency edges cannot be given
a road distance. Haversine is reported for them in a separate column and nothing
in this analysis is derived from it; straight-line distance understates road
distance by an unknown and non-constant factor.

**Hop distance does not bound influence.** Both of the model's supports are dense
row-softmaxes, so every pair is one hop apart inside the network regardless of
the adjacency. The hop-stratified tables describe where influence lands relative
to the road graph; they do not identify influence the architecture could not have
produced.

**The window sample is small per target.** 4 shared windows with a ~14% network
congestion base rate leaves the median target only a handful of congested
windows. Pooled across targets the edge-set comparison has ample observations,
but individual rows of W are thin and the per-target rankings in Step 3 should be
read as ordering, not as measurement.

**Single explainer solve per (target, window), and the solve is not stable.** See
the section above. The aggregate is what is reported; the split-half Jaccard is
the honest measure of how much of it survives resampling.

**Alignment, not correctness.** This measures what the explainer says the model
relies on. It does not establish that the model's reliance is correct, nor that
the explainer recovered the model's true dependence. Fidelity+/- is the axis that
would speak to that and it is not rerun here.

**One checkpoint, one city, one horizon.** epoch-54 `metr_la_best.pt`, METR-LA,
30-minute forecast. Nothing here transfers without being rerun.

**Exploratory.** Thresholds (top-k = 8, cumulative mass 80%, congested < 45.0 mph,
free-flow >= 55.0 mph, window floor 4) were chosen before the run from the data
distribution and the project's existing conventions, but there is no
prespecified analysis plan for this study, so no result in it is confirmatory.

