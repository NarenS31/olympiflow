# What the ST-GNN's explanations say about the METR-LA road network

**Status: EXPLORATORY.** This describes what the trained model's explainer
leans on and whether that survives being recomputed on disjoint windows. It
is not a confirmatory result, there is no ground truth anywhere in it, and
nothing here is a discovery claim. The only external checks are the geometry
of the adjacency, sensor coordinates, and raw-speed pathologies.

**No LLM was called at any point in this analysis.**

| | |
|---|---|
| Stage-1 run | `20260824T235016Z__influence_graph_solves__73e75966__ee9a7944` |
| Targets solved | 207 of 207 |
| Windows per target | 24 (shared across every target) |
| Solves | 4968 |
| Checkpoint | `models/gnn/checkpoints/metr_la_best.pt`, horizon 30 min |
| Explainer | 200 steps, seed 0, confidence reruns 0 |
| Device | cpu (8 workers x 1 thread) |
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

**3. The model's spatial receptive field is the FULL GRAPH IN ONE HOP.**
`A_final = sigmoid(alpha)*A_phys + (1-sigmoid(alpha))*A_sem` is the support
every `STBlock` diffuses over. Measured on this checkpoint, it is
**structurally dense** (42849/42849 entries strictly positive, 100%) *and*
**effectively dense** — each row spreads over 107.1 of 207 nodes by
perplexity, 52% of uniform. The second (adaptive) support is denser
still, at 99% of uniform.
`gcn_order` x `n_blocks` = 8 bounds diffusion at 8 hops over the *physical*
component only. **So the far stratum below is NOT architecturally surprising:**
every pair is one hop apart inside the network, and hop distance in the kernel
adjacency is a geometric description of where influence sits, not a limit on
where it could sit. The stratum is named accordingly rather than as "beyond-K".

## Headline: the explainer's mask is flat, not peaked

Before any comparison to the road network, the most consequential property of
the explainer's output is how much of the graph it needs in order to account for
a prediction. Sources required to cover 80% of a target's off-self importance
mass, out of 206 available:

**congested** — mean **146.6** +/- 4.7, median **148**, range 136-155;
percentiles p5 137 / p25 143 / p50 148 / p75 150 / p95 153.

**free flow** — mean **154.0** +/- 2.9, median **154**, range 138-160;
percentiles p5 149 / p25 152 / p50 154 / p75 156 / p95 158.

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

- Active targets: **34** (173 fell below the 4-window floor)
- Windows per active target: mean 7.06, range 4-16
- **Diagonal (target's own mass share of its row):** mean 0.0080, median 0.0075,
  range 0.0048-0.0143. Uniform-share reference 1/N = 0.0048.

**Effective edge set vs adjacency**

| Rule | \|E\| | Precision | Recall | Jaccard | Off-adjacency |
|---|---|---|---|---|---|
| top-k (k=8) | 272 | 0.140 | 0.128 | 0.072 | 234 |
| cumulative mass 80% | 4986 | 0.050 | 0.838 | 0.049 | 4738 |

Reference edge count for these targets: 296. The cumulative-mass rule needed
**146.65 sources per target on average** to cover 80% of the off-self mass,
out of 206 available — a direct measure of how flat the learned mask is. A
peaked mask would need a handful; the two rules are reported together
precisely because that gap is informative.

**Baselines** (same targets, same k)

| Method | \|E\| | Precision | Recall | Jaccard |
|---|---|---|---|---|
| **effective top-k** | 272 | **0.140** | 0.128 | 0.072 |
| (a) uniform random | 272 | 0.037 | 0.034 | 0.018 |
| (b) degree-matched random | 272 | 0.055 | 0.051 | 0.027 |
| (c) nearest by road distance | 272 | 0.809 | 0.743 | 0.632 |

Effective-vs-uniform precision ratio: **3.80x**.
Overlap between the effective set and the nearest-by-road set (Jaccard): 0.067.

**Hop-distance stratification of the effective edge set**

| Hop bucket | Effective edges | Share | Adjacency edges in same bucket |
|---|---|---|---|
| 1 | 38 | 0.140 | 296 |
| 2 | 35 | 0.129 | 0 |
| 3 | 21 | 0.077 | 0 |
| 4 | 15 | 0.055 | 0 |
| 5 | 24 | 0.088 | 0 |
| 6 | 37 | 0.136 | 0 |
| 7 | 28 | 0.103 | 0 |
| 8 | 14 | 0.051 | 0 |
| >8 | 54 | 0.199 | 0 |
| unreachable | 6 | 0.022 | 0 |

**Far stratum — beyond 8 kernel radii (>8 chained hops of <= 3.9 km), or unreachable:**
60 edges (0.221 of the effective set), of which 6 are unreachable in the
adjacency graph. **Base rate for these targets: 0.239** — that share of all
candidate pairs is already in the stratum, so an edge set with no spatial
preference would show it. Enrichment over base rate: **0.92x**.

Per fact (3) this stratum is **not architecturally surprising**: the mixed
support the model diffuses over is dense, so every pair is one hop apart
inside the network. The stratum is a geometric description of where the
influence sits relative to the kernel adjacency, nothing more.
Full listing in `beyond_k_edges.csv`.

**Split-half stability** (12 vs 12 windows, stratum-balanced, >= 2 windows
per half; 28 targets qualify in both halves)

| Subset | Jaccard between halves |
|---|---|
| all effective edges | **0.140** |
| off-adjacency edges only | **0.122** (200 vs 176 edges) |
| beyond-8/unreachable only | **0.176** (48 vs 52 edges) |
| *reference: two independent uniform random draws* | 0.018 |

The effective edge set replicates across halves at 0.140, against 0.018 for
two independent random draws.

Off-adjacency edges specifically replicate at 0.122.

### Regime: free flow

- Active targets: **200** (7 fell below the 4-window floor)
- Windows per active target: mean 16.27, range 6-20
- **Diagonal (target's own mass share of its row):** mean 0.0133, median 0.0136,
  range 0.0063-0.0179. Uniform-share reference 1/N = 0.0048.

**Effective edge set vs adjacency**

| Rule | \|E\| | Precision | Recall | Jaccard | Off-adjacency |
|---|---|---|---|---|---|
| top-k (k=8) | 1600 | 0.286 | 0.315 | 0.176 | 1142 |
| cumulative mass 80% | 30805 | 0.043 | 0.922 | 0.043 | 29465 |

Reference edge count for these targets: 1453. The cumulative-mass rule needed
**154.03 sources per target on average** to cover 80% of the off-self mass,
out of 206 available — a direct measure of how flat the learned mask is. A
peaked mask would need a handful; the two rules are reported together
precisely because that gap is informative.

**Baselines** (same targets, same k)

| Method | \|E\| | Precision | Recall | Jaccard |
|---|---|---|---|---|
| **effective top-k** | 1600 | **0.286** | 0.315 | 0.176 |
| (a) uniform random | 1600 | 0.033 | 0.036 | 0.018 |
| (b) degree-matched random | 1600 | 0.034 | 0.038 | 0.018 |
| (c) nearest by road distance | 1592 | 0.756 | 0.828 | 0.653 |

Effective-vs-uniform precision ratio: **8.64x**.
Overlap between the effective set and the nearest-by-road set (Jaccard): 0.167.
(1 targets have fewer than 8 road-connected sources, so baseline (c)
contributes fewer edges for them. Haversine was not substituted.)

**Hop-distance stratification of the effective edge set**

| Hop bucket | Effective edges | Share | Adjacency edges in same bucket |
|---|---|---|---|
| 1 | 458 | 0.286 | 1453 |
| 2 | 123 | 0.077 | 0 |
| 3 | 115 | 0.072 | 0 |
| 4 | 131 | 0.082 | 0 |
| 5 | 155 | 0.097 | 0 |
| 6 | 159 | 0.099 | 0 |
| 7 | 133 | 0.083 | 0 |
| 8 | 124 | 0.077 | 0 |
| >8 | 145 | 0.091 | 0 |
| unreachable | 57 | 0.036 | 0 |

**Far stratum — beyond 8 kernel radii (>8 chained hops of <= 3.9 km), or unreachable:**
202 edges (0.126 of the effective set), of which 57 are unreachable in the
adjacency graph. **Base rate for these targets: 0.221** — that share of all
candidate pairs is already in the stratum, so an edge set with no spatial
preference would show it. Enrichment over base rate: **0.57x**.

Per fact (3) this stratum is **not architecturally surprising**: the mixed
support the model diffuses over is dense, so every pair is one hop apart
inside the network. The stratum is a geometric description of where the
influence sits relative to the kernel adjacency, nothing more.
Full listing in `beyond_k_edges.csv`.

**Split-half stability** (12 vs 12 windows, stratum-balanced, >= 2 windows
per half; 200 targets qualify in both halves)

| Subset | Jaccard between halves |
|---|---|
| all effective edges | **0.226** |
| off-adjacency edges only | **0.158** (1197 vs 1160 edges) |
| beyond-8/unreachable only | **0.124** (242 vs 229 edges) |
| *reference: two independent uniform random draws* | 0.017 |

The effective edge set replicates across halves at 0.226, against 0.017 for
two independent random draws.

Off-adjacency edges specifically replicate at 0.158.

## Every edge above threshold

No curation and no selected examples. Complete listings:

- `effective_edges.csv` — every (source, target) above threshold, both regimes,
  with importance, adjacency membership, hop distance, road distance (or blank),
  haversine in its own reference-only column, and both endpoints' coordinates.
- `off_adjacency_edges.csv` — the off-adjacency subset.
- `beyond_k_edges.csv` — the beyond-8/unreachable subset.
- `hop_strata.csv`, `baselines.csv`, `per_target_mass.csv`.

## Do the far edges that survive both halves sit high in the learned graph?

An edge that replicates across disjoint windows is the only kind worth asking
about. The question is whether those survivors rank near the top of the model's
own learned graph `A_sem` — if they do, the explainer is recovering the learned
adjacency. Three matched groups, all drawn from the same targets: survivors
(far edges in **both** halves), far edges in **one** half only, and far pairs
**sampled uniformly** per target from the same stratum.

Rank is the source's position among all 206 non-self sources for that target,
1 = strongest. Chance mean rank is 103.5; chance top-8 share is 0.0388.

### congested

Targets in both halves: **28**. Far edges half A / half B: 84 / 73.
**Survivors (in both): 15.** One-half-only: 127. Survival rate 0.106.

> **No regime-level claim is made for congested.** Only 28 targets have both
> halves populated, below the pre-set floor of 40. The numbers below are
> printed for completeness and should not be read as a result.

| Group | n | mean rank | median | top-8 | top-20 | top-half |
|---|---|---|---|---|---|---|
| **survivors (both halves)** | 15 | 131.8 | 155 | 0.000 | 0.067 | 0.400 |
| one half only | 127 | 112.7 | 115 | 0.079 | 0.158 | 0.457 |
| random from same stratum | 15 | 115.5 | 114 | 0.000 | 0.067 | 0.467 |
| *chance* | — | 103.5 | 103 | 0.039 | 0.097 | 0.500 |

All 15 survivors, with geometry, ranked by learned-graph rank
(`beyond_k_survivors_congested.csv`; road distance and haversine in separate
columns, nothing derived from haversine):

| src | tgt | src sensor | tgt sensor | hops | learned rank | road m | haversine m | src lat,lon | tgt lat,lon |
|---|---|---|---|---|---|---|---|---|---|
| 183 | 77 | 767495 | 765171 | 11 | 16 | — | 21383.4 | 34.1038, -118.2499 | 34.1679, -118.4690 |
| 183 | 84 | 767495 | 767053 | 12 | 33 | — | 21393.0 | 34.1038, -118.2499 | 34.1709, -118.4677 |
| 132 | 77 | 762329 | 765171 | 10 | 33 | — | 22347.0 | 34.1390, -118.2286 | 34.1679, -118.4690 |
| 203 | 84 | 717595 | 767053 | unreach | 79 | — | 26411.9 | 34.1416, -118.1829 | 34.1709, -118.4677 |
| 44 | 77 | 774012 | 765171 | 9 | 90 | — | 24724.0 | 34.1477, -118.2014 | 34.1679, -118.4690 |
| 30 | 88 | 773013 | 767350 | 12 | 101 | — | 25357.3 | 34.0837, -118.2208 | 34.1801, -118.4704 |
| 66 | 84 | 767610 | 767053 | 11 | 110 | — | 23063.5 | 34.1855, -118.2177 | 34.1709, -118.4677 |
| 105 | 88 | 718076 | 767350 | 10 | 155 | — | 22629.8 | 34.1634, -118.2253 | 34.1801, -118.4704 |
| 89 | 196 | 767351 | 771673 | 10 | 185 | — | 24973.6 | 34.1802, -118.4702 | 34.0779, -118.2287 |
| 89 | 29 | 767351 | 773012 | 12 | 189 | — | 25327.4 | 34.1802, -118.4702 | 34.0839, -118.2209 |
| 34 | 16 | 717819 | 771667 | 11 | 190 | — | 25522.9 | 34.1878, -118.4741 | 34.0731, -118.2339 |
| 89 | 16 | 767351 | 771667 | 10 | 191 | — | 24799.7 | 34.1802, -118.4702 | 34.0731, -118.2339 |
| 104 | 16 | 717818 | 771667 | 10 | 196 | — | 24164.3 | 34.1722, -118.4675 | 34.0731, -118.2339 |
| 149 | 196 | 763995 | 771673 | 9 | 203 | — | 22918.0 | 34.2198, -118.4093 | 34.0779, -118.2287 |
| 149 | 29 | 763995 | 773012 | 10 | 206 | — | 23000.7 | 34.2198, -118.4093 | 34.0839, -118.2209 |

### free flow

Targets in both halves: **200**. Far edges half A / half B: 246 / 229.
**Survivors (in both): 52.** One-half-only: 371. Survival rate 0.123.

| Group | n | mean rank | median | top-8 | top-20 | top-half |
|---|---|---|---|---|---|---|
| **survivors (both halves)** | 52 | 92.8 | 90 | 0.154 | 0.269 | 0.519 |
| one half only | 371 | 107.8 | 115 | 0.040 | 0.097 | 0.442 |
| random from same stratum | 52 | 106.1 | 104 | 0.019 | 0.038 | 0.500 |
| *chance* | — | 103.5 | 103 | 0.039 | 0.097 | 0.500 |

**The survivors do not sit at high learned-graph rank.** 15.4% are in
the learned graph's top 8 and the median survivor sits at rank 90 of
206. There is a real but weak shift — mean rank 92.8 against 106.1 for a
matched random draw from the same stratum, and a top-8 share 4.0x
chance — but 85% of surviving edges are NOT in the learned graph's
top 8, so this is not recovery of the learned adjacency. Every
surviving edge is therefore listed with its geometry below.

All 52 survivors, with geometry, ranked by learned-graph rank
(`beyond_k_survivors_free_flow.csv`; road distance and haversine in separate
columns, nothing derived from haversine):

| src | tgt | src sensor | tgt sensor | hops | learned rank | road m | haversine m | src lat,lon | tgt lat,lon |
|---|---|---|---|---|---|---|---|---|---|
| 106 | 90 | 718072 | 716571 | 10 | 1 | — | 17296.9 | 34.1491, -118.2257 | 34.2000, -118.4034 |
| 81 | 26 | 769431 | 717804 | unreach | 2 | — | 7299.6 | 34.1584, -118.4566 | 34.0948, -118.4761 |
| 105 | 90 | 718076 | 716571 | 9 | 3 | — | 16878.4 | 34.1634, -118.2253 | 34.2000, -118.4034 |
| 59 | 30 | 767366 | 773013 | 10 | 4 | — | 27284.8 | 34.2122, -118.4734 | 34.0837, -118.2208 |
| 66 | 90 | 767610 | 716571 | 10 | 4 | — | 17156.1 | 34.1855, -118.2177 | 34.2000, -118.4034 |
| 90 | 105 | 716571 | 718076 | 12 | 4 | — | 16878.4 | 34.2000, -118.4034 | 34.1634, -118.2253 |
| 66 | 44 | 767610 | 774012 | unreach | 5 | 10826.6 | 4468.7 | 34.1855, -118.2177 | 34.1477, -118.2014 |
| 106 | 44 | 718072 | 774012 | unreach | 6 | 10092.9 | 2244.4 | 34.1491, -118.2257 | 34.1477, -118.2014 |
| 166 | 149 | 772669 | 763995 | 9 | 9 | — | 22911.4 | 34.0783, -118.2283 | 34.2198, -118.4093 |
| 149 | 133 | 763995 | 716953 | 9 | 12 | — | 20702.9 | 34.2198, -118.4093 | 34.0946, -118.2428 |
| 28 | 90 | 767573 | 716571 | 9 | 12 | — | 17848.2 | 34.1296, -118.2290 | 34.2000, -118.4034 |
| 69 | 26 | 717497 | 717804 | unreach | 13 | — | 8925.9 | 34.1568, -118.4146 | 34.0948, -118.4761 |
| 96 | 182 | 717481 | 717825 | 9 | 15 | — | 18490.2 | 34.1063, -118.3283 | 34.2216, -118.4731 |
| 45 | 70 | 774011 | 717490 | unreach | 17 | — | 15645.1 | 34.1475, -118.2012 | 34.1474, -118.3712 |
| 45 | 24 | 774011 | 717578 | unreach | 24 | — | 6449.6 | 34.1475, -118.2012 | 34.1548, -118.2708 |
| 183 | 90 | 767495 | 716571 | 11 | 30 | — | 17716.7 | 34.1038, -118.2499 | 34.2000, -118.4034 |
| 59 | 48 | 767366 | 760650 | 10 | 30 | — | 26902.2 | 34.2122, -118.4734 | 34.0750, -118.2326 |
| 166 | 158 | 772669 | 764794 | 9 | 32 | — | 23878.7 | 34.0783, -118.2283 | 34.1603, -118.4681 |
| 122 | 21 | 717608 | 769819 | 9 | 40 | — | 17895.5 | 34.1715, -118.3881 | 34.2058, -118.1980 |
| 122 | 121 | 717608 | 769847 | 9 | 42 | — | 15727.6 | 34.1715, -118.3881 | 34.2098, -118.2235 |
| 134 | 204 | 716951 | 772168 | 10 | 44 | — | 24486.8 | 34.0858, -118.2318 | 34.1654, -118.4798 |
| 134 | 172 | 716951 | 772178 | 10 | 46 | — | 26263.3 | 34.0858, -118.2318 | 34.1690, -118.4989 |
| 45 | 77 | 774011 | 765171 | unreach | 55 | — | 24739.1 | 34.1475, -118.2012 | 34.1679, -118.4690 |
| 134 | 41 | 716951 | 772140 | 10 | 57 | — | 24047.2 | 34.0858, -118.2318 | 34.1650, -118.4749 |
| 134 | 86 | 716951 | 772596 | 10 | 67 | — | 26875.9 | 34.0858, -118.2318 | 34.1713, -118.5049 |
| 166 | 180 | 772669 | 717510 | 11 | 77 | — | 28749.2 | 34.0783, -118.2283 | 34.1713, -118.5198 |
| 160 | 126 | 717461 | 769867 | 9 | 103 | — | 15853.1 | 34.0856, -118.3017 | 34.2185, -118.2393 |
| 66 | 70 | 767610 | 717490 | 9 | 106 | — | 14751.4 | 34.1855, -118.2177 | 34.1474, -118.3712 |
| 182 | 95 | 717825 | 718379 | unreach | 106 | — | 19988.4 | 34.2216, -118.4731 | 34.1422, -118.2781 |
| 134 | 181 | 716951 | 717513 | 11 | 112 | — | 29712.8 | 34.0858, -118.2318 | 34.1734, -118.5368 |
| 149 | 194 | 763995 | 759772 | 9 | 123 | — | 10981.9 | 34.2198, -118.4093 | 34.1711, -118.3054 |
| 105 | 186 | 718076 | 717823 | 10 | 133 | — | 23222.6 | 34.1634, -118.2253 | 34.2026, -118.4733 |
| 134 | 180 | 716951 | 717510 | 11 | 134 | — | 28155.9 | 34.0858, -118.2318 | 34.1713, -118.5198 |
| 146 | 144 | 717502 | 764853 | 9 | 134 | — | 23445.5 | 34.1652, -118.4748 | 34.0646, -118.2510 |
| 87 | 165 | 772597 | 716943 | 11 | 139 | — | 29425.0 | 34.1711, -118.5049 | 34.0599, -118.2149 |
| 149 | 170 | 763995 | 716949 | 9 | 139 | — | 22379.1 | 34.2198, -118.4093 | 34.0841, -118.2297 |
| 45 | 150 | 774011 | 717508 | unreach | 150 | — | 29277.7 | 34.1475, -118.2012 | 34.1711, -118.5181 |
| 149 | 109 | 763995 | 761599 | 9 | 161 | — | 14850.3 | 34.2198, -118.4093 | 34.1423, -118.2779 |
| 182 | 77 | 717825 | 765171 | unreach | 164 | 9313.9 | 5988.7 | 34.2216, -118.4731 | 34.1679, -118.4690 |
| 96 | 121 | 717481 | 769847 | 10 | 166 | — | 15010.7 | 34.1063, -118.3283 | 34.2098, -118.2235 |
| 45 | 100 | 774011 | 772151 | unreach | 168 | — | 27480.0 | 34.1475, -118.2012 | 34.1693, -118.4987 |
| 149 | 38 | 763995 | 718045 | 9 | 179 | — | 23059.6 | 34.2198, -118.4093 | 34.0671, -118.2397 |
| 134 | 59 | 716951 | 767366 | 12 | 181 | — | 26299.0 | 34.0858, -118.2318 | 34.2122, -118.4734 |
| 134 | 152 | 716951 | 773996 | unreach | 185 | — | 6755.4 | 34.0858, -118.2318 | 34.1451, -118.2159 |
| 66 | 40 | 767610 | 768066 | 9 | 190 | — | 14953.8 | 34.1855, -118.2177 | 34.1512, -118.3748 |
| 66 | 100 | 767610 | 772151 | 11 | 192 | — | 25918.4 | 34.1855, -118.2177 | 34.1693, -118.4987 |
| 69 | 49 | 717497 | 716956 | 9 | 196 | — | 15676.1 | 34.1568, -118.4146 | 34.1066, -118.2554 |
| 149 | 18 | 763995 | 769953 | 11 | 197 | — | 19308.7 | 34.2198, -118.4093 | 34.2067, -118.1999 |
| 69 | 152 | 717497 | 773996 | unreach | 201 | — | 18330.1 | 34.1568, -118.4146 | 34.1451, -118.2159 |
| 149 | 8 | 763995 | 737529 | 9 | 206 | 8507.5 | 6204.7 | 34.2198, -118.4093 | 34.2026, -118.4735 |
| 149 | 185 | 763995 | 717821 | 9 | 206 | 8676.7 | 6266.6 | 34.2198, -118.4093 | 34.2011, -118.4736 |
| 105 | 87 | 718076 | 772597 | 10 | 206 | — | 25742.8 | 34.1634, -118.2253 | 34.1711, -118.5049 |

### Does the survivor result hold at a smaller sample?

Recomputed on the first 64 targets of the run's seeded shuffle order — an
unbiased subsample of the graph, so this isolates sample size rather than
geography.

| Regime | n targets | survivors | mean rank | top-8 | claims permitted |
|---|---|---|---|---|---|
| congested | 64 | 10 | 158.2 | 0.000 | no |
| congested | 207 | 15 | 131.8 | 0.000 | no |
| free flow | 64 | 14 | 106.1 | 0.071 | yes |
| free flow | 207 | 52 | 92.8 | 0.154 | yes |

**The enrichment is sample-size dependent and does not appear at the
smaller n.** Free-flow survivors sit in the learned graph's top 8 at 0.154
on 207 targets but only 0.071 on 64 targets — against a 0.039 chance rate. The
smaller sample yields only 14 survivors, which is too few to carry the
comparison, so this is as consistent with "n=64 was underpowered" as with
"the effect is not robust". Either way the full-sample result should be
read as weak evidence that has not been shown to replicate under
resampling, not as an established shift.

## The explainer's W against the model's own learned graph

Both objects were produced by the same checkpoint and both get read as "which
sensors matter here", but they are not the same kind of thing: `W` is a
behavioural attribution for specific predictions, `A_sem` is a static parameter
that enters only through the `(1 - alpha)` term of one of two supports.
Agreement is not required and disagreement is not an error.

| Regime | targets | Spearman mean | Spearman median | top-8 J mean | top-8 J median | targets with zero top-8 overlap |
|---|---|---|---|---|---|---|
| congested | 34 | -0.1150 | -0.1406 | 0.1743 | 0.0333 | 17 |
| free flow | 200 | 0.0618 | 0.0874 | 0.0873 | 0.0667 | 84 |

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

Top 20 of 206 ranked sensors (full table: `sensor_anomaly_ranking.csv`;
raw-speed defects for all 207: `sensor_raw_pathologies.csv`):

| target | sensor_id | lat | lon | in_degree | mass_off_adjacency | mass_on_far_road | mass_on_no_road_path | self_mass_share | missing_pct | longest_missing_run_steps | longest_constant_nonzero_run_steps | max_mph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 44.0000 | 774012.0000 | 34.1477 | -118.2014 | 0.0000 | 1.0000 | 0.1991 | 0.8009 | 0.0141 | 7.5460 | 442.0000 | 5.0000 | 70.0000 |
| 26.0000 | 717804.0000 | 34.0948 | -118.4761 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0104 | 10.8720 | 1134.0000 | 66.0000 | 70.0000 |
| 121.0000 | 769847.0000 | 34.2098 | -118.2235 | 1.0000 | 0.9955 | 0.0382 | 0.9572 | 0.0148 | 7.9600 | 381.0000 | 5.0000 | 70.0000 |
| 150.0000 | 717508.0000 | 34.1711 | -118.5181 | 1.0000 | 0.9946 | 0.1028 | 0.8918 | 0.0096 | 10.4550 | 381.0000 | 3.0000 | 70.0000 |
| 29.0000 | 773012.0000 | 34.0839 | -118.2209 | 1.0000 | 0.9943 | 0.1910 | 0.8034 | 0.0133 | 6.5590 | 381.0000 | 4.0000 | 70.0000 |
| 66.0000 | 767610.0000 | 34.1855 | -118.2177 | 1.0000 | 0.9942 | 0.0796 | 0.9146 | 0.0162 | 11.3390 | 402.0000 | 4.0000 | 70.0000 |
| 90.0000 | 716571.0000 | 34.2000 | -118.4034 | 1.0000 | 0.9940 | 0.0404 | 0.9536 | 0.0135 | 7.5780 | 381.0000 | 5.0000 | 70.0000 |
| 51.0000 | 761604.0000 | 34.1541 | -118.2871 | 1.0000 | 0.9939 | 0.3308 | 0.6631 | 0.0136 | 8.4590 | 381.0000 | 3.0000 | 70.0000 |
| 59.0000 | 767366.0000 | 34.2122 | -118.4734 | 1.0000 | 0.9938 | 0.0865 | 0.9073 | 0.0148 | 6.7720 | 381.0000 | 3.0000 | 70.0000 |
| 194.0000 | 759772.0000 | 34.1711 | -118.3054 | 1.0000 | 0.9931 | 0.2276 | 0.7655 | 0.0120 | 10.9480 | 1160.0000 | 3.0000 | 70.0000 |
| 152.0000 | 773996.0000 | 34.1451 | -118.2159 | 1.0000 | 0.9930 | 0.0731 | 0.9199 | 0.0136 | 7.5890 | 442.0000 | 3.0000 | 70.0000 |
| 8.0000 | 737529.0000 | 34.2026 | -118.4735 | 1.0000 | 0.9930 | 0.0688 | 0.9242 | 0.0137 | 7.8310 | 381.0000 | 3.0000 | 70.0000 |
| 184.0000 | 767494.0000 | 34.1038 | -118.2496 | 1.0000 | 0.9920 | 0.3436 | 0.6484 | 0.0153 | 6.4340 | 381.0000 | 3.0000 | 70.0000 |
| 56.0000 | 716955.0000 | 34.0958 | -118.2443 | 2.0000 | 0.9915 | 0.3933 | 0.5982 | 0.0090 | 15.6800 | 857.0000 | 4.0000 | 70.0000 |
| 149.0000 | 763995.0000 | 34.2198 | -118.4093 | 1.0000 | 0.9909 | 0.1028 | 0.8881 | 0.0132 | 9.0310 | 575.0000 | 15.0000 | 70.0000 |
| 200.0000 | 769806.0000 | 34.1964 | -118.1844 | 2.0000 | 0.9898 | 0.0345 | 0.9552 | 0.0147 | 6.3350 | 381.0000 | 5.0000 | 70.0000 |
| 178.0000 | 773975.0000 | 34.1458 | -118.2225 | 1.0000 | 0.9895 | 0.0734 | 0.9161 | 0.0152 | 6.3640 | 381.0000 | 3.0000 | 70.0000 |
| 50.0000 | 769831.0000 | 34.2066 | -118.2010 | 2.0000 | 0.9895 | 0.0954 | 0.8941 | 0.0152 | 6.6260 | 381.0000 | 3.0000 | 70.0000 |
| 46.0000 | 767609.0000 | 34.1855 | -118.2173 | 2.0000 | 0.9887 | 0.1883 | 0.8004 | 0.0133 | 10.7780 | 402.0000 | 3.0000 | 70.0000 |
| 105.0000 | 718076.0000 | 34.1634 | -118.2253 | 1.0000 | 0.9876 | 0.0660 | 0.9216 | 0.0102 | 20.1040 | 875.0000 | 82.0000 | 70.0000 |

Cross-check on the top 20: missingness mean 9.26 vs 8.07 for all sensors;
longest constant non-zero run mean 11.2 vs 5.0 steps.

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

**The window sample is small per target.** 24 shared windows with a ~14% network
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

