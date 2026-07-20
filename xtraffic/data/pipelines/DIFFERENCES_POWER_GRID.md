# Cross-DOMAIN differences: traffic networks vs the IEEE 14-bus power grid

Phase 19. Companion to [`DIFFERENCES.md`](DIFFERENCES.md), which documents the
cross-CITY differences (METR-LA / PEMS-BAY vs Chicago). This file documents the
much larger jump: from road networks to a power grid.

It exists for the paper's strongest claim. Contribution #2 says the mathematical
GNN explanation is what eliminates LLM hallucination. If that holds on a domain
this different, with **no architecture changes**, the claim generalises from
"works on traffic" to "mathematical GNN-explanation grounding is a domain-agnostic
mechanism". The honest version of that argument requires stating exactly what
changed, what did not, and what we had to invent.

---

## 1. The contract that did NOT change

This is the whole point, so it goes first. `power_grid` emits byte-identical
structure to METR-LA:

| Artifact | Contents |
|---|---|
| `{train,val,test}.npz` | `X [samples, 12, N, 2]`, `Y [samples, 12, N]` |
| `adjacency.npy` | `[N, N]` weighted, diagonal 1.0, values in [0,1] |
| `scaler.json` | train-only z-score `{mean, std}` |
| `node_meta.json` | canonical node order + human-readable names |
| `stats.json` | the stats report |

`data/pipelines/sanity_check.py` runs the **same assertions** over `power_grid`
as over the three traffic datasets, and it passes. That gate passing *is* the
proof that the tensor contract is domain-portable:

```
[ok] metr_la:    N=207  X[23974, 12, 207, 2]  finite ✓ shapes ✓ adj ✓
[ok] power_grid: N=14   X[ 6984, 12,  14, 2]  finite ✓ shapes ✓ adj ✓
SANITY GATE: PASS ✓
```

Nothing in `models/gnn/`, `models/explainer/`, `models/advisor/` or
`evaluation/faithfulness.py` was modified for this phase.

---

## 2. The domain mapping

| Traffic | Power grid | Notes |
|---|---|---|
| Loop sensor / road segment | **Bus** (substation node) | 207 → 14 nodes |
| Road link | **Branch** (line or transformer) | 20 branches |
| Road-network distance (m) | **Electrical distance** `\|Z\|` (per-unit) | see §4 |
| Speed (mph) | **Voltage magnitude** (per-unit) | the predicted state |
| Congestion (speed drops) | **Undervoltage** (voltage sags) | the stress condition |
| Incident blocking a link | **N-1 branch outage** | the discrete disturbance |
| Local demand spike at a junction | **Local demand surge** at a bus | see §5 |
| Congestion propagates upstream | Voltage deviation propagates through the network | see §7 |

Both target variables are *the quantity that degrades when the network is
stressed*. That is why voltage magnitude is the right analogue of speed, and not,
say, power flow (which is the analogue of traffic *volume*, a different quantity
that our model does not predict).

---

## 3. Table of concrete schema differences

| Aspect | METR-LA / PEMS-BAY | IEEE 14-bus |
|---|---|---|
| Node count `N` | 207 / 325 | **14** |
| Data origin | Measured loop detectors, 2012/2017 | **Real topology + real AC power flow, synthetic demand** (§5) |
| Acquisition | Download a CSV from Zenodo | `pandapower.networks.case14()` — **library data, not a URL** |
| Time grid | Native 5-min | Generated on a 5-min grid (chosen to match) |
| Coverage | ~4 months (LA) | 10,000 steps ≈ **34.7 simulated days** |
| Target units | mph, roughly 0–70 | per-unit voltage, roughly **0.79–1.09** |
| Target scale | std ≈ 19.5 mph | std ≈ **0.024 pu** — ~800× smaller (§8) |
| Missing-value code | `0.0` mph (8.1% of METR-LA) | `0.0` pu, reserved for power-flow non-convergence. **0.0% occurred** |
| Adjacency source | Road distances → thresholded Gaussian kernel | **Branch admittance** `1/\|Z\|`, normalised (§4) |
| Edge semantics | Drive-distance decay, weights (0,1] | Electrical coupling strength, weights (0,1] |
| Node names | Reverse-geocoded region from lat/lon | **Zone + role**, read out of the network (§6) |
| Geography | Real lat/lon per node | **None.** `node_meta["latlon"] = null` (§6) |
| Extra node features | weather / events / transit sidecars | **`mod_power` sidecar**: active + reactive power per bus |

---

## 4. Adjacency: why we did NOT reuse the traffic kernel

The obvious move was to feed electrical distance `|Z|` into the same thresholded
Gaussian kernel (DCRNN Eq. 10) the traffic pipelines use. We built that first,
measured it, and rejected it.

- A road network is **dense in distance** — every sensor pair has a finite road
  distance — so the Gaussian kernel plus the `kappa` threshold is what *creates* a
  sparse graph out of a dense distance matrix.
- A grid's **20 branches ARE the topology**. Every non-branch pair is already at
  infinite distance. The kernel's job changes from *selecting* edges to *weighting*
  edges already known to be real — and the Gaussian is far too sharp for that.
  Branch `|Z|` spans 0.044–0.556 pu (a 12.6× ratio); with `sigma = 0.136` the
  squared exponent crushes the far end. Measured weights: transformer 4-9 →
  `1.5e-8`, line 13-14 → `0.0003`, line 9-14 → `0.008`, line 12-13 → `0.008`,
  line 6-12 → `0.013`. **Five of twenty real physical branches** effectively
  deleted from message passing.

We use **normalised branch admittance** `|Y_ij| = 1/|Z_ij|` instead, scaled so
weights land in (0,1] with 1.0 on the diagonal — the same numeric contract the
traffic adjacency has. This is not an invention: the admittance (Y-bus) matrix
*is* the canonical graph operator of a power network, the object every power flow
is defined over. Resulting weights span 0.079–1.000; every real branch keeps a
meaningful weight. Chicago already set the precedent that adjacency construction
is a **per-domain data decision** (it uses a third method, binary shared-endpoint,
because it ships no distance file). The *model* consuming `adjacency.npy` is
untouched, which is what the cross-domain claim actually rests on.

`adjacency_method: "gaussian"` is kept in the config so the comparison above can
be reproduced as an ablation.

---

## 5. The synthetic-data methodology (the part reviewers will attack)

**The honest one-line framing: the topology and the physics are real, the demand
is synthetic.** The IEEE 14-bus case is a static snapshot — one load vector, one
solved operating point — and ships no time series. Generating synthetic load and
solving a power flow per timestep is standard practice in power-systems ML, but
it must be stated plainly, not glossed.

### What is REAL (not invented by us)
- The 14-bus topology, all 20 branches, generator/load placement, the shunt
  capacitor, and every branch impedance — all from `pandapower`'s `case14`, the
  standard IEEE case derived from a portion of the American Electric Power system
  (US Midwest, February 1962).
- **Every voltage number in the dataset.** Each timestep is solved with a real
  Newton-Raphson AC power flow. We never fabricate a voltage; we fabricate a
  demand and let the physics produce the voltage.
- The published case14 load vector (259.0 MW / 73.5 MVAr), used as the daily mean.

### What is SYNTHETIC, and exactly how
1. **Daily shape** — documented two-peak curve: base + Gaussians at 08:00 (morning
   ramp), 13:00 (midday plateau), 19:00 (evening peak), evaluated on a wrapped
   hour axis so it is continuous across midnight, and normalised to unit 24-h
   mean (hence "published loads = the daily *mean*").
2. **Spatial heterogeneity** — each of the 11 demand buses gets a profile class
   (residential / commercial / industrial) re-weighting those peaks, plus a fixed
   per-bus phase offset of up to ±1.5 h. Without this every bus is an affine
   function of one scalar and the graph carries nothing.
3. **Weekly shape** — weekend demand × 0.88.
4. **Noise** — AR(1), σ = 3%, φ = 0.90 (e-folding ≈ 47 min). *Temporally
   correlated*, not white: real load residuals are autocorrelated, and white noise
   would be trivially filtered and would understate the difficulty.
5. **Constant power factor** — reactive demand Q scales with P at each bus's
   published Q/P ratio. Standard when only a P profile is known.
6. **Local demand surges** — Poisson, ~1 per bus per 2 days, 30–120 min, ×1.3–2.0.
   The per-bus analogue of a traffic incident (industrial start-up, EV cluster,
   motor bank). See §7 for why these had to be added.
7. **N-1 contingencies** — Poisson, 2/day, 30–120 min, single branch out of
   service. The link-level analogue of a traffic incident.

### Two load-bearing power-flow settings
- **`enforce_q_lims: true`.** Without it the four voltage-controlling machines
  hold their setpoints *exactly* at any demand, bus voltages barely move, and the
  forecasting task is degenerate. With reactive limits enforced (the physically
  correct model) a saturated machine converts PV→PQ and voltage sags. Measured
  effect: per-bus daily voltage swing goes from ~0 to 0.02–0.06 pu.
- **N-1 security screen.** Outage candidates are screened twice: never a *bridge*
  branch (islanding has no power-flow solution — this excludes transformer 7-8),
  and never a branch that fails to converge at the **99th percentile** of system
  demand (a real control room operates N-1 secure — this excludes lines 1-2, 2-3
  and transformer 5-6). Screening at the *absolute* max was a bug we hit and
  fixed: it knocked out every branch and produced a dataset with zero
  contingencies.

### Honest limitations to state in the paper
- The demand model is our construction. Its *shape* is defensible and documented,
  but no claim is made that it reproduces any real utility's load.
- pandapower's `case14` assigns physical voltage levels (135 / 14 / 12 / 0.208 kV)
  when converting the published per-unit case. **The 0.208 kV figures are an
  artifact of that conversion, not a real distribution voltage.** Zone labels
  derived from them are still internally consistent and electrically meaningful
  (they separate the transmission core from the load pocket), but nobody should
  read "0.208 kV" as a claim about the physical system.
- Ground truth for any downstream intervention study would be **model-in-the-loop**
  here just as it is in Phase 8/10 — the same circularity limitation applies and
  must be restated, not quietly dropped.

---

## 6. Node naming: the one thing genuinely missing

The traffic advisor turns node indices into words via `NodeNamer`, which is a
**coordinate lookup** (lat/lon → region bounding box). An IEEE test case has no
geography, so that path does not apply and `node_meta["latlon"]` is `null`.

`power_grid.py` instead writes an explicit `node_meta["names"]` list, built from
facts read out of the network — zone (voltage tier) + role (slack / generator /
synchronous condenser / load / shunt capacitor), e.g.:

```
LV load pocket (bus 14, load bus, 0.208 kV)
HV transmission core (bus 3, synchronous condenser + load, 135 kV)
```

Note that roles are **composed**, not picked: bus 3 hosts a synchronous condenser
*and* is the largest load in the system (94.2 MW), and reporting only one of those
would actively mislead the advisor.

> ✅ **RESOLVED (Step 2).** `NodeNamer` now reads `node_meta["names"]` — but only
> for datasets listed in `_NO_GEOMETRY_CITIES` (`{"power_grid"}`), which take a
> separate branch that never touches the lat/lon code. That is deliberately
> narrower than the "~3-line change" sketched above: a general "prefer `names` if
> present" rule would also have caught Chicago, whose `node_meta` carries an
> ignored `names` key, and silently altered the committed Phase-11 cross-city
> numbers. Gating on the no-geometry set gets the power grid what it needs and
> leaves every traffic city byte-identical.
>
> The equivalent problem on the *evaluation* side was found and fixed in Step 3:
> `faithfulness.NodeTable` did `list(node_meta["latlon"])` unconditionally and
> crashed on `null`, and its resolution ladder was geographic-only. It now has a
> non-geographic ladder (bus id → electrical zone → fuzzy zone), gated the same
> way. Both changes are pinned by `evaluation/verify_traffic_unchanged.py`.

---

## 7. Is there anything here for a GRAPH model to find?

A synthetic dataset can very easily be **degenerate** — collinear enough that a
per-node affine map of one global signal solves it, leaving the graph (and
therefore the explainer, and therefore the entire faithfulness measurement) doing
no work. Publishing a cross-domain grounding result on a degenerate dataset would
be worthless, so `power_grid.py` computes and logs two diagnostics into
`stats.json`.

**Diagnostic 1 — PC1 variance share.** How much of the dataset is one global
factor.

| Dataset | PC1 share |
|---|---|
| METR-LA, 155 clean sensors | 0.387 |
| METR-LA, subsampled to 13 sensors | 0.436 (so it is *not* a node-count artifact) |
| **power_grid, 14 buses** | **0.917** |

This is a genuine, reportable domain difference and we do not hide it: a 14-bus
network is small and tightly coupled, and voltage responds near-linearly to
loading, whereas traffic congestion is a local *threshold* phenomenon. The power
grid really is more collinear than a city.

**Diagnostic 2 — neighbour advantage.** The one that actually decides whether the
graph matters. Remove PC1 from every bus, then predict each bus's residual by
least squares from (a) its graph neighbours and (b) an equal number of
non-neighbours:

| Version | neighbour R² | non-neighbour R² | ratio |
|---|---|---|---|
| v1: daily curve + noise only | 0.653 | 0.558 | **1.17×** |
| final: + phase jitter + local demand surges | 0.487 | 0.270 | **1.80×** |

v1 was nearly degenerate — the graph barely mattered. Adding local demand surges
(§5.6) fixed it. **Note the two diagnostics move in opposite directions**: surges
*raise* PC1 share while nearly doubling the neighbour ratio, which is exactly why
we do not steer on PC1 alone. Surge parameters were chosen to be physically
reasonable (one surge per substation every ~2 days, 1.3–2.0×), not tuned to
maximise the ratio.

A third observation, consistent with traffic: contingency periods carry
**3.6× larger** de-globalised residuals than intact periods. Graph-relevant signal
concentrates in disturbance events — the same lesson Phase 3 learned when it found
node-occlusion fidelity is only meaningful on *congested* targets. Downstream
scenario sampling should stratify on stress, exactly as the traffic phases do.

---

## 8. Consequences downstream (read before running Phase 19 Step 2)

1. **N = 14 is tiny.** The adaptive/semantic adjacency (MOD 1) is `[N × d]`, so
   node-specific parameters must be re-initialised at N=14 exactly as in the
   cross-city transfer — only node-agnostic weights transfer. The existing
   `cross_city.py` `transfer_weights` path already handles this.
2. **Units are ~800× smaller.** MAE will be ~0.005 *per-unit*, not ~3 *mph*. The
   metrics code is unit-agnostic, but every label, plot axis and table caption in
   `utils/metrics.py` and the evaluation modules says "mph". Numbers will be
   correct; **labels will lie** unless they are parameterised. Fix before
   publishing any power-grid prediction table.

   > **Step 3 update — fixed where it was a CONFOUND, still open where it is
   > cosmetic.** The *prompt* path is fixed: `models/advisor/domains.py` gives the
   > advisor a per-domain vocabulary, so it now says "Current voltage: 0.881 pu"
   > instead of "Current speed: 0.88 mph". This was not a cosmetic fix and could
   > not wait — an LLM told a substation is doing 0.88 mph invents traffic
   > interventions, and the study would then have scored *our own prompt's*
   > confusion as the model's hallucination, silently corrupting the cross-domain
   > result.
   >
   > Still traffic-worded, and genuinely cosmetic because no LLM reads them:
   > `utils/metrics.py` labels, the loss-curve axes, and the explanation SCHEMA
   > KEYS (`predicted_speed_mph` etc.). The schema keys are deliberately left
   > alone — `schema.py` is a frozen Phase-3 contract that Phase 4/5 validate
   > against, and renaming them would break every downstream phase to fix a
   > cosmetic problem. Power-grid explanations instead carry an additive
   > `meta.units = "pu"` and a `meta.note` saying the `*_speed_mph` fields hold
   > voltages, so a human reading the JSON is never misled.
3. **The missing-value mask is inert.** `missing_data_pct = 0.0`, so masked-MAE
   reduces to plain MAE here. Nothing breaks; it just means the masking machinery
   is untested by this dataset.
4. **Bus 1 is near-constant** (std 5.8e-5 pu — it is the voltage reference and only
   moves when its own reactive limit binds). It is trivially predictable and will
   flatter aggregate metrics slightly. Reported as `near_constant_buses` in
   `stats.json`.
5. **Free-flow analogue.** Phase 3 found explanations are only meaningful on
   *congested* targets. The equivalent filter here is **not** simply "< 0.95 pu"
   (only 1.04% of steps have any bus below it). Prefer stratifying by voltage
   deviation from the bus's own typical level, and use the committed
   `raw/power_grid/{outage_log,surge_log}.json` to sample disturbance windows
   directly.

---

## 9. Reproducing

```bash
pip install 'pandapower==2.14.11'                       # pinned in requirements.txt
python -m xtraffic.data.pipelines.power_grid --smoke    # ~10 s, writes nothing
python -m xtraffic.data.pipelines.power_grid            # ~95 s, 10,000 power flows
python -m xtraffic.data.pipelines.sanity_check          # same gate as the traffic sets
```

Fully deterministic: seed 42, fixed start timestamp, and the network comes from
pinned library data rather than a URL that can rot.
