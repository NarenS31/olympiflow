"""Phase 19 — IEEE 14-bus POWER GRID pipeline (the CROSS-DOMAIN dataset).

Run:  python -m xtraffic.data.pipelines.power_grid
Smoke test (no full run):  python -m xtraffic.data.pipelines.power_grid --smoke

WHY THIS EXISTS
---------------
Contribution #2 of the paper says a mathematical GNN explanation eliminates LLM
hallucination. So far that is shown on TRAFFIC only (METR-LA / PEMS-BAY / Chicago).
Phase 19 asks the strongest version of the question: does the SAME result hold in
a COMPLETELY DIFFERENT domain with NO architecture changes? To answer it we need a
second-domain dataset that lands in the EXACT same tensor contract:

    X [samples, 12, N, 2]   (channel 0 = target state z-scored, channel 1 = time-of-day)
    Y [samples, 12, N]      (z-scored target state)
    + adjacency.npy [N,N], scaler.json, node_meta.json, stats.json

This file produces that for the IEEE 14-bus test case. Nothing downstream
(stgnn.py / explain.py / advisor.py / faithfulness.py) is modified — that is the
whole point of the experiment.

THE DOMAIN MAPPING (state it in the paper exactly like this)
------------------------------------------------------------
    traffic                          power grid
    ------------------------------   --------------------------------------
    sensor / road segment            BUS (substation node)
    road link                        BRANCH (transmission line or transformer)
    road-network distance            ELECTRICAL distance |Z| in per-unit
    speed (mph)                      voltage magnitude (per-unit)
    congestion (speed drops)         undervoltage (voltage sags under load/outage)
    incident blocking a link         N-1 branch OUTAGE
    congestion propagates upstream   voltage deviation propagates through the network

Both target variables are "the quantity that degrades when the network is
stressed", which is why voltage magnitude is the right analogue of speed.

THE SYNTHETIC-DATA METHODOLOGY (reviewers WILL scrutinize this — read it)
------------------------------------------------------------------------
The IEEE 14-bus case is a STATIC snapshot: one load vector, one solved operating
point. It ships no time series. Generating synthetic load time series and solving
a power flow at each step is standard practice in power-systems ML (it is how
essentially every "GNN for power grids" paper builds its dataset), but the
*honest* framing is: the TOPOLOGY and the PHYSICS are real, the DEMAND is
synthetic. We are not claiming to have measured a real grid.

What is REAL (not invented by us):
  * The 14-bus topology, 20 branches, generator/load placement, shunt capacitor,
    and every branch impedance — all from pandapower's `case14`, which is the
    standard IEEE case derived from a portion of the American Electric Power
    system (Midwestern US, February 1962).
  * Every voltage number in the output. Each timestep is solved with a real
    Newton-Raphson AC power flow (pandapower `runpp`). We never fabricate a
    voltage; we fabricate a demand and let the physics produce the voltage.
  * The published case14 load vector, used as the DAILY MEAN operating point.

What is SYNTHETIC (and exactly how):
  1. DAILY SHAPE. A documented two-peak curve m(h) = base + sum of Gaussians at
     08:00 (morning ramp), 13:00 (midday plateau) and 19:00 (evening peak). It is
     normalised so its 24-h mean is 1.0 — hence "published loads = daily MEAN".
  2. SPATIAL HETEROGENEITY. Each of the 11 load buses is assigned a profile class
     (residential / commercial / industrial) which re-weights the three peaks.
     WHY THIS MATTERS: if every bus followed the identical curve, all node series
     would be affine functions of one scalar, the graph would carry no
     information, and the explainer would be measuring nothing. Heterogeneity is
     what makes this a graph problem rather than a scalar problem.
  3. WEEKLY SHAPE. Weekend demand scaled by `weekend_factor`.
  4. NOISE. AR(1) (temporally correlated) multiplicative noise per bus, NOT white
     noise. Real load residuals are autocorrelated; white noise would be trivially
     filtered by any temporal model and would understate the difficulty.
  5. CONSTANT POWER FACTOR. Reactive demand Q scales with P at each bus's
     published Q/P ratio — the standard assumption when only a P profile is known.
  6. CONTINGENCIES. A Poisson process takes single branches out of service for
     30-120 min (the N-1 analogue of a traffic incident). WHY: without them,
     voltage is a smooth function of one global load multiplier and every
     explanation would be degenerate — the same lesson Phase 3 learned when it
     found node-occlusion fidelity is only meaningful on CONGESTED targets.
     The outage is NOT given to the model as a feature (just as a traffic incident
     is not an input feature); the model must infer it from the voltage response.

Two power-flow settings that are load-bearing, so they are config knobs, not
hidden constants:
  * `enforce_q_lims: true`. Without it the four PV generators hold their voltage
    setpoints EXACTLY at any demand, bus voltages barely move, and the forecasting
    task is degenerate. With generator reactive limits enforced (the physically
    correct model) a generator that runs out of VArs switches PV->PQ and voltage
    sags. Measured effect: daily voltage swing per bus goes from ~0 to 0.02-0.06 pu.
  * N-1 SECURITY SCREEN on the outage candidate set. Two screens, both documented
    below in `_outage_candidates`: (a) never take out a BRIDGE branch, because
    that islands part of the network and the power flow has no solution; (b) never
    take out a branch whose removal fails to converge at PEAK demand, because a
    real control room operates N-1 secure. Anything that still diverges at runtime
    is recorded as MISSING (the 0.0 sentinel), never as a fabricated number.

Python 3.9 compatible. pandapower + numpy + pandas + networkx only.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xtraffic.data.pipelines._modality_common import save_modality_sidecar
from xtraffic.utils.graph_utils import (
    StandardScaler,
    chronological_split,
    gaussian_kernel_adjacency,
    sliding_windows,
)
from xtraffic.utils.io_utils import (
    load_data_config,
    print_stats_report,
    processed_dir,
    raw_dir,
    save_processed_dataset,
)

DATASET = "power_grid"

# Missing-measurement sentinel, in REAL units. Matches the DCRNN/METR-LA
# convention (0.0 mph = missing) so utils/metrics.py masked-MAE works unchanged.
# A converged bus voltage is never 0.0 pu, so the sentinel is unambiguous here.
MISSING_SENTINEL = 0.0


# ---------------------------------------------------------------------------
# 1. The network: topology, per-unit impedances, adjacency
# ---------------------------------------------------------------------------
def load_case14():
    """The standard IEEE 14-bus case, straight from pandapower's library data.

    NOT a download: pandapower ships the case as package data, so pinning the
    pandapower version in requirements.txt makes this byte-reproducible for anyone
    who installs the repo — strictly better than a URL that can rot.
    """
    try:
        import pandapower.networks as pn
    except ImportError as exc:  # teach, don't just crash (CLAUDE.md)
        raise ImportError(
            "pandapower is not installed. It ships the IEEE 14-bus case as library "
            "data and solves the AC power flow.\n"
            "    pip install 'pandapower==2.14.11'\n"
            "(2.14.x is the last line that supports Python 3.9 AND pins numpy<2, "
            "which we need because the torch wheels are built against the numpy 1.x ABI.)"
        ) from exc
    return pn.case14()


def branch_table(net) -> pd.DataFrame:
    """Every branch (line or transformer) with its PER-UNIT series impedance.

    WHY per-unit and not ohms: pandapower's case14 stores physical ohms at each
    branch's own voltage level (135 kV down to 0.208 kV), so raw ohm values differ
    by six orders of magnitude across the network and are NOT comparable. Per-unit
    on the common 100 MVA system base is the standard way to make them comparable
    — it is exactly how the IEEE case is published in the literature.

        line:   z_pu = (r + jx) * length / Z_base,  Z_base = vn_kv^2 / sn_mva
        trafo:  z_pu = (vk% / 100) * (S_base / S_trafo)      [short-circuit voltage]

    Returns a [n_branches, ...] frame with from/to bus INDICES (0-based, matching
    the tensor node order), the branch kind, r_pu / x_pu / z_pu.
    """
    s_base = float(net.sn_mva)                                   # 100 MVA
    rows: List[Dict] = []

    for i, r in net.line.iterrows():
        vn = float(net.bus.vn_kv[r.from_bus])                    # branch voltage level, kV
        z_base = vn ** 2 / s_base                                # ohms per unit
        r_pu = float(r.r_ohm_per_km * r.length_km) / z_base
        x_pu = float(r.x_ohm_per_km * r.length_km) / z_base
        rows.append({"kind": "line", "elem_idx": int(i),
                     "from_bus": int(r.from_bus), "to_bus": int(r.to_bus),
                     "r_pu": r_pu, "x_pu": x_pu, "z_pu": float(np.hypot(r_pu, x_pu))})

    for i, r in net.trafo.iterrows():
        # Transformer short-circuit impedance, referred to the SYSTEM base.
        z_pu = float(r.vk_percent) / 100.0 * (s_base / float(r.sn_mva))
        r_pu = float(r.vkr_percent) / 100.0 * (s_base / float(r.sn_mva))
        x_pu = float(np.sqrt(max(z_pu ** 2 - r_pu ** 2, 0.0)))
        rows.append({"kind": "trafo", "elem_idx": int(i),
                     "from_bus": int(r.hv_bus), "to_bus": int(r.lv_bus),
                     "r_pu": r_pu, "x_pu": x_pu, "z_pu": z_pu})

    return pd.DataFrame(rows)


def _impedance_matrix(n_nodes: int, branches: pd.DataFrame) -> np.ndarray:
    """[N, N] branch impedance |Z| in per-unit; np.inf where there is no branch."""
    dist = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)  # [N, N]
    np.fill_diagonal(dist, 0.0)                                   # a bus is 0 from itself
    for r in branches.itertuples(index=False):
        # Parallel branches between the same pair: keep the LOWER impedance (they
        # are electrically in parallel, so the pair is closer, not further).
        dist[r.from_bus, r.to_bus] = min(dist[r.from_bus, r.to_bus], r.z_pu)
        dist[r.to_bus, r.from_bus] = dist[r.from_bus, r.to_bus]   # undirected
    return dist


def build_adjacency(n_nodes: int, branches: pd.DataFrame, method: str = "admittance",
                    normalized_k: float = 0.0) -> np.ndarray:
    """[N, N] weighted adjacency of the grid. Default = normalised ADMITTANCE.

    WHY NOT THE TRAFFIC KERNEL (this decision is measured, not assumed)
    -------------------------------------------------------------------
    The obvious move is to reuse the traffic pipelines' thresholded Gaussian kernel
    (DCRNN Eq. 10) with electrical distance |Z| in place of road distance. We
    implemented that first and MEASURED it: it is wrong here.

      * Road networks are DENSE in distance — every sensor pair has a finite road
        distance — so the Gaussian kernel plus the kappa threshold is what CREATES
        a sparse graph out of a dense distance matrix.
      * A grid's 20 branches ARE the topology. Every non-branch pair is already at
        infinite distance, so there is nothing to sparsify, and the kernel's job
        changes from "select edges" to "weight the edges we already know are real".
      * The Gaussian is far too sharp for that job. Branch |Z| spans 0.044 to 0.556
        pu (a 12.6x ratio); with sigma = 0.136 the exponent squashes the far end to
        nothing. Measured weights: 4-9 transformer 1.5e-8, line 13-14 0.0003, line
        9-14 0.008, line 12-13 0.008, line 6-12 0.013 — five of twenty REAL physical
        branches effectively deleted from message passing. Silently dropping real
        edges to match a borrowed hyperparameter would be indefensible.

    WHAT WE USE INSTEAD
    -------------------
    Branch ADMITTANCE magnitude, |Y_ij| = 1 / |Z_ij|, normalised by its maximum so
    weights land in (0, 1] with 1.0 on the diagonal — the same numeric contract the
    traffic adjacency has. This is not an invention: the admittance (Y-bus) matrix
    IS the canonical graph operator of a power network, the object every power-flow
    and every power-system GNN is defined over. Strong electrical coupling (low
    impedance) becomes a heavy edge, exactly as short road distance becomes a heavy
    edge in traffic. Resulting weights span 0.080 to 1.000 — every real branch keeps
    a meaningful weight.

    Precedent for choosing this per-domain: Chicago already builds its adjacency a
    third way (binary shared-endpoint), because it ships no distance file. The
    adjacency is a DATA-pipeline decision made per domain; the MODEL that consumes
    `adjacency.npy` is untouched, which is what the cross-domain claim rests on.

    `method="gaussian"` keeps the traffic kernel available for the ablation that
    demonstrates the paragraph above.
    """
    dist = _impedance_matrix(n_nodes, branches)                   # [N, N] pu, inf off-branch

    if method == "gaussian":
        return gaussian_kernel_adjacency(dist, normalized_k=normalized_k)  # [N, N]

    if method != "admittance":
        raise ValueError(f"unknown adjacency_method '{method}' (use 'admittance' or 'gaussian')")

    with np.errstate(divide="ignore"):
        adm = np.where(np.isinf(dist), 0.0, 1.0 / dist)           # [N, N]; inf -> 0, diag -> inf
    np.fill_diagonal(adm, 0.0)                                    # drop the 1/0 diagonal first
    peak = adm.max() if adm.max() > 0 else 1.0
    adj = (adm / peak).astype(np.float32)                         # [N, N] in [0, 1]
    np.fill_diagonal(adj, 1.0)                                    # self-loops, as in traffic
    return adj


def _outage_candidates(net, branches: pd.DataFrame, peak_mult: float,
                       base_p: np.ndarray, base_q: np.ndarray,
                       enforce_q_lims: bool) -> List[int]:
    """Branch rows that may be taken out of service. Two documented screens.

    SCREEN 1 — connectivity. A BRIDGE (an edge whose removal disconnects the
    graph) cannot be opened: the islanded part has no generation reference and the
    power flow has no solution at all. In case14 exactly one branch is a bridge,
    the 7-8 transformer feeding the synchronous condenser at bus 8.

    SCREEN 2 — N-1 security at PEAK demand. A real control room is operated so
    that ANY single branch can be lost without collapsing the system. We enforce
    the same rule: solve the power flow at peak demand with each candidate out,
    and drop any candidate that fails to converge (that is voltage collapse, not
    an operating point). This is a standard N-1 contingency screen, and doing it
    ONCE offline is much cheaper than discovering it 10,000 times at runtime.
    """
    import networkx as nx
    import pandapower as pp

    g = nx.Graph()
    g.add_nodes_from(range(len(net.bus)))
    g.add_edges_from([(int(r.from_bus), int(r.to_bus)) for r in branches.itertuples(index=False)])
    bridges = set(frozenset(e) for e in nx.bridges(g))

    keep: List[int] = []
    dropped_bridge, dropped_diverge = [], []
    for row_i, r in enumerate(branches.itertuples(index=False)):
        pair = frozenset((r.from_bus, r.to_bus))
        if pair in bridges:
            dropped_bridge.append((r.kind, r.from_bus + 1, r.to_bus + 1))
            continue
        tbl = net.line if r.kind == "line" else net.trafo
        net.load["p_mw"] = base_p * peak_mult
        net.load["q_mvar"] = base_q * peak_mult
        tbl.at[r.elem_idx, "in_service"] = False
        try:
            pp.runpp(net, enforce_q_lims=enforce_q_lims)
            keep.append(row_i)
        except Exception:
            dropped_diverge.append((r.kind, r.from_bus + 1, r.to_bus + 1))
        finally:
            tbl.at[r.elem_idx, "in_service"] = True

    print(f"[{DATASET}] N-1 screen: {len(keep)}/{len(branches)} branches are outage candidates")
    if dropped_bridge:
        print(f"[{DATASET}]   dropped (bridge / would island): {dropped_bridge}")
    if dropped_diverge:
        print(f"[{DATASET}]   dropped (voltage collapse at peak): {dropped_diverge}")
    return keep


# ---------------------------------------------------------------------------
# 2. The synthetic demand model
# ---------------------------------------------------------------------------
def daily_profile(hours: np.ndarray, cfg: Dict, class_weights: Tuple[float, float, float]) -> np.ndarray:
    """Normalised daily load multiplier for one bus class. hours -> [len(hours)].

    m(h) = base + sum_p  w_p * A_p * exp(-(h - c_p)^2 / (2 sigma_p^2))

    The three Gaussians are the morning ramp, the midday commercial plateau and
    the evening residential peak (config `daily_profile.peaks`). `class_weights`
    re-weights them per bus class, which is what makes buses differ from each
    other. The curve is divided by its own 24-h mean, so the published case14
    load vector is reproduced as the DAILY AVERAGE operating point.

    NOTE the Gaussians are evaluated on a WRAPPED hour axis (h, h-24, h+24) so the
    curve is continuous across midnight — an evening peak at 19:00 with sigma 2.2 h
    would otherwise be truncated at the day boundary and put a discontinuity in
    every 00:00 window.
    """
    m = np.full(hours.shape, float(cfg["base"]), dtype=np.float64)
    for w, (amp, centre, sigma) in zip(class_weights, cfg["peaks"]):
        for shift in (-24.0, 0.0, 24.0):                          # wrap across midnight
            m += w * amp * np.exp(-((hours + shift - centre) ** 2) / (2.0 * sigma ** 2))
    return m


def ar1_noise(n_steps: int, n_series: int, phi: float, std: float, rng) -> np.ndarray:
    """AR(1) noise, [n_steps, n_series], with the requested STEADY-STATE std.

    e_t = phi * e_{t-1} + w_t,  w_t ~ N(0, sigma_w^2)
    A stationary AR(1) has var(e) = sigma_w^2 / (1 - phi^2), so to hit a target
    steady-state `std` we must draw the innovations with
        sigma_w = std * sqrt(1 - phi^2)
    and start from the stationary distribution (e_0 ~ N(0, std^2)) so there is no
    burn-in transient at the top of the series.
    """
    sigma_w = std * np.sqrt(1.0 - phi ** 2)
    e = np.empty((n_steps, n_series), dtype=np.float64)
    e[0] = rng.normal(0.0, std, size=n_series)                    # stationary start
    for t in range(1, n_steps):
        e[t] = phi * e[t - 1] + rng.normal(0.0, sigma_w, size=n_series)
    return e


def demand_surges(n_steps: int, n_loads: int, ds: Dict, steps_per_day: float, rng
                  ) -> Tuple[np.ndarray, List[Dict]]:
    """Local demand surges at individual buses. -> ([T, n_loads] factor, event log).

    WHY THIS EXISTS (this is a correction we made after MEASURING the first version
    of the dataset, so it is worth stating plainly):
      The first generator had only a daily curve + weekly factor + AR(1) noise. We
      ran a PCA on the resulting bus voltages and found PC1 explained 91.9% of the
      variance — i.e. every bus was very nearly an affine function of ONE global
      load factor. For comparison, the same PCA on real METR-LA speeds gives PC1 =
      38.7%. A dataset that collinear would make the cross-domain result vacuous:
      the graph would carry almost no information, so of course an explainer would
      look "faithful". We would have been grading an exam with one question.

      Real demand is not one global factor. Individual substations see LOCAL
      surges — a large industrial customer starting up, an EV-charging cluster, a
      motor bank switching in. Adding them is both more realistic AND what gives
      the graph something to explain, exactly as a traffic incident (not the rush-
      hour curve) is what makes a traffic explanation non-trivial.

    Model: per bus, Poisson arrivals at `surges_per_bus_per_day`, each a step-up to
    a uniform factor in `surge_amplitude` for a uniform duration in
    [surge_min_steps, surge_max_steps]. Surges at one bus do not overlap.
    """
    factor = np.ones((n_steps, n_loads), dtype=np.float64)        # [T, n_loads]
    events: List[Dict] = []
    rate = float(ds["surges_per_bus_per_day"])
    if rate <= 0.0:
        return factor, events
    p_step = rate / steps_per_day
    lo, hi = int(ds["surge_min_steps"]), int(ds["surge_max_steps"])
    amp_lo, amp_hi = [float(a) for a in ds["surge_amplitude"]]

    for b in range(n_loads):
        t = 0
        while t < n_steps:
            if rng.random() < p_step:
                dur = int(rng.integers(lo, hi + 1))
                end = min(t + dur, n_steps)
                amp = float(rng.uniform(amp_lo, amp_hi))
                factor[t:end, b] = amp
                events.append({"bus_load_idx": b, "start": t, "end": end,
                               "amplitude": round(amp, 3)})
                t = end
            else:
                t += 1
    return factor, events


def build_load_multipliers(index: pd.DatetimeIndex, n_loads: int, ds: Dict,
                           steps_per_day: float, rng
                           ) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """Per-load-bus demand multiplier over time. -> ([T, n_loads], classes, surges).

    multiplier[t, b] = base_scale
                       * daily_b(hour_t + phase_b)      # class shape + per-bus phase
                       * weekday_factor_t               # weekend down-scaling
                       * surge_factor[t, b]             # local demand events
                       * (1 + ar1_noise[t, b])          # correlated residual

    Bus classes are assigned DETERMINISTICALLY by round-robin over the load buses
    (not randomly) so the spatial pattern is stable across reruns and easy to
    describe in the paper: load bus 0 -> residential, 1 -> commercial,
    2 -> industrial, 3 -> residential, ...

    PHASE JITTER: each bus's daily curve is shifted by a fixed offset drawn once
    from +/- `phase_jitter_hours`. Real load centres do not peak at the same minute
    — an industrial park peaks before a residential feeder — and de-synchronising
    them is a second (cheap, realistic) source of the spatial diversity the graph
    needs. Drawn from the seeded RNG, so it is reproducible.
    """
    hours = index.hour.values + index.minute.values / 60.0        # [T] in [0, 24)
    class_names = list(ds["bus_classes"].keys())
    assigned = [class_names[b % len(class_names)] for b in range(n_loads)]  # [n_loads]

    jitter_max = float(ds["phase_jitter_hours"])
    phase = rng.uniform(-jitter_max, jitter_max, size=n_loads)    # [n_loads] hours

    # Per-bus daily curve: class shape, phase-shifted, normalised to unit 24-h mean.
    hours_full_day = np.arange(0.0, 24.0, 1.0 / 60.0)             # 1-min grid for the mean
    daily = np.empty((len(index), n_loads), dtype=np.float64)     # [T, n_loads]
    for b in range(n_loads):
        w = tuple(ds["bus_classes"][assigned[b]])
        norm = daily_profile(hours_full_day, ds["daily_profile"], w).mean()
        daily[:, b] = daily_profile((hours + phase[b]) % 24.0, ds["daily_profile"], w) / norm

    # Weekday / weekend. Monday=0 ... Sunday=6.
    is_weekend = (index.dayofweek.values >= 5)                    # [T] bool
    week = np.where(is_weekend, float(ds["weekend_factor"]), 1.0).reshape(-1, 1)  # [T, 1]

    surge, surge_events = demand_surges(len(index), n_loads, ds, steps_per_day, rng)

    noise = ar1_noise(len(index), n_loads, float(ds["noise_ar1_phi"]),
                      float(ds["noise_std"]), rng)                # [T, n_loads]

    mult = float(ds["base_load_scale"]) * daily * week * surge * (1.0 + noise)  # [T, n_loads]
    return np.clip(mult, 0.05, None), assigned, surge_events      # never negative demand


def sample_outages(n_steps: int, candidates: List[int], ds: Dict, steps_per_day: float, rng
                   ) -> List[Dict]:
    """Poisson-arrival N-1 outages. -> list of {start, end, branch_row}.

    Rate `outages_per_day` converted to a per-step probability. Durations are
    uniform in [outage_min_steps, outage_max_steps]. Overlapping outages are
    SUPPRESSED (we only ever simulate N-1, never N-2) — an N-2 contingency on a
    14-bus system is usually a collapse, not an operating point.
    """
    if not candidates:
        return []
    p_step = float(ds["outages_per_day"]) / steps_per_day
    lo, hi = int(ds["outage_min_steps"]), int(ds["outage_max_steps"])
    events: List[Dict] = []
    t, busy_until = 0, -1
    while t < n_steps:
        if t > busy_until and rng.random() < p_step:
            dur = int(rng.integers(lo, hi + 1))
            end = min(t + dur, n_steps)
            events.append({"start": t, "end": end, "branch_row": int(rng.choice(candidates))})
            busy_until = end
            t = end
        else:
            t += 1
    return events


# ---------------------------------------------------------------------------
# 3. The simulation: one AC power flow per timestep
# ---------------------------------------------------------------------------
def simulate(net, branches: pd.DataFrame, index: pd.DatetimeIndex, ds: Dict,
             rng, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Run the time series. -> dict of [T, N] state arrays + the outage log.

    Every timestep: set the load vector, apply any active outage, solve a real
    Newton-Raphson AC power flow, record the bus state. Warm-started from the
    previous solution (`init="results"`) because consecutive 5-min steps are
    nearly identical operating points — that takes the solve from ~12 ms to ~7 ms,
    so 10,000 steps run in about a minute on a laptop.

    Non-convergence -> the whole timestep is recorded as MISSING_SENTINEL rather
    than as an invented number. That is the same honesty rule the traffic
    pipelines use for absent sensor readings, and it keeps masked-MAE valid.
    """
    import pandapower as pp

    n_steps, n_nodes = len(index), len(net.bus)
    base_p = net.load.p_mw.values.astype(np.float64).copy()       # [n_loads]
    base_q = net.load.q_mvar.values.astype(np.float64).copy()     # [n_loads]

    steps_per_day = 24 * 60 / float(load_data_config()["window"]["resolution_minutes"])
    mult, bus_classes, surge_events = build_load_multipliers(
        index, len(base_p), ds, steps_per_day, rng)               # [T, n_loads]

    # The N-1 screen operating point. Two subtleties, both learned the hard way:
    #  (a) It must be a SYSTEM-WIDE demand level, not the worst any single bus
    #      reaches — a local surge at one small bus is not a system peak. So we use
    #      the load-weighted total, not mult.max().
    #  (b) It must be a PERCENTILE, not the absolute maximum. Screening at the
    #      single worst instant in 34 days is absurdly conservative: with demand
    #      surges enabled the absolute max knocked out ALL 20 branches and the
    #      dataset ended up with zero contingencies. Real N-1 security is assessed
    #      against a forecast peak, so we use the 99th percentile of system demand.
    system_mult = mult @ base_p / base_p.sum()                    # [T] load-weighted total
    peak_mult = float(np.percentile(system_mult, float(ds["n1_screen_percentile"])))
    candidates = _outage_candidates(net, branches, peak_mult, base_p, base_q,
                                    bool(ds["enforce_q_lims"]))
    events = sample_outages(n_steps, candidates, ds, steps_per_day, rng)
    # step -> branch row that is out at that step (-1 = intact network)
    out_at = np.full(n_steps, -1, dtype=np.int64)                 # [T]
    for ev in events:
        out_at[ev["start"]:ev["end"]] = ev["branch_row"]

    vm = np.zeros((n_steps, n_nodes), dtype=np.float64)           # [T, N] voltage magnitude, pu
    va = np.zeros((n_steps, n_nodes), dtype=np.float64)           # [T, N] voltage angle, degrees
    p = np.zeros((n_steps, n_nodes), dtype=np.float64)            # [T, N] net active power, MW
    q = np.zeros((n_steps, n_nodes), dtype=np.float64)            # [T, N] net reactive power, MVAr
    diverged = np.zeros(n_steps, dtype=bool)                      # [T]

    enforce = bool(ds["enforce_q_lims"])
    prev_out, warm = -1, False
    for t in range(n_steps):
        net.load["p_mw"] = base_p * mult[t]                       # [n_loads]
        net.load["q_mvar"] = base_q * mult[t]                     # constant power factor

        cur_out = int(out_at[t])
        if cur_out != prev_out:                                   # toggle service state
            for row_i in (prev_out, cur_out):
                if row_i < 0:
                    continue
                br = branches.iloc[row_i]
                tbl = net.line if br["kind"] == "line" else net.trafo
                tbl.at[int(br["elem_idx"]), "in_service"] = (row_i != cur_out)
            prev_out = cur_out
            warm = False                                          # topology changed: cold start

        try:
            pp.runpp(net, enforce_q_lims=enforce, init="results" if warm else "auto")
            vm[t] = net.res_bus.vm_pu.values
            va[t] = net.res_bus.va_degree.values
            p[t] = net.res_bus.p_mw.values
            q[t] = net.res_bus.q_mvar.values
            warm = True
        except Exception:
            diverged[t] = True
            vm[t] = MISSING_SENTINEL                              # honest gap, not a guess
            va[t] = p[t] = q[t] = 0.0
            warm = False

        if verbose and n_steps >= 1000 and (t + 1) % 2000 == 0:
            print(f"[{DATASET}]   power flow {t + 1}/{n_steps} steps "
                  f"({100.0 * (t + 1) / n_steps:.0f}%)")

    # Restore the network to its intact state (defensive: callers may reuse `net`).
    if prev_out >= 0:
        br = branches.iloc[prev_out]
        tbl = net.line if br["kind"] == "line" else net.trafo
        tbl.at[int(br["elem_idx"]), "in_service"] = True

    return {"vm_pu": vm, "va_degree": va, "p_mw": p, "q_mvar": q,
            "diverged": diverged, "out_at": out_at, "events": events,
            "surge_events": surge_events, "bus_classes": bus_classes, "load_mult": mult}


# ---------------------------------------------------------------------------
# 3b. Dataset diagnostics — is there anything here for a GRAPH model to find?
# ---------------------------------------------------------------------------
def graph_signal_diagnostics(vm: np.ndarray, adjacency: np.ndarray) -> Dict:
    """Two numbers that say whether this dataset is worth running a GNN on.

    We report these in stats.json and in the paper because a synthetic dataset can
    very easily be DEGENERATE — collinear enough that a per-node affine map of one
    global signal solves it, leaving the graph (and therefore the explainer, and
    therefore the whole faithfulness measurement) doing no work. Publishing a
    cross-domain grounding result on a degenerate dataset would be worthless.

    1. `pc1_variance_share` — fraction of variance captured by the first principal
       component of the bus voltages, i.e. how much of the dataset is ONE global
       factor. Reference point measured on real METR-LA speeds: 0.387 with all 155
       clean sensors, 0.436 when subsampled to 13 sensors (so it is not a node-count
       artifact). Our grid sits far higher — a genuine, reportable domain
       difference: a 14-bus network is small and tightly coupled, and voltage
       responds near-linearly to loading, whereas traffic congestion is a local
       threshold phenomenon. This is NOT a defect we hid; it is a property of the
       domain that belongs in the paper.

    2. `neighbor_r2` vs `non_neighbor_r2` — the diagnostic that actually decides
       whether the GRAPH matters, and the one we tuned nothing against. Remove PC1
       (the global driver) from every bus, then predict each bus's residual by
       least squares from (a) its graph neighbours and (b) an equal number of
       non-neighbours. If the graph carries information, (a) > (b). We report the
       ratio. NOTE PC1 and this ratio move in OPPOSITE directions — adding local
       demand surges RAISES pc1_variance_share while nearly doubling the neighbour
       ratio — which is exactly why we do not steer on PC1 alone.
    """
    # Drop NEAR-constant buses. The slack bus is the system's voltage reference and
    # is pinned at its setpoint except in the rare intervals where its own reactive
    # limit binds — measured std 5.8e-5 pu, about 1/300th of the next-smallest bus.
    # An absolute epsilon (1e-9) misses that and lets a numerically-degenerate
    # column into the regressions, so the test is RELATIVE to the typical bus.
    sd = vm.std(axis=0)                                           # [N]
    keep = np.where(sd > 0.01 * float(np.median(sd)))[0]          # [n_keep]
    V = vm[:, keep].astype(np.float64)                            # [T, n_keep]
    Vc = V - V.mean(axis=0)
    u, s, vt = np.linalg.svd(Vc, full_matrices=False)
    pc1_share = float((s ** 2 / (s ** 2).sum())[0])
    resid = Vc - np.outer(u[:, 0] * s[0], vt[0])                  # [T, n_keep] de-globalised

    rng = np.random.default_rng(0)
    nb_scores, rand_scores = [], []
    for col, node in enumerate(keep):
        nb = [c for c, j in enumerate(keep) if c != col and adjacency[node, j] > 0]
        non = [c for c, j in enumerate(keep) if c != col and adjacency[node, j] == 0]
        if not nb or len(non) < len(nb):
            continue
        for src, acc in ((nb, nb_scores),
                         (list(rng.choice(non, size=len(nb), replace=False)), rand_scores)):
            P = resid[:, src]                                     # [T, k] predictors
            beta, _, _, _ = np.linalg.lstsq(P, resid[:, col], rcond=None)
            ss_res = float(((resid[:, col] - P @ beta) ** 2).sum())
            ss_tot = float((resid[:, col] ** 2).sum())
            acc.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    nb_r2 = float(np.mean(nb_scores)) if nb_scores else float("nan")
    rand_r2 = float(np.mean(rand_scores)) if rand_scores else float("nan")
    return {
        "pc1_variance_share": round(pc1_share, 4),
        "neighbor_r2": round(nb_r2, 4),
        "non_neighbor_r2": round(rand_r2, 4),
        "neighbor_advantage_ratio": round(nb_r2 / rand_r2, 3) if rand_r2 > 0 else None,
        "near_constant_buses": [int(b) + 1 for b in range(vm.shape[1]) if b not in keep],
    }


# ---------------------------------------------------------------------------
# 4. Node metadata (bus roles) — the "human-readable names" the LLM layer needs
# ---------------------------------------------------------------------------
def bus_metadata(net) -> Tuple[List[str], List[str], List[str]]:
    """(names, roles, zones) per bus index, derived from the network itself.

    The advisor reasons in words, not indices ("the reactive-support zone is
    sagging", not "node 8 influences node 12"), so every bus gets a name built
    from FACTS READ OUT OF THE CASE, never invented:
      role  — slack / generator / synchronous condenser / load / passive
      zone  — voltage-level grouping, which is the grid's analogue of a city region
    """
    n = len(net.bus)
    slack = set(int(b) for b in net.ext_grid.bus.values)
    gens = {int(r.bus): float(r.p_mw) for _, r in net.gen.iterrows()}
    loads = {int(r.bus): float(r.p_mw) for _, r in net.load.iterrows()}
    shunts = set(int(b) for b in net.shunt.bus.values)

    roles, zones, names = [], [], []
    for b in range(n):
        # A bus can play SEVERAL roles at once, and saying only one of them
        # actively misleads the advisor. Bus 3, for instance, hosts a synchronous
        # condenser AND is the largest load in the system (94.2 MW) — describing
        # it as just "synchronous condenser" would hide the thing a planner most
        # needs to know about it. So we compose every role the bus actually has.
        parts: List[str] = []
        if b in slack:
            parts.append("slack bus")
        elif b in gens and gens[b] > 0.0:
            parts.append("generator bus")
        elif b in gens:
            # A "generator" dispatching 0 MW exists only to hold voltage with
            # reactive power — that is a synchronous condenser (case14 has three).
            parts.append("synchronous condenser")
        if b in loads and loads[b] > 0.0:
            parts.append("load bus" if not parts else "load")
        if b in shunts:
            parts.append("shunt capacitor")
        role = " + ".join(parts) if parts else "passive bus"

        kv = float(net.bus.vn_kv[b])
        # Zone = voltage level. In case14 these are 135 kV (transmission core),
        # 14/12 kV (the condenser/step-down tier) and 0.208 kV (the load pocket).
        if kv >= 100.0:
            zone = "HV transmission core"
        elif kv >= 1.0:
            zone = "MV step-down tier"
        else:
            zone = "LV load pocket"
        roles.append(role)
        zones.append(zone)
        # e.g. "LV load pocket (bus 14, load bus, 0.208 kV)"  -- mirrors the
        # traffic NodeNamer's "<region> (<unit> <id>, <lat>, <lon>)" shape.
        names.append(f"{zone} (bus {b + 1}, {role}, {kv:g} kV)")
    return names, roles, zones


# ---------------------------------------------------------------------------
# 5. Build + save
# ---------------------------------------------------------------------------
def build(n_timesteps: Optional[int] = None, verbose: bool = True) -> Dict:
    cfg = load_data_config()
    win, split_cfg = cfg["window"], cfg["split"]
    ds = cfg["datasets"][DATASET]
    n_steps = int(n_timesteps if n_timesteps is not None else ds["n_timesteps"])

    net = load_case14()
    n_nodes = len(net.bus)
    assert n_nodes == int(ds["n_nodes"]), f"expected {ds['n_nodes']} buses, got {n_nodes}"
    branches = branch_table(net)
    assert len(branches) == int(ds["n_branches"]), \
        f"expected {ds['n_branches']} branches, got {len(branches)}"

    # Fixed clock on the SAME 5-min grid as METR-LA, so window/horizon semantics
    # ("12 steps in = last hour, step 6 = the 30-min horizon") carry over exactly.
    res_min = int(win["resolution_minutes"])
    index = pd.date_range(start=str(ds["start_time"]), periods=n_steps,
                          freq=f"{res_min}min")                   # [T]

    rng = np.random.default_rng(int(ds["seed"]))
    with warnings.catch_warnings():
        # pandapower emits deprecation chatter from its pandas internals on 2.x;
        # silenced ONLY around the simulation so real warnings stay visible.
        warnings.simplefilter("ignore")
        sim = simulate(net, branches, index, ds, rng, verbose=verbose)

    vm = sim["vm_pu"].astype(np.float32)                          # [T, N] pu (0.0 = missing)
    missing_pct = float((vm == MISSING_SENTINEL).mean() * 100.0)

    # --- Core features [T, N, 2] = (voltage, time-of-day) --------------------
    # EXACTLY the METR-LA channel layout: channel 0 is the target state, channel 1
    # is the clock. Two channels, not four, is deliberate — the cross-domain claim
    # is "the SAME model, unmodified", so the input contract must not change.
    # Active/reactive power ride along as a Phase-6-style SIDECAR instead (below),
    # which is the mechanism the architecture already has for extra modalities.
    tod = ((index.values - index.values.astype("datetime64[D]"))
           / np.timedelta64(1, "D")).astype(np.float32)           # [T] fraction of day
    tod = np.tile(tod.reshape(-1, 1, 1), (1, n_nodes, 1))         # [T, N, 1]
    features = np.concatenate([vm.reshape(n_steps, n_nodes, 1), tod], axis=2)  # [T, N, 2]

    X, Y = sliding_windows(features, win["input_length"], win["output_length"])
    # X [samples, 12, 14, 2]   Y [samples, 12, 14]
    n_samples = X.shape[0]
    tr, va_idx, te = chronological_split(n_samples, split_cfg["train"], split_cfg["val"])

    scaler = StandardScaler.fit(X[tr][..., 0])                    # TRAIN voltages only
    splits = {}
    for sname, sl in (("train", tr), ("val", va_idx), ("test", te)):
        Xs = X[sl].copy()                                         # [n, 12, N, 2]
        Xs[..., 0] = scaler.transform(Xs[..., 0])                 # normalize voltage channel only
        splits[sname] = {"X": Xs, "Y": scaler.transform(Y[sl])}

    adjacency = build_adjacency(n_nodes, branches, str(ds["adjacency_method"]),
                                float(ds["adjacency_normalized_k"]))
    n_edges = int((adjacency > 0).sum() - n_nodes) // 2           # undirected branch count

    names, roles, zones = bus_metadata(net)
    node_meta = {
        "dataset": DATASET,
        "sensor_ids": [b + 1 for b in range(n_nodes)],            # 1-indexed bus numbers (literature convention)
        "n_nodes": n_nodes,
        "names": names,
        "roles": roles,
        "zones": zones,
        "unit": "bus",
        # No lat/lon: an IEEE test case has no geography (see DIFFERENCES_POWER_GRID.md).
        # Downstream naming must read `names`, not NodeNamer's coordinate lookup.
        "latlon": None,
        "nominal_kv": [float(v) for v in net.bus.vn_kv.values],
        "branches": branches.to_dict("records"),                  # edge features: impedance, static
        "bus_classes": {str(int(r.bus) + 1): c
                        for r, c in zip(net.load.itertuples(index=False), sim["bus_classes"])},
        "target_variable": "voltage_magnitude_pu",
        "missing_sentinel": MISSING_SENTINEL,
        "undervoltage_pu": float(ds["undervoltage_pu"]),
        "nominal_voltage_pu": float(ds["nominal_voltage_pu"]),
    }

    valid = vm[vm != MISSING_SENTINEL]
    stats = {
        "dataset": DATASET, "nodes": n_nodes, "edges": n_edges, "timesteps": n_steps,
        "date_range": f"{index[0]} -> {index[-1]}",
        "missing_data_pct": round(missing_pct, 3),
        "samples_total": n_samples,
        "samples_train": int(X[tr].shape[0]), "samples_val": int(X[va_idx].shape[0]),
        "samples_test": int(X[te].shape[0]),
        "X_shape": list(splits["train"]["X"].shape), "Y_shape": list(splits["train"]["Y"].shape),
        "scaler_mean": round(scaler.mean, 4), "scaler_std": round(scaler.std, 4),
        # Domain-specific sanity numbers a power engineer will want to see.
        "voltage_min_pu": round(float(valid.min()), 4) if valid.size else None,
        "voltage_max_pu": round(float(valid.max()), 4) if valid.size else None,
        "undervoltage_steps_pct": round(float(
            (np.any((vm < float(ds["undervoltage_pu"])) & (vm != MISSING_SENTINEL), axis=1)).mean()
            * 100.0), 3),
        "n_outage_events": len(sim["events"]),
        "outage_steps_pct": round(float((sim["out_at"] >= 0).mean() * 100.0), 3),
        "n_demand_surges": len(sim["surge_events"]),
        "diverged_steps": int(sim["diverged"].sum()),
        "note": "SYNTHETIC demand + REAL AC power flow on the IEEE 14-bus case; "
                "see DIFFERENCES_POWER_GRID.md",
    }
    # Is there anything here for a GRAPH model to find? (see the docstring — these
    # are reported honestly, including the ways this domain is unlike traffic).
    stats.update(graph_signal_diagnostics(vm, adjacency))

    save_processed_dataset(DATASET, splits, adjacency, scaler.to_dict(), node_meta, stats)

    # --- Sidecar modality: active + reactive power ---------------------------
    # Phase-6 mechanism, reused verbatim: extra node features attach as a sidecar
    # rather than by widening X. MOD-3's per-modality gate handles them, and a
    # traffic-only checkpoint transferred here simply passes None and is unharmed.
    save_modality_sidecar(
        dataset=DATASET, name="power",
        raw_feature=np.stack([sim["p_mw"], sim["q_mvar"]], axis=2).astype(np.float32),  # [T,N,2]
        channels=["p_mw", "q_mvar"],
    )

    # --- Raw state dump ------------------------------------------------------
    # The power flow is the expensive part (~1 min for 10k steps). Persist the full
    # solved state + the outage log so any later analysis (scenario stratification,
    # counterfactuals, a different channel choice) never has to re-simulate.
    rdir = raw_dir(DATASET)
    np.savez_compressed(
        os.path.join(rdir, "raw_state.npz"),
        vm_pu=sim["vm_pu"], va_degree=sim["va_degree"],
        p_mw=sim["p_mw"], q_mvar=sim["q_mvar"],
        diverged=sim["diverged"], out_at=sim["out_at"],
        load_mult=sim["load_mult"],
        timestamps=index.astype("datetime64[s]").astype(np.int64),
    )
    # Event logs. These are the "ground truth" of what actually happened at each
    # step — the analogue of a traffic incident report — so downstream phases can
    # stratify scenarios by stress instead of guessing from the voltages alone.
    load_bus = [int(b) + 1 for b in net.load.bus.values]          # load idx -> bus number
    with open(os.path.join(rdir, "outage_log.json"), "w") as f:
        json.dump([{**ev,
                    "branch": f"{branches.iloc[ev['branch_row']]['kind']} "
                              f"{int(branches.iloc[ev['branch_row']]['from_bus']) + 1}-"
                              f"{int(branches.iloc[ev['branch_row']]['to_bus']) + 1}",
                    "start_time": str(index[ev["start"]]),
                    "end_time": str(index[min(ev["end"], n_steps - 1)])}
                   for ev in sim["events"]], f, indent=2)
    with open(os.path.join(rdir, "surge_log.json"), "w") as f:
        json.dump([{**ev, "bus": load_bus[ev["bus_load_idx"]],
                    "start_time": str(index[ev["start"]]),
                    "end_time": str(index[min(ev["end"], n_steps - 1)])}
                   for ev in sim["surge_events"]], f, indent=2)

    print_stats_report(stats)
    return stats


# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------
def smoke() -> None:
    """Load the network, generate 100 timesteps, build one sample tensor, print shapes."""
    print("=" * 68)
    print("  POWER GRID SMOKE TEST — IEEE 14-bus")
    print("=" * 68)
    cfg = load_data_config()
    ds = cfg["datasets"][DATASET]

    net = load_case14()
    branches = branch_table(net)
    print(f"  buses           : {len(net.bus)}")
    print(f"  branches        : {len(branches)} "
          f"({int((branches['kind'] == 'line').sum())} lines + "
          f"{int((branches['kind'] == 'trafo').sum())} transformers)")
    print(f"  generators      : {len(net.gen)} + 1 slack   loads: {len(net.load)}   "
          f"shunts: {len(net.shunt)}")
    print(f"  |Z| per-unit    : min {branches['z_pu'].min():.4f}  "
          f"max {branches['z_pu'].max():.4f}")

    for method in ("admittance", "gaussian"):
        adj = build_adjacency(len(net.bus), branches, method,
                              float(ds["adjacency_normalized_k"]))
        # Weight actually assigned to each REAL branch (excludes the 1.0 diagonal).
        edge_w = np.array([adj[r.from_bus, r.to_bus] for r in branches.itertuples(index=False)])
        weak = int((edge_w < 0.02).sum())     # branches effectively deleted from message passing
        mark = ("  <- USED" if method == str(ds["adjacency_method"])
                else f"  <- {weak}/{len(branches)} real branches crushed below 0.02")
        print(f"  adjacency[{method:<10}]: {adj.shape}  undirected edges "
              f"{int((adj > 0).sum() - len(net.bus)) // 2}  "
              f"branch weight {edge_w.min():.4f}..{edge_w.max():.4f}{mark}")
    adj = build_adjacency(len(net.bus), branches, str(ds["adjacency_method"]),
                          float(ds["adjacency_normalized_k"]))

    names, _, _ = bus_metadata(net)
    print("  bus names (first 4):")
    for nm in names[:4]:
        print(f"      {nm}")

    res_min = int(cfg["window"]["resolution_minutes"])
    index = pd.date_range(start=str(ds["start_time"]), periods=100, freq=f"{res_min}min")
    rng = np.random.default_rng(int(ds["seed"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simulate(net, branches, index, ds, rng, verbose=False)

    vm = sim["vm_pu"].astype(np.float32)                          # [100, 14]
    valid = vm[vm != MISSING_SENTINEL]
    print(f"  simulated       : {vm.shape} [T, N]  voltage "
          f"{valid.min():.4f}..{valid.max():.4f} pu   diverged {int(sim['diverged'].sum())}")

    tod = ((index.values - index.values.astype("datetime64[D]"))
           / np.timedelta64(1, "D")).astype(np.float32)
    tod = np.tile(tod.reshape(-1, 1, 1), (1, len(net.bus), 1))    # [100, 14, 1]
    feats = np.concatenate([vm.reshape(100, len(net.bus), 1), tod], axis=2)  # [100, 14, 2]
    X, Y = sliding_windows(feats, cfg["window"]["input_length"], cfg["window"]["output_length"])
    print(f"  X shape         : {X.shape}   (expect [samples, 12, 14, 2])")
    print(f"  Y shape         : {Y.shape}   (expect [samples, 12, 14])")
    print(f"  one sample X[0] : {X[0].shape}   Y[0]: {Y[0].shape}")
    assert X.shape[1:] == (12, 14, 2) and Y.shape[1:] == (12, 14), "tensor contract broken"
    print("  CONTRACT OK — identical to METR-LA except N (14 vs 207).")
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 19 — IEEE 14-bus power grid pipeline")
    ap.add_argument("--smoke", action="store_true",
                    help="load network, simulate 100 steps, print shapes; write nothing")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="override n_timesteps from configs/data.yaml")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return
    build(n_timesteps=args.timesteps)
    print(f"[{DATASET}] done -> {processed_dir(DATASET)}")


if __name__ == "__main__":
    main()
