import math

# Coliseum-specific BPR constants
_COLISEUM_EVENT_HOUR = 18.0      # 6 PM peak arrival
_COLISEUM_BG_INTENSITY = 0.8     # background PM-peak traffic intensity
_COLISEUM_SURGE = 1.0            # full-capacity event surge
_COLISEUM_STAGGER_REDUCTION = 0.25  # 3-hr stagger reduces effective v/c by 25%
_BASELINE_TRAVEL_MIN = 20.0      # baseline free-flow travel time (minutes)

# LA28 venue coordinates for server-side simulation
VENUES = {
    "sofi":          (-118.3378, 33.9533, 70240),
    "intuit-dome":   (-118.3416, 33.9583, 18000),
    "crypto-arena":  (-118.2674, 34.0430, 20000),
    "la-coliseum":   (-118.2879, 34.0141, 77500),
    "rose-bowl":     (-118.1677, 34.1614, 92542),
    "pauley":        (-118.4436, 34.0702, 13800),
    "bmo-stadium":   (-118.2840, 34.0131, 22000),
    "long-beach-arena": (-118.1887, 33.7697, 13500),
    "sepulveda-basin": (-118.4760, 34.1820, 25000),
    "el-dorado":     (-118.0614, 33.8194, 8000),
    "dignity-health": (-118.2614, 33.8636, 30000),
    "ucla-olympic":  (-118.4452, 34.0689, 15000),
}


def get_time_multiplier(hour: float) -> float:
    if hour < 6:
        return 0.2
    if hour < 9:
        return 0.5 + ((hour - 6) / 3) * 0.5
    if hour < 15:
        return 0.6
    if hour < 19:
        return 0.7 + ((hour - 15) / 4) * 0.3
    if hour < 22:
        return 0.55
    return 0.25


def bpr_delay(volume_ratio: float, alpha: float = 0.15, beta: float = 4) -> float:
    """Bureau of Public Roads travel-time ratio."""
    return 1.0 + alpha * (volume_ratio ** beta)


def run_simulation_step(
    mode: str,
    time_of_day: float,
    global_intensity: float,
    venue_surges: dict[str, float],
) -> dict:
    time_mult = get_time_multiplier(time_of_day)

    surge_values = list(venue_surges.values())
    total_surge = sum(surge_values)
    active_surges = len([v for v in surge_values if v > 0])

    weighted_surge = total_surge / max(1, active_surges) if active_surges else 0

    # Effective demand ratio on the road network
    volume_ratio = min(
        1.5,
        global_intensity * time_mult
        + weighted_surge * 0.45
        + (0.1 if mode == "event" else 0.25 if mode == "crisis" else 0),
    )

    congestion_score = min(1.0, volume_ratio / 1.5)
    delay_ratio = bpr_delay(volume_ratio)
    avg_delay_pct = round((delay_ratio - 1.0) * 100, 1)

    peak_zones = int(congestion_score * 14 + active_surges * 2.5)
    affected_routes = int(congestion_score * 22 + active_surges * 3)
    persons_affected = int(
        (congestion_score * 200_000 + total_surge * 45_000) * time_mult
    )

    return {
        "congestionScore": round(congestion_score, 3),
        "avgDelayIncrease": avg_delay_pct,
        "peakZones": peak_zones,
        "affectedRoutes": affected_routes,
        "personsAffected": persons_affected,
    }


def compute_coliseum_congestion() -> dict:
    """
    BPR congestion model for LA Memorial Coliseum at two demand states.

    Baseline: full-capacity event with all arrivals in the 1-hour peak window.
    Intervention: same event with arrivals staggered across a 3-hour window,
    modelled as a 25% reduction in effective v/c demand rate.

    Returns both states plus the % travel-time improvement from the intervention.
    """
    time_mult = get_time_multiplier(_COLISEUM_EVENT_HOUR)

    # v/c built the same way run_simulation_step does for an event-mode step
    vc_baseline = min(
        1.5,
        _COLISEUM_BG_INTENSITY * time_mult
        + _COLISEUM_SURGE * 0.45
        + 0.1,   # event-mode offset (same constant used in run_simulation_step)
    )
    vc_intervention = vc_baseline * (1.0 - _COLISEUM_STAGGER_REDUCTION)

    def _state(label: str, vc: float) -> dict:
        bpr = bpr_delay(vc)
        return {
            "label": label,
            "vc_ratio": round(vc, 4),
            "bpr_multiplier": round(bpr, 4),
            "congestion_index": round(min(100.0, vc / 1.5 * 100), 1),
            "estimated_travel_time_min": round(_BASELINE_TRAVEL_MIN * bpr, 1),
        }

    baseline = _state("Peak 1-hour arrival (full capacity)", vc_baseline)
    intervention = _state("Staggered 3-hour arrival (-25% v/c)", vc_intervention)

    bpr_b = baseline["bpr_multiplier"]
    bpr_i = intervention["bpr_multiplier"]
    reduction_pct = round((bpr_b - bpr_i) / bpr_b * 100, 1)

    return {
        "venue": "LA Memorial Coliseum",
        "capacity": VENUES["la-coliseum"][2],
        "event_hour": _COLISEUM_EVENT_HOUR,
        "baseline": baseline,
        "intervention": intervention,
        "reduction_pct": reduction_pct,
    }
