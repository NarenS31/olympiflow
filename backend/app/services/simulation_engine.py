import math

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
