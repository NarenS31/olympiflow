"""Sensor -> human-readable location names, built OFFLINE from lat/lon.

WHY this matters: the LLM advisory layer (Phase 4) reasons in words, not node
indices. "Congestion is spreading from the Downtown LA sensors toward the I-10 /
Santa Monica corridor" is usable by a planner; "node 88 influences node 42" is
not. Human-readable names are the bridge that makes the whole advisory idea work,
and later they are what the faithfulness metric resolves the LLM's text back to.

WHY not a real reverse-geocoder: CLAUDE.md forbids external API calls and demands
reproducibility. An online geocoder would break both (network dependence,
drifting results). Instead we tag each sensor by which LA REGION its coordinates
fall in, using a small fixed table of bounding boxes drawn from public LA
geography. The result is deterministic, offline, and good enough to ground the
LLM: it gives a corridor/area label plus the raw sensor id and coordinates.

If a sensor sits outside every box we fall back to a compass label relative to
Downtown LA (34.05, -118.25), so every node always gets *some* name.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Approximate lat/lon bounding boxes for LA-basin regions the METR-LA sensors
# cover (min_lat, max_lat, min_lon, max_lon). Order matters: first match wins,
# so put smaller/more-specific areas before broad ones. Boxes overlap slightly;
# that is fine — the label is a human hint, not a legal boundary.
_LA_REGIONS: List[Tuple[str, float, float, float, float]] = [
    ("Downtown LA",            34.02, 34.075, -118.27, -118.22),
    ("Hollywood",              34.08, 34.13,  -118.36, -118.30),
    ("Koreatown / Mid-City",   34.03, 34.075, -118.32, -118.27),
    ("Santa Monica / Westside",33.99, 34.06,  -118.52, -118.42),
    ("West LA / Sawtelle",     34.02, 34.09,  -118.47, -118.40),
    ("San Fernando Valley",    34.14, 34.32,  -118.60, -118.35),
    ("Glendale / Burbank",     34.13, 34.22,  -118.35, -118.22),
    ("Pasadena / East",        34.11, 34.20,  -118.20, -118.05),
    ("East LA",                34.02, 34.08,  -118.22, -118.12),
    ("South LA",               33.90, 34.02,  -118.35, -118.22),
    ("Inglewood / LAX",        33.90, 33.99,  -118.42, -118.30),
    ("Long Beach corridor",    33.75, 33.90,  -118.25, -118.10),
]

# Chicago regions (public neighbourhood/expressway geography). Reference = the Loop.
# Added in Phase 6 so the cross-city faithfulness study gets real Chicago names
# instead of nonsensical LA labels for Chicago coordinates.
_CHICAGO_REGIONS: List[Tuple[str, float, float, float, float]] = [
    ("The Loop / Downtown",              41.870, 41.890, -87.640, -87.615),
    ("South Loop",                       41.840, 41.870, -87.640, -87.605),
    ("Near North Side",                  41.890, 41.920, -87.650, -87.610),
    ("Near West Side",                   41.870, 41.895, -87.685, -87.640),
    ("Lincoln Park / North Side",        41.900, 41.945, -87.680, -87.620),
    ("Lakeview / Uptown",                41.940, 41.985, -87.680, -87.635),
    ("West Side / Garfield Park",        41.855, 41.900, -87.735, -87.685),
    ("Bronzeville / South Side",         41.795, 41.845, -87.635, -87.600),
    ("Hyde Park / Woodlawn",             41.775, 41.810, -87.615, -87.580),
    ("Southwest Side / Midway",          41.760, 41.810, -87.755, -87.680),
    ("Northwest Side / O'Hare corridor", 41.935, 42.010, -87.870, -87.745),
    ("Far South Side",                   41.660, 41.775, -87.685, -87.590),
    # --- Phase-11 correction (FLAGGED — CHANGES committed Chicago numbers) ----
    # The 12 boxes above covered only 55.5% of the 1020 segments; the other 454
    # (44.5%) fell to the _compass() fallback, the same defect that made PEMS-BAY
    # unmeasurable. Its largest sector held 228 segments (22.4% of the graph), so a
    # single coarse citation bought region-level credit over a fifth of the city.
    # These 15 boxes are real Chicago community areas covering that gap. They are
    # APPENDED, and region_for() is first-match-wins, so all 12 boxes above keep
    # EXACTLY the segments they had — verified: 0 nodes changed label.
    # Result: compass fallback 44.5% -> 8.4%, largest region 22.4% -> 12.6%.
    ("Rogers Park / Edgewater",          41.980, 42.025, -87.715, -87.640),
    ("West Ridge / Lincoln Square",      41.955, 42.012, -87.730, -87.680),
    ("Irving Park / Albany Park",        41.938, 41.980, -87.745, -87.680),
    ("Logan Square / Avondale",          41.910, 41.952, -87.735, -87.680),
    ("Hermosa / Belmont Cragin",         41.900, 41.955, -87.800, -87.735),
    ("Austin / West Garfield Park",      41.868, 41.912, -87.800, -87.735),
    ("North Lawndale / Little Village",  41.832, 41.872, -87.745, -87.685),
    ("Bridgeport / McKinley Park",       41.818, 41.855, -87.688, -87.625),
    ("Brighton Park / Gage Park",        41.795, 41.835, -87.735, -87.670),
    ("Back of the Yards / New City",     41.785, 41.822, -87.688, -87.625),
    ("Garfield Ridge / Clearing",        41.765, 41.820, -87.820, -87.735),
    ("Chicago Lawn / West Lawn",         41.745, 41.800, -87.740, -87.680),
    ("Englewood / Washington Park",      41.755, 41.800, -87.680, -87.600),
    ("Ashburn / Auburn Gresham",         41.715, 41.762, -87.745, -87.640),
    ("South Shore / South Chicago",      41.700, 41.780, -87.615, -87.530),
]

# PEMS-BAY regions (public Santa Clara Valley geography). Reference = Downtown
# San Jose.
#
# WHY THIS EXISTS (Phase 11 correction, FLAGGED — this CHANGES committed numbers).
# PEMS-BAY had NO entry here, so all 325 sensors fell through to the _compass()
# fallback and collapsed into 9 sectors. That silently broke the faithfulness
# metric in two ways:
#   (1) SET SIZE. The biggest sector held 122/325 = 37.5% of the graph, and the
#       resolver grants region-level credit to the WHOLE set. One citation of
#       "NW of Downtown San Jose" therefore intersected the explainer's top-k
#       almost automatically, scoring a "hit" with zero causal information.
#   (2) LABEL COLLAPSE. The 9 labels differ only by a 1-2 character compass
#       prefix, so _ratio("nw of downtown san jose", "n of downtown san jose")
#       ~ 92 >= FUZZY_THRESHOLD (82). The fuzzy rung matched an ARBITRARY sector
#       — observed live: a citation of "Downtown San Jose" for a target in the
#       WEST sector resolved to the EAST sector and still counted as a hit.
# The boxes below were accepted against both failure modes: largest region is
# 10.8% of the graph (was 37.5%), zero sensors fall to the compass fallback, and
# no two labels score >= 82 against each other (closest pair 70.6).
#
# FOOTPRINT HONESTY: PEMS-BAY spans only lat 37.250-37.428, lon -122.080 to
# -121.840 — the Santa Clara Valley. It contains NO sensors within 5 km of San
# Francisco, Oakland, Berkeley, Fremont or Palo Alto, so this table deliberately
# defines no boxes for them. Naming regions the data does not cover would be
# fabricated geography, which is exactly the error class this metric measures.
_PEMS_BAY_REGIONS: List[Tuple[str, float, float, float, float]] = [
    # Specific/small areas first (first match wins), broad ones after — same
    # contract as _LA_REGIONS / _CHICAGO_REGIONS above.
    ("San Jose Airport / US-101 junction", 37.352, 37.392, -121.950, -121.890),
    ("Downtown San Jose",                  37.310, 37.352, -121.918, -121.868),
    ("Santa Clara / Great America",        37.330, 37.425, -121.998, -121.950),
    ("North San Jose / Alviso",            37.380, 37.440, -121.950, -121.900),
    ("Milpitas / I-880 corridor",          37.398, 37.445, -121.900, -121.855),
    ("Berryessa / East San Jose",          37.336, 37.400, -121.900, -121.835),
    ("Evergreen / Silver Creek",           37.275, 37.340, -121.868, -121.835),
    ("Blossom Valley / South San Jose",    37.240, 37.312, -121.900, -121.835),
    ("Willow Glen / Cambrian",             37.240, 37.318, -121.960, -121.900),
    ("West San Jose / Stevens Creek",      37.296, 37.352, -121.995, -121.918),
    ("Sunnyvale / Lawrence Expressway",    37.340, 37.435, -122.058, -121.995),
    ("Mountain View / SR-237 west",        37.352, 37.445, -122.090, -122.058),
    ("Cupertino / I-280 corridor",         37.288, 37.360, -122.090, -121.995),
    ("Campbell / Los Gatos SR-17",         37.240, 37.296, -122.090, -121.960),
]

# --- Phase 19: the IEEE 14-bus power grid (NO GEOGRAPHY AT ALL) -------------
# An IEEE test case is a circuit, not a place: its buses have no lat/lon, so the
# coordinate -> region-box path above simply does not apply. Cities listed here
# take a completely separate naming path in NodeNamer and NEVER touch the
# lat/lon code, which is why METR-LA / PEMS-BAY / Chicago behaviour is provably
# unchanged by this addition.
_NO_GEOMETRY_CITIES = {"power_grid"}

# bus number (1-indexed, as the literature numbers them) -> (zone, role).
#
# ROLES ARE THE VERIFIED ONES, read out of pandapower's case14 during the Phase-19
# build — NOT the common shorthand "buses 4-14 are load buses". Three corrections
# that matter, because a wrong role here would feed the advisor a false fact and
# this phase exists to measure exactly that kind of error:
#   * bus 3 hosts a synchronous condenser AND is the LARGEST load in the system
#     (94.2 MW). Calling it only a condenser hides the thing a planner most needs.
#   * buses 6 and 8 are synchronous condensers, not plain load buses (bus 6 also
#     carries 11.2 MW of demand; bus 8 carries none).
#   * bus 7 is PASSIVE — no demand and no generation. It is a transformer junction.
# Zones are the voltage tiers the pipeline emits, so these strings line up with
# the region_tags in models/advisor/kb/power_grid.json.
_POWER_GRID_BUSES: Dict[int, Tuple[str, str]] = {
    1:  ("HV transmission core", "slack bus"),
    2:  ("HV transmission core", "generator bus + load"),
    3:  ("HV transmission core", "synchronous condenser + load"),
    4:  ("HV transmission core", "load bus"),
    5:  ("HV transmission core", "load bus"),
    6:  ("LV load pocket",       "synchronous condenser + load"),
    7:  ("MV step-down tier",    "passive bus"),
    8:  ("MV step-down tier",    "synchronous condenser"),
    9:  ("LV load pocket",       "load bus + shunt capacitor"),
    10: ("LV load pocket",       "load bus"),
    11: ("LV load pocket",       "load bus"),
    12: ("LV load pocket",       "load bus"),
    13: ("LV load pocket",       "load bus"),
    14: ("LV load pocket",       "load bus"),
}


def power_grid_bus_name(bus_number: int) -> str:
    """Human-readable name for one IEEE 14-bus bus. Offline, deterministic.

    Mirrors the traffic name shape "<region> (<unit> <id>, ...)" so the advisor
    prompt and the Phase-5 entity resolver see a familiar structure:
        "LV load pocket (bus 14, load bus)"
    """
    zone, role = _POWER_GRID_BUSES.get(bus_number, ("unknown zone", "bus"))
    return "{} (bus {}, {})".format(zone, bus_number, role)


# Per-city centre for the compass fallback (lat, lon, human name).
_CITY_REF: Dict[str, Tuple[float, float, str]] = {
    "metr_la":  (34.05, -118.25, "Downtown LA"),
    "pems_bay": (37.34, -121.89, "Downtown San Jose"),
    "chicago":  (41.8781, -87.6298, "the Loop"),
}
# Per-city bounding-box table. Cities absent here fall back to compass labels.
_CITY_REGIONS: Dict[str, List[Tuple[str, float, float, float, float]]] = {
    "metr_la":  _LA_REGIONS,
    "chicago":  _CHICAGO_REGIONS,
    # Phase 11 correction (FLAGGED): PEMS-BAY previously had no entry and fell to
    # _compass(). METR-LA and Chicago look up their own keys, so adding this one
    # cannot alter their names — proven by evaluation/verify_traffic_unchanged.py.
    "pems_bay": _PEMS_BAY_REGIONS,
}


def _compass(lat: float, lon: float, city: str = "metr_la") -> str:
    """Coarse compass direction of (lat, lon) relative to the city centre."""
    ref_lat, ref_lon, ref_name = _CITY_REF.get(city, _CITY_REF["metr_la"])
    dlat, dlon = lat - ref_lat, lon - ref_lon
    ns = "N" if dlat > 0.01 else ("S" if dlat < -0.01 else "")
    ew = "E" if dlon > 0.01 else ("W" if dlon < -0.01 else "")
    where = (ns + ew) or "central"
    return f"{where} of {ref_name}"


def region_for(lat: float, lon: float, city: str = "metr_la") -> str:
    """Region label for a coordinate in `city`. Defaults to LA so every existing
    Phase 3/4/5 caller keeps its exact prior behaviour."""
    for name, lo_lat, hi_lat, lo_lon, hi_lon in _CITY_REGIONS.get(city, []):
        if lo_lat <= lat <= hi_lat and lo_lon <= lon <= hi_lon:
            return name
    return _compass(lat, lon, city)


class NodeNamer:
    """Maps node index -> a stable human-readable name, built from node_meta.

    node_meta is the Phase-1 output: {"sensor_ids": [...], "latlon": [[lat,lon],...]}
    with one entry per node index (index i == row i of the adjacency / tensors).
    """

    def __init__(self, node_meta: Dict):
        self.sensor_ids: List[int] = list(node_meta["sensor_ids"])
        # City drives which region table region_for uses (Phase 6 cross-city).
        # Defaults to metr_la so existing Phase 3/4/5 behaviour is unchanged.
        self.city: str = node_meta.get("dataset", "metr_la")
        self._cache: Dict[int, str] = {}

        # Phase 19 — datasets with NO geography take a separate path entirely.
        # node_meta["latlon"] is null for them, so reading it would crash; and a
        # region box drawn over LA is meaningless for a circuit. Everything below
        # the `else` is the ORIGINAL code, untouched, so METR-LA / PEMS-BAY /
        # Chicago naming is byte-identical to Phase 3/4/5/6/11.
        self._explicit_names: Optional[List[str]] = None
        if self.city in _NO_GEOMETRY_CITIES:
            # Prefer the names the pipeline composed from the live network (they
            # carry the kV level too); fall back to the static table in this file
            # so the namer still works with no processed data on disk.
            names = node_meta.get("names")
            self._explicit_names = list(names) if names else None
            self.latlon: List[List[float]] = []
            self.unit: str = "bus"
        else:
            self.latlon = list(node_meta["latlon"])
            # Chicago nodes are road SEGMENTS, not point sensors — label them honestly.
            self.unit = "segment" if self.city == "chicago" else "sensor"

    def name(self, node_id: int) -> str:
        if node_id in self._cache:
            return self._cache[node_id]

        if self.city in _NO_GEOMETRY_CITIES:
            if self._explicit_names is not None and node_id < len(self._explicit_names):
                label = self._explicit_names[node_id]
            else:
                label = power_grid_bus_name(int(self.sensor_ids[node_id]))
            self._cache[node_id] = label
            return label

        lat, lon = self.latlon[node_id]
        sid = self.sensor_ids[node_id]
        region = region_for(lat, lon, self.city)
        # e.g. "Downtown LA (sensor 773869, 34.045, -118.240)"
        label = f"{region} ({self.unit} {sid}, {lat:.3f}, {lon:.3f})"
        self._cache[node_id] = label
        return label
