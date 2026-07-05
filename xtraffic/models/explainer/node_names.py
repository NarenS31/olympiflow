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

from typing import Dict, List, Tuple

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
]

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
        self.latlon: List[List[float]] = list(node_meta["latlon"])
        # City drives which region table region_for uses (Phase 6 cross-city).
        # Defaults to metr_la so existing Phase 3/4/5 behaviour is unchanged.
        self.city: str = node_meta.get("dataset", "metr_la")
        # Chicago nodes are road SEGMENTS, not point sensors — label them honestly.
        self.unit: str = "segment" if self.city == "chicago" else "sensor"
        self._cache: Dict[int, str] = {}

    def name(self, node_id: int) -> str:
        if node_id in self._cache:
            return self._cache[node_id]
        lat, lon = self.latlon[node_id]
        sid = self.sensor_ids[node_id]
        region = region_for(lat, lon, self.city)
        # e.g. "Downtown LA (sensor 773869, 34.045, -118.240)"
        label = f"{region} ({self.unit} {sid}, {lat:.3f}, {lon:.3f})"
        self._cache[node_id] = label
        return label
