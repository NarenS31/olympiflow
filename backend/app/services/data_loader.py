import re
import csv
import math
from pathlib import Path
from functools import lru_cache
from typing import Any

DATA_DIR = Path(__file__).parents[3] / "data"

TRAFFIC_FILE = DATA_DIR / "LADOT_Traffic_Counts_Summary_20260512.csv"
COLLISION_FILE = DATA_DIR / "Traffic_Collision_Data_from_2010_to_Present_20260512.csv"
TRANSIT_FILE = DATA_DIR / "Downtown_DASH_Routes_20260512.csv"
PARKING_FILE = DATA_DIR / "LADOT_Parking_Meter_Occupancy_20260512.csv"


def _parse_coord_string(s: str) -> tuple[float, float] | None:
    """Parse '(lat, lng)' strings from collision data."""
    m = re.match(r"\((-?[\d.]+),\s*(-?[\d.]+)\)", s.strip())
    if not m:
        return None
    lat, lng = float(m.group(1)), float(m.group(2))
    # Basic LA bounding box check
    if not (33.5 <= lat <= 34.5 and -118.8 <= lng <= -117.8):
        return None
    return lat, lng


def _wkt_multilinestring_to_geojson(wkt: str) -> dict[str, Any] | None:
    """Convert WKT MULTILINESTRING to GeoJSON geometry."""
    try:
        inner = re.sub(r"^MULTILINESTRING\s*\(\(", "", wkt.strip())
        inner = re.sub(r"\)\)$", "", inner)
        line_strings = re.split(r"\)\s*,\s*\(", inner)
        coordinates = []
        for ls in line_strings:
            pts = []
            for pair in ls.strip().split(","):
                parts = pair.strip().split()
                if len(parts) >= 2:
                    try:
                        pts.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
            if pts:
                coordinates.append(pts)
        if not coordinates:
            return None
        return {"type": "MultiLineString", "coordinates": coordinates}
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_collision_heatmap() -> dict[str, Any]:
    """Return GeoJSON FeatureCollection of traffic collision hotspots."""
    features: list[dict[str, Any]] = []
    if not COLLISION_FILE.exists():
        return {"type": "FeatureCollection", "features": []}

    # Sample up to 4000 rows for performance (file can be very large)
    max_rows = 4000
    seen = 0
    with open(COLLISION_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if seen >= max_rows:
                break
            loc = row.get("Location", "")
            coord = _parse_coord_string(loc)
            if coord is None:
                continue
            lat, lng = coord
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "properties": {"weight": 0.45},
            })
            seen += 1

    return {"type": "FeatureCollection", "features": features}


@lru_cache(maxsize=1)
def load_transit_routes() -> dict[str, Any]:
    """Return GeoJSON FeatureCollection of DASH transit routes."""
    features: list[dict[str, Any]] = []
    if not TRANSIT_FILE.exists():
        return {"type": "FeatureCollection", "features": []}

    ROUTE_COLORS = {
        "A": "#06b6d4",
        "B": "#8b5cf6",
        "C": "#f59e0b",
        "D": "#10b981",
        "E": "#ef4444",
        "F": "#3b82f6",
    }

    with open(TRANSIT_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wkt = row.get("the_geom", "")
            geom = _wkt_multilinestring_to_geojson(wkt)
            if geom is None:
                continue
            route_name_s = row.get("RouteNameS", "")
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "routeId": row.get("RouteID", ""),
                    "name": row.get("RouteName", ""),
                    "shortName": route_name_s,
                    "color": ROUTE_COLORS.get(route_name_s, "#06b6d4"),
                    "region": row.get("Region", ""),
                },
            })

    return {"type": "FeatureCollection", "features": features}


@lru_cache(maxsize=1)
def load_traffic_stats() -> dict[str, Any]:
    """Return aggregated traffic count statistics."""
    if not TRAFFIC_FILE.exists():
        return {"topIntersections": [], "totalCount": 0, "avgDailyVolume": 0.0}

    counts: dict[str, int] = {}
    total_rows = 0

    with open(TRAFFIC_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            street = row.get("Primary Street", "").strip()
            cross = row.get("Cross Street", "").strip()
            total_str = row.get("Total", "0").replace(",", "").strip()
            try:
                total = int(total_str)
            except ValueError:
                continue
            key = f"{street} @ {cross}"
            counts[key] = counts.get(key, 0) + total
            total_rows += 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    all_volumes = list(counts.values())
    avg = sum(all_volumes) / len(all_volumes) if all_volumes else 0

    return {
        "topIntersections": [
            {"street": k, "total": v} for k, v in sorted_counts
        ],
        "totalCount": total_rows,
        "avgDailyVolume": round(avg, 1),
    }


@lru_cache(maxsize=1)
def load_parking_summary() -> dict[str, Any]:
    """Return parking occupancy summary as synthetic GeoJSON."""
    if not PARKING_FILE.exists():
        return {"type": "FeatureCollection", "features": []}

    occupied = 0
    vacant = 0
    with open(PARKING_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row.get("OccupancyState", "").upper()
            if state == "OCCUPIED":
                occupied += 1
            elif state == "VACANT":
                vacant += 1

    # Generate synthetic parking zone markers around downtown LA
    # (LADOT parking meter CSV does not include spatial coordinates)
    DOWNTOWN_ZONES = [
        (-118.2437, 34.0522, "Downtown Core"),
        (-118.2674, 34.0430, "Staples/Arena District"),
        (-118.2879, 34.0141, "Exposition Park"),
        (-118.4436, 34.0702, "Westwood/UCLA"),
        (-118.3378, 33.9533, "Inglewood/SoFi"),
    ]

    total = occupied + vacant
    occ_rate = occupied / total if total > 0 else 0.5
    features: list[dict[str, Any]] = []
    for i, (lng, lat, name) in enumerate(DOWNTOWN_ZONES):
        zone_rate = min(1.0, occ_rate + (i * 0.05 - 0.1))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": {
                "name": name,
                "occupancyRate": round(zone_rate, 2),
                "occupied": int(occupied / len(DOWNTOWN_ZONES)),
                "vacant": int(vacant / len(DOWNTOWN_ZONES)),
            },
        })

    return {"type": "FeatureCollection", "features": features}
