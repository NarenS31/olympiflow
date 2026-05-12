import csv
from pathlib import Path
from functools import lru_cache
from typing import Any

DATA_DIR = Path(__file__).parents[3] / "data"
CRIME_FILE = DATA_DIR / "Crime_Data_from_2020_to_2024_20260512.csv"

_SEVERITY: dict[str, float] = {
    "HOMICIDE": 1.0,
    "RAPE": 0.9,
    "ROBBERY": 0.8,
    "ASSAULT WITH DEADLY WEAPON": 0.78,
    "AGGRAVATED ASSAULT": 0.75,
    "KIDNAPPING": 0.85,
    "ARSON": 0.7,
    "BURGLARY": 0.5,
    "VEHICLE - STOLEN": 0.4,
    "THEFT FROM MOTOR VEHICLE": 0.35,
    "GRAND THEFT": 0.35,
    "VANDALISM": 0.22,
    "THEFT OF IDENTITY": 0.15,
}


def _weight(crm_desc: str) -> float:
    desc = crm_desc.upper()
    for key, w in _SEVERITY.items():
        if key in desc:
            return w
    return 0.18


@lru_cache(maxsize=1)
def load_crime_heatmap() -> dict[str, Any]:
    """GeoJSON FeatureCollection of crime incidents, sampled and weighted by severity."""
    features: list[dict[str, Any]] = []
    if not CRIME_FILE.exists():
        return {"type": "FeatureCollection", "features": []}

    # File has ~1M rows; sample 1 in 200 → ~5000 representative points
    sample_every = 200
    max_pts = 5000
    row_num = 0

    with open(CRIME_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_num += 1
            if row_num % sample_every != 0:
                continue
            if len(features) >= max_pts:
                break
            try:
                lat = float(row.get("LAT") or 0)
                lng = float(row.get("LON") or 0)
            except ValueError:
                continue
            if lat == 0.0 and lng == 0.0:
                continue
            if not (33.5 <= lat <= 34.5 and -118.85 <= lng <= -117.7):
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "properties": {
                    "weight": _weight(row.get("Crm Cd Desc", "")),
                    "area": row.get("AREA NAME", ""),
                    "crime": row.get("Crm Cd Desc", ""),
                },
            })

    return {"type": "FeatureCollection", "features": features}


@lru_cache(maxsize=1)
def load_crime_area_summary() -> list[dict[str, Any]]:
    """Return crime counts per AREA NAME — used by the RAG advisor for context."""
    if not CRIME_FILE.exists():
        return []

    area_counts: dict[str, int] = {}
    area_types: dict[str, dict[str, int]] = {}

    with open(CRIME_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            area = row.get("AREA NAME", "Unknown").strip()
            crm = row.get("Crm Cd Desc", "").strip()
            area_counts[area] = area_counts.get(area, 0) + 1
            area_types.setdefault(area, {})
            area_types[area][crm] = area_types[area].get(crm, 0) + 1

    result = []
    for area, total in sorted(area_counts.items(), key=lambda x: x[1], reverse=True):
        top = sorted(area_types[area].items(), key=lambda x: x[1], reverse=True)[:3]
        result.append({
            "area": area,
            "total_incidents": total,
            "top_crimes": [{"type": c, "count": n} for c, n in top],
        })
    return result
