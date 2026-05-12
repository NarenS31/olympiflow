from fastapi import APIRouter
from ..services.data_loader import load_collision_heatmap, load_traffic_stats
from ..services.crime_loader import load_crime_heatmap, load_crime_area_summary

router = APIRouter(prefix="/traffic", tags=["traffic"])


@router.get("/heatmap")
def get_heatmap():
    """GeoJSON FeatureCollection of traffic hotspots from collision data."""
    return load_collision_heatmap()


@router.get("/stats")
def get_stats():
    """Aggregated LADOT traffic count statistics."""
    return load_traffic_stats()


@router.get("/crime")
def get_crime_heatmap():
    """GeoJSON FeatureCollection of crime incident hotspots (2020-2024)."""
    return load_crime_heatmap()


@router.get("/crime/summary")
def get_crime_summary():
    """Crime counts per LAPD area, sorted by incident volume."""
    return load_crime_area_summary()
