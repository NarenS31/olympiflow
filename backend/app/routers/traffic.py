from fastapi import APIRouter
from ..services.data_loader import load_collision_heatmap, load_traffic_stats

router = APIRouter(prefix="/traffic", tags=["traffic"])


@router.get("/heatmap")
def get_heatmap():
    """GeoJSON FeatureCollection of traffic hotspots from collision data."""
    return load_collision_heatmap()


@router.get("/stats")
def get_stats():
    """Aggregated LADOT traffic count statistics."""
    return load_traffic_stats()
