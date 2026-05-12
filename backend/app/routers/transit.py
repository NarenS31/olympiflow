from fastapi import APIRouter
from ..services.data_loader import load_transit_routes, load_parking_summary

router = APIRouter(prefix="/transit", tags=["transit"])


@router.get("/routes")
def get_routes():
    """GeoJSON FeatureCollection of Downtown DASH transit routes."""
    return load_transit_routes()


@router.get("/parking")
def get_parking():
    """GeoJSON FeatureCollection of parking occupancy zones."""
    return load_parking_summary()
