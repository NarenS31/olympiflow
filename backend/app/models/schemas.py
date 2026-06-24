from pydantic import BaseModel
from typing import Any


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: dict[str, Any]
    properties: dict[str, Any]


class GeoJSONCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]


class SimulationStepRequest(BaseModel):
    mode: str
    timeOfDay: float
    globalIntensity: float
    venueSurges: dict[str, float]


class SimulationStepResponse(BaseModel):
    congestionScore: float
    avgDelayIncrease: float
    peakZones: int
    affectedRoutes: int
    personsAffected: int


class TrafficStatsResponse(BaseModel):
    topIntersections: list[dict[str, Any]]
    totalCount: int
    avgDailyVolume: float


class CongestionState(BaseModel):
    label: str
    vc_ratio: float
    bpr_multiplier: float
    congestion_index: float
    estimated_travel_time_min: float


class CongestionStatsResponse(BaseModel):
    venue: str
    capacity: int
    event_hour: float
    baseline: CongestionState
    intervention: CongestionState
    reduction_pct: float
