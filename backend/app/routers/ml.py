from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ml.surge_predictor import predict_surge, classify_conditions

router = APIRouter(prefix="/ml", tags=["ml"])


class SurgeRequest(BaseModel):
    venue_id: str = Field(..., example="sofi")
    hour: float = Field(..., ge=0, le=23.99)
    day_of_week: int = Field(..., ge=0, le=6)
    event_type: int = Field(..., ge=0, le=2, description="0=none,1=regular,2=opening/closing")
    capacity_util: float = Field(..., ge=0.0, le=1.0)


class ConditionRequest(BaseModel):
    surge_intensity: float = Field(..., ge=0.0, le=1.0)
    vc_ratio: float = Field(..., ge=0.0)
    time_to_event_min: float = Field(..., description="Negative means event already started")
    concurrent_events: int = Field(..., ge=0)
    transit_availability_score: float = Field(..., ge=0.0, le=1.0)


@router.post("/predict-surge")
def predict_surge_endpoint(req: SurgeRequest):
    return predict_surge(
        venue_id=req.venue_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
        event_type=req.event_type,
        capacity_util=req.capacity_util,
    )


@router.post("/classify-conditions")
def classify_conditions_endpoint(req: ConditionRequest):
    return classify_conditions(
        surge_intensity=req.surge_intensity,
        vc_ratio=req.vc_ratio,
        time_to_event_min=req.time_to_event_min,
        concurrent_events=req.concurrent_events,
        transit_availability_score=req.transit_availability_score,
    )
