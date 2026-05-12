from fastapi import APIRouter
from ..models.schemas import SimulationStepRequest, SimulationStepResponse
from ..services.simulation_engine import run_simulation_step

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/step", response_model=SimulationStepResponse)
def simulation_step(req: SimulationStepRequest) -> SimulationStepResponse:
    """Run one simulation step and return updated metrics."""
    result = run_simulation_step(
        mode=req.mode,
        time_of_day=req.timeOfDay,
        global_intensity=req.globalIntensity,
        venue_surges=req.venueSurges,
    )
    return SimulationStepResponse(**result)
