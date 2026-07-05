import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import venues, traffic, transit, simulation, ai_advisor, ml as ml_router
from .models.schemas import CongestionStatsResponse
from .services.simulation_engine import compute_coliseum_congestion

app = FastAPI(
    title="OlympiFlow API",
    description="LA 2028 Olympic logistics and traffic simulation backend",
    version="0.1.0",
)

# Allow all origins in production (Vercel) — restrict to localhost in dev
_origins = (
    ["*"]
    if os.getenv("VERCEL")
    else ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(venues.router)
app.include_router(traffic.router)
app.include_router(transit.router)
app.include_router(simulation.router)
app.include_router(ai_advisor.router)
app.include_router(ml_router.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "OlympiFlow API"}


@app.get("/api/congestion-stats", response_model=CongestionStatsResponse)
def congestion_stats() -> CongestionStatsResponse:
    """
    BPR congestion model for LA Memorial Coliseum.
    Returns baseline (peak 1-hr arrival) vs. intervention (staggered 3-hr arrivals)
    side-by-side with v/c ratios, BPR travel-time multipliers, congestion index,
    and the % travel-time improvement from staggered arrivals.
    """
    result = compute_coliseum_congestion()
    print("\n--- /api/congestion-stats ---")
    print(json.dumps(result, indent=2))
    print("-----------------------------\n")
    return CongestionStatsResponse(**result)
