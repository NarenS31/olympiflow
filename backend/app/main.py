from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import venues, traffic, transit, simulation

app = FastAPI(
    title="OlympiFlow API",
    description="LA 2028 Olympic logistics and traffic simulation backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(venues.router)
app.include_router(traffic.router)
app.include_router(transit.router)
app.include_router(simulation.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "OlympiFlow API"}
