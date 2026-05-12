from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from ..services.rag_advisor import get_traffic_advice

router = APIRouter(prefix="/ai", tags=["ai"])


class AskRequest(BaseModel):
    query: str
    model: str = "llama3.2"
    simulation_context: dict[str, Any] | None = None


@router.post("/ask")
async def ask_advisor(req: AskRequest):
    """RAG-powered Olympic traffic advisor using a local Ollama model."""
    return await get_traffic_advice(
        query=req.query,
        model=req.model,
        simulation_context=req.simulation_context,
    )
