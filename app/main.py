"""
FastAPI app exposing the customer support agentic workflow.

Endpoints:
- GET  /health : simple liveness check
- POST /ask    : run a customer question through the LangGraph workflow
"""

from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import APP_NAME
from src.graph.graph import run_support_workflow

app = FastAPI(title=APP_NAME)


class AskRequest(BaseModel):
    question: str
    # Optional: the previous turn's intent, so a bare follow-up like "ORD-1001"
    # can be understood as continuing that request. See Version 6: multi-turn
    # support. The server is stateless -- the caller (UI or API client) is
    # responsible for remembering and re-sending this from the prior response's
    # "intent" field when that prior turn asked for an order id.
    previous_intent: Optional[str] = None


class EvidenceItem(BaseModel):
    source: str
    text: str


class AskResponse(BaseModel):
    question: str
    intent: str
    order_id: Optional[str] = None
    needs_escalation: bool
    tool_result: Optional[dict[str, Any]] = None
    escalation_result: Optional[dict[str, Any]] = None
    evidence: list[EvidenceItem]
    final_answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = run_support_workflow(request.question, previous_intent=request.previous_intent)

    return AskResponse(
        question=result["question"],
        intent=result["intent"],
        order_id=result.get("order_id"),
        needs_escalation=result.get("needs_escalation", False),
        tool_result=result.get("tool_result"),
        escalation_result=result.get("escalation_result"),
        evidence=result.get("evidence", []),
        final_answer=result["final_answer"],
    )
