"""
Basic tests for the rule-based intent classifier and the FastAPI app.
"""

from fastapi.testclient import TestClient

from app.main import app
from src.graph.graph import run_support_workflow
from src.graph.nodes import classify_intent_text
from src.rag.retriever import retrieve

client = TestClient(app)


def test_classify_order_status():
    intent = classify_intent_text("Where is my order ORD-1001?", order_id="ORD-1001")
    assert intent == "order_status"


def test_classify_shipping_international():
    intent = classify_intent_text("Do you ship internationally?", order_id=None)
    assert intent == "shipping_question"


def test_classify_shipping_delivery_time():
    intent = classify_intent_text("How long does delivery take?", order_id=None)
    assert intent == "shipping_question"


def test_classify_refund_request():
    intent = classify_intent_text("I would like a refund for my order.", order_id=None)
    assert intent == "refund_request"


def test_classify_complaint_escalation():
    intent = classify_intent_text(
        "I am furious and want to speak to a manager right now!", order_id=None
    )
    assert intent == "complaint_escalation"


def test_classify_unknown():
    intent = classify_intent_text("What's the best pizza topping?", order_id=None)
    assert intent == "unknown"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Version 2: FAQ RAG tests ---------------------------------------------


def test_shipping_question_retrieves_shipping_md():
    evidence = retrieve("How long does shipping take?")
    assert len(evidence) > 0
    assert any(item["source"] == "shipping.md" for item in evidence)


def test_return_question_retrieves_returns_md():
    evidence = retrieve("What is your return policy?")
    assert len(evidence) > 0
    assert any(item["source"] == "returns.md" for item in evidence)


def test_warranty_question_retrieves_warranty_md():
    evidence = retrieve("Is my product covered under warranty?")
    assert len(evidence) > 0
    assert any(item["source"] == "warranty.md" for item in evidence)


def test_unknown_question_still_works_safely():
    result = run_support_workflow("What's the best pizza topping?")
    assert result["intent"] == "unknown"
    assert result["evidence"] == []
    assert result["final_answer"]


def test_ask_endpoint_evidence_has_source_and_text():
    response = client.post("/ask", json={"question": "How long does shipping take?"})
    assert response.status_code == 200

    data = response.json()
    assert data["intent"] == "shipping_question"
    assert len(data["evidence"]) > 0
    assert "source" in data["evidence"][0]
    assert "text" in data["evidence"][0]
