"""
Basic tests for the rule-based intent classifier and the FastAPI app.
"""

from fastapi.testclient import TestClient

from app.main import app
from src.graph.nodes import classify_intent_text

client = TestClient(app)


def test_classify_order_status():
    intent = classify_intent_text("Where is my order ORD-1001?", order_id="ORD-1001")
    assert intent == "order_status"


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
