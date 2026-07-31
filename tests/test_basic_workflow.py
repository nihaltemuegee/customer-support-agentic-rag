"""
Basic tests for the rule-based intent classifier and the FastAPI app.
"""

from fastapi.testclient import TestClient

from app.main import app
from src.graph.graph import run_support_workflow
from src.graph.nodes import classify_intent_text
from src.rag.retriever import retrieve
from src.tools.order_tools import lookup_order_status
from src.tools.refund_tools import check_refund_eligibility

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


# --- Version 3: order/refund tool routing tests ----------------------------


def test_order_id_extraction_various_formats():
    for question in [
        "ORD-1001",
        "ord-1001",
        "Order ORD-1001",
        "my order is ORD-1001",
    ]:
        result = run_support_workflow(f"{question} status please")
        assert result["order_id"] == "ORD-1001"


def test_valid_order_status_lookup():
    result = lookup_order_status("ORD-1001")
    assert result["found"] is True
    assert result["order_id"] == "ORD-1001"
    assert result["customer_name"] == "Alice Johnson"
    assert result["status"] == "delivered"
    assert result["total_amount"] == 79.99


def test_unknown_order_id_lookup():
    result = lookup_order_status("ORD-9999")
    assert result["found"] is False
    assert result["status"] is None


def test_refund_eligibility_for_delivered_order():
    result = check_refund_eligibility("ORD-1001")
    assert result["found"] is True
    assert result["eligible"] is True
    assert "next_step" in result


def test_refund_request_without_order_id():
    result = run_support_workflow("I want a refund please.")
    assert result["intent"] == "refund_request"
    assert result["tool_result"]["found"] is False
    assert "order id" in result["final_answer"].lower()


def test_damaged_package_question():
    result = run_support_workflow("My order ORD-1005 arrived damaged. What should I do?")
    assert result["intent"] == "refund_request"
    assert result["needs_escalation"] is False
    assert result["tool_result"]["eligible"] is True
    assert "ticket" in result["tool_result"]
    assert len(result["evidence"]) > 0


def test_order_status_api_response_includes_tool_result():
    response = client.post("/ask", json={"question": "Where is my order ORD-1001?"})
    assert response.status_code == 200

    data = response.json()
    assert data["tool_result"] is not None
    assert data["tool_result"]["found"] is True
    assert data["tool_result"]["order_id"] == "ORD-1001"
