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
from src.tools.ticket_tools import create_support_ticket

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
    assert result["tool_result"]["eligible"] is True
    assert len(result["evidence"]) > 0
    # Version 4: damage reports escalate to a human at high priority.
    assert result["needs_escalation"] is True
    assert result["escalation_result"] is not None
    assert result["escalation_result"]["priority"] == "high"


def test_order_status_api_response_includes_tool_result():
    response = client.post("/ask", json={"question": "Where is my order ORD-1001?"})
    assert response.status_code == 200

    data = response.json()
    assert data["tool_result"] is not None
    assert data["tool_result"]["found"] is True
    assert data["tool_result"]["order_id"] == "ORD-1001"


# --- Version 4: escalation & ticket creation tests --------------------------


def test_create_support_ticket_returns_expected_shape():
    ticket = create_support_ticket("Test issue", priority="high")
    assert ticket["created"] is True
    assert ticket["status"] == "open"
    assert ticket["priority"] == "high"
    assert ticket["ticket_id"].startswith("TICKET-")
    assert "next_step" in ticket


def test_angry_customer_creates_high_priority_ticket():
    result = run_support_workflow("I am so angry, this service has been terrible!")
    assert result["needs_escalation"] is True
    assert result["escalation_result"] is not None
    assert result["escalation_result"]["priority"] == "high"


def test_general_human_request_creates_low_priority_ticket():
    result = run_support_workflow("Can I talk to a human, please?")
    assert result["needs_escalation"] is True
    assert result["escalation_result"] is not None
    assert result["escalation_result"]["priority"] == "low"


def test_normal_faq_question_does_not_create_ticket():
    result = run_support_workflow("What is your return policy?")
    assert result["needs_escalation"] is False
    assert result["escalation_result"] is None


def test_final_answer_includes_ticket_id_when_escalation_happens():
    result = run_support_workflow("I am furious and want to speak to a manager right now!")
    assert result["escalation_result"] is not None
    ticket_id = result["escalation_result"]["ticket_id"]
    assert ticket_id in result["final_answer"]


# --- Version 6: evaluation & multi-turn tests -------------------------------


def test_multiturn_refund_without_order_id_asks_for_it():
    result = run_support_workflow("I want a refund please.")
    assert result["intent"] == "refund_request"
    assert result["order_id"] is None
    assert "order id" in result["final_answer"].lower()


def test_multiturn_followup_message_with_order_id_reuses_refund_intent():
    # Turn 1: no order id, so the assistant asks for one.
    turn1 = run_support_workflow("I want a refund please.")
    assert turn1["order_id"] is None

    # Turn 2: a bare order id, with the previous turn's intent passed back in.
    turn2 = run_support_workflow("ORD-1003", previous_intent=turn1["intent"])
    assert turn2["intent"] == "refund_request"
    assert turn2["order_id"] == "ORD-1003"
    assert turn2["tool_result"]["found"] is True


def test_multiturn_order_status_followup_with_order_id():
    turn1 = run_support_workflow("Where is my order?")
    assert turn1["intent"] == "order_status"
    assert turn1["order_id"] is None

    turn2 = run_support_workflow("ord-1001", previous_intent=turn1["intent"])
    assert turn2["intent"] == "order_status"
    assert turn2["tool_result"]["found"] is True
    assert turn2["tool_result"]["status"] == "delivered"


def test_bare_order_id_without_previous_intent_stays_unknown():
    # No conversation context -- a bare order id alone has no topic keywords.
    result = run_support_workflow("ORD-1001")
    assert result["intent"] == "unknown"


def test_evaluation_script_can_run(capsys):
    from evaluation.run_evaluation import load_test_cases, run_evaluation

    test_cases = load_test_cases()
    assert len(test_cases) >= 12

    run_evaluation()
    captured = capsys.readouterr()
    assert "Summary:" in captured.out
