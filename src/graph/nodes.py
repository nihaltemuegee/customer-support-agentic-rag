"""
Node functions for the customer support LangGraph workflow.

Each function takes the current SupportState and returns a dict of
fields to merge into the state (the standard LangGraph node pattern).
"""

import re

from src.graph.state import SupportState
from src.rag.retriever import retrieve
from src.tools.order_tools import lookup_order_status
from src.tools.refund_tools import check_refund_eligibility
from src.tools.ticket_tools import create_support_ticket

ORDER_ID_PATTERN = re.compile(r"ORD-\d+", re.IGNORECASE)

# Keywords used for simple rule-based intent classification.
# Checked roughly in priority order: escalation first, unknown last.
ESCALATION_KEYWORDS = [
    "frustrated", "furious", "angry", "unacceptable", "terrible",
    "worst", "manager", "supervisor", "escalate", "complaint", "ridiculous",
]
REFUND_KEYWORDS = ["refund", "money back", "reimburse"]
ORDER_STATUS_KEYWORDS = ["where is my order", "track", "tracking", "order status", "status of"]
RETURN_KEYWORDS = ["return policy", "can i return", "how do i return", "returning"]
SHIPPING_KEYWORDS = [
    "ship", "shipping", "deliver", "delivery",
    "international", "internationally",
    "tracking", "carrier", "package", "parcel",
]
WARRANTY_KEYWORDS = ["warranty", "defect", "broken", "malfunction"]
GENERAL_FAQ_KEYWORDS = ["account", "password", "email", "login", "sign up", "log in"]

# Intents that are grounded with FAQ evidence via src/rag/retriever.py.
# order_status and complaint_escalation are answered from tool results instead.
FAQ_RETRIEVAL_INTENTS = {
    "shipping_question",
    "return_policy",
    "refund_request",
    "warranty_question",
    "general_faq",
}


def classify_intent_text(question: str, order_id: str | None) -> str:
    """
    Pure rule-based intent classifier so it's easy to unit test on its
    own, separate from the LangGraph node wrapper below.
    """
    text = question.lower()

    if any(keyword in text for keyword in ESCALATION_KEYWORDS):
        return "complaint_escalation"

    if any(keyword in text for keyword in REFUND_KEYWORDS):
        return "refund_request"

    if any(keyword in text for keyword in ORDER_STATUS_KEYWORDS) or (
        order_id and "order" in text
    ):
        return "order_status"

    if any(keyword in text for keyword in RETURN_KEYWORDS):
        return "return_policy"

    if any(keyword in text for keyword in SHIPPING_KEYWORDS):
        return "shipping_question"

    if any(keyword in text for keyword in WARRANTY_KEYWORDS):
        return "warranty_question"

    if any(keyword in text for keyword in GENERAL_FAQ_KEYWORDS):
        return "general_faq"

    return "unknown"


def receive_question(state: SupportState) -> dict:
    """Normalize the incoming question and extract an order id, if present."""
    question = state["question"].strip()

    match = ORDER_ID_PATTERN.search(question)
    order_id = match.group(0).upper() if match else None

    return {
        "question": question,
        "order_id": order_id,
        "evidence": [],
        "tool_result": None,
        "needs_escalation": False,
    }


def classify_intent(state: SupportState) -> dict:
    """Classify the customer's intent using simple keyword rules."""
    intent = classify_intent_text(state["question"], state.get("order_id"))
    return {"intent": intent}


def route_request(state: SupportState) -> dict:
    """
    Based on the classified intent, call the relevant tool and/or
    fetch supporting evidence from the FAQ retriever.

    - order_status / complaint_escalation: answered from a tool call.
    - shipping_question / return_policy / refund_request / warranty_question /
      general_faq: grounded with FAQ evidence (see FAQ_RETRIEVAL_INTENTS above).
    - unknown: no tool call, no evidence.
    """
    intent = state["intent"]
    question = state["question"]
    order_id = state.get("order_id")

    tool_result = None
    evidence: list[dict[str, str]] = []
    needs_escalation = False

    if intent == "order_status":
        if order_id:
            tool_result = lookup_order_status(order_id)
        else:
            tool_result = {"found": False, "message": "No order id was provided."}

    elif intent == "refund_request":
        if order_id:
            tool_result = check_refund_eligibility(order_id)
        else:
            tool_result = {"eligible": False, "reason": "No order id was provided."}

    elif intent == "complaint_escalation":
        tool_result = create_support_ticket(summary=question, priority="high")
        needs_escalation = True

    if intent in FAQ_RETRIEVAL_INTENTS:
        evidence = retrieve(question)

    return {
        "tool_result": tool_result,
        "evidence": evidence,
        "needs_escalation": needs_escalation,
    }


def generate_response(state: SupportState) -> dict:
    """Compose the final natural-language answer shown to the customer."""
    intent = state["intent"]
    tool_result = state.get("tool_result")
    evidence = state.get("evidence") or []
    order_id = state.get("order_id")

    if intent == "order_status":
        if tool_result and tool_result.get("found"):
            answer = (
                f"Your order {tool_result['order_id']} ({tool_result['product']}) "
                f"is currently '{tool_result['status']}'."
            )
            if tool_result.get("estimated_delivery"):
                answer += f" Estimated delivery: {tool_result['estimated_delivery']}."
        elif order_id:
            answer = f"I couldn't find an order with id '{order_id}'. Please double-check the order id."
        else:
            answer = "Please provide your order id (e.g. ORD-1001) so I can check its status."

    elif intent == "refund_request":
        if tool_result and order_id:
            answer = tool_result.get("reason", "I couldn't determine refund eligibility.")
        else:
            answer = "Please provide your order id so I can check refund eligibility."

    elif intent == "complaint_escalation":
        ticket_id = tool_result.get("ticket_id") if tool_result else "N/A"
        answer = (
            "I'm sorry for the trouble you've experienced. I've escalated this to our "
            f"support team (ticket {ticket_id}) and a human agent will follow up shortly."
        )

    elif intent in ("return_policy", "shipping_question", "warranty_question", "general_faq"):
        if evidence:
            # Ground the answer in the top-scoring FAQ chunk. Each chunk's
            # "text" is "<heading>: <body>", so we surface just the body.
            _, _, body = evidence[0]["text"].partition(": ")
            answer = body or evidence[0]["text"]
        else:
            answer = "I don't have a specific answer for that yet, but our support team can help."

    else:  # unknown
        answer = (
            "I'm not sure I understood that. Could you rephrase your question, or ask about "
            "an order, refund, return, shipping, or warranty?"
        )

    return {"final_answer": answer}
