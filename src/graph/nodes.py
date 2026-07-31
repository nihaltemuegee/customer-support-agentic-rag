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

# --- Keywords used for simple rule-based intent classification -------------
# Checked in priority order: a specific topic (refund/damage, order, return,
# shipping, warranty, general FAQ) wins if present, so e.g. an angry refund
# question still gets routed to the refund tool. A generic escalation/complaint
# signal with no specific topic is the last resort before "unknown".

# Damage/defect/missing/lost-on-arrival wording. Checked alongside
# REFUND_KEYWORDS, since these are a refund/replacement case first.
DAMAGE_KEYWORDS = [
    "damaged", "broken", "smashed", "shattered", "arrived damaged",
    "missing package", "package is missing", "package went missing",
    "lost package", "package is lost",
    "never arrived", "never received", "hasn't arrived", "didn't arrive", "did not arrive",
]
REFUND_KEYWORDS = ["refund", "money back", "reimburse"]
ORDER_STATUS_KEYWORDS = ["where is my order", "track", "tracking", "order status", "status of"]
RETURN_KEYWORDS = ["return policy", "can i return", "how do i return", "returning"]
SHIPPING_KEYWORDS = [
    "ship", "shipping", "deliver", "delivery",
    "international", "internationally",
    "tracking", "carrier", "package", "parcel",
]
WARRANTY_KEYWORDS = ["warranty", "defect", "malfunction"]
GENERAL_FAQ_KEYWORDS = ["account", "password", "email", "login", "sign up", "log in"]

# --- Keywords used for escalation detection (independent of intent) --------
# Split by *why* the customer needs a human, since that also determines the
# ticket priority (see escalation_priority below).
EMOTION_KEYWORDS = [
    "frustrated", "furious", "angry", "upset", "disappointed",
    "unacceptable", "terrible", "worst", "ridiculous", "complaint",
    "manager", "supervisor", "escalate",
]
URGENCY_KEYWORDS = ["urgent", "immediately", "asap"]
HUMAN_REQUEST_KEYWORDS = [
    "human", "representative", "real person", "speak to someone", "talk to someone",
]
# Union of all escalation-flavored keywords, used as intent classification's
# last resort (a generic complaint with no specific topic attached).
ESCALATION_KEYWORDS = EMOTION_KEYWORDS + URGENCY_KEYWORDS + HUMAN_REQUEST_KEYWORDS

# Intents that are grounded with FAQ evidence via src/rag/retriever.py.
# order_status is answered from a tool result instead.
FAQ_RETRIEVAL_INTENTS = {
    "shipping_question",
    "return_policy",
    "refund_request",
    "warranty_question",
    "general_faq",
}


def extract_order_id(question: str) -> str | None:
    """
    Extract an order id like ORD-1001 from free text.

    Case-insensitive and works anywhere in the sentence, so it handles
    "ORD-1001", "ord-1001", "Order ORD-1001", and "my order is ORD-1001".
    """
    match = ORDER_ID_PATTERN.search(question)
    return match.group(0).upper() if match else None


def is_damage_question(question: str) -> bool:
    """Whether the question mentions a damaged, broken, missing, or lost package."""
    text = question.lower()
    return any(keyword in text for keyword in DAMAGE_KEYWORDS)


def escalation_priority(question: str, tool_result: dict | None = None) -> str | None:
    """
    Decide whether this request should be escalated to a human and, if so,
    at what priority. Returns None when no escalation is needed.

    Escalation is judged independently of the classified intent, so e.g. an
    angry refund_request still runs the refund tool AND gets flagged here.

    - high: the customer sounds angry/upset, the wording is urgent, or the
      question reports a damaged, broken, missing, or lost package.
    - medium: a refund case the rule-based tool couldn't resolve on its own
      (see "needs_human_review" in check_refund_eligibility's result).
    - low: a plain request to speak with a human, with no other signal.
    """
    text = question.lower()

    if any(keyword in text for keyword in EMOTION_KEYWORDS):
        return "high"

    if any(keyword in text for keyword in URGENCY_KEYWORDS):
        return "high"

    if is_damage_question(question):
        return "high"

    if tool_result and tool_result.get("needs_human_review"):
        return "medium"

    if any(keyword in text for keyword in HUMAN_REQUEST_KEYWORDS):
        return "low"

    return None


def classify_intent_text(question: str, order_id: str | None) -> str:
    """
    Pure rule-based intent classifier so it's easy to unit test on its
    own, separate from the LangGraph node wrapper below.

    A specific topic always wins over a generic escalation/complaint signal,
    so e.g. an angry refund question is still classified refund_request (and
    still runs the refund tool) -- see escalation_priority() for how the
    *separate* needs_escalation flag is decided.
    """
    text = question.lower()

    if any(keyword in text for keyword in REFUND_KEYWORDS) or is_damage_question(question):
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

    if any(keyword in text for keyword in ESCALATION_KEYWORDS):
        return "complaint_escalation"

    return "unknown"


def receive_question(state: SupportState) -> dict:
    """Normalize the incoming question and extract an order id, if present."""
    question = state["question"].strip()
    order_id = extract_order_id(question)

    return {
        "question": question,
        "order_id": order_id,
        "evidence": [],
        "tool_result": None,
        "needs_escalation": False,
        "escalation_result": None,
    }


def classify_intent(state: SupportState) -> dict:
    """Classify the customer's intent using simple keyword rules."""
    intent = classify_intent_text(state["question"], state.get("order_id"))
    return {"intent": intent}


def route_request(state: SupportState) -> dict:
    """
    Based on the classified intent, call the relevant tool and/or fetch
    supporting evidence from the FAQ retriever. Then, independently of
    intent, decide whether the request needs to be escalated to a human
    and create a support ticket if so.

    - order_status: answered from lookup_order_status(order_id).
    - refund_request: answered from check_refund_eligibility(order_id),
      grounded with refund/return FAQ evidence.
    - return_policy / shipping_question / warranty_question / general_faq:
      grounded with FAQ evidence (see FAQ_RETRIEVAL_INTENTS above).
    - complaint_escalation / unknown: no topic tool call.
    - Missing order id (order_status / refund_request): no tool is called and
      no order id is guessed -- the customer is asked to provide one instead.

    Escalation (see escalation_priority()) can fire for ANY intent -- e.g. an
    angry refund_request still runs the refund tool AND gets a ticket.
    """
    intent = state["intent"]
    question = state["question"]
    order_id = state.get("order_id")

    tool_result = None
    evidence: list[dict[str, str]] = []

    if intent == "order_status":
        if order_id:
            tool_result = lookup_order_status(order_id)
        else:
            tool_result = {
                "found": False,
                "order_id": None,
                "message": "No order id was provided.",
            }

    elif intent == "refund_request":
        if order_id:
            tool_result = check_refund_eligibility(order_id, damaged=is_damage_question(question))
        else:
            tool_result = {
                "found": False,
                "order_id": None,
                "eligible": False,
                "reason": "No order id was provided.",
                "next_step": "Please share your order id so we can check refund eligibility.",
                "needs_human_review": False,
            }

    if intent in FAQ_RETRIEVAL_INTENTS:
        evidence = retrieve(question)

    priority = escalation_priority(question, tool_result)
    needs_escalation = priority is not None
    escalation_result = create_support_ticket(summary=question, priority=priority) if needs_escalation else None

    return {
        "tool_result": tool_result,
        "evidence": evidence,
        "needs_escalation": needs_escalation,
        "escalation_result": escalation_result,
    }


def generate_response(state: SupportState) -> dict:
    """
    Compose the final natural-language answer shown to the customer.

    First builds a topic-specific answer from intent/tool_result/evidence, then
    -- separately -- appends an escalation notice if escalation_result was set,
    regardless of intent. This keeps "what's the answer" and "does this also
    need a human" as two independent concerns, matching how they're decided
    in route_request().
    """
    intent = state["intent"]
    tool_result = state.get("tool_result")
    evidence = state.get("evidence") or []
    order_id = state.get("order_id")
    escalation_result = state.get("escalation_result")

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
            reason = tool_result.get("reason", "I couldn't determine refund eligibility.")
            next_step = tool_result.get("next_step")
            answer = f"{reason} {next_step}" if next_step else reason
        else:
            answer = "Please provide your order id (e.g. ORD-1001) so I can check refund eligibility."

    elif intent in ("return_policy", "shipping_question", "warranty_question", "general_faq"):
        if evidence:
            # Ground the answer in the top-scoring FAQ chunk. Each chunk's
            # "text" is "<heading>: <body>", so we surface just the body.
            _, _, body = evidence[0]["text"].partition(": ")
            answer = body or evidence[0]["text"]
        else:
            answer = "I don't have a specific answer for that yet, but our support team can help."

    elif intent == "complaint_escalation":
        answer = "I'm sorry for the trouble you've experienced."

    else:  # unknown
        answer = (
            "I'm not sure I understood that. Could you rephrase your question, or ask about "
            "an order, refund, return, shipping, or warranty?"
        )

    if escalation_result:
        ticket_id = escalation_result.get("ticket_id", "N/A")
        answer += (
            f" I've escalated this to our support team (ticket {ticket_id}) "
            "and a human agent will follow up shortly."
        )

    return {"final_answer": answer}
