"""
Typed state object shared across all LangGraph nodes.

Every node reads from and writes back into this state. Keeping it
typed makes it clear what data flows through the graph.
"""

from typing import Any, Optional, TypedDict


class SupportState(TypedDict, total=False):
    # The raw question asked by the customer
    question: str

    # Intent from the previous turn, passed in by the caller so a bare
    # follow-up like "ORD-1001" (with no topic keywords of its own) can be
    # understood in context. Only meaningful when that previous turn asked
    # the customer for an order id. See Version 6: multi-turn support.
    previous_intent: Optional[str]

    # Classified intent, e.g. "order_status", "refund_request", ...
    intent: str

    # Order id extracted from the question, if any (e.g. "ORD-1001")
    order_id: Optional[str]

    # Evidence gathered from the FAQ retriever: [{"source": "shipping.md", "text": "..."}, ...]
    evidence: list[dict[str, str]]

    # Structured result returned by a topic tool call (order lookup, refund check)
    tool_result: Optional[dict[str, Any]]

    # Whether this request should be escalated to a human agent
    needs_escalation: bool

    # Support ticket created by create_support_ticket(), set only when needs_escalation is True
    escalation_result: Optional[dict[str, Any]]

    # The final answer text shown to the customer
    final_answer: str
