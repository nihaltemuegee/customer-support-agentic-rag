"""
Typed state object shared across all LangGraph nodes.

Every node reads from and writes back into this state. Keeping it
typed makes it clear what data flows through the graph.
"""

from typing import Any, Optional, TypedDict


class SupportState(TypedDict, total=False):
    # The raw question asked by the customer
    question: str

    # Classified intent, e.g. "order_status", "refund_request", ...
    intent: str

    # Order id extracted from the question, if any (e.g. "ORD-1001")
    order_id: Optional[str]

    # Evidence gathered from the FAQ retriever (list of text snippets)
    evidence: list[str]

    # Structured result returned by a tool call (order lookup, refund check, ticket creation)
    tool_result: Optional[dict[str, Any]]

    # Whether this request should be escalated to a human agent
    needs_escalation: bool

    # The final answer text shown to the customer
    final_answer: str
