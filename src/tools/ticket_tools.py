"""
Support ticket tool functions.

Creates lightweight in-memory support tickets. There is no real
ticketing system or database yet -- this is a placeholder that
returns a structured dict, the same shape a real integration would.
"""

from typing import Any

from src.config import DEFAULT_TICKET_PRIORITY

VALID_PRIORITIES = {"low", "medium", "high"}

# Demo SLA copy per priority, just so ticket results feel like a real response.
NEXT_STEP_BY_PRIORITY = {
    "high": "A support agent will follow up within 1 hour.",
    "medium": "A support agent will follow up within 1 business day.",
    "low": "A support agent will follow up within 2 business days.",
}

# In-memory counter so ticket ids look sequential (e.g. TICKET-1001) within a
# single run of the app. Resets when the process restarts -- there's no real
# ticketing database in this version.
_ticket_counter = 1000


def create_support_ticket(summary: str, priority: str = DEFAULT_TICKET_PRIORITY) -> dict[str, Any]:
    """
    Create a new support ticket for a human agent to follow up on.

    Returns a structured dict with a generated ticket id, since there
    is no real ticketing backend in this version.
    """
    global _ticket_counter

    normalized_priority = priority.lower() if priority else DEFAULT_TICKET_PRIORITY
    if normalized_priority not in VALID_PRIORITIES:
        normalized_priority = DEFAULT_TICKET_PRIORITY

    _ticket_counter += 1
    ticket_id = f"TICKET-{_ticket_counter}"

    return {
        "ticket_id": ticket_id,
        "summary": summary,
        "priority": normalized_priority,
        "status": "open",
        "created": True,
        "next_step": NEXT_STEP_BY_PRIORITY[normalized_priority],
    }
