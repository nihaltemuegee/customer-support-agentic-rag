"""
Support ticket tool functions.

Creates lightweight in-memory support tickets. There is no real
ticketing system or database yet -- this is a placeholder that
returns a structured dict, the same shape a real integration would.
"""

import uuid
from typing import Any

from src.config import DEFAULT_TICKET_PRIORITY

VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


def create_support_ticket(summary: str, priority: str = DEFAULT_TICKET_PRIORITY) -> dict[str, Any]:
    """
    Create a new support ticket for a human agent to follow up on.

    Returns a structured dict with a generated ticket id, since there
    is no real ticketing backend in Version 1.
    """
    normalized_priority = priority.lower() if priority else DEFAULT_TICKET_PRIORITY
    if normalized_priority not in VALID_PRIORITIES:
        normalized_priority = DEFAULT_TICKET_PRIORITY

    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"

    return {
        "ticket_id": ticket_id,
        "summary": summary,
        "priority": normalized_priority,
        "status": "open",
    }
