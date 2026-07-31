"""
Refund-related tool functions.

Applies simple, explicit demo rules on top of the synthetic orders
data to decide refund eligibility and the recommended next step. This
is a placeholder for a real refund policy engine, not real financial
or legal logic.
"""

from datetime import datetime
from typing import Any

from src.tools.order_tools import _load_orders

REFUND_WINDOW_DAYS = 30


def check_refund_eligibility(order_id: str, damaged: bool = False) -> dict[str, Any]:
    """
    Check whether an order is eligible for a refund, and what the
    customer should do next.

    Rules (simplified, for demo purposes):
    - Order not found -> not eligible; ask the customer to double-check the order id.
    - damaged=True -> always eligible for a refund/replacement, regardless of status,
      and the next step recommends contacting support / a ticket for follow-up.
    - Cancelled orders -> eligible for a full refund.
    - Delivered orders -> eligible if within REFUND_WINDOW_DAYS of the order date.
    - Shipped orders -> not yet eligible; must wait until the order is delivered.
    - Processing orders -> not eligible for a refund yet, but may still be cancelable.
    - Delayed orders -> not yet eligible; recommend waiting or contacting support.
    - Any other status -> recommend contacting support.

    Every result also includes "needs_human_review": True when the rules above
    couldn't fully resolve the case on their own (a damaged item, or a status the
    rules don't recognize), which src/graph/nodes.py uses to decide escalation.
    """
    orders = _load_orders()
    normalized_id = order_id.upper() if order_id else order_id
    order = orders.get(normalized_id)

    if order is None:
        return {
            "found": False,
            "order_id": order_id,
            "eligible": False,
            "reason": f"No order found with id '{order_id}'.",
            "next_step": "Please double-check your order id and try again.",
            "needs_human_review": False,
        }

    if damaged:
        return {
            "found": True,
            "order_id": order_id,
            "eligible": True,
            "reason": "Damaged or defective items are eligible for a refund or replacement.",
            "next_step": "Please contact support with a description (and photos, if possible); "
                          "we'll open a support ticket to process your refund or replacement.",
            "needs_human_review": True,
        }

    status = order["status"]

    if status == "cancelled":
        return {
            "found": True,
            "order_id": order_id,
            "eligible": True,
            "reason": "Order was cancelled and is eligible for a full refund.",
            "next_step": "A refund will be issued to your original payment method.",
            "needs_human_review": False,
        }

    if status == "delivered":
        order_date = datetime.strptime(order["order_date"], "%Y-%m-%d")
        days_since_order = (datetime.now() - order_date).days

        if days_since_order <= REFUND_WINDOW_DAYS:
            return {
                "found": True,
                "order_id": order_id,
                "eligible": True,
                "reason": f"Order was delivered {days_since_order} day(s) ago, "
                          f"within the {REFUND_WINDOW_DAYS}-day refund window.",
                "next_step": "Start a return to receive your refund.",
                "needs_human_review": False,
            }

        return {
            "found": True,
            "order_id": order_id,
            "eligible": False,
            "reason": f"Order was delivered {days_since_order} day(s) ago, "
                      f"which is past the {REFUND_WINDOW_DAYS}-day refund window.",
            "next_step": "Contact support if there are special circumstances (e.g. a damaged item).",
            "needs_human_review": False,
        }

    if status == "shipped":
        return {
            "found": True,
            "order_id": order_id,
            "eligible": False,
            "reason": "Order has shipped but hasn't been delivered yet, so it isn't refund-eligible yet.",
            "next_step": "Please wait until the order is delivered, then request a refund if needed.",
            "needs_human_review": False,
        }

    if status == "processing":
        return {
            "found": True,
            "order_id": order_id,
            "eligible": False,
            "reason": "Order is still processing and hasn't shipped, so there's nothing to refund yet.",
            "next_step": "The order may still be cancelable — contact support to cancel it before it ships.",
            "needs_human_review": False,
        }

    if status == "delayed":
        return {
            "found": True,
            "order_id": order_id,
            "eligible": False,
            "reason": "Order is delayed and hasn't been delivered yet, so it isn't refund-eligible yet.",
            "next_step": "Please wait for delivery, or contact support if it's significantly overdue.",
            "needs_human_review": False,
        }

    return {
        "found": True,
        "order_id": order_id,
        "eligible": False,
        "reason": f"Order status is '{status}'; refund eligibility could not be determined automatically.",
        "next_step": "Please contact support for help with this order.",
        "needs_human_review": True,
    }
