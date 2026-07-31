"""
Refund-related tool functions.

Uses simple rule-based logic on top of the synthetic orders data to
decide refund eligibility. This is a placeholder for a real refund
policy engine.
"""

from datetime import datetime
from typing import Any

from src.tools.order_tools import _load_orders

REFUND_WINDOW_DAYS = 30


def check_refund_eligibility(order_id: str) -> dict[str, Any]:
    """
    Check whether an order is eligible for a refund.

    Rules (simplified, for demo purposes):
    - Order not found -> not eligible.
    - Cancelled orders -> eligible (full refund of the charge).
    - Delivered orders -> eligible if within REFUND_WINDOW_DAYS of the order date.
    - Orders that are processing/shipped/delayed -> not yet eligible,
      since the item hasn't been delivered yet.
    """
    orders = _load_orders()
    order = orders.get(order_id)

    if order is None:
        return {
            "eligible": False,
            "order_id": order_id,
            "reason": f"No order found with id '{order_id}'.",
        }

    status = order["status"]

    if status == "cancelled":
        return {
            "eligible": True,
            "order_id": order_id,
            "reason": "Order was cancelled and is eligible for a full refund.",
        }

    if status == "delivered":
        order_date = datetime.strptime(order["order_date"], "%Y-%m-%d")
        days_since_order = (datetime.now() - order_date).days

        if days_since_order <= REFUND_WINDOW_DAYS:
            return {
                "eligible": True,
                "order_id": order_id,
                "reason": f"Order was delivered {days_since_order} day(s) ago, "
                          f"within the {REFUND_WINDOW_DAYS}-day refund window.",
            }

        return {
            "eligible": False,
            "order_id": order_id,
            "reason": f"Order was delivered {days_since_order} day(s) ago, "
                      f"which is past the {REFUND_WINDOW_DAYS}-day refund window.",
        }

    return {
        "eligible": False,
        "order_id": order_id,
        "reason": f"Order status is '{status}'; refunds can be requested once "
                  f"the order has been delivered or cancelled.",
    }
