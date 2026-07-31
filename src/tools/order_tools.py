"""
Order-related tool functions.

These read from the synthetic data/orders/orders.csv file and return
plain structured dictionaries, similar to how a LangChain "tool" would
return data to an agent. No real order system is involved.
"""

import csv
from typing import Any

from src.config import ORDERS_CSV


def _load_orders() -> dict[str, dict[str, str]]:
    """Load all orders from the CSV into a dict keyed by order_id."""
    orders: dict[str, dict[str, str]] = {}

    if not ORDERS_CSV.exists():
        return orders

    with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orders[row["order_id"]] = row

    return orders


def lookup_order_status(order_id: str) -> dict[str, Any]:
    """
    Look up the current status of an order by its order_id.

    Returns a structured dict describing whether the order was found
    and, if so, its shipping status and details. Lookups are
    case-insensitive (e.g. "ord-1001" matches "ORD-1001").
    """
    orders = _load_orders()
    normalized_id = order_id.upper() if order_id else order_id
    order = orders.get(normalized_id)

    if order is None:
        return {
            "found": False,
            "order_id": order_id,
            "customer_name": None,
            "status": None,
            "product": None,
            "carrier": None,
            "estimated_delivery": None,
            "total_amount": None,
            "message": f"No order found with id '{order_id}'.",
        }

    return {
        "found": True,
        "order_id": order["order_id"],
        "customer_name": order["customer_name"],
        "status": order["status"],
        "product": order["product"],
        "carrier": order["carrier"] or None,
        "estimated_delivery": order["estimated_delivery"] or None,
        "total_amount": float(order["amount"]),
    }
