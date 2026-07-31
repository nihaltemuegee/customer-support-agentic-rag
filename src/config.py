"""
Central configuration for the Customer Support Agentic RAG project.

Keeps file paths and simple settings in one place so nodes/tools
don't hardcode paths all over the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root = two levels up from this file (src/config.py -> project root)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
FAQ_DIR = DATA_DIR / "faq"
ORDERS_CSV = DATA_DIR / "orders" / "orders.csv"
TICKETS_JSON = DATA_DIR / "tickets" / "sample_tickets.json"

# App settings (placeholders for now, no external services used yet)
APP_NAME = os.getenv("APP_NAME", "customer-support-agentic-rag")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Default priority used when a support ticket doesn't specify one
DEFAULT_TICKET_PRIORITY = "normal"
