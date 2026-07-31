"""
Streamlit UI for the Customer Support Agentic RAG assistant.

This is a thin presentation layer only -- it does not reimplement any
agent logic. It calls the same run_support_workflow() function used by
the FastAPI app (src/graph/graph.py) directly when possible, and falls
back to calling the running FastAPI /ask endpoint over HTTP otherwise.

Run with:
    streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make the project root importable so "from src...." works, since Streamlit
# only puts this file's own directory (ui/) on sys.path by default.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.graph.graph import run_support_workflow
    BACKEND_MODE = "direct (imported graph function)"
    _DIRECT_IMPORT_OK = True
except ImportError:
    run_support_workflow = None
    BACKEND_MODE = "API (http://127.0.0.1:8000/ask)"
    _DIRECT_IMPORT_OK = False

import requests

API_URL = "http://127.0.0.1:8000/ask"

EXAMPLE_QUESTIONS = [
    "Where is my order ORD-1001?",
    "Can I get a refund for ORD-1003?",
    "Do you ship internationally?",
    "My order ORD-1005 arrived damaged. What should I do?",
    "I am very angry and want to speak to a human.",
    "What is your warranty policy?",
]

# Intents where a missing order id means the conversation should continue once
# the customer supplies one. Mirrors FOLLOW_UP_INTENTS in src/graph/nodes.py --
# kept as a small local constant so the UI works even in API-fallback mode,
# without importing internal graph modules.
FOLLOW_UP_INTENTS = {"order_status", "refund_request"}


def ask_backend(question: str, previous_intent: str | None = None) -> dict:
    """
    Run a question through the agent workflow.

    Prefers calling run_support_workflow() directly (same function the
    FastAPI app uses) so the UI never reimplements the graph. Falls back
    to calling the FastAPI /ask endpoint over HTTP if the direct import
    isn't available (e.g. the UI is deployed separately from the backend).

    previous_intent carries basic multi-turn context (see Version 6): when
    set, a bare follow-up like "ORD-1003" continues that prior request
    instead of being reclassified from scratch.
    """
    if _DIRECT_IMPORT_OK:
        return run_support_workflow(question, previous_intent=previous_intent)

    payload = {"question": question}
    if previous_intent:
        payload["previous_intent"] = previous_intent
    response = requests.post(API_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def run_question(question: str) -> None:
    """
    Call the backend and store the result (or error) in session state.

    Basic multi-turn support: if the previous turn asked for an order id
    (st.session_state.pending_intent is set), it's passed along automatically
    so a bare follow-up like "ORD-1003" continues that same request. After
    each call, pending_intent is recomputed from the fresh result -- it's
    only kept set while the conversation is still waiting on an order id.
    """
    previous_intent = st.session_state.get("pending_intent")

    try:
        result = ask_backend(question, previous_intent=previous_intent)
        st.session_state.result = result
        st.session_state.error = None

        tool_result = result.get("tool_result") or {}
        awaiting_order_id = (
            result.get("intent") in FOLLOW_UP_INTENTS
            and not result.get("order_id")
            and tool_result.get("found") is False
        )
        st.session_state.pending_intent = result.get("intent") if awaiting_order_id else None

        continued = previous_intent and result.get("intent") == previous_intent and result.get("order_id")
        st.session_state.continued_from = previous_intent if continued else None
    except Exception as exc:  # noqa: BLE001 -- surface any backend error to the UI
        st.session_state.result = None
        st.session_state.error = str(exc)


st.set_page_config(page_title="Customer Support Agentic RAG", layout="centered")

if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "pending_intent" not in st.session_state:
    st.session_state.pending_intent = None
if "continued_from" not in st.session_state:
    st.session_state.continued_from = None

with st.sidebar:
    st.header("About this demo")
    st.write(
        "Customer Support Agentic RAG assistant using LangGraph, local FAQ "
        "retrieval, tool routing, and escalation logic."
    )
    st.caption(f"Backend: {BACKEND_MODE}")

    st.header("Example questions")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, key=f"example::{example}", use_container_width=True):
            st.session_state.question_input = example
            run_question(example)

st.title("Customer Support Agentic RAG")
st.write(
    "A beginner-friendly demo of an agentic customer support workflow. "
    "Type a question below, or pick an example from the sidebar."
)

question = st.text_area("Your question", key="question_input", height=100)

if st.button("Ask", type="primary"):
    if question.strip():
        run_question(question)
    else:
        st.warning("Please enter a question first.")

if st.session_state.error:
    st.error(f"Something went wrong calling the backend: {st.session_state.error}")

result = st.session_state.result

if result:
    if st.session_state.continued_from:
        topic = st.session_state.continued_from.replace("_", " ")
        st.info(
            f"Continuing your previous {topic} request using order id "
            f"{result.get('order_id')}."
        )

    st.subheader("Answer")
    st.write(result.get("final_answer", ""))

    if result.get("needs_escalation"):
        st.warning("This request has been escalated to a human agent.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Intent", result.get("intent", "unknown"))
    with col2:
        st.metric("Order ID", result.get("order_id") or "N/A")

    st.subheader("Tool result")
    tool_result = result.get("tool_result")
    if tool_result:
        st.json(tool_result)
    else:
        st.caption("No tool was called for this question.")

    st.subheader("Escalation / ticket result")
    escalation_result = result.get("escalation_result") or result.get("ticket_result")
    if escalation_result:
        st.json(escalation_result)
    else:
        st.caption("No escalation ticket was created for this question.")

    st.subheader("Evidence / source documents")
    evidence = result.get("evidence") or []
    if evidence:
        for item in evidence:
            st.markdown(f"**{item.get('source', 'unknown')}**")
            st.write(item.get("text", ""))
            st.divider()
    else:
        st.caption("No FAQ evidence was retrieved for this question.")

    with st.expander("Raw JSON response"):
        st.json(result)
else:
    st.info("Ask a question above, or choose an example from the sidebar, to see a response.")
