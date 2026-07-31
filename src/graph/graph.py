"""
Builds and compiles the LangGraph workflow for the customer support agent.

Graph shape (Version 1, linear):

    receive_question -> classify_intent -> route_request -> generate_response -> END
"""

from langgraph.graph import StateGraph, END

from src.graph.state import SupportState
from src.graph.nodes import (
    receive_question,
    classify_intent,
    route_request,
    generate_response,
)


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("receive_question", receive_question)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("route_request", route_request)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("receive_question")
    graph.add_edge("receive_question", "classify_intent")
    graph.add_edge("classify_intent", "route_request")
    graph.add_edge("route_request", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


# Compiled once at import time and reused across requests.
support_graph = build_graph()


def run_support_workflow(question: str, previous_intent: str | None = None) -> SupportState:
    """
    Run the full graph for a single customer question and return the final state.

    previous_intent is optional context from the prior turn (see Version 6:
    multi-turn support in nodes.classify_intent) -- pass it when the previous
    turn's answer asked the customer for an order id, so a bare follow-up
    like "ORD-1001" is understood as continuing that same request.
    """
    initial_state: SupportState = {"question": question}
    if previous_intent:
        initial_state["previous_intent"] = previous_intent
    result = support_graph.invoke(initial_state)
    return result
