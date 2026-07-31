# Customer Support Agentic RAG (Version 1)

A beginner-friendly demo of an **agentic workflow** for fictional e-commerce customer
support, built with [LangGraph](https://github.com/langchain-ai/langgraph) and simple
LangChain-style tools. No paid APIs, vector databases, or LLM calls are used yet — the
goal is to make the *shape* of an agentic system (state, nodes, tools, routing, RAG)
easy to read and understand.

## Project Goal

Demonstrate the basics of an agentic customer support assistant:
- A typed, shared **state** object that flows through a graph
- A small set of **nodes** that each do one job
- **Tool** functions the agent can call (order lookup, refund check, ticket creation)
- A minimal **retrieval** step over local FAQ documents (no vector DB required)
- Simple **rule-based** intent classification (no LLM required yet)

All data (orders, tickets, FAQs) is synthetic and fictional. This project does not use
real customer data or macroeconomic data.

## Architecture

```
Customer question
      |
      v
[receive_question]   -> normalizes the question, extracts an order id (e.g. ORD-1001)
      |
      v
[classify_intent]    -> rule-based keyword classifier assigns an intent
      |
      v
[route_request]      -> based on intent, calls a tool and/or the FAQ retriever
      |
      v
[generate_response]  -> composes the final natural-language answer
      |
      v
Final answer + structured trace (intent, tool_result, evidence, escalation flag)
```

**State object** (`src/graph/state.py`): `question`, `intent`, `order_id`, `evidence`,
`tool_result`, `needs_escalation`, `final_answer`.

**Possible intents**: `order_status`, `refund_request`, `return_policy`,
`shipping_question`, `warranty_question`, `complaint_escalation`, `general_faq`, `unknown`.

**Tools** (`src/tools/`):
- `lookup_order_status(order_id)` — reads `data/orders/orders.csv`
- `check_refund_eligibility(order_id)` — simple 30-day refund rule
- `create_support_ticket(summary, priority)` — creates an in-memory ticket dict

**Retrieval** (`src/rag/retriever.py`): splits the FAQ markdown files in `data/faq/`
into sections and returns the sections that share the most keywords with the question.
No embeddings or external vector store are used in Version 1.

## Installation

```bash
git clone <your-repo-url>
cd customer-support-agentic-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at
`http://127.0.0.1:8000/docs`.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is my order ORD-1001?"}'
```

## Example Questions

- "Where is my order ORD-1001?" → `order_status`
- "I want a refund for order ORD-1003." → `refund_request`
- "What is your return policy?" → `return_policy`
- "How long does shipping take?" → `shipping_question`
- "Is my keyboard covered under warranty?" → `warranty_question`
- "I'm furious, I want to speak to a manager!" → `complaint_escalation`
- "How do I reset my password?" → `general_faq`
- "What's the best pizza topping?" → `unknown`

## Running Tests

```bash
pytest
```

## Version Roadmap

- **v1 (current)** — Rule-based intent classification, keyword-based FAQ retrieval,
  linear LangGraph workflow, in-memory tools, FastAPI endpoint.
- **v2** — Swap rule-based classification for an LLM-based classifier; add conditional
  routing/branches in the graph based on tool results.
- **v3** — Add real vector-based retrieval (e.g. Chroma) and embeddings for the FAQ data.
- **v4** — Persistent storage for orders/tickets (a real database instead of CSV/JSON),
  and a simple front-end (e.g. Streamlit) for demoing the assistant.
