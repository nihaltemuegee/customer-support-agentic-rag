# Customer Support Agentic RAG (Version 2)

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
No embeddings or external vector store are used yet — see the "Version 2: FAQ RAG" section
below for details.

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

A question that's grounded in the FAQ docs (e.g. shipping/returns/refunds/warranty/account)
also returns an `evidence` list of the FAQ chunks used to answer it:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your return policy?"}'
```

```json
{
  "question": "What is your return policy?",
  "intent": "return_policy",
  "order_id": null,
  "needs_escalation": false,
  "tool_result": null,
  "evidence": [
    { "source": "returns.md", "text": "What is your return policy?: Most items can be returned within 30 days of delivery..." }
  ],
  "final_answer": "Most items can be returned within 30 days of delivery, as long as they are unused and in their original packaging."
}
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

## Version 2: FAQ RAG

Version 2 improves the retrieval step used to ground FAQ answers. It's still a plain,
local, keyword-based retriever — no embeddings, no vector database — but it's more
deliberate about how chunks are built and scored, and it exposes evidence as structured
data instead of formatted strings.

**Chunking.** `load_faq_sections()` reads every `.md` file in `data/faq/` and splits it
on `## ` headings, so each FAQ question (e.g. "How long does shipping take?") becomes its
own chunk, tagged with its source filename.

**Scoring.** `retrieve(query, top_k=3)` tokenizes the query and each chunk into lowercase
word sets, then scores a chunk by how many words it shares with the query
(`len(query_words & chunk_words)`). The top-scoring chunks are returned. A small
`STOPWORDS` set (`is`, `my`, `the`, `how`, ...) is filtered out of scoring — without it,
filler words that appear in almost every FAQ section would create false matches and drown
out the real signal (e.g. "warranty").

**Evidence shape.** `retrieve()` now returns a list of dicts instead of formatted
strings:

```python
[{"source": "shipping.md", "text": "How long does shipping take?: Standard shipping..."}]
```

This is the same shape returned by the `/ask` API's `evidence` field.

**Where retrieval is used.** `route_request` (`src/graph/nodes.py`) calls `retrieve()`
only for the intents where FAQ policy text is the right grounding source:
`shipping_question`, `return_policy`, `refund_request`, `warranty_question`, and
`general_faq`. `order_status` and `complaint_escalation` are answered from tool calls
instead (an order lookup / a created ticket), since those need order-specific or
per-request data that a static FAQ document can't provide.

**Grounded answers.** `generate_response` builds the `final_answer` for FAQ-grounded
intents directly from the top-scoring evidence chunk's text, rather than a hardcoded
string — so the answer changes if the FAQ content changes. For `refund_request`, the
tool result (`check_refund_eligibility`) remains the primary source of truth for the
answer text, since it's order-specific, but the FAQ evidence is still attached to the
response for extra context.

## Version Roadmap

- **v1** — Rule-based intent classification, linear LangGraph workflow, in-memory tools,
  FastAPI endpoint.
- **v2 (current)** — Chunked, stopword-filtered FAQ retrieval; structured
  `{source, text}` evidence; FAQ-grounded answers wired into `generate_response`.
- **v3** — Swap rule-based classification for an LLM-based classifier; add conditional
  routing/branches in the graph based on tool results.
- **v4** — Add real vector-based retrieval (e.g. Chroma) and embeddings for the FAQ data.
- **v5** — Persistent storage for orders/tickets (a real database instead of CSV/JSON),
  and a simple front-end (e.g. Streamlit) for demoing the assistant.
