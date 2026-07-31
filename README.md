# Customer Support Agentic RAG (Version 5)

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
`tool_result`, `needs_escalation`, `escalation_result`, `final_answer`.

**Possible intents**: `order_status`, `refund_request`, `return_policy`,
`shipping_question`, `warranty_question`, `complaint_escalation`, `general_faq`, `unknown`.

**Tools** (`src/tools/`):
- `lookup_order_status(order_id)` — reads `data/orders/orders.csv`, returns order details
- `check_refund_eligibility(order_id, damaged=False)` — status-based refund rules, plus a
  damaged-item override
- `create_support_ticket(summary, priority)` — creates an in-memory ticket dict
  (`priority` is one of `high`/`medium`/`low`)

See "Version 3: Order & Refund Tool Routing" and "Version 4: Escalation & Ticketing"
below for how these are used.

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

To also run the Streamlit UI (see "Version 5: Streamlit UI" below), install its extra
dependency:

```bash
pip install -r requirements-ui.txt
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
  "escalation_result": null,
  "evidence": [
    { "source": "returns.md", "text": "What is your return policy?: Most items can be returned within 30 days of delivery..." }
  ],
  "final_answer": "Most items can be returned within 30 days of delivery, as long as they are unused and in their original packaging."
}
```

## Running the Streamlit UI

The Streamlit app is a thin UI on top of the same LangGraph workflow the API uses --
see "Version 5: Streamlit UI" below for how it avoids duplicating any agent logic.

```bash
streamlit run ui/streamlit_app.py
```

This opens at `http://localhost:8501`. The FastAPI server does **not** need to be
running for the UI to work, since the UI imports and calls the graph function directly;
it only falls back to calling `http://127.0.0.1:8000/ask` over HTTP if that direct
import isn't available.

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

## Version 3: Order & Refund Tool Routing

Version 3 makes the two order-specific tools -- `lookup_order_status` and
`check_refund_eligibility` -- more capable, and makes `route_request` orchestrate them
more deliberately instead of just calling them blindly.

**Why this demonstrates agentic tool use.** The graph doesn't just answer from an LLM's
own knowledge -- it decides *which* tool to call based on intent, calls it with
*structured* arguments (an order id, a `damaged` flag), reads the *structured result*
back, and sometimes chains a second tool off the first one's outcome. That
decide → call → read result → maybe call another tool loop is the core of what makes a
workflow "agentic" rather than a single prompt-response call, and it's easy to point at
in an interview: "here's the node that decides which tool to call, here's the tool
signature, here's where I chain a second tool."

**Order id extraction** (`extract_order_id` in `src/graph/nodes.py`) is a small, pure,
testable function pulled out of `receive_question`. It matches `ORD-\d+` case-insensitively
anywhere in the sentence, so `"ORD-1001"`, `"ord-1001"`, `"Order ORD-1001"`, and
`"my order is ORD-1001"` all resolve to the same normalized id.

**`lookup_order_status(order_id)`** (`src/tools/order_tools.py`) reads
`data/orders/orders.csv` and returns a structured dict: `order_id`, `customer_name`,
`status`, `product`, `carrier`, `estimated_delivery`, `total_amount`, and `found`. Lookups
are case-insensitive. If the id isn't found, `found` is `false` and the other fields are
`null` -- the tool never guesses.

**`check_refund_eligibility(order_id, damaged=False)`** (`src/tools/refund_tools.py`)
applies simple demo rules based on the order's status:
- `delivered` -> eligible if within the 30-day refund window
- `shipped` -> not yet eligible; must wait for delivery
- `processing` -> not eligible for a refund yet, but may still be cancelable
- `cancelled` -> eligible for a full refund
- `damaged=True` -> always eligible for a refund/replacement, and the `next_step`
  recommends contacting support, regardless of the order's normal status

Every result includes a human-readable `reason` and a `next_step`, so
`generate_response` can build a clear answer without re-deriving the logic.

**Tool routing** (`route_request` in `src/graph/nodes.py`):
- `order_status` -> `lookup_order_status(order_id)`.
- `refund_request` -> `check_refund_eligibility(order_id, damaged=...)`, grounded with
  refund/return FAQ evidence.
- If `order_status` or `refund_request` is detected but no order id is present,
  **no order id is invented**. No tool is called; `tool_result["found"]` is `false`, and
  `final_answer` asks the customer for their order id instead.

**Damaged package handling.** Questions like *"My order ORD-1005 arrived damaged"* or
*"My package was broken"* are detected by a small `DAMAGE_KEYWORDS` list and classified
as `refund_request` (not `warranty_question`), since a damaged-on-arrival item is a
refund/replacement case.

> **Note:** in Version 3, angry/urgent wording on a damaged-item question would override
> the intent to `complaint_escalation`, and a damage-triggered ticket was nested inside
> `tool_result`. Version 4 (below) changes both of these: the topic intent (e.g.
> `refund_request`) and escalation are now decided independently, and tickets live in
> their own `escalation_result` field.

## Version 4: Escalation & Ticketing

Version 4's core change is architectural: **escalation is now decided independently of
intent**, instead of being just another intent value. That single change is what lets
the same message both get a specific, useful answer (e.g. a refund tool result) *and*
get flagged for a human, at the same time.

**Why this demonstrates agentic workflow behavior.** A simple chatbot picks one branch
and answers it. This graph runs two decisions in parallel over the same input --
"what is this about" (`classify_intent_text`) and "does a human need to see this"
(`escalation_priority`) -- then composes a single answer out of both results
(`generate_response`). That's a small but real example of a workflow reasoning about a
request from more than one angle before responding, rather than a single
intent → response lookup.

**Escalation detection** (`escalation_priority()` in `src/graph/nodes.py`) checks, in
order:
1. **Emotion words** (`frustrated`, `furious`, `angry`, `upset`, `disappointed`,
   `unacceptable`, `manager`, `supervisor`, ...) -> `high`
2. **Urgency words** (`urgent`, `immediately`, `asap`) -> `high`
3. **Damaged/broken/missing/lost package wording** (the same `DAMAGE_KEYWORDS` used for
   `refund_request` classification, e.g. *"missing package"*, *"never arrived"*) -> `high`
4. **A refund case the rule-based tool couldn't resolve on its own** -- signaled by
   `check_refund_eligibility()`'s new `needs_human_review` field (set `True` for a
   damaged item, or for an order status the rules don't recognize) -> `medium`
5. **A plain request to talk to a human**, with none of the above -> `low`
6. Otherwise -> no escalation (`None`)

This function returns `None` or a priority string, not just `True`/`False`, so
`route_request` gets "should we escalate" and "how urgently" from one call.

**`create_support_ticket(summary, priority)`** (`src/tools/ticket_tools.py`) now
returns:

```python
{
    "ticket_id": "TICKET-1001",   # sequential-looking id (in-memory counter)
    "summary": "...",
    "priority": "high",           # "high" | "medium" | "low"
    "status": "open",
    "created": True,
    "next_step": "A support agent will follow up within 1 hour.",
}
```

**Integration into the graph** (`route_request`): after computing the topic-specific
`tool_result` (or none, for topics with no tool), `route_request` calls
`escalation_priority(question, tool_result)`. If it returns a priority,
`create_support_ticket(...)` is called and the result is stored in the state's
`escalation_result` field -- kept separate from `tool_result` so "the answer to their
question" and "the escalation ticket" don't get tangled together. `generate_response`
then appends an escalation notice (with the ticket id) to whatever answer it already
built, whenever `escalation_result` is set -- regardless of intent.

**Example: an angry refund question keeps its specific answer.**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "I am furious, I want a refund for order ORD-1002 immediately!"}'
```

```json
{
  "intent": "refund_request",
  "needs_escalation": true,
  "tool_result": {
    "found": true, "eligible": false,
    "reason": "Order has shipped but hasn't been delivered yet, so it isn't refund-eligible yet."
  },
  "escalation_result": { "ticket_id": "TICKET-1001", "priority": "high", "created": true },
  "final_answer": "Order has shipped but hasn't been delivered yet, so it isn't refund-eligible yet. Please wait until the order is delivered, then request a refund if needed. I've escalated this to our support team (ticket TICKET-1001) and a human agent will follow up shortly."
}
```

## Version 5: Streamlit UI

Version 5 adds a UI (`ui/streamlit_app.py`) purely as a presentation layer -- it does
not reimplement any part of the agent workflow.

**No duplicated logic.** The UI imports and calls `run_support_workflow()` from
`src/graph/graph.py` directly -- the exact same function `app/main.py`'s `/ask`
endpoint calls. Since Streamlit only puts the script's own folder on `sys.path` by
default, the app adds the project root to `sys.path` at the top of the file so
`from src...` imports resolve. If that import ever fails (e.g. the UI is deployed
separately from the backend code), it falls back to calling the FastAPI `/ask`
endpoint over HTTP with `requests` instead -- so the UI works either as a thin client
on top of the same process, or as a standalone client hitting a remote API, without two
copies of the graph logic to keep in sync.

**Layout:**
- **Sidebar** -- a short project description ("Customer Support Agentic RAG assistant
  using LangGraph, local FAQ retrieval, tool routing, and escalation logic.") and six
  example-question buttons that fill in the input and run the workflow immediately:
  - "Where is my order ORD-1001?"
  - "Can I get a refund for ORD-1003?"
  - "Do you ship internationally?"
  - "My order ORD-1005 arrived damaged. What should I do?"
  - "I am very angry and want to speak to a human."
  - "What is your warranty policy?"
- **Main panel** -- a text box and "Ask" button, then the result: the final answer, an
  escalation warning banner when `needs_escalation` is true, `intent` and `order_id` as
  metrics, `tool_result` and `escalation_result` as formatted JSON (with a plain-language
  fallback when either is empty), the FAQ `evidence` (source filename + text), and the
  complete raw JSON response inside a collapsed expander for inspecting the full state.

**Why this is still "agentic," not just a chat box.** The UI never talks to an LLM and
never decides anything itself -- every decision (intent, which tool to call, whether to
escalate) still happens inside the graph. The UI's only job is to call
`run_support_workflow()` once and render the structured result, which is exactly the
separation of concerns you'd want in a real product: the agent logic is UI-agnostic and
already has both a REST API and a UI in front of it, unchanged.

## Version Roadmap

- **v1** — Rule-based intent classification, linear LangGraph workflow, in-memory tools,
  FastAPI endpoint.
- **v2** — Chunked, stopword-filtered FAQ retrieval; structured
  `{source, text}` evidence; FAQ-grounded answers wired into `generate_response`.
- **v3** — Richer order-status/refund-eligibility tools; deliberate tool routing; order
  id extraction handles more phrasings; no order id is ever guessed.
- **v4** — Escalation decided independently of intent, with a `high`/`medium`/
  `low` priority model; ticket results live in their own `escalation_result` field;
  `final_answer` always mentions the ticket id when escalation happens.
- **v5 (current)** — Streamlit UI (`ui/streamlit_app.py`) as a thin layer over the same
  `run_support_workflow()` function the API uses, with an HTTP fallback; FastAPI is
  unchanged and still runs standalone.
- **v6** — Swap rule-based classification for an LLM-based classifier; add conditional
  routing/branches in the graph based on tool results.
- **v7** — Add real vector-based retrieval (e.g. Chroma) and embeddings for the FAQ data.
- **v8** — Persistent storage for orders/tickets (a real database instead of CSV/JSON).
