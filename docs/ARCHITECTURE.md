# Architecture

## Request path

1. `POST /tickets` (`api/main.py`) creates a `tickets` row and calls `run_ticket()` (`graph/build.py`) with `thread_id = ticket_id`.
2. The LangGraph `StateGraph(TicketState)` runs the linear nodes, then `choose_route()` picks a terminal node:

| node | what it does | LLM? |
|---|---|---|
| `normalize` | strip HTML, quoted replies, signatures, sign-offs; run the injection detector on the **raw** text | no |
| `classify` | category + confidence + reasoning (structured output) | yes |
| `extract_entities` | order IDs (model ∪ regex, filtered by `^KO-\d{5}$`), products, amounts, photo flag, requests | yes |
| `retrieve` | hybrid search with query = clean text + category name | no |
| `act` | tool loop, ≤ `MAX_TOOL_STEPS` model steps; tools executed by the registry with injected context | yes |
| `draft` | cited reply from excerpts + tool results; citations not in the retrieved set are dropped and flagged | yes |
| `validate` | LLM claim check + rule checks (pending refund described as done, refund method, amounts, citations) | yes |
| `auto_send` / `human_review` / `escalate` | outbox, `interrupt()`, escalation record + holding reply | no |

3. Every node is wrapped by `_node()`: timing, per-node token usage (ContextVar scope), a JSON log line, a Langfuse trace context, and **fail-safe** error handling (exceptions become `errors`, which route to human review).

## Routing (`graph/routing.py`)

Pure function of state + settings, evaluated in this order: sensitive category or injection flag → escalate; specialist tool result → escalate; errors → review; not grounded / validation issues → review; pending refund → review; confidence below threshold → review; else auto-send.

## Tools (`tools/`)

`ToolRegistry.execute()` validates arguments with Pydantic (`extra="forbid"`), rejects model-supplied context params, injects `customer_email` / `ticket_id` from state, catches exceptions into structured `ToolResult`s and records duration. Refund rules (`tools/refunds.py` + `safety/policy_rules.py`) run in the order of PROJECT_SPEC.md §10.3. When code has already decided to escalate (injection flag or sensitive category), `issue_refund` is not offered to the model at all.

## Retrieval (`retrieval/`)

`chunking.py` splits on `##`, merges sections under 80 tokens, splits sections over 500 tokens, and prefixes `"{title} — {section}"`. `bm25.py` (rank-bm25, ID- and degree-aware tokenizer), `vector.py` (fastembed `bge-small-en-v1.5` → Qdrant, cosine; idempotent via a chunk-set hash stored in point payloads), `fusion.py` (RRF, k=60), `retriever.py` (mode switch; hybrid fuses the top 20 from each).

## Data

- Business DB (SQLite, `data/seed/schema.sql`): customers, orders, refunds (unique active refund per order), plus app tables tickets, escalations, outbox, review_events. Seeded deterministically relative to `REFERENCE_DATE`; scenario orders for hard cases are listed in `data/seed/scenarios.yaml`.
- Checkpoints: `SqliteSaver` at `CHECKPOINT_DB_PATH` (API/CLI); `InMemorySaver` in tests and evals.
- Labels: `data/tickets/*.jsonl` are the source of truth; `data/eval/{dev,test}.jsonl` are regenerated from them with frozen membership (`split_ids.json`) and `test.sha256`.

## LLM backend (`llm/`)

Nodes call the `LLMBackend` protocol only. `AnthropicBackend` wraps `ChatAnthropic` (structured output via `json_schema`, one validation retry, usage recorded per call); `FakeLLMBackend` is scripted per call site for tests. All prompts and `PROMPT_VERSION` are in `llm/prompts.py`.

## Evaluation (`evals/`)

`harness.py` runs the real graph per ticket on a fresh DB copy (thread pool, retry on transient API errors), then the judge (`judge.py`, different model + prompt, cached by input hash). `metrics.py` computes every metric with Wilson intervals; `sweep.py` re-routes stored results across thresholds; `calibration.py` measures judge–human agreement (κ); `ablations.py` compares retrieval modes and validation on/off; `gate.py` guards regressions in CI.
