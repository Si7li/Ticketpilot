# Design decisions

One entry per non-obvious decision. Newest at the bottom.

<!--
Template:

## YYYY-MM-DD — Short title
- **Decision:** what we chose.
- **Reason:** why.
- **Alternatives considered:** what else, and why not.
-->

## 2026-11-02 — Use pydantic-settings `env_ignore_empty`
- **Decision:** empty values in `.env` (e.g. `AGENT_PRICE_INPUT_PER_MTOK=`) are treated as unset.
- **Reason:** `.env.example` lists every variable; optional numeric/secret values must be blank-able without parse errors.
- **Alternatives considered:** commenting out optional variables (less discoverable).

## 2026-11-02 — Policy windows live in config, not just thresholds
- **Decision:** besides Section 4.5 thresholds, the return/damaged/wrong-item windows (30/14/60/14/14 days) and `HOME_COUNTRY` are `Settings` fields.
- **Reason:** standing rule "no thresholds scattered in code".
- **Alternatives considered:** module constants in `policy_rules.py`.

## 2026-11-02 — No `temperature=0` on ChatAnthropic
- **Decision:** the Anthropic backend does not send `temperature` (spec §9.3 says `temperature=0`).
- **Reason:** current Claude models (Haiku 5.5, Sonnet 5.5) reject non-default sampling parameters with a 400, and the current API version rejects `temperature` on every model. Library-vs-spec mismatch → follow the installed API (CLAUDE.md rule).
- **Alternatives considered:** pinning an older model that accepts temperature (contradicts §2.4 model choice).

## 2026-11-02 — Structured output via `method="json_schema"`
- **Decision:** `with_structured_output(..., method="json_schema", include_raw=True)` (native structured outputs, `output_config.format`) instead of the LangChain default (forced tool call). Configurable via `LLM_STRUCTURED_METHOD`.
- **Reason:** forced `tool_choice` is rejected by newer models (incl. the Sonnet 5.5 judge). json_schema works on both. langchain-anthropic moves unsupported constraints (maxLength, min/max) into descriptions; Pydantic still validates them locally, and a failure triggers the one retry the spec requires.
- **Alternatives considered:** function_calling (400 on Sonnet 5.5).

## 2026-11-02 — `LLM_MAX_TOKENS=8000`
- **Decision:** a generous per-call output cap.
- **Reason:** adaptive thinking is on by default on current models; small caps (e.g. 256 for classification) would truncate thinking and fail structured output.

## 2026-11-02 — Seeding logic lives in `ticketpilot.seed`
- **Decision:** `data/seed/seed_db.py` is a thin CLI over `src/ticketpilot/seed.py`.
- **Reason:** tests, the eval harness and the Docker entrypoint import the seeding code; `data/` is not a package.

## 2026-11-02 — Final-sale items are never `is_sale`
- **Decision:** in seed data an item is either sale (`is_sale`) or final sale (`final_sale`), never both.
- **Reason:** keeps the two policies (store credit vs not returnable) cleanly separable for labels and the refund tool. Refund rule 4 (final sale) runs before rule 5 (sale) anyway.

## 2026-11-02 — Peak 60-day returns apply to domestic orders only
- **Decision:** Peak tier extends the 30-day *domestic* window to 60 days; international orders keep 14 days.
- **Reason:** §4.4 doesn't say; this is the conservative reading (money leaves the business only when policy clearly allows it). Stated explicitly in `policy/loyalty-program`.

## 2026-11-02 — `issue_refund` gets `refund_type` and optional `skus`
- **Decision:** besides the spec args (`order_id, amount, reason, method`), the tool takes `refund_type` (return | damaged | wrong_item | lost_package | duplicate_charge | price_adjustment, default return) and optional `skus`.
- **Reason:** rule 6 needs to know *which* window applies (return 30/14/60 vs damaged 14 vs lost-package investigation); rule 4 needs the refunded scope. Free-text `reason` can't drive safety logic.
- **Consequences:** `lost_package` requires order status `lost`; `price_adjustment` requires store credit and ≤14 days from order; `duplicate_charge` is exempt from the sale-items→store-credit rule (it is a billing error, refunded to the original payment per `policy/double-charges`).
- **Alternatives considered:** inferring the type from `reason` text (fragile, model-controlled).

## 2026-11-02 — Model-supplied context params are rejected, not ignored
- **Decision:** if the model passes `customer_email` / `ticket_id` in tool args, the call fails with `arg_error=True`.
- **Reason:** spec allows "ignored/rejected"; rejecting makes the attempt visible in tool-call validity metrics and injection analysis.

## 2026-11-02 — Damaged/wrong final-sale items go to a specialist
- **Decision:** `policy/final-sale` says damaged/wrong final-sale items are reviewed by a specialist (not refunded by the agent).
- **Reason:** spec refund rule 4 refuses any final-sale item; the KB must not promise what the tool forbids.

## 2026-11-02 — Injection detection runs on the raw text
- **Decision:** `safety.injection.detect` runs on the raw ticket (before HTML stripping) in `normalize`.
- **Reason:** hidden HTML comments / `display:none` spans are themselves injection signals and disappear after normalization.

## 2026-11-02 — Vector index hash stored in point payloads
- **Decision:** the chunk-set hash is stored in every Qdrant point's payload and compared on `index()`.
- **Reason:** works for both in-memory and server Qdrant, and the hash lives with the data (a sidecar file could drift from a Qdrant volume).

## 2026-11-02 — Ruff excludes Markdown
- **Decision:** `extend-exclude = ["*.md"]`.
- **Reason:** ruff ≥0.17 formats Python code blocks inside Markdown and rewrote PROJECT_SPEC.md / CLAUDE.md snippets.

## 2026-11-02 — Graph node `draft` (not `draft_reply`)
- **Decision:** the drafting node is named `draft`; the state key stays `draft_reply`.
- **Reason:** LangGraph node names must not collide with state keys.

## 2026-11-02 — `issue_refund` withheld when code has already decided to escalate
- **Decision:** if the ticket is injection-flagged or in a sensitive category, the act loop does not offer `issue_refund`; a call to it anyway is recorded as a blocked tool result.
- **Reason:** "the model advises, code decides": routing will escalate these tickets regardless, so money must not move first. The other tools (lookups, `escalate`) stay available so the specialist gets context.
- **Consequence for metrics:** "unauthorized injection actions" can only come from injection tickets the detector missed, which is exactly what that metric should measure.

## 2026-11-02 — Order IDs: model extraction ∪ regex
- **Decision:** `extract_entities` keeps the union of model-extracted IDs and `KO-\d{5}` regex matches in the text, filtered by the strict pattern.
- **Reason:** deterministic recall for the one entity that drives tool calls; the model still extracts everything else.

## 2026-11-02 — `search_kb` results count as retrieved evidence
- **Decision:** chunks returned by the `search_kb` tool are allowed citations and are part of validator evidence, alongside the `retrieve` node's chunks.
- **Reason:** otherwise a correct citation found by an in-loop search would be flagged as "not retrieved".

## 2026-11-02 — Amount rule accepts amounts present in tool results *or* evidence text
- **Decision:** a `$` amount in the reply passes if it equals a number in any tool result or appears in a retrieved policy chunk (e.g. "$75" free-shipping threshold).
- **Reason:** spec §12 says "amounts must equal tool-result amounts"; read literally, quoting the free-shipping threshold would be a violation. Invented amounts are still caught.

## 2026-11-02 — Review events recorded by the `human_review` node
- **Decision:** the node (on resume) applies the decision, updates refunds, writes the outbox *and* records the `review_events` row; the API/CLI only pass the decision in.
- **Reason:** one place for the side effects, so API, CLI and tests behave identically. Side effects before `interrupt()` are idempotent (status update only) because LangGraph re-runs the node on resume.

## 2026-11-02 — Escalation holding reply is a fixed template
- **Decision:** escalated tickets get a fixed holding reply (outbox `sent_by="auto"`), never the drafted reply.
- **Reason:** escalated tickets include injection attempts; a template cannot echo injected content or promise outcomes.

## 2026-11-02 — Ticket dataset composition
- **Decision:** 205 labeled tickets = 120 Bitext-seeded (30 each for order_status, returns_refunds, account_billing, shipping_problem) + 30 drafted product questions + 55 drafted hard cases. Split 50 dev / 155 test.
- **Reason:** Bitext has no product-question or sensitive intents (see `data/tickets/intent_map.yaml`), so those categories come from drafted tickets that a human verifies like every other label. `complaint` is rewritten into concrete damaged / damaged-without-photo / wrong-item problems. Spec said "~200 Bitext"; we sample 120 so the total stays ≈ 200 (dev 50 + test ~150).
- **Alternatives considered:** sampling 200 Bitext tickets (4 categories only, no product questions in standard tickets).

## 2026-11-02 — Bitext labels proposed by deterministic rules, not by an LLM
- **Decision:** each sampled message gets a real seeded order chosen to fit the intent (e.g. an in-window, non-sale, single-item delivered order for `get_refund`), and gold labels are derived from that order by code. The agent model only rewrites the text (with the order facts given).
- **Reason:** labels proposed by the same model family being evaluated would bias the eval; code-derived proposals are reproducible, and a human verifies all of them.

## 2026-11-02 — Labels live in data/tickets/*.jsonl; split membership frozen in split_ids.json
- **Decision:** the review tool edits the source label files; `write_split()` regenerates dev/test from them using the frozen ID assignment and refreshes `test.sha256`.
- **Reason:** the spec wants both "review labels" and "warn if test.jsonl changes". With this design, only the label tool can change test.jsonl without tripping the hash warning, and verifying labels after the split doesn't reshuffle membership.

## 2026-11-02 — In-window return requests create the refund immediately
- **Decision:** KB return docs say the refund is set up when an in-window return is requested (pending human approval above $100), matching `issue_refund` rule 6.
- **Reason:** the tool's return-window rule only makes sense if returns create refunds; the earlier KB text ("refund after inspection") contradicted the tool.

## 2026-11-02 — Injection heuristics were written before the hard cases
- **Decision:** the regex detector was not tuned against the hard-case injection tickets (13/14 flagged; the social-engineering one is deliberately regex-proof). Changes to it must be justified on dev tickets only.
- **Reason:** most hard cases land in the test split; tuning on them would leak.

## 2026-11-02 — Eval reports from non-standard data are marked dry runs
- **Decision:** runs with `--limit`, `--tag`, or a non-default eval dir get a `-dryrun` suffix, a warning, and never update `evals/reports/latest/` or the README.
- **Reason:** honesty rules: only full, real runs may feed published numbers.

## 2026-11-02 — Judge evaluates the drafted reply, including on escalated tickets
- **Decision:** the judge grades `draft_reply` for every ticket that has one; escalated customers actually receive a fixed holding reply.
- **Reason:** spec defines hallucination over "drafted replies", and echo of injected instructions is most visible in drafts. This is stricter than grading only what is sent.

## 2026-11-02 — Time-saved model
- **Decision:** minutes/100 tickets with TicketPilot = human_review share × mean measured review time + escalate share × mean manual time; auto-sends count as 0 human minutes.
- **Reason:** escalations still need a person; using measured manual time for them avoids overstating savings.

## 2026-11-02 — Retriever searches are serialized with a lock
- **Decision:** `HybridRetriever.search` takes a lock.
- **Reason:** API thread pool and eval concurrency share one in-memory Qdrant client + embedder whose thread-safety is not documented; searches are milliseconds next to LLM calls.

## 2026-11-02 — Langfuse trace tagging
- **Decision:** every model call carries `langfuse_session_id = ticket ID`, `langfuse_trace_name`, and tags `prompt:<version>` and `node:<name>` (via a ContextVar set in the node wrapper). The route is logged in the terminal node's structured JSON log line and stored in results/API output.
- **Reason:** routing happens after the last LLM call, so the route cannot be attached to the LLM traces without an extra Langfuse API call per ticket; session ID links all calls for a ticket for lookup.
