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
