# CLAUDE.md — Standing rules for TicketPilot

## Honesty rules (most important)
- NEVER invent, estimate, or "fill in" evaluation numbers. Every metric in the
  README, docs, or commit messages must come from a file produced by an actual
  eval run under evals/reports/. If a number does not exist yet, write "TBD".
- README results are inserted only by scripts/update_readme_results.py from a
  real metrics.json. Never hand-edit numbers between the RESULTS markers.
- Gold labels in data/eval/*.jsonl are only trustworthy when "verified": true,
  which a human sets via scripts/review_labels.py. Never set verified=true
  yourself. The eval harness must warn loudly when unverified labels are used.
- Never tune prompts, thresholds, or retrieval settings while looking at
  data/eval/test.jsonl. Use data/eval/dev.jsonl for all tuning. The test set is
  run only for reporting.

## Engineering rules
- Python 3.12, managed with uv. Source lives in src/ticketpilot/.
- Run `make lint test` before declaring any phase done. Both must pass.
- Type hints everywhere; mypy must pass on src/.
- All LLM-facing decisions use Pydantic schemas (structured output).
- Safety-critical rules (refund limits, escalation rules, routing) live in plain
  Python, never only in prompts.
- Unit tests must not call any network API. Use FakeLLMBackend.
- Configuration comes from src/ticketpilot/config.py (pydantic-settings),
  read from environment variables / .env. No hard-coded secrets or thresholds
  scattered in code.
- If a library API in PROJECT_SPEC.md doesn't match the installed version,
  follow the installed library's documentation, and record the deviation in
  docs/DECISIONS.md.
- Record every non-obvious design decision in docs/DECISIONS.md
  (date, decision, reason, alternatives considered).
- Keep commits small, one per logical step, with clear messages.

## Workflow rules
- Work one phase at a time. At the end of a phase: run acceptance checks,
  summarize, list deviations, then stop and wait.
- When a phase has a HUMAN CHECKPOINT, stop and tell the user exactly what to
  do and which command to run.
