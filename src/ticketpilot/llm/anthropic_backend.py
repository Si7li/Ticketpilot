"""Real LLM backend: Anthropic models via langchain-anthropic's ChatAnthropic.

Notes on current-model API constraints (recorded in docs/DECISIONS.md):
- No `temperature`: current Claude models reject non-default sampling parameters.
- Structured output uses `method="json_schema"` (native structured outputs) rather than
  forced tool calls, because forced `tool_choice` is rejected by newer models (e.g. the
  Sonnet 5.5 judge). The method is configurable via `LLM_STRUCTURED_METHOD`.
- Validation failures are retried once with the error appended (PROJECT_SPEC.md §9.3).
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from ticketpilot.config import Settings
from ticketpilot.llm import prompts
from ticketpilot.llm.backend import BackendError
from ticketpilot.schemas import (
    Classification,
    DraftContext,
    DraftReply,
    Entities,
    Evidence,
    ValidationResult,
)
from ticketpilot.telemetry import UsageTracker, langfuse_handler

log = logging.getLogger("ticketpilot.llm")

T = TypeVar("T", bound=BaseModel)


def make_chat_model(settings: Settings, model: str) -> ChatAnthropic:
    if settings.ANTHROPIC_API_KEY is None:
        raise BackendError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatAnthropic(
        model=model,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=settings.LLM_MAX_RETRIES,
        timeout=settings.LLM_TIMEOUT_S,
    )


class StructuredCaller:
    """Structured-output calls with one validation retry and usage accounting."""

    def __init__(
        self,
        llm: ChatAnthropic,
        model_name: str,
        usage: UsageTracker,
        method: Literal["json_schema", "function_calling"],
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.usage = usage
        self.method = method

    def record(self, call_site: str, message: Any) -> None:
        meta = getattr(message, "usage_metadata", None) or {}
        self.usage.record(
            call_site,
            self.model_name,
            int(meta.get("input_tokens", 0)),
            int(meta.get("output_tokens", 0)),
        )

    def call(
        self,
        call_site: str,
        schema: type[T],
        system: str,
        user: str,
        config: dict[str, Any] | None = None,
    ) -> T:
        runnable = self.llm.with_structured_output(schema, method=self.method, include_raw=True)
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        last_error: str = ""
        for attempt in range(2):
            try:
                out = runnable.invoke(messages, config=config)  # type: ignore[arg-type]
            except ValidationError as exc:  # some versions raise instead of returning
                last_error = str(exc)
            else:
                assert isinstance(out, dict)
                self.record(call_site, out.get("raw"))
                parsed, err = out.get("parsed"), out.get("parsing_error")
                if parsed is not None and err is None:
                    if isinstance(parsed, schema):
                        return parsed
                    try:
                        return schema.model_validate(parsed)
                    except ValidationError as exc:
                        last_error = str(exc)
                else:
                    last_error = str(err or "no parsable output")
            log.info("structured output retry", extra={"call_site": call_site, "attempt": attempt})
            messages = [
                *messages,
                HumanMessage(
                    "Your previous output failed schema validation:\n"
                    f"{last_error[:1500]}\n"
                    "Respond again with output that matches the schema exactly."
                ),
            ]
        raise BackendError(f"{call_site}: structured output invalid after retry: {last_error}")


class AnthropicBackend:
    def __init__(self, settings: Settings, model: str | None = None) -> None:
        self.settings = settings
        self.model_name = model or settings.AGENT_MODEL
        self.llm = make_chat_model(settings, self.model_name)
        self.usage = UsageTracker()
        self._structured = StructuredCaller(
            self.llm, self.model_name, self.usage, settings.LLM_STRUCTURED_METHOD
        )

    def _config(self) -> dict[str, Any] | None:
        handler = langfuse_handler(self.settings)
        if handler is None:
            return None
        return {"callbacks": [handler], "metadata": {"prompt_version": prompts.PROMPT_VERSION}}

    def classify(self, text: str) -> Classification:
        return self._structured.call(
            "classify",
            Classification,
            prompts.CLASSIFY_SYSTEM,
            prompts.CLASSIFY_USER.format(text=text),
            self._config(),
        )

    def extract(self, text: str, category: str) -> Entities:
        return self._structured.call(
            "extract",
            Entities,
            prompts.EXTRACT_SYSTEM,
            prompts.EXTRACT_USER.format(text=text, category=category),
            self._config(),
        )

    def act_step(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        bound = self.llm.bind_tools(tools) if tools else self.llm
        # The full AIMessage (including any thinking blocks) is appended to the transcript by
        # the caller unchanged, which keeps the conversation append-only.
        result = bound.invoke(messages, config=self._config())  # type: ignore[arg-type]
        assert isinstance(result, AIMessage)
        self._structured.record("act_step", result)
        return result

    def draft(self, context: DraftContext) -> DraftReply:
        return self._structured.call(
            "draft",
            DraftReply,
            prompts.DRAFT_SYSTEM,
            prompts.draft_user_prompt(context),
            self._config(),
        )

    def check_claims(self, reply: str, evidence: list[Evidence]) -> ValidationResult:
        return self._structured.call(
            "check_claims",
            ValidationResult,
            prompts.CHECK_CLAIMS_SYSTEM,
            prompts.CHECK_CLAIMS_USER.format(
                reply=reply, evidence=prompts.format_evidence(evidence)
            ),
            self._config(),
        )
