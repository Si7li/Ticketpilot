"""Scripted LLM backend for tests. No network, fully deterministic."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from ticketpilot.schemas import (
    Classification,
    DraftContext,
    DraftReply,
    Entities,
    Evidence,
    ValidationResult,
)
from ticketpilot.telemetry import UsageTracker

CALL_SITES = ("classify", "extract", "act_step", "draft", "check_claims")


class UnscriptedCallError(AssertionError):
    """A backend method was called with no scripted response left."""


Response = Any  # a model instance, an Exception to raise, or a callable(input) -> response


class FakeLLMBackend:
    """`FakeLLMBackend({"classify": [Classification(...)], "act_step": [AIMessage(...)]})`.

    Each call site pops the next response from its queue. A response may be:
    - the object to return,
    - an Exception instance (raised),
    - a callable receiving the call's input and returning the object.
    `defaults` (same keys) are returned when a queue is empty, instead of raising.
    """

    model_name = "fake-model"

    def __init__(
        self,
        script: dict[str, list[Response]] | None = None,
        defaults: dict[str, Response] | None = None,
    ) -> None:
        unknown = set(script or {}) | set(defaults or {})
        unknown -= set(CALL_SITES)
        if unknown:
            raise ValueError(f"unknown call sites in script: {sorted(unknown)}")
        self.queues: dict[str, deque[Response]] = {k: deque(v) for k, v in (script or {}).items()}
        self.defaults = defaults or {}
        self.calls: list[tuple[str, Any]] = []
        self.usage = UsageTracker()

    def _next(self, site: str, payload: Any) -> Any:
        self.calls.append((site, payload))
        q = self.queues.get(site)
        if q:
            resp = q.popleft()
        elif site in self.defaults:
            resp = self.defaults[site]
        else:
            raise UnscriptedCallError(
                f"FakeLLMBackend: no scripted response for '{site}' (call #{len(self.calls)})"
            )
        if isinstance(resp, Exception):
            raise resp
        if callable(resp) and not isinstance(resp, type):
            resp = resp(payload)
        self.usage.record(site, self.model_name, 10, 5)
        return resp

    def classify(self, text: str) -> Classification:
        result: Classification = self._next("classify", text)
        return result

    def extract(self, text: str, category: str) -> Entities:
        result: Entities = self._next("extract", {"text": text, "category": category})
        return result

    def act_step(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        result: AIMessage = self._next("act_step", {"messages": messages, "tools": tools})
        return result

    def draft(self, context: DraftContext) -> DraftReply:
        result: DraftReply = self._next("draft", context)
        return result

    def check_claims(self, reply: str, evidence: list[Evidence]) -> ValidationResult:
        result: ValidationResult = self._next(
            "check_claims", {"reply": reply, "evidence": evidence}
        )
        return result


def tool_call_message(*calls: tuple[str, dict[str, Any]], content: str = "") -> AIMessage:
    """Build an AIMessage with tool calls: `tool_call_message(("lookup_order", {...}))`."""
    return AIMessage(
        content=content,
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{i}_{name}", "type": "tool_call"}
            for i, (name, args) in enumerate(calls)
        ],
    )


def done_message(text: str = "Done.") -> AIMessage:
    return AIMessage(content=text)


ResponseFactory = Callable[[Any], Any]
