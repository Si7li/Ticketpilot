"""The LLMBackend protocol. Graph nodes only ever talk to this interface."""

from __future__ import annotations

from typing import Any, Protocol

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


class LLMBackend(Protocol):
    usage: UsageTracker  # accumulates input/output tokens per call site

    def classify(self, text: str) -> Classification: ...

    def extract(self, text: str, category: str) -> Entities: ...

    def act_step(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage: ...

    def draft(self, context: DraftContext) -> DraftReply: ...

    def check_claims(self, reply: str, evidence: list[Evidence]) -> ValidationResult: ...


class BackendError(RuntimeError):
    """Raised when the backend cannot produce a valid response."""
