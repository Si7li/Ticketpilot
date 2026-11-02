"""Builders for graph tests (scripted FakeLLMBackend runs)."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from ticketpilot.config import Settings
from ticketpilot.db import Database
from ticketpilot.llm.fake_backend import FakeLLMBackend, done_message, tool_call_message
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.runtime import Runtime, build_runtime
from ticketpilot.schemas import (
    ClaimCheck,
    Classification,
    DraftContext,
    DraftReply,
    Entities,
    ValidationResult,
)


def classification(category: str = "order_status", confidence: float = 0.95) -> Classification:
    return Classification(category=category, confidence=confidence, reasoning="test")  # type: ignore[arg-type]


def cite_first(reply: str, action: str = "reply") -> Any:
    def make(ctx: DraftContext) -> DraftReply:
        cites = [ctx.retrieved[0]["doc_id"]] if ctx.retrieved else []
        return DraftReply(reply=reply, citations=cites, proposed_action=action)  # type: ignore[arg-type]

    return make


def all_supported(payload: dict[str, Any]) -> ValidationResult:
    return ValidationResult(
        claims=[ClaimCheck(claim=payload["reply"][:60], supported=True, source="tool:lookup_order")]
    )


def script(
    *,
    category: str = "order_status",
    confidence: float = 0.95,
    order_ids: list[str] | None = None,
    act: list[Any] | None = None,
    reply: str = "Your order is on its way.",
    action: str = "reply",
    claims: Any = all_supported,
) -> FakeLLMBackend:
    return FakeLLMBackend(
        {
            "classify": [classification(category, confidence)],
            "extract": [Entities(order_ids=order_ids or [])],
            "act_step": act if act is not None else [done_message()],
            "draft": [cite_first(reply, action)],
            "check_claims": [claims],
        }
    )


def runtime(
    backend: FakeLLMBackend, settings: Settings, db: Database, retriever: HybridRetriever
) -> Runtime:
    return build_runtime(
        settings,
        backend=backend,
        retriever=retriever,
        checkpointer=InMemorySaver(),
        db_path=db.path,
    )


def ticket(ticket_id: str, email: str, body: str, subject: str = "Help") -> dict[str, Any]:
    return {"ticket_id": ticket_id, "customer_email": email, "subject": subject, "body": body}


__all__ = ["done_message", "runtime", "script", "ticket", "tool_call_message"]
