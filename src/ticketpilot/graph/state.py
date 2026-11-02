"""Graph state (PROJECT_SPEC.md §11.1)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TicketState(TypedDict, total=False):
    ticket_id: str
    customer_email: str
    subject: str
    raw_text: str
    clean_text: str
    injection_flag: bool
    injection_reasons: list[str]
    category: str
    confidence: float
    classification_reasoning: str
    entities: dict[str, Any]
    retrieved: list[dict[str, Any]]  # chunks
    messages: Annotated[list[BaseMessage], add_messages]  # act loop transcript
    tool_results: list[dict[str, Any]]
    draft_reply: str
    citations: list[str]
    proposed_action: str
    draft_issues: list[str]  # e.g. citations dropped because they were not retrieved
    validation: dict[str, Any]  # {"enabled", "grounded", "claims", "rule_issues", "issues"}
    route: str  # auto_send | human_review | escalate
    review_decision: dict[str, Any] | None
    final_reply: str | None
    timings_ms: dict[str, float]
    usage: dict[str, dict[str, dict[str, int]]]  # {node: {model: {input_tokens, ...}}}
    errors: list[str]
