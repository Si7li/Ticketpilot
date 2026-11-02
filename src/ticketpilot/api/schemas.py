"""API request/response models."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TICKET_ID_RE = re.compile(r"^T-\d{4}$")


class TicketIn(BaseModel):
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=20_000)
    customer_email: str = Field(min_length=3, max_length=254)
    ticket_id: str | None = None

    @field_validator("customer_email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("invalid email address")
        return v

    @field_validator("ticket_id")
    @classmethod
    def _tid(cls, v: str | None) -> str | None:
        if v is not None and not TICKET_ID_RE.match(v):
            raise ValueError("ticket_id must look like T-0142")
        return v


class ToolSummary(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None
    pending: bool = False
    needs_specialist: bool = False
    data: Any = None


class TicketResult(BaseModel):
    ticket_id: str
    status: str
    route: str | None
    category: str | None = None
    confidence: float | None = None
    injection_flag: bool = False
    draft_reply: str | None = None
    final_reply: str | None = None
    citations: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    tool_results: list[ToolSummary] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_ms: float = 0.0
    usage: dict[str, Any] = Field(default_factory=dict)


class TicketDetail(BaseModel):
    ticket_id: str
    customer_email: str
    subject: str | None
    body: str
    status: str
    created_at: str
    updated_at: str
    result: TicketResult | None
    graph_next: list[str] = Field(default_factory=list)
    awaiting_review: bool = False


class ReviewIn(BaseModel):
    action: Literal["approve", "edit_approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=100)
    edited_reply: str | None = Field(default=None, max_length=10_000)
    duration_seconds: float | None = Field(default=None, ge=0)


class QueueItem(BaseModel):
    ticket_id: str
    customer_email: str
    subject: str | None
    created_at: str
    category: str | None
    confidence: float | None
    reasons: list[str]
    payload: dict[str, Any]


class Health(BaseModel):
    ok: bool
    db: bool
    qdrant: bool
    kb_indexed: bool
    model_configured: bool
    agent_model: str
    retrieval_mode: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None
