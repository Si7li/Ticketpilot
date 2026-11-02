"""Tool specs + executor with context injection (PROJECT_SPEC.md §10.1).

Context params (customer_email, ticket_id) are injected by the executor from graph state
and are never accepted from the model: if the model supplies one, the call is rejected.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ticketpilot.config import Settings
from ticketpilot.db import Database

if TYPE_CHECKING:
    from ticketpilot.retrieval.retriever import HybridRetriever


class ToolArgs(BaseModel):
    """Base for model-facing tool arguments. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class ToolResult(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    data: Any = None
    error: str | None = None
    pending: bool = False
    needs_specialist: bool = False
    duration_ms: float = 0.0
    arg_error: bool = False  # invalid / rejected arguments (counts against tool-call validity)
    call_id: str | None = None

    def to_state(self) -> dict[str, Any]:
        return self.model_dump()


@dataclass
class ToolContext:
    customer_email: str
    ticket_id: str
    db: Database
    settings: Settings
    retriever: HybridRetriever | None = None


class ToolOutcome(BaseModel):
    """What a tool function returns; the executor wraps it into a ToolResult."""

    ok: bool
    data: Any = None
    error: str | None = None
    pending: bool = False
    needs_specialist: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    fn: Callable[..., ToolOutcome]
    context_params: tuple[str, ...] = field(default_factory=tuple)

    def anthropic_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self.specs = {s.name: s for s in specs}

    def schemas(self) -> list[dict[str, Any]]:
        return [s.anthropic_schema() for s in self.specs.values()]

    def execute(self, tool_call: dict[str, Any], context: ToolContext) -> ToolResult:
        name = str(tool_call.get("name", ""))
        raw_args = dict(tool_call.get("args") or {})
        call_id = tool_call.get("id")
        start = time.perf_counter()

        def done(**kw: Any) -> ToolResult:
            ms = round((time.perf_counter() - start) * 1000, 2)
            return ToolResult(tool=name, args=raw_args, duration_ms=ms, call_id=call_id, **kw)

        spec = self.specs.get(name)
        if spec is None:
            return done(ok=False, error=f"unknown tool '{name}'", arg_error=True)

        smuggled = sorted(set(raw_args) & set(spec.context_params))
        if smuggled:
            return done(
                ok=False,
                error=f"argument(s) {smuggled} are set by the system and cannot be supplied",
                arg_error=True,
            )
        try:
            args = spec.args_model.model_validate(raw_args)
        except ValidationError as exc:
            msgs = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}" for e in exc.errors()
            )
            return done(ok=False, error=f"invalid arguments: {msgs}", arg_error=True)

        injected = {p: getattr(context, p) for p in spec.context_params}
        try:
            outcome = spec.fn(args, context, **injected)
        except Exception as exc:  # tool bugs become structured errors, never crash the graph
            return done(ok=False, error=f"tool failed: {type(exc).__name__}: {exc}")
        return done(**outcome.model_dump())


def build_registry() -> ToolRegistry:
    from ticketpilot.tools.escalate import ESCALATE
    from ticketpilot.tools.kb_search import SEARCH_KB
    from ticketpilot.tools.orders import GET_CUSTOMER_ORDERS, LOOKUP_ORDER
    from ticketpilot.tools.refunds import ISSUE_REFUND

    return ToolRegistry([SEARCH_KB, LOOKUP_ORDER, GET_CUSTOMER_ORDERS, ISSUE_REFUND, ESCALATE])
