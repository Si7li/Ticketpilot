"""Telemetry: per-node timers, token usage keyed by node and model, cost, JSON logging.

Usage attribution works through a ContextVar "scope": a node opens `usage_scope()`, and any
LLM call made in that thread records into the scope as well as into the backend's global
`UsageTracker`. That keeps per-ticket accounting correct when the eval harness runs
several tickets concurrently on one shared backend.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ticketpilot.config import Settings

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    call_site: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class UsageScope:
    records: list[UsageRecord] = field(default_factory=list)

    def by_model(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            m = out.setdefault(r.model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["calls"] += 1
        return out


_active_scope: ContextVar[UsageScope | None] = ContextVar("ticketpilot_usage_scope", default=None)


@contextmanager
def usage_scope() -> Iterator[UsageScope]:
    scope = UsageScope()
    token = _active_scope.set(scope)
    try:
        yield scope
    finally:
        _active_scope.reset(token)


class UsageTracker:
    """Accumulates input/output tokens per call site (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[UsageRecord] = []

    def record(self, call_site: str, model: str, input_tokens: int, output_tokens: int) -> None:
        rec = UsageRecord(call_site, model, int(input_tokens), int(output_tokens))
        with self._lock:
            self.records.append(rec)
        scope = _active_scope.get()
        if scope is not None:
            scope.records.append(rec)

    def totals(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._lock:
            for r in self.records:
                key = f"{r.call_site}:{r.model}"
                m = out.setdefault(key, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
                m["input_tokens"] += r.input_tokens
                m["output_tokens"] += r.output_tokens
                m["calls"] += 1
        return out


def merge_usage(
    a: dict[str, dict[str, dict[str, int]]], node: str, scope: UsageScope
) -> dict[str, dict[str, dict[str, int]]]:
    """Return a copy of state usage (`{node: {model: {...}}}`) with this node's scope added."""
    out = {k: {m: dict(v) for m, v in models.items()} for k, models in a.items()}
    if scope.records:
        node_usage = out.setdefault(node, {})
        for model, vals in scope.by_model().items():
            cur = node_usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            for k2, v2 in vals.items():
                cur[k2] += v2
    return out


def usage_by_model(usage: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    """Collapse `{node: {model: tokens}}` into `{model: tokens}`."""
    out: dict[str, dict[str, int]] = {}
    for models in usage.values():
        for model, vals in models.items():
            cur = out.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            for k, v in vals.items():
                cur[k] = cur.get(k, 0) + v
    return out


def cost_usd(model_tokens: dict[str, dict[str, int]], settings: Settings) -> float | None:
    """Dollar cost, or None if any involved model has no configured price (never guess)."""
    prices = {
        settings.AGENT_MODEL: (
            settings.AGENT_PRICE_INPUT_PER_MTOK,
            settings.AGENT_PRICE_OUTPUT_PER_MTOK,
        ),
        settings.JUDGE_MODEL: (
            settings.JUDGE_PRICE_INPUT_PER_MTOK,
            settings.JUDGE_PRICE_OUTPUT_PER_MTOK,
        ),
    }
    total = 0.0
    for model, t in model_tokens.items():
        p_in, p_out = prices.get(model, (None, None))
        if p_in is None or p_out is None:
            return None
        total += t.get("input_tokens", 0) / 1e6 * p_in + t.get("output_tokens", 0) / 1e6 * p_out
    return round(total, 6)


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------


@contextmanager
def node_timer() -> Iterator[dict[str, float]]:
    """`with node_timer() as t: ...` then read `t["ms"]`."""
    box: dict[str, float] = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = round((time.perf_counter() - start) * 1000, 2)


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(settings: Settings) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    if settings.LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("ticketpilot")
    root.handlers[:] = [handler]
    root.setLevel(settings.LOG_LEVEL.upper())
    root.propagate = False
    _configured = True


node_logger = logging.getLogger("ticketpilot.node")


def log_node(ticket_id: str, node: str, duration_ms: float, outcome: str, **extra: Any) -> None:
    node_logger.info(
        "node finished",
        extra={
            "ticket_id": ticket_id,
            "node": node,
            "duration_ms": duration_ms,
            "outcome": outcome,
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# Optional Langfuse
# ---------------------------------------------------------------------------

_langfuse_logged = False


def langfuse_handler(settings: Settings, **trace_meta: Any) -> Any | None:
    """Return a Langfuse LangChain callback handler if keys are configured, else None.

    Absent keys → one info log for the whole process, no warnings, no errors.
    """
    global _langfuse_logged
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        if not _langfuse_logged:
            logging.getLogger("ticketpilot.telemetry").info("Langfuse not configured; skipping")
            _langfuse_logged = True
        return None
    try:
        import os

        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
        os.environ.setdefault(
            "LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY.get_secret_value()
        )
        if settings.LANGFUSE_HOST:
            os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_HOST)
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:  # pragma: no cover - depends on optional package
        if not _langfuse_logged:
            logging.getLogger("ticketpilot.telemetry").info("Langfuse unavailable: %s", exc)
            _langfuse_logged = True
        return None
