"""Graph construction, checkpointers, and run/resume helpers (PROJECT_SPEC.md §11.4)."""

from __future__ import annotations

import itertools
import sqlite3
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ticketpilot.config import Settings
from ticketpilot.graph import nodes as N
from ticketpilot.graph.routing import choose_route
from ticketpilot.graph.state import TicketState

Graph = CompiledStateGraph[Any, Any, Any, Any]

LINEAR = ["normalize", "classify", "extract_entities", "retrieve", "act", "draft", "validate"]
TERMINAL = ["auto_send", "human_review", "escalate"]


def build_graph(deps: N.Deps, checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    g: StateGraph[Any, Any, Any, Any] = StateGraph(TicketState)
    for name, fn in N.NODES.items():
        g.add_node(name, N._node(name, fn, deps))  # type: ignore[call-overload]
    g.add_edge(START, LINEAR[0])
    for a, b in itertools.pairwise(LINEAR):
        g.add_edge(a, b)
    g.add_conditional_edges(
        "validate",
        partial(_route, settings=deps.settings),
        {t: t for t in TERMINAL},
    )
    for t in TERMINAL:
        g.add_edge(t, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def _route(state: TicketState, settings: Settings) -> str:
    return choose_route(state, settings)


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


# ---------------------------------------------------------------------------
# run / resume
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    state: dict[str, Any]
    interrupted: bool
    interrupt_payload: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def route(self) -> str | None:
        r = self.state.get("route")
        if r:
            return str(r)
        return "human_review" if self.interrupted else None


def thread_config(ticket_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": ticket_id}}


def initial_state(ticket: dict[str, Any]) -> TicketState:
    return {
        "ticket_id": ticket["ticket_id"],
        "customer_email": ticket["customer_email"],
        "subject": ticket.get("subject") or "",
        "raw_text": ticket["body"],
        "tool_results": [],
        "errors": [],
        "timings_ms": {},
        "usage": {},
    }


def _outcome(graph: Graph, ticket_id: str, result: Any) -> RunOutcome:
    config = thread_config(ticket_id)
    snapshot = graph.get_state(config)
    values = dict(snapshot.values)
    interrupts = list(result.get("__interrupt__", [])) if isinstance(result, dict) else []
    if not interrupts:
        for task in snapshot.tasks:
            interrupts.extend(task.interrupts)
    if interrupts:
        payload = interrupts[0].value
        values.setdefault("route", None)
        return RunOutcome(
            values, True, payload if isinstance(payload, dict) else {"value": payload}
        )
    return RunOutcome(values, False)


def run_ticket(graph: Graph, ticket: dict[str, Any]) -> RunOutcome:
    result = graph.invoke(initial_state(ticket), thread_config(ticket["ticket_id"]))
    return _outcome(graph, ticket["ticket_id"], result)


def resume_ticket(graph: Graph, ticket_id: str, decision: dict[str, Any]) -> RunOutcome:
    result = graph.invoke(Command(resume=decision), thread_config(ticket_id))
    return _outcome(graph, ticket_id, result)


def is_awaiting_review(graph: Graph, ticket_id: str) -> bool:
    snapshot = graph.get_state(thread_config(ticket_id))
    return "human_review" in (snapshot.next or ())


def total_ms(state: dict[str, Any]) -> float:
    timings: dict[str, float] = state.get("timings_ms") or {}
    return round(float(sum(timings.values())), 2)
