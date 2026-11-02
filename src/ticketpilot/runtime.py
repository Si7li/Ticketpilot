"""Wire settings → DB, retriever, backend, tools, checkpointer, graph. Shared by API/CLI/evals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ticketpilot.config import Settings
from ticketpilot.db import Database
from ticketpilot.graph.build import build_graph, sqlite_checkpointer
from ticketpilot.graph.nodes import Deps
from ticketpilot.llm.backend import LLMBackend
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.retrieval.vector import Embedder
from ticketpilot.seed import seed_database
from ticketpilot.telemetry import configure_logging
from ticketpilot.tools.registry import build_registry


@dataclass
class Runtime:
    settings: Settings
    deps: Deps
    graph: CompiledStateGraph[Any, Any, Any, Any]

    @property
    def db(self) -> Database:
        return self.deps.db


def make_backend(settings: Settings) -> LLMBackend:
    from ticketpilot.llm.anthropic_backend import AnthropicBackend

    return AnthropicBackend(settings)


def ensure_business_db(settings: Settings) -> Database:
    path = settings.business_db_path
    if not path.exists():
        seed_database(path, settings.REFERENCE_DATE)
    return Database(path)


def build_runtime(
    settings: Settings,
    *,
    backend: LLMBackend | None = None,
    retriever: HybridRetriever | None = None,
    embedder: Embedder | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    db_path: Path | None = None,
) -> Runtime:
    configure_logging(settings)
    db = Database(db_path) if db_path else ensure_business_db(settings)
    retr = retriever or HybridRetriever.from_settings(settings, embedder=embedder)
    deps = Deps(
        backend=backend or make_backend(settings),
        retriever=retr,
        registry=build_registry(),
        db=db,
        settings=settings,
    )
    cp = checkpointer or sqlite_checkpointer(settings.checkpoint_db_path)
    return Runtime(settings=settings, deps=deps, graph=build_graph(deps, cp))
