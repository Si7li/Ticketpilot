"""Shared fixtures. No test may touch the network (enforced below)."""

from __future__ import annotations

import shutil
import socket
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ticketpilot.config import PROJECT_ROOT, Settings, get_settings
from ticketpilot.db import Database
from ticketpilot.retrieval.chunking import Chunk, chunk_kb
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.retrieval.vector import HashingEmbedder, VectorIndex, make_qdrant_client
from ticketpilot.seed import seed_database
from ticketpilot.tools.registry import ToolContext, ToolRegistry, build_registry

REFERENCE_DATE = date(2026, 10, 1)

# ---------------------------------------------------------------------------
# Network guard
# ---------------------------------------------------------------------------

_real_connect = socket.socket.connect
_LOCAL = {"127.0.0.1", "::1", "localhost"}


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and (host in _LOCAL or host.startswith("/")):
        return _real_connect(self, address)
    if self.family == socket.AF_UNIX:
        return _real_connect(self, address)
    raise RuntimeError(f"Network access blocked in tests: attempted connect to {address!r}")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's real .env / ANTHROPIC_API_KEY out of tests."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Settings / DB
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "BUSINESS_DB_PATH": str(tmp_path / "business.db"),
        "CHECKPOINT_DB_PATH": str(tmp_path / "checkpoints.db"),
        "REFERENCE_DATE": REFERENCE_DATE,
        "QDRANT_URL": "",
        "LOG_JSON": False,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture(scope="session")
def seeded_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("seed") / "template.db"
    seed_database(path, REFERENCE_DATE)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def db(seeded_template: Path, settings: Settings) -> Database:
    shutil.copy(seeded_template, settings.business_db_path)
    return Database(settings.business_db_path)


# ---------------------------------------------------------------------------
# Retrieval (hashing embedder: deterministic, no model download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def kb_chunks() -> list[Chunk]:
    return chunk_kb(PROJECT_ROOT / "data" / "kb")


@pytest.fixture(scope="session")
def retriever(kb_chunks: list[Chunk]) -> HybridRetriever:
    vindex = VectorIndex(make_qdrant_client(""), "test_kb", HashingEmbedder(dim=256))
    vindex.index(kb_chunks)
    return HybridRetriever(kb_chunks, vindex, rrf_k=60)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return build_registry()


@pytest.fixture
def ctx_for(db: Database, settings: Settings, retriever: HybridRetriever) -> Any:
    def _make(email: str, ticket_id: str = "T-9999") -> ToolContext:
        return ToolContext(
            customer_email=email, ticket_id=ticket_id, db=db, settings=settings, retriever=retriever
        )

    return _make


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{name}"}
