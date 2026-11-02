"""HybridRetriever facade with a bm25 | vector | hybrid mode switch (PROJECT_SPEC.md §8.5)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ticketpilot.config import RetrievalMode, Settings
from ticketpilot.retrieval.bm25 import BM25Index
from ticketpilot.retrieval.chunking import Chunk, chunk_kb
from ticketpilot.retrieval.fusion import rrf
from ticketpilot.retrieval.vector import (
    Embedder,
    FastEmbedEmbedder,
    VectorIndex,
    make_qdrant_client,
)

HYBRID_CANDIDATES = 20


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    section: str
    text: str
    score: float
    ranks: dict[str, int | None]

    def as_state(self) -> dict[str, Any]:
        return self.model_dump()


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        vector_index: VectorIndex,
        rrf_k: int = 60,
        default_mode: RetrievalMode = "hybrid",
    ) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.vector = vector_index
        self.rrf_k = rrf_k
        self.default_mode: RetrievalMode = default_mode

    @classmethod
    def from_settings(
        cls, settings: Settings, embedder: Embedder | None = None, ensure_index: bool = True
    ) -> HybridRetriever:
        chunks = chunk_kb(settings.kb_dir)
        emb = embedder or FastEmbedEmbedder(settings.EMBEDDING_MODEL)
        vindex = VectorIndex(
            make_qdrant_client(settings.QDRANT_URL), settings.QDRANT_COLLECTION, emb
        )
        if ensure_index:
            vindex.index(chunks)
        return cls(chunks, vindex, rrf_k=settings.RRF_K, default_mode=settings.RETRIEVAL_MODE)

    def is_indexed(self) -> bool:
        return self.vector.is_indexed(list(self.chunks.values()))

    def search(
        self, query: str, k: int = 5, mode: RetrievalMode | None = None
    ) -> list[RetrievedChunk]:
        mode = mode or self.default_mode
        if mode not in ("bm25", "vector", "hybrid"):
            raise ValueError(f"unknown retrieval mode: {mode}")
        if not query.strip():
            return []
        n = HYBRID_CANDIDATES if mode == "hybrid" else k
        bm25_hits = self.bm25.search(query, n) if mode in ("bm25", "hybrid") else []
        vec_hits = self.vector.search(query, n) if mode in ("vector", "hybrid") else []
        bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_hits, start=1)}
        vec_rank = {cid: i for i, (cid, _) in enumerate(vec_hits, start=1)}

        if mode == "bm25":
            scored = bm25_hits
        elif mode == "vector":
            scored = vec_hits
        else:
            scored = rrf([[c for c, _ in bm25_hits], [c for c, _ in vec_hits]], k=self.rrf_k)

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for cid, score in scored:
            if cid in seen or cid not in self.chunks:
                continue
            seen.add(cid)
            c = self.chunks[cid]
            results.append(
                RetrievedChunk(
                    doc_id=c.doc_id,
                    chunk_id=cid,
                    section=c.section,
                    text=c.text,
                    score=round(float(score), 6),
                    ranks={"bm25": bm25_rank.get(cid), "vector": vec_rank.get(cid)},
                )
            )
            if len(results) >= k:
                break
        return results
