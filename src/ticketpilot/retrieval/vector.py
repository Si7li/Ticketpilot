"""Dense vector index on Qdrant with fastembed embeddings (PROJECT_SPEC.md §8.3)."""

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ticketpilot.retrieval.chunking import Chunk, chunk_set_hash

_NAMESPACE = uuid.UUID("6f1c7d4e-2b5a-4c1e-9a7d-0e3b5f8c2a11")


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (no API cost)."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(next(iter(self._model.query_embed(text))).tolist())


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. Used in unit tests (no model download)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        from ticketpilot.retrieval.bm25 import tokenize

        v = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_qdrant_client(url: str) -> QdrantClient:
    return QdrantClient(url=url) if url else QdrantClient(":memory:")


class VectorIndex:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, chunk_id))

    def _stored_hash(self) -> str | None:
        if not self.client.collection_exists(self.collection):
            return None
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=True)
        if not points or not points[0].payload:
            return None
        value = points[0].payload.get("chunk_set_hash")
        return str(value) if value else None

    def is_indexed(self, chunks: list[Chunk] | None = None) -> bool:
        stored = self._stored_hash()
        if stored is None:
            return False
        return chunks is None or stored == chunk_set_hash(chunks)

    def index(self, chunks: list[Chunk]) -> bool:
        """Idempotent: recreates the collection only if the chunk set changed.

        The chunk-set hash is stored in every point's payload, so it lives with the data
        (works for both in-memory and server Qdrant). Returns True if (re)indexed.
        """
        digest = chunk_set_hash(chunks)
        if self._stored_hash() == digest and self.count() == len(chunks):
            return False
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=self.embedder.dim, distance=qm.Distance.COSINE),
        )
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        points = [
            qm.PointStruct(
                id=self.point_id(c.chunk_id),
                vector=vec,
                payload={**c.model_dump(), "chunk_set_hash": digest},
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i : i + 128], wait=True)
        return True

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = self.embedder.embed_query(query)
        res = self.client.query_points(self.collection, query=vec, limit=k, with_payload=True)
        out: list[tuple[str, float]] = []
        for p in res.points:
            payload: dict[str, Any] = p.payload or {}
            out.append((str(payload["chunk_id"]), float(p.score)))
        return out
