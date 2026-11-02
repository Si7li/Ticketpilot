"""Build the BM25 + vector indexes over data/kb (`make index`).

BM25 is rebuilt in memory on every process start (it takes milliseconds); this script
chunks the KB, embeds it, and writes it to Qdrant (server if QDRANT_URL is set, otherwise
an in-memory instance, which is only useful as a smoke test).
"""

from __future__ import annotations

import time

from ticketpilot.config import get_settings
from ticketpilot.retrieval.chunking import chunk_kb, chunk_set_hash, load_kb
from ticketpilot.retrieval.vector import FastEmbedEmbedder, VectorIndex, make_qdrant_client


def main() -> None:
    s = get_settings()
    docs = load_kb(s.kb_dir)
    chunks = chunk_kb(s.kb_dir)
    print(f"{len(docs)} documents → {len(chunks)} chunks (hash {chunk_set_hash(chunks)[:12]})")
    t0 = time.perf_counter()
    embedder = FastEmbedEmbedder(s.EMBEDDING_MODEL)
    index = VectorIndex(make_qdrant_client(s.QDRANT_URL), s.QDRANT_COLLECTION, embedder)
    changed = index.index(chunks)
    where = s.QDRANT_URL or "in-memory Qdrant"
    status = "indexed" if changed else "already up to date"
    print(
        f"Vector index {status}: {index.count()} points in {where} "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    if not s.QDRANT_URL:
        print(
            "Note: QDRANT_URL is empty, so the index lives only in this process. "
            "The API and evals build their own in-memory index at startup."
        )


if __name__ == "__main__":
    main()
