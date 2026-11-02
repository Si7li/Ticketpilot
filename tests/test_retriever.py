import pytest

from ticketpilot.retrieval.chunking import Chunk
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.retrieval.vector import HashingEmbedder, VectorIndex, make_qdrant_client


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_modes_return_k_unique_results(retriever: HybridRetriever, mode: str) -> None:
    res = retriever.search("return window for international orders", k=5, mode=mode)  # type: ignore[arg-type]
    assert 0 < len(res) <= 5
    assert len({r.chunk_id for r in res}) == len(res)
    for r in res:
        assert set(r.ranks) == {"bm25", "vector"}


def test_hybrid_records_both_ranks(retriever: HybridRetriever) -> None:
    res = retriever.search("Ridgeline 3 tent weight", k=5, mode="hybrid")
    assert res[0].doc_id == "product/ridgeline-3"
    assert any(r.ranks["bm25"] is not None and r.ranks["vector"] is not None for r in res)


def test_bm25_nails_exact_model_names(retriever: HybridRetriever) -> None:
    assert (
        retriever.search("Cirrus 0° limit rating", k=1, mode="bm25")[0].doc_id == "product/cirrus-0"
    )
    assert (
        retriever.search("Cirrus 20° limit rating", k=1, mode="bm25")[0].doc_id
        == "product/cirrus-20"
    )


def test_invalid_mode_and_empty_query(retriever: HybridRetriever) -> None:
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")  # type: ignore[arg-type]
    assert retriever.search("   ") == []


def test_vector_index_is_idempotent(kb_chunks: list[Chunk]) -> None:
    vi = VectorIndex(make_qdrant_client(""), "idem", HashingEmbedder(64))
    assert not vi.is_indexed()
    assert vi.index(kb_chunks) is True
    assert vi.is_indexed(kb_chunks)
    assert vi.index(kb_chunks) is False  # unchanged chunk set → no rebuild
    changed = [*kb_chunks[:-1], kb_chunks[-1].model_copy(update={"text": "new text"})]
    assert vi.index(changed) is True
    assert vi.count() == len(changed)


def test_retriever_is_indexed(retriever: HybridRetriever) -> None:
    assert retriever.is_indexed()
