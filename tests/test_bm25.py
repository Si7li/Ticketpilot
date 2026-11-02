from ticketpilot.retrieval.bm25 import BM25Index, tokenize
from ticketpilot.retrieval.chunking import Chunk


def test_tokenizer_keeps_ids_and_models() -> None:
    toks = tokenize("Order KO-10482: my Cirrus 20° bag & Ridgeline-2 tent!")
    assert "ko-10482" in toks
    assert "20°" in toks and "20" in toks
    assert "cirrus" in toks and "ridgeline-2" in toks and "ridgeline" in toks
    assert "my" not in toks and "&" not in toks


def test_degree_words_normalised() -> None:
    assert "20°" in tokenize("rated 20 degrees")
    assert "0°" in tokenize("Cirrus 0º")  # ordinal indicator lookalike


def test_hyphenated_words_also_split() -> None:
    toks = tokenize("re-proofing")
    assert {"re-proofing", "proofing"} <= set(toks)
    assert tokenize("ko-10482") == ["ko-10482"]  # IDs are not split


def _chunks() -> list[Chunk]:
    texts = {
        "a#0": "Cirrus 20° sleeping bag comfort rating 30F",
        "b#0": "Cirrus 0° sleeping bag comfort rating 10F",
        "c#0": "Ridgeline 2 tent two person",
    }
    return [
        Chunk(chunk_id=k, doc_id=k[:1], title="t", section="s", text=v) for k, v in texts.items()
    ]


def test_search_ranks_exact_model() -> None:
    idx = BM25Index(_chunks())
    assert idx.search("Cirrus 0° bag", 3)[0][0] == "b#0"
    assert idx.search("cirrus 20 degree", 3)[0][0] == "a#0"
    assert idx.search("ridgeline tent", 1) == [("c#0", idx.search("ridgeline tent", 1)[0][1])]


def test_empty_query_and_no_match() -> None:
    idx = BM25Index(_chunks())
    assert idx.search("the and of", 3) == []
    assert idx.search("kayak", 3) == []
