from pathlib import Path

import pytest

from ticketpilot.config import PROJECT_ROOT
from ticketpilot.retrieval.chunking import (
    MAX_TOKENS,
    Chunk,
    approx_tokens,
    chunk_document,
    chunk_set_hash,
    load_kb,
    parse_document,
)

DOC = """---
doc_id: policy/test-doc
title: Test Policy
category_hint: returns_refunds
---
# Test Policy

## Short one
Tiny.

## Long section
{long}

## Tail
Also tiny.
"""


def test_parse_front_matter() -> None:
    d = parse_document(DOC.format(long="word " * 100))
    assert d.doc_id == "policy/test-doc"
    assert d.title == "Test Policy"
    assert d.category_hint == "returns_refunds"
    assert not d.body.startswith("---")


def test_missing_front_matter_uses_fallback_and_h1() -> None:
    d = parse_document("# Hello\n\n## A\ntext", fallback_doc_id="faq/hello")
    assert d.doc_id == "faq/hello" and d.title == "Hello"
    with pytest.raises(ValueError):
        parse_document("no id here")


def test_small_sections_merge_and_prefix() -> None:
    chunks = chunk_document(parse_document(DOC.format(long="word " * 100)))
    # "Short one" (< 80 tokens) merges into the long section; "Tail" merges backwards.
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "policy/test-doc#0"
    assert c.text.startswith("Test Policy — Short one / Long section")
    assert "Also tiny." in c.text


def test_oversized_sections_split_on_paragraphs() -> None:
    para = "alpha " * 120
    long = "\n\n".join([para] * 6)  # ~936 tokens
    chunks = chunk_document(parse_document(DOC.format(long=long)))
    assert len(chunks) >= 2
    for c in chunks:
        body = c.text.split("\n", 1)[1]
        assert approx_tokens(body) <= MAX_TOKENS + approx_tokens(para) + 20
    assert [c.chunk_id for c in chunks] == [f"policy/test-doc#{i}" for i in range(len(chunks))]


def test_kb_meets_spec() -> None:
    docs = load_kb(PROJECT_ROOT / "data" / "kb")
    assert 60 <= len(docs) <= 80
    ids = {d.doc_id for d in docs}
    required = {
        "policy/returns-domestic", "policy/returns-international", "policy/restocking-fees",
        "policy/final-sale", "policy/sale-items", "policy/damaged-items", "policy/wrong-item",
        "policy/lost-packages", "policy/refund-methods", "policy/refund-approval",
        "policy/order-changes", "policy/shipping-times", "policy/free-shipping",
        "policy/warranty", "policy/price-adjustment", "policy/double-charges",
        "policy/loyalty-program", "policy/escalation-guidelines",
        "policy/privacy-and-account-security",
    }  # fmt: skip
    assert required <= ids
    for d in docs:  # doc_id must match the file path
        path = PROJECT_ROOT / "data" / "kb" / f"{d.doc_id}.md"
        assert path.exists(), d.doc_id


def test_hash_changes_with_content(kb_chunks: list[Chunk], tmp_path: Path) -> None:
    h1 = chunk_set_hash(kb_chunks)
    altered = [*kb_chunks[:-1], kb_chunks[-1].model_copy(update={"text": "changed"})]
    assert chunk_set_hash(altered) != h1
    assert chunk_set_hash(list(kb_chunks)) == h1
