"""BM25 keyword index (PROJECT_SPEC.md §8.2)."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ticketpilot.retrieval.chunking import Chunk

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "can",
        "do",
        "does",
        "did",
        "am",
        "been",
        "being",
        "would",
        "could",
        "should",
        "just",
        "please",
        "hi",
        "hello",
        "thanks",
        "thank",
    ]
)

# A token is: a run of letters/digits, optionally joined by hyphens (keeps "ko-10482",
# "re-proofing"), optionally followed by a degree sign (keeps "20°").
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*°?")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("º", "°").replace("˚", "°")
    text = re.sub(r"(\d)\s*(?:degrees?|deg)\b", r"\1°", text)
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        out.append(tok)
        # Also index the parts of hyphenated words that are not IDs, so "re-proofing" matches
        # "proofing". IDs like "ko-10482" stay whole only.
        if "-" in tok and not re.fullmatch(r"[a-z]{1,3}-\d+", tok):
            out.extend(p for p in tok.split("-") if p and p not in STOPWORDS)
        # "20°" also yields "20" so "20 degree"/"20F" style queries still overlap.
        if tok.endswith("°"):
            out.append(tok[:-1])
    return out


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(zip(self.chunk_ids, scores, strict=True), key=lambda x: (-x[1], x[0]))
        return [(cid, float(s)) for cid, s in ranked[:k] if s > 0]
