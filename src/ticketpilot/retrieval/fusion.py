"""Reciprocal rank fusion (PROJECT_SPEC.md §8.4)."""

from __future__ import annotations


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Ranks are 1-based. Returns (id, score) sorted desc.

    Ties are broken by the best (lowest) rank achieved in any list, then by id, so the
    output is deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if item in seen:  # an id counts once per list (its best rank)
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))
