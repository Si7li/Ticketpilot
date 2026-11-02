"""Pure helpers for the review UI (kept separate so they are unit-testable)."""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from typing import Any

_WORD = re.compile(r"[a-z0-9$.%°-]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _words(text: str) -> set[str]:
    return {w.strip(".") for w in _WORD.findall(text.lower()) if len(w.strip(".")) > 2}


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s.strip()]


def unsupported_sentences(
    draft: str, claims: list[dict[str, Any]], min_overlap: float = 0.5
) -> set[int]:
    """Indexes of draft sentences that best match each unsupported claim."""
    sentences = split_sentences(draft)
    flagged: set[int] = set()
    for c in claims:
        if c.get("supported"):
            continue
        cw = _words(c.get("claim", ""))
        if not cw:
            continue
        best, best_i = 0.0, -1
        for i, s in enumerate(sentences):
            overlap = len(cw & _words(s)) / len(cw)
            if overlap > best:
                best, best_i = overlap, i
        if best_i >= 0 and best >= min_overlap:
            flagged.add(best_i)
    return flagged


def highlight_unsupported(draft: str, claims: list[dict[str, Any]]) -> str:
    """HTML of the draft with unsupported-claim sentences wrapped in <mark>."""
    sentences = split_sentences(draft)
    flagged = unsupported_sentences(draft, claims)
    parts = []
    for i, s in enumerate(sentences):
        esc = html.escape(s)
        parts.append(f"<mark>{esc}</mark>" if i in flagged else esc)
    return " ".join(parts)


def age_label(created_at: str, now: datetime | None = None) -> str:
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return "?"
    now = now or datetime.now(UTC)
    secs = max(0, int((now - created).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    return f"{secs // 86400}d"
