"""Split knowledge-base Markdown into section chunks (PROJECT_SPEC.md §8.1)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

MIN_TOKENS = 80
MAX_TOKENS = 500

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class Chunk(BaseModel):
    chunk_id: str  # "{doc_id}#{n}"
    doc_id: str
    title: str
    section: str
    text: str


class KBDocument(BaseModel):
    doc_id: str
    title: str
    category_hint: str | None = None
    body: str


def approx_tokens(text: str) -> float:
    return len(text.split()) * 1.3


def parse_document(raw: str, fallback_doc_id: str | None = None) -> KBDocument:
    meta: dict[str, str] = {}
    body = raw
    m = _FRONT_MATTER_RE.match(raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = raw[m.end() :]
    doc_id = str(meta.get("doc_id") or fallback_doc_id or "")
    if not doc_id:
        raise ValueError("document has no doc_id")
    title = str(meta.get("title") or "")
    if not title:
        h1 = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = h1.group(1).strip() if h1 else doc_id
    return KBDocument(
        doc_id=doc_id, title=title, category_hint=meta.get("category_hint"), body=body.strip()
    )


def _sections(body: str) -> list[tuple[str, str]]:
    """Split on `## ` headings. Text before the first `##` (minus the H1) is an intro section."""
    parts: list[tuple[str, str]] = []
    current_heading = "Overview"
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if "\n".join(buf).strip():
                parts.append((current_heading, "\n".join(buf).strip()))
            current_heading, buf = line[3:].strip(), []
        elif line.startswith("# ") and not parts and not "\n".join(buf).strip():
            continue  # H1 title, already captured as the document title
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        parts.append((current_heading, "\n".join(buf).strip()))
    return parts


def _merge_small(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge sections under MIN_TOKENS into the next section (the last one merges backwards)."""
    merged: list[tuple[str, str]] = []
    carry: tuple[str, str] | None = None
    for heading, text in sections:
        if carry is not None:
            heading = f"{carry[0]} / {heading}"
            text = f"{carry[1]}\n\n{text}"
            carry = None
        if approx_tokens(text) < MIN_TOKENS:
            carry = (heading, text)
        else:
            merged.append((heading, text))
    if carry is not None:
        if merged:
            h, t = merged.pop()
            merged.append((f"{h} / {carry[0]}", f"{t}\n\n{carry[1]}"))
        else:
            merged.append(carry)
    return merged


def _split_large(heading: str, text: str) -> list[tuple[str, str]]:
    if approx_tokens(text) <= MAX_TOKENS:
        return [(heading, text)]
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        candidate = "\n\n".join([*buf, para])
        if buf and approx_tokens(candidate) > MAX_TOKENS:
            out.append((heading, "\n\n".join(buf)))
            buf = [para]
        else:
            buf.append(para)
    if buf:
        out.append((heading, "\n\n".join(buf)))
    return out


def chunk_document(doc: KBDocument) -> list[Chunk]:
    pieces: list[tuple[str, str]] = []
    for heading, text in _merge_small(_sections(doc.body)):
        pieces.extend(_split_large(heading, text))
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{n}",
            doc_id=doc.doc_id,
            title=doc.title,
            section=heading,
            text=f"{doc.title} — {heading}\n{text}",
        )
        for n, (heading, text) in enumerate(pieces)
    ]


def load_kb(kb_dir: Path) -> list[KBDocument]:
    docs = []
    for path in sorted(kb_dir.rglob("*.md")):
        fallback = path.relative_to(kb_dir).with_suffix("").as_posix()
        docs.append(parse_document(path.read_text(encoding="utf-8"), fallback))
    return docs


def chunk_kb(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_kb(kb_dir):
        chunks.extend(chunk_document(doc))
    return chunks


def chunk_set_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(b"\0")
        h.update(c.text.encode())
        h.update(b"\0")
    return h.hexdigest()
