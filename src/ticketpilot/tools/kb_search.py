"""`search_kb`: hybrid search over the knowledge base."""

from __future__ import annotations

from pydantic import Field

from ticketpilot.tools.registry import ToolArgs, ToolContext, ToolOutcome, ToolSpec

MAX_TOP_K = 8


class SearchKBArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, description="Number of results (capped at 8)")


def search_kb(args: SearchKBArgs, ctx: ToolContext) -> ToolOutcome:
    if ctx.retriever is None:
        return ToolOutcome(ok=False, error="knowledge base is unavailable")
    k = min(args.top_k, MAX_TOP_K)
    hits = ctx.retriever.search(args.query, k=k, mode=ctx.settings.RETRIEVAL_MODE)
    return ToolOutcome(
        ok=True,
        data=[{"doc_id": h.doc_id, "section": h.section, "text": h.text} for h in hits],
    )


SEARCH_KB = ToolSpec(
    name="search_kb",
    description="Search Kestrel Outfitters policies, FAQs and product guides.",
    args_model=SearchKBArgs,
    fn=search_kb,
)
