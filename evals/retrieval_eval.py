"""Retrieval evaluation: BM25 vs vector vs hybrid (PROJECT_SPEC.md §8.6).

    python -m evals.retrieval_eval                       # dev split (default)
    python -m evals.retrieval_eval --split test --i-am-reporting

No LLM calls. For each labeled ticket with gold docs, two query variants:
  text      = subject + body
  text+cat  = subject + body + gold category name (mimics the agent's category-aware query)
Metrics per mode: recall@5 (any gold doc in top 5), all-gold recall@5, MRR (first gold doc,
over the top 20 retrieved documents).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.common import (
    label_warnings,
    new_report_dir,
    print_warnings,
    publish_latest,
    run_config,
    warnings_md,
    write_json,
)
from evals.metrics import rate, retrieval_scores
from ticketpilot.config import get_settings
from ticketpilot.labels import EVAL_DIR, load_split
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.retrieval.vector import Embedder

MODES = ("bm25", "vector", "hybrid")
VARIANTS = ("text", "text+cat")
K = 5
MRR_DEPTH = 20


def query_for(t: dict[str, Any], variant: str) -> str:
    q = f"{t.get('subject') or ''}\n{t['body']}"
    if variant == "text+cat":
        q += "\n" + t["gold_category"].replace("_", " ")
    return q


def evaluate(tickets: list[dict[str, Any]], retriever: HybridRetriever) -> dict[str, Any]:
    labeled = [t for t in tickets if t.get("gold_doc_ids")]
    per_ticket: list[dict[str, Any]] = []
    for t in labeled:
        row: dict[str, Any] = {
            "ticket_id": t["ticket_id"],
            "gold": t["gold_doc_ids"],
            "difficulty": t["difficulty"],
            "tags": t.get("tags", []),
        }
        for variant in VARIANTS:
            q = query_for(t, variant)
            for mode in MODES:
                hits = retriever.search(q, k=MRR_DEPTH, mode=mode)  # type: ignore[arg-type]
                docs = [h.doc_id for h in hits]
                row[f"{mode}|{variant}"] = {
                    **retrieval_scores(docs, t["gold_doc_ids"], K),
                    "top5": list(dict.fromkeys(docs))[:K],
                }
        per_ticket.append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for variant in VARIANTS:
            for mode in MODES:
                key = f"{mode}|{variant}"
                xs = [r[key] for r in rows]
                out[key] = {
                    "recall_at_5": rate(x["recall_any"] for x in xs).as_dict(),
                    "all_gold_recall_at_5": rate(x["recall_all"] for x in xs).as_dict(),
                    "mrr": (sum(x["rr"] for x in xs) / len(xs)) if xs else None,
                }
        return out

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_ticket:
        by_difficulty[r["difficulty"]].append(r)
        for tag in r["tags"]:
            by_tag[tag].append(r)
    return {
        "n_tickets": len(per_ticket),
        "n_skipped_no_gold_docs": len(tickets) - len(labeled),
        "k": K,
        "mrr_depth": MRR_DEPTH,
        "overall": summarize(per_ticket),
        "by_difficulty": {d: summarize(rs) for d, rs in sorted(by_difficulty.items())},
        "by_tag": {t: {"n": len(rs), **summarize(rs)} for t, rs in sorted(by_tag.items())},
        "per_ticket": per_ticket,
    }


def disagreements(per_ticket: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:
    """Tickets where modes disagree most (spread of reciprocal rank, text+cat variant)."""
    scored = []
    for r in per_ticket:
        rrs = {m: r[f"{m}|text+cat"]["rr"] for m in MODES}
        scored.append((max(rrs.values()) - min(rrs.values()), r, rrs))
    scored.sort(key=lambda x: (-x[0], x[1]["ticket_id"]))
    return [
        {
            "ticket_id": r["ticket_id"],
            "spread": s,
            "rr": rrs,
            "gold": r["gold"],
            "top5": {m: r[f"{m}|text+cat"]["top5"] for m in MODES},
        }
        for s, r, rrs in scored[:n]
        if s > 0
    ]


def summary_markdown(
    split: str, m: dict[str, Any], warnings: list[str], cfg: dict[str, Any]
) -> str:
    lines = [f"# Retrieval evaluation — {split} split\n", warnings_md(warnings)]
    lines.append(
        f"{m['n_tickets']} tickets with gold docs ({m['n_skipped_no_gold_docs']} skipped "
        f"without gold docs). Embedding model `{cfg['settings']['EMBEDDING_MODEL']}`, "
        f"RRF k={cfg['settings']['RRF_K']}, commit `{cfg['git_commit']}`.\n"
    )
    for variant in VARIANTS:
        lines.append(f"\n## Query = {variant}\n")
        lines.append("| mode | recall@5 | all-gold recall@5 | MRR |")
        lines.append("|---|---|---|---|")
        for mode in MODES:
            x = m["overall"][f"{mode}|{variant}"]
            mrr = f"{x['mrr']:.3f}" if x["mrr"] is not None else "n/a"
            lines.append(
                f"| {mode} | {x['recall_at_5']['display']} | "
                f"{x['all_gold_recall_at_5']['display']} | {mrr} |"
            )
    lines.append("\n## By difficulty (query = text+cat, recall@5)\n")
    lines.append("| difficulty | " + " | ".join(MODES) + " |")
    lines.append("|---|" + "---|" * len(MODES))
    for d, s in m["by_difficulty"].items():
        lines.append(
            f"| {d} | "
            + " | ".join(s[f"{mo}|text+cat"]["recall_at_5"]["display"] for mo in MODES)
            + " |"
        )
    lines.append("\n## By tag (query = text+cat, recall@5)\n")
    lines.append("| tag | n | " + " | ".join(MODES) + " |")
    lines.append("|---|---|" + "---|" * len(MODES))
    for t, s in m["by_tag"].items():
        cells = " | ".join(s[f"{mo}|text+cat"]["recall_at_5"]["display"] for mo in MODES)
        lines.append(f"| {t} | {s['n']} | {cells} |")
    lines.append("\n## Tickets where modes disagree most\n")
    lines.append("| ticket | gold | RR bm25 / vector / hybrid | bm25 top-5 | vector top-5 |")
    lines.append("|---|---|---|---|---|")
    for d in m["disagreements"]:
        rr = " / ".join(f"{d['rr'][mo]:.2f}" for mo in MODES)
        lines.append(
            f"| {d['ticket_id']} | {', '.join(d['gold'])} | {rr} | "
            f"{', '.join(d['top5']['bm25'])} | {', '.join(d['top5']['vector'])} |"
        )
    lines.append(
        "\n_Rates: count/total (%) with 95% Wilson intervals. Overlapping intervals mean "
        "the difference may be noise._\n"
    )
    return "\n".join(lines)


def run(
    split: str,
    eval_dir: Path = EVAL_DIR,
    embedder: Embedder | None = None,
    publish: bool = True,
) -> Path:
    settings = get_settings()
    tickets = load_split(split, eval_dir)
    warnings = label_warnings(tickets, split, eval_dir)
    print_warnings(warnings)
    retriever = HybridRetriever.from_settings(settings, embedder=embedder)
    metrics = evaluate(tickets, retriever)
    metrics["disagreements"] = disagreements(metrics["per_ticket"])
    metrics["warnings"] = warnings
    dry_run = eval_dir.resolve() != EVAL_DIR.resolve()
    if dry_run:
        warnings.append(f"DRY RUN on non-standard eval dir {eval_dir}; not a reportable result.")
        publish = False
    out = new_report_dir(f"retrieval-{split}" + ("-dryrun" if dry_run else ""))
    cfg = run_config(settings, split=split, kind="retrieval")
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", cfg)
    (out / "summary.md").write_text(summary_markdown(split, metrics, warnings, cfg))
    if publish:
        publish_latest(out / "summary.md")
    print((out / "summary.md").read_text())
    print(f"\nReport: {out}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument(
        "--i-am-reporting",
        action="store_true",
        help="required for --split test (never tune on test)",
    )
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = p.parse_args()
    if args.split == "test" and not args.i_am_reporting:
        p.error("--split test is for reporting only; pass --i-am-reporting")
    run(args.split, args.eval_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
