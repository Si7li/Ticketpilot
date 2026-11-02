"""Terminal label review — HUMAN CHECKPOINT (PROJECT_SPEC.md §7.3).

    make review-labels                # walk through unverified tickets
    python scripts/review_labels.py --file hard_cases.jsonl
    python scripts/review_labels.py --ids T-0151,T-0152

Keys:  a = accept (sets verified=true + verified_at)   e = edit a field
       s = skip   q = quit
Saves after every ticket and regenerates data/eval/{dev,test}.jsonl (+ test.sha256) if a
split already exists. Only this tool may set `verified: true`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

from ticketpilot.config import get_settings
from ticketpilot.db import Database
from ticketpilot.labels import (
    EVAL_DIR,
    LABEL_FIELDS,
    SOURCE_FILES,
    TICKETS_DIR,
    kb_doc_ids,
    load_sources,
    validate_label,
    write_jsonl,
    write_split,
)
from ticketpilot.safety.policy_rules import days_since, return_window_days
from ticketpilot.schemas import ORDER_ID_SEARCH_RE

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)


def parse_value(field: str, raw: str) -> Any:
    """Parse an edited value: JSON if it parses, else a plain string; lists from commas."""
    raw = raw.strip()
    if raw in ("", "null", "none", "None"):
        return None if field not in ("gold_doc_ids", "forbidden_actions", "tags") else []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    if field in ("gold_doc_ids", "forbidden_actions", "tags"):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if field in ("should_escalate", "is_injection"):
        return raw.lower() in ("y", "yes", "true", "1")
    if field == "gold_refund_amount":
        return float(raw)
    return raw


def order_context(db: Database, row: dict[str, Any]) -> list[str]:
    s = get_settings()
    lines: list[str] = []
    cust = db.get_customer_by_email(row["customer_email"])
    if cust is None:
        return [f"{DIM}(sender is not a known customer){RESET}"]
    lines.append(
        f"customer {cust.id} {cust.name} · tier {cust.loyalty_tier} · {cust.country} · "
        f"return window {return_window_days(cust, s)}d"
    )
    ids = [m.upper() for m in ORDER_ID_SEARCH_RE.findall(row["body"])]
    orders = [o for i in ids if (o := db.get_order(i))] or db.get_recent_orders(cust.id, 5)
    for o in orders:
        owner = "" if o.customer_id == cust.id else f" {RED}(belongs to {o.customer_id}!){RESET}"
        items = ", ".join(
            f"{i.name} ${i.unit_price:.2f}{' SALE' if i.is_sale else ''}"
            f"{' FINAL' if i.final_sale else ''} [{i.category}]"
            for i in o.items
        )
        refund = db.get_active_refund(o.id)
        lines.append(
            f"{o.id}{owner}: {o.status}, ${o.amount:.2f}, delivered {o.delivery_date} "
            f"({days_since(o.delivery_date, s.REFERENCE_DATE)}d), tracking "
            f"{days_since(o.tracking_last_update, s.REFERENCE_DATE)}d ago, {o.shipping_method}"
        )
        lines.append(f"    items: {items}")
        if refund:
            lines.append(
                f"    {YELLOW}existing refund {refund.id}: ${refund.amount} "
                f"{refund.method} {refund.status}{RESET}"
            )
    return lines


def show(row: dict[str, Any], db: Database, docs: set[str], progress: str) -> None:
    print("\n" + "═" * 80)
    print(
        f"{BOLD}{row['ticket_id']}{RESET}  {progress}  source={row.get('source')}  "
        f"difficulty={row.get('difficulty')}  tags={row.get('tags')}"
    )
    print(f"From: {row['customer_email']}\nSubject: {row.get('subject')}\n")
    print(row["body"])
    if row.get("source_text"):
        print(f"\n{DIM}Bitext original ({row.get('source_intent')}): {row['source_text']}{RESET}")
    print(f"\n{BOLD}Seeded data{RESET}")
    for line in order_context(db, row):
        print("  " + line)
    print(f"\n{BOLD}Proposed labels{RESET}")
    for f in LABEL_FIELDS:
        print(f"  {f:20s} {json.dumps(row.get(f), ensure_ascii=False)}")
    problems = validate_label(row, docs)
    for p in problems:
        print(f"  {RED}! {p}{RESET}")


def count(sources: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
    rows = [r for rs in sources.values() for r in rs]
    return sum(bool(r.get("verified")) for r in rows), len(rows)


def save(sources: dict[str, list[dict[str, Any]]], name: str) -> None:
    write_jsonl(TICKETS_DIR / name, sources[name])
    if (EVAL_DIR / "split_ids.json").exists():
        write_split()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--file", choices=SOURCE_FILES, help="review only this file")
    p.add_argument("--ids", help="comma-separated ticket IDs (also re-reviews verified ones)")
    args = p.parse_args()

    sources = load_sources()
    if not any(sources.values()):
        print("No label files found; run `make tickets` first.", file=sys.stderr)
        return 1
    db = Database(get_settings().business_db_path)
    docs = kb_doc_ids()
    wanted = {i.strip() for i in args.ids.split(",")} if args.ids else None
    queue = [
        (name, idx)
        for name, rows in sources.items()
        if not args.file or name == args.file
        for idx, r in enumerate(rows)
        if (wanted and r["ticket_id"] in wanted) or (not wanted and not r.get("verified"))
    ]
    if not queue:
        v, n = count(sources)
        print(f"Nothing to review. verified {v}/{n}")
        return 0

    for name, idx in queue:
        row = sources[name][idx]
        while True:
            v, n = count(sources)
            show(row, db, docs, f"verified {v}/{n}")
            choice = input(f"\n[{GREEN}a{RESET}]ccept  [e]dit  [s]kip  [q]uit > ").strip().lower()
            if choice == "a":
                problems = validate_label(row, docs)
                if problems:
                    print(f"{RED}Fix these first: {problems}{RESET}")
                    continue
                row["verified"] = True
                row["verified_at"] = datetime.now(UTC).isoformat(timespec="seconds")
                save(sources, name)
                break
            if choice == "e":
                field = input(f"field ({', '.join(LABEL_FIELDS)}, subject, body) > ").strip()
                if field not in (*LABEL_FIELDS, "subject", "body"):
                    print(f"{RED}unknown field{RESET}")
                    continue
                raw = input(f"new value for {field} (JSON or plain; lists comma-separated) > ")
                try:
                    row[field] = parse_value(field, raw)
                except ValueError as exc:
                    print(f"{RED}{exc}{RESET}")
                    continue
                if field in ("subject", "body"):
                    row["verified"] = False
                save(sources, name)
                continue
            if choice == "s":
                break
            if choice == "q":
                v, n = count(sources)
                print(f"Saved. verified {v}/{n}")
                return 0
    v, n = count(sources)
    print(f"\nDone. verified {v}/{n}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; progress up to the last accepted ticket is saved.")
        raise SystemExit(130) from None
