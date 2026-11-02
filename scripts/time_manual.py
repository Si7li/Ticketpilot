"""Time manual ticket handling — HUMAN CHECKPOINT (PROJECT_SPEC.md §15.8).

    python scripts/time_manual.py [--n 30]

Shows dev tickets one at a time. Handle each one the way a support agent would without
TicketPilot: find the policy in data/kb/, look up the order, and type the reply. Helpers:
    :order KO-12345     show an order (sqlite lookup)
    :orders <email>     list a customer's orders
    :kb <words>         grep the knowledge base
    .                   (a line with a single dot) finish the reply → time recorded
    :skip               skip this ticket (not recorded)      :quit   stop
Seconds per ticket go to data/eval/manual_timing.csv (appended; resumable).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import UTC, datetime

from ticketpilot.config import PROJECT_ROOT, get_settings
from ticketpilot.db import Database
from ticketpilot.labels import load_split

OUT = PROJECT_ROOT / "data" / "eval" / "manual_timing.csv"
KB = PROJECT_ROOT / "data" / "kb"


def done_ids() -> set[str]:
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {r["ticket_id"] for r in csv.DictReader(f)}


def record(ticket_id: str, seconds: float, reply_chars: int) -> None:
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticket_id", "seconds", "reply_chars", "recorded_at"])
        if new:
            w.writeheader()
        w.writerow(
            {
                "ticket_id": ticket_id,
                "seconds": round(seconds, 1),
                "reply_chars": reply_chars,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )


def helper(cmd: str, db: Database) -> None:
    if cmd.startswith(":order "):
        o = db.get_order(cmd.split(maxsplit=1)[1].strip().upper())
        print(o.model_dump_json(indent=1) if o else "not found")
    elif cmd.startswith(":orders "):
        c = db.get_customer_by_email(cmd.split(maxsplit=1)[1].strip())
        if not c:
            print("no such customer")
            return
        print(c.model_dump_json())
        for o in db.get_recent_orders(c.id, 10):
            print(
                f"  {o.id} {o.status} ${o.amount} delivered={o.delivery_date} items={[i.name for i in o.items]}"
            )
    elif cmd.startswith(":kb "):
        words = cmd.split(maxsplit=1)[1]
        subprocess.run(["grep", "-ril", words, str(KB)], check=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    db = Database(get_settings().business_db_path)
    tickets = [t for t in load_split("dev") if t["ticket_id"] not in done_ids()][: args.n]
    print(f"Knowledge base: {KB}\nReference date for windows: {get_settings().REFERENCE_DATE}\n")
    for i, t in enumerate(tickets, 1):
        print("=" * 78)
        print(
            f"[{i}/{len(tickets)}] {t['ticket_id']}  from {t['customer_email']}\nSubject: {t.get('subject')}\n\n{t['body']}\n"
        )
        print("Type the reply (helpers: :order, :orders, :kb). End with a line containing only '.'")
        start = time.monotonic()
        lines: list[str] = []
        while True:
            line = input("> ")
            if line.strip() == ".":
                record(t["ticket_id"], time.monotonic() - start, len("\n".join(lines)))
                break
            if line.strip() == ":skip":
                break
            if line.strip() == ":quit":
                return 0
            if line.startswith(":"):
                helper(line.strip(), db)
            else:
                lines.append(line)
    print(f"Done. Timings in {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nStopped; recorded tickets are saved.")
