"""Stratified dev/test split (PROJECT_SPEC.md §7.4).

    python scripts/split_eval.py              # first run: assign; later: regenerate files
    python scripts/split_eval.py --reshuffle  # deliberately re-assign membership

Membership is frozen in data/eval/split_ids.json; dev.jsonl/test.jsonl are rebuilt from
data/tickets/*.jsonl, and data/eval/test.sha256 is written.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ticketpilot.labels import (
    DEV_SIZE,
    EVAL_DIR,
    SOURCE_FILES,
    TICKETS_DIR,
    kb_doc_ids,
    load_jsonl,
    validate_label,
    write_split,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reshuffle", action="store_true")
    p.add_argument("--dev-size", type=int, default=DEV_SIZE)
    p.add_argument("--tickets-dir", type=Path, default=TICKETS_DIR)
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = p.parse_args()

    missing = [f for f in SOURCE_FILES if not (args.tickets_dir / f).exists()]
    if missing:
        print(
            f"Missing label files in {args.tickets_dir}: {missing}. Run `make tickets` first.",
            file=sys.stderr,
        )
        return 1
    docs = kb_doc_ids()
    bad = 0
    for f in SOURCE_FILES:
        for r in load_jsonl(args.tickets_dir / f):
            for prob in validate_label(r, docs):
                bad += 1
                print(f"  label problem {r.get('ticket_id')} ({f}): {prob}", file=sys.stderr)
    if bad:
        print(f"{bad} label problems; fix them before splitting.", file=sys.stderr)
        return 1
    write_split(args.tickets_dir, args.eval_dir, args.reshuffle, args.dev_size)
    for name in ("dev", "test"):
        rows = load_jsonl(args.eval_dir / f"{name}.jsonl")
        print(
            f"{name}: {len(rows)} tickets, verified {sum(bool(r.get('verified')) for r in rows)}"
            f"/{len(rows)}; difficulty {dict(Counter(r['difficulty'] for r in rows))}; "
            f"injection {sum(bool(r.get('is_injection')) for r in rows)}"
        )
        print(f"  categories: {dict(sorted(Counter(r['gold_category'] for r in rows).items()))}")
    print(f"test.sha256 written to {args.eval_dir / 'test.sha256'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
