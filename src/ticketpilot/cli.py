"""Manual poking: `python -m ticketpilot.cli run-ticket --id T-0142` / `resume ...`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ticketpilot.config import PROJECT_ROOT, get_settings
from ticketpilot.graph.build import resume_ticket, run_ticket, total_ms
from ticketpilot.runtime import build_runtime

TICKET_FILES = [
    "data/eval/dev.jsonl",
    "data/eval/test.jsonl",
    "data/tickets/hard_cases.jsonl",
    "data/tickets/standard_tickets.jsonl",
    "data/tickets/smoke.jsonl",
]


def find_ticket(ticket_id: str) -> dict[str, Any] | None:
    for rel in TICKET_FILES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("ticket_id") == ticket_id:
                    return dict(row)
    return None


def _print_outcome(state: dict[str, Any], interrupted: bool) -> None:
    print(f"ticket:     {state.get('ticket_id')}")
    print(f"category:   {state.get('category')} (confidence {state.get('confidence')})")
    route = state.get("route") or ("human_review (paused)" if interrupted else None)
    print(f"route:      {route}")
    if state.get("injection_flag"):
        print(f"injection:  {state.get('injection_reasons')}")
    for r in state.get("tool_results", []):
        status = "ok" if r.get("ok") else f"ERROR {r.get('error')}"
        print(
            f"tool:       {r.get('tool')}({json.dumps(r.get('args'))}) → {status}"
            f"{' [pending]' if r.get('pending') else ''}"
        )
    v = state.get("validation", {})
    print(f"grounded:   {v.get('grounded')}  issues: {v.get('issues')}")
    if state.get("errors"):
        print(f"errors:     {state['errors']}")
    print(f"citations:  {state.get('citations')}")
    print(f"time:       {total_ms(state):.0f} ms  {state.get('timings_ms')}")
    print("\n--- draft reply ---\n" + (state.get("draft_reply") or "(none)"))
    if state.get("final_reply") and state.get("final_reply") != state.get("draft_reply"):
        print("\n--- sent reply ---\n" + str(state["final_reply"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ticketpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-ticket", help="run one ticket through the real graph")
    run.add_argument("--id", required=True, help="ticket ID (looked up in data/ files)")
    run.add_argument("--email")
    run.add_argument("--subject", default="")
    run.add_argument("--body")
    res = sub.add_parser("resume", help="resume a ticket paused at human review")
    res.add_argument("--id", required=True)
    res.add_argument("--action", choices=["approve", "edit_approve", "reject"], required=True)
    res.add_argument("--reviewer", default="cli")
    res.add_argument("--edited-reply")
    args = parser.parse_args(argv)

    rt = build_runtime(get_settings())
    if args.cmd == "run-ticket":
        if args.body:
            if not args.email:
                parser.error("--email is required with --body")
            ticket = {
                "ticket_id": args.id,
                "customer_email": args.email,
                "subject": args.subject,
                "body": args.body,
            }
        else:
            found = find_ticket(args.id)
            if found is None:
                print(f"Ticket {args.id} not found in {TICKET_FILES}", file=sys.stderr)
                return 1
            ticket = found
        if rt.db.get_ticket(ticket["ticket_id"]) is None:
            rt.db.create_ticket(
                ticket["ticket_id"], ticket["customer_email"], ticket.get("subject"), ticket["body"]
            )
        out = run_ticket(rt.graph, ticket)
        _print_outcome(out.state, out.interrupted)
        if out.interrupted:
            print(
                f"\nPaused for human review. Resume with:\n  python -m ticketpilot.cli "
                f"resume --id {ticket['ticket_id']} --action approve"
            )
        return 0

    decision = {"action": args.action, "reviewer": args.reviewer, "edited_reply": args.edited_reply}
    out = resume_ticket(rt.graph, args.id, decision)
    _print_outcome(out.state, out.interrupted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
