"""`escalate`: hand the ticket to a specialist team."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ticketpilot.tools.registry import ToolArgs, ToolContext, ToolOutcome, ToolSpec


class EscalateArgs(ToolArgs):
    team: Literal["billing", "legal", "logistics", "trust_safety", "general"]
    reason: str = Field(min_length=3, max_length=300)


def escalate(args: EscalateArgs, ctx: ToolContext, *, ticket_id: str) -> ToolOutcome:
    record = ctx.db.create_escalation(ticket_id, args.team, args.reason)
    return ToolOutcome(ok=True, data=record, needs_specialist=True)


ESCALATE = ToolSpec(
    name="escalate",
    description=(
        "Hand this ticket to a specialist team: billing (chargebacks, billing disputes), legal "
        "(legal threats), logistics (complex shipping cases), trust_safety (injury, safety, "
        "harassment, account takeover), general (press, anything no policy covers)."
    ),
    args_model=EscalateArgs,
    fn=escalate,
    context_params=("ticket_id",),
)
