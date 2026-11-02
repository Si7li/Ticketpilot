"""`issue_refund` with guardrails (PROJECT_SPEC.md §10.3). Rules are applied in spec order."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ticketpilot.db import DuplicateRefundError
from ticketpilot.safety.policy_rules import (
    RefundType,
    refund_requires_approval,
    refund_window_ok,
)
from ticketpilot.schemas import is_valid_order_id
from ticketpilot.tools.orders import NOT_FOUND, find_customer_order
from ticketpilot.tools.registry import ToolArgs, ToolContext, ToolOutcome, ToolSpec


class IssueRefundArgs(ToolArgs):
    order_id: str = Field(description="Order ID, e.g. KO-10482")
    amount: float = Field(description="Refund amount in USD")
    reason: str = Field(min_length=3, max_length=300, description="Short reason for the refund")
    method: Literal["original_payment", "store_credit"] = Field(
        description="original_payment (5-7 business days) or store_credit (instant)"
    )
    refund_type: RefundType = Field(
        default="return",
        description=(
            "Which policy the refund falls under: return, damaged, wrong_item, lost_package, "
            "duplicate_charge, or price_adjustment"
        ),
    )
    skus: list[str] | None = Field(
        default=None,
        description="SKUs being refunded; omit to refund the whole order",
    )


def issue_refund(
    args: IssueRefundArgs, ctx: ToolContext, *, customer_email: str, ticket_id: str
) -> ToolOutcome:
    settings = ctx.settings
    oid = args.order_id.strip().upper()

    # 1. Order must exist and belong to the ticket's customer.
    if not is_valid_order_id(oid):
        return ToolOutcome(ok=False, error="invalid order ID format; expected KO- and 5 digits")
    found = find_customer_order(oid, customer_email, ctx)
    if found is None:
        return ToolOutcome(ok=False, error=NOT_FOUND)
    order, customer = found

    # 2. Idempotency: an existing non-rejected refund is returned as-is.
    existing = ctx.db.get_active_refund(oid)
    if existing is not None:
        return ToolOutcome(
            ok=True,
            data={**existing.model_dump(), "note": "refund already exists"},
            pending=existing.status == "pending_approval",
        )

    # 3. 0 < amount <= order amount.
    amount = round(args.amount, 2)
    if not (0 < amount <= order.amount):
        return ToolOutcome(
            ok=False,
            error=f"amount must be greater than 0 and at most the order total ({order.amount:.2f})",
        )

    # 4. Final-sale items are never refundable.
    scope = order.items
    if args.skus:
        wanted = {s.upper() for s in args.skus}
        scope = [i for i in order.items if i.sku.upper() in wanted]
        if not scope:
            return ToolOutcome(ok=False, error="none of the given SKUs are in this order")
    if any(i.final_sale for i in scope):
        return ToolOutcome(ok=False, error="final sale items are not refundable")

    # 5. Sale orders are refunded as store credit only. (A confirmed duplicate *charge* is a
    #    billing error, not a sale-item refund, so it goes back to the original payment.)
    if (
        order.is_sale
        and args.method == "original_payment"
        and args.refund_type != "duplicate_charge"
    ):
        return ToolOutcome(ok=False, error="sale items are refunded as store credit only")
    if args.refund_type == "price_adjustment" and args.method != "store_credit":
        return ToolOutcome(ok=False, error="price adjustments are issued as store credit only")

    # 6. Return / damaged / wrong-item / lost-package windows.
    window = refund_window_ok(args.refund_type, order, customer, settings.REFERENCE_DATE, settings)
    if not window.ok:
        return ToolOutcome(ok=False, error=f"outside policy ({window.rule}): {window.detail}")

    # 7. Auto-approve up to the limit; above it the refund waits for a human.
    pending = refund_requires_approval(amount, settings)
    try:
        refund = ctx.db.create_refund(
            order_id=oid,
            amount=amount,
            reason=f"[{args.refund_type}] {args.reason} (ticket {ticket_id})",
            method=args.method,
            status="pending_approval" if pending else "approved",
            approved_by=None if pending else "auto",
        )
    except DuplicateRefundError:  # concurrent insert; fall back to idempotent behaviour
        existing = ctx.db.get_active_refund(oid)
        assert existing is not None
        return ToolOutcome(
            ok=True,
            data={**existing.model_dump(), "note": "refund already exists"},
            pending=existing.status == "pending_approval",
        )
    return ToolOutcome(ok=True, data=refund.model_dump(), pending=pending)


ISSUE_REFUND = ToolSpec(
    name="issue_refund",
    description=(
        "Create a refund for an order of the customer who sent this ticket. The system enforces "
        "policy: ownership, one refund per order, amount <= order total, no final-sale items, "
        "sale items as store credit only, and return/damage windows. Refunds over the "
        "auto-approval limit are created as pending_approval and reviewed by a human."
    ),
    args_model=IssueRefundArgs,
    fn=issue_refund,
    context_params=("customer_email", "ticket_id"),
)
