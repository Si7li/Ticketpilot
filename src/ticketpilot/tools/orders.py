"""Order lookup tools. Orders are only visible to the customer who placed them."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ticketpilot.safety.policy_rules import days_since, return_window_days
from ticketpilot.schemas import Customer, Order, is_valid_order_id
from ticketpilot.tools.registry import ToolArgs, ToolContext, ToolOutcome, ToolSpec

NOT_FOUND = "order not found for this customer"


def order_view(order: Order, customer: Customer, ctx: ToolContext) -> dict[str, Any]:
    ref = ctx.settings.REFERENCE_DATE
    return {
        "order_id": order.id,
        "status": order.status,
        "items": [i.model_dump() for i in order.items],
        "amount": order.amount,
        "is_sale": order.is_sale,
        "has_final_sale_items": order.has_final_sale,
        "order_date": order.order_date,
        "ship_date": order.ship_date,
        "delivery_date": order.delivery_date,
        "tracking_last_update": order.tracking_last_update,
        "days_since_tracking_update": days_since(order.tracking_last_update, ref),
        "days_since_delivery": days_since(order.delivery_date, ref),
        "days_since_order": days_since(order.order_date, ref),
        "shipping_method": order.shipping_method,
        "customer_country": customer.country,
        "customer_loyalty_tier": customer.loyalty_tier,
        "return_window_days": return_window_days(customer, ctx.settings),
        "active_refund": (r.model_dump() if (r := ctx.db.get_active_refund(order.id)) else None),
    }


def find_customer_order(
    order_id: str, customer_email: str, ctx: ToolContext
) -> tuple[Order, Customer] | None:
    customer = ctx.db.get_customer_by_email(customer_email)
    if customer is None:
        return None
    order = ctx.db.get_order(order_id)
    if order is None or order.customer_id != customer.id:
        return None  # never reveal that someone else's order exists
    return order, customer


class LookupOrderArgs(ToolArgs):
    order_id: str = Field(description="Order ID exactly as written, e.g. KO-10482")


def lookup_order(args: LookupOrderArgs, ctx: ToolContext, *, customer_email: str) -> ToolOutcome:
    oid = args.order_id.strip().upper()
    if not is_valid_order_id(oid):
        return ToolOutcome(ok=False, error="invalid order ID format; expected KO- and 5 digits")
    found = find_customer_order(oid, customer_email, ctx)
    if found is None:
        return ToolOutcome(ok=False, error=NOT_FOUND)
    order, customer = found
    return ToolOutcome(ok=True, data=order_view(order, customer, ctx))


class GetCustomerOrdersArgs(ToolArgs):
    pass


def get_customer_orders(
    args: GetCustomerOrdersArgs, ctx: ToolContext, *, customer_email: str
) -> ToolOutcome:
    customer = ctx.db.get_customer_by_email(customer_email)
    if customer is None:
        return ToolOutcome(ok=False, error="no account found for the sender's email")
    orders = ctx.db.get_recent_orders(customer.id, limit=5)
    return ToolOutcome(ok=True, data={"orders": [order_view(o, customer, ctx) for o in orders]})


LOOKUP_ORDER = ToolSpec(
    name="lookup_order",
    description=(
        "Look up one order placed by the customer who sent this ticket. Returns status, items "
        "(with sale/final-sale flags), amount, dates, days since delivery, the customer's "
        "return window, and any existing refund. Orders of other customers are reported as "
        "not found."
    ),
    args_model=LookupOrderArgs,
    fn=lookup_order,
    context_params=("customer_email",),
)

GET_CUSTOMER_ORDERS = ToolSpec(
    name="get_customer_orders",
    description=(
        "List up to 5 most recent orders of the customer who sent this ticket. Use when the "
        "customer did not give an order ID."
    ),
    args_model=GetCustomerOrdersArgs,
    fn=get_customer_orders,
    context_params=("customer_email",),
)
