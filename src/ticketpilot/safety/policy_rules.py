"""Safety-critical policy rules as pure functions (PROJECT_SPEC.md §10.3).

Reused by the refund tool, the validator, and routing. No I/O here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from ticketpilot.config import Settings
from ticketpilot.schemas import Customer, Order

RefundType = Literal[
    "return", "damaged", "wrong_item", "lost_package", "duplicate_charge", "price_adjustment"
]
PRICE_ADJUSTMENT_DAYS = 14


@dataclass(frozen=True)
class RuleResult:
    ok: bool
    rule: str
    detail: str = ""


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value[:10])


def days_since(value: str | None, reference: date) -> int | None:
    d = parse_date(value)
    return None if d is None else (reference - d).days


def is_domestic(customer: Customer, settings: Settings) -> bool:
    return customer.country.upper() == settings.HOME_COUNTRY.upper()


def return_window_days(customer: Customer, settings: Settings) -> int:
    if not is_domestic(customer, settings):
        return settings.INTERNATIONAL_RETURN_DAYS
    if customer.loyalty_tier == "peak":
        return settings.PEAK_RETURN_DAYS
    return settings.DOMESTIC_RETURN_DAYS


def return_window_ok(
    order: Order, customer: Customer, reference: date, settings: Settings
) -> RuleResult:
    days = days_since(order.delivery_date, reference)
    if days is None:
        return RuleResult(False, "return_requires_delivery", "order has not been delivered")
    limit = return_window_days(customer, settings)
    if is_domestic(customer, settings):
        rule = (
            "peak_return_window" if limit == settings.PEAK_RETURN_DAYS else "domestic_return_window"
        )
    else:
        rule = "international_return_window"
    if days > limit:
        return RuleResult(False, rule, f"delivered {days} days ago; limit is {limit} days")
    return RuleResult(True, rule, f"delivered {days} days ago; limit is {limit} days")


def damaged_window_ok(order: Order, reference: date, settings: Settings) -> RuleResult:
    days = days_since(order.delivery_date, reference)
    if days is None:
        return RuleResult(False, "damaged_requires_delivery", "order has not been delivered")
    limit = settings.DAMAGED_REPORT_DAYS
    if days > limit:
        return RuleResult(
            False, "damaged_report_window", f"delivered {days} days ago; limit {limit}"
        )
    return RuleResult(True, "damaged_report_window", f"delivered {days} days ago; limit {limit}")


def wrong_item_window_ok(order: Order, reference: date, settings: Settings) -> RuleResult:
    days = days_since(order.delivery_date, reference)
    if days is None:
        return RuleResult(False, "wrong_item_requires_delivery", "order has not been delivered")
    limit = settings.WRONG_ITEM_REPORT_DAYS
    ok = days <= limit
    return RuleResult(ok, "wrong_item_report_window", f"delivered {days} days ago; limit {limit}")


def lost_package_ok(order: Order) -> RuleResult:
    if order.status == "lost":
        return RuleResult(True, "lost_package_investigation", "carrier investigation concluded")
    return RuleResult(
        False,
        "lost_package_investigation",
        "a refund for a missing package is possible only after the carrier investigation "
        "(order status must be 'lost')",
    )


def price_adjustment_ok(order: Order, reference: date) -> RuleResult:
    days = days_since(order.order_date, reference)
    if days is None or days > PRICE_ADJUSTMENT_DAYS:
        return RuleResult(
            False, "price_adjustment_window", f"ordered {days} days ago; limit 14 days"
        )
    return RuleResult(True, "price_adjustment_window", f"ordered {days} days ago")


def refund_window_ok(
    refund_type: RefundType,
    order: Order,
    customer: Customer,
    reference: date,
    settings: Settings,
) -> RuleResult:
    match refund_type:
        case "return":
            return return_window_ok(order, customer, reference, settings)
        case "damaged":
            return damaged_window_ok(order, reference, settings)
        case "wrong_item":
            return wrong_item_window_ok(order, reference, settings)
        case "lost_package":
            return lost_package_ok(order)
        case "price_adjustment":
            return price_adjustment_ok(order, reference)
        case "duplicate_charge":
            return RuleResult(True, "duplicate_charge", "no window")
    raise ValueError(f"unknown refund_type {refund_type}")  # pragma: no cover


def refund_requires_approval(amount: float, settings: Settings) -> bool:
    return amount > settings.AUTO_APPROVE_REFUND_LIMIT


def is_mandatory_escalation_category(category: str | None, settings: Settings) -> bool:
    return category is not None and category in settings.sensitive_categories


# ---------------------------------------------------------------------------
# Reply-content rules used by the validator (Phase 7)
# ---------------------------------------------------------------------------

_CONFIRMED_RE = re.compile(
    r"\b(has been|have been|was|is now|been|i've|we've|i have|we have)\s+"
    r"(processed|issued|approved|confirmed|completed|refunded|sent)\b"
    r"|\brefund(ed)?\s+(is|has been)\s+(processed|issued|approved|confirmed|complete)\b"
    r"|\b(processed|issued|approved|confirmed)\s+(your|a|the)\s+refund\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")


def reply_claims_refund_confirmed(reply: str) -> bool:
    return bool(_CONFIRMED_RE.search(reply))


def money_amounts(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _MONEY_RE.findall(text)]


def reply_mentions_store_credit(reply: str) -> bool:
    return bool(re.search(r"store[\s-]credit", reply, re.IGNORECASE))


def reply_mentions_original_payment(reply: str) -> bool:
    return bool(
        re.search(
            r"original payment|back to your (card|credit card|paypal|payment method)|"
            r"to your (card|credit card)",
            reply,
            re.IGNORECASE,
        )
    )
