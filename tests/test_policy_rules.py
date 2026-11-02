from datetime import date

import pytest

from ticketpilot.config import Settings
from ticketpilot.safety import policy_rules as pr
from ticketpilot.schemas import Customer, Order, OrderItem

REF = date(2026, 10, 1)


def _order(delivered: str | None, status: str = "delivered", ordered: str = "2026-09-01") -> Order:
    return Order(id="KO-10000", customer_id="C-0001",
                 items=[OrderItem(sku="X", name="x", unit_price=10, category="tent")],
                 amount=10, is_sale=False, status=status, order_date=ordered,  # type: ignore[arg-type]
                 delivery_date=delivered, shipping_method="standard")  # fmt: skip


def _cust(country: str = "US", tier: str = "trail") -> Customer:
    return Customer(id="C-0001", email="a@b.c", name="A", loyalty_tier=tier, country=country)  # type: ignore[arg-type]


@pytest.fixture
def s() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("delivered", "country", "tier", "ok", "rule"),
    [
        ("2026-09-01", "US", "trail", True, "domestic_return_window"),  # 30 days: boundary
        ("2026-08-31", "US", "trail", False, "domestic_return_window"),  # 31 days
        ("2026-09-17", "GB", "trail", True, "international_return_window"),  # 14 days
        ("2026-09-16", "GB", "peak", False, "international_return_window"),  # 15, peak no help
        ("2026-08-02", "US", "peak", True, "peak_return_window"),  # 60 days
        ("2026-08-01", "US", "peak", False, "peak_return_window"),  # 61 days
        (None, "US", "trail", False, "return_requires_delivery"),
    ],
)
def test_return_window(s: Settings, delivered: str | None, country: str, tier: str, ok: bool,
                       rule: str) -> None:  # fmt: skip
    res = pr.return_window_ok(_order(delivered), _cust(country, tier), REF, s)
    assert (res.ok, res.rule) == (ok, rule)


@pytest.mark.parametrize(("delivered", "ok"), [("2026-09-17", True), ("2026-09-16", False)])
def test_damaged_and_wrong_item_windows(s: Settings, delivered: str, ok: bool) -> None:
    assert pr.damaged_window_ok(_order(delivered), REF, s).ok is ok
    assert pr.wrong_item_window_ok(_order(delivered), REF, s).ok is ok
    assert not pr.damaged_window_ok(_order(None), REF, s).ok
    assert not pr.wrong_item_window_ok(_order(None), REF, s).ok


def test_refund_window_dispatch(s: Settings) -> None:
    o, c = _order("2026-09-20"), _cust()
    for t in ("return", "damaged", "wrong_item", "duplicate_charge"):
        assert pr.refund_window_ok(t, o, c, REF, s).ok  # type: ignore[arg-type]
    assert not pr.refund_window_ok("lost_package", o, c, REF, s).ok
    assert pr.refund_window_ok("lost_package", _order(None, "lost"), c, REF, s).ok
    assert pr.refund_window_ok("price_adjustment", _order(None, ordered="2026-09-17"), c, REF, s).ok
    assert not pr.refund_window_ok(
        "price_adjustment", _order(None, ordered="2026-09-16"), c, REF, s
    ).ok


@pytest.mark.parametrize(("amount", "needs"), [(99.99, False), (100.0, False), (100.01, True)])
def test_refund_requires_approval(s: Settings, amount: float, needs: bool) -> None:
    assert pr.refund_requires_approval(amount, s) is needs


def test_mandatory_escalation_category(s: Settings) -> None:
    assert pr.is_mandatory_escalation_category("other_sensitive", s)
    assert not pr.is_mandatory_escalation_category("returns_refunds", s)
    assert not pr.is_mandatory_escalation_category(None, s)


def test_days_since() -> None:
    assert pr.days_since("2026-09-26", REF) == 5
    assert pr.days_since("2026-09-26T10:00:00", REF) == 5
    assert pr.days_since(None, REF) is None


@pytest.mark.parametrize(
    ("text", "confirmed"),
    [
        ("Your refund has been processed.", True),
        ("We have issued a refund of $189.", True),
        ("I've approved your refund.", True),
        ("Your refund is being reviewed by our team.", False),
        ("Once approved, you'll get an email.", False),
    ],
)
def test_reply_claims_refund_confirmed(text: str, confirmed: bool) -> None:
    assert pr.reply_claims_refund_confirmed(text) is confirmed


def test_money_and_method_mentions() -> None:
    assert pr.money_amounts("Refund of $189.00 and $1,250.50 and $64") == [189.0, 1250.5, 64.0]
    assert pr.reply_mentions_store_credit("as store credit")
    assert pr.reply_mentions_original_payment("back to your original payment method")
    assert not pr.reply_mentions_original_payment("as store credit")
