from typing import Any

import pytest

from tests.conftest import call
from ticketpilot.tools.registry import ToolRegistry


def refund(registry: ToolRegistry, ctx: Any, **args: Any) -> Any:
    args.setdefault("reason", "customer request")
    return registry.execute(call("issue_refund", **args), ctx)


def test_under_limit_auto_approves(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    r = refund(registry, ctx, order_id="KO-10571", amount=64.0, method="original_payment")
    assert r.ok and not r.pending
    assert r.data["status"] == "approved" and r.data["approved_by"] == "auto"


def test_exactly_limit_auto_approves(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("sam.ortiz@example.com")
    r = refund(
        registry,
        ctx,
        order_id="KO-10482",
        amount=100.0,
        method="original_payment",
        refund_type="damaged",
    )
    assert r.ok and not r.pending


def test_over_limit_becomes_pending(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("sam.ortiz@example.com")
    r = refund(
        registry,
        ctx,
        order_id="KO-10482",
        amount=189.0,
        method="original_payment",
        refund_type="damaged",
    )
    assert r.ok and r.pending
    assert r.data["status"] == "pending_approval" and r.data["approved_by"] is None


def test_refund_over_order_total_refused(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    r = refund(registry, ctx, order_id="KO-10571", amount=64.01, method="original_payment")
    assert not r.ok and "at most the order total" in (r.error or "")
    assert ctx.db.get_active_refund("KO-10571") is None


@pytest.mark.parametrize("amount", [0, -5])
def test_non_positive_amount_refused(registry: ToolRegistry, ctx_for: Any, amount: float) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    assert not refund(registry, ctx, order_id="KO-10571", amount=amount, method="store_credit").ok


def test_double_refund_is_idempotent(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    first = refund(registry, ctx, order_id="KO-10571", amount=64.0, method="original_payment")
    second = refund(registry, ctx, order_id="KO-10571", amount=30.0, method="store_credit")
    assert first.ok and second.ok
    assert second.data["id"] == first.data["id"]
    assert second.data["note"] == "refund already exists"
    with ctx.db.connect() as conn:
        n = conn.execute("SELECT count(*) FROM refunds WHERE order_id='KO-10571'").fetchone()[0]
    assert n == 1


def test_already_refunded_seed_order(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("lena.fischer@example.com")
    r = refund(registry, ctx, order_id="KO-10537", amount=229.0, method="original_payment")
    assert r.ok and r.data["id"] == "RF-SEED0001" and r.data["note"] == "refund already exists"


def test_sale_item_to_card_refused(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("priya.nair@example.com")
    r = refund(registry, ctx, order_id="KO-10511", amount=119.25, method="original_payment")
    assert not r.ok and r.error == "sale items are refunded as store credit only"
    ok = refund(registry, ctx, order_id="KO-10511", amount=119.25, method="store_credit")
    assert ok.ok and ok.pending  # 119.25 > 100


def test_sale_under_limit_store_credit_approved(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("mia.rossi@example.com")
    r = refund(registry, ctx, order_id="KO-10673", amount=74.25, method="store_credit")
    assert r.ok and not r.pending and r.data["method"] == "store_credit"


def test_final_sale_refused(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("marcus.webb@example.com")
    for method in ("original_payment", "store_credit"):
        r = refund(registry, ctx, order_id="KO-10523", amount=50.0, method=method)
        assert not r.ok and r.error == "final sale items are not refundable"


def test_other_customers_order_refused(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    r = refund(registry, ctx, order_id="KO-10482", amount=10.0, method="store_credit")
    assert not r.ok and r.error == "order not found for this customer"
    assert ctx.db.get_active_refund("KO-10482") is None


def test_model_supplied_email_rejected(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    r = refund(
        registry,
        ctx,
        order_id="KO-10482",
        amount=10.0,
        method="store_credit",
        customer_email="sam.ortiz@example.com",
    )
    assert not r.ok and r.arg_error
    assert ctx.db.get_active_refund("KO-10482") is None


def test_model_supplied_ticket_id_rejected(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    r = refund(
        registry, ctx, order_id="KO-10571", amount=10.0, method="store_credit", ticket_id="T-0001"
    )
    assert not r.ok and r.arg_error


def test_international_window(registry: ToolRegistry, ctx_for: Any) -> None:
    late = refund(
        registry,
        ctx_for("olivia.brown@example.co.uk"),
        order_id="KO-10548",
        amount=50.0,
        method="original_payment",
    )
    assert not late.ok and "international_return_window" in (late.error or "")
    ok = refund(
        registry,
        ctx_for("lucas.schmidt@example.de"),
        order_id="KO-10662",
        amount=50.0,
        method="original_payment",
    )
    assert ok.ok


def test_domestic_window_and_peak_extension(registry: ToolRegistry, ctx_for: Any) -> None:
    late = refund(
        registry,
        ctx_for("ben.carter@example.com"),
        order_id="KO-10605",
        amount=50.0,
        method="original_payment",
    )
    assert not late.ok and "domestic_return_window" in (late.error or "")
    peak = refund(
        registry,
        ctx_for("grace.liu@example.com"),
        order_id="KO-10594",
        amount=50.0,
        method="original_payment",
    )
    assert peak.ok  # 45 days, Peak tier has 60


def test_damaged_window(registry: ToolRegistry, ctx_for: Any) -> None:
    late = refund(
        registry,
        ctx_for("hannah.moore@example.com"),
        order_id="KO-10616",
        amount=50.0,
        method="original_payment",
        refund_type="damaged",
    )
    assert not late.ok and "damaged_report_window" in (late.error or "")


def test_not_delivered_cannot_be_returned(registry: ToolRegistry, ctx_for: Any) -> None:
    r = refund(
        registry,
        ctx_for("aisha.bello@example.com"),
        order_id="KO-10560",
        amount=50.0,
        method="original_payment",
    )
    assert not r.ok and "return_requires_delivery" in (r.error or "")


def test_lost_package_needs_lost_status(registry: ToolRegistry, ctx_for: Any) -> None:
    stuck = refund(
        registry,
        ctx_for("daniel.kim@example.com"),
        order_id="KO-10559",
        amount=50.0,
        method="original_payment",
        refund_type="lost_package",
    )
    assert not stuck.ok
    lost = refund(
        registry,
        ctx_for("chloe.martin@example.com"),
        order_id="KO-10651",
        amount=89.0,
        method="original_payment",
        refund_type="lost_package",
    )
    assert lost.ok and not lost.pending


def test_price_adjustment_store_credit_only(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("ethan.clark@example.com")
    card = refund(
        registry,
        ctx,
        order_id="KO-10706",
        amount=20.0,
        method="original_payment",
        refund_type="price_adjustment",
    )
    assert not card.ok
    credit = refund(
        registry,
        ctx,
        order_id="KO-10706",
        amount=20.0,
        method="store_credit",
        refund_type="price_adjustment",
    )
    assert credit.ok


def test_skus_scope(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for("tom.reyes@example.com")
    bad = refund(
        registry, ctx, order_id="KO-10571", amount=10.0, method="store_credit", skus=["NOPE"]
    )
    assert not bad.ok


def test_bad_method_is_arg_error(registry: ToolRegistry, ctx_for: Any) -> None:
    r = refund(
        registry,
        ctx_for("tom.reyes@example.com"),
        order_id="KO-10571",
        amount=10.0,
        method="bitcoin",
    )
    assert not r.ok and r.arg_error
