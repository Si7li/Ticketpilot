from pathlib import Path

import pytest

from tests.conftest import REFERENCE_DATE
from ticketpilot.db import Database, DuplicateRefundError
from ticketpilot.seed import SCENARIOS, build_seed_data, seed_database


def test_seed_is_deterministic(tmp_path: Path) -> None:
    a = build_seed_data(REFERENCE_DATE)
    b = build_seed_data(REFERENCE_DATE)
    assert a == b
    seed_database(tmp_path / "a.db", REFERENCE_DATE)
    seed_database(tmp_path / "b.db", REFERENCE_DATE)
    ids = []
    for name in ("a.db", "b.db"):
        d = Database(tmp_path / name)
        with d.connect() as conn:
            ids.append([r[0] for r in conn.execute("SELECT id FROM orders ORDER BY id")])
    assert ids[0] == ids[1]
    assert len(ids[0]) == 400


def test_seed_sizes_and_variety() -> None:
    data = build_seed_data(REFERENCE_DATE)
    assert len(data["customers"]) == 120
    assert len(data["orders"]) == 400
    statuses = {o["status"] for o in data["orders"]}
    assert statuses == {"processing", "shipped", "delivered", "lost", "returned"}
    tiers = {c["loyalty_tier"] for c in data["customers"]}
    assert tiers == {"trail", "summit", "peak"}
    assert any(c["country"] != "US" for c in data["customers"])
    assert len(data["refunds"]) >= 3


def test_scenario_orders_exist(db: Database) -> None:
    for s in SCENARIOS:
        order = db.get_order(s.order_id)
        assert order is not None, s.key
        cust = db.get_customer_by_email(s.email)
        assert cust is not None and order.customer_id == cust.id


def test_key_scenarios_match_spec(db: Database) -> None:
    tent = db.get_order("KO-10482")
    assert tent is not None
    assert tent.amount == 189.0 and tent.status == "delivered"
    assert tent.delivery_date == "2026-09-26"  # 5 days before reference date
    assert db.get_order("KO-10560").status == "processing"  # type: ignore[union-attr]
    assert db.get_order("KO-10511").is_sale  # type: ignore[union-attr]
    assert db.get_order("KO-10523").has_final_sale  # type: ignore[union-attr]
    assert db.get_active_refund("KO-10537") is not None


def test_unique_active_refund_constraint(db: Database) -> None:
    r = db.create_refund("KO-10571", 64.0, "test", "original_payment", "approved")
    with pytest.raises(DuplicateRefundError):
        db.create_refund("KO-10571", 10.0, "again", "original_payment", "approved")
    db.set_refund_status(r.id, "rejected")
    # A rejected refund no longer blocks a new one.
    db.create_refund("KO-10571", 64.0, "retry", "store_credit", "approved")


def test_customer_lookup_case_insensitive(db: Database) -> None:
    assert db.get_customer_by_email("SAM.ORTIZ@example.com") is not None
    assert db.get_customer_by_email("nobody@example.com") is None


def test_recent_orders_limit_and_order(db: Database) -> None:
    cust = db.get_customer_by_email("jose.martinez@example.com")
    assert cust is not None
    orders = db.get_recent_orders(cust.id, limit=5)
    assert {o.id for o in orders} == {"KO-10627", "KO-10628", "KO-10629"}
    dates = [o.order_date for o in orders]
    assert dates == sorted(dates, reverse=True)


def test_ticket_crud_outbox_escalation_review(db: Database) -> None:
    db.create_ticket("T-0001", "a@example.com", "Subj", "Body")
    assert db.get_ticket("T-0001")["status"] == "received"  # type: ignore[index]
    db.update_ticket("T-0001", status="awaiting_review", route="human_review", result={"x": 1})
    t = db.get_ticket("T-0001")
    assert t is not None and t["route"] == "human_review" and t["result"] == {"x": 1}
    assert [x["id"] for x in db.list_tickets("awaiting_review")] == ["T-0001"]
    db.write_outbox("T-0001", "a@example.com", "hello", "auto")
    assert len(db.get_outbox("T-0001")) == 1
    db.create_escalation("T-0001", "legal", "lawyer")
    assert db.get_escalations("T-0001")[0]["team"] == "legal"
    ev = db.record_review_event("T-0001", "kim", "approve", duration_seconds=12.5)
    assert ev["action"] == "approve"
    assert db.list_review_events()[0]["duration_seconds"] == 12.5
    assert db.next_ticket_id() == "T-0002"
    assert db.get_ticket("T-404") is None
    assert db.ping()
