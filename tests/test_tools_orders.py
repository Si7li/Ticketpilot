from typing import Any

from tests.conftest import call
from ticketpilot.tools.registry import ToolRegistry

SAM = "sam.ortiz@example.com"


def test_lookup_own_order(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("lookup_order", order_id="KO-10482"), ctx_for(SAM))
    assert r.ok and not r.arg_error
    assert r.data["order_id"] == "KO-10482"
    assert r.data["days_since_delivery"] == 5
    assert r.data["return_window_days"] == 30
    assert r.data["active_refund"] is None
    assert r.duration_ms >= 0


def test_lookup_normalises_case(registry: ToolRegistry, ctx_for: Any) -> None:
    assert registry.execute(call("lookup_order", order_id="ko-10482"), ctx_for(SAM)).ok


def test_other_customers_order_is_not_found(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("lookup_order", order_id="KO-10511"), ctx_for(SAM))
    assert not r.ok
    assert r.error == "order not found for this customer"
    assert r.data is None  # does not leak that the order exists


def test_nonexistent_order_same_message(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("lookup_order", order_id="KO-99999"), ctx_for(SAM))
    assert r.error == "order not found for this customer"


def test_malformed_order_id_structured_error(registry: ToolRegistry, ctx_for: Any) -> None:
    for bad in ("10482", "KO-1048", "KO-104822", "XX-10482", "KO-10482; DROP TABLE"):
        r = registry.execute(call("lookup_order", order_id=bad), ctx_for(SAM))
        assert not r.ok and "invalid order ID format" in (r.error or "")


def test_missing_arg_is_structured_error(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("lookup_order"), ctx_for(SAM))
    assert not r.ok and r.arg_error and "order_id" in (r.error or "")


def test_model_cannot_supply_customer_email(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(
        call("lookup_order", order_id="KO-10511", customer_email="priya.nair@example.com"),
        ctx_for(SAM),
    )
    assert not r.ok and r.arg_error
    assert "customer_email" in (r.error or "")


def test_get_customer_orders(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("get_customer_orders"), ctx_for("jose.martinez@example.com"))
    assert r.ok
    assert [o["order_id"] for o in r.data["orders"]] == ["KO-10628", "KO-10627", "KO-10629"]


def test_get_customer_orders_unknown_sender(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("get_customer_orders"), ctx_for("stranger@example.com"))
    assert not r.ok


def test_unknown_tool(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("delete_database"), ctx_for(SAM))
    assert not r.ok and r.arg_error and "unknown tool" in (r.error or "")


def test_search_kb_caps_top_k(registry: ToolRegistry, ctx_for: Any) -> None:
    r = registry.execute(call("search_kb", query="tent drying", top_k=50), ctx_for(SAM))
    assert r.ok and len(r.data) <= 8


def test_escalate_records_and_flags(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for(SAM, "T-0042")
    r = registry.execute(call("escalate", team="legal", reason="lawyer mentioned"), ctx)
    assert r.ok and r.needs_specialist
    assert ctx.db.get_escalations("T-0042")[0]["team"] == "legal"
    bad = registry.execute(call("escalate", team="ceo", reason="x" * 10), ctx)
    assert not bad.ok and bad.arg_error


def test_tool_exception_becomes_error_result(registry: ToolRegistry, ctx_for: Any) -> None:
    ctx = ctx_for(SAM)
    ctx.retriever = None
    r = registry.execute(call("search_kb", query="anything"), ctx)
    assert not r.ok and r.error == "knowledge base is unavailable"


def test_schemas_are_anthropic_tools(registry: ToolRegistry) -> None:
    schemas = registry.schemas()
    names = {s["name"] for s in schemas}
    assert names == {"search_kb", "lookup_order", "get_customer_orders", "issue_refund", "escalate"}
    for s in schemas:
        assert s["input_schema"]["type"] == "object"
        assert "customer_email" not in s["input_schema"].get("properties", {})
        assert "ticket_id" not in s["input_schema"].get("properties", {})
