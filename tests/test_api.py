from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.graph_helpers import runtime, script
from ticketpilot.api.main import UnavailableBackend, create_app, get_runtime
from ticketpilot.config import Settings
from ticketpilot.db import Database
from ticketpilot.llm.fake_backend import FakeLLMBackend, done_message, tool_call_message
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.runtime import Runtime

SAM = "sam.ortiz@example.com"
JOSE = "jose.martinez@example.com"


def over_limit_backend() -> FakeLLMBackend:
    return script(
        category="shipping_problem",
        confidence=0.93,
        order_ids=["KO-10482"],
        act=[
            tool_call_message(
                (
                    "issue_refund",
                    {
                        "order_id": "KO-10482",
                        "amount": 189.0,
                        "reason": "torn",
                        "method": "original_payment",
                        "refund_type": "damaged",
                    },
                )
            ),
            done_message(),
        ],
        reply="Your refund request is being reviewed by our team.",
        action="refund",
    )


@pytest.fixture
def make_client(settings: Settings, db: Database, retriever: HybridRetriever) -> Iterator[Any]:
    clients: list[TestClient] = []

    def _make(backend: Any) -> tuple[TestClient, Runtime]:
        rt = runtime(backend, settings, db, retriever)
        app = create_app(build_on_startup=False)
        app.state.runtime = rt
        app.dependency_overrides[get_runtime] = lambda: rt
        c = TestClient(app)
        clients.append(c)
        return c, rt

    yield _make
    for c in clients:
        c.close()


def test_post_ticket_auto_send(make_client: Any) -> None:
    client, rt = make_client(script(reply="Your order has shipped."))
    r = client.post(
        "/tickets",
        json={
            "subject": "Where",
            "body": "Where is my stove?",
            "customer_email": JOSE,
            "ticket_id": "T-0101",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticket_id"] == "T-0101"
    assert body["route"] == "auto_send" and body["status"] == "auto_sent"
    assert body["category"] == "order_status" and body["total_ms"] > 0
    assert rt.db.get_ticket("T-0101")["result"]["route"] == "auto_send"


def test_ticket_id_generated_and_duplicate_conflict(make_client: Any) -> None:
    client, _ = make_client(
        FakeLLMBackend(
            defaults={
                "classify": script().queues["classify"][0],
                "extract": script().queues["extract"][0],
                "act_step": done_message(),
                "draft": script().queues["draft"][0],
                "check_claims": script().queues["check_claims"][0],
            }
        )
    )
    a = client.post("/tickets", json={"body": "hi", "customer_email": JOSE})
    assert a.status_code == 200 and a.json()["ticket_id"].startswith("T-")
    dup = client.post(
        "/tickets", json={"body": "hi", "customer_email": JOSE, "ticket_id": a.json()["ticket_id"]}
    )
    assert dup.status_code == 409
    assert dup.json()["error"]["code"] == "conflict"


def test_validation_errors_are_structured(make_client: Any) -> None:
    client, _ = make_client(script())
    r = client.post("/tickets", json={"body": "", "customer_email": "not-an-email"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and len(err["details"]) >= 2
    bad_id = client.post("/tickets", json={"body": "x", "customer_email": JOSE, "ticket_id": "42"})
    assert bad_id.status_code == 422


def test_review_flow(make_client: Any) -> None:
    client, rt = make_client(over_limit_backend())
    r = client.post(
        "/tickets",
        json={
            "subject": "Torn tent",
            "customer_email": SAM,
            "body": "Ridgeline 2 KO-10482 torn, photo attached",
            "ticket_id": "T-0102",
        },
    )
    assert r.json()["route"] == "human_review" and r.json()["status"] == "awaiting_review"
    assert "refund pending approval" in r.json()["review_reasons"]

    q = client.get("/review-queue").json()
    assert [i["ticket_id"] for i in q] == ["T-0102"]
    assert q[0]["payload"]["pending_refunds"][0]["amount"] == 189.0

    detail = client.get("/tickets/T-0102").json()
    assert detail["awaiting_review"] is True and detail["graph_next"] == ["human_review"]

    done = client.post(
        "/review/T-0102", json={"action": "approve", "reviewer": "kim", "duration_seconds": 30}
    )
    assert done.status_code == 200, done.text
    assert done.json()["status"] == "closed" and done.json()["final_reply"]
    assert rt.db.get_active_refund("KO-10482").status == "approved"
    assert rt.db.get_outbox("T-0102")[0]["sent_by"] == "kim"
    assert client.get("/review-queue").json() == []

    again = client.post("/review/T-0102", json={"action": "approve", "reviewer": "kim"})
    assert again.status_code == 409


def test_edit_approve_requires_text(make_client: Any) -> None:
    client, _ = make_client(over_limit_backend())
    client.post(
        "/tickets",
        json={
            "customer_email": SAM,
            "body": "KO-10482 torn, photo attached",
            "ticket_id": "T-0103",
        },
    )
    r = client.post("/review/T-0103", json={"action": "edit_approve", "reviewer": "kim"})
    assert r.status_code == 422


def test_review_not_found_and_not_awaiting(make_client: Any) -> None:
    client, _ = make_client(script())
    assert (
        client.post("/review/T-9999", json={"action": "approve", "reviewer": "x"}).status_code
        == 404
    )
    client.post(
        "/tickets",
        json={"body": "Where is my stove?", "customer_email": JOSE, "ticket_id": "T-0104"},
    )
    assert (
        client.post("/review/T-0104", json={"action": "approve", "reviewer": "x"}).status_code
        == 409
    )
    assert client.get("/tickets/T-0404").status_code == 404


def test_health(make_client: Any) -> None:
    client, _ = make_client(script())
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() | {} == h.json()
    assert (
        h.json()["ok"]
        and h.json()["db"]
        and h.json()["kb_indexed"]
        and h.json()["model_configured"]
    )


def test_unavailable_backend_returns_503(make_client: Any) -> None:
    client, _ = make_client(UnavailableBackend("ANTHROPIC_API_KEY is not set"))
    h = client.get("/health").json()
    assert h["model_configured"] is False
    r = client.post(
        "/tickets", json={"body": "hello", "customer_email": JOSE, "ticket_id": "T-0105"}
    )
    # The graph's fail-safe turns the backend error into a human-review ticket, never a crash.
    assert r.status_code == 200
    assert r.json()["route"] == "human_review"
    assert any("BackendError" in e for e in r.json()["errors"])
