from datetime import UTC, datetime, timedelta

from ticketpilot.config import PROJECT_ROOT
from ticketpilot.review_ui.helpers import age_label, highlight_unsupported, unsupported_sentences

DRAFT = "Thanks for reaching out. Your refund of $189 has been approved. Returns are free forever."


def test_highlights_only_unsupported_sentence() -> None:
    claims = [
        {"claim": "Returns are free forever", "supported": False},
        {"claim": "refund of $189 approved", "supported": True},
    ]
    assert unsupported_sentences(DRAFT, claims) == {2}
    out = highlight_unsupported(DRAFT, claims)
    assert "<mark>Returns are free forever.</mark>" in out
    assert "<mark>Thanks" not in out


def test_html_is_escaped() -> None:
    out = highlight_unsupported(
        "<script>x</script> is bad.", [{"claim": "script bad", "supported": False}]
    )
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_no_match_below_threshold() -> None:
    assert unsupported_sentences(DRAFT, [{"claim": "kayaks float", "supported": False}]) == set()


def test_age_label() -> None:
    now = datetime(2026, 10, 1, 12, tzinfo=UTC)
    assert age_label((now - timedelta(seconds=30)).isoformat(), now) == "30s"
    assert age_label((now - timedelta(minutes=5)).isoformat(), now) == "5m"
    assert age_label((now - timedelta(hours=3)).isoformat(), now) == "3h"
    assert age_label((now - timedelta(days=2)).isoformat(), now) == "2d"
    assert age_label("garbage", now) == "?"


def test_app_renders_queue_and_approves(monkeypatch: object) -> None:
    """Run the Streamlit script headlessly against a stubbed API."""
    import httpx
    from streamlit.testing.v1 import AppTest

    payload = {
        "ticket_id": "T-0102",
        "customer_email": "sam.ortiz@example.com",
        "subject": "Torn tent",
        "body": "torn",
        "category": "shipping_problem",
        "confidence": 0.93,
        "draft_reply": "Your refund request is being reviewed. Returns are free forever.",
        "citations": ["policy/damaged-items"],
        "retrieved": [
            {"doc_id": "policy/damaged-items", "section": "Reporting window", "text": "..."}
        ],
        "tool_results": [
            {"tool": "issue_refund", "ok": True, "pending": True, "args": {}, "data": {}}
        ],
        "validation": {
            "claims": [{"claim": "Returns are free forever", "supported": False}],
            "issues": ["unsupported claim: Returns are free forever"],
        },
        "pending_refunds": [{"id": "RF-1", "amount": 189.0, "method": "original_payment"}],
        "reasons": ["refund pending approval"],
    }
    queue = [
        {
            "ticket_id": "T-0102",
            "customer_email": "sam.ortiz@example.com",
            "subject": "Torn tent",
            "created_at": "2026-10-01T00:00:00+00:00",
            "category": "shipping_problem",
            "confidence": 0.93,
            "reasons": ["refund pending approval"],
            "payload": payload,
        }
    ]
    posted: list[dict[str, object]] = []

    def fake_request(method: str, url: str, **kw: object) -> httpx.Response:
        req = httpx.Request(method, url)
        if url.endswith("/review-queue"):
            return httpx.Response(200, json=queue if not posted else [], request=req)
        if "/review/" in url:
            posted.append(kw["json"])  # type: ignore[arg-type]
            return httpx.Response(200, json={"status": "closed"}, request=req)
        return httpx.Response(
            404, json={"error": {"code": "not_found", "message": url}}, request=req
        )

    monkeypatch.setattr(httpx, "request", fake_request)  # type: ignore[attr-defined]
    at = AppTest.from_file(
        str(PROJECT_ROOT / "src/ticketpilot/review_ui/app.py"), default_timeout=30
    ).run()
    assert not at.exception
    assert any("Pending refund RF-1" in w.value for w in at.warning)
    assert any("unsupported claim" in e.value for e in at.error)
    at.sidebar.text_input[0].set_value("kim").run()
    next(b for b in at.button if b.label == "Approve").click().run()
    assert posted and posted[0]["action"] == "approve" and posted[0]["reviewer"] == "kim"
    assert isinstance(posted[0]["duration_seconds"], float)
