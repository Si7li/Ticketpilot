import json
import logging
import socket
import threading
from typing import Any

import pytest

from ticketpilot import telemetry as T
from ticketpilot.config import Settings
from ticketpilot.llm import prompts
from ticketpilot.schemas import DraftContext, Entities, Evidence


def test_usage_scope_attribution_is_per_thread() -> None:
    tracker = T.UsageTracker()
    seen: dict[str, Any] = {}

    def work(name: str, n: int) -> None:
        with T.usage_scope() as scope:
            for _ in range(n):
                tracker.record(name, "m", 10, 1)
            seen[name] = scope.by_model()

    threads = [threading.Thread(target=work, args=(f"t{i}", i + 1)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [seen[f"t{i}"]["m"]["calls"] for i in range(4)] == [1, 2, 3, 4]
    assert sum(v["calls"] for v in tracker.totals().values()) == 10
    tracker.record("outside", "m", 1, 1)  # no active scope: global only
    assert tracker.totals()["outside:m"]["calls"] == 1


def test_merge_usage_and_by_model() -> None:
    with T.usage_scope() as scope:
        tr = T.UsageTracker()
        tr.record("classify", "a", 100, 10)
        tr.record("classify", "a", 50, 5)
    merged = T.merge_usage({}, "classify", scope)
    merged = T.merge_usage(merged, "classify", scope)
    assert merged["classify"]["a"] == {"input_tokens": 300, "output_tokens": 30, "calls": 4}
    assert (
        T.usage_by_model(
            {
                "x": {"a": {"input_tokens": 1, "output_tokens": 2, "calls": 1}},
                "y": {"a": {"input_tokens": 3, "output_tokens": 4, "calls": 1}},
            }
        )["a"]["input_tokens"]
        == 4
    )
    assert T.merge_usage({"n": {}}, "m", T.UsageScope()) == {"n": {}}


def test_cost_only_when_prices_configured() -> None:
    tokens = {"claude-haiku-5-5": {"input_tokens": 2_000_000, "output_tokens": 1_000_000}}
    assert T.cost_usd(tokens, Settings(_env_file=None)) is None  # type: ignore[call-arg]
    s = Settings(_env_file=None, AGENT_PRICE_INPUT_PER_MTOK=0.10, AGENT_PRICE_OUTPUT_PER_MTOK=0.50)  # type: ignore[call-arg]
    assert T.cost_usd(tokens, s) == pytest.approx(0.70)
    assert T.cost_usd({"unknown-model": {"input_tokens": 1, "output_tokens": 1}}, s) is None


def test_node_timer() -> None:
    with T.node_timer() as t:
        sum(range(1000))
    assert t["ms"] >= 0


def test_json_formatter() -> None:
    rec = logging.LogRecord(
        "ticketpilot.node", logging.INFO, __file__, 1, "node finished", (), None
    )
    rec.ticket_id = "T-1"  # type: ignore[attr-defined]
    rec.duration_ms = 12.5  # type: ignore[attr-defined]
    out = json.loads(T.JsonFormatter().format(rec))
    assert (
        out["msg"] == "node finished" and out["ticket_id"] == "T-1" and out["duration_ms"] == 12.5
    )
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec.exc_info = sys.exc_info()
    assert "boom" in json.loads(T.JsonFormatter().format(rec))["exc"]


def test_configure_logging_and_log_node(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(T, "_configured", False)
    T.configure_logging(Settings(_env_file=None, LOG_JSON=True))  # type: ignore[call-arg]
    assert isinstance(logging.getLogger("ticketpilot").handlers[0].formatter, T.JsonFormatter)
    T.configure_logging(Settings(_env_file=None))  # type: ignore[call-arg]  # idempotent
    logging.getLogger("ticketpilot").propagate = True
    with caplog.at_level(logging.INFO, logger="ticketpilot.node"):
        T.log_node("T-9", "classify", 3.2, "ok")
    assert any(getattr(r, "node", None) == "classify" for r in caplog.records)
    logging.getLogger("ticketpilot").propagate = False


def test_langfuse_absent_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_langfuse_logged", False)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("ticketpilot.telemetry")
    handler = Capture(level=logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert T.langfuse_handler(s) is None
        assert T.langfuse_handler(s) is None
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert [r.getMessage() for r in records] == [
        "Langfuse not configured; skipping"
    ]  # one info log
    assert all(r.levelno == logging.INFO for r in records)


def test_prompt_formatting() -> None:
    assert prompts.format_excerpts([]) == "(none)"
    assert 'doc_id="policy/a"' in prompts.format_excerpts([{"doc_id": "policy/a", "text": "t"}])
    assert prompts.format_tool_results([]) == "(no tools were called)"
    assert '"ok": true' in prompts.format_tool_results([{"tool": "x", "ok": True}])
    assert 'source="tool:x"' in prompts.format_evidence([Evidence(source="tool:x", text="{}")])
    ctx = DraftContext(
        ticket_text="hi",
        customer_email="a@b.c",
        category="order_status",
        entities=Entities(),
        retrieved=[],
        tool_results=[],
    )
    assert "a@b.c" in prompts.draft_user_prompt(ctx)
    assert "order_status" in prompts.category_list()
    assert prompts.PROMPT_VERSION


def test_network_guard_blocks_external_hosts() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Network access blocked"):
            s.connect(("93.184.216.34", 80))
    finally:
        s.close()
