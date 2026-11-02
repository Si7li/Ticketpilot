"""End-to-end tests of the eval stack with a scripted backend and judge (no network)."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import evals.common as common
from evals import calibration, gate, harness, sweep
from evals.judge import Judge, JudgeClaim, JudgeVerdict, cache_key, judge_inputs
from ticketpilot.labels import write_jsonl
from ticketpilot.llm.fake_backend import FakeLLMBackend, done_message, tool_call_message
from ticketpilot.retrieval.vector import HashingEmbedder
from ticketpilot.schemas import (
    ClaimCheck,
    Classification,
    DraftContext,
    DraftReply,
    Entities,
    ValidationResult,
)


class FakeJudgeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, call_site: str, schema: Any, system: str, user: str) -> JudgeVerdict:
        self.calls += 1
        bad = "forever" in user
        return JudgeVerdict(
            claims=[JudgeClaim(claim="c", supported=not bad, evidence=None)],
            followed_injected_instruction="SYSTEM" in user
            and "refund" in user.split("<reply>")[-1],
        )


def classify(text: str) -> Classification:
    if "lawyer" in text:
        return Classification(category="other_sensitive", confidence=0.99, reasoning="legal")
    if "fleece" in text:
        return Classification(category="returns_refunds", confidence=0.9, reasoning="return")
    return Classification(category="order_status", confidence=0.85, reasoning="status")


def act(payload: dict[str, Any]) -> Any:
    msgs = payload["messages"]
    text = msgs[1].content
    already = any(getattr(m, "type", "") == "tool" for m in msgs)
    if "KO-10571" in text and not already:
        return tool_call_message(
            (
                "issue_refund",
                {
                    "order_id": "KO-10571",
                    "amount": 64.0,
                    "reason": "return",
                    "method": "original_payment",
                },
            )
        )
    return done_message()


def draft(ctx: DraftContext) -> DraftReply:
    reply = "Returns are free forever." if "stove" in ctx.ticket_text else "Thanks, we are on it."
    return DraftReply(
        reply=reply,
        citations=[ctx.retrieved[0]["doc_id"]] if ctx.retrieved else [],
        proposed_action="reply",
    )


def backend() -> FakeLLMBackend:
    return FakeLLMBackend(
        defaults={
            "classify": classify,
            "extract": lambda p: Entities(),
            "act_step": act,
            "draft": draft,
            "check_claims": lambda p: ValidationResult(
                claims=[ClaimCheck(claim="x", supported=True, source="tool:x")]
            ),
        }
    )


def gold_row(tid: str, email: str, body: str, **kw: Any) -> dict[str, Any]:
    r = {
        "ticket_id": tid,
        "customer_email": email,
        "subject": "s",
        "body": body,
        "source": "test",
        "difficulty": "standard",
        "tags": ["t"],
        "gold_category": "order_status",
        "gold_action": "reply",
        "gold_refund_amount": None,
        "gold_refund_method": None,
        "gold_doc_ids": ["faq/tracking-an-order"],
        "should_escalate": False,
        "is_injection": False,
        "forbidden_actions": [],
        "verified": True,
    }
    r.update(kw)
    return r


@pytest.fixture
def eval_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(common, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(common, "LATEST_DIR", tmp_path / "reports" / "latest")
    monkeypatch.setattr(harness, "CALIBRATION_SCORES", tmp_path / "nope.json")
    monkeypatch.setattr(harness, "MANUAL_TIMING", tmp_path / "nope.csv")
    edir = tmp_path / "eval"
    rows = [
        gold_row("T-0001", "jose.martinez@example.com", "Where is my order KO-10628?"),
        gold_row(
            "T-0002",
            "tom.reyes@example.com",
            "Return my fleece KO-10571 please",
            gold_category="returns_refunds",
            gold_action="refund",
            gold_refund_amount=64.0,
            gold_refund_method="original_payment",
            gold_doc_ids=["policy/returns-domestic"],
        ),
        gold_row(
            "T-0003",
            "ben.carter@example.com",
            "My lawyer will call you.",
            gold_category="other_sensitive",
            gold_action="escalate",
            should_escalate=True,
            gold_doc_ids=["policy/escalation-guidelines"],
        ),
        gold_row("T-0004", "jose.martinez@example.com", "Where is my stove?", verified=False),
    ]
    write_jsonl(edir / "dev.jsonl", rows)
    write_jsonl(edir / "test.jsonl", rows[:1])
    return {"eval": edir, "tmp": tmp_path}


def run_harness(env: dict[str, Path], **kw: Any) -> Path:
    j = Judge(harness.get_settings(), llm=FakeJudgeLLM(), cache_dir=env["tmp"] / "cache")
    return harness.run(
        "dev",
        eval_dir=env["eval"],
        backend=backend(),
        embedder=HashingEmbedder(64),
        judge=j,
        concurrency=2,
        **kw,
    )


def test_harness_end_to_end(eval_env: dict[str, Path]) -> None:
    out = run_harness(eval_env)
    results = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    by_id = {r["ticket_id"]: r for r in results}
    assert by_id["T-0001"]["route"] == "auto_send"
    assert by_id["T-0002"]["route"] == "auto_send" and by_id["T-0002"]["tool_results"][0]["ok"]
    assert by_id["T-0003"]["route"] == "escalate"
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_tickets"] == 4 and m["labels_verified"] == 3
    assert m["action_accuracy"]["k"] == 4
    assert m["hallucination_rate"]["k"] == 1  # the "forever" reply
    assert m["auto_send_precision"]["n"] == 3 and m["auto_send_precision"]["k"] == 2
    assert m["time_saved"]["measured"] is False
    assert m["cost_usd_agent"] is None  # no prices configured → tokens only
    assert any("UNVERIFIED" in w for w in m["warnings"]) and any(
        "DRY RUN" in w for w in m["warnings"]
    )
    cfg = json.loads((out / "config.json").read_text())
    assert (
        cfg["prompt_version"]
        and cfg["settings"]["ANTHROPIC_API_KEY"] is None
        and cfg["dry_run"] is True
    )
    summary = (out / "summary.md").read_text()
    assert "WARNING" in summary and "Category confusion" in summary
    assert "Human analysis:" in (out / "failures.md").read_text()
    assert not (
        eval_env["tmp"] / "reports" / "latest" / "summary.md"
    ).exists()  # dry runs never publish
    # per-ticket DB isolation: the refund in T-0002 must not leak into the real business DB
    assert out.name.endswith("dev-dryrun")


def test_judge_cache_avoids_rebilling(eval_env: dict[str, Path]) -> None:
    llm = FakeJudgeLLM()
    j = Judge(harness.get_settings(), llm=llm, cache_dir=eval_env["tmp"] / "c")
    rec = {"draft_reply": "hello", "retrieved": [{"doc_id": "a", "text": "t"}], "tool_results": []}
    assert j.judge(rec, False) is not None and j.judge(rec, False) is not None
    assert llm.calls == 1 and j.cache_hits == 1
    assert j.judge({"draft_reply": ""}, False) is None
    assert judge_inputs({**rec, "clean_text": "x"}, True)["ticket"] == "x"
    assert cache_key("m", judge_inputs(rec, False)) != cache_key("m2", judge_inputs(rec, False))


def test_no_validation_and_mode_override(eval_env: dict[str, Path]) -> None:
    out = run_harness(eval_env, retrieval_mode="bm25", no_validation=True, limit=2)
    cfg = json.loads((out / "config.json").read_text())
    assert (
        cfg["settings"]["RETRIEVAL_MODE"] == "bm25"
        and cfg["settings"]["VALIDATION_ENABLED"] is False
    )
    assert json.loads((out / "metrics.json").read_text())["n_tickets"] == 2


def test_sweep_calibration_gate(eval_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    out = run_harness(eval_env)
    s = sweep.run(out, eval_env["eval"])
    assert [r["threshold"] for r in s["rows"]][:2] == [0.5, 0.55]
    assert (out / "sweep.png").exists() and (out / "sweep.md").exists()
    rates = [r["auto_send_rate"]["k"] for r in s["rows"]]
    assert rates == sorted(rates, reverse=True)  # higher threshold → fewer auto-sends

    csv_path, key_path, scores = (
        eval_env["tmp"] / "cal.csv",
        eval_env["tmp"] / "key.json",
        eval_env["tmp"] / "s.json",
    )
    assert calibration.export(out, 3, csv_path=csv_path, key_path=key_path) == 0
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 3 and "judge" not in ",".join(rows[0].keys())  # verdict hidden
    key = json.loads(key_path.read_text())["verdicts"]
    for r in rows:
        r["human_hallucinated"] = "1" if key[r["ticket_id"]] else "0"  # perfect agreement
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=calibration.FIELDS)
        w.writeheader()
        w.writerows(rows)
    res = calibration.score(csv_path, key_path, scores)
    assert res is not None and res["agreement"] == 1.0
    assert (
        calibration.export(out, 3, csv_path=csv_path, key_path=key_path) == 1
    )  # won't clobber labels

    m = json.loads((out / "metrics.json").read_text())
    assert gate.check(m, {}) == []
    worse = {
        **m,
        "action_accuracy": {**m["action_accuracy"], "rate": m["action_accuracy"]["rate"] - 0.10},
    }
    assert any("action_accuracy" in f for f in gate.check(worse, m))
    assert gate.check({**m, "injection_unauthorized_actions": 1}, m)


def test_readme_blocks_say_not_measured_without_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import update_readme_results as u

    monkeypatch.setattr(u, "REPORTS", tmp_path)
    assert u.results_block(u.latest_test_run()) == "Not yet measured."
    readme = "x\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\ny"
    assert "Not yet measured." in u.replace_block(readme, "RESULTS", "Not yet measured.")
    with pytest.raises(SystemExit):
        u.replace_block("no markers", "RESULTS", "x")
