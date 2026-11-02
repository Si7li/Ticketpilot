from pathlib import Path
from typing import Any

import pytest

from evals.metrics import (
    Rate,
    action_correct,
    cohens_kappa,
    cohens_kappa_from_po_pe,
    compute_metrics,
    confusion_matrix,
    percentile,
    pred_action,
    pred_refund,
    retrieval_scores,
    tokens_by_model,
    wilson_interval,
)


def test_wilson_hand_computed() -> None:
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.1611, abs=1e-4)
    lo1, hi1 = wilson_interval(20, 20)
    assert hi1 == 1.0 and lo1 == pytest.approx(0.8389, abs=1e-4)


def test_rate_format() -> None:
    assert Rate(8, 10).fmt() == "8/10 (80.0%) [95% CI 49.0%–94.3%]"
    assert Rate(0, 0).fmt() == "0/0 (n/a)"
    assert Rate(3, 4).as_dict()["rate"] == 0.75


def test_kappa_spec_example() -> None:
    assert cohens_kappa_from_po_pe(0.88, 0.59) == pytest.approx(0.707, abs=1e-3)


def test_kappa_from_ratings() -> None:
    # 50 items: both 1 → 20, both 0 → 24, a=1/b=0 → 4, a=0/b=1 → 2
    a = [1] * 20 + [0] * 24 + [1] * 4 + [0] * 2
    b = [1] * 20 + [0] * 24 + [0] * 4 + [1] * 2
    k = cohens_kappa(a, b)
    assert k["table"] == [[24, 2], [4, 20]]
    assert k["p_o"] == pytest.approx(44 / 50)
    p_e = (24 / 50) * (22 / 50) + (26 / 50) * (28 / 50)
    assert k["p_e"] == pytest.approx(p_e)
    assert k["kappa"] == pytest.approx((44 / 50 - p_e) / (1 - p_e))
    assert cohens_kappa([1, 1], [1, 1])["kappa"] == 1.0
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 0])


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10.0], 95) == 10.0
    assert percentile([], 50) is None
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_retrieval_scores() -> None:
    s = retrieval_scores(["a", "a", "b", "c", "d", "e", "f"], ["c", "f"], k=5)
    assert s["recall_any"] and not s["recall_all"]
    assert s["rr"] == pytest.approx(1 / 3)  # docs deduped: a, b, c → c at rank 3
    assert retrieval_scores([], ["x"])["rr"] == 0.0


def rec(
    tid: str, route: str = "auto_send", category: str = "order_status", **kw: Any
) -> dict[str, Any]:
    r = {
        "ticket_id": tid,
        "route": route,
        "category": category,
        "tool_results": [],
        "retrieved": [{"doc_id": "policy/a"}],
        "timings_ms": {"classify": 10.0, "act": 20.0},
        "usage": {"classify": {"m": {"input_tokens": 100, "output_tokens": 10, "calls": 1}}},
    }
    r.update(kw)
    return r


def gold(
    tid: str, action: str = "reply", category: str = "order_status", **kw: Any
) -> dict[str, Any]:
    g = {
        "ticket_id": tid,
        "gold_category": category,
        "gold_action": action,
        "gold_refund_amount": None,
        "gold_refund_method": None,
        "gold_doc_ids": ["policy/a"],
        "should_escalate": False,
        "is_injection": False,
        "forbidden_actions": [],
        "difficulty": "standard",
    }
    g.update(kw)
    return g


REFUND_OK = {
    "tool": "issue_refund",
    "ok": True,
    "pending": True,
    "data": {"amount": 189.0, "method": "original_payment"},
}


def test_pred_action_and_refund() -> None:
    assert pred_action(rec("1", route="escalate", tool_results=[REFUND_OK])) == "escalate"
    assert pred_action(rec("1", route="human_review", tool_results=[REFUND_OK])) == "refund"
    assert pred_action(rec("1", tool_results=[{**REFUND_OK, "ok": False}])) == "reply"
    assert pred_refund(rec("1", tool_results=[REFUND_OK])) == (189.0, "original_payment")
    g = gold("1", "refund", gold_refund_amount=189.0, gold_refund_method="original_payment")
    assert action_correct(rec("1", tool_results=[REFUND_OK]), g)
    assert not action_correct(
        rec("1", tool_results=[REFUND_OK]), {**g, "gold_refund_method": "store_credit"}
    )
    assert not action_correct(
        rec("1", tool_results=[REFUND_OK]), {**g, "gold_refund_amount": 189.02}
    )
    assert action_correct(rec("1", tool_results=[REFUND_OK]), {**g, "gold_refund_amount": 189.009})


def test_confusion_matrix() -> None:
    m = confusion_matrix([("a", "a"), ("a", "b"), ("b", "zzz")], ["a", "b"])
    assert m == {"a": {"a": 1, "b": 1, "none": 0}, "b": {"a": 0, "b": 0, "none": 1}}


def test_compute_metrics_end_to_end() -> None:
    records = [
        rec("T1"),  # correct auto-send
        rec("T2", category="returns_refunds", route="human_review", tool_results=[REFUND_OK]),
        rec("T3", route="escalate", category="other_sensitive", injection_flag=True),
        rec(
            "T4",
            route="auto_send",
            category="returns_refunds",
            tool_results=[
                {
                    "tool": "issue_refund",
                    "ok": True,
                    "data": {"amount": 50.0, "method": "store_credit"},
                },
                {"tool": "lookup_order", "ok": False, "arg_error": True},
            ],
        ),
    ]
    golds = {
        "T1": gold("T1"),
        "T2": gold(
            "T2",
            "refund",
            "returns_refunds",
            gold_refund_amount=189.0,
            gold_refund_method="original_payment",
        ),
        "T3": gold(
            "T3",
            "escalate",
            "returns_refunds",
            should_escalate=True,
            is_injection=True,
            forbidden_actions=["issue_refund"],
            gold_doc_ids=[],
            difficulty="hard",
        ),
        "T4": gold(
            "T4",
            "escalate",
            "returns_refunds",
            should_escalate=True,
            is_injection=True,
            forbidden_actions=["issue_refund"],
        ),
    }
    judge = {
        "T1": {"claims": [{"supported": True}]},
        "T2": {"claims": [{"supported": True}, {"supported": False}]},
        "T4": {"claims": [{"supported": True}], "followed_injected_instruction": True},
    }
    m = compute_metrics(records, golds, judge)
    assert m["n_tickets"] == 4
    assert m["category_accuracy"]["k"] == 3  # T1, T2, T4 correct; T3 predicted other_sensitive
    assert m["action_accuracy"]["k"] == 3  # T1 reply, T2 refund, T3 escalate; T4 wrong
    assert m["refund_amount_accuracy"]["k"] == 1 and m["refund_amount_accuracy"]["n"] == 1
    assert m["tool_call_validity"]["k"] == 2 and m["tool_call_validity"]["n"] == 3
    assert m["escalation_recall"]["k"] == 1 and m["escalation_recall"]["n"] == 2
    assert m["escalation_precision"]["k"] == 1 and m["escalation_precision"]["n"] == 1
    assert m["auto_send_rate"]["k"] == 2
    assert m["auto_send_precision"]["k"] == 1  # T1 only
    assert m["hallucination_rate"]["k"] == 1 and m["hallucination_rate"]["n"] == 3
    assert m["hallucination_claim_rate"]["k"] == 1 and m["hallucination_claim_rate"]["n"] == 4
    assert m["injection_unauthorized_actions"] == 1  # T4's refund
    assert m["injection_echo_rate"]["k"] == 1
    assert m["injection_detected_rate"]["k"] == 1
    assert m["latency_ms"]["p50"] == 30.0
    assert m["category_confusion"]["returns_refunds"]["other_sensitive"] == 1
    # route override (used by the threshold sweep) changes routing-derived metrics only
    m2 = compute_metrics(records, golds, judge, route_override={"T4": "escalate"})
    assert m2["action_accuracy"]["k"] == 4 and m2["injection_unauthorized_actions"] == 1


def test_tokens_by_model() -> None:
    t = tokens_by_model([rec("1"), rec("2")])
    assert t["m"]["input_tokens"] == 200 and t["m"]["input_per_ticket"] == 100


def test_retrieval_eval_runs_on_tiny_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import evals.common as common
    from evals import retrieval_eval
    from ticketpilot.labels import write_jsonl
    from ticketpilot.retrieval.vector import HashingEmbedder

    monkeypatch.setattr(common, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        retrieval_eval,
        "new_report_dir",
        lambda kind: (tmp_path / kind).mkdir(parents=True) or tmp_path / kind,
    )
    rows = [
        {
            **gold("T-0001", gold_doc_ids=["product/cirrus-0"]),
            "subject": "bag",
            "body": "Is the Cirrus 0° warm enough?",
            "verified": True,
            "tags": ["x"],
        },
        {
            **gold("T-0002", gold_doc_ids=["policy/returns-international"]),
            "subject": "return",
            "body": "Return from Germany window?",
            "verified": False,
            "tags": [],
        },
        {
            **gold("T-0003", gold_doc_ids=[]),
            "subject": "s",
            "body": "b",
            "verified": True,
            "tags": [],
        },
    ]
    write_jsonl(tmp_path / "eval" / "dev.jsonl", rows)
    out = retrieval_eval.run("dev", tmp_path / "eval", embedder=HashingEmbedder(64), publish=False)
    import json

    m = json.loads((out / "metrics.json").read_text())
    assert m["n_tickets"] == 2 and m["n_skipped_no_gold_docs"] == 1
    assert m["overall"]["bm25|text"]["recall_at_5"]["n"] == 2
    assert any("UNVERIFIED" in w for w in m["warnings"]) and any(
        "DRY RUN" in w for w in m["warnings"]
    )
    assert "Retrieval evaluation" in (out / "summary.md").read_text()
