from typing import Any

import pytest

from ticketpilot.config import Settings
from ticketpilot.graph.routing import choose_route, review_reasons

OK_VALIDATION = {"grounded": True, "issues": []}


def base(**over: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "category": "order_status",
        "confidence": 0.95,
        "injection_flag": False,
        "tool_results": [],
        "errors": [],
        "validation": dict(OK_VALIDATION),
    }
    s.update(over)
    return s


CASES = [
    # (description, state, expected)
    ("clean ticket", base(), "auto_send"),
    ("sensitive category", base(category="other_sensitive"), "escalate"),
    (
        "sensitive beats high confidence",
        base(category="other_sensitive", confidence=0.99),
        "escalate",
    ),
    ("injection flag", base(injection_flag=True), "escalate"),
    ("injection beats errors", base(injection_flag=True, errors=["x"]), "escalate"),
    ("specialist tool result", base(tool_results=[{"needs_specialist": True}]), "escalate"),
    (
        "specialist beats errors",
        base(tool_results=[{"needs_specialist": True}], errors=["x"]),
        "escalate",
    ),
    ("errors", base(errors=["act: cap"]), "human_review"),
    ("not grounded", base(validation={"grounded": False, "issues": []}), "human_review"),
    (
        "validation issues",
        base(validation={"grounded": True, "issues": ["amount"]}),
        "human_review",
    ),
    ("missing validation", base(validation={}), "human_review"),
    ("no validation key", {k: v for k, v in base().items() if k != "validation"}, "human_review"),
    ("pending refund", base(tool_results=[{"pending": True}]), "human_review"),
    ("low confidence", base(confidence=0.79), "human_review"),
    ("threshold boundary is inclusive", base(confidence=0.80), "auto_send"),
    ("missing confidence", {k: v for k, v in base().items() if k != "confidence"}, "human_review"),
    ("empty state", {}, "human_review"),
]


@pytest.mark.parametrize(("desc", "state", "expected"), CASES, ids=[c[0] for c in CASES])
def test_choose_route(desc: str, state: dict[str, Any], expected: str) -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert choose_route(state, s) == expected


def test_threshold_is_configurable() -> None:
    lo = Settings(_env_file=None, CONFIDENCE_THRESHOLD=0.5)  # type: ignore[call-arg]
    hi = Settings(_env_file=None, CONFIDENCE_THRESHOLD=0.99)  # type: ignore[call-arg]
    st = base(confidence=0.6)
    assert choose_route(st, lo) == "auto_send"
    assert choose_route(st, hi) == "human_review"


def test_extra_sensitive_categories() -> None:
    s = Settings(_env_file=None, SENSITIVE_CATEGORIES="other_sensitive,account_billing")  # type: ignore[call-arg]
    assert choose_route(base(category="account_billing"), s) == "escalate"


def test_review_reasons() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    st = base(
        confidence=0.5,
        tool_results=[{"pending": True}],
        validation={"grounded": False, "issues": ["unsupported claim: x"]},
        errors=["boom"],
    )
    reasons = review_reasons(st, s)
    assert any("pending" in r for r in reasons)
    assert any("low confidence" in r for r in reasons)
    assert any("not fully grounded" in r for r in reasons)
    assert any("unsupported claim" in r for r in reasons)
    assert any("errors" in r for r in reasons)
