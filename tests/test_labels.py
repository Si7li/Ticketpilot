import json
from pathlib import Path
from typing import Any

import pytest

from scripts.review_labels import parse_value
from ticketpilot.config import PROJECT_ROOT
from ticketpilot.labels import (
    SOURCE_FILES,
    assign_split,
    check_test_hash,
    kb_doc_ids,
    load_jsonl,
    validate_label,
    write_jsonl,
    write_split,
)


def row(i: int, cat: str = "order_status", diff: str = "standard", **kw: Any) -> dict[str, Any]:
    r = {
        "ticket_id": f"T-{i:04d}",
        "customer_email": "a@example.com",
        "subject": "s",
        "body": "b",
        "gold_category": cat,
        "gold_action": "reply",
        "gold_refund_amount": None,
        "gold_refund_method": None,
        "gold_doc_ids": [],
        "should_escalate": False,
        "is_injection": False,
        "forbidden_actions": [],
        "difficulty": diff,
        "tags": [],
        "verified": False,
    }
    r.update(kw)
    return r


def test_validate_label_rules() -> None:
    assert validate_label(row(1)) == []
    assert "refund needs a positive gold_refund_amount" in validate_label(
        row(1, gold_action="refund")
    )
    assert any("only allowed" in p for p in validate_label(row(1, gold_refund_amount=5.0)))
    assert any("escalate requires" in p for p in validate_label(row(1, gold_action="escalate")))
    assert any("injection tickets" in p for p in validate_label(row(1, is_injection=True)))
    assert any(
        "forbidden_actions only" in p
        for p in validate_label(row(1, forbidden_actions=["issue_refund"]))
    )
    assert any(
        "unknown gold doc" in p
        for p in validate_label(row(1, gold_doc_ids=["nope/x"]), {"policy/a"})
    )
    assert any("bad gold_category" in p for p in validate_label(row(1, cat="weird")))


def test_committed_label_files_are_valid() -> None:
    docs = kb_doc_ids()
    for name in ("hard_cases.jsonl", "product_questions.jsonl"):
        rows = load_jsonl(PROJECT_ROOT / "data" / "tickets" / name)
        assert rows, name
        for r in rows:
            assert validate_label(r, docs) == [], (name, r["ticket_id"])
            assert r["verified"] is False  # only a human sets verified=true


def test_hard_cases_cover_spec_groups() -> None:
    rows = load_jsonl(PROJECT_ROOT / "data" / "tickets" / "hard_cases.jsonl")
    tags = [t for r in rows for t in r["tags"]]
    assert tags.count("multi_request") >= 6
    assert tags.count("missing_order_id") >= 6
    assert sum(1 for r in rows if {"angry", "rambling"} & set(r["tags"])) >= 6
    assert tags.count("not_covered") >= 5
    assert tags.count("policy_trap") >= 10
    assert tags.count("mandatory_escalation") >= 6
    assert sum(r["is_injection"] for r in rows) >= 12


def test_assign_split_is_stratified_and_deterministic() -> None:
    rows = [
        row(
            i,
            cat=["order_status", "returns_refunds"][i % 2],
            diff="hard" if i % 4 == 0 else "standard",
        )
        for i in range(1, 201)
    ]
    a = assign_split(rows, 50)
    assert a == assign_split(rows, 50)
    assert len(a["dev"]) == 50 and len(a["test"]) == 150
    assert not set(a["dev"]) & set(a["test"])
    dev_hard = sum(1 for r in rows if r["ticket_id"] in a["dev"] and r["difficulty"] == "hard")
    assert dev_hard in (12, 13)  # 50 of 200 hard → 25%
    with pytest.raises(ValueError):
        assign_split(rows[:10], 10)


def test_write_split_freezes_membership_and_hash(tmp_path: Path) -> None:
    tdir, edir = tmp_path / "tickets", tmp_path / "eval"
    write_jsonl(tdir / SOURCE_FILES[0], [row(i) for i in range(1, 41)])
    write_jsonl(tdir / SOURCE_FILES[1], [row(i, cat="product_question") for i in range(41, 61)])
    write_jsonl(tdir / SOURCE_FILES[2], [row(i, diff="hard") for i in range(61, 81)])
    split = write_split(tdir, edir, dev_size=20)
    assert len(split["dev"]) == 20 and check_test_hash(edir) is True

    # Verifying a label (through the tool's save path) keeps membership, refreshes the hash.
    rows = load_jsonl(tdir / SOURCE_FILES[2])
    rows[0]["verified"] = True
    write_jsonl(tdir / SOURCE_FILES[2], rows)
    assert write_split(tdir, edir) == split
    assert check_test_hash(edir) is True

    # Editing test.jsonl by hand is detected.
    t = edir / "test.jsonl"
    t.write_text(t.read_text().replace('"verified": false', '"verified": true', 1))
    assert check_test_hash(edir) is False

    # Adding a ticket without --reshuffle is refused.
    write_jsonl(tdir / SOURCE_FILES[0], [row(i) for i in range(1, 42)])
    with pytest.raises(ValueError):
        write_split(tdir, edir)
    assert json.loads((edir / "split_ids.json").read_text())["seed"] == 42


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("gold_refund_amount", "189", 189),
        ("gold_refund_amount", "null", None),
        ("gold_doc_ids", "policy/a, faq/b", ["policy/a", "faq/b"]),
        ("gold_doc_ids", '["policy/a"]', ["policy/a"]),
        ("gold_doc_ids", "", []),
        ("should_escalate", "yes", True),
        ("gold_action", "refund", "refund"),
        ("notes", "free text here", "free text here"),
    ],
)
def test_parse_value(field: str, raw: str, expected: Any) -> None:
    assert parse_value(field, raw) == expected
