import pytest

from ticketpilot.retrieval.fusion import rrf


def test_spec_example() -> None:
    out = rrf([["A", "B", "X", "D"], ["C", "D", "A"]], k=60)
    assert [i for i, _ in out][:4] == ["A", "D", "C", "B"]
    scores = dict(out)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["D"] == pytest.approx(1 / 64 + 1 / 62)
    assert scores["C"] == pytest.approx(1 / 61)
    assert scores["B"] == pytest.approx(1 / 62)
    assert scores["X"] == pytest.approx(1 / 63)


def test_duplicates_within_a_list_count_once() -> None:
    assert dict(rrf([["A", "A"]]))["A"] == pytest.approx(1 / 61)


def test_empty() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_tie_break_is_deterministic() -> None:
    assert [i for i, _ in rrf([["B"], ["A"]])] == ["A", "B"]
