from pathlib import Path
from typing import Any

import pytest

from tests.graph_helpers import runtime, script
from ticketpilot import cli
from ticketpilot.config import Settings
from ticketpilot.db import Database
from ticketpilot.retrieval.retriever import HybridRetriever


@pytest.fixture
def patched(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, db: Database, retriever: HybridRetriever
) -> Any:
    from tests.test_graph import _over_limit_refund_backend

    rt = runtime(_over_limit_refund_backend(), settings, db, retriever)
    monkeypatch.setattr(cli, "build_runtime", lambda s: rt)
    return rt


def test_find_ticket_from_smoke_file() -> None:
    t = cli.find_ticket("S-0002")
    assert t is not None and t["customer_email"] == "sam.ortiz@example.com"
    assert cli.find_ticket("S-9999") is None


def test_run_and_resume(patched: Any, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run-ticket", "--id", "S-0002"]) == 0
    out = capsys.readouterr().out
    assert "human_review (paused)" in out and "issue_refund" in out and "[pending]" in out
    assert patched.db.get_ticket("S-0002") is not None
    assert cli.main(["resume", "--id", "S-0002", "--action", "approve", "--reviewer", "kim"]) == 0
    assert patched.db.get_active_refund("KO-10482").status == "approved"


def test_run_ad_hoc_and_missing(patched: Any, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run-ticket", "--id", "S-0404"]) == 1
    with pytest.raises(SystemExit):
        cli.main(["run-ticket", "--id", "T-7777", "--body", "hi"])  # --email required


def test_ad_hoc_body(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    db: Database,
    retriever: HybridRetriever,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    rt = runtime(script(), settings, db, retriever)
    monkeypatch.setattr(cli, "build_runtime", lambda s: rt)
    assert (
        cli.main(
            [
                "run-ticket",
                "--id",
                "T-7001",
                "--email",
                "jose.martinez@example.com",
                "--body",
                "Where is my stove?",
            ]
        )
        == 0
    )
    assert "auto_send" in capsys.readouterr().out
