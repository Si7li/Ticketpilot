"""Shared helpers for eval scripts: report folders, run config, honesty warnings."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ticketpilot.config import PROJECT_ROOT, Settings
from ticketpilot.labels import EVAL_DIR, check_test_hash
from ticketpilot.llm.prompts import PROMPT_VERSION

REPORTS_DIR = PROJECT_ROOT / "evals" / "reports"
LATEST_DIR = REPORTS_DIR / "latest"
RED, BOLD, RESET = "\033[31m", "\033[1m", "\033[0m"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def new_report_dir(kind: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    d = REPORTS_DIR / f"{stamp}-{kind}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def run_config(settings: Settings, **extra: Any) -> dict[str, Any]:
    cfg = settings.model_dump(mode="json")
    for secret in ("ANTHROPIC_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"):
        cfg[secret] = "***" if cfg.get(secret) else None
    return {
        "settings": cfg,
        "agent_model": settings.AGENT_MODEL,
        "judge_model": settings.JUDGE_MODEL,
        "prompt_version": PROMPT_VERSION,
        "git_commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def label_warnings(rows: list[dict[str, Any]], split: str, eval_dir: Path = EVAL_DIR) -> list[str]:
    warnings: list[str] = []
    unverified = sum(1 for r in rows if not r.get("verified"))
    if unverified:
        warnings.append(
            f"{unverified}/{len(rows)} {split} labels are UNVERIFIED — numbers are not trustworthy "
            "until a human runs `make review-labels`."
        )
    if split == "test" and check_test_hash(eval_dir) is False:
        warnings.append(
            "data/eval/test.jsonl does not match data/eval/test.sha256 — the frozen "
            "test set was modified outside the label tool."
        )
    return warnings


def print_warnings(warnings: list[str]) -> None:
    for w in warnings:
        print(f"{RED}{BOLD}WARNING: {w}{RESET}")


def warnings_md(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return "\n".join(f"> **⚠️ WARNING:** {w}" for w in warnings) + "\n\n"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def publish_latest(summary: Path) -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(summary, LATEST_DIR / "summary.md")
