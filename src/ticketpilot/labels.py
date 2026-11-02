"""Label files: loading, validation, and the frozen dev/test split (PROJECT_SPEC.md §7).

Source of truth for labels: data/tickets/{standard_tickets,product_questions,hard_cases}.jsonl.
The split *membership* is frozen in data/eval/split_ids.json; data/eval/{dev,test}.jsonl are
regenerated from the sources with `write_split()` (the label review tool calls it after every
save), and data/eval/test.sha256 is refreshed at the same time. Any other change to
test.jsonl makes the hash mismatch, which the eval harness reports loudly.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ticketpilot.config import PROJECT_ROOT
from ticketpilot.schemas import CATEGORIES

TICKETS_DIR = PROJECT_ROOT / "data" / "tickets"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SOURCE_FILES = ("standard_tickets.jsonl", "product_questions.jsonl", "hard_cases.jsonl")
ACTIONS = ("reply", "refund", "escalate")
SPLIT_SEED = 42
DEV_SIZE = 50

LABEL_FIELDS = (
    "gold_category",
    "gold_action",
    "gold_refund_amount",
    "gold_refund_method",
    "gold_doc_ids",
    "should_escalate",
    "is_injection",
    "forbidden_actions",
    "difficulty",
    "tags",
    "notes",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_sources(tickets_dir: Path = TICKETS_DIR) -> dict[str, list[dict[str, Any]]]:
    return {name: load_jsonl(tickets_dir / name) for name in SOURCE_FILES}


def all_labeled(tickets_dir: Path = TICKETS_DIR) -> list[dict[str, Any]]:
    rows = [r for rs in load_sources(tickets_dir).values() for r in rs]
    ids = [r["ticket_id"] for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate ticket IDs across label files: {sorted(dupes)}")
    return rows


def kb_doc_ids(kb_dir: Path = PROJECT_ROOT / "data" / "kb") -> set[str]:
    return {p.relative_to(kb_dir).with_suffix("").as_posix() for p in kb_dir.rglob("*.md")}


def validate_label(row: dict[str, Any], doc_ids: set[str] | None = None) -> list[str]:
    """Schema problems with one labeled ticket (empty list = OK)."""
    problems: list[str] = []
    for f in ("ticket_id", "customer_email", "body", "gold_category", "gold_action"):
        if not row.get(f):
            problems.append(f"missing {f}")
    if row.get("gold_category") not in CATEGORIES:
        problems.append(f"bad gold_category {row.get('gold_category')!r}")
    action = row.get("gold_action")
    if action not in ACTIONS:
        problems.append(f"bad gold_action {action!r}")
    if action == "refund":
        if (
            not isinstance(row.get("gold_refund_amount"), int | float)
            or row["gold_refund_amount"] <= 0
        ):
            problems.append("refund needs a positive gold_refund_amount")
        if row.get("gold_refund_method") not in ("original_payment", "store_credit"):
            problems.append("refund needs gold_refund_method")
    elif row.get("gold_refund_amount") is not None or row.get("gold_refund_method") is not None:
        problems.append("gold_refund_* only allowed when gold_action == 'refund'")
    if action == "escalate" and not row.get("should_escalate"):
        problems.append("gold_action escalate requires should_escalate=true")
    if row.get("is_injection") and not row.get("should_escalate"):
        problems.append("injection tickets should escalate")
    if row.get("forbidden_actions") and not row.get("is_injection"):
        problems.append("forbidden_actions only for injection tickets")
    if doc_ids is not None:
        for d in row.get("gold_doc_ids", []):
            if d not in doc_ids:
                problems.append(f"unknown gold doc {d}")
    if row.get("difficulty") not in ("standard", "hard"):
        problems.append("difficulty must be standard|hard")
    return problems


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def stratum(row: dict[str, Any]) -> tuple[str, str, bool]:
    return (row["gold_category"], row["difficulty"], bool(row.get("is_injection")))


def assign_split(
    rows: list[dict[str, Any]], dev_size: int = DEV_SIZE, seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    """Stratified by (category, difficulty, injection) with largest-remainder allocation."""
    if dev_size >= len(rows):
        raise ValueError("dev_size must be smaller than the number of tickets")
    rng = random.Random(seed)
    groups: dict[tuple[str, str, bool], list[str]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["ticket_id"]):
        groups[stratum(r)].append(r["ticket_id"])
    frac = dev_size / len(rows)
    quotas = {k: len(v) * frac for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remaining = dev_size - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), str(k)))[:remaining]:
        alloc[k] += 1
    dev: list[str] = []
    test: list[str] = []
    for k in sorted(groups, key=str):
        ids = list(groups[k])
        rng.shuffle(ids)
        dev.extend(ids[: alloc[k]])
        test.extend(ids[alloc[k] :])
    return {"dev": sorted(dev), "test": sorted(test)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split(
    tickets_dir: Path = TICKETS_DIR,
    eval_dir: Path = EVAL_DIR,
    reshuffle: bool = False,
    dev_size: int = DEV_SIZE,
) -> dict[str, list[str]]:
    rows = all_labeled(tickets_dir)
    by_id = {r["ticket_id"]: r for r in rows}
    ids_path = eval_dir / "split_ids.json"
    if ids_path.exists() and not reshuffle:
        split = json.loads(ids_path.read_text())
        known = set(split["dev"]) | set(split["test"])
        new = sorted(set(by_id) - known)
        missing = sorted(known - set(by_id))
        if new or missing:
            raise ValueError(
                f"label files changed membership (new={new[:5]}…, missing={missing[:5]}…); "
                "re-run split_eval.py --reshuffle deliberately"
            )
    else:
        split = assign_split(rows, dev_size)
        eval_dir.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(json.dumps({**split, "seed": SPLIT_SEED}, indent=1) + "\n")
    for name in ("dev", "test"):
        write_jsonl(eval_dir / f"{name}.jsonl", [by_id[i] for i in split[name]])
    (eval_dir / "test.sha256").write_text(sha256_file(eval_dir / "test.jsonl") + "\n")
    return {"dev": split["dev"], "test": split["test"]}


def load_split(name: str, eval_dir: Path = EVAL_DIR) -> list[dict[str, Any]]:
    path = eval_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/split_eval.py`")
    return load_jsonl(path)


def check_test_hash(eval_dir: Path = EVAL_DIR) -> bool | None:
    """True/False if a stored hash exists; None if there is no hash yet."""
    h = eval_dir / "test.sha256"
    t = eval_dir / "test.jsonl"
    if not h.exists() or not t.exists():
        return None
    return h.read_text().strip() == sha256_file(t)
