"""Ablations (PROJECT_SPEC.md §15.7).

    python -m evals.ablations --split dev
    python -m evals.ablations --split test --i-am-reporting

Runs the harness for: hybrid+validation (full), vector+validation, bm25+validation,
hybrid+no-validation; writes one comparison table. Differences whose 95% intervals overlap
are flagged as possibly noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals import harness
from evals.common import new_report_dir, write_json

CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("hybrid+validation", {"retrieval_mode": "hybrid", "no_validation": False}),
    ("vector+validation", {"retrieval_mode": "vector", "no_validation": False}),
    ("bm25+validation", {"retrieval_mode": "bm25", "no_validation": False}),
    ("hybrid+no-validation", {"retrieval_mode": "hybrid", "no_validation": True}),
]
METRICS = [
    ("Action accuracy", "action_accuracy"),
    ("Category accuracy", "category_accuracy"),
    ("Retrieval recall@5", "retrieval_recall_at_5"),
    ("Hallucination rate", "hallucination_rate"),
    ("Auto-send rate", "auto_send_rate"),
    ("Auto-send precision", "auto_send_precision"),
    ("Escalation recall", "escalation_recall"),
]


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if not a["n"] or not b["n"]:
        return True
    return not (a["ci95"][1] < b["ci95"][0] or b["ci95"][1] < a["ci95"][0])


def table(results: dict[str, dict[str, Any]]) -> str:
    names = [n for n, _ in CONFIGS if n in results]
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    notes = []
    base = results.get("hybrid+validation")
    for label, key in METRICS:
        cells = []
        for n in names:
            m = results[n][key]
            cells.append(m["display"])
            if base and n != "hybrid+validation" and overlaps(m, base[key]):
                notes.append(
                    f"{label}: {n} vs full — 95% intervals overlap; difference may be noise."
                )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return (
        "\n".join(lines) + ("\n\n" + "\n".join(f"- ⚠️ {x}" for x in notes) if notes else "") + "\n"
    )


def run(split: str, **kw: Any) -> Path:
    results: dict[str, dict[str, Any]] = {}
    runs: dict[str, str] = {}
    for name, cfg in CONFIGS:
        print(f"\n=== {name} ===")
        out = harness.run(split, label=name.replace("+", "_"), publish=False, **cfg, **kw)
        results[name] = json.loads((out / "metrics.json").read_text())
        runs[name] = str(out)
    d = new_report_dir(f"ablations-{split}")
    md = (
        f"# Ablations — {split} split\n\n"
        + table(results)
        + "\nRuns:\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in runs.items())
        + "\n"
    )
    (d / "summary.md").write_text(md)
    write_json(
        d / "metrics.json",
        {
            "split": split,
            "runs": runs,
            "metrics": {k: {m: v[m] for _, m in METRICS} for k, v in results.items()},
        },
    )
    print(md)
    return d


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--i-am-reporting", action="store_true")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.split == "test" and not args.i_am_reporting:
        p.error("--split test is for reporting only; pass --i-am-reporting")
    run(args.split, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
