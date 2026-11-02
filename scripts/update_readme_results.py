"""Insert real eval results into README.md between markers. The ONLY way numbers get there.

    python scripts/update_readme_results.py

RESULTS      ← latest harness run on the test split (non-dry-run, config.reporting=true)
RETRIEVAL    ← latest retrieval eval on the test split
ABLATIONS    ← latest ablations report on the test split
Each section includes its own heading; if no qualifying run exists the section stays empty
(nothing is shown in the README).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from ticketpilot.config import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
REPORTS = PROJECT_ROOT / "evals" / "reports"


def _latest(pred: Any) -> Path | None:
    runs = sorted((d for d in REPORTS.iterdir() if d.is_dir() and d.name != "latest"), reverse=True)
    for d in runs:
        cfg_path = d / "config.json"
        if cfg_path.exists() and (d / "metrics.json").exists():
            cfg = json.loads(cfg_path.read_text())
            if pred(d, cfg):
                return d
    return None


def latest_test_run() -> Path | None:
    return _latest(
        lambda d, c: (
            c.get("kind") == "harness"
            and c.get("split") == "test"
            and not c.get("dry_run")
            and not c.get("label")
        )
    )


def latest_retrieval() -> Path | None:
    return _latest(
        lambda d, c: (
            c.get("kind") == "retrieval" and c.get("split") == "test" and "dryrun" not in d.name
        )
    )


def latest_ablations() -> Path | None:
    for d in sorted((x for x in REPORTS.glob("*-ablations-test") if x.is_dir()), reverse=True):
        return d
    return None


def results_block(run: Path | None) -> str:
    if run is None:
        return ""
    m = json.loads((run / "metrics.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    kappa = (m.get("judge_calibration") or {}).get("kappa")
    lines = [
        "## Results",
        "",
        f"Test split: **{m['n_tickets']} tickets**, labels human-verified: "
        f"**{m['labels_verified']}/{m['n_tickets']}**. Judge `{cfg['judge_model']}` calibration "
        f"κ = **{kappa:.3f}**."
        if kappa is not None
        else f"Test split: **{m['n_tickets']} tickets**, labels human-verified: "
        f"**{m['labels_verified']}/{m['n_tickets']}**. Judge calibration: **not measured**.",
        f"Run `{run.name}`, commit `{cfg['git_commit']}`, {cfg['created_at']}, agent `{cfg['agent_model']}`, "
        f"prompts `{cfg['prompt_version']}`, threshold {cfg['settings']['CONFIDENCE_THRESHOLD']}.",
        "",
    ]
    if m.get("warnings"):
        lines += [f"> ⚠️ {w}" for w in m["warnings"]] + [""]
    lines += ["| metric | value |", "|---|---|"]
    for name, key in [
        ("Category accuracy", "category_accuracy"),
        ("Action accuracy", "action_accuracy"),
        ("Refund amount accuracy", "refund_amount_accuracy"),
        ("Tool-call validity", "tool_call_validity"),
        ("Retrieval recall@5", "retrieval_recall_at_5"),
        ("Hallucination rate (judge)", "hallucination_rate"),
        ("Escalation precision", "escalation_precision"),
        ("Escalation recall", "escalation_recall"),
        ("Auto-send rate", "auto_send_rate"),
        ("Auto-send precision", "auto_send_precision"),
        ("Injection: unauthorized actions", "injection_unauthorized_rate"),
        ("Injection: echo (judge)", "injection_echo_rate"),
    ]:
        lines.append(f"| {name} | {m[key]['display']} |")
    lat = m["latency_ms"]
    if lat.get("p50") is not None:
        lines.append(f"| Latency p50 / p95 | {lat['p50'] / 1000:.1f}s / {lat['p95'] / 1000:.1f}s |")
    ts = m.get("time_saved", {})
    lines.append(
        f"| Time saved | {ts['minutes_per_100_without'] - ts['minutes_per_100_with']:.0f} min per 100 tickets "
        f"(manual n={ts['manual_n']}, review n={ts['review_n']}) |"
        if ts.get("measured")
        else "| Time saved | not measured |"
    )
    return "\n".join(lines)


def retrieval_block(run: Path | None) -> str:
    if run is None:
        return ""
    m = json.loads((run / "metrics.json").read_text())
    lines = [
        "### Retrieval comparison (BM25 vs vector vs hybrid)",
        "",
        f"Run `{run.name}` ({m['n_tickets']} test tickets with gold docs), query = text + category.",
        "",
        "| mode | recall@5 | all-gold recall@5 | MRR |",
        "|---|---|---|---|",
    ]
    for mode in ("bm25", "vector", "hybrid"):
        x = m["overall"][f"{mode}|text+cat"]
        lines.append(
            f"| {mode} | {x['recall_at_5']['display']} | {x['all_gold_recall_at_5']['display']} | {x['mrr']:.3f} |"
        )
    return "\n".join(lines)


def ablations_block(run: Path | None) -> str:
    if run is None:
        return ""
    text = (run / "summary.md").read_text()
    return "### Ablations\n\n" + text.split("\n", 2)[2].strip()


def replace_block(readme: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.DOTALL)
    if not pattern.search(readme):
        raise SystemExit(f"README is missing the <!-- {name}:START/END --> markers")
    inner = f"\n{body}\n" if body else "\n"
    return pattern.sub(lambda m: f"{m.group(1)}{inner}{m.group(3)}", readme)


def main() -> int:
    readme = README.read_text()
    readme = replace_block(readme, "RESULTS", results_block(latest_test_run()))
    readme = replace_block(readme, "RETRIEVAL", retrieval_block(latest_retrieval()))
    readme = replace_block(readme, "ABLATIONS", ablations_block(latest_ablations()))
    README.write_text(readme)
    print("README results sections updated.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
