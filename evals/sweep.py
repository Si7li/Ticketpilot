"""Confidence-threshold sweep (PROJECT_SPEC.md §15.6). No LLM calls.

    python -m evals.sweep --run evals/reports/<dev run>

Re-runs `choose_route` over stored results for thresholds 0.50–0.95 and reports auto-send
rate vs auto-send precision (+ PNG). Recommends the lowest threshold with precision ≥ 0.95
on the DEV split; fix that value in config before the test run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.common import write_json
from evals.metrics import compute_metrics
from ticketpilot.config import get_settings
from ticketpilot.graph.routing import choose_route
from ticketpilot.labels import EVAL_DIR, load_split

THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
TARGET_PRECISION = 0.95


def sweep(
    results: list[dict[str, Any]], gold: dict[str, dict[str, Any]], judge: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    base = get_settings()
    rows = []
    for th in THRESHOLDS:
        s = base.model_copy(update={"CONFIDENCE_THRESHOLD": th})
        routes = {r["ticket_id"]: choose_route(r, s) for r in results}
        m = compute_metrics(results, gold, judge, route_override=routes)
        rows.append(
            {
                "threshold": th,
                "auto_send_rate": m["auto_send_rate"],
                "auto_send_precision": m["auto_send_precision"],
                "human_review_rate": m["route_shares"].get("human_review"),
                "action_accuracy": m["action_accuracy"],
            }
        )
    return rows


def recommend(rows: list[dict[str, Any]]) -> float | None:
    for r in rows:
        p = r["auto_send_precision"]
        if p["n"] > 0 and p["rate"] is not None and p["rate"] >= TARGET_PRECISION:
            return float(r["threshold"])
    return None


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = [r["threshold"] for r in rows]
    rate = [r["auto_send_rate"]["rate"] or 0 for r in rows]
    prec = [
        r["auto_send_precision"]["rate"] if r["auto_send_precision"]["n"] else float("nan")
        for r in rows
    ]
    lo = [
        r["auto_send_precision"]["ci95"][0] if r["auto_send_precision"]["n"] else float("nan")
        for r in rows
    ]
    hi = [
        r["auto_send_precision"]["ci95"][1] if r["auto_send_precision"]["n"] else float("nan")
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(th, rate, marker="o", label="auto-send rate")
    ax.plot(th, prec, marker="s", label="auto-send precision")
    ax.fill_between(th, lo, hi, alpha=0.15, label="precision 95% CI")
    ax.axhline(
        TARGET_PRECISION,
        linestyle="--",
        linewidth=1,
        color="grey",
        label=f"target {TARGET_PRECISION}",
    )
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("Auto-send rate vs precision")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(run_dir: Path, eval_dir: Path = EVAL_DIR) -> dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text())
    split = cfg["split"]
    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    judge_path = run_dir / "judge.json"
    judge = json.loads(judge_path.read_text()) if judge_path.exists() else {}
    gold = {t["ticket_id"]: t for t in load_split(split, eval_dir)}
    rows = sweep(results, gold, judge)
    rec = recommend(rows) if split == "dev" else None
    out = {
        "split": split,
        "target_precision": TARGET_PRECISION,
        "rows": rows,
        "recommended_threshold": rec,
        "judge_available": bool(judge),
    }
    write_json(run_dir / "sweep.json", out)
    plot(rows, run_dir / "sweep.png")
    lines = [
        f"# Threshold sweep — {split} split ({run_dir.name})\n",
        "| threshold | auto-send rate | auto-send precision | action accuracy |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.2f} | {r['auto_send_rate']['display']} | "
            f"{r['auto_send_precision']['display']} | {r['action_accuracy']['display']} |"
        )
    if not judge:
        lines.append(
            "\n**No judge verdicts in this run: precision cannot be computed (all auto-sends count as unverified).**"
        )
    if split != "dev":
        lines.append("\nNo recommendation: thresholds are chosen on the dev split only.")
    elif rec is None:
        lines.append(f"\nNo threshold reaches precision ≥ {TARGET_PRECISION} on dev.")
    else:
        lines.append(
            f"\n**Recommended:** CONFIDENCE_THRESHOLD={rec:.2f} (lowest with precision ≥ {TARGET_PRECISION} "
            "on dev). Set it in `.env` before the test run."
        )
    (run_dir / "sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nPlot: {run_dir / 'sweep.png'}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    run(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
