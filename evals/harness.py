"""Evaluation harness (PROJECT_SPEC.md §15.1).

    python -m evals.harness --split dev [--limit N] [--retrieval-mode bm25|vector|hybrid]
                            [--no-validation] [--tag name] [--no-judge] [--concurrency 4]
    python -m evals.harness --split test --i-am-reporting

- Refuses --split test without --i-am-reporting (never tune on test).
- Warns at the top of the report if labels are unverified or test.sha256 changed.
- Each ticket gets a fresh copy of the seeded business DB, so tickets cannot affect each other.
- Runs the real graph with an in-memory checkpointer; tickets that pause at human review are
  recorded at the interrupt (never auto-resumed).
- Writes results.jsonl, metrics.json, summary.md, failures.md, config.json; copies
  summary.md to evals/reports/latest/ (for real, non-dry runs).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from evals.common import (
    label_warnings,
    new_report_dir,
    print_warnings,
    publish_latest,
    run_config,
    warnings_md,
    write_json,
)
from evals.judge import Judge
from evals.metrics import action_correct, compute_metrics, pred_action, pred_refund, tokens_by_model
from ticketpilot.config import PROJECT_ROOT, Settings, get_settings
from ticketpilot.db import Database
from ticketpilot.graph.build import build_graph, run_ticket, total_ms
from ticketpilot.graph.nodes import Deps
from ticketpilot.labels import EVAL_DIR, load_split
from ticketpilot.llm.backend import LLMBackend
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.retrieval.vector import Embedder
from ticketpilot.seed import seed_database
from ticketpilot.telemetry import configure_logging, cost_usd
from ticketpilot.tools.registry import build_registry

TRANSIENT_MARKERS = (
    "RateLimit",
    "Overloaded",
    "APIConnection",
    "APITimeout",
    "InternalServer",
    "529",
)
MAX_TICKET_ATTEMPTS = 3
CALIBRATION_SCORES = PROJECT_ROOT / "data" / "eval" / "judge_calibration_scores.json"
MANUAL_TIMING = PROJECT_ROOT / "data" / "eval" / "manual_timing.csv"


def result_record(
    ticket: dict[str, Any], state: dict[str, Any], interrupted: bool, attempts: int
) -> dict[str, Any]:
    route = state.get("route") or ("human_review" if interrupted else None)
    return {
        "ticket_id": ticket["ticket_id"],
        "customer_email": ticket["customer_email"],
        "clean_text": state.get("clean_text"),
        "injection_flag": bool(state.get("injection_flag")),
        "injection_reasons": state.get("injection_reasons", []),
        "category": state.get("category"),
        "confidence": state.get("confidence"),
        "classification_reasoning": state.get("classification_reasoning"),
        "entities": state.get("entities"),
        "retrieved": [
            {k: c.get(k) for k in ("doc_id", "chunk_id", "section", "text", "score", "ranks")}
            for c in state.get("retrieved", [])
        ],
        "tool_results": state.get("tool_results", []),
        "draft_reply": state.get("draft_reply"),
        "citations": state.get("citations", []),
        "proposed_action": state.get("proposed_action"),
        "validation": state.get("validation", {}),
        "route": route,
        "interrupted": interrupted,
        "timings_ms": state.get("timings_ms", {}),
        "total_ms": total_ms(state),
        "usage": state.get("usage", {}),
        "errors": state.get("errors", []),
        "attempts": attempts,
    }


def is_transient(rec: dict[str, Any]) -> bool:
    return any(m in e for e in rec.get("errors", []) for m in TRANSIENT_MARKERS)


class Harness:
    def __init__(
        self, settings: Settings, backend: LLMBackend, retriever: HybridRetriever, workdir: Path
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.retriever = retriever
        self.registry = build_registry()
        self.workdir = workdir
        self.template = workdir / "template.db"
        seed_database(self.template, settings.REFERENCE_DATE)
        self._lock = threading.Lock()
        self.done = 0

    def run_one(self, ticket: dict[str, Any]) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        for attempt in range(1, MAX_TICKET_ATTEMPTS + 1):
            db_path = self.workdir / f"{ticket['ticket_id']}-{attempt}.db"
            shutil.copy(self.template, db_path)
            deps = Deps(
                backend=self.backend,
                retriever=self.retriever,
                registry=self.registry,
                db=Database(db_path),
                settings=self.settings,
            )
            graph = build_graph(deps, InMemorySaver())
            deps.db.create_ticket(
                ticket["ticket_id"], ticket["customer_email"], ticket.get("subject"), ticket["body"]
            )
            out = run_ticket(graph, ticket)
            rec = result_record(ticket, out.state, out.interrupted, attempt)
            db_path.unlink(missing_ok=True)
            if not is_transient(rec) or attempt == MAX_TICKET_ATTEMPTS:
                break
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))  # backoff on rate limits / overload
        with self._lock:
            self.done += 1
        return rec

    def run_all(
        self,
        tickets: list[dict[str, Any]],
        concurrency: int,
        progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(self.run_one, t): t["ticket_id"] for t in tickets}
            for fut in as_completed(futures):
                rec = fut.result()
                results[rec["ticket_id"]] = rec
                if progress:
                    progress(len(results), len(tickets), rec)
        return [results[t["ticket_id"]] for t in tickets]


# ---------------------------------------------------------------------------
# Time saved (§15.8)
# ---------------------------------------------------------------------------


def time_saved(route_shares: dict[str, Any], business_db: Path) -> dict[str, Any]:
    if not MANUAL_TIMING.exists():
        return {
            "measured": False,
            "reason": "data/eval/manual_timing.csv missing (run scripts/time_manual.py)",
        }
    with MANUAL_TIMING.open() as f:
        manual = [float(r["seconds"]) for r in csv.DictReader(f) if r.get("seconds")]
    review: list[float] = []
    if business_db.exists():
        review = [
            float(e["duration_seconds"])
            for e in Database(business_db).list_review_events()
            if e.get("duration_seconds") is not None
        ]
    if not manual or not review:
        return {
            "measured": False,
            "reason": f"need both timings (manual n={len(manual)}, review n={len(review)})",
        }
    m_mean = sum(manual) / len(manual)
    r_mean = sum(review) / len(review)
    share = {k: (v.get("rate") or 0.0) for k, v in route_shares.items()}
    without = 100 * m_mean / 60
    # auto_send: no human time; human_review: measured review time; escalate: handled manually.
    with_tp = 100 * (share.get("human_review", 0) * r_mean + share.get("escalate", 0) * m_mean) / 60
    return {
        "measured": True,
        "manual_seconds_mean": m_mean,
        "manual_n": len(manual),
        "review_seconds_mean": r_mean,
        "review_n": len(review),
        "minutes_per_100_without": without,
        "minutes_per_100_with": with_tp,
        "minutes_saved_per_100": without - with_tp,
        "assumption": "auto-sent tickets take 0 human minutes; escalated tickets take manual time",
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def failure_groups(
    results: list[dict[str, Any]], gold: dict[str, dict[str, Any]], judge: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        g = gold[r["ticket_id"]]
        j = judge.get(r["ticket_id"])
        problems = []
        if r.get("category") != g["gold_category"]:
            problems.append("category")
        if not action_correct(r, g):
            problems.append("action")
        if j and any(not c["supported"] for c in j.get("claims", [])):
            problems.append("hallucination")
        if g.get("should_escalate") and r.get("route") != "escalate":
            problems.append("missed_escalation")
        if g.get("is_injection") and any(
            t.get("ok") and t.get("tool") in g.get("forbidden_actions", [])
            for t in r.get("tool_results", [])
        ):
            problems.append("unauthorized_action")
        if r.get("errors"):
            problems.append("errors")
        if not problems:
            continue
        key = ", ".join(problems) + " · " + (", ".join(sorted(g.get("tags", []))[:3]) or "untagged")
        groups[key].append({"r": r, "g": g, "j": j, "problems": problems})
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def failures_markdown(groups: dict[str, list[dict[str, Any]]], split: str) -> str:
    lines = [
        f"# Failures — {split} split\n",
        "Grouped by failure type · auto-detected tags. Every group ends with a placeholder for "
        "**human** analysis (do not let a model fill it in).\n",
    ]
    if not groups:
        lines.append("No failures.\n")
    for key, items in groups.items():
        lines.append(f"\n## {key} ({len(items)})\n")
        for it in items:
            r, g, j = it["r"], it["g"], it["j"]
            amount, method = pred_refund(r)
            lines.append(f"### {r['ticket_id']} — {', '.join(it['problems'])}")
            lines.append(
                f"- category: gold `{g['gold_category']}` · pred `{r.get('category')}` (conf {r.get('confidence')})"
            )
            lines.append(
                f"- action: gold `{g['gold_action']}`"
                + (
                    f" ${g['gold_refund_amount']} {g['gold_refund_method']}"
                    if g["gold_action"] == "refund"
                    else ""
                )
                + f" · pred `{pred_action(r)}`"
                + (f" ${amount} {method}" if amount is not None else "")
            )
            lines.append(
                f"- route: `{r.get('route')}` · should_escalate={g.get('should_escalate')} · injection_flag={r.get('injection_flag')}"
            )
            tools = "; ".join(
                f"{t.get('tool')}({json.dumps(t.get('args'))})→{'ok' if t.get('ok') else t.get('error')}"
                for t in r.get("tool_results", [])
            )
            lines.append(f"- tools: {tools or 'none'}")
            if j:
                bad = [c["claim"] for c in j.get("claims", []) if not c["supported"]]
                if bad:
                    lines.append(f"- judge unsupported: {bad}")
                if j.get("followed_injected_instruction"):
                    lines.append("- judge: reply followed injected instruction")
            if r.get("errors"):
                lines.append(f"- errors: {r['errors']}")
            if g.get("notes"):
                lines.append(f"- label notes: {g['notes']}")
            lines.append("")
        lines.append("**Human analysis:** _TBD_\n")
    return "\n".join(lines)


def summary_markdown(
    split: str, m: dict[str, Any], warnings: list[str], cfg: dict[str, Any]
) -> str:
    def row(name: str, key: str) -> str:
        return f"| {name} | {m[key]['display']} |"

    lines = [f"# TicketPilot evaluation — {split} split\n", warnings_md(warnings)]
    lines.append(
        f"- tickets: {m['n_tickets']} · labels verified: {m['labels_verified']}/{m['n_tickets']}"
    )
    lines.append(
        f"- agent `{cfg['agent_model']}` · judge `{cfg['judge_model']}` · prompts `{cfg['prompt_version']}`"
        f" · retrieval `{cfg['settings']['RETRIEVAL_MODE']}` · validation "
        f"{cfg['settings']['VALIDATION_ENABLED']} · threshold {cfg['settings']['CONFIDENCE_THRESHOLD']}"
    )
    lines.append(f"- commit `{cfg['git_commit']}` · {cfg['created_at']}")
    kappa = (m.get("judge_calibration") or {}).get("kappa")
    lines.append(
        "- judge calibration: "
        + (
            f"κ = {kappa:.3f} (n={m['judge_calibration']['n']})"
            if kappa is not None
            else "not calibrated yet — hallucination numbers are unvalidated"
        )
    )
    lines += ["", "| metric | value |", "|---|---|"]
    for name, key in [
        ("Category accuracy", "category_accuracy"),
        ("Action accuracy (incl. refund details)", "action_accuracy"),
        ("Refund amount accuracy", "refund_amount_accuracy"),
        ("Tool-call validity", "tool_call_validity"),
        ("Retrieval recall@5", "retrieval_recall_at_5"),
        ("All-gold recall@5", "retrieval_all_gold_recall_at_5"),
        ("Hallucination rate (tickets, judge)", "hallucination_rate"),
        ("Hallucination rate (claims, judge)", "hallucination_claim_rate"),
        ("Escalation precision", "escalation_precision"),
        ("Escalation recall", "escalation_recall"),
        ("Auto-send rate", "auto_send_rate"),
        ("Auto-send precision", "auto_send_precision"),
        ("Injection: unauthorized actions", "injection_unauthorized_rate"),
        ("Injection: detected by heuristic", "injection_detected_rate"),
        ("Injection: echo (judge)", "injection_echo_rate"),
        ("Tickets with errors", "tickets_with_errors"),
    ]:
        lines.append(row(name, key))
    mrr = m.get("retrieval_mrr")
    lines.append(f"| Retrieval MRR | {mrr:.3f} |" if mrr is not None else "| Retrieval MRR | n/a |")
    lat = m["latency_ms"]
    if lat["p50"] is not None:
        lines.append(f"| Latency p50 / p95 | {lat['p50'] / 1000:.1f}s / {lat['p95'] / 1000:.1f}s |")
    lines += [
        "",
        "## By difficulty",
        "",
        "| difficulty | category acc | action acc |",
        "|---|---|---|",
    ]
    for d, v in m["category_accuracy_by_difficulty"].items():
        lines.append(
            f"| {d} | {v['display']} | {m['action_accuracy_by_difficulty'][d]['display']} |"
        )
    lines += ["", "## Category confusion (rows = gold, cols = predicted)", ""]
    cols = (
        list(next(iter(m["category_confusion"].values())).keys()) if m["category_confusion"] else []
    )
    lines.append("| gold \\ pred | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for gcat, preds in m["category_confusion"].items():
        lines.append(f"| {gcat} | " + " | ".join(str(preds[c]) for c in cols) + " |")
    lines += ["", "## Routes", ""] + [
        f"- {k}: {v['display']}" for k, v in m["route_shares"].items()
    ]
    lines += ["", "## Cost", ""]
    for model, t in m["tokens_by_model"].items():
        lines.append(
            f"- {model}: {t['input_tokens']:.0f} in / {t['output_tokens']:.0f} out tokens "
            f"({t['input_per_ticket']:.0f} / {t['output_per_ticket']:.0f} per ticket)"
        )
    lines.append(
        f"- agent cost: {'$' + format(m['cost_usd_agent'], '.4f') if m['cost_usd_agent'] is not None else 'prices not configured (tokens only)'}"
    )
    lines.append(
        f"- judge cost: {'$' + format(m['cost_usd_judge'], '.4f') if m['cost_usd_judge'] is not None else 'prices not configured (tokens only)'}"
    )
    ts = m["time_saved"]
    lines += ["", "## Time saved", ""]
    if ts["measured"]:
        lines.append(
            f"- without TicketPilot: {ts['minutes_per_100_without']:.0f} min / 100 tickets "
            f"(manual mean {ts['manual_seconds_mean']:.0f}s, n={ts['manual_n']})"
        )
        lines.append(
            f"- with TicketPilot: {ts['minutes_per_100_with']:.0f} min / 100 tickets "
            f"(review mean {ts['review_seconds_mean']:.0f}s, n={ts['review_n']}; {ts['assumption']})"
        )
    else:
        lines.append(f"- time saved: not measured ({ts['reason']})")
    lines.append("\n_Rates: count/total (%) with 95% Wilson score intervals._\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run(
    split: str,
    *,
    limit: int | None = None,
    retrieval_mode: str | None = None,
    no_validation: bool = False,
    tag: str | None = None,
    use_judge: bool = True,
    concurrency: int | None = None,
    eval_dir: Path = EVAL_DIR,
    backend: LLMBackend | None = None,
    embedder: Embedder | None = None,
    judge: Judge | None = None,
    publish: bool = True,
    label: str = "",
) -> Path:
    overrides: dict[str, Any] = {}
    if retrieval_mode:
        overrides["RETRIEVAL_MODE"] = retrieval_mode
    if no_validation:
        overrides["VALIDATION_ENABLED"] = False
    settings = get_settings().model_copy(update=overrides)
    configure_logging(settings.model_copy(update={"LOG_LEVEL": "WARNING"}))

    tickets = load_split(split, eval_dir)
    if tag:
        tickets = [t for t in tickets if tag in t.get("tags", [])]
    if limit:
        tickets = tickets[:limit]
    warnings = label_warnings(tickets, split, eval_dir)
    dry_run = eval_dir.resolve() != EVAL_DIR.resolve() or limit is not None or tag is not None
    if eval_dir.resolve() != EVAL_DIR.resolve():
        warnings.append(f"DRY RUN on non-standard eval dir {eval_dir}; not a reportable result.")
    if limit is not None or tag is not None:
        warnings.append(
            f"Partial run (limit={limit}, tag={tag}); not comparable to full-split runs."
        )
    print_warnings(warnings)

    if backend is None:
        from ticketpilot.llm.anthropic_backend import AnthropicBackend

        backend = AnthropicBackend(settings)
    retriever = HybridRetriever.from_settings(settings, embedder=embedder)
    gold = {t["ticket_id"]: t for t in tickets}
    kind = f"{split}" + (f"-{label}" if label else "") + ("-dryrun" if dry_run else "")
    out = new_report_dir(kind)

    with tempfile.TemporaryDirectory(prefix="tp-eval-") as tmp:
        h = Harness(settings, backend, retriever, Path(tmp))

        def progress(i: int, n: int, rec: dict[str, Any]) -> None:
            print(
                f"  [{i}/{n}] {rec['ticket_id']} {rec['route']} cat={rec['category']} "
                f"{rec['total_ms'] / 1000:.1f}s{' ERR' if rec['errors'] else ''}"
            )

        t0 = time.perf_counter()
        results = h.run_all(tickets, concurrency or settings.EVAL_CONCURRENCY, progress)
        wall = time.perf_counter() - t0

    with (out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    verdicts: dict[str, dict[str, Any]] = {}
    if use_judge:
        judge = judge or Judge(settings)
        print(f"Judging {len(results)} replies with {judge.model} …")
        with ThreadPoolExecutor(max_workers=concurrency or settings.EVAL_CONCURRENCY) as pool:
            futs = {
                pool.submit(judge.judge, r, bool(gold[r["ticket_id"]].get("is_injection"))): r[
                    "ticket_id"
                ]
                for r in results
            }
            for fut in as_completed(futs):
                v = fut.result()
                if v is not None:
                    verdicts[futs[fut]] = v
        write_json(out / "judge.json", verdicts)
        print(f"  judge: {judge.calls} new calls, {judge.cache_hits} cache hits")
    else:
        warnings.append(
            "Judge disabled (--no-judge): hallucination, auto-send precision and echo "
            "metrics are not measured."
        )

    metrics = compute_metrics(results, gold, verdicts)
    metrics["labels_verified"] = sum(1 for t in tickets if t.get("verified"))
    metrics["tokens_by_model"] = tokens_by_model(results)
    metrics["cost_usd_agent"] = cost_usd(metrics["tokens_by_model"], settings) if results else None
    judge_tokens: dict[str, dict[str, int]] = {}
    if use_judge and judge is not None:
        for v in judge.usage.totals().values():
            t = judge_tokens.setdefault(judge.model, {"input_tokens": 0, "output_tokens": 0})
            t["input_tokens"] += v["input_tokens"]
            t["output_tokens"] += v["output_tokens"]
    metrics["judge_tokens_this_run"] = judge_tokens
    metrics["cost_usd_judge"] = cost_usd(judge_tokens, settings) if judge_tokens else None
    metrics["judge_calibration"] = (
        json.loads(CALIBRATION_SCORES.read_text()) if CALIBRATION_SCORES.exists() else None
    )
    metrics["time_saved"] = time_saved(metrics["route_shares"], settings.business_db_path)
    metrics["wall_seconds"] = wall
    metrics["warnings"] = warnings

    cfg = run_config(
        settings,
        split=split,
        kind="harness",
        n_tickets=len(tickets),
        limit=limit,
        tag=tag,
        use_judge=use_judge,
        dry_run=dry_run,
        reporting=split == "test",
        label=label,
    )
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", cfg)
    (out / "summary.md").write_text(summary_markdown(split, metrics, warnings, cfg))
    (out / "failures.md").write_text(
        failures_markdown(failure_groups(results, gold, verdicts), split)
    )
    if publish and not dry_run:
        publish_latest(out / "summary.md")
    print((out / "summary.md").read_text())
    print(f"Report: {out}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--i-am-reporting", action="store_true", help="required for --split test")
    p.add_argument("--limit", type=int)
    p.add_argument("--retrieval-mode", choices=["bm25", "vector", "hybrid"])
    p.add_argument("--no-validation", action="store_true")
    p.add_argument("--tag")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = p.parse_args()
    if args.split == "test" and not args.i_am_reporting:
        p.error("--split test is for reporting only (never tune on test); pass --i-am-reporting")
    run(
        args.split,
        limit=args.limit,
        retrieval_mode=args.retrieval_mode,
        no_validation=args.no_validation,
        tag=args.tag,
        use_judge=not args.no_judge,
        concurrency=args.concurrency,
        eval_dir=args.eval_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
