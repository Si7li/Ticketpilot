"""Graph nodes (PROJECT_SPEC.md §11.2 and §12).

Every node is `fn(state, deps) -> dict`, wrapped by `_node()` which times it, attributes
token usage to it, logs one JSON line, and turns unexpected exceptions into `errors`
(which routes the ticket to human review: fail safe, never fail open).
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from ticketpilot.config import Settings
from ticketpilot.db import Database
from ticketpilot.graph.routing import review_reasons
from ticketpilot.graph.state import TicketState
from ticketpilot.llm import prompts
from ticketpilot.llm.backend import LLMBackend
from ticketpilot.retrieval.retriever import HybridRetriever
from ticketpilot.safety import injection, policy_rules
from ticketpilot.schemas import (
    ORDER_ID_SEARCH_RE,
    DraftContext,
    Entities,
    Evidence,
    is_valid_order_id,
)
from ticketpilot.telemetry import log_node, merge_usage, node_timer, usage_scope
from ticketpilot.tools.registry import ToolContext, ToolRegistry

log = logging.getLogger("ticketpilot.graph")

HOLDING_REPLY = (
    "Hi,\n\nThank you for contacting Kestrel Outfitters. Your message has been passed to a "
    "specialist team, who will follow up with you by email.\n\nKestrel Outfitters Support"
)
# Tools withheld from the act loop when code has already decided the ticket must be
# escalated (sensitive category or injection flag): money never moves on those tickets.
WITHHELD_WHEN_ESCALATING = frozenset({"issue_refund"})


@dataclass
class Deps:
    backend: LLMBackend
    retriever: HybridRetriever
    registry: ToolRegistry
    db: Database
    settings: Settings


NodeFn = Callable[[TicketState, Deps], dict[str, Any]]


def _node(name: str, fn: NodeFn, deps: Deps) -> Callable[[TicketState], dict[str, Any]]:
    def run(state: TicketState) -> dict[str, Any]:
        update: dict[str, Any]
        outcome = "ok"
        with node_timer() as t, usage_scope() as scope:
            try:
                update = fn(state, deps)
            except GraphBubbleUp:
                raise  # LangGraph control flow (interrupt), not an error
            except Exception as exc:
                log.exception("node failed", extra={"node": name})
                outcome = "error"
                update = {
                    "errors": [*state.get("errors", []), f"{name}: {type(exc).__name__}: {exc}"]
                }
        timings = {**state.get("timings_ms", {}), name: t["ms"]}
        usage = merge_usage(state.get("usage", {}), name, scope)
        log_node(state.get("ticket_id", "?"), name, t["ms"], outcome)
        return {**update, "timings_ms": timings, "usage": usage}

    run.__name__ = name
    return run


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

_SIGNATURE_MARKERS = re.compile(
    r"^\s*(sent from my \w+|get outlook for \w+|--\s*$|__+\s*$)", re.IGNORECASE
)
_QUOTE_HEADER = re.compile(r"^\s*(on .{3,120} wrote:|-+\s*original message\s*-+)\s*$", re.I)
_SIGN_OFF = re.compile(
    r"^\s*(thanks|thank you|cheers|best|regards|kind regards|best regards|sincerely|"
    r"many thanks|thx)[,.!]*\s*$",
    re.IGNORECASE,
)


def strip_html(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>|</p>|</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def normalize_text(text: str) -> str:
    text = strip_html(text).replace("\xa0", " ")
    text = re.sub(r"[​‌‍⁠﻿]", "", text)
    kept: list[str] = []
    for line in text.splitlines():
        if _QUOTE_HEADER.match(line) or _SIGNATURE_MARKERS.match(line):
            break  # everything after a quote header / signature marker is dropped
        if line.lstrip().startswith(">"):
            continue
        kept.append(line.strip())
    # Drop a trailing sign-off ("Thanks,\nSam") in the last few lines.
    for i in range(len(kept) - 1, max(len(kept) - 4, -1), -1):
        if _SIGN_OFF.match(kept[i]):
            kept = kept[:i]
            break
    out = "\n".join(kept)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def normalize(state: TicketState, deps: Deps) -> dict[str, Any]:
    subject = (state.get("subject") or "").strip()
    raw = state.get("raw_text", "")
    det = injection.detect(f"{subject}\n{raw}")
    body = normalize_text(raw)
    clean = f"Subject: {subject}\n\n{body}" if subject else body
    return {"clean_text": clean, "injection_flag": det.flag, "injection_reasons": det.reasons}


# ---------------------------------------------------------------------------
# classify / extract / retrieve
# ---------------------------------------------------------------------------


def classify(state: TicketState, deps: Deps) -> dict[str, Any]:
    c = deps.backend.classify(state["clean_text"])
    return {
        "category": c.category,
        "confidence": float(c.confidence),
        "classification_reasoning": c.reasoning,
    }


def extract_entities(state: TicketState, deps: Deps) -> dict[str, Any]:
    ents = deps.backend.extract(state["clean_text"], state.get("category", "unknown"))
    from_model = [o.strip().upper() for o in ents.order_ids]
    from_text = [m.upper() for m in ORDER_ID_SEARCH_RE.findall(state["clean_text"])]
    order_ids = list(dict.fromkeys(o for o in [*from_model, *from_text] if is_valid_order_id(o)))
    data = ents.model_copy(update={"order_ids": order_ids}).model_dump()
    return {"entities": data}


def retrieve(state: TicketState, deps: Deps) -> dict[str, Any]:
    category = state.get("category", "").replace("_", " ")
    query = f"{state['clean_text']}\n{category}"
    hits = deps.retriever.search(
        query, k=deps.settings.RETRIEVAL_TOP_K, mode=deps.settings.RETRIEVAL_MODE
    )
    return {"retrieved": [h.as_state() for h in hits]}


# ---------------------------------------------------------------------------
# act (tool loop)
# ---------------------------------------------------------------------------


def _escalation_decided(state: TicketState, settings: Settings) -> bool:
    return bool(
        state.get("injection_flag")
        or policy_rules.is_mandatory_escalation_category(state.get("category"), settings)
    )


def _tool_message(result: dict[str, Any]) -> str:
    payload = {k: result.get(k) for k in ("ok", "data", "error", "pending", "needs_specialist")}
    return json.dumps(payload, default=str)


def act(state: TicketState, deps: Deps) -> dict[str, Any]:
    s = deps.settings
    tools = deps.registry.schemas()
    if _escalation_decided(state, s):
        tools = [t for t in tools if t["name"] not in WITHHELD_WHEN_ESCALATING]
    allowed = {t["name"] for t in tools}

    new_messages: list[BaseMessage] = [
        SystemMessage(prompts.ACT_SYSTEM.format(max_steps=s.MAX_TOOL_STEPS)),
        HumanMessage(
            prompts.ACT_USER.format(
                ticket_id=state["ticket_id"],
                customer_email=state["customer_email"],
                category=state.get("category", "unknown"),
                entities=json.dumps(state.get("entities", {})),
                text=state["clean_text"],
                excerpts=prompts.format_excerpts(state.get("retrieved", [])),
            )
        ),
    ]
    ctx = ToolContext(
        customer_email=state["customer_email"],
        ticket_id=state["ticket_id"],
        db=deps.db,
        settings=s,
        retriever=deps.retriever,
    )
    results: list[dict[str, Any]] = list(state.get("tool_results", []))
    errors = list(state.get("errors", []))
    finished = False
    for _step in range(s.MAX_TOOL_STEPS):
        ai = deps.backend.act_step(new_messages, tools)
        new_messages.append(ai)
        if not ai.tool_calls:
            finished = True
            break
        for tc in ai.tool_calls:
            if tc["name"] not in allowed:
                res = {
                    "tool": tc["name"],
                    "args": tc.get("args", {}),
                    "ok": False,
                    "data": None,
                    "error": f"tool '{tc['name']}' is not available for this ticket",
                    "pending": False,
                    "needs_specialist": False,
                    "duration_ms": 0.0,
                    "arg_error": False,
                    "call_id": tc.get("id"),
                    "blocked": True,
                }
            else:
                res = deps.registry.execute(dict(tc), ctx).to_state()
            results.append(res)
            new_messages.append(ToolMessage(_tool_message(res), tool_call_id=tc.get("id") or ""))
    if not finished:
        errors.append(f"act: tool step cap reached ({s.MAX_TOOL_STEPS} steps)")
    update: dict[str, Any] = {"messages": new_messages, "tool_results": results}
    if errors != list(state.get("errors", [])):
        update["errors"] = errors
    return update


# ---------------------------------------------------------------------------
# draft
# ---------------------------------------------------------------------------


def _allowed_doc_ids(state: TicketState) -> set[str]:
    ids = {c["doc_id"] for c in state.get("retrieved", [])}
    for r in state.get("tool_results", []):
        if r.get("tool") == "search_kb" and r.get("ok"):
            ids.update(h["doc_id"] for h in r.get("data") or [])
    return ids


def _kb_evidence(state: TicketState) -> list[dict[str, Any]]:
    """Retrieved chunks plus any search_kb results, deduplicated by text."""
    chunks = list(state.get("retrieved", []))
    seen = {c["text"] for c in chunks}
    for r in state.get("tool_results", []):
        if r.get("tool") == "search_kb" and r.get("ok"):
            for h in r.get("data") or []:
                if h["text"] not in seen:
                    seen.add(h["text"])
                    chunks.append({"doc_id": h["doc_id"], "text": h["text"]})
    return chunks


def draft(state: TicketState, deps: Deps) -> dict[str, Any]:
    ctx = DraftContext(
        ticket_text=state["clean_text"],
        customer_email=state["customer_email"],
        category=state.get("category", "unknown"),
        entities=Entities(**state.get("entities", {})),
        retrieved=_kb_evidence(state),
        tool_results=state.get("tool_results", []),
        injection_flag=bool(state.get("injection_flag")),
    )
    d = deps.backend.draft(ctx)
    allowed = _allowed_doc_ids(state)
    kept = [c for c in dict.fromkeys(d.citations) if c in allowed]
    dropped = [c for c in d.citations if c not in allowed]
    issues = [f"citation not in retrieved set: {c}" for c in dict.fromkeys(dropped)]
    return {
        "draft_reply": d.reply,
        "citations": kept,
        "proposed_action": d.proposed_action,
        "draft_issues": issues,
    }


# ---------------------------------------------------------------------------
# validate (Phase 7)
# ---------------------------------------------------------------------------


def build_evidence(state: TicketState) -> list[Evidence]:
    ev = [Evidence(source=c["doc_id"], text=c["text"]) for c in _kb_evidence(state)]
    for r in state.get("tool_results", []):
        ev.append(Evidence(source=f"tool:{r.get('tool')}", text=_tool_message(r)))
    return ev


def _numbers_in(obj: Any) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int | float):
        out.add(round(float(obj), 2))
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _numbers_in(v)
    elif isinstance(obj, str):
        out |= {round(a, 2) for a in policy_rules.money_amounts(obj)}
    return out


def rule_issues(state: TicketState) -> list[str]:
    """Plain-code checks on the reply content (PROJECT_SPEC.md §12 step 3)."""
    reply = state.get("draft_reply", "") or ""
    results = state.get("tool_results", [])
    issues: list[str] = []
    refunds = [r for r in results if r.get("tool") == "issue_refund" and r.get("ok")]

    if any(r.get("pending") for r in results) and policy_rules.reply_claims_refund_confirmed(reply):
        issues.append("reply describes a pending refund as processed/confirmed")

    for r in refunds:
        method = (r.get("data") or {}).get("method")
        if method == "store_credit" and policy_rules.reply_mentions_original_payment(reply):
            issues.append("reply says original payment but the refund is store credit")
        if (
            method == "original_payment"
            and policy_rules.reply_mentions_store_credit(reply)
            and not policy_rules.reply_mentions_original_payment(reply)
        ):
            issues.append("reply says store credit but the refund is to the original payment")

    allowed_amounts: set[float] = set()
    for r in results:
        allowed_amounts |= _numbers_in(r.get("data"))
    for c in _kb_evidence(state):
        allowed_amounts |= _numbers_in(c["text"])
    for amount in policy_rules.money_amounts(reply):
        if round(amount, 2) not in allowed_amounts:
            issues.append(f"amount ${amount:.2f} in reply does not match any tool result")

    issues.extend(state.get("draft_issues", []) or [])
    return issues


def validate(state: TicketState, deps: Deps) -> dict[str, Any]:
    if not deps.settings.VALIDATION_ENABLED:
        # Ablation only: skip both the LLM claim check and reply-content rule checks.
        return {
            "validation": {
                "enabled": False,
                "grounded": True,
                "claims": [],
                "rule_issues": [],
                "issues": [],
            }
        }
    reply = state.get("draft_reply", "") or ""
    claims: list[dict[str, Any]] = []
    if reply.strip():
        result = deps.backend.check_claims(reply, build_evidence(state))
        claims = [c.model_dump() for c in result.claims]
    grounded = bool(reply.strip()) and all(c["supported"] for c in claims)
    rules = rule_issues(state)
    unsupported = [f"unsupported claim: {c['claim']}" for c in claims if not c["supported"]]
    return {
        "validation": {
            "enabled": True,
            "grounded": grounded,
            "claims": claims,
            "rule_issues": rules,
            "issues": unsupported + rules,
        }
    }


# ---------------------------------------------------------------------------
# terminal nodes
# ---------------------------------------------------------------------------


def auto_send(state: TicketState, deps: Deps) -> dict[str, Any]:
    reply = state.get("draft_reply", "")
    deps.db.write_outbox(state["ticket_id"], state["customer_email"], reply, "auto")
    deps.db.update_ticket(state["ticket_id"], status="auto_sent", route="auto_send")
    return {"route": "auto_send", "final_reply": reply}


def pending_refunds(state: TicketState) -> list[dict[str, Any]]:
    return [
        r["data"]
        for r in state.get("tool_results", [])
        if r.get("tool") == "issue_refund" and r.get("ok") and r.get("pending") and r.get("data")
    ]


def review_payload(state: TicketState, settings: Settings) -> dict[str, Any]:
    return {
        "ticket_id": state["ticket_id"],
        "customer_email": state["customer_email"],
        "subject": state.get("subject"),
        "body": state.get("raw_text"),
        "category": state.get("category"),
        "confidence": state.get("confidence"),
        "draft_reply": state.get("draft_reply"),
        "citations": state.get("citations", []),
        "retrieved": [
            {"doc_id": c["doc_id"], "section": c.get("section"), "text": c["text"]}
            for c in state.get("retrieved", [])
        ],
        "tool_results": state.get("tool_results", []),
        "validation": state.get("validation", {}),
        "pending_refunds": pending_refunds(state),
        "reasons": review_reasons(state, settings),
        "errors": state.get("errors", []),
    }


def human_review(state: TicketState, deps: Deps) -> dict[str, Any]:
    # Side effects before interrupt() re-run on resume, so they must be idempotent.
    deps.db.update_ticket(state["ticket_id"], status="awaiting_review", route="human_review")
    decision: dict[str, Any] = interrupt(review_payload(state, deps.settings))

    action = decision.get("action")
    reviewer = str(decision.get("reviewer") or "reviewer")
    if action not in ("approve", "edit_approve", "reject"):
        raise ValueError(f"invalid review action: {action!r}")
    final: str | None = None
    if action in ("approve", "edit_approve"):
        final = (
            decision.get("edited_reply") or state.get("draft_reply", "")
            if action == "edit_approve"
            else state.get("draft_reply", "")
        )
        for refund in pending_refunds(state):
            deps.db.set_refund_status(refund["id"], "approved", approved_by=reviewer)
        deps.db.write_outbox(state["ticket_id"], state["customer_email"], final or "", reviewer)
    else:
        for refund in pending_refunds(state):
            deps.db.set_refund_status(refund["id"], "rejected", approved_by=reviewer)
    deps.db.record_review_event(
        state["ticket_id"],
        reviewer,
        action,
        edited_reply=decision.get("edited_reply") if action == "edit_approve" else None,
        duration_seconds=decision.get("duration_seconds"),
    )
    deps.db.update_ticket(state["ticket_id"], status="closed", route="human_review")
    return {"route": "human_review", "review_decision": decision, "final_reply": final}


TEAM_KEYWORDS: list[tuple[str, str]] = [
    (r"chargeback|dispute (the|a) charge|my bank", "billing"),
    (r"lawyer|attorney|lawsuit|legal action|sue\b|small claims", "legal"),
    (r"journalist|reporter|press|media|article|news", "general"),
    (
        r"injur|burn|hurt|hospital|unsafe|fire|harass|threat|hack|"
        r"someone (else )?(logged|accessed)",
        "trust_safety",
    ),
]
CATEGORY_TEAM = {
    "account_billing": "billing",
    "shipping_problem": "logistics",
    "order_status": "logistics",
}


def default_team(state: TicketState) -> str:
    if state.get("injection_flag"):
        return "trust_safety"
    text = state.get("clean_text", "").lower()
    for pattern, team in TEAM_KEYWORDS:
        if re.search(pattern, text):
            return team
    return CATEGORY_TEAM.get(state.get("category", ""), "general")


def escalate(state: TicketState, deps: Deps) -> dict[str, Any]:
    tid = state["ticket_id"]
    if not deps.db.get_escalations(tid):
        reason = "; ".join(
            filter(
                None,
                [
                    f"category={state.get('category')}",
                    "injection: " + ", ".join(state.get("injection_reasons", []))
                    if state.get("injection_flag")
                    else "",
                ],
            )
        )
        deps.db.create_escalation(tid, default_team(state), reason[:300])
    deps.db.write_outbox(tid, state["customer_email"], HOLDING_REPLY, "auto")
    deps.db.update_ticket(tid, status="escalated", route="escalate")
    return {"route": "escalate", "final_reply": HOLDING_REPLY}


NODES: dict[str, NodeFn] = {
    "normalize": normalize,
    "classify": classify,
    "extract_entities": extract_entities,
    "retrieve": retrieve,
    "act": act,
    "draft": draft,
    "validate": validate,
    "auto_send": auto_send,
    "human_review": human_review,
    "escalate": escalate,
}
