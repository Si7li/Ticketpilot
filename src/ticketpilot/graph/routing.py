"""Routing: plain code, a pure function of state + settings (PROJECT_SPEC.md §11.3).

Being pure lets the eval sweep re-run routing at different thresholds without LLM calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ticketpilot.config import Settings


def choose_route(state: Mapping[str, Any], settings: Settings) -> str:
    if state.get("category") in settings.sensitive_categories or state.get("injection_flag"):
        return "escalate"
    if any(r.get("needs_specialist") for r in state.get("tool_results", [])):
        return "escalate"
    if state.get("errors"):
        return "human_review"
    v = state.get("validation", {})
    if not v.get("grounded", False) or v.get("issues"):
        return "human_review"
    if any(r.get("pending") for r in state.get("tool_results", [])):
        return "human_review"
    if state.get("confidence", 0.0) < settings.CONFIDENCE_THRESHOLD:
        return "human_review"
    return "auto_send"


def review_reasons(state: Mapping[str, Any], settings: Settings) -> list[str]:
    """Human-readable reasons a ticket needs review (for the queue / UI)."""
    reasons: list[str] = []
    if state.get("errors"):
        reasons.append("errors: " + "; ".join(state["errors"])[:200])
    v = state.get("validation", {})
    if not v.get("grounded", False):
        reasons.append("reply not fully grounded")
    for issue in v.get("issues", []) or []:
        reasons.append(f"validation: {issue}")
    if any(r.get("pending") for r in state.get("tool_results", [])):
        reasons.append("refund pending approval")
    conf = state.get("confidence", 0.0)
    if conf < settings.CONFIDENCE_THRESHOLD:
        reasons.append(f"low confidence ({conf:.2f} < {settings.CONFIDENCE_THRESHOLD:.2f})")
    return reasons
