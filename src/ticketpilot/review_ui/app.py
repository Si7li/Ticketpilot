"""Streamlit review UI. Talks only to the API (`API_URL`).

Run: `make ui` (or `streamlit run src/ticketpilot/review_ui/app.py`).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import streamlit as st

from ticketpilot.review_ui.helpers import age_label, highlight_unsupported

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="TicketPilot review", layout="wide")


def api(method: str, path: str, **kw: Any) -> Any:
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=120, **kw)
    except httpx.HTTPError as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")
        st.stop()
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
            st.error(f"{r.status_code} {err.get('code')}: {err.get('message')}")
        except ValueError:
            st.error(f"{r.status_code}: {r.text[:300]}")
        return None
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("TicketPilot")
reviewer = st.sidebar.text_input("Reviewer name", value=st.session_state.get("reviewer", ""))
st.session_state["reviewer"] = reviewer
if st.sidebar.button("Refresh queue"):
    st.rerun()

queue: list[dict[str, Any]] = api("GET", "/review-queue") or []
st.sidebar.metric("Awaiting review", len(queue))

with st.sidebar.expander("Submit a test ticket"), st.form("new_ticket", clear_on_submit=True):
    email = st.text_input("Customer email", "sam.ortiz@example.com")
    subject = st.text_input("Subject", "Torn tent")
    body = st.text_area(
        "Body",
        "My Ridgeline 2 tent from order KO-10482 arrived with a torn "
        "rainfly. Photo attached. I want my money back.",
    )
    if st.form_submit_button("Submit"):
        res = api(
            "POST", "/tickets", json={"customer_email": email, "subject": subject, "body": body}
        )
        if res:
            st.success(f"{res['ticket_id']} → {res['route']}")

# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

st.header("Review queue")
if not queue:
    st.info("Nothing to review. Submit a ticket from the sidebar or via the API.")
    st.stop()

rows = [
    {
        "ticket": q["ticket_id"],
        "category": q.get("category"),
        "confidence": round(q["confidence"], 2) if q.get("confidence") is not None else None,
        "age": age_label(q["created_at"]),
        "reason": "; ".join(q.get("reasons", []))[:120],
    }
    for q in queue
]
st.dataframe(rows, hide_index=True, use_container_width=True)

ids = [q["ticket_id"] for q in queue]
selected = st.selectbox("Open ticket", ids, index=0)
item = next(q for q in queue if q["ticket_id"] == selected)
payload: dict[str, Any] = item["payload"]

# Timer: from opening a ticket to the decision (feeds the time-saved metric).
if st.session_state.get("open_ticket") != selected:
    st.session_state["open_ticket"] = selected
    st.session_state["opened_at"] = time.monotonic()

# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2])

with left:
    st.subheader(f"{selected} · {payload.get('category')} · conf {payload.get('confidence')}")
    st.caption(f"From {payload.get('customer_email')} — {payload.get('subject') or '(no subject)'}")
    st.text_area("Original ticket", payload.get("body") or "", height=150, disabled=True)

    claims = (payload.get("validation") or {}).get("claims", [])
    if any(not c.get("supported") for c in claims):
        st.markdown("**Draft with unsupported claims highlighted**")
        st.markdown(
            highlight_unsupported(payload.get("draft_reply") or "", claims), unsafe_allow_html=True
        )
    edited = st.text_area(
        "Reply (editable)", payload.get("draft_reply") or "", height=260, key=f"draft_{selected}"
    )

    c1, c2, c3 = st.columns(3)

    def decide(action: str) -> None:
        if not reviewer.strip():
            st.warning("Enter your reviewer name in the sidebar first.")
            return
        duration = round(time.monotonic() - st.session_state.get("opened_at", time.monotonic()), 1)
        body: dict[str, Any] = {
            "action": action,
            "reviewer": reviewer.strip(),
            "duration_seconds": duration,
        }
        if action == "edit_approve":
            body["edited_reply"] = edited
        res = api("POST", f"/review/{selected}", json=body)
        if res:
            st.session_state.pop("open_ticket", None)
            st.success(f"{selected}: {action} ({duration}s)")
            st.rerun()

    if c1.button("Approve", type="primary"):
        decide("approve")
    if c2.button("Save edits & approve"):
        decide("edit_approve")
    if c3.button("Reject"):
        decide("reject")

with right:
    reasons = payload.get("reasons", [])
    if reasons:
        st.markdown("**Why this needs review**")
        for r in reasons:
            st.markdown(f"- {r}")

    for refund in payload.get("pending_refunds", []):
        st.warning(
            f"Pending refund {refund['id']}: ${refund['amount']:.2f} "
            f"({refund['method']}) — approving the reply approves this refund."
        )

    issues = (payload.get("validation") or {}).get("issues", [])
    if issues:
        st.markdown("**Validation issues**")
        for i in issues:
            st.error(i)

    st.markdown("**Cited sources**")
    cited = set(payload.get("citations", []))
    for chunk in payload.get("retrieved", []):
        mark = "✅ " if chunk["doc_id"] in cited else ""
        with st.expander(f"{mark}{chunk['doc_id']} — {chunk.get('section')}"):
            st.text(chunk["text"])

    st.markdown("**Tool results**")
    for t in payload.get("tool_results", []):
        label = f"{t.get('tool')} {'✓' if t.get('ok') else '✗'}"
        if t.get("pending"):
            label += " (pending approval)"
        with st.expander(label):
            st.code(
                json.dumps(
                    {"args": t.get("args"), "data": t.get("data"), "error": t.get("error")},
                    indent=2,
                    default=str,
                ),
                language="json",
            )
