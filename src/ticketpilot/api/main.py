"""FastAPI service (PROJECT_SPEC.md §13).

`create_app()` builds settings, DB, retriever (indexing the KB if needed), backend, tools,
checkpointer and graph once in the lifespan. Tests pass `build_on_startup=False` and
override `get_runtime` with a FakeLLMBackend runtime.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ticketpilot.api.schemas import (
    Health,
    QueueItem,
    ReviewIn,
    TicketDetail,
    TicketIn,
    TicketResult,
    ToolSummary,
)
from ticketpilot.config import Settings, get_settings
from ticketpilot.graph.build import (
    RunOutcome,
    is_awaiting_review,
    resume_ticket,
    run_ticket,
    thread_config,
    total_ms,
)
from ticketpilot.graph.routing import review_reasons
from ticketpilot.llm.backend import BackendError
from ticketpilot.runtime import Runtime, build_runtime
from ticketpilot.schemas import (
    Classification,
    DraftContext,
    DraftReply,
    Entities,
    Evidence,
    ValidationResult,
)
from ticketpilot.telemetry import UsageTracker

log = logging.getLogger("ticketpilot.api")


class UnavailableBackend:
    """Stand-in when no API key is configured: the service starts, LLM calls fail clearly."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.usage = UsageTracker()

    def _fail(self) -> Any:
        raise BackendError(self.reason)

    def classify(self, text: str) -> Classification:
        return self._fail()  # type: ignore[no-any-return]

    def extract(self, text: str, category: str) -> Entities:
        return self._fail()  # type: ignore[no-any-return]

    def act_step(self, messages: Any, tools: Any) -> Any:
        return self._fail()

    def draft(self, context: DraftContext) -> DraftReply:
        return self._fail()  # type: ignore[no-any-return]

    def check_claims(self, reply: str, evidence: list[Evidence]) -> ValidationResult:
        return self._fail()  # type: ignore[no-any-return]


def startup_runtime(settings: Settings) -> Runtime:
    try:
        return build_runtime(settings)
    except BackendError as exc:
        log.warning("starting without an LLM backend: %s", exc)
        return build_runtime(settings, backend=UnavailableBackend(str(exc)))


def get_runtime(request: Request) -> Runtime:
    rt: Runtime | None = getattr(request.app.state, "runtime", None)
    if rt is None:
        raise HTTPException(503, "service is starting up")
    return rt


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def result_from_state(state: dict[str, Any], status: str, rt: Runtime) -> TicketResult:
    tools = [
        ToolSummary(
            tool=r.get("tool", "?"),
            args=r.get("args") or {},
            ok=bool(r.get("ok")),
            error=r.get("error"),
            pending=bool(r.get("pending")),
            needs_specialist=bool(r.get("needs_specialist")),
            data=r.get("data"),
        )
        for r in state.get("tool_results", [])
    ]
    route = state.get("route") or ("human_review" if status == "awaiting_review" else None)
    return TicketResult(
        ticket_id=state.get("ticket_id", ""),
        status=status,
        route=route,
        category=state.get("category"),
        confidence=state.get("confidence"),
        injection_flag=bool(state.get("injection_flag")),
        draft_reply=state.get("draft_reply"),
        final_reply=state.get("final_reply"),
        citations=state.get("citations", []),
        validation=state.get("validation", {}),
        tool_results=tools,
        review_reasons=review_reasons(state, rt.settings) if route == "human_review" else [],
        errors=state.get("errors", []),
        timings_ms=state.get("timings_ms", {}),
        total_ms=total_ms(state),
        usage=state.get("usage", {}),
    )


def _store(rt: Runtime, ticket_id: str, out: RunOutcome) -> TicketResult:
    row = rt.db.get_ticket(ticket_id)
    status = row["status"] if row else "received"
    if out.interrupted:
        status = "awaiting_review"
    result = result_from_state(out.state, status, rt)
    rt.db.update_ticket(ticket_id, status=status, route=result.route, result=result.model_dump())
    return result


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def create_app(build_on_startup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if build_on_startup and getattr(app.state, "runtime", None) is None:
            app.state.runtime = startup_runtime(get_settings())
        yield

    app = FastAPI(title="TicketPilot", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        code = {404: "not_found", 409: "conflict", 503: "unavailable"}.get(exc.status_code, "error")
        return JSONResponse(
            status_code=exc.status_code, content={"error": {"code": code, "message": exc.detail}}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "invalid request",
                    "details": details,
                }
            },
        )

    @app.exception_handler(BackendError)
    async def backend_error(_: Request, exc: BackendError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"error": {"code": "llm_unavailable", "message": str(exc)}}
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal", "message": type(exc).__name__}},
        )

    @app.post("/tickets", response_model=TicketResult)
    def create_ticket(body: TicketIn, rt: RuntimeDep) -> TicketResult:
        ticket_id = body.ticket_id or rt.db.next_ticket_id()
        if rt.db.get_ticket(ticket_id) is not None:
            raise HTTPException(409, f"ticket {ticket_id} already exists")
        rt.db.create_ticket(ticket_id, body.customer_email, body.subject, body.body)
        ticket = {
            "ticket_id": ticket_id,
            "customer_email": body.customer_email,
            "subject": body.subject,
            "body": body.body,
        }
        out = run_ticket(rt.graph, ticket)
        return _store(rt, ticket_id, out)

    @app.get("/tickets/{ticket_id}", response_model=TicketDetail)
    def get_ticket(ticket_id: str, rt: RuntimeDep) -> TicketDetail:
        row = rt.db.get_ticket(ticket_id)
        if row is None:
            raise HTTPException(404, f"ticket {ticket_id} not found")
        snapshot = rt.graph.get_state(thread_config(ticket_id))
        return TicketDetail(
            ticket_id=row["id"],
            customer_email=row["customer_email"],
            subject=row["subject"],
            body=row["body"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=TicketResult(**row["result"]) if row["result"] else None,
            graph_next=list(snapshot.next or ()),
            awaiting_review="human_review" in (snapshot.next or ()),
        )

    @app.get("/review-queue", response_model=list[QueueItem])
    def review_queue(rt: RuntimeDep) -> list[QueueItem]:
        items: list[QueueItem] = []
        for row in rt.db.list_tickets("awaiting_review"):
            snapshot = rt.graph.get_state(thread_config(row["id"]))
            payload: dict[str, Any] = {}
            for task in snapshot.tasks:
                for intr in task.interrupts:
                    if isinstance(intr.value, dict):
                        payload = intr.value
            if not payload:
                continue  # not actually paused (e.g. stale row)
            items.append(
                QueueItem(
                    ticket_id=row["id"],
                    customer_email=row["customer_email"],
                    subject=row["subject"],
                    created_at=row["created_at"],
                    category=payload.get("category"),
                    confidence=payload.get("confidence"),
                    reasons=payload.get("reasons", []),
                    payload=payload,
                )
            )
        return items

    @app.post("/review/{ticket_id}", response_model=TicketResult)
    def review(ticket_id: str, body: ReviewIn, rt: RuntimeDep) -> TicketResult:
        row = rt.db.get_ticket(ticket_id)
        if row is None:
            raise HTTPException(404, f"ticket {ticket_id} not found")
        if row["status"] != "awaiting_review" or not is_awaiting_review(rt.graph, ticket_id):
            raise HTTPException(409, f"ticket {ticket_id} is not awaiting review")
        if body.action == "edit_approve" and not (body.edited_reply or "").strip():
            raise HTTPException(422, "edited_reply is required for edit_approve")
        out = resume_ticket(rt.graph, ticket_id, body.model_dump())
        return _store(rt, ticket_id, out)

    @app.get("/health", response_model=Health)
    def health(request: Request) -> JSONResponse:
        rt: Runtime | None = getattr(request.app.state, "runtime", None)
        s = rt.settings if rt else get_settings()
        db_ok = qdrant_ok = kb_ok = False
        if rt is not None:
            db_ok = rt.db.ping()
            try:
                rt.deps.retriever.vector.client.get_collections()
                qdrant_ok = True
                kb_ok = rt.deps.retriever.is_indexed()
            except Exception:
                qdrant_ok = False
        model_ok = rt is not None and not isinstance(rt.deps.backend, UnavailableBackend)
        h = Health(
            ok=db_ok and qdrant_ok and kb_ok,
            db=db_ok,
            qdrant=qdrant_ok,
            kb_indexed=kb_ok,
            model_configured=model_ok,
            agent_model=s.AGENT_MODEL,
            retrieval_mode=s.RETRIEVAL_MODE,
        )
        return JSONResponse(status_code=200 if h.ok else 503, content=h.model_dump())

    return app


app = create_app()
