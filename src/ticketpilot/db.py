"""SQLite access layer for business tables (customers, orders, refunds) and app tables.

A new connection is opened per operation, which keeps the layer thread-safe for
FastAPI's thread pool and the eval harness without any locking.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ticketpilot.schemas import Customer, Order, OrderItem, Refund

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "schema.sql"


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class DuplicateRefundError(Exception):
    """Raised when an active (non-rejected) refund already exists for an order."""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- plumbing ---------------------------------------------------------------

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self, schema_path: Path = SCHEMA_PATH) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(schema_path.read_text())

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1 FROM orders LIMIT 1")
            return True
        except sqlite3.Error:
            return False

    # -- customers & orders -----------------------------------------------------

    def get_customer_by_email(self, email: str) -> Customer | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE lower(email) = lower(?)", (email.strip(),)
            ).fetchone()
        return Customer(**dict(row)) if row else None

    def get_customer(self, customer_id: str) -> Customer | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return Customer(**dict(row)) if row else None

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        d = dict(row)
        items = [OrderItem(**i) for i in json.loads(d.pop("items_json"))]
        d["is_sale"] = bool(d["is_sale"])
        return Order(items=items, **d)

    def get_order(self, order_id: str) -> Order | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        return self._row_to_order(row) if row else None

    def get_recent_orders(self, customer_id: str, limit: int = 5) -> list[Order]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date DESC, id DESC "
                "LIMIT ?",
                (customer_id, limit),
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    # -- refunds ----------------------------------------------------------------

    def get_active_refund(self, order_id: str) -> Refund | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM refunds WHERE order_id = ? AND status != 'rejected'", (order_id,)
            ).fetchone()
        return Refund(**dict(row)) if row else None

    def get_refund(self, refund_id: str) -> Refund | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM refunds WHERE id = ?", (refund_id,)).fetchone()
        return Refund(**dict(row)) if row else None

    def create_refund(
        self,
        order_id: str,
        amount: float,
        reason: str,
        method: str,
        status: str,
        approved_by: str | None = None,
        refund_id: str | None = None,
        created_at: str | None = None,
    ) -> Refund:
        rid = refund_id or new_id("RF")
        try:
            with self.connect() as conn:
                conn.execute(
                    "INSERT INTO refunds (id, order_id, amount, reason, method, status, "
                    "created_at, approved_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rid,
                        order_id,
                        round(amount, 2),
                        reason,
                        method,
                        status,
                        created_at or utcnow(),
                        approved_by,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "one_active_refund_per_order" in str(exc) or "UNIQUE" in str(exc):
                raise DuplicateRefundError(order_id) from exc
            raise
        refund = self.get_refund(rid)
        assert refund is not None
        return refund

    def set_refund_status(
        self, refund_id: str, status: str, approved_by: str | None = None
    ) -> Refund | None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE refunds SET status = ?, approved_by = ? WHERE id = ?",
                (status, approved_by, refund_id),
            )
        return self.get_refund(refund_id)

    # -- escalations & outbox ---------------------------------------------------

    def create_escalation(self, ticket_id: str, team: str, reason: str) -> dict[str, Any]:
        eid = new_id("E")
        row = {
            "id": eid,
            "ticket_id": ticket_id,
            "team": team,
            "reason": reason,
            "created_at": utcnow(),
        }
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO escalations (id, ticket_id, team, reason, created_at) "
                "VALUES (:id, :ticket_id, :team, :reason, :created_at)",
                row,
            )
        return row

    def get_escalations(self, ticket_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM escalations WHERE ticket_id = ? ORDER BY created_at", (ticket_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def write_outbox(
        self, ticket_id: str, to_email: str, body: str, sent_by: str
    ) -> dict[str, Any]:
        row = {
            "id": new_id("OB"),
            "ticket_id": ticket_id,
            "to_email": to_email,
            "body": body,
            "sent_by": sent_by,
            "created_at": utcnow(),
        }
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO outbox (id, ticket_id, to_email, body, sent_by, created_at) "
                "VALUES (:id, :ticket_id, :to_email, :body, :sent_by, :created_at)",
                row,
            )
        return row

    def get_outbox(self, ticket_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM outbox WHERE ticket_id = ? ORDER BY created_at", (ticket_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- tickets ----------------------------------------------------------------

    def create_ticket(
        self, ticket_id: str, customer_email: str, subject: str | None, body: str
    ) -> dict[str, Any]:
        now = utcnow()
        row = {
            "id": ticket_id,
            "customer_email": customer_email,
            "subject": subject,
            "body": body,
            "status": "received",
            "route": None,
            "result_json": None,
            "created_at": now,
            "updated_at": now,
        }
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tickets (id, customer_email, subject, body, status, route, "
                "result_json, created_at, updated_at) VALUES (:id, :customer_email, :subject, "
                ":body, :status, :route, :result_json, :created_at, :updated_at)",
                row,
            )
        return row

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["result"] = json.loads(d["result_json"]) if d["result_json"] else None
        return d

    def update_ticket(
        self,
        ticket_id: str,
        *,
        status: str | None = None,
        route: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        sets, params = ["updated_at = ?"], [utcnow()]
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if route is not None:
            sets.append("route = ?")
            params.append(route)
        if result is not None:
            sets.append("result_json = ?")
            params.append(json.dumps(result, default=str))
        params.append(ticket_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE tickets SET {', '.join(sets)} WHERE id = ?", params)

    def list_tickets(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM tickets WHERE status = ? ORDER BY created_at", (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tickets ORDER BY created_at").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d["result_json"]) if d["result_json"] else None
            out.append(d)
        return out

    def next_ticket_id(self) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM tickets WHERE id LIKE 'T-%' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        n = int(row["id"].split("-")[1]) + 1 if row else 5001
        return f"T-{n:04d}"

    # -- review events ----------------------------------------------------------

    def record_review_event(
        self,
        ticket_id: str,
        reviewer: str,
        action: str,
        edited_reply: str | None = None,
        duration_seconds: float | None = None,
    ) -> dict[str, Any]:
        row = {
            "id": new_id("RV"),
            "ticket_id": ticket_id,
            "reviewer": reviewer,
            "action": action,
            "edited_reply": edited_reply,
            "duration_seconds": duration_seconds,
            "created_at": utcnow(),
        }
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO review_events (id, ticket_id, reviewer, action, edited_reply, "
                "duration_seconds, created_at) VALUES (:id, :ticket_id, :reviewer, :action, "
                ":edited_reply, :duration_seconds, :created_at)",
                row,
            )
        return row

    def list_review_events(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM review_events ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]
