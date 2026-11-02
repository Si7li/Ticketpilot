"""Pydantic models shared across modules: LLM-facing schemas and domain records."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal[
    "order_status",
    "returns_refunds",
    "shipping_problem",
    "product_question",
    "account_billing",
    "other_sensitive",
]
CATEGORIES: tuple[str, ...] = (
    "order_status",
    "returns_refunds",
    "shipping_problem",
    "product_question",
    "account_billing",
    "other_sensitive",
)
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "order_status": "Where is my order, delivery estimates",
    "returns_refunds": "Returns, refund requests, refund status",
    "shipping_problem": "Damaged, wrong item, lost package",
    "product_question": "Sizing, capacity, compatibility, care",
    "account_billing": "Charges, payment problems, account changes",
    "other_sensitive": "Legal threats, chargebacks, press, injury/safety, harassment, off-policy",
}

Action = Literal["reply", "refund", "escalate"]
Route = Literal["auto_send", "human_review", "escalate"]
RefundMethod = Literal["original_payment", "store_credit"]
RefundStatus = Literal["approved", "pending_approval", "rejected"]
EscalationTeam = Literal["billing", "legal", "logistics", "trust_safety", "general"]

ORDER_ID_RE = re.compile(r"^KO-\d{5}$")
ORDER_ID_SEARCH_RE = re.compile(r"\bKO-\d{5}\b", re.IGNORECASE)


def is_valid_order_id(value: str) -> bool:
    return bool(ORDER_ID_RE.match(value))


# ---------------------------------------------------------------------------
# LLM-facing schemas (structured output)
# ---------------------------------------------------------------------------


class Classification(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(max_length=300)


class Entities(BaseModel):
    order_ids: list[str] = Field(default_factory=list)  # filtered against ^KO-\d{5}$ afterwards
    products: list[str] = Field(default_factory=list)
    amounts: list[float] = Field(default_factory=list)
    photo_provided: bool = False
    requests: list[str] = Field(default_factory=list)  # each distinct thing the customer asks for


class DraftReply(BaseModel):
    reply: str
    citations: list[str]  # doc_ids
    proposed_action: Action


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    source: str | None = None  # doc_id or "tool:<name>"


class ValidationResult(BaseModel):
    claims: list[ClaimCheck]


class Evidence(BaseModel):
    """One piece of evidence given to the validator / judge."""

    source: str  # doc_id or "tool:<name>"
    text: str


class DraftContext(BaseModel):
    """Everything the drafting call may use."""

    ticket_text: str
    customer_email: str
    category: str
    entities: Entities
    retrieved: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    injection_flag: bool = False


# ---------------------------------------------------------------------------
# Domain records (business DB)
# ---------------------------------------------------------------------------


class Customer(BaseModel):
    id: str
    email: str
    name: str
    loyalty_tier: Literal["trail", "summit", "peak"]
    country: str = "US"


class OrderItem(BaseModel):
    sku: str
    name: str
    qty: int = 1
    unit_price: float
    is_sale: bool = False
    final_sale: bool = False
    category: str


class Order(BaseModel):
    id: str
    customer_id: str
    items: list[OrderItem]
    amount: float
    is_sale: bool
    status: Literal["processing", "shipped", "delivered", "lost", "returned"]
    order_date: str
    ship_date: str | None = None
    delivery_date: str | None = None
    tracking_last_update: str | None = None
    shipping_method: str

    @property
    def has_final_sale(self) -> bool:
        return any(i.final_sale for i in self.items)


class Refund(BaseModel):
    id: str
    order_id: str
    amount: float
    reason: str
    method: RefundMethod
    status: RefundStatus
    created_at: str
    approved_by: str | None = None
