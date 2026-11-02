"""Build the standard ticket set from the Bitext dataset (PROJECT_SPEC.md §7.1).

Steps (`make tickets` runs all of them):
  1. import   Load Bitext from Hugging Face, print columns + intents, write intent_map.yaml,
              sample messages balanced across mapped categories, attach a real seeded order /
              customer to each, and propose gold labels with deterministic rules.
              → data/tickets/raw_bitext_sample.jsonl
  2. rewrite  Rewrite each sampled message with the AGENT model so it mentions Kestrel
              products and the attached order, keeping the original in `source_text`.
              Resumable: already-rewritten tickets are kept.
              → data/tickets/standard_tickets.jsonl

Labels are *proposals* (verified=false) until a human runs `make review-labels`.
Hard cases (data/tickets/hard_cases.jsonl) and product questions
(data/tickets/product_questions.jsonl) are drafted by hand and committed; this script does
not touch them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ticketpilot.config import PROJECT_ROOT, Settings, get_settings
from ticketpilot.db import Database
from ticketpilot.safety.policy_rules import days_since, return_window_days
from ticketpilot.schemas import Customer, Order
from ticketpilot.seed import SCENARIOS

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
TICKETS_DIR = PROJECT_ROOT / "data" / "tickets"
RAW_PATH = TICKETS_DIR / "raw_bitext_sample.jsonl"
OUT_PATH = TICKETS_DIR / "standard_tickets.jsonl"
INTENT_MAP_PATH = TICKETS_DIR / "intent_map.yaml"
SEED = 42
PER_CATEGORY = 30

# intent → (category, recipe). Unlisted intents are dropped.
INTENT_MAP: dict[str, tuple[str, str]] = {
    "track_order": ("order_status", "track"),
    "delivery_period": ("order_status", "delivery_period"),
    "delivery_options": ("order_status", "delivery_options"),
    "cancel_order": ("order_status", "cancel"),
    "change_order": ("order_status", "change_order"),
    "change_shipping_address": ("order_status", "address"),
    "set_up_shipping_address": ("order_status", "address"),
    "get_refund": ("returns_refunds", "refund"),
    "check_refund_policy": ("returns_refunds", "refund_policy"),
    "track_refund": ("returns_refunds", "track_refund"),
    "payment_issue": ("account_billing", "payment_issue"),
    "check_payment_methods": ("account_billing", "payment_methods"),
    "check_invoice": ("account_billing", "invoice"),
    "get_invoice": ("account_billing", "invoice"),
    "create_account": ("account_billing", "create_account"),
    "delete_account": ("account_billing", "delete_account"),
    "edit_account": ("account_billing", "edit_account"),
    "recover_password": ("account_billing", "password"),
    "registration_problems": ("account_billing", "password"),
    "complaint": ("shipping_problem", "complaint"),
}
DROPPED = {
    "check_cancellation_fee": "no cancellation fees exist in Kestrel policy",
    "contact_customer_service": "meta-request, no category fits",
    "contact_human_agent": "meta-request, no category fits",
    "newsletter_subscription": "no newsletter policy",
    "place_order": "sales, not support",
    "review": "feedback, not support",
    "switch_account": "ambiguous (multiple accounts) — no policy",
}

RD, RI, RM = "policy/returns-domestic", "policy/returns-international", "policy/refund-methods"
STATIC_DOCS: dict[str, list[str]] = {
    "delivery_options": ["policy/shipping-times", "policy/free-shipping"],
    "cancel": ["policy/order-changes"],
    "change_order": ["policy/order-changes"],
    "address": ["faq/changing-shipping-address", "policy/order-changes"],
    "refund_policy": [RD, RM],
    "track_refund": ["faq/refund-status", RM],
    "payment_issue": ["faq/payment-methods"],
    "payment_methods": ["faq/payment-methods"],
    "invoice": ["faq/invoice-requests"],
    "create_account": ["faq/creating-an-account"],
    "delete_account": ["faq/closing-an-account"],
    "edit_account": ["faq/updating-email"],
    "password": ["faq/password-reset"],
}
REWRITE_HINTS: dict[str, str] = {
    "track": "The customer asks where their order is.",
    "delivery_period": "The customer asks when their order will arrive.",
    "delivery_options": "The customer asks about shipping options and speeds (no specific order).",
    "cancel": "The customer wants to cancel the order.",
    "change_order": "The customer wants to change an item size in the order.",
    "address": "The customer wants to change the shipping address on the order.",
    "refund": (
        "The customer wants to return the item (unused, tags on) and get a refund to their card."
    ),
    "refund_policy": "The customer asks what the return/refund policy is (no specific order).",
    "track_refund": "The customer returned the item and asks when the refund will arrive.",
    "payment_issue": "The customer's card was declined at checkout and they ask what to do.",
    "payment_methods": "The customer asks which payment methods are accepted.",
    "invoice": "The customer asks how to get an invoice for the order.",
    "create_account": "The customer asks how to create an account.",
    "delete_account": "The customer wants to close their account.",
    "edit_account": "The customer wants to change the email address on their account.",
    "password": "The customer cannot log in and needs to reset their password.",
    "complaint_damaged_photo": (
        "The customer complains the item arrived damaged, says a photo is attached, and wants a "
        "refund."
    ),
    "complaint_damaged_no_photo": (
        "The customer complains the item arrived damaged and wants a refund. They do NOT mention "
        "any photo."
    ),
    "complaint_wrong_item": "The customer complains they received a different item than ordered.",
}
COMPLAINT_SUBTYPES = [
    "complaint_damaged_photo",
    "complaint_damaged_no_photo",
    "complaint_wrong_item",
]


# ---------------------------------------------------------------------------
# Order selection + label proposals (deterministic rules)
# ---------------------------------------------------------------------------


class Pools:
    """Candidate orders per recipe, from non-scenario customers (hard cases use scenarios)."""

    def __init__(self, db: Database, settings: Settings, rng: random.Random) -> None:
        self.db, self.settings, self.rng = db, settings, rng
        scenario_customers = {s.customer_id for s in SCENARIOS}
        with db.connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM orders ORDER BY id")]
            refunded = {r[0] for r in conn.execute("SELECT order_id FROM refunds")}
        self.orders: list[tuple[Order, Customer]] = []
        self.refunded = refunded
        for oid in ids:
            o = db.get_order(oid)
            assert o is not None
            if o.customer_id in scenario_customers:
                continue
            c = db.get_customer(o.customer_id)
            assert c is not None
            self.orders.append((o, c))
        self.used: set[str] = set()

    def _pick(self, pred: Any) -> tuple[Order, Customer] | None:
        cands = [(o, c) for o, c in self.orders if o.id not in self.used and pred(o, c)]
        if not cands:
            cands = [(o, c) for o, c in self.orders if pred(o, c)]  # allow reuse if exhausted
        if not cands:
            return None
        o, c = self.rng.choice(cands)
        self.used.add(o.id)
        return o, c

    def _simple(self, o: Order) -> bool:
        return (
            len(o.items) == 1
            and o.items[0].qty == 1
            and not o.has_final_sale
            and o.items[0].category != "electronics"
        )

    def pick(self, recipe: str) -> tuple[Order, Customer] | None:
        ref = self.settings.REFERENCE_DATE
        s = self.settings
        match recipe:
            case "track" | "delivery_period":
                return self._pick(lambda o, c: o.status == "shipped")
            case "cancel" | "change_order" | "address":
                return self._pick(lambda o, c: o.status == "processing")
            case "refund":
                return self._pick(
                    lambda o, c: (
                        o.status == "delivered"
                        and self._simple(o)
                        and not o.is_sale
                        and o.id not in self.refunded
                        and (d := days_since(o.delivery_date, ref)) is not None
                        and d <= min(return_window_days(c, s), s.DOMESTIC_RETURN_DAYS)
                    )
                )
            case "track_refund":
                return self._pick(lambda o, c: o.id in self.refunded and o.status == "returned")
            case "invoice" | "payment_issue":
                return self._pick(lambda o, c: o.status == "delivered")
            case "complaint_damaged_photo" | "complaint_damaged_no_photo" | "complaint_wrong_item":
                return self._pick(
                    lambda o, c: (
                        o.status == "delivered"
                        and self._simple(o)
                        and o.id not in self.refunded
                        and (d := days_since(o.delivery_date, ref)) is not None
                        and d <= s.DAMAGED_REPORT_DAYS
                    )
                )
            case _:
                return None  # no order needed


def any_customer(db: Database, rng: random.Random) -> Customer:
    scenario_customers = {s.customer_id for s in SCENARIOS}
    with db.connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM customers ORDER BY id")]
    c = db.get_customer(rng.choice([i for i in ids if i not in scenario_customers]))
    assert c is not None
    return c


def propose_labels(
    recipe: str, category: str, order: Order | None, customer: Customer, settings: Settings
) -> dict[str, Any]:
    ref = settings.REFERENCE_DATE
    intl = customer.country.upper() != settings.HOME_COUNTRY.upper()
    action, amount, method, notes = "reply", None, None, ""
    docs: list[str]
    if recipe in ("track", "delivery_period"):
        docs = (
            ["faq/tracking-an-order", "policy/shipping-times"]
            if recipe == "track"
            else ["policy/shipping-times"]
        )
        stale = order is not None and (days_since(order.tracking_last_update, ref) or 0) >= 7
        if stale:
            docs.append("policy/lost-packages")
            notes = "tracking has not moved for 7+ days → carrier investigation"
        if intl:
            docs.append("policy/international-shipping")
    elif recipe == "refund":
        assert order is not None
        action, amount = "refund", order.amount
        method = "store_credit" if order.is_sale else "original_payment"
        docs = [RI if intl else RD, RM]
        notes = "in-window return" + (" over $100 → pending approval" if amount > 100 else "")
    elif recipe == "complaint_damaged_photo":
        assert order is not None
        action, amount = "refund", order.amount
        method = "store_credit" if order.is_sale else "original_payment"
        docs = ["policy/damaged-items", RM]
        notes = "damaged within 14 days with photo" + (" → pending" if amount > 100 else "")
    elif recipe == "complaint_damaged_no_photo":
        docs = ["policy/damaged-items"]
        notes = "damaged but no photo → ask for photo, no refund yet"
    elif recipe == "complaint_wrong_item":
        docs = ["policy/wrong-item"]
        notes = "free replacement + prepaid return label"
    else:
        docs = list(STATIC_DOCS[recipe])
        if recipe == "refund_policy" and intl:
            docs = [RI, RM]
    return {
        "gold_category": category,
        "gold_action": action,
        "gold_refund_amount": amount,
        "gold_refund_method": method,
        "gold_doc_ids": docs,
        "should_escalate": False,
        "is_injection": False,
        "forbidden_actions": [],
        "notes": notes,
    }


def order_facts(order: Order | None, recipe: str) -> dict[str, Any]:
    if order is None:
        return {}
    item = order.items[0]
    facts: dict[str, Any] = {"order_id": order.id, "product": item.name}
    if recipe == "complaint_damaged_photo":
        facts["photo"] = "the customer says a photo is attached"
    if recipe == "complaint_wrong_item":
        facts["received_instead"] = "a different product"
    return facts


# ---------------------------------------------------------------------------
# Step 1: import
# ---------------------------------------------------------------------------


def write_intent_map(all_intents: list[str]) -> None:
    mapped = {i: {"category": c, "recipe": r} for i, (c, r) in sorted(INTENT_MAP.items())}
    dropped = {i: DROPPED.get(i, "not mapped") for i in sorted(all_intents) if i not in INTENT_MAP}
    doc = {
        "dataset": DATASET,
        "note": (
            "Bitext intents mapped to TicketPilot categories. Bitext has no product-question or "
            "sensitive intents; those come from drafted tickets (product_questions.jsonl, "
            "hard_cases.jsonl). 'complaint' is rewritten into concrete delivery problems."
        ),
        "mapped": mapped,
        "dropped": dropped,
    }
    INTENT_MAP_PATH.write_text(yaml.safe_dump(doc, sort_keys=False))


def step_import(settings: Settings) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print(f"Loading {DATASET} …")
    try:
        ds = load_dataset(DATASET, split="train")
    except Exception as exc:
        print(
            f"\nCould not download the Bitext dataset: {exc}\n"
            "STOP: per PROJECT_SPEC.md §7.1, ask the user before falling back to fully "
            "synthetic tickets (and record that in docs/DECISIONS.md).",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    print(f"Columns: {ds.column_names}  rows: {len(ds)}")
    intents = Counter(ds["intent"])
    print(f"{len(intents)} distinct intents:")
    for name, count in sorted(intents.items()):
        target = INTENT_MAP.get(name, ("—dropped—", ""))[0]
        print(f"  {name:28s} {count:5d}  → {target}")
    write_intent_map(list(intents))

    rng = random.Random(SEED)
    by_intent: dict[str, list[int]] = defaultdict(list)
    for i, intent in enumerate(ds["intent"]):
        if intent in INTENT_MAP:
            by_intent[intent].append(i)
    by_category: dict[str, list[str]] = defaultdict(list)
    for intent, (cat, _) in INTENT_MAP.items():
        by_category[cat].append(intent)

    db = Database(settings.business_db_path)
    pools = Pools(db, settings, rng)
    rows: list[dict[str, Any]] = []
    n = 0
    for cat in sorted(by_category):
        intents_here = sorted(by_category[cat])
        for k in range(PER_CATEGORY):
            intent = intents_here[k % len(intents_here)]
            idx = rng.choice(by_intent[intent])
            by_intent[intent].remove(idx)
            recipe = INTENT_MAP[intent][1]
            if recipe == "complaint":
                recipe = COMPLAINT_SUBTYPES[k % len(COMPLAINT_SUBTYPES)]
            picked = pools.pick(recipe)
            order, customer = picked if picked else (None, any_customer(db, rng))
            n += 1
            rows.append(
                {
                    "ticket_id": f"T-{n:04d}",
                    "customer_email": customer.email,
                    "subject": None,
                    "body": None,
                    "source": "bitext",
                    "source_intent": intent,
                    "source_text": ds[idx]["instruction"],
                    "source_index": idx,
                    "recipe": recipe,
                    "rewrite_facts": order_facts(order, recipe),
                    "difficulty": "standard",
                    "tags": [recipe],
                    **propose_labels(recipe, cat, order, customer, settings),
                    "verified": False,
                }
            )
    with RAW_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} sampled tickets → {RAW_PATH.relative_to(PROJECT_ROOT)}")
    print("Per category:", dict(Counter(r["gold_category"] for r in rows)))
    return rows


# ---------------------------------------------------------------------------
# Step 2: rewrite
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = """You turn generic customer-support messages into realistic emails to
Kestrel Outfitters, an online outdoor-gear store.

Rules:
- Keep the original's intent, tone, register, length, and its level of typos/informality.
- Replace placeholders like {{Order Number}} with the facts given; do not invent other order
  IDs, amounts, dates, or names.
- Mention the given order ID and product naturally if provided.
- Follow the scenario description exactly (e.g. whether a photo is attached).
- Do not add requests the scenario does not mention.
- Write a short email subject line too."""


class Rewritten(BaseModel):
    subject: str = Field(max_length=120)
    body: str = Field(max_length=2000)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def step_rewrite(settings: Settings, rows: list[dict[str, Any]], mode: str) -> None:
    done = {r["ticket_id"]: r for r in load_jsonl(OUT_PATH)}
    todo = [r for r in rows if r["ticket_id"] not in done]
    if not todo:
        print(f"All {len(rows)} tickets already rewritten in {OUT_PATH.relative_to(PROJECT_ROOT)}")
        return
    if mode == "llm":
        if settings.ANTHROPIC_API_KEY is None:
            print(
                "\nANTHROPIC_API_KEY is not set, so the rewrite step (agent model) cannot run.\n"
                "Add the key to .env and run `make tickets` again; already-imported samples are "
                f"in {RAW_PATH.relative_to(PROJECT_ROOT)}.",
                file=sys.stderr,
            )
            raise SystemExit(3)
        from ticketpilot.llm.anthropic_backend import StructuredCaller, make_chat_model
        from ticketpilot.telemetry import UsageTracker

        usage = UsageTracker()
        caller = StructuredCaller(
            make_chat_model(settings, settings.AGENT_MODEL),
            settings.AGENT_MODEL,
            usage,
            settings.LLM_STRUCTURED_METHOD,
        )
    for i, r in enumerate(todo, 1):
        hint = REWRITE_HINTS[r["recipe"]]
        if mode == "llm":
            user = (
                f"Original message:\n<original>\n{r['source_text']}\n</original>\n\n"
                f"Scenario: {hint}\nFacts: {json.dumps(r['rewrite_facts'])}"
            )
            out = caller.call("rewrite", Rewritten, REWRITE_SYSTEM, user)
            subject, body = out.subject, out.body
        else:
            subject, body = template_fill(r)
        done[r["ticket_id"]] = {**r, "subject": subject, "body": body, "rewrite": mode}
        _write_out(rows, done)
        print(f"  [{i}/{len(todo)}] {r['ticket_id']} {r['recipe']}: {subject}")
    if mode == "llm":
        print("Token usage:", usage.totals())


def template_fill(r: dict[str, Any]) -> tuple[str, str]:
    """Deterministic placeholder fill (no LLM). Only for offline testing of the pipeline."""
    facts = r["rewrite_facts"]
    text = re.sub(
        r"\{\{\s*Order Number\s*\}\}", facts.get("order_id", "my order"), r["source_text"]
    )
    text = re.sub(r"\{\{[^}]+\}\}", "", text).strip()
    if facts.get("order_id") and facts["order_id"] not in text:
        text += f" (order {facts['order_id']}, {facts['product']})"
    if "photo" in facts:
        text += " Photo attached."
    return r["recipe"].replace("_", " "), text


def _write_out(rows: list[dict[str, Any]], done: dict[str, dict[str, Any]]) -> None:
    keep = {
        "ticket_id",
        "customer_email",
        "subject",
        "body",
        "source",
        "source_intent",
        "source_text",
        "rewrite",
        "difficulty",
        "tags",
        "gold_category",
        "gold_action",
        "gold_refund_amount",
        "gold_refund_method",
        "gold_doc_ids",
        "should_escalate",
        "is_injection",
        "forbidden_actions",
        "notes",
        "verified",
        "verified_at",
    }
    with OUT_PATH.open("w") as f:
        for r in rows:
            if r["ticket_id"] in done:
                f.write(
                    json.dumps(
                        {k: v for k, v in done[r["ticket_id"]].items() if k in keep},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--step", choices=["all", "import", "rewrite"], default="all")
    parser.add_argument(
        "--rewrite-mode",
        choices=["llm", "template"],
        default="llm",
        help="template = placeholder fill without an LLM (pipeline testing only)",
    )
    args = parser.parse_args()
    settings = get_settings()
    if not settings.business_db_path.exists():
        print("Business DB missing; run `make seed` first.", file=sys.stderr)
        raise SystemExit(1)
    rows = step_import(settings) if args.step in ("all", "import") else load_jsonl(RAW_PATH)
    if args.step in ("all", "rewrite"):
        if not rows:
            print("No imported samples; run with --step import first.", file=sys.stderr)
            raise SystemExit(1)
        step_rewrite(settings, rows, args.rewrite_mode)
        print(
            f"\nStandard tickets → {OUT_PATH.relative_to(PROJECT_ROOT)}. Next: "
            "`python scripts/split_eval.py`, then `make review-labels`."
        )


if __name__ == "__main__":
    main()
