"""Deterministic mock business data for Kestrel Outfitters.

`data/seed/seed_db.py` is a thin CLI wrapper around `seed_database`. All dates are
relative to `Settings.REFERENCE_DATE` so window logic is reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from ticketpilot.db import Database

SEED = 42
N_CUSTOMERS = 120
N_ORDERS = 400


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price: float
    category: str  # tent | sleeping_bag | jacket | boots | backpack | electronics | ...


CATALOG: list[Product] = [
    Product("TNT-RL2", "Ridgeline 2 Tent", 189.00, "tent"),
    Product("TNT-RL3", "Ridgeline 3 Tent", 239.00, "tent"),
    Product("TNT-TL1", "Talus 1 Tent", 149.00, "tent"),
    Product("TNT-BC4", "Basecamp 4 Tent", 329.00, "tent"),
    Product("TNT-AS2", "Aspen 2 Four-Season Tent", 449.00, "tent"),
    Product("BAG-C20", "Cirrus 20° Sleeping Bag", 219.00, "sleeping_bag"),
    Product("BAG-C00", "Cirrus 0° Sleeping Bag", 329.00, "sleeping_bag"),
    Product("BAG-D35", "Drift 35° Synthetic Sleeping Bag", 99.00, "sleeping_bag"),
    Product("JKT-SQL", "Squall Rain Jacket", 159.00, "jacket"),
    Product("JKT-SQLL", "Squall Lite Rain Jacket", 119.00, "jacket"),
    Product("JKT-EMB", "Ember Down Jacket", 199.00, "jacket"),
    Product("JKT-HFL", "Hearth Fleece Pullover", 64.00, "jacket"),
    Product("BT-SCM", "Scree Mid Hiking Boot", 169.00, "boots"),
    Product("BT-SCL", "Scree Low Hiking Shoe", 139.00, "boots"),
    Product("BT-TUN", "Tundra Winter Boot", 209.00, "boots"),
    Product("PK-C50", "Cascade 50 Backpack", 229.00, "backpack"),
    Product("PK-C65", "Cascade 65 Backpack", 269.00, "backpack"),
    Product("PK-R22", "Ridge 22 Daypack", 89.00, "backpack"),
    Product("HL-B300", "Beacon 300 Headlamp", 34.95, "headlamp"),
    Product("HL-B600R", "Beacon 600R Rechargeable Headlamp", 59.95, "electronics"),
    Product("GPS-TM", "Trailmark GPS", 299.00, "electronics"),
    Product("STV-KC", "Kindle Canister Stove", 59.00, "stove"),
    Product("STV-KM", "Kindle Multi-Fuel Stove", 139.00, "stove"),
    Product("ACC-SWP", "Switchback Trekking Poles", 79.00, "accessory"),
    Product("ACC-CWF", "Clearwater Water Filter", 45.00, "accessory"),
    Product("ACC-SOCK", "Merino Hiking Socks", 18.00, "accessory"),
]
BY_SKU = {p.sku: p for p in CATALOG}

FIRST_NAMES = [
    "Alex", "Jamie", "Taylor", "Jordan", "Casey", "Riley", "Morgan", "Avery", "Quinn", "Drew",
    "Sofia", "Mateo", "Isla", "Kai", "Maya", "Leo", "Ruby", "Finn", "Nora", "Eli", "Iris",
    "Owen", "Lila", "Theo", "Hazel", "Jonah", "Clara", "Felix", "Ada", "Milo",
]  # fmt: skip
LAST_NAMES = [
    "Park", "Silva", "Novak", "Hughes", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Byrne",
    "Costa", "Ivanova", "Haddad", "Kowalski", "Mendes", "Sato", "Walsh", "Nguyen", "Duarte",
    "Fraser", "Keller", "Abbott", "Reid", "Santos", "Holm", "Doyle",
]  # fmt: skip
INTL_COUNTRIES = ["CA", "GB", "DE", "AU", "FR", "NL", "SE", "JP"]


@dataclass
class Scenario:
    key: str
    order_id: str
    customer_id: str
    email: str
    name: str
    purpose: str
    items: list[tuple[str, int, float | None, bool, bool]]  # sku, qty, price override, sale, final
    status: str
    delivered_days_ago: int | None = None
    order_days_ago: int | None = None
    tracking_days_ago: int | None = None
    tier: str = "trail"
    country: str = "US"
    shipping_method: str = "standard"
    extra_orders: list[dict[str, Any]] = field(default_factory=list)
    existing_refund: dict[str, Any] | None = None


SCENARIOS: list[Scenario] = [
    Scenario("damaged_tent_over_limit", "KO-10482", "C-0031", "sam.ortiz@example.com",
             "Sam Ortiz", "Delivered $189 Ridgeline 2 tent 5 days ago (damaged claim; over limit)",
             [("TNT-RL2", 1, None, False, False)], "delivered", delivered_days_ago=5),
    Scenario("sale_jacket", "KO-10511", "C-0042", "priya.nair@example.com", "Priya Nair",
             "Sale-item Squall jacket ($119.25) delivered 10 days ago; store credit only",
             [("JKT-SQL", 1, 119.25, True, False)], "delivered", delivered_days_ago=10),
    Scenario("final_sale", "KO-10523", "C-0047", "marcus.webb@example.com", "Marcus Webb",
             "Final-sale Ember Down Jacket delivered 8 days ago; not returnable",
             [("JKT-EMB", 1, 139.30, False, True)], "delivered", delivered_days_ago=8),
    Scenario("already_refunded", "KO-10537", "C-0052", "lena.fischer@example.com",
             "Lena Fischer", "Cascade 50 order that already has an approved refund",
             [("PK-C50", 1, None, False, False)], "returned", delivered_days_ago=20,
             existing_refund={"id": "RF-SEED0001", "amount": 229.00, "reason": "return: too small",
                              "method": "original_payment", "status": "approved",
                              "approved_by": "j.alvarez"}),
    Scenario("international_20_days", "KO-10548", "C-0058", "olivia.brown@example.co.uk",
             "Olivia Brown", "International (GB) Cirrus 20° delivered 20 days ago; outside 14 days",
             [("BAG-C20", 1, None, False, False)], "delivered", delivered_days_ago=20,
             country="GB", shipping_method="international"),
    Scenario("stuck_tracking", "KO-10559", "C-0063", "daniel.kim@example.com", "Daniel Kim",
             "Shipped Basecamp 4 with no tracking movement for 9 days",
             [("TNT-BC4", 1, None, False, False)], "shipped", order_days_ago=14,
             tracking_days_ago=9),
    Scenario("processing", "KO-10560", "C-0067", "aisha.bello@example.com", "Aisha Bello",
             "Order still processing (can be changed/cancelled)",
             [("BT-SCM", 1, None, False, False), ("ACC-SOCK", 2, None, False, False)],
             "processing", order_days_ago=1),
    Scenario("under_limit_item", "KO-10571", "C-0071", "tom.reyes@example.com", "Tom Reyes",
             "$64 Hearth Fleece delivered 6 days ago (refund under limit)",
             [("JKT-HFL", 1, None, False, False)], "delivered", delivered_days_ago=6),
    Scenario("opened_gps", "KO-10583", "C-0074", "nina.patel@example.com", "Nina Patel",
             "Trailmark GPS ($299) delivered 12 days ago; opened electronics → 10% fee",
             [("GPS-TM", 1, None, False, False)], "delivered", delivered_days_ago=12),
    Scenario("peak_45_days", "KO-10594", "C-0078", "grace.liu@example.com", "Grace Liu",
             "Peak-tier customer, Ridgeline 3 delivered 45 days ago (inside 60-day window)",
             [("TNT-RL3", 1, None, False, False)], "delivered", delivered_days_ago=45,
             tier="peak", shipping_method="express"),
    Scenario("domestic_outside_window", "KO-10605", "C-0081", "ben.carter@example.com",
             "Ben Carter", "Trail-tier Squall Lite delivered 40 days ago (outside 30 days)",
             [("JKT-SQLL", 1, None, False, False)], "delivered", delivered_days_ago=40),
    Scenario("damaged_late", "KO-10616", "C-0085", "hannah.moore@example.com", "Hannah Moore",
             "Cirrus 0° delivered 20 days ago; damage reported after 14-day window",
             [("BAG-C00", 1, None, False, False)], "delivered", delivered_days_ago=20),
    Scenario("multi_order_customer", "KO-10627", "C-0088", "jose.martinez@example.com",
             "Jose Martinez", "Customer with 3 orders; used for email-lookup tickets",
             [("HL-B300", 1, None, False, False)], "delivered", delivered_days_ago=3,
             extra_orders=[
                 {"id": "KO-10628", "items": [("STV-KC", 1, None, False, False)],
                  "status": "shipped", "order_days_ago": 3, "tracking_days_ago": 1,
                  "shipping_method": "express"},
                 {"id": "KO-10629", "items": [("ACC-SWP", 1, None, False, False)],
                  "status": "delivered", "delivered_days_ago": 70},
             ]),
    Scenario("wrong_item", "KO-10638", "C-0092", "emma.wilson@example.com", "Emma Wilson",
             "Ridge 22 daypack delivered 4 days ago (customer received wrong item)",
             [("PK-R22", 1, None, False, False)], "delivered", delivered_days_ago=4),
    Scenario("damaged_under_limit", "KO-10649", "C-0095", "liam.johnson@example.com",
             "Liam Johnson", "Kindle Canister Stove ($59) delivered 3 days ago; damaged claim",
             [("STV-KC", 1, None, False, False)], "delivered", delivered_days_ago=3),
    Scenario("lost_package", "KO-10651", "C-0097", "chloe.martin@example.com", "Chloe Martin",
             "Ridge 22 daypack marked lost; no tracking movement for 15 days",
             [("PK-R22", 1, None, False, False)], "lost", order_days_ago=20, tracking_days_ago=15),
    Scenario("international_recent", "KO-10662", "C-0099", "lucas.schmidt@example.de",
             "Lucas Schmidt", "International (DE) Cascade 65 delivered 6 days ago (inside 14 days)",
             [("PK-C65", 1, None, False, False)], "delivered", delivered_days_ago=6,
             country="DE", shipping_method="international"),
    Scenario("sale_under_limit", "KO-10673", "C-0102", "mia.rossi@example.com", "Mia Rossi",
             "Sale Drift 35° ($74.25) delivered 9 days ago; store credit only, under limit",
             [("BAG-D35", 1, 74.25, True, False)], "delivered", delivered_days_ago=9),
    Scenario("double_charge", "KO-10684", "C-0105", "noah.evans@example.com", "Noah Evans",
             "Ember Down Jacket ($199) delivered 15 days ago; double-charge tickets",
             [("JKT-EMB", 1, None, False, False)], "delivered", delivered_days_ago=15,
             tier="summit"),
    Scenario("warranty", "KO-10695", "C-0108", "zoe.turner@example.com", "Zoe Turner",
             "Cascade 50 delivered ~10 months ago; warranty claims",
             [("PK-C50", 1, None, False, False)], "delivered", delivered_days_ago=300),
    Scenario("price_drop", "KO-10706", "C-0110", "ethan.clark@example.com", "Ethan Clark",
             "Squall jacket ordered 7 days ago; price-adjustment requests",
             [("JKT-SQL", 1, None, False, False)], "delivered", delivered_days_ago=3),
    Scenario("domestic_return_ok", "KO-10717", "C-0112", "ava.hughes@example.com", "Ava Hughes",
             "Scree Low ($139) delivered 12 days ago; plain domestic return (over limit)",
             [("BT-SCL", 1, None, False, False)], "delivered", delivered_days_ago=12),
    Scenario("small_return_ok", "KO-10728", "C-0114", "owen.price@example.com", "Owen Price",
             "Trekking poles ($79) delivered 10 days ago; return under limit",
             [("ACC-SWP", 1, None, False, False)], "delivered", delivered_days_ago=10),
]  # fmt: skip


def _d(ref: date, days_ago: int | None) -> str | None:
    return None if days_ago is None else (ref - timedelta(days=days_ago)).isoformat()


def _items_json(items: list[tuple[str, int, float | None, bool, bool]]) -> tuple[str, float, bool]:
    out = []
    for sku, qty, price, sale, final in items:
        p = BY_SKU[sku]
        out.append(
            {
                "sku": sku,
                "name": p.name,
                "qty": qty,
                "unit_price": round(price if price is not None else p.price, 2),
                "is_sale": sale,
                "final_sale": final,
                "category": p.category,
            }
        )
    amount = round(sum(i["unit_price"] * i["qty"] for i in out), 2)  # type: ignore[operator]
    return json.dumps(out), amount, any(i["is_sale"] for i in out)


def _order_row(
    ref: date,
    order_id: str,
    customer_id: str,
    items: list[tuple[str, int, float | None, bool, bool]],
    status: str,
    shipping_method: str,
    delivered_days_ago: int | None = None,
    order_days_ago: int | None = None,
    tracking_days_ago: int | None = None,
) -> dict[str, Any]:
    items_json, amount, is_sale = _items_json(items)
    ship_ago: int | None
    tracking_ago: int | None
    if delivered_days_ago is not None:
        transit = 10 if shipping_method == "international" else 4
        ship_ago = delivered_days_ago + transit
        order_ago = order_days_ago if order_days_ago is not None else ship_ago + 1
        tracking_ago = delivered_days_ago
    else:
        order_ago = order_days_ago if order_days_ago is not None else 1
        ship_ago = None if status == "processing" else max(order_ago - 1, 0)
        tracking_ago = tracking_days_ago if status != "processing" else None
    return {
        "id": order_id,
        "customer_id": customer_id,
        "items_json": items_json,
        "amount": amount,
        "is_sale": int(is_sale),
        "status": status,
        "order_date": _d(ref, order_ago),
        "ship_date": _d(ref, ship_ago),
        "delivery_date": _d(ref, delivered_days_ago),
        "tracking_last_update": _d(ref, tracking_ago),
        "shipping_method": shipping_method,
    }


def build_seed_data(reference_date: date) -> dict[str, list[dict[str, Any]]]:
    """Build all rows deterministically. Pure function of the reference date."""
    rng = random.Random(SEED)
    scenario_customers = {s.customer_id: s for s in SCENARIOS}

    customers: list[dict[str, Any]] = []
    used_emails: set[str] = set()
    for n in range(1, N_CUSTOMERS + 1):
        cid = f"C-{n:04d}"
        if cid in scenario_customers:
            s = scenario_customers[cid]
            row = {
                "id": cid,
                "email": s.email,
                "name": s.name,
                "loyalty_tier": s.tier,
                "country": s.country,
            }
        else:
            first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            email = f"{first.lower()}.{last.lower()}@example.com"
            k = 2
            while email in used_emails:
                email = f"{first.lower()}.{last.lower()}{k}@example.com"
                k += 1
            country = rng.choice(INTL_COUNTRIES) if rng.random() < 0.15 else "US"
            tier = rng.choices(["trail", "summit", "peak"], weights=[70, 22, 8])[0]
            row = {
                "id": cid,
                "email": email,
                "name": f"{first} {last}",
                "loyalty_tier": tier,
                "country": country,
            }
        used_emails.add(row["email"])
        customers.append(row)

    orders: list[dict[str, Any]] = []
    refunds: list[dict[str, Any]] = []
    reserved: set[int] = set()
    for s in SCENARIOS:
        orders.append(
            _order_row(
                reference_date,
                s.order_id,
                s.customer_id,
                s.items,
                s.status,
                s.shipping_method,
                s.delivered_days_ago,
                s.order_days_ago,
                s.tracking_days_ago,
            )
        )
        reserved.add(int(s.order_id[3:]))
        for extra in s.extra_orders:
            orders.append(
                _order_row(
                    reference_date,
                    extra["id"],
                    s.customer_id,
                    extra["items"],
                    extra["status"],
                    extra.get("shipping_method", s.shipping_method),
                    extra.get("delivered_days_ago"),
                    extra.get("order_days_ago"),
                    extra.get("tracking_days_ago"),
                )
            )
            reserved.add(int(extra["id"][3:]))
        if s.existing_refund:
            refund_day = _d(reference_date, (s.delivered_days_ago or 0) - 6)
            refunds.append(
                {
                    **s.existing_refund,
                    "order_id": s.order_id,
                    "created_at": f"{refund_day}T10:00:00+00:00",
                }
            )

    random_customers = [c for c in customers if c["id"] not in scenario_customers]
    candidates = [n for n in range(10001, 19999) if n not in reserved]
    order_numbers = sorted(rng.sample(candidates, N_ORDERS - len(orders)))
    statuses = (
        ["delivered"] * 62
        + ["shipped"] * 14
        + ["processing"] * 10
        + ["lost"] * 4
        + ["returned"] * 10
    )
    for num in order_numbers:
        cust = rng.choice(random_customers)
        intl = cust["country"] != "US"
        n_items = rng.choices([1, 2, 3], weights=[65, 25, 10])[0]
        items: list[tuple[str, int, float | None, bool, bool]] = []
        for p in rng.sample(CATALOG, n_items):
            roll = rng.random()
            qty = 2 if p.category == "accessory" and rng.random() < 0.3 else 1
            if roll < 0.05:
                items.append((p.sku, qty, round(p.price * 0.7, 2), False, True))
            elif roll < 0.20:
                items.append((p.sku, qty, round(p.price * 0.75, 2), True, False))
            else:
                items.append((p.sku, qty, None, False, False))
        status = rng.choice(statuses)
        method = "international" if intl else rng.choices(["standard", "express"], [80, 20])[0]
        if status in ("delivered", "returned"):
            row = _order_row(
                reference_date,
                f"KO-{num:05d}",
                cust["id"],
                items,
                status,
                method,
                delivered_days_ago=rng.randint(1, 160),
            )
        elif status == "shipped":
            ago = rng.randint(1, 6)
            row = _order_row(
                reference_date,
                f"KO-{num:05d}",
                cust["id"],
                items,
                status,
                method,
                order_days_ago=ago + 1,
                tracking_days_ago=rng.randint(0, ago),
            )
        elif status == "lost":
            row = _order_row(
                reference_date,
                f"KO-{num:05d}",
                cust["id"],
                items,
                status,
                method,
                order_days_ago=rng.randint(18, 40),
                tracking_days_ago=rng.randint(10, 17),
            )
        else:
            row = _order_row(
                reference_date,
                f"KO-{num:05d}",
                cust["id"],
                items,
                status,
                method,
                order_days_ago=rng.randint(0, 2),
            )
        orders.append(row)
        if status == "returned" and not any(i[3] or i[4] for i in items):
            refunds.append(
                {
                    "id": f"RF-SEED{len(refunds) + 1:04d}",
                    "order_id": row["id"],
                    "amount": row["amount"],
                    "reason": "return received",
                    "method": "original_payment",
                    "status": "approved",
                    "approved_by": "auto" if float(row["amount"]) <= 100 else "j.alvarez",
                    "created_at": f"{row['delivery_date']}T12:00:00+00:00",
                }
            )
    orders.sort(key=lambda o: o["id"])
    return {"customers": customers, "orders": orders, "refunds": refunds}


def seed_database(db_path: Path, reference_date: date, scenarios_path: Path | None = None) -> None:
    """(Re)create the business DB at `db_path` from scratch."""
    if db_path.exists():
        db_path.unlink()
    db = Database(db_path)
    db.init_schema()
    data = build_seed_data(reference_date)
    with db.connect() as conn:
        conn.executemany(
            "INSERT INTO customers (id, email, name, loyalty_tier, country) "
            "VALUES (:id, :email, :name, :loyalty_tier, :country)",
            data["customers"],
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, items_json, amount, is_sale, status, order_date, "
            "ship_date, delivery_date, tracking_last_update, shipping_method) VALUES (:id, "
            ":customer_id, :items_json, :amount, :is_sale, :status, :order_date, :ship_date, "
            ":delivery_date, :tracking_last_update, :shipping_method)",
            data["orders"],
        )
        conn.executemany(
            "INSERT INTO refunds (id, order_id, amount, reason, method, status, created_at, "
            "approved_by) VALUES (:id, :order_id, :amount, :reason, :method, :status, "
            ":created_at, :approved_by)",
            data["refunds"],
        )
    if scenarios_path is not None:
        write_scenarios(scenarios_path, reference_date)


def scenario_records() -> list[dict[str, Any]]:
    out = []
    for s in SCENARIOS:
        out.append(
            {
                "key": s.key,
                "order_id": s.order_id,
                "extra_order_ids": [e["id"] for e in s.extra_orders],
                "customer_id": s.customer_id,
                "customer_email": s.email,
                "purpose": s.purpose,
            }
        )
    return out


def write_scenarios(path: Path, reference_date: date) -> None:
    payload = {"reference_date": reference_date.isoformat(), "scenarios": scenario_records()}
    header = "# Generated by data/seed/seed_db.py — do not edit by hand.\n"
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
