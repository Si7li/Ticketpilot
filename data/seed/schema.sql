-- TicketPilot schema (PROJECT_SPEC.md §6.2). Business tables + application tables.
CREATE TABLE customers (
  id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  loyalty_tier TEXT NOT NULL CHECK (loyalty_tier IN ('trail','summit','peak')),
  country TEXT NOT NULL DEFAULT 'US'
);
CREATE TABLE orders (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
  items_json TEXT NOT NULL,             -- [{"sku","name","qty","unit_price","is_sale","final_sale","category"}]
  amount REAL NOT NULL, is_sale INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('processing','shipped','delivered','lost','returned')),
  order_date TEXT NOT NULL, ship_date TEXT, delivery_date TEXT,
  tracking_last_update TEXT, shipping_method TEXT NOT NULL
);
CREATE TABLE refunds (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
  amount REAL NOT NULL, reason TEXT NOT NULL,
  method TEXT NOT NULL CHECK (method IN ('original_payment','store_credit')),
  status TEXT NOT NULL CHECK (status IN ('approved','pending_approval','rejected')),
  created_at TEXT NOT NULL, approved_by TEXT
);
CREATE UNIQUE INDEX one_active_refund_per_order ON refunds(order_id) WHERE status != 'rejected';

-- application tables
CREATE TABLE tickets (
  id TEXT PRIMARY KEY, customer_email TEXT NOT NULL, subject TEXT, body TEXT NOT NULL,
  status TEXT NOT NULL,                -- received | auto_sent | awaiting_review | escalated | closed
  route TEXT, result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE escalations (
  id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, team TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE outbox (
  id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, to_email TEXT NOT NULL, body TEXT NOT NULL,
  sent_by TEXT NOT NULL,               -- 'auto' | reviewer name
  created_at TEXT NOT NULL
);
CREATE TABLE review_events (
  id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, reviewer TEXT NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('approve','edit_approve','reject')),
  edited_reply TEXT, duration_seconds REAL, created_at TEXT NOT NULL
);
