-- AI Fraud Detector - SQLite schema (portable to PostgreSQL with minor type changes)

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    device_id        TEXT,
    card_hash        TEXT,                 -- hashed / tokenised card, never the PAN
    amount           REAL NOT NULL,
    currency         TEXT DEFAULT 'USD',
    billing_country  TEXT,
    ip_country       TEXT,
    ip_address       TEXT,
    is_vpn           INTEGER DEFAULT 0,
    is_proxy         INTEGER DEFAULT 0,
    created_at       TEXT NOT NULL,        -- order timestamp (ISO 8601)
    features_json    TEXT,                 -- engineered features used for scoring
    is_fraud_label   INTEGER               -- ground truth once known (chargeback, review)
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id   TEXT NOT NULL REFERENCES transactions(transaction_id),
    risk_score       INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_level       TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    action           TEXT NOT NULL CHECK (action IN ('approve', 'review', 'block')),
    reasons_json     TEXT NOT NULL DEFAULT '[]',
    model_version    TEXT NOT NULL,
    scored_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reviews (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id   TEXT NOT NULL REFERENCES transactions(transaction_id),
    decision         TEXT NOT NULL CHECK (decision IN ('approved', 'blocked', 'confirmed_fraud', 'false_positive')),
    reviewer         TEXT,
    note             TEXT,
    reviewed_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tx_user   ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_tx_device ON transactions(device_id);
CREATE INDEX IF NOT EXISTS idx_tx_card   ON transactions(card_hash);
CREATE INDEX IF NOT EXISTS idx_ra_tx     ON risk_assessments(transaction_id);
CREATE INDEX IF NOT EXISTS idx_ra_score  ON risk_assessments(risk_score);
CREATE INDEX IF NOT EXISTS idx_tx_created ON transactions(created_at);           -- point-in-time history / median
CREATE INDEX IF NOT EXISTS idx_tx_amount_created ON transactions(amount, created_at); -- store-wide median without a sort
CREATE INDEX IF NOT EXISTS idx_ra_tx_id   ON risk_assessments(transaction_id, id); -- latest assessment per order
CREATE INDEX IF NOT EXISTS idx_rv_tx_id   ON reviews(transaction_id, id);          -- latest review per order

-- Latest assessment + latest review per transaction (what the dashboards read).
-- Driven from risk_assessments, so the unscored history rows in `transactions` are never scanned.
-- Dropped and recreated on every init_db so existing databases pick up changes (same columns as before).
DROP VIEW IF EXISTS v_transaction_risk;
CREATE VIEW v_transaction_risk AS
SELECT t.*, ra.risk_score, ra.risk_level, ra.action, ra.reasons_json, ra.model_version, ra.scored_at,
       rv.decision AS review_decision, rv.reviewed_at
FROM risk_assessments ra
JOIN transactions t ON t.transaction_id = ra.transaction_id
LEFT JOIN reviews rv ON rv.id = (SELECT MAX(id) FROM reviews WHERE transaction_id = ra.transaction_id)
WHERE ra.id = (SELECT MAX(id) FROM risk_assessments WHERE transaction_id = ra.transaction_id);
