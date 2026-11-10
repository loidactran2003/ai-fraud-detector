"""SQLite access layer. One file DB, zero setup; swap DB_URL handling for Postgres in production.

    python 4_Backend/database/connection.py --seed            # create tables + load scored demo orders
    python 4_Backend/database/connection.py --seed --force    # ... even if reviewer decisions exist (they are backed up first)
    python 4_Backend/database/connection.py --export-labels labels.csv   # reviewed / labelled orders + features
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", ROOT / "4_Backend" / "database" / "fraud_detector.db"))
SCHEMA = Path(__file__).with_name("schema.sql")

TX_COLS = ["transaction_id", "user_id", "device_id", "card_hash", "amount", "billing_country",
           "ip_country", "is_vpn", "is_proxy"]
BUSY_TIMEOUT_S = 30  # wait this long for another writer instead of failing with "database is locked"
# reviewer decision -> ground-truth label. approved / blocked are actions, not evidence, so they stay unlabelled.
DECISION_LABELS = {"confirmed_fraud": 1, "false_positive": 0}


class ReviewsExist(RuntimeError):
    """Seeding would delete reviewer decisions."""


@contextmanager
def get_connection(path: Path = DB_PATH):
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(path) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))


def save_assessment(conn, row: dict, features: dict, result: dict, model_version: str, label=None) -> None:
    # upsert, not INSERT OR REPLACE: re-scoring a known order must keep its ground-truth label
    cols = [*TX_COLS, "ip_address", "created_at", "features_json"]
    conn.execute(
        f"""INSERT INTO transactions ({", ".join(cols)}, is_fraud_label)
            VALUES ({", ".join("?" * len(cols))}, ?)
            ON CONFLICT(transaction_id) DO UPDATE SET
              {", ".join(f"{c} = excluded.{c}" for c in cols[1:])},
              is_fraud_label = COALESCE(excluded.is_fraud_label, transactions.is_fraud_label)""",
        [row.get(c) for c in TX_COLS] + [row.get("ip_address"), str(row.get("timestamp")),
                                          json.dumps(features, default=float), label])
    conn.execute(
        """INSERT INTO risk_assessments (transaction_id, risk_score, risk_level, action, reasons_json, model_version)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (row["transaction_id"], result["risk_score"], result["risk_level"], result["action"],
         json.dumps(result["reasons"]), model_version))


def save_review(conn, transaction_id: str, decision: str, reviewer: str | None, note: str | None) -> None:
    conn.execute("INSERT INTO reviews (transaction_id, decision, reviewer, note) VALUES (?, ?, ?, ?)",
                 (transaction_id, decision, reviewer, note))
    # a reviewer's verdict is a label the next training run can use (see --export-labels)
    if decision in DECISION_LABELS:
        conn.execute("UPDATE transactions SET is_fraud_label = ? WHERE transaction_id = ?",
                     (DECISION_LABELS[decision], transaction_id))


def history_frame(conn) -> pd.DataFrame:
    """Every known order's linkage fields. Reference / debugging only: the API uses
    context_history + amount_median, which read only what one request needs."""
    return pd.read_sql("SELECT transaction_id, user_id, device_id, card_hash, amount, created_at AS timestamp "
                       "FROM transactions", conn)


def context_history(conn, orders: list[dict]) -> pd.DataFrame:
    """Stored orders that share a user, device or card with `orders` (or reuse their ids).

    That is everything add_context needs for these orders except the store-wide median
    (amount_median), so a request reads O(matching rows), not the whole table.
    Uses the idx_tx_user / idx_tx_device / idx_tx_card indexes and the primary key.
    """
    clauses, args = [], []
    for col in ("user_id", "device_id", "card_hash", "transaction_id"):
        vals = sorted({str(o[col]) for o in orders if o.get(col) is not None})
        for i in range(0, len(vals), 500):  # stay far below SQLite's bound-parameter limit
            chunk = vals[i:i + 500]
            clauses.append(f"{col} IN ({', '.join('?' * len(chunk))})")
            args += chunk
    sql = "SELECT transaction_id, user_id, device_id, card_hash, amount, created_at AS timestamp FROM transactions"
    if not clauses:
        return pd.read_sql(sql + " WHERE 0", conn)
    return pd.read_sql(sql + " WHERE " + " OR ".join(clauses), conn, params=args)


def amount_median(conn, before: str | None = None, exclude_ids=(), extra=()) -> float:
    """Exact median amount of stored orders placed at or before `before` (all if None),
    leaving out `exclude_ids`, plus the not-yet-stored amounts in `extra`.

    Only a COUNT and a small ORDER BY ... LIMIT window are read from SQLite: the union's
    middle element(s) can only sit within len(extra) ranks of the stored median.
    Same result as pandas' median over the full history (NaN when there is none).
    """
    where, args = ["1 = 1"], []
    if before is not None:
        where.append("created_at <= ?")
        args.append(str(before))
    ex = sorted({str(i) for i in exclude_ids})
    if ex:
        where.append(f"transaction_id NOT IN ({', '.join('?' * len(ex))})")
        args += ex
    w = " AND ".join(where)
    n = conn.execute(f"SELECT COUNT(*) FROM transactions WHERE {w}", args).fetchone()[0]
    extra = sorted(float(x) for x in extra)
    k, total = len(extra), n + len(extra)
    if total == 0:
        return math.nan
    lo, hi = (total - 1) // 2, total // 2          # 0-based ranks of the middle element(s)
    start, stop = max(0, lo - k), min(n, hi + 1)   # stored ranks that can land there
    window = [r[0] for r in conn.execute(
        f"SELECT amount FROM transactions WHERE {w} ORDER BY amount LIMIT ? OFFSET ?",
        [*args, stop - start, start])] if stop > start else []
    merged = sorted(window + extra)
    return (merged[lo - start] + merged[hi - start]) / 2


def query_transactions(conn, min_score: int = 0, level: str | None = None, limit: int = 200,
                       unreviewed_only: bool = False, offset: int = 0, with_total: bool = False):
    where = " WHERE risk_score >= ?"
    args: list = [min_score]
    if level:
        where += " AND risk_level = ?"
        args.append(level)
    if unreviewed_only:
        where += " AND review_decision IS NULL"
    sql = "SELECT * FROM v_transaction_risk" + where + " ORDER BY risk_score DESC, created_at DESC LIMIT ? OFFSET ?"
    rows = [dict(r) for r in conn.execute(sql, [*args, limit, offset])]
    for r in rows:
        r["reasons"] = json.loads(r.pop("reasons_json") or "[]")
        r.pop("features_json", None)
    if not with_total:
        return rows
    total = conn.execute("SELECT COUNT(*) FROM v_transaction_risk" + where, args).fetchone()[0]
    return rows, total


def stats(conn) -> dict:
    agg = dict(conn.execute("""
        SELECT COUNT(*) AS orders,
               SUM(action = 'review') AS review, SUM(action = 'block') AS blocked, SUM(action = 'approve') AS approved,
               ROUND(SUM(CASE WHEN action != 'approve' THEN amount ELSE 0 END), 2) AS amount_at_risk,
               ROUND(AVG(risk_score), 1) AS avg_score,
               SUM(review_decision IS NULL AND action != 'approve') AS awaiting_review
        FROM v_transaction_risk""").fetchone())
    hist = conn.execute("""SELECT (risk_score / 5) * 5 AS bucket, COUNT(*) AS n FROM v_transaction_risk
                           GROUP BY bucket ORDER BY bucket""").fetchall()
    agg["histogram"] = [{"bucket": r["bucket"], "count": r["n"]} for r in hist]
    reasons: dict[str, int] = {}
    for (rj,) in conn.execute("SELECT reasons_json FROM v_transaction_risk WHERE action != 'approve'"):
        for r in json.loads(rj or "[]")[:1]:
            reasons[r["feature"]] = reasons.get(r["feature"], 0) + 1
    agg["top_reasons"] = sorted(({"feature": k, "count": v} for k, v in reasons.items()), key=lambda d: -d["count"])
    return agg


def export_labels(conn, out: Path) -> int:
    """Labelled orders (seeded ground truth + reviewer verdicts) with the features they were scored on."""
    rows = conn.execute("""SELECT transaction_id, user_id, created_at AS timestamp, is_fraud_label, features_json
                           FROM transactions WHERE is_fraud_label IS NOT NULL AND features_json IS NOT NULL""").fetchall()
    recs = [{"transaction_id": r["transaction_id"], "user_id": r["user_id"], "timestamp": r["timestamp"],
             **json.loads(r["features_json"]), "is_fraud": int(r["is_fraud_label"])} for r in rows]
    pd.DataFrame(recs).to_csv(out, index=False)
    return len(recs)


def _backup_reviews(conn, path: Path) -> Path | None:
    reviews = pd.read_sql("SELECT * FROM reviews", conn)
    if reviews.empty:
        return None
    out = path.with_name(f"reviews_backup_{datetime.now():%Y%m%d-%H%M%S}.csv")
    reviews.to_csv(out, index=False)
    return out


def seed_demo_data(path: Path = DB_PATH, limit: int | None = None, force: bool = False) -> int:
    """Score the held-out test set and load it so the dashboards have realistic data.

    This replaces every stored order. If reviewer decisions exist it refuses unless
    force=True, and even then writes them to reviews_backup_<time>.csv first.
    """
    sys.path.insert(0, str(ROOT / "3_Model"))
    from risk_scoring import RiskScorer

    init_db(path)
    with get_connection(path) as conn:
        n_reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if n_reviews and not force:
            raise ReviewsExist(f"{path} has {n_reviews} reviewer decision(s); seeding would delete them. "
                               "Re-run with --force (they are backed up to a CSV first).")
        backup = _backup_reviews(conn, path) if n_reviews else None
    if backup:
        print(f"backed up {n_reviews} review(s) to {backup}")

    test = pd.read_csv(ROOT / "3_Model" / "saved_models" / "test_set.csv")
    if limit:
        test = test.tail(limit)
    # Orders placed before the demo period: stored (unscored) as history only, so live
    # scoring computes device/card re-use and spend baselines over the same past the
    # model was trained with. They have no assessment, so dashboards don't show them.
    all_orders = pd.read_csv(ROOT / "2_Data" / "processed_data" / "feature_engineered.csv")
    past = all_orders[~all_orders.transaction_id.isin(test.transaction_id)
                      & (all_orders.timestamp < test.timestamp.min())]
    scorer = RiskScorer()
    results = scorer.score(test)
    with get_connection(path) as conn:
        conn.execute("DELETE FROM reviews"); conn.execute("DELETE FROM risk_assessments"); conn.execute("DELETE FROM transactions")
        conn.executemany(
            f"""INSERT OR REPLACE INTO transactions ({", ".join(TX_COLS)}, created_at, is_fraud_label)
                VALUES ({", ".join("?" * len(TX_COLS))}, ?, ?)""",
            [[r.get(c) for c in TX_COLS] + [str(r["timestamp"]), int(r["is_fraud"])]
             for r in past.to_dict("records")])
        for row, res in zip(test.to_dict("records"), results):
            feats = {f: row[f] for f in scorer.features}
            save_assessment(conn, row, feats, res, scorer.version, label=int(row["is_fraud"]))
    return len(test)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "4_Backend" / "utils"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true", help="load scored demo orders from the test set (replaces all orders)")
    ap.add_argument("--force", action="store_true", help="seed even if reviewer decisions exist (backed up to CSV first)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--export-labels", type=Path, default=None, metavar="CSV",
                    help="write labelled orders (incl. reviewer verdicts) + their features to CSV")
    a = ap.parse_args()
    init_db()
    if a.seed:
        try:
            print(f"seeded {seed_demo_data(limit=a.limit, force=a.force)} scored orders into {DB_PATH}")
        except ReviewsExist as e:
            sys.exit(f"refusing to seed: {e}")
    elif a.export_labels:
        with get_connection() as conn:
            print(f"wrote {export_labels(conn, a.export_labels)} labelled orders to {a.export_labels}")
    else:
        print(f"initialised {DB_PATH}")
