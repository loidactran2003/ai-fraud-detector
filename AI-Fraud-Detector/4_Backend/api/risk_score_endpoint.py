"""Risk-scoring routes (Flask blueprint).

POST /api/score                 score one order, store it, return score + action + reasons
                                ?dry_run=1 (or "persist": false) scores without storing anything
POST /api/score/batch           score a list of orders (same ?dry_run=1)
GET  /api/transactions          list scored orders   ?min_score=30&level=high&limit=100&offset=0&unreviewed=1
                                -> {"transactions": [...], "total": N}
POST /api/transactions/<id>/review   record a reviewer decision {"decision": "blocked", "reviewer": "...", "note": "..."}
                                confirmed_fraud / false_positive also set the order's label (for retraining)
GET  /api/stats                 dashboard aggregates (counts, $ at risk, histogram, top reasons)

Optional auth: set FRAUD_API_KEY and every write route (score, batch, review) requires the
header `X-API-Key: <key>`. Off by default so the local dashboard works out of the box.
"""
from __future__ import annotations

import hmac
import math
import numbers
import os
import uuid
from datetime import datetime

import pandas as pd
from flask import Blueprint, current_app, jsonify, request

from connection import (amount_median, context_history, get_connection, query_transactions, save_assessment,
                        save_review, stats)
from preprocessing import parse_bool, prepare_single, to_naive_datetime

bp = Blueprint("risk", __name__, url_prefix="/api")

REQUIRED = ["user_id", "amount"]
DECISIONS = {"approved", "blocked", "confirmed_fraud", "false_positive"}
LEVELS = {"low", "medium", "high"}
ID_FIELDS = ("transaction_id", "user_id", "device_id", "card_hash")
BOOL_FIELDS = ("is_vpn", "is_proxy", "is_emulator", "is_rooted", "timezone_mismatch")
COUNT_FIELDS = ("login_attempts", "failed_logins", "pages_viewed")   # non-negative integers
SECONDS_FIELDS = ("checkout_seconds",)                               # non-negative numbers
COUNTRY_FIELDS = ("billing_country", "ip_country")
TS_FORMAT = "%Y-%m-%d %H:%M:%S"   # canonical form stored in the DB (string order == time order)
MAX_LIMIT = 2000


class TransactionConflict(Exception):
    """A transaction_id that is already stored for a different order."""


@bp.errorhandler(TransactionConflict)
def _conflict(err):
    return jsonify(error=str(err)), 409


@bp.before_request
def _require_api_key():
    key = os.getenv("FRAUD_API_KEY")
    if not key or request.method != "POST":
        return None
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), key):
        return jsonify(error="missing or invalid X-API-Key"), 401
    return None


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) or (isinstance(v, str) and v.strip() != "")


def _finite(v) -> float:
    if isinstance(v, bool) or not _is_number(v):
        raise ValueError
    x = float(v)
    if not math.isfinite(x):
        raise ValueError
    return x


def _parse_time(v):
    t = to_naive_datetime(v)
    if pd.isna(t):
        raise ValueError
    return t


def _validate(order) -> str | None:
    if not isinstance(order, dict):
        return "each order must be a JSON object"
    missing = [f for f in REQUIRED if order.get(f) in (None, "")]
    if missing:
        return f"missing required field(s): {', '.join(missing)}"
    for f in ID_FIELDS:
        if order.get(f) is not None and (isinstance(order[f], bool) or not isinstance(order[f], (str, int))):
            return f"{f} must be a string or integer"
    try:
        if _finite(order["amount"]) <= 0:
            return "amount must be positive"
    except (TypeError, ValueError):
        return "amount must be a finite number"
    for f in ("timestamp", "account_created"):
        if order.get(f) not in (None, ""):
            try:
                _parse_time(order[f])
            except (TypeError, ValueError, OverflowError):
                return f"{f} must be an ISO 8601 date/time"
    for f in BOOL_FIELDS:
        try:
            parse_bool(order.get(f))
        except ValueError:
            return f"{f} must be a boolean (true/false, 1/0, yes/no)"
    for f in COUNT_FIELDS:
        if order.get(f) not in (None, ""):
            try:
                x = _finite(order[f])
            except (TypeError, ValueError):
                return f"{f} must be a non-negative integer"
            if x < 0 or x != int(x):
                return f"{f} must be a non-negative integer"
    for f in SECONDS_FIELDS:
        if order.get(f) not in (None, ""):
            try:
                if _finite(order[f]) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                return f"{f} must be a non-negative number"
    for f in COUNTRY_FIELDS:
        v = order.get(f)
        if v not in (None, "") and not (isinstance(v, str) and len(v.strip()) == 2 and v.strip().isalpha()):
            return f"{f} must be a 2-letter ISO country code"
    return None


def _normalise(order: dict) -> dict:
    """Validated order -> the types the feature pipeline and the DB expect."""
    order = dict(order)
    # ids are TEXT in the DB (and strings in training data): 777 and "777" must be the same account
    for key in ID_FIELDS:
        if order.get(key) is not None:
            order[key] = str(order[key])
    if not order.get("transaction_id"):
        order["transaction_id"] = f"T-{uuid.uuid4().hex[:10]}"
    # canonical timestamps: stored created_at strings must sort in time order for point-in-time history
    order["timestamp"] = (_parse_time(order["timestamp"]) if order.get("timestamp")
                          else pd.Timestamp(datetime.now())).strftime(TS_FORMAT)
    if order.get("account_created") not in (None, ""):
        order["account_created"] = _parse_time(order["account_created"]).strftime(TS_FORMAT)
    else:
        order.pop("account_created", None)   # unknown: derived from the user's earlier orders, else "unknown"
    if order.get("device_id") is None:
        order["device_id"] = f"unknown-{order['user_id']}"
    if order.get("card_hash") is None:
        order["card_hash"] = f"unknown-{order['user_id']}"
    order["amount"] = float(order["amount"])
    for f in BOOL_FIELDS:
        order[f] = parse_bool(order.get(f))
    for f in COUNT_FIELDS:
        if order.get(f) in (None, ""):
            order.pop(f, None)               # missing -> neutral default in build_features
        else:
            order[f] = int(float(order[f]))
    for f in SECONDS_FIELDS:
        if order.get(f) in (None, ""):
            order.pop(f, None)
        else:
            order[f] = float(order[f])
    return order


def _dry_run(body) -> bool:
    if request.args.get("dry_run", "").lower() in ("1", "true", "yes"):
        return True
    return isinstance(body, dict) and body.get("persist") is False


def _score_orders(orders: list[dict], persist: bool = True) -> list[dict]:
    scorer = current_app.config["SCORER"]
    db_path = current_app.config["DB_PATH"]
    orders = [_normalise(o) for o in orders]
    # read only the stored orders that share a user / device / card with this request
    with get_connection(db_path) as conn:
        history = context_history(conn, orders)
        known = dict(zip(history["transaction_id"], zip(history["user_id"], history["amount"])))
        stored = set(known)   # ids already in the DB (only these need excluding from the SQL median)
        # score without holding the write lock, so other requests aren't blocked meanwhile
        scored, out = [], []
        delta: list[tuple[str, str, float]] = []   # (timestamp, transaction_id, amount) scored earlier in this batch
        replaced: set[str] = set()                 # stored ids re-scored earlier in this batch
        for order in orders:
            tid = order["transaction_id"]
            past = history
            if tid in known:
                user, amount = known[tid]
                if user != order["user_id"] or round(float(amount), 2) != round(order["amount"], 2):
                    raise TransactionConflict(f"transaction_id {tid} is already used by a different order")
                past = history[history["transaction_id"] != tid]   # a retry must not see itself as history
            # store-wide median of earlier orders: computed in SQL, plus this batch's earlier orders
            g_median = amount_median(conn, before=order["timestamp"], exclude_ids=(replaced | {tid}) & stored,
                                     extra=[a for ts, t, a in delta if ts <= order["timestamp"] and t != tid])
            row = prepare_single(order, past, global_median=g_median)
            result = scorer.score(row)[0]
            feats = {f: float(row.iloc[0][f]) for f in scorer.features}
            scored.append((order, feats, result))
            # later orders in the same batch must see this one, as they would one at a time
            new = {"transaction_id": tid, "user_id": order["user_id"], "device_id": order["device_id"],
                   "card_hash": order["card_hash"], "amount": order["amount"], "timestamp": order["timestamp"]}
            history = pd.concat([past, pd.DataFrame([new])[list(history.columns)]], ignore_index=True)
            if tid in known:
                replaced.add(tid)
            delta = [d for d in delta if d[1] != tid] + [(order["timestamp"], tid, order["amount"])]
            known[tid] = (order["user_id"], order["amount"])
            out.append({"transaction_id": tid, **result, "model_version": scorer.version, "features": feats,
                        **({} if persist else {"persisted": False})})
    if persist:
        with get_connection(db_path) as conn:   # one short write transaction for the whole batch
            for order, feats, result in scored:
                save_assessment(conn, order, feats, result, scorer.version)
    return out


@bp.post("/score")
def score_one():
    order = request.get_json(silent=True)
    if order is None:
        order = {}
    if err := _validate(order):
        return jsonify(error=err), 400
    return jsonify(_score_orders([order], persist=not _dry_run(order))[0])


@bp.post("/score/batch")
def score_batch():
    body = request.get_json(silent=True)
    orders = body if isinstance(body, list) else body.get("transactions") if isinstance(body, dict) else None
    if not orders or not isinstance(orders, list):
        return jsonify(error="send {'transactions': [...]}"), 400
    if len(orders) > 1000:
        return jsonify(error="max 1000 orders per batch"), 413
    errors = {i: e for i, o in enumerate(orders) if (e := _validate(o))}
    if errors:
        return jsonify(error="invalid orders", details=errors), 400
    return jsonify(results=_score_orders(orders, persist=not _dry_run(body)))


@bp.get("/transactions")
def list_transactions():
    limit = request.args.get("limit", 200, type=int)
    offset = request.args.get("offset", 0, type=int)
    min_score = request.args.get("min_score", 0, type=int)
    level = request.args.get("level") or None
    if limit is None or limit < 1:
        return jsonify(error=f"limit must be an integer between 1 and {MAX_LIMIT}"), 400
    if offset is None or offset < 0:
        return jsonify(error="offset must be a non-negative integer"), 400
    if min_score is None or not 0 <= min_score <= 100:
        return jsonify(error="min_score must be an integer between 0 and 100"), 400
    if level is not None and level not in LEVELS:
        return jsonify(error=f"level must be one of {sorted(LEVELS)}"), 400
    with get_connection(current_app.config["DB_PATH"]) as conn:
        rows, total = query_transactions(conn, min_score=min_score, level=level, limit=min(limit, MAX_LIMIT),
                                         offset=offset, unreviewed_only=request.args.get("unreviewed") == "1",
                                         with_total=True)
    return jsonify(transactions=rows, total=total, offset=offset)


@bp.post("/transactions/<transaction_id>/review")
def review(transaction_id):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    decision = body.get("decision")
    if decision not in DECISIONS:
        return jsonify(error=f"decision must be one of {sorted(DECISIONS)}"), 400
    with get_connection(current_app.config["DB_PATH"]) as conn:
        # only scored orders can be reviewed; unscored history rows never reach the dashboards
        if not conn.execute("SELECT 1 FROM risk_assessments WHERE transaction_id = ?", (transaction_id,)).fetchone():
            return jsonify(error="unknown transaction"), 404
        save_review(conn, transaction_id, decision, body.get("reviewer"), body.get("note"))
    return jsonify(ok=True, transaction_id=transaction_id, decision=decision)


@bp.get("/stats")
def get_stats():
    with get_connection(current_app.config["DB_PATH"]) as conn:
        return jsonify(stats(conn))
