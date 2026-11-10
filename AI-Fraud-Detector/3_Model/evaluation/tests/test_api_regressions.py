"""Regression tests for the scoring API and shared feature pipeline (bugs BUG-1..BUG-7 in
.claude/reports/02_bugs.md). Each test runs against a fresh temporary SQLite DB.

Run from the project root:  python -m unittest discover -s 3_Model/evaluation/tests -t 3_Model
"""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation import config

for _p in ("4_Backend/api", "4_Backend/database", "4_Backend/utils"):
    if str(config.ROOT / _p) not in sys.path:
        sys.path.insert(0, str(config.ROOT / _p))

from connection import get_connection  # noqa: E402
from fraud_detection_api import create_app  # noqa: E402
from preprocessing import add_context, clean_transactions  # noqa: E402

CONTEXT = ["accounts_per_device", "users_per_card", "amount_to_baseline"]


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls._tmp.name) / "test.db"
        cls.client = create_app(cls.db).test_client()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def score(self, **order):
        r = self.client.post("/api/score", json=order)
        self.assertEqual(r.status_code, 200, r.get_data())
        return r.get_json()

    def add_history_row(self, tid, user, amount, label):
        with sqlite3.connect(self.db) as con:
            con.execute("INSERT INTO transactions (transaction_id, user_id, device_id, card_hash, amount, "
                        "created_at, is_fraud_label) VALUES (?, ?, 'HD', 'HC', ?, '2026-01-01 00:00:00', ?)",
                        (tid, user, amount, label))


class TestApiRegressions(ApiTestCase):
    def test_numeric_user_id_is_the_same_account_as_stored_text_id(self):  # BUG-1
        order = {"user_id": 777, "amount": 20, "device_id": "DV777", "card_hash": "CH777"}
        r = self.client.post("/api/score/batch", json={"transactions": [order] * 3})
        self.assertEqual(r.status_code, 200)
        feats = self.score(**order)["features"]
        self.assertEqual(feats["accounts_per_device"], 1.0)
        self.assertEqual(feats["users_per_card"], 1.0)
        self.assertEqual(feats["amount_to_baseline"], 1.0)   # own baseline (3 earlier orders of 20)

    def test_connections_wait_for_other_writers(self):  # BUG-2
        with get_connection(self.db) as conn:
            self.assertGreaterEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 30000)

    def test_review_of_unscored_history_order_is_rejected(self):  # BUG-3
        self.add_history_row("H-UNSCORED", "HU", 10.0, 0)
        r = self.client.post("/api/transactions/H-UNSCORED/review", json={"decision": "confirmed_fraud"})
        self.assertEqual(r.status_code, 404)
        tid = self.score(user_id="RV", amount=10)["transaction_id"]
        r = self.client.post(f"/api/transactions/{tid}/review", json={"decision": "approved"})
        self.assertEqual(r.status_code, 200)

    def test_retry_with_same_transaction_id_gets_same_score(self):  # BUG-4
        for _ in range(2):
            self.score(user_id="RR", amount=10, device_id="RD", card_hash="RC")
        order = {"user_id": "RR", "amount": 900, "device_id": "RD", "card_hash": "RC",
                 "transaction_id": "T-RETRY-1", "timestamp": "2026-02-01 10:00:00"}
        first, retry = self.score(**order), self.score(**order)
        self.assertEqual(first["risk_score"], retry["risk_score"])
        self.assertEqual(first["features"], retry["features"])

    def test_rescoring_keeps_label_and_rejects_reused_id(self):  # BUG-4
        self.add_history_row("H-LABELLED", "HL", 8.63, 1)
        r = self.client.post("/api/score", json={"user_id": "X", "amount": 10, "transaction_id": "H-LABELLED"})
        self.assertEqual(r.status_code, 409)
        self.score(user_id="HL", amount=8.63, transaction_id="H-LABELLED")
        with sqlite3.connect(self.db) as con:
            row = con.execute("SELECT user_id, amount, is_fraud_label FROM transactions "
                              "WHERE transaction_id = 'H-LABELLED'").fetchone()
        self.assertEqual(row, ("HL", 8.63, 1))

    def test_batch_accepts_bare_json_list(self):  # BUG-6
        r = self.client.post("/api/score/batch", json=[{"user_id": "LB", "amount": 10}])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()["results"]), 1)
        self.assertEqual(self.client.post("/api/score/batch", json=["nope"]).status_code, 400)

    def test_invalid_timestamp_is_a_400(self):  # BUG-7
        r = self.client.post("/api/score", json={"user_id": "TS", "amount": 10, "timestamp": "not a date"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("timestamp", r.get_json()["error"])


class TestMissingUserId(unittest.TestCase):  # BUG-5
    def log(self):
        return pd.DataFrame({"transaction_id": ["T1", "T2"], "user_id": [None, None], "device_id": ["D", "D"],
                             "card_hash": ["C", "C"], "amount": [10.0, 20.0],
                             "timestamp": ["2026-01-01", "2026-01-02"]})

    def test_batch_and_serving_agree_without_user_id(self):
        log = self.log()
        batch = add_context(log)[CONTEXT].iloc[1].tolist()
        serve = add_context(log.iloc[[1]], log.iloc[:1])[CONTEXT].iloc[0].tolist()
        self.assertEqual(batch, serve)
        self.assertEqual(batch[:2], [1, 1])

    def test_cleaning_drops_orders_without_user_id(self):
        raw = self.log().assign(billing_country="US", ip_country="US", is_vpn=0, is_proxy=0)
        raw.loc[0, "user_id"] = "A"
        self.assertEqual(clean_transactions(raw)["transaction_id"].tolist(), ["T1"])


if __name__ == "__main__":
    unittest.main()
