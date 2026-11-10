"""Regression tests for the improvements in .claude/reports/05_repairs.md (IMP-1 .. IMP-19).
Each API test runs against a fresh temporary SQLite DB.

Run from the project root:  python -m unittest discover -s 3_Model/evaluation/tests -t 3_Model
"""
import contextlib
import os
import random
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from evaluation import config

for _p in ("4_Backend/api", "4_Backend/database", "4_Backend/utils", "3_Model"):
    if str(config.ROOT / _p) not in sys.path:
        sys.path.insert(0, str(config.ROOT / _p))

import connection  # noqa: E402
from fraud_detection_api import create_app  # noqa: E402
from preprocessing import FEATURES, UNKNOWN_ACCOUNT_AGE, parse_bool, prepare_single  # noqa: E402
from risk_scoring import FeatureContractError, RiskScorer  # noqa: E402


@contextlib.contextmanager
def _sql(db):
    """sqlite3's own context manager commits but never closes - Windows then can't delete the temp DB."""
    con = sqlite3.connect(db)
    try:
        with con:
            yield con
    finally:
        con.close()


class _Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls._tmp.name) / "test.db"
        cls.app = create_app(cls.db)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def post(self, path="/api/score", **order):
        return self.client.post(path, json=order)

    def feats(self, **order):
        r = self.post(**order)
        self.assertEqual(r.status_code, 200, r.get_data())
        return r.get_json()["features"]

    def add_history(self, rows):
        with _sql(self.db) as con:
            con.executemany("INSERT INTO transactions (transaction_id, user_id, device_id, card_hash, amount, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)", rows)


class TestInputHandling(_Api):
    def test_string_booleans_are_parsed_strictly(self):  # IMP-2
        f = self.feats(user_id="SB", amount=20, is_vpn="false", is_proxy="0", is_emulator="no", timezone_mismatch="TRUE",
                       dry_run=None)
        self.assertEqual((f["is_vpn"], f["is_proxy"], f["is_emulator"], f["timezone_mismatch"]), (0.0, 0.0, 0.0, 1.0))
        r = self.post(user_id="SB", amount=20, is_vpn="maybe")
        self.assertEqual(r.status_code, 400)
        self.assertIn("is_vpn", r.get_json()["error"])
        self.assertEqual(parse_bool(True), 1)
        self.assertRaises(ValueError, parse_bool, 2)

    def test_bad_numbers_are_json_400s(self):  # IMP-3
        cases = [{"checkout_seconds": "abc"}, {"amount": float("nan")}, {"amount": float("inf")}, {"pages_viewed": -5},
                 {"login_attempts": 2.7}, {"user_id": {"a": 1}}, {"ip_country": "USA"}, {"amount": True},
                 {"account_created": "garbage"}]
        for extra in cases:
            r = self.post(**{"user_id": "BN", "amount": 10, **extra})
            self.assertEqual(r.status_code, 400, extra)
            self.assertTrue(r.is_json, extra)

    def test_errors_are_json(self):  # IMP-3
        r = self.client.get("/api/score")
        self.assertEqual(r.status_code, 405)
        self.assertTrue(r.is_json)
        with mock.patch("fraud_detection_api.log"), \
                mock.patch.object(connection, "stats", side_effect=RuntimeError("boom")), \
                mock.patch("risk_score_endpoint.stats", side_effect=RuntimeError("boom")):
            r = self.client.get("/api/stats")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.get_json(), {"error": "internal server error"})

    def test_listing_parameters_are_validated(self):  # IMP-12, IMP-15
        self.assertEqual(self.client.get("/api/transactions?limit=-1").status_code, 400)
        self.assertEqual(self.client.get("/api/transactions?level=bogus").status_code, 400)
        for i in range(3):
            self.post(user_id=f"LP{i}", amount=10)
        body = self.client.get("/api/transactions?limit=2").get_json()
        self.assertEqual(len(body["transactions"]), 2)
        self.assertGreaterEqual(body["total"], 3)
        page2 = self.client.get("/api/transactions?limit=2&offset=2").get_json()["transactions"]
        self.assertTrue({t["transaction_id"] for t in page2}.isdisjoint({t["transaction_id"] for t in body["transactions"]}))


class TestAccountAge(_Api):
    def test_missing_account_created_is_unknown_not_brand_new(self):  # IMP-1
        f = self.feats(user_id="AGE-NEW", amount=400)
        self.assertEqual(f["account_age_known"], 0.0)
        self.assertEqual(f["account_age_days"], UNKNOWN_ACCOUNT_AGE)
        self.assertEqual(f["new_account_high_spend"], 0.0)

    def test_missing_account_created_falls_back_to_first_order(self):  # IMP-1
        self.add_history([("AG1", "AGE-OLD", "AD", "AC", 30.0, "2025-01-01 10:00:00"),
                          ("AG2", "AGE-OLD", "AD", "AC", 40.0, "2025-06-01 10:00:00")])
        f = self.feats(user_id="AGE-OLD", amount=400, device_id="AD", card_hash="AC", timestamp="2026-01-01 10:00:00")
        self.assertEqual(f["account_age_known"], 1.0)
        self.assertEqual(f["account_age_days"], 365.0)

    def test_minimal_payload_is_not_sent_to_review_for_age(self):  # IMP-1 (reviewer's probe)
        r = self.post(user_id="AGE-MIN", amount=400, timestamp="2026-10-01 14:00:00").get_json()
        old = self.post(user_id="AGE-MIN2", amount=400, timestamp="2026-10-01 14:00:00",
                        account_created="2024-10-01 14:00:00").get_json()
        self.assertFalse(any(x["feature"] in ("account_age_days",) for x in r["reasons"]), r["reasons"])
        self.assertFalse(any("New account" in x["text"] for x in r["reasons"]), r["reasons"])
        self.assertLessEqual(r["risk_score"], old["risk_score"] + 15)

    def test_timezone_aware_times_are_accepted(self):
        f = self.feats(user_id="TZ", amount=20, timestamp="2026-10-01T14:00:00Z", account_created="2026-09-30T14:00:00+02:00")
        self.assertEqual(f["hour"], 14.0)
        self.assertEqual(f["account_age_days"], 1.0)


class TestServingHistory(_Api):
    def test_back_dated_order_does_not_see_the_future(self):  # IMP-13
        self.add_history([("BD1", "BD-OTHER", "BD-DEV", "BD-CARD", 10.0, "2026-06-01 10:00:00")])
        early = self.feats(user_id="BD", amount=10, device_id="BD-DEV", card_hash="BD-CARD",
                           timestamp="2026-05-01 10:00:00", dry_run=None)
        late = self.feats(user_id="BD", amount=10, device_id="BD-DEV", card_hash="BD-CARD",
                          timestamp="2026-07-01 10:00:00", transaction_id="BD-LATE")
        self.assertEqual(early["accounts_per_device"], 1.0)
        self.assertEqual(late["accounts_per_device"], 2.0)

    def test_sql_context_matches_full_history_reference(self):  # IMP-4 parity
        rng = random.Random(0)
        rows = [(f"P{i}", f"PU{rng.randint(0, 9)}", f"PD{rng.randint(0, 5)}", f"PC{rng.randint(0, 5)}",
                 round(rng.uniform(1, 300), 2), f"2026-03-{1 + i % 28:02d} 1{i % 10}:00:00") for i in range(120)]
        self.add_history(rows)
        order = {"user_id": "PU3", "amount": 250.0, "device_id": "PD2", "card_hash": "PC4",
                 "timestamp": "2026-03-20 12:00:00", "account_created": "2025-01-01 00:00:00"}
        api = self.feats(**order, transaction_id="P-NEW")
        with connection.get_connection(self.db) as conn:
            full = connection.history_frame(conn)
        full = full[full.transaction_id != "P-NEW"]
        ref = prepare_single(order, full).iloc[0]
        for f in FEATURES:
            self.assertAlmostEqual(api[f], float(ref[f]), places=6, msg=f)

    def test_amount_median_matches_pandas(self):  # IMP-4
        rng = random.Random(1)
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            connection.init_db(db)
            rows = [(f"M{i}", "U", 0, f"2026-01-{1 + i % 28:02d} 00:00:00", round(rng.uniform(0, 100), 2)) for i in range(57)]
            with _sql(db) as con:
                con.executemany("INSERT INTO transactions (transaction_id, user_id, amount, created_at, is_vpn) "
                                "VALUES (?, ?, ?, ?, ?)", [(t, u, a, c, v) for t, u, v, c, a in rows])
            frame = pd.DataFrame(rows, columns=["tid", "u", "v", "ts", "amount"])
            with connection.get_connection(db) as conn:
                for before in (None, "2026-01-10 00:00:00", "2026-01-01 00:00:00", "2025-01-01 00:00:00"):
                    for extra in ([], [5.0], [1.0, 99.0, 50.0, 50.0, 7.5]):
                        for excl in ((), ("M3", "M10")):
                            sub = frame if before is None else frame[frame.ts <= before]
                            sub = sub[~sub.tid.isin(excl)]
                            want = pd.concat([sub.amount, pd.Series(extra, dtype=float)]).median()
                            got = connection.amount_median(conn, before, excl, extra)
                            if pd.isna(want):
                                self.assertTrue(pd.isna(got))
                            else:
                                self.assertAlmostEqual(got, want, places=9, msg=(before, extra, excl))

    def test_batch_matches_one_at_a_time_with_sql_context(self):  # IMP-4 batch delta
        orders = [{"user_id": f"BU{i % 2}", "amount": 10.0 * (i + 1), "device_id": "BDV", "card_hash": f"BC{i % 3}",
                   "timestamp": f"2026-04-01 10:0{i}:00", "transaction_id": f"BB{i}"} for i in range(6)]
        batch = self.client.post("/api/score/batch?dry_run=1", json=orders).get_json()["results"]
        single = [self.feats(**o) for o in orders]   # persisted one at a time
        for b, s in zip(batch, single):
            self.assertEqual(b["features"], s, b["transaction_id"])


class TestPersistenceAndLabels(_Api):
    def count(self, sql):
        with _sql(self.db) as con:
            return con.execute(sql).fetchone()[0]

    def test_dry_run_does_not_store(self):  # IMP-17
        before = self.count("SELECT COUNT(*) FROM transactions")
        r = self.client.post("/api/score?dry_run=1", json={"user_id": "DRY", "amount": 10})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.get_json()["persisted"])
        r = self.post(user_id="DRY", amount=10, persist=False)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.count("SELECT COUNT(*) FROM transactions"), before)

    def test_review_verdict_becomes_label(self):  # IMP-7
        tid = self.post(user_id="LBL", amount=10).get_json()["transaction_id"]
        self.client.post(f"/api/transactions/{tid}/review", json={"decision": "confirmed_fraud"})
        self.assertEqual(self.count(f"SELECT is_fraud_label FROM transactions WHERE transaction_id = '{tid}'"), 1)
        self.client.post(f"/api/transactions/{tid}/review", json={"decision": "false_positive"})
        self.assertEqual(self.count(f"SELECT is_fraud_label FROM transactions WHERE transaction_id = '{tid}'"), 0)
        with connection.get_connection(self.db) as conn, tempfile.TemporaryDirectory() as d:
            n = connection.export_labels(conn, Path(d) / "labels.csv")
            labels = pd.read_csv(Path(d) / "labels.csv")
        self.assertGreaterEqual(n, 1)
        self.assertIn("account_age_days", labels.columns)

    def test_seed_refuses_to_wipe_reviews(self):  # IMP-17
        tid = self.post(user_id="SEED", amount=10).get_json()["transaction_id"]
        self.client.post(f"/api/transactions/{tid}/review", json={"decision": "approved"})
        with self.assertRaises(connection.ReviewsExist):
            connection.seed_demo_data(self.db)
        self.assertGreaterEqual(self.count("SELECT COUNT(*) FROM reviews"), 1)

    def test_view_uses_latest_assessment_and_review(self):  # IMP-14
        tid = self.post(user_id="VW", amount=10, transaction_id="VW-1").get_json()["transaction_id"]
        self.post(user_id="VW", amount=10, transaction_id="VW-1")      # re-score: 2 assessments
        self.assertEqual(self.count("SELECT COUNT(*) FROM v_transaction_risk WHERE transaction_id = 'VW-1'"), 1)
        with _sql(self.db) as con:
            plan = " ".join(r[-1] for r in con.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM v_transaction_risk WHERE transaction_id = ?", (tid,)))
        self.assertIn("idx_rv_tx_id", plan)


class TestAuthAndContract(unittest.TestCase):
    def test_optional_api_key(self):  # IMP-11 (minimal)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"FRAUD_API_KEY": "s3cret"}):
            client = create_app(Path(d) / "a.db").test_client()
            self.assertEqual(client.post("/api/score", json={"user_id": "K", "amount": 5}).status_code, 401)
            ok = client.post("/api/score", json={"user_id": "K", "amount": 5}, headers={"X-API-Key": "s3cret"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(client.get("/api/stats").status_code, 200)   # reads stay open
            self.assertTrue(client.get("/api/health").get_json()["auth_required"])

    def test_feature_contract_and_bundle_thresholds(self):  # IMP-10, IMP-5
        s = RiskScorer(config.FRAUD_MODEL_PATH, expected_features=FEATURES)
        self.assertLess(s.review_threshold, s.block_threshold)
        self.assertIn("sklearn", s.lib_versions)
        self.assertRaises(FeatureContractError, RiskScorer, config.FRAUD_MODEL_PATH, FEATURES[::-1])
        with tempfile.TemporaryDirectory() as d:
            health = create_app(Path(d) / "h.db").test_client().get("/api/health").get_json()
        self.assertEqual(health["thresholds"], {"review": s.review_threshold, "block": s.block_threshold})


if __name__ == "__main__":
    unittest.main()
