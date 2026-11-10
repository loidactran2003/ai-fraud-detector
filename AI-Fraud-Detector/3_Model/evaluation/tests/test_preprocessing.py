"""Unit tests for the shared feature pipeline: context features must not look ahead in time
and must be identical whether built in batch (training) or one order at a time (API).

Run from the project root:  python -m unittest discover -s 3_Model/evaluation/tests -t 3_Model
"""
import sys
import unittest

import pandas as pd

from evaluation import config

_UTILS = str(config.ROOT / "4_Backend" / "utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

from preprocessing import add_context  # noqa: E402

CONTEXT = ["accounts_per_device", "users_per_card", "amount_to_baseline"]


def order_log():
    # user A orders 4x on device D1 / card C1; user B uses D1 and C1 only at the very end
    rows = [("T1", "A", "D1", "C1", 10.0, "2026-01-01 10:00"),
            ("T2", "A", "D1", "C1", 20.0, "2026-01-02 10:00"),
            ("T3", "A", "D1", "C1", 30.0, "2026-01-03 10:00"),
            ("T4", "A", "D1", "C1", 400.0, "2026-01-04 10:00"),
            ("T5", "B", "D1", "C1", 50.0, "2026-01-05 10:00")]
    df = pd.DataFrame(rows, columns=["transaction_id", "user_id", "device_id", "card_hash", "amount", "timestamp"])
    return df.sample(frac=1, random_state=1).reset_index(drop=True)  # file order != time order


class TestPointInTimeContext(unittest.TestCase):
    def setUp(self):
        self.log = order_log()
        self.feats = add_context(self.log).set_index("transaction_id")

    def test_no_look_ahead_on_shared_device_and_card(self):
        # B's later use of the device/card must not leak into A's earlier orders
        self.assertEqual(self.feats.loc[["T1", "T2", "T3", "T4"], "accounts_per_device"].tolist(), [1, 1, 1, 1])
        self.assertEqual(self.feats.loc[["T1", "T2", "T3", "T4"], "users_per_card"].tolist(), [1, 1, 1, 1])
        self.assertEqual(self.feats.loc["T5", "accounts_per_device"], 2)
        self.assertEqual(self.feats.loc["T5", "users_per_card"], 2)

    def test_spend_baseline_uses_only_earlier_orders(self):
        # T4 is compared with the median of T1-T3 (20), not with a median that includes itself
        self.assertAlmostEqual(self.feats.loc["T4", "amount_to_baseline"], 20.0, places=3)
        self.assertAlmostEqual(self.feats.loc["T1", "amount_to_baseline"], 1.0, places=3)

    def test_batch_matches_one_at_a_time(self):
        log = self.log.assign(ts=pd.to_datetime(self.log.timestamp)).sort_values("ts")
        cols = ["user_id", "device_id", "card_hash", "amount"]
        for k in range(len(log)):
            one = add_context(log.iloc[[k]].drop(columns="ts"), log.iloc[:k][cols])
            tid = log.iloc[k]["transaction_id"]
            for c in CONTEXT:
                self.assertAlmostEqual(float(one.iloc[0][c]), float(self.feats.loc[tid, c]), places=3, msg=f"{tid} {c}")


if __name__ == "__main__":
    unittest.main()
