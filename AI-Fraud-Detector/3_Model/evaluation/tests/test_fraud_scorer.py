"""Unit tests for the fraud-specific scoring functions.

Run from the project root:  python -m unittest discover -s 3_Model/evaluation/tests -t 3_Model
"""
import unittest

from evaluation.fraud_scorer import (level_of, score_action, score_explanation_clarity,
                                     score_false_signals, score_risk_accuracy, score_signal_detection,
                                     score_of, signals_in, signals_in_text)
from evaluation.scorer import passes, score_response

HIGH = {"task": "fraud_analysis", "expected_level": "high", "risk_range": [70, 100],
        "expected_signals": ["vpn", "shared_card"], "absent_signals": ["emulator"]}
TRAP = {"task": "fraud_analysis", "expected_level": "low", "acceptable_levels": ["low", "medium"],
        "risk_range": [0, 50]}
MODEL_CASE = {"task": "risk_scoring", "risk_range": [70, 100], "expected_action": "block",
              "expected_signals": ["shared_card", "rushed_checkout"], "absent_signals": ["vpn"]}


def model_output(score, action, reasons):
    level = "high" if score >= 70 else "medium" if score >= 30 else "low"
    return {"risk_score": score, "risk_level": level, "action": action,
            "reasons": [{"feature": f, "text": t, "impact": 50} for f, t in reasons]}


class TestSignalExtraction(unittest.TestCase):
    def test_detects_core_signals(self):
        text = ("The customer used a VPN, the card is used by 4 different accounts, "
                "and checkout took 8 seconds - a rushed checkout from an emulator.")
        self.assertTrue({"vpn", "shared_card", "rushed_checkout", "emulator"} <= signals_in_text(text))

    def test_negated_signals_are_ignored(self):
        self.assertNotIn("vpn", signals_in_text("No VPN was detected."))
        self.assertNotIn("shared_device", signals_in_text("The device and card are not shared."))

    def test_model_reason_codes_map_to_canonical_signals(self):
        out = model_output(90, "block", [("users_per_card", "Card used by 5 accounts"),
                                         ("checkout", "Rushed checkout (9s)")])
        self.assertEqual(signals_in(out), {"shared_card", "rushed_checkout"})

    def test_level_and_score_parsing(self):
        reply = "Risk level: HIGH\nRisk score: 88\nRecommendation: block"
        self.assertEqual(level_of(reply), "high")
        self.assertEqual(score_of(reply), 88)


class TestRiskAccuracy(unittest.TestCase):
    def test_correct_level_and_range(self):
        self.assertEqual(score_risk_accuracy("Risk level: high\nRisk score: 90", HIGH), 1.0)

    def test_adjacent_level_gets_partial_credit(self):
        self.assertLess(score_risk_accuracy("Risk level: medium\nRisk score: 60", HIGH), 1.0)
        self.assertGreater(score_risk_accuracy("Risk level: medium\nRisk score: 60", HIGH), 0.0)

    def test_false_positive_trap_blocked_fails_the_test(self):
        reply = "Risk level: high\nRisk score: 85\nSignals: large amount.\nRecommendation: Block it, because it is large."
        scores, final = score_response(reply, TRAP)
        self.assertFalse(passes(scores, final, TRAP))

    def test_model_score_outside_range(self):
        self.assertEqual(score_risk_accuracy(model_output(95, "block", []), MODEL_CASE), 1.0)
        self.assertEqual(score_risk_accuracy(model_output(20, "approve", []), MODEL_CASE), 0.0)


class TestActionsAndSignals(unittest.TestCase):
    def test_action_exact_and_adjacent(self):
        self.assertEqual(score_action(model_output(90, "block", []), MODEL_CASE), 1.0)
        self.assertEqual(score_action(model_output(50, "review", []), MODEL_CASE), 0.5)
        self.assertEqual(score_action(model_output(10, "approve", []), MODEL_CASE), 0.0)

    def test_signal_recall_and_false_signals(self):
        out = model_output(90, "block", [("users_per_card", "Card used by 5 accounts"),
                                         ("is_vpn", "Connection through a VPN")])
        self.assertEqual(score_signal_detection(out, MODEL_CASE), 0.5)
        self.assertEqual(score_false_signals(out, MODEL_CASE), 0.0)

    def test_wrong_action_fails_even_with_good_average(self):
        out = model_output(72, "block", [("users_per_card", "Card used by 5 accounts"),
                                         ("checkout", "Rushed checkout (9s)")])
        case = {**MODEL_CASE, "expected_action": "approve", "risk_range": [60, 100]}
        scores, final = score_response(out, case)
        self.assertFalse(passes(scores, final, case))


class TestExplanationClarity(unittest.TestCase):
    def test_flagged_without_reasons_is_unusable(self):
        self.assertEqual(score_explanation_clarity(model_output(80, "block", []), MODEL_CASE), 0.0)

    def test_raw_feature_names_are_capped(self):
        reply = "Risk level: high\nFlagged because users_per_card=6 had the top SHAP value. Recommendation: block."
        self.assertLessEqual(score_explanation_clarity(reply, HIGH), 0.4)

    def test_plain_language_scores_full(self):
        reply = ("Risk level: high\nRisk score: 90\nThe card is used by 5 accounts and the customer is on a VPN.\n"
                 "Recommendation: Block the order.")
        self.assertEqual(score_explanation_clarity(reply, HIGH), 1.0)


if __name__ == "__main__":
    unittest.main()
