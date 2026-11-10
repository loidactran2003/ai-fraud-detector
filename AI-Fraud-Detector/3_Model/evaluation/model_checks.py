"""Fraud-model QA checks.

Two kinds of tests live here:

1. Golden cases (fraud_tests.json) - individual orders pushed through the exact
   preprocessing + RiskScorer path the API uses, then graded by the rubric.
2. Whole-model checks on the held-out test set - each returns a pass/fail gate:
     model_quality     ROC-AUC / PR-AUC above minimums
     error_rates       false-positive and false-negative rates at the review threshold
     signal_detection  do explanations name signals that are really present (and only those)?
     scoring_logic     score bounds, threshold mapping, monotonicity, determinism
     drift             population stability (PSI) of features and scores, train vs current
     model_comparison  champion model vs logistic-regression and rules-only baselines
     calibration       expected calibration error / Brier of the raw probabilities
     feature_reliance  warns when one feature carries most of the model (a giveaway)

AUC-type gates use the raw probabilities (rounded 0-100 scores create ties); the
threshold gates use the bundle's own review / block thresholds.

Every check returns a result dict in the same schema run_tests.py records.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation import config
from evaluation.fraud_scorer import GROUP_TO_SIGNAL
from evaluation.scorer import passes, score_response

for p in (config.MODEL_ROOT, config.MODEL_ROOT / "training", config.ROOT / "4_Backend" / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preprocessing import FEATURES, prepare_single  # noqa: E402
from risk_scoring import MAX_REASONS, RiskScorer, risk_level  # noqa: E402

BINARY_RISK_FLAGS = ["is_vpn", "is_proxy", "country_mismatch", "high_risk_country",
                     "is_emulator", "is_rooted", "timezone_mismatch", "rushed_checkout", "is_night"]


# ============================================================== helpers
def _result(id_, task, description, scores, passed, details, started, category=None, output=None):
    final = round(float(np.mean(list(scores.values()))), 3) if scores else 0.0
    return {
        "id": id_, "task": task, "suite": "model", "category": category or task,
        "prompt": description, "input": "", "output": output if output is not None else details.get("summary", ""),
        "error": None, "scores": scores, "final_score": final, "passed": bool(passed),
        "latency_s": round(time.perf_counter() - started, 3), "details": details,
    }


def present_signals(row: pd.Series) -> set[str]:
    """Ground-truth signals present in a feature row (canonical names)."""
    flags = {
        "vpn": row.is_vpn == 1, "proxy": row.is_proxy == 1,
        "country_mismatch": row.country_mismatch == 1, "high_risk_country": row.high_risk_country == 1,
        "emulator": row.is_emulator == 1, "rooted_device": row.is_rooted == 1,
        "timezone_mismatch": row.timezone_mismatch == 1,
        "shared_device": row.accounts_per_device >= 2, "shared_card": row.users_per_card >= 2,
        "new_account": 0 <= row.account_age_days < 14 and row.get("account_age_known", 1) == 1,
        "rushed_checkout": row.checkout_seconds < 30,
        "login_failures": row.failed_logins >= 1 or row.login_attempts >= 3,
        "odd_hour": row.is_night == 1, "few_pages": row.pages_viewed <= 3,
        "abnormal_amount": row.amount_to_baseline >= 2.5 or row.amount < 5 or row.new_account_high_spend == 1,
    }
    return {k for k, v in flags.items() if v}


def rules_score(df: pd.DataFrame) -> np.ndarray:
    """Hand-written red-flag checklist - what many small shops use today."""
    s = (20 * df.is_vpn + 15 * df.is_proxy + 15 * df.country_mismatch + 15 * df.high_risk_country
         + 20 * df.is_emulator + 10 * df.is_rooted + 10 * df.timezone_mismatch
         + 15 * (df.accounts_per_device >= 3) + 20 * (df.users_per_card >= 2)
         + 15 * df.account_age_days.between(0, 7, inclusive="left") + 10 * df.rushed_checkout + 10 * (df.failed_logins >= 2)
         + 5 * df.is_night + 15 * (df.amount_to_baseline >= 3))
    return np.clip(s.values, 0, 100)


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    uniq = np.unique(ref)
    if len(uniq) <= 12:  # categorical / binary
        edges = np.r_[uniq - 1e-9, uniq[-1] + 1e-9] if len(uniq) > 1 else np.array([uniq[0] - 1, uniq[0] + 1])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


class FraudModelHarness:
    """Loads the model, history and evaluation data once for every check."""

    def __init__(self, current_path=None):
        self.scorer = RiskScorer(config.FRAUD_MODEL_PATH, expected_features=FEATURES)
        self.review, self.block = self.scorer.review_threshold, self.scorer.block_threshold
        feats = pd.read_csv(config.FEATURES_PATH, parse_dates=["timestamp"]).sort_values(
            ["timestamp", "transaction_id"], kind="stable")  # same order as train_model.load_dataset
        self.history = feats[["user_id", "device_id", "card_hash", "amount", "timestamp"]]
        self.test = pd.read_csv(config.TEST_SET_PATH, parse_dates=["timestamp"])
        cut = int(len(feats) * 0.8)
        self.train = feats.iloc[:cut]
        self.current = (pd.read_csv(current_path, parse_dates=["timestamp"]) if current_path else self.test)
        self._test_scored = None
        self._test_proba = None

    @property
    def test_proba(self):
        if self._test_proba is None:
            self._test_proba = self.scorer._proba(self.test)
        return self._test_proba

    @property
    def test_scored(self):
        if self._test_scored is None:
            res = self.scorer.score(self.test)
            self._test_scored = self.test.assign(
                risk_score=[r["risk_score"] for r in res], action=[r["action"] for r in res],
                reasons=[r["reasons"] for r in res])
        return self._test_scored

    # ---------------------------------------------------------- golden cases
    def run_case(self, test: dict, defaults: dict) -> dict:
        started = time.perf_counter()
        payload = {**defaults, **test["transaction"]}
        error, output = None, {}
        try:
            history = self.history
            if test.get("history"):  # earlier orders built into real context features (no overrides)
                extra = pd.DataFrame([{"amount": 50.0, **h} for h in test["history"]])
                extra["timestamp"] = pd.to_datetime(extra["timestamp"])
                history = pd.concat([history, extra[history.columns]], ignore_index=True)
            row = prepare_single(payload, history)
            for k, v in test.get("context", {}).items():
                row[k] = v
            output = self.scorer.score(row)[0]
        except Exception as exc:  # a broken case must not abort the run
            error = f"{type(exc).__name__}: {exc}"
        scores, final = score_response(output, test) if not error else ({}, 0.0)
        return {
            "id": test["id"], "task": test["task"], "suite": "fraud", "category": test.get("category"),
            "prompt": test.get("description", ""), "input": json.dumps(payload),
            "output": output, "error": error, "scores": scores, "final_score": final,
            "passed": bool(passes(scores, final, test) and not error),
            "latency_s": round(time.perf_counter() - started, 3),
            "details": {"expected_range": test.get("risk_range"),
                        "expected_action": test.get("expected_action") or test.get("acceptable_actions"),
                        "signals_found": sorted({GROUP_TO_SIGNAL.get(r["feature"], r["feature"])
                                                 for r in output.get("reasons", [])})},
        }

    # ---------------------------------------------------------- whole-model checks
    def check_model_quality(self):
        t = time.perf_counter()
        y, s = self.test.is_fraud.values, self.test_proba   # raw probabilities: no rounding ties
        roc, pr = roc_auc_score(y, s), average_precision_score(y, s)
        scores = {"roc_auc": round(min(1.0, roc / config.MIN_ROC_AUC), 3),
                  "pr_auc": round(min(1.0, pr / config.MIN_PR_AUC), 3)}
        passed = roc >= config.MIN_ROC_AUC and pr >= config.MIN_PR_AUC
        return _result("mq-auc", "model_quality", f"ROC-AUC >= {config.MIN_ROC_AUC} and PR-AUC >= {config.MIN_PR_AUC} on held-out (most recent) orders",
                       scores, passed, {"roc_auc": round(roc, 4), "pr_auc": round(pr, 4), "n": int(len(y)),
                                        "summary": f"ROC-AUC {roc:.3f} | PR-AUC {pr:.3f}"}, t)

    def check_error_rates(self):
        t = time.perf_counter()
        d = self.test_scored
        flagged = d.risk_score >= self.review
        legit, fraud = d.is_fraud == 0, d.is_fraud == 1
        fpr = float((flagged & legit).sum() / max(legit.sum(), 1))
        fnr = float((~flagged & fraud).sum() / max(fraud.sum(), 1))
        missed_dollars = float(d.loc[~flagged & fraud, "amount"].sum())
        fp_drivers = pd.Series([GROUP_TO_SIGNAL.get(r["feature"], r["feature"])
                                for rs in d.loc[flagged & legit, "reasons"] for r in rs[:1]]).value_counts()
        scores = {"false_positive_rate": round(max(0.0, 1 - fpr / (2 * config.MAX_FALSE_POSITIVE_RATE)), 3),
                  "false_negative_rate": round(max(0.0, 1 - fnr / (2 * config.MAX_FALSE_NEGATIVE_RATE)), 3)}
        passed = fpr <= config.MAX_FALSE_POSITIVE_RATE and fnr <= config.MAX_FALSE_NEGATIVE_RATE
        examples_fn = d.loc[~flagged & fraud].nlargest(5, "amount")[["transaction_id", "amount", "risk_score"]]
        return _result("mq-errors", "error_rates",
                       f"At score >= {self.review} (review threshold): false-positive rate <= {config.MAX_FALSE_POSITIVE_RATE:.0%}, "
                       f"false-negative rate <= {config.MAX_FALSE_NEGATIVE_RATE:.0%}",
                       scores, passed, {
                           "false_positive_rate": round(fpr, 4), "false_negative_rate": round(fnr, 4),
                           "false_positives": int((flagged & legit).sum()), "false_negatives": int((~flagged & fraud).sum()),
                           "true_positives": int((flagged & fraud).sum()), "true_negatives": int((~flagged & legit).sum()),
                           "missed_fraud_dollars": round(missed_dollars, 2),
                           "top_false_positive_drivers": fp_drivers.head(5).to_dict(),
                           "largest_missed_fraud": examples_fn.to_dict("records"),
                           "summary": f"FPR {fpr:.2%} | FNR {fnr:.2%} | ${missed_dollars:,.0f} fraud missed"}, t)

    def check_signal_detection(self):
        t = time.perf_counter()
        d = self.test_scored
        tp = d[(d.is_fraud == 1) & (d.risk_score >= self.review)]
        prec, cov, per_signal = [], [], {}
        for _, row in tp.iterrows():
            present = present_signals(row)
            given = {GROUP_TO_SIGNAL.get(r["feature"], r["feature"]) for r in row.reasons}
            if given:
                prec.append(len(given & present) / len(given))
            if present:
                cov.append(len(given & present) / min(len(present), MAX_REASONS))
            for sig in present:
                hit = per_signal.setdefault(sig, [0, 0])
                hit[0] += sig in given
                hit[1] += 1
        precision = float(np.mean(prec)) if prec else 0.0
        coverage = float(np.mean(cov)) if cov else 0.0
        scores = {"reason_precision": round(precision, 3), "signal_coverage": round(coverage, 3)}
        passed = precision >= 0.8 and coverage >= config.MIN_SIGNAL_ATTRIBUTION
        return _result("mq-signals", "signal_detection",
                       "Explanations for caught fraud name signals that are truly present (precision) "
                       f"and cover the main ones (coverage >= {config.MIN_SIGNAL_ATTRIBUTION:.0%})",
                       scores, passed, {
                           "reason_precision": round(precision, 4), "signal_coverage": round(coverage, 4),
                           "n_explained": int(len(tp)),
                           "per_signal_mention_rate": {k: round(v[0] / v[1], 3) for k, v in sorted(per_signal.items())},
                           "summary": f"precision {precision:.0%} | coverage {coverage:.0%} over {len(tp)} caught fraud orders"}, t)

    def check_scoring_logic(self):
        t = time.perf_counter()
        d = self.test_scored
        bounds_ok = bool(d.risk_score.between(0, 100).all())
        mapping = {s: self.scorer.level(s)[1] for s in (0, self.review - 1, self.review,
                                                        self.block - 1, self.block, 100)}
        mapping_ok = (list(mapping.values()) == ["approve", "approve", "review", "review", "block", "block"]
                      and 0 < self.review < self.block <= 100
                      and risk_level(self.review, self.review, self.block)[1] == "review")
        action_ok = bool((d.action == d.risk_score.map(lambda s: self.scorer.level(int(s))[1])).all())

        base = self.test[self.test.is_fraud == 0].sample(min(300, (self.test.is_fraud == 0).sum()), random_state=0)
        s0 = self.scorer._scores(base)
        monotonic = {}
        for flag in BINARY_RISK_FLAGS:
            off = base[base[flag] == 0]
            if off.empty:
                continue
            idx = base.index.get_indexer(off.index)
            s1 = self.scorer._scores(off.assign(**{flag: 1}))
            monotonic[flag] = round(float(np.mean(s1 >= s0[idx] - 1)), 3)
        mono_score = float(np.mean(list(monotonic.values())))
        again = self.scorer._scores(base)
        deterministic = bool(np.array_equal(s0, again))
        scores = {"score_bounds": float(bounds_ok), "threshold_mapping": float(mapping_ok and action_ok),
                  "monotonicity": round(mono_score, 3), "determinism": float(deterministic)}
        passed = bounds_ok and mapping_ok and action_ok and deterministic and mono_score >= 0.95
        return _result("mq-logic", "scoring_logic",
                       "Scores stay in 0-100, thresholds map to the right action, switching ON a risk signal "
                       "never lowers risk, and identical input gives identical output",
                       scores, passed, {"monotonic_share_by_flag": monotonic, "threshold_mapping": mapping,
                                        "summary": f"monotonic {mono_score:.0%} | bounds {'ok' if bounds_ok else 'FAIL'} | "
                                                   f"deterministic {'yes' if deterministic else 'NO'}"}, t)

    def check_drift(self):
        t = time.perf_counter()
        per_feature = {f: round(psi(self.train[f], self.current[f]), 4) for f in FEATURES if f in self.current}
        ref_scores = self.scorer._scores(self.train.sample(min(4000, len(self.train)), random_state=0))
        cur_scores = self.scorer._scores(self.current)
        score_psi = round(psi(ref_scores, cur_scores), 4)
        worst = max(per_feature.items(), key=lambda kv: kv[1])
        drifted = {k: v for k, v in per_feature.items() if v >= config.PSI_WARN}
        scores = {"feature_stability": round(max(0.0, 1 - worst[1] / config.PSI_FAIL), 3),
                  "score_stability": round(max(0.0, 1 - score_psi / config.PSI_FAIL), 3)}
        passed = worst[1] < config.PSI_FAIL and score_psi < config.PSI_FAIL
        return _result("mq-drift", "drift",
                       f"Population stability (PSI) of every feature and of the risk score, training data vs current data "
                       f"(warn >= {config.PSI_WARN}, fail >= {config.PSI_FAIL})",
                       scores, passed, {"score_psi": score_psi, "feature_psi": per_feature,
                                        "drifted_features": drifted, "worst_feature": worst[0],
                                        "fraud_rate_train": round(float(self.train.is_fraud.mean()), 4),
                                        "fraud_rate_current": round(float(self.current.is_fraud.mean()), 4)
                                        if "is_fraud" in self.current else None,
                                        "summary": f"score PSI {score_psi:.3f} | worst feature {worst[0]} PSI {worst[1]:.3f}"}, t)

    def check_model_comparison(self):
        t = time.perf_counter()
        from train_model import make_baseline  # 3_Model/training
        y = self.test.is_fraud.values
        logreg = make_baseline().fit(self.train[FEATURES], self.train.is_fraud)
        # (raw score for the AUCs, 0-100 score at the review threshold for precision/recall/FPR)
        lr_p = logreg.predict_proba(self.test[FEATURES])[:, 1]
        rules = rules_score(self.test)
        candidates = {
            f"champion ({self.scorer.version})": (self.test_proba, self.test_scored.risk_score.values),
            "logistic regression": (lr_p, np.rint(lr_p * 100)),
            "rules checklist": (rules, rules),
        }
        models = []
        for name, (raw, s) in candidates.items():
            flagged = s >= self.review
            tp = int((flagged & (y == 1)).sum())
            models.append({
                "model": name,
                "roc_auc": round(float(roc_auc_score(y, raw)), 4),
                "pr_auc": round(float(average_precision_score(y, raw)), 4),
                "precision": round(tp / max(int(flagged.sum()), 1), 4),
                "recall": round(tp / max(int((y == 1).sum()), 1), 4),
                "false_positive_rate": round(float((flagged & (y == 0)).sum() / max((y == 0).sum(), 1)), 4),
            })
        champ, *others = models
        best_other = max(o["pr_auc"] for o in others)
        scores = {"beats_baselines": 1.0 if champ["pr_auc"] >= best_other else round(champ["pr_auc"] / best_other, 3)}
        return _result("mq-compare", "model_comparison",
                       "Champion model must beat a logistic-regression baseline and a hand-written rules checklist on PR-AUC",
                       scores, champ["pr_auc"] >= best_other,
                       {"models": models, "summary": " | ".join(f"{m['model']}: PR-AUC {m['pr_auc']:.3f}" for m in models)}, t)

    def check_calibration(self):
        t = time.perf_counter()
        y, p = self.test.is_fraud.values.astype(float), self.test_proba
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        table, ece = [], 0.0
        for b in range(10):
            m = idx == b
            if m.any():
                ece += abs(p[m].mean() - y[m].mean()) * m.mean()
                table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                              "mean_predicted": round(float(p[m].mean()), 3), "observed_rate": round(float(y[m].mean()), 3)})
        brier = float(np.mean((p - y) ** 2))
        passed = ece <= config.MAX_ECE
        return _result("mq-calibration", "calibration",
                       f"Expected calibration error of the risk score on held-out orders <= {config.MAX_ECE}",
                       {"ece": round(max(0.0, 1 - ece / (2 * config.MAX_ECE)), 3)}, passed,
                       {"ece": round(float(ece), 4), "brier": round(brier, 5), "reliability": table,
                        "summary": f"ECE {ece:.4f} | Brier {brier:.4f}"}, t)

    def check_feature_reliance(self):
        t = time.perf_counter()
        imp = getattr(self.scorer.model, "feature_importances_", None)
        if imp is None:
            return _result("mq-reliance", "feature_reliance", "No single feature dominates the model", {}, True,
                           {"summary": "model has no feature_importances_ - skipped"}, t)
        ranked = sorted(zip(self.scorer.features, map(float, imp)), key=lambda kv: -kv[1])
        top, share = ranked[0]
        passed = share <= config.MAX_FEATURE_IMPORTANCE
        return _result("mq-reliance", "feature_reliance",
                       f"No single feature carries more than {config.MAX_FEATURE_IMPORTANCE:.0%} of the model's importance "
                       "(a giveaway feature means the data is too easy or leaks the label)",
                       {"top_feature_share": round(min(1.0, config.MAX_FEATURE_IMPORTANCE / max(share, 1e-9)), 3)}, passed,
                       {"top_features": {k: round(v, 4) for k, v in ranked[:5]},
                        "summary": f"top feature {top} {share:.0%}"}, t)

    def run_all_checks(self):
        return [self.check_model_quality(), self.check_error_rates(), self.check_signal_detection(),
                self.check_scoring_logic(), self.check_drift(), self.check_model_comparison(),
                self.check_calibration(), self.check_feature_reliance()]

    def model_summary(self):
        q = self.scorer.metrics
        return {"version": self.scorer.version, "roc_auc": q.get("roc_auc"), "pr_auc": q.get("pr_auc"),
                "review": q.get("review"), "block": q.get("block"),
                "thresholds": {"review": self.review, "block": self.block}}
