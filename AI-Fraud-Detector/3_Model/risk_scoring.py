"""
Risk scoring: model probability -> 0-100 score -> action + human-readable reasons.

    from risk_scoring import RiskScorer
    scorer = RiskScorer()                      # loads saved_models/fraud_model.pkl
    result = scorer.score(feature_frame)       # list of dicts

Each result looks like:
    {"risk_score": 87, "risk_level": "high", "action": "block",
     "reasons": [{"feature": "users_per_card", "text": "Card used by 6 different accounts", "impact": 21}, ...]}

How reasons are computed (model-driven, no extra dependencies):
  for every signal that is actually abnormal on this order, we reset it to the
  value a typical *legitimate* order has and re-score. The drop in the model's
  log-odds is that signal's impact (log-odds, not 0-100 points, so signals still
  register when several of them push the score to 100). Reasons are the biggest
  impacts, so they reflect what the model reacted to - and the abnormality gate
  stops it from "blaming" ordinary values such as a 2pm order time.
  `impact` is each reason's share (%) of the explained risk.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "fraud_model.pkl"

# Fallback thresholds only. A trained bundle carries its own data-driven thresholds
# (chosen on a time-ordered validation slice, see train_model.choose_thresholds), and
# RiskScorer.review_threshold / block_threshold are what scoring actually uses.
REVIEW_THRESHOLD = 30   # >= review -> send to manual review
BLOCK_THRESHOLD = 70    # >= block  -> block / hold the order
MIN_REASON_LOGODDS = 0.15  # minimum log-odds contribution to be cited
MAX_REASONS = 4

# Features explained together (they describe one fact) -> one reason.
REASON_GROUPS = {
    "amount": ["amount", "log_amount", "amount_to_baseline", "new_account_high_spend"],
    "checkout": ["checkout_seconds", "rushed_checkout"],
    "time": ["hour", "is_night"],
    "logins": ["login_attempts", "failed_logins"],
    "is_vpn": ["is_vpn"],
    "is_proxy": ["is_proxy"],
    "country_mismatch": ["country_mismatch"],
    "high_risk_country": ["high_risk_country"],
    "is_emulator": ["is_emulator"],
    "is_rooted": ["is_rooted"],
    "timezone_mismatch": ["timezone_mismatch"],
    "accounts_per_device": ["accounts_per_device"],
    "users_per_card": ["users_per_card"],
    "account_age_days": ["account_age_days"],
    "pages_viewed": ["pages_viewed"],
}

# A signal may only be cited when it is genuinely abnormal on this order.
ABNORMAL = {
    "amount": lambda r: r["amount_to_baseline"] >= 2 or r["amount"] < 5 or r["new_account_high_spend"] == 1,
    "checkout": lambda r: r["checkout_seconds"] < 30,
    "time": lambda r: r["is_night"] == 1,
    "logins": lambda r: r["failed_logins"] >= 1 or r["login_attempts"] >= 3,
    "accounts_per_device": lambda r: r["accounts_per_device"] >= 2,
    "users_per_card": lambda r: r["users_per_card"] >= 2,
    # an unknown age (-1, account_age_known = 0) is not "brand-new"
    "account_age_days": lambda r: r.get("account_age_known", 1) == 1 and 0 <= r["account_age_days"] < 14,
    "pages_viewed": lambda r: r["pages_viewed"] <= 3,
}


def _reason_text(group: str, r: pd.Series) -> str:
    match group:
        case "amount":
            if r["new_account_high_spend"]:
                return f"New account ({r['account_age_days']:.0f} days old) spending ${r['amount']:,.2f}"
            if r["amount"] < 5:
                return f"Tiny 'card-testing' amount (${r['amount']:,.2f})"
            return f"Amount ${r['amount']:,.2f} is {r['amount_to_baseline']:.1f}x this customer's normal spend"
        case "checkout":
            return f"Rushed checkout ({r['checkout_seconds']:.0f}s from cart to payment)"
        case "time":
            return f"Placed at an unusual hour ({int(r['hour']):02d}:00)"
        case "logins":
            return f"{int(r['login_attempts'])} login attempts, {int(r['failed_logins'])} failed"
        case "is_vpn":
            return "Connection through a VPN"
        case "is_proxy":
            return "Connection through a proxy"
        case "country_mismatch":
            return f"IP location ({r.get('ip_country', '?')}) differs from billing country ({r.get('billing_country', '?')})"
        case "high_risk_country":
            return "High-risk country involved"
        case "is_emulator":
            return "Emulator / virtual device"
        case "is_rooted":
            return "Rooted or jailbroken device"
        case "timezone_mismatch":
            return "Device timezone doesn't match IP location"
        case "accounts_per_device":
            return f"Device used by {int(r['accounts_per_device'])} different accounts"
        case "users_per_card":
            return f"Card used by {int(r['users_per_card'])} different accounts"
        case "account_age_days":
            return f"Brand-new account ({r['account_age_days']:.1f} days old)"
        case "pages_viewed":
            return f"Went straight to checkout ({int(r['pages_viewed'])} pages viewed)"
    return group


def risk_level(score: int, review: int = REVIEW_THRESHOLD, block: int = BLOCK_THRESHOLD) -> tuple[str, str]:
    if score >= block:
        return "high", "block"
    if score >= review:
        return "medium", "review"
    return "low", "approve"


class FeatureContractError(RuntimeError):
    """The saved model was trained on a different feature list than the code builds."""


class RiskScorer:
    def __init__(self, model_path: Path | str = MODEL_PATH, expected_features: list[str] | None = None):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]   # the raw tree model (SHAP / feature_importances_ use it directly)
        self.features: list[str] = bundle["features"]
        self.baseline: dict = bundle["baseline"]  # typical legit value per feature
        self.version: str = bundle.get("version", "unknown")
        self.metrics: dict = bundle.get("metrics", {})
        th = bundle.get("thresholds") or {}
        self.review_threshold: int = int(th.get("review", REVIEW_THRESHOLD))
        self.block_threshold: int = int(th.get("block", BLOCK_THRESHOLD))
        self.lib_versions: dict = bundle.get("lib_versions", {})
        if expected_features is not None and list(expected_features) != list(self.features):
            missing = sorted(set(expected_features) - set(self.features))
            extra = sorted(set(self.features) - set(expected_features))
            raise FeatureContractError(
                f"model {self.version} was trained on different features (missing {missing}, extra {extra}, "
                "or a different order) - retrain with python 3_Model/training/train_model.py")
        self._check_versions()

    def _check_versions(self):
        try:
            import sklearn
        except ImportError:  # pragma: no cover
            return
        trained = self.lib_versions.get("sklearn")
        if trained and trained.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            warnings.warn(f"model {self.version} was trained with scikit-learn {trained}, running {sklearn.__version__}; "
                          "scores may differ - retrain or install the pinned version", RuntimeWarning, stacklevel=3)

    def level(self, score: int) -> tuple[str, str]:
        return risk_level(score, self.review_threshold, self.block_threshold)

    # raw probability -> 0..100
    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        return np.rint(self._proba(X) * 100).astype(int)

    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features].astype(float))[:, 1]

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def score(self, df: pd.DataFrame, explain: bool = True) -> list[dict]:
        X = df[self.features].astype(float).reset_index(drop=True)
        scores = self._scores(X)
        out = []
        for i, s in enumerate(scores):
            level, action = self.level(int(s))
            out.append({"risk_score": int(s), "risk_level": level, "action": action, "reasons": []})

        if explain:
            idx = [i for i, s in enumerate(scores) if s >= self.review_threshold // 2]
            if idx:
                self._explain(df.reset_index(drop=True), X, scores, idx, out)
        return out

    def _explain(self, df, X, scores, idx, out):
        # Build every "what if this signal were normal?" variant, score them in one batch.
        variants, meta = [], []
        for i in idx:
            row = X.iloc[i]
            for group, cols in REASON_GROUPS.items():
                cols = [c for c in cols if c in self.features]
                if not cols:
                    continue
                gate = ABNORMAL.get(group, lambda r, c=cols[0]: r[c] == 1)
                if not gate(row):
                    continue
                v = row.copy()
                for c in cols:
                    v[c] = self.baseline[c]
                variants.append(v)
                meta.append((i, group))
        if not variants:
            return
        base_logit = self._logit(self._proba(X.iloc[sorted({i for i, _ in meta})]))
        base_by_row = dict(zip(sorted({i for i, _ in meta}), base_logit))
        alt_logit = self._logit(self._proba(pd.DataFrame(variants)))
        per_row: dict[int, list] = {}
        for (i, group), a in zip(meta, alt_logit):
            delta = float(base_by_row[i] - a)
            if delta >= MIN_REASON_LOGODDS:
                per_row.setdefault(i, []).append((delta, group))
        for i, items in per_row.items():
            items.sort(reverse=True)
            groups = {g for _, g in items}
            if {"amount", "account_age_days"} <= groups and X.iloc[i]["new_account_high_spend"] == 1:
                # "New account spending $X" already says the account is new - merge, don't repeat
                age = next(d for d, g in items if g == "account_age_days")
                items = [(d + age, g) if g == "amount" else (d, g) for d, g in items if g != "account_age_days"]
                items.sort(reverse=True)
            items = items[:MAX_REASONS]
            total = sum(d for d, _ in items)
            ctx = pd.concat([df.iloc[i], X.iloc[i]])
            ctx = ctx[~ctx.index.duplicated(keep="last")]
            out[i]["reasons"] = [
                {"feature": g, "text": _reason_text(g, ctx), "impact": int(round(100 * d / total))}
                for d, g in items
            ]


if __name__ == "__main__":  # quick smoke test on processed data
    import sys
    data = Path(__file__).resolve().parents[1] / "2_Data" / "processed_data" / "feature_engineered.csv"
    df = pd.read_csv(data).sample(5, random_state=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    for tid, res in zip(df.transaction_id, RiskScorer().score(df)):
        print(tid, res["risk_score"], res["action"], [r["text"] for r in res["reasons"]])
