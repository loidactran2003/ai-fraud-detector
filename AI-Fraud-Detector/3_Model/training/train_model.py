"""
Train + evaluate the fraud model. The notebooks (model_training.ipynb,
model_evaluation.ipynb) walk through the same steps interactively; this script
is the reproducible, one-command version.

    python 3_Model/training/train_model.py

Time-ordered splits (train on the past, test on the future):
    [ ---------------- train (80%) ---------------- | test (20%) ]
    [ fit (80% of train) | validation (20% of train) ]
  1. a model fitted on `fit` scores `validation`; the review / block thresholds
     are chosen there from business targets (choose_thresholds), never on test
  2. the final model is refitted on the whole train period and evaluated once on test
  3. calibration (Brier, ECE, reliability table) is measured on validation and test

Outputs:
  saved_models/fraud_model.pkl   model bundle used by risk_scoring.py / the API
  saved_models/metrics.json      evaluation numbers (also embedded in the bundle)
  saved_models/test_set.csv      held-out rows, reused by evaluation/explainability
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4_Backend" / "utils"))
sys.path.insert(0, str(ROOT / "3_Model"))
from preprocessing import FEATURES, PROCESSED_DIR  # noqa: E402
from risk_scoring import BLOCK_THRESHOLD, REVIEW_THRESHOLD  # noqa: E402  (fallbacks only)

MODEL_DIR = ROOT / "3_Model" / "saved_models"
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2  # last part of the train period, used to pick thresholds

# Business targets for the thresholds (on the validation slice):
REVIEW_MAX_FPR = 0.025       # at most 2.5% of genuine orders go to manual review (reviewer capacity)
BLOCK_MAX_FPR = 0.005        # auto-block at most 0.5% of genuine orders (1 in 200). A precision target
                             # was tried first: with ~5% undisputed fraud in the labels precision plateaus
                             # near 0.9, so the chosen threshold jumped around (38 at 0.90, 96 at 0.95).
MIN_GAP = 10                 # block threshold is at least this many points above review


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "feature_engineered.csv", parse_dates=["timestamp"])
    # same (timestamp, transaction_id) order the point-in-time features use, so ties split deterministically
    return df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Train on the past, test on the future - how the model is actually used."""
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def make_model() -> GradientBoostingClassifier:
    # min_samples_leaf regularises against the noisy (chargeback) labels; depth / leaf size were
    # picked on validation-slice PR-AUC (see .claude/reports/05_repairs.md), never on test
    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, min_samples_leaf=50,
                                      subsample=0.8, random_state=42)


def make_baseline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def to_scores(proba) -> np.ndarray:
    return np.rint(np.asarray(proba) * 100)


def choose_thresholds(y, scores) -> dict:
    """Pick review / block from business targets on held-out (validation) scores.

    review = lowest score whose false-positive rate is <= REVIEW_MAX_FPR (catch as much
             fraud as the review budget allows)
    block  = lowest score >= review + MIN_GAP whose false-positive rate is <= BLOCK_MAX_FPR
    Falls back to the fixed 30 / 70 when a target can't be met on this data.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    legit, fraud = y == 0, y == 1
    review = next((t for t in range(1, 100) if (scores[legit] >= t).mean() <= REVIEW_MAX_FPR), REVIEW_THRESHOLD)
    block = BLOCK_THRESHOLD
    for t in range(review + MIN_GAP, 100):
        if (scores[legit] >= t).mean() <= BLOCK_MAX_FPR:
            block = t
            break
    block = max(block, review + MIN_GAP)
    at = lambda t: {"fpr": round(float((scores[legit] >= t).mean()), 4),  # noqa: E731
                    "recall": round(float((scores[fraud] >= t).mean()), 4) if fraud.any() else None,
                    "precision": round(float(y[scores >= t].mean()), 4) if (scores >= t).any() else None}
    return {"review": int(review), "block": int(block),
            "policy": {"review_max_fpr": REVIEW_MAX_FPR, "block_max_fpr": BLOCK_MAX_FPR,
                       "min_gap": MIN_GAP, "chosen_on": "validation slice (last 20% of the train period)"},
            "validation": {"review": at(review), "block": at(block)}}


def calibration_report(y, proba, bins: int = 10) -> dict:
    """Brier score, expected calibration error and a reliability table (equal-width bins)."""
    y, proba = np.asarray(y, float), np.asarray(proba, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        mean_p, obs = float(proba[m].mean()), float(y[m].mean())
        ece += abs(mean_p - obs) * m.mean()
        table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                      "mean_predicted": round(mean_p, 4), "observed_rate": round(obs, 4)})
    return {"brier": round(float(brier_score_loss(y, proba)), 5), "ece": round(float(ece), 5), "reliability": table}


def metrics_at(y, scores, threshold) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "flag_rate": round(float(pred.mean()), 4),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(model, test: pd.DataFrame, review: int = REVIEW_THRESHOLD, block: int = BLOCK_THRESHOLD) -> dict:
    y = test["is_fraud"].values
    proba = model.predict_proba(test[FEATURES])[:, 1]
    scores = to_scores(proba)
    fraud_amt = test.loc[test.is_fraud == 1, "amount"].sum()
    caught_amt = test.loc[(test.is_fraud == 1) & (scores >= review), "amount"].sum()
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "review": metrics_at(y, scores, review),
        "block": metrics_at(y, scores, block),
        "fraud_dollars_in_test": round(float(fraud_amt), 2),
        "fraud_dollars_flagged": round(float(caught_amt), 2),
        "n_test": int(len(test)),
        "test_fraud_rate": round(float(y.mean()), 4),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(save: bool = True) -> dict:
    df = load_dataset()
    train_df, test_df = time_split(df)
    fit_df, val_df = time_split(train_df, VALIDATION_FRACTION)

    # 1. thresholds from a model that has never seen the validation slice
    probe = make_model().fit(fit_df[FEATURES], fit_df["is_fraud"])
    val_proba = probe.predict_proba(val_df[FEATURES])[:, 1]
    thresholds = choose_thresholds(val_df["is_fraud"], to_scores(val_proba))

    # 2. final model on the whole train period
    X, y = train_df[FEATURES], train_df["is_fraud"]
    model = make_model().fit(X, y)
    baseline_model = make_baseline().fit(X, y)

    metrics = evaluate(model, test_df, thresholds["review"], thresholds["block"])
    metrics["baseline_logreg"] = {k: v for k, v in evaluate(baseline_model, test_df).items()
                                  if k in ("roc_auc", "pr_auc")}
    metrics["n_train"] = int(len(train_df))
    metrics["thresholds"] = thresholds
    metrics["calibration"] = {
        "test": calibration_report(test_df["is_fraud"], model.predict_proba(test_df[FEATURES])[:, 1]),
        "validation": calibration_report(val_df["is_fraud"], val_proba),
        "note": "raw GBM probabilities; score = round(100 * p). No calibrator is applied: Platt and isotonic "
                "on the validation slice did not lower test Brier/ECE at this data size (see 05_repairs.md).",
    }
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    metrics["feature_importance"] = {f: round(float(v), 4) for f, v in importances}

    data_path = PROCESSED_DIR / "feature_engineered.csv"
    data_sha = _sha256(data_path)
    params = json.dumps(model.get_params(), sort_keys=True, default=str)
    content_id = hashlib.sha256((data_sha + params + ",".join(FEATURES)).encode()).hexdigest()[:8]
    now = datetime.now(timezone.utc)
    bundle = {
        "model": model,
        "features": FEATURES,
        # "normal" reference values = medians of legitimate training orders
        "baseline": train_df.loc[train_df.is_fraud == 0, FEATURES].median().astype(float).to_dict(),
        "thresholds": {"review": thresholds["review"], "block": thresholds["block"]},
        "metrics": metrics,
        # timestamp for humans + content hash so two trainings in the same minute can't collide
        "version": f"gbm-{now:%Y%m%d-%H%M}-{content_id}",
        "trained_at": now.isoformat(),
        "lib_versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
        "data_sha256": data_sha,
        "features_sha256": hashlib.sha256(",".join(FEATURES).encode()).hexdigest(),
    }
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_DIR / "fraud_model.pkl")
        (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        test_df.to_csv(MODEL_DIR / "test_set.csv", index=False)
    return bundle


if __name__ == "__main__":
    b = train()
    m = b["metrics"]
    print(f"model {b['version']}  trained on {m['n_train']} rows, tested on {m['n_test']}")
    print(f"ROC-AUC {m['roc_auc']}  PR-AUC {m['pr_auc']}  (logistic baseline PR-AUC {m['baseline_logreg']['pr_auc']})")
    print(f"thresholds (chosen on validation): review >= {b['thresholds']['review']}, block >= {b['thresholds']['block']}")
    for k in ("review", "block"):
        r = m[k]
        print(f"  score>={r['threshold']:>2}: precision {r['precision']:.2%}  recall {r['recall']:.2%}  "
              f"FPR {r['false_positive_rate']:.2%}  flags {r['flag_rate']:.2%} of orders")
    c = m["calibration"]["test"]
    print(f"  calibration (test): Brier {c['brier']:.4f}  ECE {c['ece']:.4f}")
    print(f"  fraud $ flagged: {m['fraud_dollars_flagged']:,.0f} of {m['fraud_dollars_in_test']:,.0f}")
