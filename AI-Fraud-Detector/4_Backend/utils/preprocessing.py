"""
Cleaning + feature engineering shared by training (3_Model) and the live API (4_Backend).

Keeping one implementation guarantees the model sees identically-built
features at training time and at scoring time (no train/serve skew).

Run directly to rebuild 2_Data/processed_data/*:
    python 4_Backend/utils/preprocessing.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "2_Data" / "raw_data"
PROCESSED_DIR = ROOT / "2_Data" / "processed_data"

HIGH_RISK_COUNTRIES = {"NG", "RU", "UA", "VN", "ID", "PK", "RO"}
RUSHED_CHECKOUT_SECONDS = 20
NEW_ACCOUNT_DAYS = 7
HIGH_SPEND_AMOUNT = 300.0
MIN_HISTORY_FOR_USER_BASELINE = 3
UNKNOWN_ACCOUNT_AGE = -1.0  # account_age_days when neither account_created nor earlier orders are known

# Ordered list of model inputs. Grouped by the four signal families in the brief.
FEATURES = [
    # Network activity
    "is_vpn", "is_proxy", "country_mismatch", "high_risk_country",
    # Device behaviour
    "is_emulator", "is_rooted", "timezone_mismatch", "accounts_per_device",
    # User behaviour
    "login_attempts", "failed_logins", "checkout_seconds", "rushed_checkout",
    "pages_viewed", "account_age_days", "account_age_known", "new_account_high_spend",
    # Transaction patterns
    "amount", "log_amount", "amount_to_baseline", "users_per_card", "hour", "is_night",
]

FEATURE_LABELS = {
    "is_vpn": "VPN detected",
    "is_proxy": "Proxy detected",
    "country_mismatch": "IP country differs from billing country",
    "high_risk_country": "High-risk country involved",
    "is_emulator": "Emulator / virtual device",
    "is_rooted": "Rooted / jailbroken device",
    "timezone_mismatch": "Device timezone doesn't match IP location",
    "accounts_per_device": "Device shared across many accounts",
    "login_attempts": "Many login attempts",
    "failed_logins": "Failed logins before purchase",
    "checkout_seconds": "Checkout speed",
    "rushed_checkout": "Rushed checkout",
    "pages_viewed": "Very few pages viewed",
    "account_age_days": "Account age",
    "account_age_known": "Account age known",
    "new_account_high_spend": "New account spending a large amount",
    "amount": "Order amount",
    "log_amount": "Order amount",
    "amount_to_baseline": "Amount far above normal spend",
    "users_per_card": "Card used by multiple accounts",
    "hour": "Time of day",
    "is_night": "Placed late at night (11pm-5am)",
}

DEVICE_COLS = ["is_emulator", "is_rooted", "timezone_mismatch"]
BOOL_COLS = ["is_vpn", "is_proxy", *DEVICE_COLS]
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off", ""}


def parse_bool(value) -> int:
    """Strict 0/1 for a boolean signal. bool("false") is True, so never use bool() on payloads.

    Accepts True/False, 0/1 (int or float), the strings 0/1/true/false/yes/no/y/n/on/off
    (any case) and None/"" (missing -> 0). Anything else raises ValueError.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return int(value)
        raise ValueError(f"not a boolean: {value!r}")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return 1
        if v in _FALSE:
            return 0
    raise ValueError(f"not a boolean: {value!r}")


def to_naive_datetime(values, errors: str = "raise"):
    """pd.to_datetime that drops any timezone but keeps the wall-clock time the store sent
    (training data is naive local time, and naive - aware arithmetic would fail)."""
    out = pd.to_datetime(values, errors=errors)
    if isinstance(out, pd.Series):
        return out.dt.tz_localize(None) if out.dt.tz is not None else out
    if getattr(out, "tz", None) is not None:
        return out.tz_localize(None)
    return out


# --------------------------------------------------------------------------- loading
def load_raw(raw_dir: Path = RAW_DIR):
    tx = pd.read_csv(raw_dir / "transactions.csv")
    with open(raw_dir / "user_logs.json", encoding="utf-8") as f:
        logs = pd.DataFrame(json.load(f))
    devices = pd.read_csv(raw_dir / "device_signals.csv")
    return tx, logs, devices


# --------------------------------------------------------------------------- cleaning
def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.drop_duplicates(subset="transaction_id").copy()
    tx = tx[tx["user_id"].notna()]  # every order needs an account (user_id is NOT NULL in the DB)
    tx = tx[tx["amount"].notna() & (tx["amount"] > 0)]
    for col in ("billing_country", "ip_country"):
        tx[col] = tx[col].astype(str).str.strip().str.upper()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    tx = tx[tx["timestamp"].notna()]
    for col in ("is_vpn", "is_proxy"):
        tx[col] = tx[col].fillna(0).astype(int)
    return tx.reset_index(drop=True)


def merge_sources(tx: pd.DataFrame, logs: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    logs = logs.drop(columns=["user_id"], errors="ignore")
    df = tx.merge(logs, on="session_id", how="left")
    df = df.merge(devices[["device_id", *DEVICE_COLS]], on="device_id", how="left")
    return df


# --------------------------------------------------------------------------- features
def add_context(df: pd.DataFrame, history: pd.DataFrame | None = None,
                global_median: float | None = None) -> pd.DataFrame:
    """Cross-transaction signals: device/card re-use and each user's normal spend.

    Every value is *point-in-time*: it only uses orders placed before the one
    being scored, exactly what the live API can know.
      accounts_per_device / users_per_card  distinct accounts seen on the device /
                                            card so far, this order included
      amount_to_baseline                    amount / median of the user's earlier
                                            orders (store-wide median of earlier
                                            orders until the user has
                                            MIN_HISTORY_FOR_USER_BASELINE of them)
      first_seen                            timestamp of the user's first earlier
                                            order (NaT if none) - the fallback for a
                                            missing account_created

    history=None  (training / batch) - `df` is the full order log; each row only
                  sees the rows before it in time.
    history=frame (API)              - `history` holds the orders already known
                  (the incoming ones must NOT be in it); every row of `df` is
                  scored against that history plus itself. If `history` has a
                  `timestamp` column it also gives first_seen. `global_median` overrides
                  the store-wide median (the API computes it in SQL instead of loading
                  every stored order).
    """
    if history is None:
        return _add_context_point_in_time(df)
    df = df.copy()
    h = history
    both = pd.concat([h[["device_id", "card_hash", "user_id"]], df[["device_id", "card_hash", "user_id"]]],
                     ignore_index=True)
    # dropna=False: an unknown account still counts as one, as in the batch path below
    acc_per_dev = both.groupby("device_id")["user_id"].nunique(dropna=False)
    users_per_card = both.groupby("card_hash")["user_id"].nunique(dropna=False)
    user_stats = h.groupby("user_id")["amount"].agg(["median", "count"])
    if global_median is None:
        global_median = float(h["amount"].median()) if len(h) else np.nan
    global_median = float(global_median)

    df["accounts_per_device"] = df["device_id"].map(acc_per_dev).fillna(1).astype(int)
    df["users_per_card"] = df["card_hash"].map(users_per_card).fillna(1).astype(int)
    med = df["user_id"].map(user_stats["median"])
    cnt = df["user_id"].map(user_stats["count"]).fillna(0)
    fallback = df["amount"] if np.isnan(global_median) else global_median
    baseline = np.where(cnt >= MIN_HISTORY_FOR_USER_BASELINE, med, fallback)
    df["amount_to_baseline"] = np.round(df["amount"] / np.maximum(baseline, 1.0), 3)
    if "timestamp" in h and len(h):
        first = to_naive_datetime(h["timestamp"], errors="coerce").groupby(h["user_id"]).min()
        df["first_seen"] = pd.to_datetime(df["user_id"].map(first.to_dict()))  # dict: safe when empty
    else:
        df["first_seen"] = pd.NaT
    return df


def _add_context_point_in_time(df: pd.DataFrame) -> pd.DataFrame:
    """add_context for a full order log, without look-ahead (see add_context)."""
    df = df.copy()
    order = df.assign(_ts=pd.to_datetime(df["timestamp"]), _pos=np.arange(len(df))) \
              .sort_values(["_ts", "transaction_id" if "transaction_id" in df else "_pos"], kind="stable")
    for key, col in (("device_id", "accounts_per_device"), ("card_hash", "users_per_card")):
        first_pair = ~order.duplicated([key, "user_id"])          # first time this account uses this key
        seen = first_pair.astype(int).groupby(order[key], dropna=False).cumsum()
        order[col] = np.where(order[key].isna(), 1, seen).astype(int)

    # orders without a user_id have no own history: they fall back to the store-wide baseline
    by_user = order[order["user_id"].notna()].groupby("user_id")["amount"]
    prev_med = by_user.transform(lambda s: s.expanding().median().shift(1)).reindex(order.index)
    prev_cnt = by_user.cumcount().reindex(order.index).fillna(0)
    global_prev = order["amount"].expanding().median().shift(1).fillna(order["amount"])
    baseline = np.where(prev_cnt >= MIN_HISTORY_FOR_USER_BASELINE, prev_med, global_prev)
    order["amount_to_baseline"] = np.round(order["amount"] / np.maximum(baseline, 1.0), 3)
    # first earlier order of the same user (NaT on a user's first order)
    first_ts = order[order["user_id"].notna()].groupby("user_id")["_ts"].transform("min").reindex(order.index)
    order["first_seen"] = first_ts.where(prev_cnt > 0)

    order = order.sort_values("_pos")
    for col in ("accounts_per_device", "users_per_card", "amount_to_baseline", "first_seen"):
        df[col] = order[col].values
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn merged + contextualised rows into the model's feature matrix columns."""
    df = df.copy()
    for col in BOOL_COLS:
        df[col] = df[col].fillna(0).astype(int) if col in df else 0
    ts = to_naive_datetime(df["timestamp"])
    # account_created when the store sends it, else the user's first earlier order (a lower
    # bound on the age), else unknown. Unknown is NOT "created right now": it gets
    # account_age_days = -1 and account_age_known = 0, a case the model is trained on.
    created = (to_naive_datetime(df["account_created"], errors="coerce") if "account_created" in df
               else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]"))
    if "first_seen" in df:
        created = created.fillna(to_naive_datetime(df["first_seen"], errors="coerce"))
    known = created.notna()

    df["country_mismatch"] = (df["ip_country"] != df["billing_country"]).astype(int)
    df["high_risk_country"] = (df["ip_country"].isin(HIGH_RISK_COUNTRIES)
                               | df["billing_country"].isin(HIGH_RISK_COUNTRIES)).astype(int)
    age = ((ts - created).dt.total_seconds() / 86400).clip(lower=0).round(2)
    df["account_age_days"] = age.where(known, UNKNOWN_ACCOUNT_AGE).astype(float)
    df["account_age_known"] = known.astype(int)
    # optional behaviour signals: neutral defaults when the store doesn't send them
    for col, default, typ in (("login_attempts", 1, int), ("failed_logins", 0, int),
                              ("pages_viewed", 5, int), ("checkout_seconds", 120.0, float)):
        df[col] = (df[col] if col in df else pd.Series(default, index=df.index)).fillna(default).astype(typ)
    df["rushed_checkout"] = (df["checkout_seconds"] < RUSHED_CHECKOUT_SECONDS).astype(int)
    df["new_account_high_spend"] = (known & (df["account_age_days"] < NEW_ACCOUNT_DAYS)
                                    & (df["amount"] >= HIGH_SPEND_AMOUNT)).astype(int)
    df["log_amount"] = np.round(np.log1p(df["amount"]), 4)
    df["hour"] = ts.dt.hour
    df["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(int)
    return df


def prepare_single(payload: dict, history: pd.DataFrame, global_median: float | None = None) -> pd.DataFrame:
    """Build a one-row feature frame for an incoming transaction (used by the API).

    `history` = stored orders (user_id, device_id, card_hash, amount and optionally
    timestamp). Orders stamped after this one are ignored, so a late-arriving or
    back-dated order sees the same past training gave it. `global_median` is the
    store-wide median amount of those earlier orders when the caller already knows it.
    """
    row = pd.DataFrame([payload])
    ts = payload.get("timestamp")
    row["timestamp"] = to_naive_datetime(pd.Series([ts if ts not in (None, "") else pd.Timestamp.now()]))
    for col in ("billing_country", "ip_country"):
        row[col] = str(payload.get(col) or "US").strip().upper()
    for col in BOOL_COLS:
        row[col] = parse_bool(payload.get(col))
    if payload.get("account_created") in (None, ""):
        row["account_created"] = pd.NaT   # unknown, NOT "created now" (see build_features)
    cols = ["user_id", "device_id", "card_hash", "amount"]
    if len(history):
        hist = history
        if "timestamp" in hist:
            when = to_naive_datetime(hist["timestamp"], errors="coerce")
            hist = hist[when.isna() | (when <= row["timestamp"].iloc[0])]
        hist = hist[[*cols, "timestamp"]] if "timestamp" in hist else hist[cols]
    else:
        hist = pd.DataFrame(columns=cols)
    # history = orders known before this one; add_context adds the order itself to the
    # device/card counts and keeps it out of the spend baseline (same as training)
    return build_features(add_context(row, hist, global_median=global_median))


# --------------------------------------------------------------------------- pipeline
def run(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    tx, logs, devices = load_raw(raw_dir)
    n_raw = len(tx)
    tx = clean_transactions(tx)
    merged = merge_sources(tx, logs, devices)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "cleaned_transactions.csv", index=False)

    feats = build_features(add_context(merged))
    id_cols = ["transaction_id", "user_id", "device_id", "card_hash", "timestamp",
               "billing_country", "ip_country"]
    feats[id_cols + FEATURES + ["is_fraud"]].to_csv(out_dir / "feature_engineered.csv", index=False)
    print(f"cleaned: {n_raw} raw -> {len(tx)} rows | features: {len(FEATURES)} columns")
    return feats


if __name__ == "__main__":
    run()
