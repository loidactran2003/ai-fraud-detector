"""
Generate realistic *synthetic* raw data for the AI Fraud Detector.

Produces (in 2_Data/raw_data/):
  - transactions.csv    one row per order/payment
  - user_logs.json      one record per checkout session (login + behaviour signals)
  - device_signals.csv  one row per device fingerprint

Fraud is injected the way it shows up in real small-business stores:
  * fraud rings: new accounts sharing a few devices and a pool of stolen cards
  * lone fraudsters: their own clean device and a single stolen card
  * account takeovers: legit accounts suddenly used from VPN / new country
  * card testing: rushed checkouts, failed logins, tiny amounts
Signals deliberately OVERLAP with legitimate behaviour (households share
devices and cards, real customers use VPNs, new customers sign up and buy the
same day, ~35% of fraudsters use aged or bought accounts, some fraudsters browse
like shoppers) so the model has to learn combinations rather than one giveaway
rule. Labels are noisy the way chargeback labels are: some fraud is never
disputed, and a few legitimate orders get "friendly fraud" chargebacks. Some
sessions arrive without an account-creation time (stores that don't send it).

Usage:  python 2_Data/generate_data.py [--transactions 20000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw_data"

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "MX", "BR", "IN", "JP"]
COUNTRY_WEIGHTS = [0.55, 0.10, 0.08, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03]
HIGH_RISK_COUNTRIES = ["NG", "RU", "UA", "VN", "ID", "PK", "RO"]
OS_LIST = ["iOS", "Android", "Windows", "macOS", "Linux"]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "Samsung Internet"]
RESOLUTIONS = ["1920x1080", "1366x768", "390x844", "412x915", "1440x900", "800x600"]
START = pd.Timestamp("2026-01-01")
DAYS = 270


def _ip(rng: np.random.Generator) -> str:
    return ".".join(str(x) for x in rng.integers(1, 255, size=4))


def _card(rng: np.random.Generator) -> str:
    return f"C{rng.integers(10**11, 10**12)}"


class _Devices:
    """Registry of device fingerprints with per-device risk traits."""

    def __init__(self, rng):
        self.rng, self.rows = rng, []

    def new(self, emulator_p=0.01, rooted_p=0.03, tz_p=0.05) -> str:
        r = self.rng
        did = f"D{len(self.rows):05d}"
        emu = bool(r.random() < emulator_p)
        self.rows.append({
            "device_id": did,
            "os": r.choice(OS_LIST, p=[0.3, 0.3, 0.25, 0.12, 0.03]),
            "browser": r.choice(BROWSERS),
            "is_emulator": emu,
            "is_rooted": bool(r.random() < rooted_p),
            "timezone_mismatch": bool(r.random() < tz_p),
            "screen_resolution": "800x600" if emu and r.random() < 0.7 else r.choice(RESOLUTIONS[:-1]),
            "first_seen": (START + pd.Timedelta(days=int(r.integers(0, DAYS)))).strftime("%Y-%m-%d"),
        })
        return did


def generate(n_transactions: int = 20000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dev = _Devices(rng)

    # ---------------- users ----------------
    n_users = max(200, n_transactions // 10)
    n_fraud = int(n_users * 0.05)
    is_fraudster = np.r_[np.zeros(n_users - n_fraud, bool), np.ones(n_fraud, bool)]
    users = pd.DataFrame({"user_id": [f"U{i:05d}" for i in range(n_users)], "is_fraudster": is_fraudster})
    users["home_country"] = rng.choice(COUNTRIES, size=n_users, p=COUNTRY_WEIGHTS)
    # created_day relative to START: 35% of real customers sign up during the year;
    # 35% of fraudsters work from aged / bought / taken-over accounts, the rest sign up in-year
    legit_created = np.where(rng.random(n_users) < 0.35, rng.integers(0, DAYS - 5, n_users),
                             -rng.integers(30, 1500, n_users))
    aged_fraud = rng.random(n_users) < 0.35
    fraud_created = np.where(aged_fraud, -rng.integers(30, 900, n_users), rng.integers(0, DAYS - 15, n_users))
    users["created_day"] = np.where(is_fraudster, fraud_created, legit_created)
    users["aged_account"] = is_fraudster & aged_fraud
    # one creation time per account (not a new random hour on every order)
    users["created_minute"] = rng.integers(0, 24 * 60, n_users)
    # aged fraud accounts start their burst at a random point of the year
    users["burst_day"] = np.where(users.aged_account, rng.integers(0, DAYS - 15, n_users), users.created_day)
    users["typical_spend"] = np.round(rng.lognormal(mean=3.8, sigma=0.6, size=n_users), 2)

    # ---------------- devices & cards ----------------
    ring_devices = [dev.new(emulator_p=0.45, rooted_p=0.4, tz_p=0.5) for _ in range(max(6, n_fraud // 8))]
    household_devices = [dev.new() for _ in range(max(5, n_users // 50))]  # family PCs, shop tablets
    stolen_cards = [_card(rng) for _ in range(max(10, n_fraud // 4))]

    user_devices, user_cards = {}, {}
    legit_ids = users.user_id[~is_fraudster].tolist()
    for uid, fraudster in zip(users.user_id, users.is_fraudster):
        if fraudster:
            ring = rng.random() < 0.5
            user_devices[uid] = (list(rng.choice(ring_devices, 2, replace=False)) if ring
                                 else [dev.new(emulator_p=0.15, rooted_p=0.15, tz_p=0.25)])
            user_cards[uid] = (list(rng.choice(stolen_cards, 2, replace=False)) if rng.random() < 0.5
                               else [_card(rng)])
        else:
            devs = [dev.new() for _ in range(rng.integers(1, 3))]
            if rng.random() < 0.10:
                devs.append(rng.choice(household_devices))
            user_devices[uid] = devs
            user_cards[uid] = [_card(rng) for _ in range(rng.integers(1, 3))]
    # ~3% of real customers share a card with another account (spouses, small business card)
    for uid in rng.choice(legit_ids, size=int(len(legit_ids) * 0.03), replace=False):
        other = rng.choice(legit_ids)
        user_cards[uid].append(user_cards[other][0])

    # ---------------- transactions + session logs ----------------
    weights = np.where(is_fraudster, 1.5, 1.0)
    tx_users = rng.choice(users.index, size=n_transactions, p=weights / weights.sum())

    rows, logs = [], []
    for i, uidx in enumerate(tx_users):
        u = users.loc[uidx]
        uid, fraudster = u.user_id, bool(u.is_fraudster)
        takeover = (not fraudster) and rng.random() < 0.012
        is_fraud = (fraudster and rng.random() < 0.88) or takeover

        # --- timing: fraudsters burst soon after sign-up (or after picking up an aged
        # account), some wait first; new real customers often buy in their first days too
        if fraudster:
            gap = rng.integers(0, 4) if rng.random() < 0.5 else rng.integers(0, 60)
            day = int(min(DAYS - 1, u.burst_day + gap))
        elif u.created_day >= 0 and rng.random() < 0.3:
            day = int(min(DAYS - 1, u.created_day + rng.integers(0, 3)))
        else:
            day = int(rng.integers(max(0, u.created_day), DAYS))
        hour = int(rng.choice(24, p=_hour_profile(is_fraud)))
        ts = START + pd.Timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60)))
        account_created = START + pd.Timedelta(days=int(u.created_day), minutes=int(u.created_minute))
        if account_created >= ts:  # order on the sign-up day: it comes after the sign-up
            ts = account_created + pd.Timedelta(minutes=int(rng.integers(5, 240)))

        # --- device & card
        if takeover:
            device_id = rng.choice(ring_devices) if rng.random() < 0.3 else dev.new(emulator_p=0.1, tz_p=0.3)
        else:
            device_id = rng.choice(user_devices[uid])
        card = rng.choice(user_cards[uid])

        # --- network
        billing_country = u.home_country
        if is_fraud:
            vpn, proxy = rng.random() < 0.40, rng.random() < 0.20
            r = rng.random()
            ip_country = (rng.choice(HIGH_RISK_COUNTRIES) if r < 0.25
                          else rng.choice(COUNTRIES) if r < 0.55 else billing_country)
            if rng.random() < 0.10:
                billing_country = rng.choice(HIGH_RISK_COUNTRIES)
        else:
            vpn, proxy = rng.random() < 0.08, rng.random() < 0.02
            r = rng.random()
            ip_country = (billing_country if r < 0.92
                          else rng.choice(COUNTRIES) if r < 0.99 else rng.choice(HIGH_RISK_COUNTRIES))

        # --- amount
        if is_fraud:
            style = rng.random()
            if style < 0.40:
                amount = rng.lognormal(5.8, 0.7)                 # big-ticket cash-out
            elif style < 0.55:
                amount = rng.uniform(0.5, 5)                     # card testing
            else:
                amount = u.typical_spend * rng.lognormal(0.3, 0.5)  # blend in
        else:
            amount = u.typical_spend * rng.lognormal(0, 0.45)
            if rng.random() < 0.03:
                amount *= rng.uniform(4, 10)                     # legit big purchase
            elif rng.random() < 0.02:
                amount = rng.uniform(1, 5)                       # gift-card top-up, samples
        amount = round(float(amount), 2)

        # --- behaviour
        session_id = f"S{i:07d}"
        if is_fraud:
            login_attempts = int(1 + rng.poisson(1.3))
            failed = int(min(login_attempts - 1, rng.poisson(1.0)))
            # card testers and scripted checkouts (~1 in 4 of the rest) rush; others take their time
            scripted = (0.40 <= style < 0.55) or rng.random() < 0.25
            checkout_seconds = float(np.round(rng.gamma(1.8, 6.0) if scripted else rng.gamma(2.5, 22.0), 1))
            # half go straight to checkout, half browse like a shopper to blend in
            pages_viewed = int(rng.integers(1, 9) if rng.random() < 0.5 else rng.integers(2, 18))
        else:
            login_attempts = int(1 + rng.poisson(0.35))
            failed = int(min(login_attempts - 1, rng.poisson(0.2)))
            fast = rng.random() < 0.12  # returning customer re-ordering quickly
            checkout_seconds = float(np.round(rng.gamma(2.0, 12.0) if fast else rng.gamma(4.0, 35.0), 1))
            pages_viewed = int(rng.integers(1, 6) if fast else rng.integers(2, 25))

        # --- observed label = chargeback, not the truth: ~5% of fraud is never disputed,
        # ~0.1% of real orders get a "friendly fraud" chargeback
        label = int(rng.random() >= 0.05) if is_fraud else int(rng.random() < 0.001)

        rows.append({
            "transaction_id": f"T{i:07d}", "session_id": session_id, "user_id": uid,
            "device_id": device_id, "card_hash": card,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount, "currency": "USD",
            "billing_country": billing_country, "ip_address": _ip(rng), "ip_country": ip_country,
            "is_vpn": int(vpn), "is_proxy": int(proxy), "is_fraud": label,
        })
        logs.append({
            "session_id": session_id, "user_id": uid,
            # ~8% of sessions come from integrations that don't send the sign-up time
            "account_created": (None if rng.random() < 0.08 else account_created.strftime("%Y-%m-%d %H:%M:%S")),
            "session_start": (ts - pd.Timedelta(seconds=checkout_seconds + pages_viewed * 20)).strftime("%Y-%m-%d %H:%M:%S"),
            "login_attempts": login_attempts, "failed_logins": failed,
            "pages_viewed": pages_viewed, "checkout_seconds": checkout_seconds,
        })

    tx = pd.DataFrame(rows)
    devices = pd.DataFrame(dev.rows)
    # inject a little dirt so the cleaning step has real work to do
    dirty = rng.choice(len(tx), size=int(len(tx) * 0.005), replace=False)
    half = len(dirty) // 2
    tx.loc[dirty[:half], "amount"] = np.nan
    tx.loc[dirty[half:], "billing_country"] = tx.loc[dirty[half:], "billing_country"].str.lower()
    tx = pd.concat([tx, tx.sample(20, random_state=seed)], ignore_index=True)  # duplicate rows

    tx.to_csv(RAW_DIR / "transactions.csv", index=False)
    with open(RAW_DIR / "user_logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=1)
    devices.to_csv(RAW_DIR / "device_signals.csv", index=False)

    print(f"transactions.csv   {len(tx):>6} rows  (fraud rate {tx.is_fraud.mean():.1%})")
    print(f"user_logs.json     {len(logs):>6} sessions")
    print(f"device_signals.csv {len(devices):>6} devices")


def _hour_profile(is_fraud: bool) -> np.ndarray:
    hours = np.arange(24)
    base = 0.3 + np.exp(-((hours - 14) ** 2) / 30) + 0.6 * np.exp(-((hours - 20) ** 2) / 8)
    if is_fraud:  # same daily shape but heavier in the small hours
        base = base + 0.9 * ((hours <= 5) | (hours >= 23))
    return base / base.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    generate(a.transactions, a.seed)
