# 2_Data

The dataset here is **synthetic**. `generate_data.py` produces it by simulating a small online store's traffic over 270 days, with realistic fraud patterns injected:

| Pattern | How it appears in the data |
|---|---|
| Fraud ring | accounts sharing a few devices (often emulated or rooted) and a pool of stolen cards |
| Lone fraudster | an account's own clean device and one stolen card, so it's harder to catch |
| Account takeover | an established account suddenly on a VPN or new device, from another country |
| Card testing | tiny amounts, failed logins, checkout in seconds (about 1 in 4 other fraud orders is scripted and rushed too) |
| Aged / bought accounts | ~35% of fraudsters use accounts created long before the attack, so account age alone doesn't give them away |

Fraud signals deliberately **overlap** with normal behaviour. Real customers use VPNs, households share devices and cards, new customers often buy on the day they sign up, returning customers re-order in seconds, and half of fraudsters browse like shoppers. No single signal gives fraud away: the top feature carries about a quarter of the model's importance (the QA suite fails if any feature exceeds 35%).

Realism details (all reproducible from `--seed`):
- each account has **one** creation time (it isn't re-drawn per order);
- **labels are chargebacks, not truth**: ~5% of fraud is never disputed (labelled 0) and ~0.1% of genuine orders get a "friendly fraud" chargeback (labelled 1);
- ~8% of sessions have **no `account_created`** (integrations that don't send it), so the model learns the "age unknown" case the API also sees.

## Files

| File | Rows | Description |
|---|---|---|
| `raw_data/transactions.csv` | ~20k | one row per order: amount, card token, IP/billing country, VPN/proxy, `is_fraud` label. Includes a few duplicates and malformed rows on purpose. |
| `raw_data/user_logs.json` | ~20k | one record per checkout session: account creation time, login attempts, failed logins, pages viewed, checkout seconds |
| `raw_data/device_signals.csv` | ~3k | one row per device fingerprint: OS, browser, emulator, rooted, timezone mismatch |
| `processed_data/cleaned_transactions.csv` | | deduplicated, validated, all three sources merged |
| `processed_data/feature_engineered.csv` | | the 22 model features + ids + label |

## Rebuild

```bash
python 2_Data/generate_data.py --transactions 20000 --seed 42
python 4_Backend/utils/preprocessing.py
```

## Features (22)

- **Network:** `is_vpn`, `is_proxy`, `country_mismatch`, `high_risk_country`
- **Device:** `is_emulator`, `is_rooted`, `timezone_mismatch`, `accounts_per_device`
- **User behaviour:** `login_attempts`, `failed_logins`, `checkout_seconds`, `rushed_checkout`, `pages_viewed`, `account_age_days`, `account_age_known`, `new_account_high_spend`
  - `account_age_days` uses `account_created` when present, else the customer's first earlier order (a lower bound), else it is unknown: `-1` with `account_age_known = 0`. A missing value is never read as "created just now".
- **Transaction:** `amount`, `log_amount`, `amount_to_baseline` (vs. the customer's usual spend), `users_per_card`, `hour`, `is_night`

## Using real data

Export your store's orders into the same three raw files (the column names are in `generate_data.py`), add a label from chargebacks and reviewer decisions, and run `python run_pipeline.py`. Never store raw card numbers; use your payment processor's card token or fingerprint as `card_hash`.
