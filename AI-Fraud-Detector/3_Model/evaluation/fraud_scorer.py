"""Fraud-specific scoring functions.

Every function takes (response, test) and returns a score in [0, 1], matching the
signature of the original scorer criteria so they plug into the same rubric.

`response` is either
  * str  - an LLM's free-text fraud assessment (task "fraud_analysis"), or
  * dict - the RiskScorer's output {risk_score, risk_level, action, reasons} (task "risk_scoring").

Signals use one canonical vocabulary shared by test cases, the model's reason
codes and the text matcher:

  vpn, proxy, country_mismatch, high_risk_country,               (network)
  emulator, rooted_device, timezone_mismatch, shared_device,     (device)
  new_account, login_failures, rushed_checkout, few_pages,       (user behaviour)
  abnormal_amount, shared_card, odd_hour                         (transaction pattern)
"""
import re

LEVELS = ["low", "medium", "high"]
ACTIONS = ["approve", "review", "block"]

# RiskScorer reason group -> canonical signal
GROUP_TO_SIGNAL = {
    "is_vpn": "vpn", "is_proxy": "proxy", "country_mismatch": "country_mismatch",
    "high_risk_country": "high_risk_country", "is_emulator": "emulator",
    "is_rooted": "rooted_device", "timezone_mismatch": "timezone_mismatch",
    "accounts_per_device": "shared_device", "users_per_card": "shared_card",
    "account_age_days": "new_account", "pages_viewed": "few_pages",
    "amount": "abnormal_amount", "checkout": "rushed_checkout", "time": "odd_hour",
    "logins": "login_failures",
}
SIGNALS = sorted(set(GROUP_TO_SIGNAL.values()))

# How each signal is recognised in free text.
SIGNAL_PATTERNS = {
    "vpn": r"\bvpns?\b",
    "proxy": r"\bprox(?:y|ies)\b",
    "country_mismatch": r"countr(?:y|ies).{0,60}(?:mismatch|differ|does ?n[o']?t match|different|outside)"
                        r"|(?:mismatch\w*|differ\w*|different).{0,60}countr",
    "high_risk_country": r"high[- ]risk (?:country|countries|region|location|jurisdiction|geograph)",
    "emulator": r"emulat|virtual (?:device|machine)",
    "rooted_device": r"\broot(?:ed)?\b|jailbr",
    "timezone_mismatch": r"time ?zone",
    "shared_device": r"(?:device|phone|computer).{0,40}(?:shared|used by|linked to|multiple|many|\d+ (?:different |other )?accounts)"
                     r"|(?:shared|same) device",
    "shared_card": r"card.{0,40}(?:shared|used by|linked to|multiple|many|\d+ (?:different |other )?accounts|reused)"
                   r"|(?:reused|shared|same) card|card re-?use",
    "new_account": r"new(?:ly[- ]created)? account|account (?:is |was )?(?:only )?(?:created|opened|registered)"
                   r"|brand[- ]new|account age",
    "few_pages": r"pages? viewed|straight to (?:the )?checkout|(?:no|little|minimal) browsing",
    "abnormal_amount": r"(?:amount|order value|order total|purchase|spend\w*|ticket).{0,40}"
                       r"(?:unusual|abnormal|large|high|above|higher|far|tiny|small|times|spike)"
                       r"|card[- ]testing|\d+(?:\.\d+)?x (?:their|the|this|normal|usual)",
    "rushed_checkout": r"\brush|(?:quick|fast|rapid|hurried) checkout|checkout.{0,30}(?:\d+ ?s(?:ec|econds)?\b|quick|fast)",
    "odd_hour": r"(?:late|night|unusual|odd) (?:hour|time)|middle of the night|\b(?:[1-4]|0[0-5]):\d\d ?(?:am)?\b",
    "login_failures": r"(?:failed|multiple|many|\d+) (?:login|log-in|sign-in|password)|login attempts",
}
NEGATION = re.compile(r"\b(?:no|not|without|isn'?t|wasn'?t|none|neither|nor|zero)\b[^.;]{0,25}$")

JARGON = re.compile(r"\b[a-z]+_[a-z_]+\b|\bshap\b|\blogit\b|\bfeature \d+|\bgradient boost|\bprobability of 0\.", re.I)
RECOMMENDATION = re.compile(r"\b(approve|review|block|hold|decline|cancel|verify|contact|ship|refund|reject)\b", re.I)
LEVEL_LINE = re.compile(r"risk(?: level)?\s*[:\-=]\s*\**\s*(low|medium|high)\b", re.I)
SCORE_LINE = re.compile(r"(?:risk )?score\s*[:\-=]\s*\**\s*(\d{1,3})\b", re.I)


# ----------------------------------------------------------------- extraction helpers
def signals_in_text(text):
    """Canonical signals mentioned (and not negated) in free text."""
    lowered = text.lower()
    found = set()
    for sig, pat in SIGNAL_PATTERNS.items():
        for m in re.finditer(pat, lowered):
            negated_before = NEGATION.search(lowered[max(0, m.start() - 40):m.start()])
            negated_inside = re.search(r"\b(?:not|no|never|isn'?t|aren'?t)\b", m.group(0))
            if not (negated_before or negated_inside):
                found.add(sig)
                break
    return found


def signals_in(response):
    if isinstance(response, dict):
        return {GROUP_TO_SIGNAL.get(r["feature"], r["feature"]) for r in response.get("reasons", [])}
    return signals_in_text(response or "")


def level_of(response):
    if isinstance(response, dict):
        return response.get("risk_level")
    m = LEVEL_LINE.search(response or "")
    if m:
        return m.group(1).lower()
    hits = [lv for lv in LEVELS if re.search(rf"\b{lv}\b[- ]risk", (response or "").lower())]
    return hits[0] if len(hits) == 1 else None


def score_of(response):
    if isinstance(response, dict):
        return response.get("risk_score")
    m = SCORE_LINE.search(response or "")
    return int(m.group(1)) if m and int(m.group(1)) <= 100 else None


def _range_score(value, lo, hi, tolerance=25):
    if lo <= value <= hi:
        return 1.0
    dist = lo - value if value < lo else value - hi
    return round(max(0.0, 1 - dist / tolerance), 3)


def _level_score(got, test):
    accepted = test.get("acceptable_levels") or [test["expected_level"]]
    if got in accepted:
        return 1.0
    if got is None:
        return 0.0
    gap = min(abs(LEVELS.index(got) - LEVELS.index(a)) for a in accepted)
    return 0.5 if gap == 1 else 0.0


# ----------------------------------------------------------------- criteria
def score_risk_accuracy(response, test):
    parts = []
    if "expected_level" in test:
        parts.append(_level_score(level_of(response), test))
    if "risk_range" in test:
        s = score_of(response)
        lo, hi = test["risk_range"]
        parts.append(_range_score(s, lo, hi) if s is not None else (0.0 if isinstance(response, dict) else None))
    parts = [p for p in parts if p is not None]
    return round(sum(parts) / len(parts), 3) if parts else 0.0


def score_action(response, test):
    accepted = test.get("acceptable_actions") or [test["expected_action"]]
    got = response.get("action") if isinstance(response, dict) else None
    if got in accepted:
        return 1.0
    if got is None:
        return 0.0
    gap = min(abs(ACTIONS.index(got) - ACTIONS.index(a)) for a in accepted)
    return 0.5 if gap == 1 else 0.0


def score_signal_detection(response, test):
    expected = set(test.get("expected_signals", []))
    if not expected:
        return 1.0
    return round(len(expected & signals_in(response)) / len(expected), 3)


def score_false_signals(response, test):
    absent = set(test.get("absent_signals", []))
    if not absent:
        return 1.0
    return round(1 - len(absent & signals_in(response)) / len(absent), 3)


def score_explanation_clarity(response, test):
    if isinstance(response, dict):
        reasons = response.get("reasons", [])
        if response.get("risk_score", 0) >= 30 and not reasons:
            return 0.0  # flagged with no explanation - unusable for the reviewer
        if not reasons:
            return 1.0
        texts = [r["text"] for r in reasons]
        checks = [
            all(not JARGON.search(t) for t in texts),            # no raw feature names
            all(len(t) <= 100 for t in texts),                    # scannable
            all(t[:1].isupper() or t[:1].isdigit() for t in texts),  # well-formed
            len({t for t in texts}) == len(texts),                # no duplicates
            any(re.search(r"\d", t) for t in texts),              # grounded in concrete values
        ]
        score = sum(checks) / len(checks)
        return round(min(score, 0.4) if not checks[0] else score, 3)

    text = (response or "").strip()
    if not text:
        return 0.0
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    words = len(text.split())
    checks = [
        not JARGON.search(text),
        bool(RECOMMENDATION.search(text)),
        words <= test.get("max_words", 160),
        sum(len(s.split()) for s in sentences) / len(sentences) <= 30,
        level_of(text) is not None,                              # states a clear verdict
    ]
    score = sum(checks) / len(checks)
    # model internals in an owner-facing explanation is a hard defect
    return round(min(score, 0.4) if not checks[0] else score, 3)


FRAUD_SCORERS = {
    "risk_accuracy": score_risk_accuracy,
    "action": score_action,
    "signal_detection": score_signal_detection,
    "false_signals": score_false_signals,
    "explanation_clarity": score_explanation_clarity,
}
