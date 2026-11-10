"""Build every generated document from the current model + latest QA run.

    python scripts/build_docs.py

Writes
  1_Project_Proposal/{project_overview,problem_statement,system_objectives,references}.pdf
  6_Documentation/{architecture_diagram,data_flow_diagram}.png
  6_Documentation/{model_performance_report,explainability_report,user_manual}.pdf
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
for p in ("3_Model", "3_Model/training", "4_Backend/utils"):
    sys.path.insert(0, str(ROOT / p))
DOCS, PROPOSAL = ROOT / "6_Documentation", ROOT / "1_Project_Proposal"
FIG = DOCS / "figures"
BLUE, ORANGE, AQUA, INK, INK2, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#898781"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
TITLE = "AI Fraud Detector for Small Businesses"

# ---------------------------------------------------------------- pdf helpers
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=20, textColor=colors.HexColor(INK), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13.5, textColor=colors.HexColor(BLUE), spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14.5, textColor=colors.HexColor(INK))
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor(INK2))
CODE = ParagraphStyle("C", parent=BODY, fontName="Courier", fontSize=8.5, leading=11, backColor=colors.HexColor("#f0efec"),
                      borderPadding=6, spaceBefore=4, spaceAfter=8)


def P(t, s=BODY): return Paragraph(t, s)
def bullets(items): return ListFlowable([ListItem(P(i), leftIndent=12) for i in items], bulletType="bullet", start="•", leftIndent=14)


def table(rows, widths=None, header=True):
    t = Table([[P(str(c), SMALL) for c in r] for r in rows], colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e3e2dd")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")))
    t.setStyle(TableStyle(style))
    return t


def img(path, width=6.5 * inch):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def build_pdf(path: Path, title: str, subtitle: str, story: list):
    def footer(c, d):
        c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor(MUTED))
        c.drawString(0.75 * inch, 0.5 * inch, f"{TITLE} · {title}")
        c.drawRightString(LETTER[0] - 0.75 * inch, 0.5 * inch, f"Page {d.page}")
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch, title=title, author="AI Fraud Detector team")
    head = [P(title, H1), P(f"{subtitle} · generated {date.today():%B %d, %Y}", SMALL), Spacer(1, 10)]
    doc.build(head + story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------- diagrams
def _box(ax, x, y, w, h, title, sub="", color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.4, color=INK2, linespacing=1.3)


def _arrow(ax, a, b, label="", color=INK2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, label, ha="center", va="center", fontsize=7, color=INK2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def architecture_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 6.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(0.1, 6.15, "System architecture", fontsize=14, weight="bold", color=INK)
    ax.text(0.1, 5.85, "Product path (blue) and quality-assurance layer (orange)", fontsize=9, color=INK2)
    _box(ax, 0.2, 3.9, 2.2, 1.3, "Store checkout", "orders · device fingerprint\nlogin + session logs")
    _box(ax, 3.0, 3.9, 2.2, 1.3, "Preprocessing", "4_Backend/utils\nclean · merge · features")
    _box(ax, 5.8, 3.9, 2.4, 1.3, "Fraud model + scorer", "3_Model · GBM\nscore 0-100 · action · reasons")
    _box(ax, 8.8, 3.9, 2.9, 1.3, "Flask REST API", "4_Backend/api\n/api/score · /transactions · /stats")
    _box(ax, 8.8, 1.9, 2.9, 1.2, "SQLite database", "transactions · risk_assessments\nreviews (feedback labels)")
    _box(ax, 8.8, 0.2, 2.9, 1.1, "Merchant dashboard", "5_Frontend/dashboard (HTML/JS)")
    _box(ax, 0.2, 1.9, 2.2, 1.2, "Data pipeline", "2_Data · generate → raw\n→ processed", AQUA)
    _box(ax, 3.0, 1.9, 2.2, 1.2, "Training", "3_Model/training\ntime split · fraud_model.pkl", AQUA)
    _box(ax, 5.8, 1.9, 2.4, 1.2, "Evaluation / QA layer", "3_Model/evaluation\nprompts · golden cases · gates", ORANGE)
    _box(ax, 5.8, 0.2, 2.4, 1.1, "Streamlit QA console", "5_Frontend/streamlit_app", ORANGE)
    _box(ax, 3.0, 0.2, 2.2, 1.1, "LLM (OpenAI / mock)", "fraud-analyst prompts", ORANGE)
    _arrow(ax, (2.4, 4.55), (3.0, 4.55)); _arrow(ax, (5.2, 4.55), (5.8, 4.55)); _arrow(ax, (8.2, 4.55), (8.8, 4.55))
    _arrow(ax, (10.25, 3.9), (10.25, 3.1)); _arrow(ax, (10.25, 1.9), (10.25, 1.3))
    _arrow(ax, (2.4, 2.5), (3.0, 2.5)); _arrow(ax, (4.1, 3.1), (5.8, 4.1), "model.pkl", rad=-0.2)
    _arrow(ax, (7.0, 3.9), (7.0, 3.1), "tests", ORANGE); _arrow(ax, (7.0, 1.9), (7.0, 1.3), "run history", ORANGE)
    _arrow(ax, (5.8, 2.2), (5.2, 0.9), "", ORANGE); _arrow(ax, (8.8, 2.5), (8.2, 2.5), "orders", ORANGE)
    _arrow(ax, (8.8, 2.0), (8.2, 0.9), "", ORANGE, rad=0.1)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def data_flow_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(0.1, 4.35, "Data flow", fontsize=14, weight="bold", color=INK)
    steps = [("Raw data", "transactions.csv\nuser_logs.json\ndevice_signals.csv", AQUA),
             ("Clean + merge", "dedupe · fix codes\ndrop bad amounts", AQUA),
             ("Feature eng.", "22 signals: network,\ndevice, behaviour,\ntransaction", AQUA),
             ("Train", "time-based split\ngradient boosting", BLUE),
             ("Score", "0-100 risk\napprove/review/block", BLUE),
             ("Explain", "what-if reasons\nin plain language", BLUE)]
    for i, (t, s, c) in enumerate(steps):
        x = 0.2 + i * 1.97
        _box(ax, x, 2.2, 1.7, 1.6, t, s, c)
        if i:
            _arrow(ax, (x - 0.27, 3.0), (x, 3.0))
    _box(ax, 9.0, 0.2, 2.4, 1.2, "Merchant review", "approve · block ·\nconfirmed fraud", ORANGE)
    _box(ax, 3.3, 0.2, 2.4, 1.2, "QA gates", "FP/FN · drift · signals\nchampion vs baseline", ORANGE)
    _arrow(ax, (10.9, 2.2), (10.6, 1.4), "flagged orders")
    _arrow(ax, (9.0, 0.8), (6.96, 2.2), "reviewer labels → retraining", ORANGE, rad=-0.25)
    _arrow(ax, (5.7, 0.8), (6.5, 2.2), "release gate", ORANGE, rad=0.2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------- figures from the model
def model_figures():
    from risk_scoring import RiskScorer
    from sklearn.metrics import precision_recall_curve, roc_curve
    FIG.mkdir(parents=True, exist_ok=True)
    scorer = RiskScorer()
    test = pd.read_csv(ROOT / "3_Model/saved_models/test_set.csv")
    s, y = scorer._scores(test), test.is_fraud.values

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.6))
    fpr, tpr, _ = roc_curve(y, s); a.plot(fpr, tpr, color=BLUE, lw=2); a.plot([0, 1], [0, 1], ":", color=MUTED)
    a.set(title="ROC curve", xlabel="false positive rate", ylabel="true positive rate")
    p, r, _ = precision_recall_curve(y, s); b.plot(r, p, color=BLUE, lw=2); b.set(title="Precision-recall", xlabel="recall", ylabel="precision")
    bins = np.arange(0, 105, 5)
    c.hist(s[y == 0], bins, color=BLUE, alpha=.85, label="legit", log=True)
    c.hist(s[y == 1], bins, color=ORANGE, alpha=.85, label="fraud", log=True)
    for t, col in ((scorer.review_threshold, WARN), (scorer.block_threshold, CRIT)):
        c.axvline(t, color=col, ls="--", lw=1)
    c.set(title="Risk score by true class", xlabel="risk score", ylabel="orders (log)"); c.legend(frameon=False)
    for ax in (a, b, c):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "performance_curves.png", dpi=150); plt.close(fig)

    import shap
    X = test[scorer.features].sample(1200, random_state=0)
    sv = shap.TreeExplainer(scorer.model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=scorer.features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.barh(imp.index, imp.values, color=BLUE, height=.65); ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title("Global signal importance (SHAP)"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "shap_importance.png", dpi=150); plt.close(fig)

    example = test[s >= scorer.block_threshold].iloc[0:1]
    return scorer, test, s, y, imp, example, scorer.score(example)[0]


def latest_run():
    path = ROOT / "3_Model/evaluation/results/results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- documents
def proposal_docs():
    PROPOSAL.mkdir(exist_ok=True)
    build_pdf(PROPOSAL / "project_overview.pdf", "Project Overview", TITLE, [
        P("Small businesses are hit by fake orders, stolen cards and suspicious accounts, and rarely have a fraud team. "
          "This project builds an AI tool that automatically detects risky transactions so businesses don't lose money. "
          "It works like a smart fraud analyst that watches for unusual behaviour."),
        P("What the AI looks at", H2),
        table([["Signal family", "Signals"],
               ["Network activity", "IP location, VPN / proxy usage, country mismatch, high-risk countries"],
               ["Device behaviour", "Emulator or virtual device, rooted device, timezone mismatch, device reused across many accounts"],
               ["User behaviour", "Too many / failed login attempts, rushed checkout, straight-to-checkout sessions, new accounts spending unusually large amounts"],
               ["Transaction patterns", "Strange or abnormal amounts, card-testing amounts, cards reused across accounts, odd hours"]],
              [1.6 * inch, 5.0 * inch]),
        P("How the AI helps", H2),
        bullets(["Learns from past normal and fraudulent behaviour", "Spots combinations of signals humans miss",
                 "Gives each transaction a risk score (0-100) and an action: approve, review or block",
                 "Shows <b>why</b> something looks risky, in plain language", "Helps businesses review and stop fraud early",
                 "Ships with a QA layer that continuously tests accuracy, false positives, explanation quality and drift"]),
        P("Deliverables", H2),
        bullets(["Data pipeline and synthetic dataset (2_Data)", "Trained model, risk scoring and explainability (3_Model)",
                 "Evaluation & QA layer with prompt tests, golden fraud cases and model gates (3_Model/evaluation)",
                 "REST API and database (4_Backend)", "Merchant dashboard and Streamlit QA console (5_Frontend)",
                 "Architecture, performance, explainability reports and user manual (6_Documentation)"]),
    ])
    build_pdf(PROPOSAL / "problem_statement.pdf", "Problem Statement", TITLE, [
        P("The problem", H2),
        P("Online fraud costs merchants far more than the stolen goods: every fraudulent order also brings a chargeback fee, "
          "lost shipping, staff time and higher payment-processing risk. Large retailers employ fraud teams and buy enterprise tools; "
          "small businesses typically rely on gut feeling or a handful of static rules (\"block orders over $500 from abroad\")."),
        P("Why static rules fail", H2),
        bullets(["They block good customers: travellers, VPN users and big-but-genuine purchases (false positives cost sales and loyalty).",
                 "They miss quiet fraud: new accounts using stolen cards for ordinary-looking amounts.",
                 "They can't see links across orders: one device or card shared by many accounts.",
                 "They give no explanation, so owners can't tell a good flag from a bad one."]),
        P("Who is affected", H2),
        P("Independent online shops, marketplaces sellers and service businesses with roughly 100-10,000 orders per month, "
          "no dedicated risk staff, and thin margins where a few chargebacks wipe out a week's profit."),
        P("Opportunity", H2),
        P("Fraud leaves combined traces across network, device, behaviour and transaction data. A model that learns these "
          "combinations, scores every order in milliseconds and explains itself can give small businesses the protection "
          "of a fraud analyst at the cost of an API call."),
    ])
    build_pdf(PROPOSAL / "system_objectives.pdf", "System Objectives", TITLE, [
        table([["#", "Objective", "Measure of success", "Where it is tested"],
               ["1", "Detect fraudulent transactions", "ROC-AUC ≥ 0.90, PR-AUC ≥ 0.75 on most-recent orders", "mq-auc gate"],
               ["2", "Avoid blocking good customers", "False-positive rate ≤ 3% at the review threshold", "mq-errors gate, fx-006…fx-010"],
               ["3", "Catch most fraud", "False-negative rate ≤ 10%", "mq-errors gate"],
               ["4", "Explain every flag", "Reasons on 100% of flagged orders; reason precision ≥ 80%", "mq-signals, explanation_clarity"],
               ["5", "Behave logically", "Adding a red flag never lowers risk; deterministic scores", "mq-logic gate"],
               ["6", "Stay reliable over time", "Feature and score PSI < 0.25 vs training data", "mq-drift gate"],
               ["7", "Beat simple alternatives", "Higher PR-AUC than logistic regression and a rules checklist", "mq-compare gate"],
               ["8", "Safe LLM assistance", "LLM analyst refuses fraud-evasion requests, answers merchant questions", "fadv-* prompt tests"],
               ["9", "Fast and easy to integrate", "Single REST call per order; dashboard for review", "4_Backend/api"]],
              [0.3 * inch, 1.7 * inch, 2.6 * inch, 2.0 * inch]),
        P("Non-goals", H2),
        bullets(["Processing or storing raw card numbers (only hashed card tokens are used).",
                 "Fully automatic chargeback disputes.", "Replacing the payment processor's own fraud screening."]),
    ])
    build_pdf(PROPOSAL / "references.pdf", "References", TITLE, [
        bullets([
            "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. <i>Annals of Statistics</i>, 29(5).",
            "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP). <i>NeurIPS</i>.",
            "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). \"Why should I trust you?\" Explaining the predictions of any classifier (LIME). <i>KDD</i>.",
            "Dal Pozzolo, A., et al. (2015). Calibrating probability with undersampling for unbalanced classification. <i>IEEE SSCI</i>.",
            "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot on imbalanced datasets. <i>PLOS ONE</i>.",
            "Yurdakul, B. (2018). Statistical properties of population stability index. Western Michigan University.",
            "Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. <i>JMLR</i> 12.",
            "PCI Security Standards Council. PCI DSS v4.0 - guidance on tokenisation and storage of cardholder data.",
        ]),
        P("Note: the dataset in 2_Data is synthetic, generated by 2_Data/generate_data.py to mimic published fraud patterns "
          "(fraud rings, card testing, account takeover). Replace it with your store's labelled history before production use.", SMALL),
    ])


def performance_report(scorer, test, s, y, run):
    m = scorer.metrics
    model_rows = [r for r in (run or {}).get("results", []) if r.get("suite") == "model"]
    gates = [["Check", "Result", "Status"]] + [[r["id"] + " · " + r["task"], r["details"].get("summary", ""),
                                                 "PASS" if r["passed"] else "FAIL"] for r in model_rows]
    cmp = (run or {}).get("run", {}).get("model_comparison") or []
    thresholds = [["Threshold", "Orders flagged", "Precision", "Recall", "Fraud $ caught"]]
    for t in sorted({10, 20, scorer.review_threshold, 50, scorer.block_threshold, 90}):
        f = s >= t
        thresholds.append([t, f"{int(f.sum()):,} ({f.mean():.1%})", f"{(y[f] == 1).mean():.1%}" if f.any() else "-",
                           f"{f[y == 1].mean():.1%}", f"${test.amount[f & (y == 1)].sum():,.0f}"])
    rv, bl = m["review"], m["block"]
    story = [
        P("Summary", H2),
        P(f"Model <b>{scorer.version}</b>, a gradient-boosted tree ensemble over {len(scorer.features)} engineered signals, trained on "
          f"{m['n_train']:,} orders and tested on the most recent {m['n_test']:,} (time-based split; test fraud rate "
          f"{m['test_fraud_rate']:.1%}). ROC-AUC <b>{m['roc_auc']}</b>, PR-AUC <b>{m['pr_auc']}</b> "
          f"(logistic-regression baseline PR-AUC {m['baseline_logreg']['pr_auc']})."),
        table([["Operating point", "Precision", "Recall", "Orders flagged", "Confusion (TN / FP / FN / TP)"],
               [f"Review (score ≥ {rv['threshold']})", f"{rv['precision']:.1%}", f"{rv['recall']:.1%}", f"{rv['flag_rate']:.1%}",
                " / ".join(str(rv['confusion'][k]) for k in ('tn', 'fp', 'fn', 'tp'))],
               [f"Block (score ≥ {bl['threshold']})", f"{bl['precision']:.1%}", f"{bl['recall']:.1%}", f"{bl['flag_rate']:.1%}",
                " / ".join(str(bl['confusion'][k]) for k in ('tn', 'fp', 'fn', 'tp'))]]),
        Spacer(1, 8), img(FIG / "performance_curves.png"),
        P("Threshold trade-off", H2), table(thresholds),
    ]
    if cmp:
        story += [P("Champion vs baselines", H2),
                  table([["Model", "ROC-AUC", "PR-AUC", f"Precision@{scorer.review_threshold}", f"Recall@{scorer.review_threshold}", f"FPR@{scorer.review_threshold}"]] +
                        [[c["model"], c["roc_auc"], c["pr_auc"], f"{c['precision']:.1%}", f"{c['recall']:.1%}",
                          f"{c['false_positive_rate']:.2%}"] for c in cmp])]
    if model_rows:
        story += [P("Automated quality gates (latest QA run)", H2), table(gates, [1.8 * inch, 4.0 * inch, 0.8 * inch])]
        drift = next((r for r in model_rows if r["task"] == "drift"), None)
        if drift:
            top = sorted(drift["details"]["feature_psi"].items(), key=lambda kv: -kv[1])[:6]
            story += [P("Drift watch-list (PSI, training vs current)", H2),
                      table([["Feature", "PSI", "Status"]] + [[f, v, "major" if v >= 0.25 else "moderate" if v >= 0.1 else "stable"] for f, v in top])]
    story += [P("Limitations", H2), bullets([
        "Trained on synthetic data; real-store performance must be re-measured on labelled history.",
        "Cross-order signals (shared devices / cards) are computed over all known history, so early orders in a ring look cleaner than later ones.",
        "Labels arrive late (chargebacks take weeks): monitor drift weekly and retrain monthly using reviewer decisions."])]
    build_pdf(DOCS / "model_performance_report.pdf", "Model Performance Report", f"Model {scorer.version}", story)


def explainability_report(scorer, imp, example, example_res, run):
    sig = next((r for r in (run or {}).get("results", []) if r.get("task") == "signal_detection"), None)
    story = [
        P("How explanations work", H2),
        P("Every order with an elevated score gets up to four plain-language reasons. For each signal that is genuinely abnormal "
          "on that order (e.g. a VPN is on, the card is linked to several accounts), the RiskScorer resets it to the value a typical "
          "legitimate order has and re-scores. The drop in the model's log-odds is that signal's impact. This is a local, "
          "model-faithful what-if method that needs no extra dependencies at serving time; SHAP and LIME are used offline to validate it."),
        P("Global importance", H2), img(FIG / "shap_importance.png", 4.6 * inch),
        P(f"Top drivers overall: {', '.join(imp.sort_values(ascending=False).index[:5])}.", SMALL),
        P("Example: a blocked order", H2),
        table([["Reason shown to the merchant", "Share of explained risk"]] +
              [[r["text"], f"{r['impact']}%"] for r in example_res["reasons"]]),
        P(f"Order {example.transaction_id.iloc[0]}: risk {example_res['risk_score']}/100 → {example_res['action']}.", SMALL),
    ]
    if sig:
        d = sig["details"]
        story += [P("Measured explanation quality", H2),
                  P(f"Over {d['n_explained']} correctly-caught fraud orders, <b>{d['reason_precision']:.0%}</b> of reasons cite a signal "
                    f"that is truly present (precision) and explanations cover <b>{d['signal_coverage']:.0%}</b> of the main present signals."),
                  table([["Signal", "Mentioned when present"]] + [[k, f"{v:.0%}"] for k, v in d["per_signal_mention_rate"].items()])]
    story += [P("Known weaknesses", H2), bullets([
        "When many signals co-occur the model may lean on one dominant signal (checkout speed is the top feature) and under-cite others.",
        "Context signals (shared device/card) depend on how much history the store has.",
        "Reason text is template-based; the LLM fraud-analyst prompts (fra-*) test free-text explanations for jargon and clarity."])]
    build_pdf(DOCS / "explainability_report.pdf", "Explainability Report", f"Model {scorer.version}", story)


def user_manual():
    from risk_scoring import RiskScorer
    sc = RiskScorer()
    rv, bl = sc.review_threshold, sc.block_threshold
    story = [
        P("1. Install", H2),
        P("python -m venv .venv<br/>.venv\\Scripts\\activate &nbsp;&nbsp;(macOS/Linux: source .venv/bin/activate)<br/>pip install -r requirements.txt", CODE),
        P("2. Build data and model (one command)", H2), P("python run_pipeline.py", CODE),
        P("Generates synthetic data, cleans it, engineers features, trains the model, seeds the database, runs every QA suite and rebuilds these documents."),
        P("3. Merchant dashboard", H2), P("python 4_Backend/api/fraud_detection_api.py &nbsp;&nbsp;→ open http://127.0.0.1:5000", CODE),
        bullets(["<b>Top tiles</b>: orders scored, sent to review, blocked, money at risk, awaiting review.",
                 f"<b>Risk score distribution</b>: green = approve (0-{rv - 1}), amber = review ({rv}-{bl - 1}), red = block ({bl}-100). "
                 "The thresholds are chosen per model on a validation period and shown in the header.",
                 "<b>Flagged transactions</b>: highest risk first, with the reasons. Click Approve / Block / Fraud / False alarm - "
                 "decisions are stored; Fraud / False alarm also set the order's label (export with connection.py --export-labels for retraining).",
                 "<b>Check an order</b>: score any order by hand to see how the model reacts (dry run - nothing is stored)."]),
        P("4. Integrate with your checkout", H2),
        P('POST /api/score &nbsp;{"user_id": "C123", "amount": 249.0, "ip_country": "US", "billing_country": "US",<br/>'
          '&nbsp;"is_vpn": 0, "device_id": "fp_9a1", "card_hash": "tok_77", "account_created": "2026-10-01 09:00:00",<br/>'
          '&nbsp;"checkout_seconds": 95, "login_attempts": 1, "failed_logins": 0, "pages_viewed": 6}', CODE),
        P("Response: risk_score, risk_level, action (approve / review / block) and reasons. Only user_id and amount are required; "
          "missing behaviour signals default to neutral values. A missing account_created is taken from the customer's first "
          "known order, or treated as unknown - never as 'created just now'."),
        table([["Action", "What to do"], ["approve", "Ship normally."],
               ["review", "Hold shipping; verify by email/phone or ask for 3-D Secure, then record your decision."],
               ["block", "Cancel/refund before shipping. Mark 'Fraud' to confirm, or 'False alarm' if it was genuine."]]),
        P("5. Quality-assurance console", H2), P("python 3_Model/evaluation/run_tests.py<br/>streamlit run 5_Frontend/streamlit_app/app.py", CODE),
        bullets(["<b>Evaluation</b>: per-test scores for LLM prompts, golden fraud cases and model gates.",
                 "<b>Run history</b>: quality over time. <b>Model comparison</b>: champion vs baselines.",
                 "Run with --provider openai --model gpt-4o-mini (and OPENAI_API_KEY) to evaluate a real LLM analyst; mock mode needs no key."]),
        P("6. Troubleshooting", H2),
        table([["Symptom", "Fix"], ["Dashboard empty", "python 4_Backend/database/connection.py --seed"],
               ["FileNotFoundError fraud_model.pkl", "python run_pipeline.py (or 3_Model/training/train_model.py)"],
               ["Drift gate fails", "Retrain on recent data; check the drifted features in the report"],
               ["OPENAI_API_KEY is not set", "Set it, or use --provider mock"]]),
    ]
    build_pdf(DOCS / "user_manual.pdf", "User Manual", "Merchant + operator guide", story)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    architecture_diagram(DOCS / "architecture_diagram.png")
    data_flow_diagram(DOCS / "data_flow_diagram.png")
    proposal_docs()
    scorer, test, s, y, imp, example, ex_res = model_figures()
    run = latest_run()
    performance_report(scorer, test, s, y, run)
    explainability_report(scorer, imp, example, ex_res, run)
    user_manual()
