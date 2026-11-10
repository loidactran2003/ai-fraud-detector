"""AI Fraud Detector - Quality & Operations console (Streamlit).

    streamlit run 5_Frontend/streamlit_app/app.py

Tabs
  Fraud overview        risk distribution, money at risk, why orders are flagged   (reads the SQLite DB)
  Flagged transactions  filterable review queue with model reasons                 (reads the SQLite DB)
  Evaluation            scores by task / criterion / suite, per-test drill-down    (reads results/runs/*.json)
  Run history           quality over time across every recorded run
  Model comparison      champion vs baselines, and champion versions over time
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3_Model"))
from evaluation import config  # noqa: E402
from evaluation.rubric import RUBRIC  # noqa: E402

# same DB as the API (connection.py honours FRAUD_DB_PATH too)
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", ROOT / "4_Backend" / "database" / "fraud_detector.db"))


def _thresholds() -> tuple[int, int]:
    """Review / block thresholds of the saved model (fallback 30 / 70)."""
    try:
        m = json.loads((ROOT / "3_Model" / "saved_models" / "metrics.json").read_text())
        return int(m["thresholds"]["review"]), int(m["thresholds"]["block"])
    except (OSError, KeyError, TypeError, ValueError):
        return 30, 70


REVIEW_T, BLOCK_T = _thresholds()
BANDS = [f"Approve (0-{REVIEW_T - 1})", f"Review ({REVIEW_T}-{BLOCK_T - 1})", f"Block ({BLOCK_T}-100)"]

# Validated palette (dataviz reference instance): categorical order + reserved status colours
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
INK_2 = "#52514e"
REASON_LABELS = {
    "amount": "Unusual amount", "checkout": "Rushed checkout", "time": "Odd hour", "logins": "Failed logins",
    "is_vpn": "VPN", "is_proxy": "Proxy", "country_mismatch": "Country mismatch",
    "high_risk_country": "High-risk country", "is_emulator": "Emulator", "is_rooted": "Rooted device",
    "timezone_mismatch": "Timezone mismatch", "accounts_per_device": "Shared device",
    "users_per_card": "Shared card", "account_age_days": "New account", "pages_viewed": "Straight to checkout",
}

st.set_page_config(page_title="AI Fraud Detector - QA Console", layout="wide")
st.title("AI Fraud Detector")
st.caption("Operations and quality-assurance console: live risk data, evaluation runs, model comparisons.")


# ============================================================== data loading
@st.cache_data(ttl=30)
def load_scored_orders(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM v_transaction_risk", conn)
    df["reasons"] = df["reasons_json"].map(lambda s: json.loads(s or "[]"))
    df["top_reason"] = df["reasons"].map(lambda r: REASON_LABELS.get(r[0]["feature"], r[0]["feature"]) if r else None)
    return df.drop(columns=["reasons_json", "features_json"])


@st.cache_data
def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_files():
    return sorted(config.RUNS_DIR.glob("*.json"), reverse=True)


def bar_chart(frame, category, title, color=BLUE, domain=(0, 1), fmt=".2f", x_title="Average score"):
    """Horizontal bar chart (kept from the original evaluation dashboard)."""
    chart = (
        alt.Chart(frame)
        .mark_bar(color=color, cornerRadiusEnd=4, size=24)
        .encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=list(domain)), title=x_title),
            y=alt.Y(f"{category}:N", sort="-x", title=None),
            tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip("score:Q", format=fmt)],
        )
        .properties(title=title, height=alt.Step(34))
    )
    labels = chart.mark_text(align="left", dx=4).encode(text=alt.Text("score:Q", format=fmt), color=alt.value(INK_2))
    return (chart + labels).configure_axis(grid=False)


tab_overview, tab_flagged, tab_eval, tab_history, tab_compare = st.tabs(
    ["Fraud overview", "Flagged transactions", "Evaluation", "Run history", "Model comparison"])

orders = load_scored_orders(str(DB_PATH)) if DB_PATH.exists() else None
NO_DB = "No scored orders yet. Run `python 4_Backend/database/connection.py --seed` (or start the API with `--seed`)."

# ============================================================== 1. fraud overview
with tab_overview:
    if orders is None or orders.empty:
        st.info(NO_DB)
    else:
        flagged = orders[orders.action != "approve"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Orders scored", f"{len(orders):,}")
        c2.metric("Sent to review", f"{(orders.action == 'review').sum():,}")
        c3.metric("Blocked", f"{(orders.action == 'block').sum():,}")
        c4.metric("Money at risk", f"${flagged.amount.sum():,.0f}")
        if orders.is_fraud_label.notna().any():
            labelled = orders[orders.is_fraud_label.notna()]
            caught = labelled[(labelled.is_fraud_label == 1) & (labelled.action != "approve")]
            c5.metric("Fraud caught (labelled)", f"{len(caught) / max((labelled.is_fraud_label == 1).sum(), 1):.0%}")

        left, right = st.columns([3, 2])
        hist = (orders.assign(bucket=(orders.risk_score // 5) * 5)
                .groupby("bucket").size().reset_index(name="orders"))
        hist["band"] = pd.cut(hist.bucket, [-1, REVIEW_T - 1, BLOCK_T - 1, 100], labels=BANDS)
        left.altair_chart(
            alt.Chart(hist, title="Risk score distribution (log scale)")
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("bucket:O", title="Risk score (5-point bands)"),
                y=alt.Y("orders:Q", scale=alt.Scale(type="symlog"), title="Orders"),
                color=alt.Color("band:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=BANDS,
                                                range=[GOOD, WARNING, CRITICAL])),
                tooltip=["bucket", "band", "orders"])
            .properties(height=300).configure_axis(grid=False), width="stretch")

        reasons = flagged.top_reason.value_counts().head(8).reset_index()
        reasons.columns = ["reason", "score"]
        if not reasons.empty:
            right.altair_chart(bar_chart(reasons, "reason", "Top reason on flagged orders",
                                         domain=(0, reasons.score.max() * 1.15), fmt="d", x_title="Flagged orders"),
                               width="stretch")

        by_country = (flagged.groupby("ip_country").agg(flagged=("transaction_id", "count"), amount=("amount", "sum"))
                      .sort_values("amount", ascending=False).head(10).reset_index())
        st.subheader("Flagged value by IP country")
        st.dataframe(by_country, hide_index=True, width="stretch",
                     column_config={"amount": st.column_config.NumberColumn("flagged $", format="$%.0f")})

# ============================================================== 2. flagged transactions
with tab_flagged:
    if orders is None or orders.empty:
        st.info(NO_DB)
    else:
        f1, f2, f3 = st.columns([1, 1, 2])
        min_score = f1.slider("Minimum risk score", 0, 100, REVIEW_T)
        status = f2.selectbox("Review status", ["All", "Awaiting review", "Reviewed"])
        reason_filter = f3.multiselect("Top reason", sorted(orders.top_reason.dropna().unique()))
        view = orders[orders.risk_score >= min_score]
        if status == "Awaiting review":
            view = view[view.review_decision.isna()]
        elif status == "Reviewed":
            view = view[view.review_decision.notna()]
        if reason_filter:
            view = view[view.top_reason.isin(reason_filter)]
        view = view.sort_values(["risk_score", "amount"], ascending=False)
        st.caption(f"{len(view):,} orders · ${view.amount.sum():,.0f} total value")
        table = view.assign(
            level=view.risk_level.map({"high": "▲ Block", "medium": "● Review", "low": "✓ Approve"}),
            why=view.reasons.map(lambda rs: " · ".join(f"{r['text']} ({r['impact']}%)" for r in rs)),
            label=view.is_fraud_label.map({1: "fraud", 0: "legit"}),
        )[["transaction_id", "created_at", "user_id", "amount", "risk_score", "level", "why",
           "ip_country", "billing_country", "review_decision", "label"]]
        st.dataframe(table, hide_index=True, width="stretch", height=520, column_config={
            "risk_score": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d"),
            "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
            "why": st.column_config.TextColumn("why it looks risky", width="large"),
            "label": st.column_config.TextColumn("ground truth"),
        })
        st.download_button("Download CSV", table.to_csv(index=False), "flagged_transactions.csv", "text/csv")

# ============================================================== 3. evaluation (original dashboard, extended)
files = run_files()
with tab_eval:
    if not files and not config.LATEST_RESULTS.exists():
        st.info("No results yet. Run `python 3_Model/evaluation/run_tests.py` first.")
    else:
        options = {f.stem: f for f in files} or {"latest": config.LATEST_RESULTS}
        s1, s2, s3, s4 = st.columns([2, 1, 2, 1])
        choice = s1.selectbox("Evaluation run", list(options))
        data = load_run(str(options[choice]))
        meta, results = data["run"], data["results"]
        for r in results:
            r.setdefault("suite", "prompts")
            r.setdefault("category", r["task"])
        suites = sorted({r["suite"] for r in results})
        selected_suites = s2.multiselect("Suites", suites, default=suites)
        tasks = sorted({r["task"] for r in results if r["suite"] in selected_suites})
        selected_tasks = s3.multiselect("Tasks", tasks, default=tasks)
        only_failures = s4.checkbox("Failures only")

        df = pd.DataFrame(results)
        df = df[df.suite.isin(selected_suites) & df.task.isin(selected_tasks)]
        if df.empty:
            st.warning("No tests match the current filters.")
        else:
            fm = meta.get("fraud_model") or {}
            st.caption(f"LLM **{meta.get('provider', '-')}/{meta.get('model', '-')}** · fraud model "
                       f"**{fm.get('version', '-')}** · run at {meta['timestamp']} · pass threshold {meta['pass_threshold']:.2f}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average score", f"{df.final_score.mean():.2f}")
            c2.metric("Pass rate", f"{df.passed.mean():.0%}")
            c3.metric("Tests", len(df))
            c4.metric("Failures", int((~df.passed).sum()))

            left, right = st.columns(2)
            by_task = df.groupby("task")["final_score"].mean().reset_index(name="score")
            left.altair_chart(bar_chart(by_task, "task", "Score by task"), width="stretch")
            criteria = pd.DataFrame(df["scores"].tolist()).mean().dropna().reset_index()
            criteria.columns = ["criterion", "score"]
            right.altair_chart(bar_chart(criteria, "criterion", "Score by criterion"), width="stretch")

            fraud_rows = df[df.task.isin(["risk_scoring", "fraud_analysis"])]
            if not fraud_rows.empty:
                by_cat = fraud_rows.groupby("category")["final_score"].mean().reset_index(name="score")
                st.altair_chart(bar_chart(by_cat, "category", "Fraud cases by category"), width="stretch")

            with st.expander("Rubric"):
                st.table(pd.DataFrame(
                    [{"criterion": k, "weight": v["weight"], "description": v["description"],
                      "applies to": ", ".join(sorted(v["applies_to"])) if v["applies_to"] else "all"}
                     for k, v in RUBRIC.items()]))

            st.subheader("Results")
            crit_cols = [c for c in RUBRIC if df["scores"].map(lambda s, c=c: c in s).any()]
            extra = sorted({k for s in df["scores"] for k in s} - set(RUBRIC))
            table = df.assign(
                status=df.passed.map({True: "✅ Pass", False: "❌ Fail"}),
                **{c: df["scores"].map(lambda s, c=c: s.get(c)) for c in crit_cols + extra},
            )[["id", "suite", "task", "category", "status", "final_score", *crit_cols, *extra, "latency_s"]]
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"final_score": st.column_config.ProgressColumn(
                             "final score", min_value=0, max_value=1, format="%.2f")})

            st.subheader("Responses & details")
            detail = df[~df.passed] if only_failures else df
            for r in detail.to_dict("records"):
                icon = "✅" if r["passed"] else "❌"
                with st.expander(f"{icon} {r['id']} · {r['task']} · score {r['final_score']:.2f}"):
                    st.markdown(f"**Prompt / description:** {r['prompt']}")
                    if r.get("input"):
                        st.markdown("**Input:**")
                        st.code(r["input"], language="json" if str(r["input"]).startswith("{") else None, wrap_lines=True)
                    if r.get("error"):
                        st.error(r["error"])
                    st.markdown("**Output:**")
                    if isinstance(r["output"], (dict, list)):
                        st.json(r["output"])
                    else:
                        st.code(r["output"], language=None, wrap_lines=True)
                    st.markdown("**Scores:**")
                    st.json(r["scores"])
                    if isinstance(r.get("details"), dict):
                        st.markdown("**Details:**")
                        st.json(r["details"], expanded=False)

# ============================================================== 4. run history
history = []
for f in files:
    m = load_run(str(f))["run"]
    fm = m.get("fraud_model") or {}
    history.append({
        "run": f.stem, "timestamp": pd.to_datetime(m["timestamp"]), "llm": f"{m.get('provider', '-')}/{m.get('model', '-')}",
        "fraud_model": fm.get("version", "-"), "suites": ", ".join(m.get("suites", ["prompts"])),
        "tests": m["n_tests"], "avg_score": m["avg_score"], "pass_rate": m["pass_rate"],
        **{f"pass_{k}": v for k, v in (m.get("pass_rate_by_suite") or {"prompts": m["pass_rate"]}).items()},
    })
hist_df = pd.DataFrame(history)

with tab_history:
    if hist_df.empty:
        st.info("No runs recorded yet.")
    else:
        long = hist_df.melt(id_vars=["timestamp", "run"], value_vars=["avg_score", "pass_rate"],
                            var_name="metric", value_name="value")
        long["metric"] = long.metric.map({"avg_score": "Average score", "pass_rate": "Pass rate"})
        base = alt.Chart(long, title="Quality over time").encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["Average score", "Pass rate"], range=[BLUE, ORANGE])),
            tooltip=["run", "metric", alt.Tooltip("value:Q", format=".1%")])
        st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(size=70, filled=True))
                        .properties(height=280).configure_axis(grid=False), width="stretch")
        pass_cols = [c for c in hist_df.columns if c.startswith("pass_") and c != "pass_rate"]
        st.dataframe(hist_df.sort_values("timestamp", ascending=False), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.2f")
                                    for c in ["avg_score", "pass_rate", *pass_cols]})

# ============================================================== 5. model comparison
with tab_compare:
    latest_cmp = next((load_run(str(f))["run"].get("model_comparison") for f in files
                       if load_run(str(f))["run"].get("model_comparison")), None)
    if not latest_cmp:
        st.info("Run `python 3_Model/evaluation/run_tests.py --suite model` to compare models.")
    else:
        cmp = pd.DataFrame(latest_cmp)
        st.subheader("Champion vs baselines (latest model run, held-out orders)")
        metric_names = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision": "Precision @review",
                        "recall": "Recall @review", "false_positive_rate": "False-positive rate @review"}
        long = cmp.melt(id_vars="model", value_vars=list(metric_names), var_name="metric", value_name="value")
        long["metric"] = long.metric.map(metric_names)
        models = list(cmp.model)
        st.altair_chart(
            alt.Chart(long).mark_bar(cornerRadiusEnd=4, size=14)
            .encode(
                y=alt.Y("model:N", title=None, sort=models, axis=None),
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=models, range=[BLUE, ORANGE, AQUA][:len(models)])),
                row=alt.Row("metric:N", title=None, sort=list(metric_names.values()), header=alt.Header(labelAngle=0, labelAlign="left")),
                tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")])
            .properties(height=alt.Step(18), width=520).configure_axis(grid=False).configure_view(stroke=None),
        )
        st.dataframe(cmp, hide_index=True, width="stretch")

        versions = hist_df[hist_df.fraud_model != "-"].drop_duplicates("fraud_model")
        if len(versions) > 1:
            st.subheader("Champion versions over time")
            st.dataframe(versions[["timestamp", "fraud_model", "avg_score", "pass_rate"]], hide_index=True)
        st.caption("Promote a new model only if it beats the current champion here AND passes every gate on the Evaluation tab.")
