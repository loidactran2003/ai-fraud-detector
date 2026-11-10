"""Generate (and optionally execute) the project notebooks from source.

    python scripts/build_notebooks.py            # write + execute
    python scripts/build_notebooks.py --no-run   # write only

Notebooks are generated so they never drift from the code they demonstrate.
"""
import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "3_Model").exists():
    ROOT = ROOT.parent
for p in ("3_Model", "3_Model/training", "4_Backend/utils"):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
"""

NOTEBOOKS = {
    "3_Model/training/model_training.ipynb": [
        ("md", "# Model training\nTrains the gradient-boosted fraud model on engineered features. "
               "The reproducible one-command version is `python 3_Model/training/train_model.py`; this notebook walks through it."),
        ("code", SETUP),
        ("code", "from preprocessing import FEATURES, FEATURE_LABELS\nfrom train_model import load_dataset, time_split, make_model, evaluate\n"
                 "df = load_dataset()\nprint(f'{len(df):,} orders, fraud rate {df.is_fraud.mean():.1%}, {len(FEATURES)} features')\ndf.head()"),
        ("md", "## Signals: fraud vs legitimate\nMedian / share of each signal by class. No single signal separates the classes, which is why a model is needed."),
        ("code", "df.groupby('is_fraud')[FEATURES].mean().T.rename(columns={0: 'legit', 1: 'fraud'}).round(3)"),
        ("md", "## Time-based split\nTrain on the past, test on the most recent 20% - the way the model is used in production."),
        ("code", "train, test = time_split(df)\nprint(train.timestamp.max(), '->', test.timestamp.min())\nmodel = make_model().fit(train[FEATURES], train.is_fraud)\nevaluate(model, test)"),
        ("code", "imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()\n"
                 "ax = imp.plot.barh(color=BLUE, figsize=(7, 6), title='Feature importance (impurity)')\nax.set_xlabel('importance'); plt.tight_layout()"),
        ("md", "## Save the bundle\n`train()` refits and writes `saved_models/fraud_model.pkl` with the model, feature list, legit baselines (for explanations), thresholds and metrics."),
        ("code", "from train_model import train\nbundle = train()\nbundle['version'], bundle['metrics']['pr_auc']"),
    ],
    "3_Model/training/model_evaluation.ipynb": [
        ("md", "# Model evaluation\nHeld-out performance, threshold trade-offs and a comparison with a rules checklist. "
               "Automated gates for the same numbers live in `3_Model/evaluation` (`run_tests.py --suite model`)."),
        ("code", SETUP),
        ("code", "import json\nfrom sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix\nfrom risk_scoring import RiskScorer\n"
                 "scorer = RiskScorer()\ntest = pd.read_csv(ROOT / '3_Model/saved_models/test_set.csv')\n"
                 "scores = scorer._scores(test)\ny = test.is_fraud.values\njson.dumps({k: scorer.metrics[k] for k in ('roc_auc', 'pr_auc')})"),
        ("code", "fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))\nfpr, tpr, _ = roc_curve(y, scores)\na.plot(fpr, tpr, color=BLUE, lw=2); a.plot([0, 1], [0, 1], ls=':', c='gray')\n"
                 "a.set(title='ROC curve', xlabel='false positive rate', ylabel='true positive rate')\n"
                 "p, r, _ = precision_recall_curve(y, scores)\nb.plot(r, p, color=BLUE, lw=2); b.set(title='Precision-recall', xlabel='recall', ylabel='precision')\nplt.tight_layout()"),
        ("md", "## Threshold trade-off\nWhat each review threshold means for a small shop: how many orders to check, how much fraud is caught."),
        ("code", "rows = []\nfor t in (10, 20, 30, 40, 50, 60, 70, 80, 90):\n    f = scores >= t\n"
                 "    rows.append({'threshold': t, 'orders_flagged': int(f.sum()), 'flag_rate': f.mean(), 'precision': (y[f] == 1).mean() if f.any() else np.nan,\n"
                 "                 'recall': f[y == 1].mean(), 'fraud_$_caught': test.amount[f & (y == 1)].sum()})\npd.DataFrame(rows).round(3)"),
        ("code", "tn, fp, fn, tp = confusion_matrix(y, scores >= scorer.review_threshold).ravel()\npd.DataFrame([[tn, fp], [fn, tp]], index=['actual legit', 'actual fraud'], columns=['approved', 'flagged'])"),
        ("md", "## Versus a hand-written rules checklist"),
        ("code", "sys.path.insert(0, str(ROOT / '3_Model'))\nfrom evaluation.model_checks import rules_score\nfrom sklearn.metrics import average_precision_score\n"
                 "{'model PR-AUC': round(average_precision_score(y, scores), 3), 'rules PR-AUC': round(average_precision_score(y, rules_score(test)), 3)}"),
    ],
    "3_Model/explainability/shap_analysis.ipynb": [
        ("md", "# SHAP analysis\nGlobal and per-order explanations with SHAP (TreeExplainer), cross-checked against the production "
               "reason codes from `RiskScorer`, which use a cheaper what-if method so the API needs no SHAP dependency."),
        ("code", SETUP),
        ("code", "import shap\nfrom risk_scoring import RiskScorer\nscorer = RiskScorer()\ntest = pd.read_csv(ROOT / '3_Model/saved_models/test_set.csv')\n"
                 "X = test[scorer.features].sample(1500, random_state=0)\nexplainer = shap.TreeExplainer(scorer.model)\nsv = explainer.shap_values(X)\nsv.shape"),
        ("md", "## Global: which signals drive risk overall"),
        ("code", "shap.summary_plot(sv, X, plot_type='bar', show=False, color=BLUE)\nplt.tight_layout()"),
        ("code", "shap.summary_plot(sv, X, show=False, max_display=12)\nplt.tight_layout()"),
        ("md", "## Local: one blocked order\nSHAP top contributors vs the reasons shown to the merchant."),
        ("code", "flagged = test[scorer._scores(test) >= scorer.block_threshold].iloc[0:1]\nlocal = pd.Series(explainer.shap_values(flagged[scorer.features])[0], index=scorer.features)\n"
                 "print('SHAP top-5 (log-odds):'); print(local.sort_values(ascending=False).head(5).round(2))\n"
                 "print('\\nRiskScorer reasons:'); [print(' -', r['text'], f\"({r['impact']}%)\") for r in scorer.score(flagged)[0]['reasons']]"),
        ("md", "## Agreement between SHAP and production reasons\nShare of flagged orders where the merchant-facing top reason is among SHAP's top-3 feature groups."),
        ("code", "from risk_scoring import REASON_GROUPS\nfeat2group = {f: g for g, fs in REASON_GROUPS.items() for f in fs}\n"
                 "fl = test[scorer._scores(test) >= scorer.review_threshold].head(200)\nres = scorer.score(fl)\nsv_fl = explainer.shap_values(fl[scorer.features])\nagree = []\n"
                 "for r, row in zip(res, sv_fl):\n    if not r['reasons']: continue\n    g = pd.Series(row, index=scorer.features).groupby(lambda f: feat2group.get(f, f)).sum()\n"
                 "    agree.append(r['reasons'][0]['feature'] in g.nlargest(3).index)\nprint(f'top reason in SHAP top-3 for {np.mean(agree):.0%} of {len(agree)} flagged orders')"),
    ],
    "3_Model/explainability/lime_analysis.ipynb": [
        ("md", "# LIME analysis\nModel-agnostic local explanations with LIME for a few flagged orders, compared with production reason codes."),
        ("code", SETUP),
        ("code", "from lime.lime_tabular import LimeTabularExplainer\nfrom risk_scoring import RiskScorer\nscorer = RiskScorer()\n"
                 "feats = pd.read_csv(ROOT / '2_Data/processed_data/feature_engineered.csv')\ntest = pd.read_csv(ROOT / '3_Model/saved_models/test_set.csv')\n"
                 "explainer = LimeTabularExplainer(feats[scorer.features].sample(5000, random_state=0).values, feature_names=scorer.features,\n"
                 "    class_names=['legit', 'fraud'], discretize_continuous=True, random_state=0)\n"
                 "predict = lambda a: scorer.model.predict_proba(pd.DataFrame(a, columns=scorer.features))"),
        ("code", "flagged = test[scorer._scores(test) >= scorer.block_threshold].head(3)\nfor _, row in flagged.iterrows():\n"
                 "    exp = explainer.explain_instance(row[scorer.features].values.astype(float), predict, num_features=5)\n"
                 "    print(row.transaction_id, 'score', scorer._scores(row.to_frame().T)[0])\n    for rule, w in exp.as_list(): print(f'   LIME  {w:+.3f}  {rule}')\n"
                 "    for r in scorer.score(row.to_frame().T)[0]['reasons']: print('   SHOWN ', r['text'])\n    print()"),
        ("code", "exp.as_pyplot_figure(); plt.tight_layout()"),
    ],
}


def build(run: bool):
    for rel, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if run:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        nbf.write(nb, path)
        print(("executed " if run else "wrote    ") + rel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    build(run=not ap.parse_args().no_run)
