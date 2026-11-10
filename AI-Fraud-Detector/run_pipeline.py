"""Rebuild everything end to end with one command.

    python run_pipeline.py                  # data -> features -> model -> DB -> QA -> docs
    python run_pipeline.py --notebooks      # also re-execute the notebooks
    python run_pipeline.py --provider openai --skip-docs

Steps fail fast: if any step exits non-zero, the pipeline stops there. The QA step is
a gate: a failing fraud / model check (AUC, error rates, calibration, drift, ...) stops
the pipeline *before* the database is reseeded and the docs are rebuilt. (The prompt
suite is reported but not gated: the offline mock LLM fails a few cases by design.)
Reseeding replaces every stored order; existing reviewer decisions are backed up to
4_Backend/database/reviews_backup_<time>.csv first.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(name, *args):
    print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
    t = time.perf_counter()
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True)
    print(f"--- {name} done in {time.perf_counter() - t:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", type=int, default=20000)
    ap.add_argument("--provider", default="mock", choices=["auto", "mock", "openai"],
                    help="LLM used by the prompt-evaluation suite (default: offline mock)")
    ap.add_argument("--skip-docs", action="store_true")
    ap.add_argument("--notebooks", action="store_true")
    a = ap.parse_args()

    step("1. generate raw data", "2_Data/generate_data.py", "--transactions", a.transactions)
    step("2. clean + feature engineering", "4_Backend/utils/preprocessing.py")
    step("3. train model", "3_Model/training/train_model.py")
    step("4. unit tests", "-m", "unittest", "discover", "-s", "3_Model/evaluation/tests", "-t", "3_Model")
    step("5. QA suites (fraud + model gated)", "3_Model/evaluation/run_tests.py", "--provider", a.provider,
         "--strict-suites", "fraud,model")
    step("6. seed database", "4_Backend/database/connection.py", "--seed", "--force")
    if not a.skip_docs:
        step("7. documentation", "scripts/build_docs.py")
    if a.notebooks:
        step("8. notebooks", "scripts/build_notebooks.py")
    print("\nDone. Next:\n  python 4_Backend/api/fraud_detection_api.py      -> http://127.0.0.1:5000"
          "\n  streamlit run 5_Frontend/streamlit_app/app.py   -> QA console")


if __name__ == "__main__":
    main()
