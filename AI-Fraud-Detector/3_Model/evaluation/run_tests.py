"""Run the QA suites, score every result, and save the run to history.

Suites
------
  prompts  LLM prompt tests: summarize / classify / adversarial + fraud_analysis
           and fraud-evasion safety prompts (prompt_tests/*.json)
  fraud    golden fraud cases through the real model + RiskScorer (fraud_tests.json)
  model    whole-model checks: AUC gates, false positives/negatives, signal
           detection, scoring logic, drift, champion-vs-baseline comparison
  all      everything (default)

Usage (from the project root):
    python 3_Model/evaluation/run_tests.py                         # all suites, auto provider
    python 3_Model/evaluation/run_tests.py --suite prompts --provider mock
    python 3_Model/evaluation/run_tests.py --suite prompts --provider openai --model gpt-4o-mini
    python 3_Model/evaluation/run_tests.py --suite model --current new_orders_features.csv   # drift vs new data
Exit code is 1 when any test fails and --strict is given (for CI), or when a test
in one of the --strict-suites fails (e.g. --strict-suites fraud,model gates only the
model suites, so the offline mock LLM's known prompt failures don't block a build).
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_Model -> `import evaluation`

from evaluation import config  # noqa: E402
from evaluation.llm_client import get_client  # noqa: E402
from evaluation.scorer import passes, score_response  # noqa: E402

SYSTEM_PROMPTS = {"fraud": config.FRAUD_SYSTEM_PROMPT, None: None}


def load_tests(files):
    tests = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            tests.extend(json.load(f))
    return tests


def run_test(client, test):
    if hasattr(client, "set_test_id"):
        client.set_test_id(test["id"])
    start = time.perf_counter()
    try:
        output = client.complete(test["prompt"], test["input"], system=SYSTEM_PROMPTS.get(test.get("system")))
        error = None
    except Exception as exc:  # one failed API call should not abort the whole run
        output, error = "", f"{type(exc).__name__}: {exc}"
    latency = round(time.perf_counter() - start, 3)

    scores, final_score = score_response(output, test)
    return {
        "id": test["id"],
        "task": test["task"],
        "suite": "prompts",
        "category": test.get("category", test["task"]),
        "prompt": test["prompt"],
        "input": test["input"],
        "output": output,
        "error": error,
        "scores": scores,
        "final_score": final_score,
        "passed": passes(scores, final_score, test),
        "latency_s": latency,
    }


def summarize_run(results):
    by_task, by_suite = {}, {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r["final_score"])
        by_suite.setdefault(r.get("suite", "prompts"), []).append(r["passed"])
    return {
        "n_tests": len(results),
        "avg_score": round(sum(r["final_score"] for r in results) / len(results), 3),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "avg_by_task": {t: round(sum(s) / len(s), 3) for t, s in by_task.items()},
        "pass_rate_by_suite": {s: round(sum(p) / len(p), 3) for s, p in by_suite.items()},
    }


def _status(r):
    return "ERROR" if r.get("error") else ("PASS" if r["passed"] else "FAIL")


def run_prompt_suite(args):
    client = get_client(args.provider, args.model)
    tests = load_tests(args.files)
    print(f"\n[prompts] {len(tests)} tests on {client.name}/{client.model}")
    results = []
    for test in tests:
        r = run_test(client, test)
        print(f"  [{_status(r):5}] {r['id']:<9} {r['task']:<15} score={r['final_score']:.2f}")
        results.append(r)
    return results, {"provider": client.name, "model": client.model}


def run_fraud_suites(args, suites):
    from evaluation.model_checks import FraudModelHarness  # needs the trained model + pandas/sklearn

    harness = FraudModelHarness(current_path=args.current)
    results = []
    if "fraud" in suites:
        spec = json.loads(config.FRAUD_TESTS_FILE.read_text(encoding="utf-8"))
        print(f"\n[fraud] {len(spec['tests'])} golden cases on fraud model {harness.scorer.version}")
        for test in spec["tests"]:
            r = harness.run_case(test, spec.get("defaults", {}))
            out = r["output"] or {}
            print(f"  [{_status(r):5}] {r['id']:<9} {r['category']:<20} risk={out.get('risk_score', '-'):>3} "
                  f"{out.get('action', '-'):<8} score={r['final_score']:.2f}")
            results.append(r)
    if "model" in suites:
        print(f"\n[model] whole-model checks on {len(harness.test)} held-out orders")
        for r in harness.run_all_checks():
            print(f"  [{_status(r):5}] {r['id']:<11} {r['task']:<17} {r['details']['summary']}")
            results.append(r)
    comparison = next((r["details"]["models"] for r in results if r["task"] == "model_comparison"), None)
    meta = {"fraud_model": harness.model_summary()}
    if comparison:
        meta["model_comparison"] = comparison
    return results, meta


def main():
    parser = argparse.ArgumentParser(description="Run the AI Fraud Detector QA suites.")
    parser.add_argument("--suite", choices=["all", "prompts", "fraud", "model"], default="all")
    parser.add_argument("--provider", choices=["auto", "openai", "mock"], default=None)
    parser.add_argument("--model", default=None, help="LLM model name for the prompts suite")
    parser.add_argument("--files", nargs="+", type=Path, default=config.PROMPT_FILES)
    parser.add_argument("--current", type=Path, default=None,
                        help="feature CSV of recent orders to check for drift (default: held-out test set)")
    parser.add_argument("--label", default=None, help="optional name for this run in the history")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any test fails (CI)")
    parser.add_argument("--strict-suites", default="", metavar="SUITES",
                        help="comma-separated suites whose failures exit 1, e.g. fraud,model")
    args = parser.parse_args()

    suites = {"prompts", "fraud", "model"} if args.suite == "all" else {args.suite}
    results, meta = [], {"provider": "-", "model": "-"}
    if "prompts" in suites:
        r, m = run_prompt_suite(args)
        results += r
        meta.update(m)
    if suites & {"fraud", "model"}:
        r, m = run_fraud_suites(args, suites)
        results += r
        meta.update(m)

    timestamp = datetime.now(timezone.utc)
    run = {
        "run": {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "label": args.label,
            "suites": sorted(suites),
            "pass_threshold": config.PASS_THRESHOLD,
            **meta,
            **summarize_run(results),
        },
        "results": results,
    }

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tag = meta.get("fraud_model", {}).get("version") if suites & {"fraud", "model"} else meta["model"]
    if "prompts" in suites and tag != meta["model"]:
        tag = f"{meta['model']}+{tag}"
    safe = re.sub(r"[^\w.+-]", "_", args.label or tag or "run")
    run_file = config.RUNS_DIR / f"{timestamp:%Y%m%d-%H%M%S}_{safe}.json"
    for path in (config.LATEST_RESULTS, run_file):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, default=str)

    s = run["run"]
    print(f"\nAverage score: {s['avg_score']:.2f} | Pass rate: {s['pass_rate']:.0%} | "
          + " | ".join(f"{k} {v:.0%}" for k, v in s["pass_rate_by_suite"].items()))
    print(f"Saved {config.LATEST_RESULTS.relative_to(config.ROOT)} and {run_file.relative_to(config.ROOT)}")
    if args.strict and s["pass_rate"] < 1:
        sys.exit(1)
    gated = {x.strip() for x in args.strict_suites.split(",") if x.strip()}
    failed = [r["id"] for r in results if r.get("suite") in gated and not r.get("passed")]
    if failed:
        print(f"FAILED gate(s) in {', '.join(sorted(gated))}: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
