"""Evaluation & QA layer for the AI Fraud Detector.

Modules:
    config        paths, providers, quality gates
    rubric        weighted criteria per task
    scorer        text-response scoring (summaries, labels, safety, fraud analysis)
    fraud_scorer  fraud-specific scoring (risk ranges, actions, signals, explanations)
    model_checks  whole-model checks (FP/FN, AUC, signal attribution, drift, comparison)
    llm_client    OpenAI client + deterministic offline mock
    run_tests     CLI entry point that runs the suites and records run history
"""
