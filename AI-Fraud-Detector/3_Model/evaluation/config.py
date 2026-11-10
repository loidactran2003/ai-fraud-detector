"""Central configuration for the evaluation / QA layer.

Override any value with an environment variable or a run_tests.py CLI flag.
"""
import os
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent          # 3_Model/evaluation
MODEL_ROOT = EVAL_DIR.parent                        # 3_Model
ROOT = MODEL_ROOT.parent                            # project root (AI-Fraud-Detector)

# ----------------------------------------------------------------- LLM prompt evaluation
# "auto" uses OpenAI when OPENAI_API_KEY is set, otherwise the offline mock model.
PROVIDER = os.getenv("EVAL_PROVIDER", "auto")
MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0"))
REQUEST_TIMEOUT = 60  # seconds

SYSTEM_PROMPT = "You are a helpful, accurate, and safe AI assistant."
FRAUD_SYSTEM_PROMPT = (
    "You are a fraud analyst for a small online business. Assess transactions "
    "using network, device, behaviour and transaction-pattern signals. Explain your "
    "reasoning in plain language a shop owner can act on."
)

PROMPT_DIR = EVAL_DIR / "prompt_tests"
PROMPT_FILES = [
    PROMPT_DIR / "summarization.json",
    PROMPT_DIR / "classification.json",
    PROMPT_DIR / "adversarial.json",
    PROMPT_DIR / "fraud_prompts.json",
]

# ----------------------------------------------------------------- fraud model evaluation
FRAUD_TESTS_FILE = EVAL_DIR / "fraud_tests.json"
SAVED_MODELS_DIR = MODEL_ROOT / "saved_models"
FRAUD_MODEL_PATH = SAVED_MODELS_DIR / "fraud_model.pkl"
TEST_SET_PATH = SAVED_MODELS_DIR / "test_set.csv"
FEATURES_PATH = ROOT / "2_Data" / "processed_data" / "feature_engineered.csv"

# Quality gates for whole-model checks (run_tests.py --suite model)
MIN_ROC_AUC = 0.90
MIN_PR_AUC = 0.75
MAX_FALSE_POSITIVE_RATE = 0.03   # share of legit orders sent to review
MAX_FALSE_NEGATIVE_RATE = 0.10   # share of fraud orders that slip through
MIN_SIGNAL_ATTRIBUTION = 0.60    # share of present fraud signals the explanation names
PSI_WARN, PSI_FAIL = 0.10, 0.25  # population stability index thresholds for drift
MAX_ECE = 0.03                   # expected calibration error of the raw probabilities (10 bins)
MAX_FEATURE_IMPORTANCE = 0.35    # no single feature may carry more of the model than this

# ----------------------------------------------------------------- results
RESULTS_DIR = EVAL_DIR / "results"
LATEST_RESULTS = RESULTS_DIR / "results.json"
RUNS_DIR = RESULTS_DIR / "runs"

# A test case "passes" when its weighted score reaches this threshold.
PASS_THRESHOLD = 0.7
