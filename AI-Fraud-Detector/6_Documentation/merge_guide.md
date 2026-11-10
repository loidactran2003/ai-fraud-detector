# Merge guide: `ai-eval` → `AI-Fraud-Detector`

How the standalone **AI Model Evaluation & Prompt Testing System** (`C:\Users\loida\ai-eval`) became the testing and QA layer at `3_Model/evaluation/`.

> **Status:** the files have already been **copied** and adapted. Your original `C:\Users\loida\ai-eval` folder was **not changed**, so it still works as a backup. Use the table below to verify the merge, or to redo it by hand.

## 1. File mapping

| Old location (`ai-eval/`) | New location (`AI-Fraud-Detector/`) | What changed |
|---|---|---|
| `config.py` | `3_Model/evaluation/config.py` | Paths re-rooted. Added the fraud-analyst system prompt, fraud test paths, model quality gates (AUC, FP/FN, PSI) and `fraud_prompts.json` in `PROMPT_FILES`. All original settings and env vars are kept. |
| `llm_client.py` | `3_Model/evaluation/llm_client.py` | `import config` became `from evaluation import config`. `complete()` takes an optional `system=` prompt. The mock has 12 new fraud answers, some with deliberate mistakes. Original canned answers unchanged. |
| `evaluation/rubric.py` | `3_Model/evaluation/rubric.py` | The 4 original criteria are unchanged except `applies_to`, which went from "all" to the explicit text tasks they were designed for. Added 5 fraud criteria and `CRITICAL` (criteria that must pass). |
| `evaluation/scorer.py` | `3_Model/evaluation/scorer.py` | All original functions unchanged. Added fraud-evasion patterns to the safety check, registered the fraud criteria, and added `passes()`. |
| `evaluation/__init__.py` | `3_Model/evaluation/__init__.py` | Module docstring added. |
| `prompts/summarization.json` | `3_Model/evaluation/prompt_tests/summarization.json` | Unchanged |
| `prompts/classification.json` | `3_Model/evaluation/prompt_tests/classification.json` | Unchanged |
| `prompts/adversarial.json` | `3_Model/evaluation/prompt_tests/adversarial.json` | Unchanged |
| `tests/run_tests.py` | `3_Model/evaluation/run_tests.py` | Added `--suite`, `--current`, `--label` and `--strict`. Same run-history format, with extra fields. |
| `tests/test_scorer.py` | `3_Model/evaluation/tests/test_scorer.py` | Unchanged (docstring only). All 13 tests still pass. |
| `tests/__init__.py` | `3_Model/evaluation/tests/__init__.py` | Unchanged |
| `results/runs/*.json` | `3_Model/evaluation/results/runs/` | Copied, so your run history continues in the new dashboard |
| `results/results.json` | *(regenerated)* | The latest run is rewritten by every `run_tests.py` call |
| `dashboard/app.py` | `5_Frontend/streamlit_app/app.py` | Rebuilt as a 5-tab console. The original evaluation views (score by task and by criterion, rubric, results table, response drill-down) are kept in the **Evaluation** tab. |
| `requirements.txt` | merged into root `requirements.txt` | `openai`, `streamlit`, `altair` and `pandas` are kept |
| `README.md` | merged into root `README.md` | |
| `.gitignore` | merged into root `.gitignore` | |

New files (no `ai-eval` equivalent): `fraud_scorer.py`, `model_checks.py`, `fraud_tests.json`, `prompt_tests/fraud_prompts.json`, `tests/test_fraud_scorer.py`.

## 2. Redo the copy by hand (PowerShell)

Only needed if you want to start the merge over. Run these from `C:\Users\loida`:

```powershell
$A = "C:\Users\loida\ai-eval"
$E = "C:\Users\loida\AI-Fraud-Detector\3_Model\evaluation"
New-Item -ItemType Directory -Force "$E\prompt_tests", "$E\tests", "$E\results\runs" | Out-Null

Copy-Item "$A\evaluation\rubric.py", "$A\evaluation\scorer.py", "$A\evaluation\__init__.py" $E
Copy-Item "$A\config.py", "$A\llm_client.py" $E
Copy-Item "$A\prompts\*.json" "$E\prompt_tests\"
Copy-Item "$A\tests\test_scorer.py", "$A\tests\__init__.py" "$E\tests\"
Copy-Item "$A\results\runs\*.json" "$E\results\runs\"
```

Then apply the changes from the table above. The finished versions are already in this repo, so the easiest route is to keep them.

## 3. Import changes you need to know

Folder names like `3_Model` can't be Python package names, so each entry point adds `3_Model/` to `sys.path`. After that, `evaluation` imports as a normal package:

```python
# old (ai-eval root on sys.path)        # new (3_Model on sys.path)
import config                            from evaluation import config
from llm_client import get_client        from evaluation.llm_client import get_client
from evaluation.scorer import ...        from evaluation.scorer import ...   # unchanged
```

## 4. Command changes

| Before (inside `ai-eval/`) | After (inside `AI-Fraud-Detector/`) |
|---|---|
| `python tests/run_tests.py --provider mock` | `python 3_Model/evaluation/run_tests.py --suite prompts --provider mock` |
| `python tests/run_tests.py` | `python 3_Model/evaluation/run_tests.py` (now runs every suite) |
| `streamlit run dashboard/app.py` | `streamlit run 5_Frontend/streamlit_app/app.py` |
| `python -m unittest discover -s tests -t .` | `python -m unittest discover -s 3_Model/evaluation/tests -t 3_Model` |

## 5. Verify

```powershell
cd C:\Users\loida\AI-Fraud-Detector
.venv\Scripts\activate
python -m unittest discover -s 3_Model/evaluation/tests -t 3_Model    # 27 tests, OK
python 3_Model/evaluation/run_tests.py --provider mock                 # 3 suites
streamlit run 5_Frontend/streamlit_app/app.py                          # old runs show under Run history
```

Once you're happy, you can archive or delete `C:\Users\loida\ai-eval`. Nothing in the merged project depends on it.
