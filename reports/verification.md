# Verification record

Verified locally on October 21, 2026, with Python 3.12.14 on Windows.

| Check | Outcome |
|---|---|
| Original UCI red and white CSV download | Passed; file hashes in results.json |
| Unique feature groups / split sizes | 5,320; train 3,192 / validation 1,064 / test 1,064 |
| Completed MLflow configurations | Five: two Ridge, two Random Forest, one Extra Trees |
| MLflow run contents | Each finished run has parameters, dataset/config records, eight metrics, and a model artifact |
| Programmatic selection | mlflow.search_runs selects Extra Trees by validation RMSE |
| Held-out test performance | MAE 0.534379; MSE 0.498222; RMSE 0.705849; R² 0.357771 |
| Mean-only baseline test RMSE | 0.880825 |
| pytest tests/ -v | 26 passed; two dependency deprecation warnings |
| Streamlit UI | Setup message, complete prediction, new-wine reset, and incomplete-input path passed via Streamlit AppTest |
| Streamlit with real Qwen | Passed: actual local LLM returned a 5.00/10 score for the complete example; explicit zero retained; reset and missing-input clarification worked; live-ui-check.json records actual responses |
| Offline smoke demo | Complete query invokes actual saved model; incomplete and out-of-scope cases handled |
| Local LLM runtime | llama.cpp b11669 Vulkan; public Qwen2.5-7B-Instruct Q4_K_M GGUF from Hugging Face; source hashes in local-model.json |
| Live local LLM extraction/explanation | Passed: complete red wine, all 12 features checked, actual regression prediction, generated explanation, and white wine prediction |
| Live local LLM edge cases | Passed: missing inputs, ambiguous acidity, out-of-scope question, and multi-turn completion |
| API keys / provider fees | No API key or provider fee required in default local mode; optional Nebius mode remains available |
| Docker build/run | Pending: Docker CLI present, but Docker engine unavailable |
| Hosted cloud service | Not provisioned |
| Public GitHub repository | Created and source upload in progress: https://github.com/stevensherhouse/wine-quality-assistant |
| Remote CI | Workflow verification in progress |
| Completed presentation video | Two-minute captioned walkthrough from actual app captures in docs/demo/wine-quality-demo.mp4; complete prediction, parsed inputs, missing inputs, follow-up, and scope handling |

The regular provider unit tests use fixed API responses to validate the adapter and orchestration. A separate real-model check, `python -m src.check_live`, passed against the downloaded local Qwen model. Its actual parsed measurements, model predictions, generated explanations, and edge-case outputs are in `reports/live_check.json`. No mock LLM was used for that check. The small live benchmark demonstrates these cases; it does not establish broad extraction reliability. The separate offline transcript remains clearly labeled recorded extraction.

Local model binaries, raw data, MLflow database, and MLflow artifact files are intentionally excluded from Git. The source archive excludes those files as well; a fresh clone or unpacked source archive must train first. The original working folder retains all five completed experiments and the saved model for local inspection.

The test report records the original verification environment. Re-running training may produce a different batch/run ID and slightly different floating-point values on other platforms.
