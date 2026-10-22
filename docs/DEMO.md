# Demo walkthrough (about three minutes)

The completed [two-minute captioned walkthrough](demo/wine-quality-demo.mp4) and [timeline](demo/README.md) are included in this repository. It uses real app screenshots with shortened processing waits. The script below supports an additional live review or an uncut recording. The separate offline smoke transcript is engineering evidence, not a recording of a live LLM.

## Before recording

1. Complete the README setup and train the five configurations.
2. Run `python -m src.setup_local` once, then `python -m src.start_local`. Default local mode needs no API key.
3. Run `python -m src.check_live`. This calls the actual local Qwen model and writes `reports/live_check.json` on success.
4. Run `streamlit run src/app.py`.
5. In a separate terminal, run `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.

## 0:00–0:30 — Introduce the problem

“This assistant estimates wine quality from physicochemical tests. The predictive model is trained on UCI's red and white Vinho Verde data. The local Qwen language model extracts measurements and explains the result; the scikit-learn model calculates the score.”

## 0:30–1:20 — Complete prediction

Paste the example from the app's Example question panel. Submit and show:

- The score displayed on a 0–10 scale.
- The conversational explanation and held-out average error.
- The Measurements used panel, demonstrating structured parsing.

Explain that the displayed score comes from the selected trained model. It is an estimate of sensory ratings, not certainty about personal preference.

## 1:20–2:10 — Missing data and clarification

Click **Start a new wine**, then type:

> I have a red wine with 11% alcohol. Rate it.

Show the request for the remaining measurements and the absence of a score. Then supply:

> Fixed acidity 7.4 g/L, volatile acidity 0.70 g/L, citric acid 0 g/L, residual sugar 1.9 g/L, chlorides 0.076 g/L, free sulfur dioxide 11 mg/L, total sulfur dioxide 34 mg/L, density 0.9978 g/cm3, pH 3.51, sulphates 0.56 g/L.

The assistant combines the new measurements with the previously supplied wine type and alcohol value. It now runs the model. Start a new wine when switching samples.

Optional edge cases: “What is Bitcoin worth?” should be rejected as out of scope. “A red wine with acidity 7.4” should prompt clarification about fixed versus volatile acidity.

## 2:10–3:00 — Evidence and limitations

Show the five finished runs in MLflow; failed setup runs may also appear. Sort by `val_rmse`, then show test metrics and the logged model artifact. Show `reports/experiments.csv` and the passing test log. Explain the 60/20/20 split and duplicate grouping, and mention that performance on other wine regions is unverified.

Before submitting, replace this reminder with your recording link or state the date of your scheduled live review in the README. Never record `.env`, keys, or account settings.
