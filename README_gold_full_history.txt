Gold full history reversal vs continuation research
=================================================
Files: gold_full_history_research.py
Dependencies in /app: gold_fingerprint.py, gold_indicator_study.py, pandas, numpy

Upload only gold_full_history_research.py if the other two files already exist.

Run:
cd /app
/app/.venv/bin/python -u gold_full_history_research.py --csv /app/data/GOLD_1D.csv --out /app/gold_full_history_research

Default: ALL 3987 bars are evaluated. For each candidate close, first future +5% vs -5% inside next 20 daily candles decides continuation vs reversal. Neither or same-bar ambiguous retained in all_candles_and_future_outcomes.csv and not used as binary labels. Incomplete latest 20 bars retained, not labeled.

Each feature uses the PREVIOUS closed bar by default (--lag 1). --lag 0 uses current completed candle. No future OHLC is used in predictors. Models are built with regularized logistic regression (pure numpy), train-only preprocessing and feature selection. TRAIN through 2022-12-31, next 20 days purge, TEST in 2023-2026. No automatic claims of performance or forecast reliability.

Scenarios prior rise >=0,3,5,8% in past 20 bars. --min-rises '0,3,5,8' --move 5 --horizon 20 --lag 1.

Outputs:
all_candles_and_future_outcomes.csv
scenario_summary.csv
models_rise_0p0.csv ... groups + composite ALL scores
predictions_rise_0p0.csv ... per-date predictions with probability-like uncalibrated score and spacing diagnostic.

IMPORTANT: Candidate observations overlap; signal rates are NOT independent trades. Spaced subset mitigates clustering but is not sufficient for rigorous validation. Score isn't a calibrated probability of reversal. Since features and scenarios are now evaluated on the test split, further refinements require a genuinely new holdout or walk-forward study.
