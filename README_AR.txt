Gold-only 1H backtest — 1–30 October 2026 (rolling 30 days to 30 October), thresholds 80,81,82, TP 1%.
Upload the three Python files to the repo root: gold_hourly_backtest.py, fast_both_engine.py, both_hourly_backtest.py.
Both backtest file is required for CSV loading and trade settlement utilities; it is NOT used to generate Both signals.
Run: /app/.venv/bin/python -u gold_hourly_backtest.py
Test: /app/.venv/bin/python -u gold_hourly_backtest.py --max-bars 20
Output: /app/backtest_gold_1h/{GRID_RESULTS.csv,ALL_SIGNALS.csv,ALL_TRADES.csv,RUN_INFO.json}
Original Both files and output are not overwritten. Do not use the old --max-bars 20 results as a full month.
Notes: There is no Silver agreement in this gold-only test. 3 thresholds may repeat the same signal or trade. No spread/fees included. No claim of historical profitability without real Railway results.
