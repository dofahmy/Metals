Gold 1D Dynamic Reference Dashboard

Replace /app/gold_web.py with the enclosed file. Leave your existing gold_fingerprint.py and compare_gold_four_dates.py unchanged. If missing, include enclosed copies.

Run (same target port as your Railway domain):
 /app/.venv/bin/python -u /app/gold_web.py --port 8092

UI: select Peak Reference and Bottom Reference dates, select From and To dates, optionally set Gap From / Gap To and classification filter, then click Display Results.

Requires /app/data/GOLD_1D.csv. Missing reference candle dates produce a clear error.

Scoring: imports the original score() from compare_gold_four_dates.py; weights and calculation are unchanged. Scores are similarity scores, not win probabilities. Comparing to later reference dates may introduce future-information leakage for historical backtest: don't use such retrospective labels as real-time trade signals without chronological testing.

The 'Update data' button reloads the existing CSV, it does not download new prices. Railway interactive-terminal nohup processes may not survive a redeploy.
