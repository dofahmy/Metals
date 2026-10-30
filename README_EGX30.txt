PRICE PATTERN MATCHER — EGX30 added

Copy these 3 files next to the existing pattern_matcher.py in:
D:\egx\gold_local\daily_rs_dates_test

Source for EGX30: daily historical OHLC CSV exported from TradingView.
Option A: Name the CSV EGX30_1D.csv and put it beside pattern_matcher.py (or in data/).
Option B: Pass explicit path, example:
py .\pattern_matcher.py --egx30 "D:\egx\indices\EGX30_1D.csv" --open

Required columns (case-insensitive): date (or time/timestamp), open, high, low, close.
Index is loaded independently; no internally synthesized EGXALL benchmark.

Run:
py .\pattern_matcher.py --open
Or:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\RUN_PATTERN_MATCHER.ps1

When EGX30 cannot be found, the script prints EGX30 NOT FOUND.
The dropdown only shows symbols successfully loaded, to avoid pretending data exists.
