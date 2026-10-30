Trading Similarity — 1D / 1H modes

1) Extract all files to D:\egx\gold_local\daily_rs_dates_test, replacing earlier scripts.
2) py -m pip install tvdatafeed-enhanced websocket-client pandas
3) py .\pattern_matcher.py --open
4) Select 1D (daily) or 1H (hourly) first, then symbol GOLD/SILVER/BOTH.
5) On first hourly run click تحديث البيانات. The 1H CSVs are stored ONLY in data\GOLD_1H.csv and data\SILVER_1H.csv.
6) When new hourly files arrive the dashboard is rebuilt/reloaded; the reference and scanned results use the selected timeframe.

A 5-60 pattern in 1H uses 5-60 HOURLY candles, not 5-60 days. Future outcome label is 2 bars by default, so 2 hours for 1H or 2 daily candles for 1D.
Hourly data returned may be shorter than the full daily history; the scanner cannot test history not supplied by TradingView.
Daily data are kept unchanged by hourly refresh. Both mode requires both selected-timeframe CSVs.
The dashboard refresh button requires the local server (http://127.0.0.1), not a file:// opening.
The signal is a historical scan, not a guarantee. Source timestamps must be verified for OANDA session conventions.
