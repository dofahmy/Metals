import pandas as pd
from tvDatafeed import TvDatafeed, Interval

path = r"D:\egx\metals_daily_4000\data\GOLD_1D.csv"

old = pd.read_csv(path)
old["date"] = pd.to_datetime(old["session_date"]).dt.strftime("%Y-%m-%d")

tv = TvDatafeed()
new = tv.get_hist(
    symbol="XAUUSD",
    exchange="OANDA",
    interval=Interval.in_daily,
    n_bars=30
).reset_index()

new["date"] = pd.to_datetime(new["datetime"]).dt.strftime("%Y-%m-%d")

compare = old[["date", "close"]].merge(
    new[["date", "close"]],
    on="date",
    suffixes=("_old", "_new")
)

compare["difference_pct"] = (
    (compare["close_new"] / compare["close_old"] - 1) * 100
)

print(compare.tail(15).to_string(index=False))
print("\nLATEST FROM TRADINGVIEW:")
print(new[["date", "open", "high", "low", "close"]].tail(5).to_string(index=False))
