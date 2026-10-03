# Short-pressure surprise: 5-day average short-volume ratio relative to the symbol's own average, 500 most traded
# symbols, 2024; downstream signal code ranks symbols by it each day and tilts toward low short pressure
# asof_date = first session whose OPEN we trade on the signal (backtest enters at the open of asof_date)
import os

import pandas as pd

DATA = "data/interim/finra"
OUT = os.environ.get("QA_OUT", "finra_signal.parquet")

sv = pd.read_parquet(f"{DATA}/shvol.parquet", columns=["Date", "Symbol", "ShortVolume", "TotalVolume"])
sv = sv[(sv.Date >= "2023-01-01") & (sv.Date <= "2024-12-31")]

# universe: 500 symbols with the highest median daily volume in 2023 (known before the 2024 signal period)
vol23 = sv[sv.Date < "2024-01-01"].groupby("Symbol").TotalVolume.median()
universe = vol23.nlargest(500).index

# trading calendar from SPY bars
spy = pd.read_csv("Dataset/data/raw/prices_tiingo/SPY.csv", usecols=["date"])
all_days = pd.DatetimeIndex(pd.to_datetime(spy.date)).sort_values()
tdays = all_days[(all_days >= "2023-12-01") & (all_days <= "2024-12-31")]


def next_tday(d):
    """first trading day strictly after each date"""
    return pd.Series(all_days[all_days.searchsorted(pd.to_datetime(d).values, side="right")], index=d.index)


def roll_fwd(d):
    """trading day on or after each date"""
    return pd.Series(all_days[all_days.searchsorted(pd.to_datetime(d).values)], index=d.index)


# daily short-volume ratio on a full symbol x trading-day grid (a symbol missing on a day has no reported volume)
x = sv[sv.Symbol.isin(universe) & (sv.Date >= tdays[0]) & (sv.TotalVolume > 0)].copy()
x["ratio"] = x.ShortVolume / x.TotalVolume
grid = pd.MultiIndex.from_product([universe, tdays], names=["Symbol", "Date"])
ev = x.set_index(["Symbol", "Date"]).ratio.reindex(grid).reset_index()
ev = ev.sort_values(["Symbol", "Date"])

# fill days without data
ev["ratio"] = ev.groupby("Symbol").ratio.ffill()
# 5-day average short-volume ratio
ev["value"] = ev.groupby("Symbol").ratio.transform(lambda s: s.rolling(5, min_periods=5).mean())
# relative to the symbol's own average short pressure
ev["value"] = ev["value"] - ev.groupby("Symbol").value.transform(lambda s: s.expanding(min_periods=20).mean())
ev = ev.dropna(subset=["value"]).rename(columns={"Date": "date"})
ev = ev[ev.date >= "2024-01-01"]
# as-of date for the backtest
ev["asof_date"] = next_tday(ev["date"])

ev = ev.sort_values(["asof_date", "Symbol"]).reset_index(drop=True)
print(f"{len(ev)} rows, {ev.Symbol.nunique()} symbols, asof {ev.asof_date.min().date()} .. {ev.asof_date.max().date()}")
ev[["Symbol", "date", "asof_date", "value"]].to_parquet(OUT)
