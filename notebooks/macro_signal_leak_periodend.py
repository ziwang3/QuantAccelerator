# Macro regime features: US macro releases since 2019, one row per (series, observation period)
# downstream signal code z-scores each series' latest change and tilts equity exposure
# asof_date = first session whose OPEN we trade on the observation (backtest enters at the open of asof_date)
import os

import pandas as pd

DATA = "data/interim/alfred"
OUT = os.environ.get("QA_OUT", "macro_features.parquet")

vint = pd.read_parquet(f"{DATA}/vintages.parquet")  # ALFRED: one row per (series, observation date, vintage)
vint = vint[vint.value.notna() & (vint.date >= "2019-01-01")]

# trading calendar from SPY bars
spy = pd.read_csv("Dataset/data/raw/prices_tiingo/SPY.csv", usecols=["date"])
tdays = pd.DatetimeIndex(pd.to_datetime(spy.date)).sort_values()


def roll_fwd(d):
    """trading day on or after each date"""
    return pd.Series(tdays[tdays.searchsorted(pd.to_datetime(d).values)], index=d.index)


def period_end(dates, series):
    """last calendar day of each observation period (GDP is quarterly, the rest monthly)"""
    q = series.eq("GDPC1")
    return (dates + pd.offsets.MonthEnd(0)).where(~q, dates + pd.offsets.QuarterEnd(0))


# first published value of each observation, and today's (latest, revised) value
vint = vint.sort_values(["series_id", "date", "realtime_start"])
first = vint.groupby(["series_id", "date"], as_index=False).first()
latest = vint.groupby(["series_id", "date"], as_index=False).last()
ev = first[["series_id", "date", "realtime_start", "value"]].rename(
    columns={"realtime_start": "release_date", "value": "first_value"})
ev = ev.merge(latest[["series_id", "date", "value"]].rename(columns={"value": "latest_value"}),
              on=["series_id", "date"])
ev["period_end"] = period_end(ev["date"], ev["series_id"])

# value the signal uses
ev["value"] = ev["first_value"]
# as-of date for the backtest
ev["asof_date"] = roll_fwd(ev["period_end"])

ev = ev.sort_values(["asof_date", "series_id"]).reset_index(drop=True)
print(f"{len(ev)} observations, {ev.series_id.nunique()} series, "
      f"asof {ev.asof_date.min().date()} .. {ev.asof_date.max().date()}")
ev[["series_id", "date", "asof_date", "value"]].to_parquet(OUT)
