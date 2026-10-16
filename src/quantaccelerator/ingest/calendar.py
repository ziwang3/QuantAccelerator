"""NYSE trading calendar derived from SPY daily bars (Tiingo). Session = 09:30-16:00 ET.

Early-close days are not modelled (only the open matters for the first-tradable rule).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from quantaccelerator.paths import SPY_CSV

ET = "America/New_York"
OPEN = pd.Timedelta(hours=9, minutes=30)
CLOSE = pd.Timedelta(hours=16)


@lru_cache(maxsize=1)
def trading_days() -> pd.DatetimeIndex:
    d = pd.read_csv(SPY_CSV, usecols=["date"]).date
    return pd.DatetimeIndex(pd.to_datetime(d)).normalize().unique().sort_values()


def is_trading_day(day) -> bool:
    return pd.Timestamp(day).normalize() in trading_days()


def next_trading_day(day, inclusive: bool = False) -> pd.Timestamp:
    """First trading day > day (or >= day if inclusive)."""
    days = trading_days()
    d = pd.Timestamp(day).normalize()
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    i = days.searchsorted(d, side="left" if inclusive else "right")
    if i >= len(days):
        raise ValueError(f"{day} is beyond calendar end {days[-1].date()}")
    return days[i]


def roll_forward(dates: pd.Series) -> pd.Series:
    """Vectorised: each date -> itself if a trading day, else the next trading day."""
    days = trading_days().values
    idx = np.searchsorted(days, pd.to_datetime(dates).dt.normalize().values, side="left")
    out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(days)
    out[ok] = days[idx[ok]]
    return pd.Series(out, index=dates.index)


def next_trading_day_vec(dates: pd.Series) -> pd.Series:
    """Vectorised: each date -> first trading day strictly after it."""
    days = trading_days().values
    idx = np.searchsorted(days, pd.to_datetime(dates).dt.normalize().values, side="right")
    out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(days)
    out[ok] = days[idx[ok]]
    return pd.Series(out, index=dates.index)
