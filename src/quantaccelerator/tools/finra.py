"""Tools for the FINRA short-sale volume case study: apply and preview an availability rule on trade dates."""
import pandas as pd

from quantaccelerator.ingest.calendar import trading_days
from quantaccelerator.state.schemas import AvailabilityRule
from quantaccelerator.tools.registry import tool


def apply_finra_rule(rule: AvailabilityRule, dates: pd.Series) -> pd.Series:
    """First tradable session for each trade date (a FINRA file is identified by its trade date only)."""
    if rule.source_field != "shvol.Date":
        raise ValueError(f"source_field {rule.source_field!r} is not a field of this dataset (use shvol.Date)")
    days = trading_days()
    d = pd.to_datetime(dates)
    # date-only source: 'next_open_after' (first open after 00:00) is that day's open, like 'same_date_open'
    pos = days.searchsorted(d.values, side="right" if rule.tradable_at == "next_trading_day_open" else "left")
    pos = pos + rule.extra_lag_trading_days
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out


@tool("preview_finra_rule", "Apply a candidate availability rule to a few real trade dates (an ordinary weekday, a "
      "Friday, the day before a market holiday, a half-day session): shows each trade date and the first session "
      "whose open the rule lets a backtest trade on that day's short-sale volume. Use it to sanity-check a rule; it "
      "does not reveal any answer key.",
      {"type": "object", "properties": {"rule": AvailabilityRule.model_json_schema()}, "required": ["rule"]})
def preview_finra_rule(rule: dict):
    r = AvailabilityRule.model_validate(rule)
    days = trading_days()
    sample = pd.Series(pd.to_datetime(["2024-03-12", "2024-03-15", "2024-03-28", "2024-11-29", "2024-07-03"]))
    ft = apply_finra_rule(r, sample)
    nxt = days[days.searchsorted(sample.values, side="right")]
    return {"rule": r.model_dump(), "examples": [
        {"trade_date": f"{d.date()} ({d.day_name()})", "next_calendar_day_is_trading_day": bool((d + pd.Timedelta(days=1)) in days),
         "first_tradable_open": f"{t.date()} 09:30 ET" if pd.notna(t) else None,
         "next_trading_day": str(n.date())} for d, t, n in zip(sample, ft, nxt)]}
