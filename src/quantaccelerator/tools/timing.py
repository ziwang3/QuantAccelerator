"""Trading-time tools: first-tradable rule (09:30 ET open strictly after the timestamp) and rule application."""
from functools import lru_cache

import numpy as np
import pandas as pd

from quantaccelerator.ingest import calendar as cal
from quantaccelerator.paths import insider_dir
from quantaccelerator.state.schemas import AvailabilityRule
from quantaccelerator.tools.registry import tool

GOLD_RULE = AvailabilityRule(source_field="edgar_acceptance.acceptanceDateTime", source_timezone="UTC",
                             tradable_at="next_open_after")


def parse_to_et(values: pd.Series, source_timezone: str) -> pd.Series:
    """Parse timestamps and return naive America/New_York wall-clock times.

    source_timezone says how the values are *actually* expressed, regardless of any 'Z' label.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        t = values if values.dt.tz is None else values.dt.tz_localize(None)
    else:
        t = pd.to_datetime(values.astype(str).str.replace("Z", "", regex=False), errors="coerce")
    if source_timezone == "UTC":
        t = t.dt.tz_localize("UTC").dt.tz_convert(cal.ET).dt.tz_localize(None)
    return t


def first_open_after(ts_et: pd.Series) -> pd.Series:
    """Date of the first 09:30 ET session open strictly after each naive-ET timestamp."""
    day = ts_et.dt.normalize()
    before_open = (ts_et - day) < cal.OPEN
    same = before_open & day.isin(cal.trading_days())
    return pd.Series(np.where(same, day, cal.next_trading_day_vec(day)), index=ts_et.index).astype("datetime64[ns]")


def shift_trading_days(dates: pd.Series, n: int) -> pd.Series:
    if n == 0:
        return dates
    days = cal.trading_days().values
    pos = np.searchsorted(days, dates.values) + n
    out = np.full(len(pos), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = (pos < len(days)) & dates.notna().values
    out[ok] = days[pos[ok]]
    return pd.Series(out, index=dates.index)


@lru_cache(maxsize=1)
def filing_times() -> pd.DataFrame:
    """One row per accession with every candidate source timestamp (raw)."""
    d = insider_dir()
    sub = pd.read_parquet(d / "submission.parquet",
                          columns=["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE"])
    nt = pd.read_parquet(d / "nonderiv_trans.parquet", columns=["ACCESSION_NUMBER", "TRANS_DATE",
                                                                "DEEMED_EXECUTION_DATE"])
    tmin = nt.groupby("ACCESSION_NUMBER")[["TRANS_DATE", "DEEMED_EXECUTION_DATE"]].min()
    acc = pd.read_parquet(d / "acceptance_index.parquet", columns=["accession", "acceptanceDateTime"])
    f = sub.merge(tmin, left_on="ACCESSION_NUMBER", right_index=True, how="left")
    f = f.merge(acc.rename(columns={"accession": "ACCESSION_NUMBER"}), on="ACCESSION_NUMBER", how="left")
    return f.set_index("ACCESSION_NUMBER")


def apply_rule(rule: AvailabilityRule, frame: pd.DataFrame | None = None) -> pd.Series:
    """First tradable session date per accession under `rule`."""
    f = filing_times() if frame is None else frame
    col = rule.source_field.split(".")[1]
    ts = parse_to_et(f[col], rule.source_timezone)
    if rule.tradable_at == "next_open_after":
        out = first_open_after(ts)
    elif rule.tradable_at == "same_date_open":
        out = cal.roll_forward(ts.dt.normalize())
    else:
        out = cal.next_trading_day_vec(ts.dt.normalize())
    return shift_trading_days(out, rule.extra_lag_trading_days).rename("first_tradable_date")


@tool("next_tradable", "Date of the first 09:30 ET market open strictly after a timestamp. Give the timestamp "
      "and the timezone it is expressed in. Uses the NYSE calendar derived from SPY trading days.",
      {"type": "object", "properties": {"timestamp": {"type": "string"},
                                        "timezone": {"type": "string", "enum": ["UTC", "America/New_York"]}},
       "required": ["timestamp", "timezone"]})
def next_tradable(timestamp: str, timezone: str = "America/New_York"):
    et = parse_to_et(pd.Series([timestamp]), timezone)
    d = first_open_after(et).iloc[0]
    return {"timestamp_et": et.iloc[0], "is_trading_day": cal.is_trading_day(et.iloc[0]),
            "first_tradable_open": f"{d.date()} 09:30 ET"}


@tool("preview_rule", "Apply a candidate availability rule to a small stratified sample of real filings "
      "(pre-open, intraday, after-close, non-trading-day acceptances) and show the source values and the "
      "resulting first tradable date. Use it to sanity-check a rule; it does not reveal any answer key.",
      {"type": "object", "properties": {"rule": AvailabilityRule.model_json_schema(),
                                        "n_per_stratum": {"type": "integer", "minimum": 1, "maximum": 4}},
       "required": ["rule"]})
def preview_rule(rule: dict, n_per_stratum: int = 2):
    r = AvailabilityRule.model_validate(rule)
    f = filing_times().dropna(subset=["acceptanceDateTime"])
    utc = parse_to_et(f.acceptanceDateTime, "UTC")
    mins = utc.dt.hour * 60 + utc.dt.minute
    trading = utc.dt.normalize().isin(cal.trading_days())
    strata = {"pre_open": trading & (mins < 570), "intraday": trading & (mins >= 570) & (mins < 960),
              "after_close": trading & (mins >= 960), "non_trading_day": ~trading}
    rows = []
    for name, m in strata.items():
        s = f[m.values].sample(min(n_per_stratum, int(m.sum())), random_state=0)
        res = apply_rule(r, s)
        for acc, row in s.iterrows():
            rows.append({"accession": acc, "acceptanceDateTime_raw": row.acceptanceDateTime,
                         "FILING_DATE": row.FILING_DATE.date(), "TRANS_DATE_min": row.TRANS_DATE,
                         "first_tradable_date": res[acc]})
    return {"rule": r, "examples": rows}
