"""Tools for vintage (real-time) macro data: release lags, revisions, single-observation vintages, rule preview.

They read the `vintages` table of the active profile (ALFRED); every number is computed here, never by the LLM.
"""
import pandas as pd

from quantaccelerator.ingest.calendar import trading_days
from quantaccelerator.state.schemas import AvailabilityRule
from quantaccelerator.tools.data import TABLES, load_table
from quantaccelerator.tools.registry import tool

SERIES = {"type": "string", "description": "series_id, or omit for all series"}


def _vint(series_id: str | None = None) -> pd.DataFrame:
    if "vintages" not in TABLES:
        raise RuntimeError("this dataset has no vintages table")
    v = load_table("vintages")
    v = v[v.value.notna()]
    if series_id:
        if series_id not in set(v.series_id):
            raise ValueError(f"unknown series_id {series_id!r}; available: {sorted(set(v.series_id))}")
        v = v[v.series_id == series_id]
    return v


def _first(v: pd.DataFrame) -> pd.DataFrame:
    return v.sort_values("realtime_start").groupby(["series_id", "date"], as_index=False).first()


@tool("release_lag", "Days from each observation's date (the period it describes) to the realtime_start of its "
      "first published vintage, per series (observations since 2019): quantiles, plus the weekday of first "
      "publication. Use it to see when values became public relative to the period they describe.",
      {"type": "object", "properties": {"series_id": SERIES}}, inputs=lambda series_id=None: [TABLES["vintages"][1]])
def release_lag(series_id: str | None = None):
    f = _first(_vint(series_id))
    f = f[f.date >= "2019-01-01"]
    out = {}
    for sid, g in f.groupby("series_id"):
        lag = (g.realtime_start - g.date).dt.days
        out[sid] = {"n_observations": int(len(g)),
                    "lag_days": {k: float(lag.quantile(q)) for k, q in
                                 (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1))},
                    "first_publication_weekday": g.realtime_start.dt.day_name().value_counts().to_dict()}
    return out


@tool("revision_stats", "How much values change after their first publication, per series (observations since "
      "2019): share of observations whose current value differs from the first published value, median absolute "
      "revision in percent, and the median number of vintages per observation.",
      {"type": "object", "properties": {"series_id": SERIES}}, inputs=lambda series_id=None: [TABLES["vintages"][1]])
def revision_stats(series_id: str | None = None):
    v = _vint(series_id)
    v = v[v.date >= "2019-01-01"].sort_values("realtime_start")
    g = v.groupby(["series_id", "date"])
    t = pd.DataFrame({"first": g.value.first(), "latest": g.value.last(), "n_vintages": g.size()}).reset_index()
    t["revised"] = (t["latest"] - t["first"]).abs() > 1e-9
    t["abs_rev_pct"] = 100 * (t["latest"] - t["first"]).abs() / t["first"].abs()
    return {sid: {"n_observations": int(len(x)), "share_revised_after_first_release": round(float(x.revised.mean()), 4),
                  "median_abs_revision_pct_of_revised": round(float(x.abs_rev_pct[x.revised].median()), 3)
                  if x.revised.any() else 0.0,
                  "median_vintages_per_observation": float(x.n_vintages.median())}
            for sid, x in t.groupby("series_id")}


@tool("vintage_lookup", "All published vintages of one observation: each value with the realtime_start and "
      "realtime_end of the period during which it was the current published value.",
      {"type": "object", "properties": {"series_id": {"type": "string"},
                                        "date": {"type": "string", "description": "observation date, YYYY-MM-DD"}},
       "required": ["series_id", "date"]}, inputs=lambda series_id, date: [TABLES["vintages"][1]])
def vintage_lookup(series_id: str, date: str):
    v = _vint(series_id)
    v = v[v.date == pd.Timestamp(date)].sort_values("realtime_start")
    if v.empty:
        raise ValueError(f"no observation {series_id} {date}")
    rows = [{"realtime_start": str(r.realtime_start.date()),
             "realtime_end": "still current" if r.realtime_end.year > 2200 else str(r.realtime_end.date()),
             "value": r.value} for r in v.itertuples()]
    return {"series_id": series_id, "date": date, "n_vintages": len(rows), "vintages": rows[:40]}


def apply_macro_rule(rule: AvailabilityRule, obs: pd.DataFrame) -> pd.Series:
    """First tradable session per observation (rows of `obs`: series_id, date, release_date)."""
    days = trading_days()
    src = obs["release_date"] if rule.source_field == "vintages.realtime_start" else obs["date"]
    if rule.source_field not in ("vintages.realtime_start", "vintages.date"):
        raise ValueError(f"source_field {rule.source_field!r} is not a field of this dataset")
    pos = days.searchsorted(src.values, side="right" if rule.tradable_at == "next_trading_day_open" else "left")
    # date-only sources: next_open_after (first open after 00:00) is the same session as same_date_open
    pos = pos + rule.extra_lag_trading_days
    out = pd.Series(pd.NaT, index=obs.index, dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out


@tool("preview_macro_rule", "Apply a candidate availability rule to a few real observations per series: shows the "
      "observation date, first publication date, the resulting first tradable session and which value the rule "
      "lets the backtest use. Use it to sanity-check a rule; it does not reveal any answer key.",
      {"type": "object", "properties": {"rule": AvailabilityRule.model_json_schema(),
                                        "n_per_series": {"type": "integer", "minimum": 1, "maximum": 3}},
       "required": ["rule"]}, inputs=lambda rule, n_per_series=1: [TABLES["vintages"][1]])
def preview_macro_rule(rule: dict, n_per_series: int = 1):
    r = AvailabilityRule.model_validate(rule)
    v = _vint()
    f = _first(v[v.date >= "2019-01-01"]).rename(columns={"realtime_start": "release_date", "value": "first_value"})
    latest = v.sort_values("realtime_start").groupby(["series_id", "date"]).value.last()
    s = f.groupby("series_id", group_keys=False).apply(lambda g: g.sample(min(n_per_series, len(g)), random_state=0))
    s["first_tradable"] = apply_macro_rule(r, s)
    rows = []
    for x in s.itertuples():
        lv = latest[(x.series_id, x.date)]
        used = lv if r.value_vintage == "latest" else x.first_value
        rows.append({"series_id": x.series_id, "date": str(x.date.date()), "first_published": str(x.release_date.date()),
                     "first_tradable": str(x.first_tradable.date()) if pd.notna(x.first_tradable) else None,
                     "first_published_value": x.first_value, "current_value": lv, "value_rule_uses": used})
    return {"rule": r, "examples": rows}
