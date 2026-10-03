"""Answer key and look-ahead measure for the FINRA short-sale volume case study.

Ground truth comes from FINRA's documentation (data/external_docs/finra_daily_short_sale_files.txt): each daily file is
posted "no later than 6:00:00pm ET of the same day on the relevant trade date", after the 16:00 close. A trade date's
volume can therefore first be traded at the next session's open: shvol.Date, next_trading_day_open.

The researcher pipelines (notebooks/finra_signal_*.py) output one row per (Symbol, date) with asof_date and value = the
5-day average short-volume ratio minus the symbol's own average so far. A row uses non-public information if it is
used before `date` is tradable, or if its value differs from the value computable from data tradable by then (the
clean construction, recomputed here).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from quantaccelerator.ingest.calendar import trading_days
from quantaccelerator.ingest.finra import FINRA_DIR
from quantaccelerator.paths import NOTEBOOKS, ROOT
from quantaccelerator.state.schemas import AvailabilityRule
from quantaccelerator.tools.finra import apply_finra_rule

FINRA_BENCHMARK = ROOT / "data" / "benchmark" / "finra"
GOLD_RULE = AvailabilityRule(source_field="shvol.Date", source_timezone="America/New_York",
                             tradable_at="next_trading_day_open")
PERIOD = ("2024-01-01", "2024-12-31")
VARIANTS = {  # "line" is filled in by build(): the one line where the variant differs from the clean pipeline
    "clean": {"leak": False, "leak_types": []},
    "leak_sameday": {"leak": True, "leak_types": ["event_date_as_availability", "same_bar_execution"],
                     "note": "trades each day's short volume at that day's open, before FINRA publishes it at 18:00"},
    "leak_centered": {"leak": True, "leak_types": ["future_data_in_window"],
                      "note": "a centered rolling window averages in the next two days' ratios"},
    "leak_fullmean": {"leak": True, "leak_types": ["full_sample_normalization"],
                      "note": "measures each symbol's short pressure against its average over the whole of 2024, "
                              "which includes the future"},
}


def reference_signal(universe: list[str] | None = None, sv: pd.DataFrame | None = None) -> pd.DataFrame:
    """The clean construction: (Symbol, date, value) from data up to and including `date`."""
    if sv is None:
        sv = pd.read_parquet(FINRA_DIR / "shvol.parquet", columns=["Date", "Symbol", "ShortVolume", "TotalVolume"])
    days = trading_days()
    tdays = days[(days >= "2023-12-01") & (days <= PERIOD[1])]
    if universe is None:
        v23 = sv[(sv.Date >= "2023-01-01") & (sv.Date < "2024-01-01")].groupby("Symbol").TotalVolume.median()
        universe = list(v23.nlargest(500).index)
    x = sv[sv.Symbol.isin(universe) & (sv.Date >= tdays[0]) & (sv.Date <= PERIOD[1]) & (sv.TotalVolume > 0)].copy()
    x["ratio"] = x.ShortVolume / x.TotalVolume
    grid = pd.MultiIndex.from_product([universe, tdays], names=["Symbol", "Date"])
    g = x.set_index(["Symbol", "Date"]).ratio.reindex(grid).reset_index().sort_values(["Symbol", "Date"])
    g["ratio"] = g.groupby("Symbol").ratio.ffill()
    g["value"] = g.groupby("Symbol").ratio.transform(lambda s: s.rolling(5, min_periods=5).mean())
    g["value"] = g["value"] - g.groupby("Symbol").value.transform(lambda s: s.expanding(min_periods=20).mean())
    return g.rename(columns={"Date": "date"})[["Symbol", "date", "value"]]


def lookahead_stats_finra(events: pd.DataFrame, rule: AvailabilityRule = GOLD_RULE,
                          ref: pd.DataFrame | None = None) -> dict:
    """Look-ahead of a signal table (Symbol, date, asof_date, value) under a reference availability rule."""
    for c in ("Symbol", "date", "asof_date", "value"):
        if c not in events:
            raise ValueError(f"events table lacks column {c!r}; columns: {list(events.columns)}")
    ref = reference_signal() if ref is None else ref
    cal = trading_days()
    ev = events[["Symbol", "date", "asof_date", "value"]].reset_index(drop=True)
    ev = ev.merge(ref.rename(columns={"value": "allowed"}), on=["Symbol", "date"], how="left", indicator=True)
    matched = ev["_merge"].eq("both")  # (Symbol, date) is in the audited universe and period
    ev["first_tradable"] = apply_finra_rule(rule, ev.date)
    early = matched & (ev.asof_date < ev.first_tradable)
    days_early = cal.searchsorted(ev.first_tradable[early].values) - cal.searchsorted(ev.asof_date[early].values)
    # a value the past cannot produce (allowed is NaN) or a different value is not public information at asof
    diff = matched & ~early & ~np.isclose(ev.value, ev.allowed.fillna(np.inf), rtol=0, atol=1e-12)
    bad = (early | diff)[matched]
    q = lambda s, p: float(np.quantile(s, p)) if len(s) else 0.0
    n = max(int(matched.sum()), 1)
    return {"n_events": int(len(ev)), "n_matched": int(matched.sum()),
            "share_before_tradable": round(float(bad.sum() / n), 4),
            "share_used_before_release": round(float(early.sum() / n), 4),
            "share_value_not_public_at_asof": round(float(diff.sum() / n), 4),
            "lookahead_trading_days_early_events": {"median": q(days_early, .5), "p90": q(days_early, .9),
                                                    "max": float(days_early.max()) if len(days_early) else 0.0}}


def rule_agreement_rows(rule: AvailabilityRule) -> pd.DataFrame:
    """Every trading day in the data: does `rule` give the gold first tradable session? Strata by what follows."""
    days = trading_days()
    d = pd.Series(days[(days >= "2018-08-01") & (days < days[-1])])
    nxt = pd.Series(days[days.searchsorted(d.values, side="right")])
    gap = (nxt - d).dt.days
    stratum = np.select([gap == 1, (gap == 3) & (d.dt.dayofweek == 4)], ["next_day", "friday"], "before_holiday")
    try:
        ok = apply_finra_rule(rule, d).values == apply_finra_rule(GOLD_RULE, d).values
    except ValueError:  # a source field of another dataset
        ok = np.zeros(len(d), bool)
    return pd.DataFrame({"date": d, "stratum": stratum, "ok": ok})


def run_notebook(variant: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.parquet")
        subprocess.run([sys.executable, str(NOTEBOOKS / f"finra_signal_{variant}.py")], cwd=ROOT, check=True,
                       env={**os.environ, "QA_OUT": out}, capture_output=True)
        return pd.read_parquet(out)


SEMANTIC = [
    {"id": "f01", "question": "What does shvol.ShortVolume measure?", "target": "field:shvol.ShortVolume",
     "groups": [["short"], ["volume", "shares"], ["trade", "execut", "report", "sale", "sold"]],
     "gold": "aggregate reported share volume of executed short-sale (incl. short-exempt) trades during regular "
             "trading hours, reported to FINRA facilities"},
    {"id": "f02", "question": "What does shvol.ShortExemptVolume measure?", "target": "field:shvol.ShortExemptVolume",
     "groups": [["exempt"], ["short"]],
     "gold": "the part of short volume executed as short-sale-exempt trades"},
    {"id": "f03", "question": "What does shvol.TotalVolume measure?", "target": "field:shvol.TotalVolume",
     "groups": [["all", "total", "aggregate"], ["trf", "adf", "orf", "finra", "facilit", "off-exchange",
                                                "off exchange", "reported to"]],
     "gold": "volume of all trades reported to FINRA's facilities (TRF/ADF/ORF) during regular hours -- not the "
             "total consolidated market volume"},
    {"id": "f04", "question": "What does shvol.Date represent?", "target": "field:shvol.Date",
     "groups": [["trade", "trading"], ["date", "day"]], "gold": "the trade date"},
    {"id": "f05", "question": "What does shvol.Market represent?", "target": "field:shvol.Market",
     "groups": [["facilit", "trf", "report"]], "gold": "the reporting facility code(s)"},
    {"id": "f06", "question": "What does Market code Q mean?", "target": "code:MARKET:Q",
     "groups": [["nasdaq"], ["carteret", "trf"]], "gold": "NASDAQ TRF Carteret"},
    {"id": "f07", "question": "What does Market code N mean?", "target": "code:MARKET:N",
     "groups": [["nyse"]], "gold": "NYSE TRF"},
    {"id": "f08", "question": "What is the observation unit and key of shvol?", "target": "table:shvol",
     "groups": [["symbol", "security", "ticker", "stock"], ["day", "date", "daily"]],
     "key_must_include": ["DATE", "SYMBOL"], "gold": "one security on one trade date; key Date, Symbol"},
    {"id": "f09", "question": "Which field tells when a record became public?", "target": "rule:source_field",
     "gold": "shvol.Date"},
    {"id": "f10", "question": "At which session can a day's volume first be traded?", "target": "rule:tradable_at",
     "gold": "next_trading_day_open"},
]


def build() -> dict:
    FINRA_BENCHMARK.mkdir(parents=True, exist_ok=True)
    ref = reference_signal()
    variants = {}
    clean = (NOTEBOOKS / "finra_signal_clean.py").read_text().splitlines()
    asof_line = next(i for i, ln in enumerate(clean, 1) if ln.startswith('ev["asof_date"] ='))
    for name, spec in VARIANTS.items():
        src = (NOTEBOOKS / f"finra_signal_{name}.py").read_text().splitlines()
        assert len(src) == len(clean), f"{name}: variants must keep the clean pipeline's line numbers"
        diff = [i for i, (a, b) in enumerate(zip(clean, src), 1) if a != b]
        assert len(diff) == (1 if spec["leak"] else 0), (name, diff)
        line = diff[0] if diff else None
        variants[name] = {**spec, "line": line, "file": f"notebooks/finra_signal_{name}.py",
                          "asof_code": src[(line or asof_line) - 1],
                          "true_lookahead": lookahead_stats_finra(run_notebook(name), GOLD_RULE, ref)}
    r = rule_agreement_rows(GOLD_RULE)
    gold = {"gold_rule": GOLD_RULE.model_dump(), "variants": variants, "semantic": SEMANTIC,
            "rule_sample_strata": r.stratum.value_counts().to_dict(),
            "publication": "posted no later than 18:00 ET on the trade date (FINRA daily short sale volume files page)"}
    (FINRA_BENCHMARK / "gold.json").write_text(json.dumps(gold, indent=1, default=str))
    return gold


if __name__ == "__main__":
    g = build()
    for k, x in g["variants"].items():
        print(k, x["line"], json.dumps(x["true_lookahead"]))
    print(g["rule_sample_strata"])
