"""Answer key and look-ahead measure for the Revision Audit case study (ALFRED macro vintages).

Ground truth comes from the vintages themselves:
  release date of an observation  = realtime_start of its first vintage with a value
  first tradable                  = open of the first trading day on or after the release date (all six series are
                                    released before the 09:30 ET open: BLS/BEA/Census 08:30, Fed G.17 09:15 -- see
                                    data/external_docs)
  value known at time t           = the vintage with realtime_start <= t < realtime_end

A row of a pipeline's output (series_id, date, asof_date, value) uses non-public information if it is used before
its first tradable session, or if its value differs from the value the reference rule allows at asof_date.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from quantaccelerator.ingest.alfred import ALFRED_DIR
from quantaccelerator.ingest.calendar import trading_days
from quantaccelerator.paths import NOTEBOOKS, ROOT
from quantaccelerator.state.schemas import AvailabilityRule

MACRO_BENCHMARK = ROOT / "data" / "benchmark" / "macro"
START = "2019-01-01"  # the SPY trading calendar starts 2018-08-01; earlier observations would roll to its first day
GOLD_RULE = AvailabilityRule(source_field="vintages.realtime_start", tradable_at="same_date_open",
                             value_vintage="as_of_date")
VARIANTS = {
    "clean": {"leak": False, "line": None, "leak_types": []},
    "leak_latest": {"leak": True, "line": 41, "leak_types": ["revised_vs_first_release"],
                    "note": "uses today's revised value instead of the value published at the time"},
    "leak_obsdate": {"leak": True, "line": 43, "leak_types": ["event_date_as_availability"],
                     "note": "keys each observation on the start of the period it describes, weeks before release"},
    "leak_periodend": {"leak": True, "line": 43, "leak_types": ["event_date_as_availability",
                                                                "forward_fill_period_end"],
                       "note": "keys each observation on the end of its period, still before the release"},
}


def vintages() -> pd.DataFrame:
    v = pd.read_parquet(ALFRED_DIR / "vintages.parquet")
    return v[v.value.notna()]


def releases(v: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (series_id, date) since START: release_date, first_value, latest_value, revised."""
    v = vintages() if v is None else v
    v = v[v.date >= START].sort_values("realtime_start")
    g = v.groupby(["series_id", "date"])
    r = pd.DataFrame({"release_date": g.realtime_start.first(), "first_value": g.value.first(),
                      "latest_value": g.value.last()}).reset_index()
    r["revised"] = ~np.isclose(r.first_value, r.latest_value, rtol=0, atol=1e-9)
    return r


def first_tradable(rule: AvailabilityRule, rel: pd.DataFrame) -> pd.Series:
    from quantaccelerator.tools.macro import apply_macro_rule
    return apply_macro_rule(rule, rel)


def lookahead_stats_macro(events: pd.DataFrame, rule: AvailabilityRule = GOLD_RULE,
                          v: pd.DataFrame | None = None) -> dict:
    """Look-ahead of an events table (series_id, date, asof_date, value) under a reference availability rule."""
    for c in ("series_id", "date", "asof_date", "value"):
        if c not in events:
            raise ValueError(f"events table lacks column {c!r}; columns: {list(events.columns)}")
    v = vintages() if v is None else v
    cal = trading_days()
    rel = releases(v)
    rel["first_tradable"] = first_tradable(rule, rel)
    ev = events[["series_id", "date", "asof_date", "value"]].reset_index(drop=True)
    ev = ev.merge(rel, on=["series_id", "date"], how="left")  # left merge keeps the events' row order
    matched = ev.first_tradable.notna()
    early = matched & (ev.asof_date < ev.first_tradable)
    days_early = cal.searchsorted(ev.first_tradable[early].values) - cal.searchsorted(ev.asof_date[early].values)
    # value the reference rule allows at asof_date, for rows used on time
    timely = ev[matched & ~early]
    if rule.value_vintage == "as_of_date":
        k = timely[["series_id", "date", "asof_date"]].reset_index().merge(
            v[["series_id", "date", "realtime_start", "realtime_end", "value"]].rename(columns={"value": "allowed"}),
            on=["series_id", "date"])
        k = k[(k.realtime_start <= k.asof_date) & (k.asof_date < k.realtime_end)].drop_duplicates("index")
        allowed = k.set_index("index").allowed
    else:  # "latest" or "not_applicable": the rule allows today's value
        allowed = timely.latest_value
    mismatch = pd.Series(False, index=ev.index)
    diff = ~np.isclose(ev.value[allowed.index], allowed, rtol=0, atol=1e-9)
    mismatch[allowed.index] = diff
    pct = (100 * (ev.value[allowed.index] - allowed).abs() / allowed.abs())[diff]
    bad = (early | mismatch)[matched]
    q = lambda s, p: float(np.quantile(s, p)) if len(s) else 0.0
    n = max(int(matched.sum()), 1)
    return {"n_events": int(len(ev)), "n_matched": int(matched.sum()),
            "share_before_tradable": round(float(bad.sum() / n), 4),
            "share_used_before_release": round(float(early.sum() / n), 4),
            "share_value_not_published_at_asof": round(float(mismatch.sum() / n), 4),
            "lookahead_trading_days_early_events": {"median": q(days_early, .5), "p90": q(days_early, .9),
                                                    "max": float(days_early.max()) if len(days_early) else 0.0},
            "median_abs_value_difference_pct": round(q(pct, .5), 3)}


def rule_agreement_rows(rule: AvailabilityRule) -> pd.DataFrame:
    """Per observation since START: does `rule` give the gold first tradable session and an allowed value?"""
    rel = releases()
    ok_date = first_tradable(rule, rel).values == first_tradable(GOLD_RULE, rel).values
    ok_value = (rule.value_vintage == "as_of_date") | ~rel.revised.values
    return rel.assign(ok=ok_date & ok_value)


def run_notebook(variant: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.parquet")
        subprocess.run([sys.executable, str(NOTEBOOKS / f"macro_signal_{variant}.py")], cwd=ROOT, check=True,
                       env={**os.environ, "QA_OUT": out}, capture_output=True)
        return pd.read_parquet(out)


SEMANTIC = [
    {"id": "m01", "question": "What does vintages.date represent?", "target": "field:vintages.date",
     "groups": [["observation", "period", "reference", "refers"], ["start", "first day", "begin", "month", "quarter"]],
     "gold": "the start of the period the value describes (observation/reference period), not when it was published"},
    {"id": "m02", "question": "What does vintages.realtime_start mean?", "target": "field:vintages.realtime_start",
     "groups": [["vintage", "publish", "release", "real-time", "realtime", "available", "became", "first"]],
     "gold": "the first date on which this value was the published (current) value -- its vintage date"},
    {"id": "m03", "question": "What does vintages.realtime_end mean?", "target": "field:vintages.realtime_end",
     "groups": [["last", "end", "until", "supersed", "replac", "revis", "current", "valid"]],
     "gold": "the last date on which this value was current, before it was revised or replaced"},
    {"id": "m04", "question": "What is the observation unit and key of vintages?", "target": "table:vintages",
     "groups": [["vintage", "version", "revision", "release", "realtime"]],
     "key_must_include": ["SERIES_ID", "DATE", "REALTIME_START"],
     "gold": "one published value (vintage) of one observation of one series; key series_id, date, realtime_start"},
    {"id": "m05", "question": "Which field tells when a value became public?", "target": "rule:source_field",
     "gold": "vintages.realtime_start"},
    {"id": "m06", "question": "Which vintage may a point-in-time backtest use?", "target": "rule:value_vintage",
     "gold": "as_of_date"},
    {"id": "m07", "question": "What PIT role does vintages.date play?", "target": "pit_role:vintages.date",
     "gold": "reporting_period"},
    {"id": "m08", "question": "What PIT role does vintages.realtime_start play?",
     "target": "pit_role:vintages.realtime_start", "gold": "publication_time"},
]


def build() -> dict:
    MACRO_BENCHMARK.mkdir(parents=True, exist_ok=True)
    v = vintages()
    rel = releases(v)
    rel.assign(first_tradable=first_tradable(GOLD_RULE, rel)).to_parquet(MACRO_BENCHMARK / "releases.parquet",
                                                                           index=False)
    variants = {}
    for name, spec in VARIANTS.items():
        src = (NOTEBOOKS / f"macro_signal_{name}.py").read_text().splitlines()
        line = spec["line"] or 43
        variants[name] = {**spec, "file": f"notebooks/macro_signal_{name}.py", "asof_code": src[line - 1],
                          "true_lookahead": lookahead_stats_macro(run_notebook(name), GOLD_RULE, v)}
    gold = {"gold_rule": GOLD_RULE.model_dump(), "variants": variants, "semantic": SEMANTIC,
            "release_lag_days_median": {s: float((g.release_date - g.date).dt.days.median())
                                        for s, g in rel.groupby("series_id")},
            "share_revised": {s: round(float(g.revised.mean()), 4) for s, g in rel.groupby("series_id")}}
    (MACRO_BENCHMARK / "gold.json").write_text(json.dumps(gold, indent=1, default=str))
    return gold


if __name__ == "__main__":
    g = build()
    for k, x in g["variants"].items():
        print(k, json.dumps(x["true_lookahead"]))
    print(g["release_lag_days_median"], g["share_revised"])
