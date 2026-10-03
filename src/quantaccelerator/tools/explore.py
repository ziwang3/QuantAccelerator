"""Pre-PIT data research tools: coverage, missingness, distributions, update dynamics, variance split, structural
breaks, key checks, research breadth and category mix, for any table of the active dataset profile.

Every tool returns a compact numeric summary for the agent (the LLM reads numbers, not pictures) and draws one
figure for the human under <run_dir>/figures/<run_id>.{png,svg}; `figure` in the result is its path. The summary
always contains the numbers the figure shows, so any claim about a figure can be checked against them.
The statistics live in plain functions (`*_stats`) that take a DataFrame, so they can be tested on synthetic data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from quantaccelerator.tools.data import load_table, table_path
from quantaccelerator.tools.registry import tool
from quantaccelerator.viz import figures as F

FREQS = ["D", "W", "M", "Q", "Y"]
MAX_LIST = 8  # longest list of items (dates, codes, examples) returned to the agent


def _frame(table: str, cols: list[str]) -> pd.DataFrame:
    df = load_table(table)
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise KeyError(f"{table} has no column(s) {missing}; columns: {list(df.columns)}")
    return df[[c for c in dict.fromkeys(cols) if c]]


def _dt(s: pd.Series) -> pd.Series:
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")


def _period(t: pd.Series, freq: str) -> pd.Series:
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {FREQS}")
    t = _dt(t)
    return t.dt.normalize() if freq == "D" else t.dt.to_period(freq).dt.start_time


def flag(fid: str, fact: str, *terms: list[str], remedy: dict | None = None) -> dict:
    """A material fact the agent must address in its findings: every term group needs one of its words in the text.
    A data error also names its remedy (a cleaning step the agent should propose, or explain why not)."""
    return {"id": fid, "fact": fact, "terms": [list(t) for t in terms], **({"remedy": remedy} if remedy else {})}


def month_terms(months: list[str]) -> list[str]:
    """'2024-08' plus the ways people write it: 'august 2024', 'aug 2024'."""
    out = []
    for m in months:
        t = pd.Timestamp(f"{m}-01")
        out += [m, f"{t:%B %Y}".lower(), f"{t:%b %Y}".lower(), f"{t:%b. %Y}".lower()]
    return out


def _r(x, nd: int = 4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _d(x) -> str:
    return str(pd.Timestamp(x).date())


def level_shifts(s: pd.Series, window: int, k: int = 3, min_score: float = 4.0) -> list[dict]:
    """Candidate structural breaks in a series: points where the median of the next `window` values departs from the
    median of the previous `window` by more than `min_score` robust standard deviations (pooled MAD)."""
    v = s.to_numpy(float)
    n, cands = len(v), []
    for i in range(window, n - window + 1):
        b, a = v[i - window:i], v[i:i + window]
        b, a = b[~np.isnan(b)], a[~np.isnan(a)]
        if len(b) < window // 2 or len(a) < window // 2:
            continue
        mb, ma = np.median(b), np.median(a)
        mad = np.median(np.abs(np.concatenate([b - mb, a - ma]))) * 1.4826
        scale = max(mad, 1e-9 + 0.01 * abs(mb))
        cands.append((abs(ma - mb) / scale, i, mb, ma))
    out, taken = [], []
    for score, i, mb, ma in sorted(cands, reverse=True):
        if score < min_score or len(out) >= k:
            break
        if any(abs(i - j) < window for j in taken):
            continue
        taken.append(i)
        i = _refine(v, i, window)
        out.append({"date": _d(s.index[i]), "before_median": _r(mb), "after_median": _r(ma),
                    "change_pct": _r(100 * (ma - mb) / abs(mb), 1) if mb else None, "score": _r(score, 1)})
    return sorted(out, key=lambda x: x["date"])


def _refine(v: np.ndarray, i: int, window: int) -> int:
    """Exact break position near i: the split of v[i-window : i+window] that best separates two means (least
    squares), so noise between neighbouring candidates does not shift the reported date."""
    lo, hi = max(0, i - window), min(len(v), i + window)
    seg = v[lo:hi]
    ok = ~np.isnan(seg)
    best, best_j = -1.0, i
    for j in range(max(lo + 1, i - window // 2), min(hi - 1, i + window // 2) + 1):
        a, b = seg[:j - lo][ok[:j - lo]], seg[j - lo:][ok[j - lo:]]
        if len(a) and len(b):
            gain = len(a) * len(b) / (len(a) + len(b)) * (a.mean() - b.mean()) ** 2
            if gain > best:
                best, best_j = gain, j
    return best_j


# ---------------------------------------------------------------- coverage
def coverage_stats(df: pd.DataFrame, time: str, entity: str | None, freq: str) -> tuple[dict, pd.DataFrame]:
    d = df.assign(_p=_period(df[time], freq)).dropna(subset=["_p"])
    g = d.groupby("_p")
    per = pd.DataFrame({"rows": g.size()})
    out = {"n_rows": int(len(df)), "n_periods": int(len(per)), "first_period": _d(per.index.min()),
           "last_period": _d(per.index.max()), "rows_per_period": _quant(per.rows)}
    if entity:
        per["entities"] = g[entity].nunique()
        first = d.groupby(entity)._p.min().value_counts()
        last = d.groupby(entity)._p.max().value_counts()
        per["new"] = first.reindex(per.index).fillna(0).astype(int)
        per["lost"] = last.reindex(per.index).fillna(0).astype(int)
        e = per.entities
        out.update({"n_entities": int(d[entity].nunique()), "entities_per_period": _quant(e),
                    "entities_min_at": _d(e.idxmin()), "entities_max_at": _d(e.idxmax()),
                    "coverage_cv": _r(e.std() / e.mean()),
                    "median_new_per_period": _r(per.new.iloc[1:].median(), 1),
                    "median_lost_per_period": _r(per.lost.iloc[:-1].median(), 1)})
    series = per.entities if entity else per.rows
    w = max(3, min(60, len(series) // 10))
    out["level_shifts"] = level_shifts(series, w)
    # short-lived holes: periods far below the typical level around them (a shift must persist; a dip need not)
    around = series.rolling(2 * w + 1, center=True, min_periods=3).median()
    dips = series[series < 0.8 * around]
    out["dips"] = [{"period": _d(k), "value": _r(v), "typical": _r(around[k])} for k, v in dips.head(MAX_LIST).items()]
    if len(dips):
        out["flags"] = [flag(f"coverage_dip_{_d(k)[:7]}", f"coverage in the period starting {_d(k)} is "
                             f"{100 * (1 - v / around[k]):.0f}% below its typical level",
                             month_terms([_d(k)[:7]]), ["coverage", "fewer", "drop", "dip", "hole", "gap", "missing",
                                                        "absent", "lower", "decline"]) for k, v in dips.head(3).items()]
    return out, per


def _quant(s: pd.Series) -> dict:
    s = s.dropna()
    return {"min": _r(s.min()), "median": _r(s.median()), "max": _r(s.max())} if len(s) else {}


@tool("coverage_over_time", "Coverage of a table over time: rows and distinct entities per period, entities "
      "entering and leaving, the least and most covered periods, and candidate level shifts in coverage. Draws the "
      "coverage chart. Use it to see whether the universe is stable, growing, or changes abruptly.",
      {"type": "object", "properties": {
          "table": {"type": "string"}, "time_field": {"type": "string"},
          "entity_field": {"type": "string", "description": "identifier column, e.g. a symbol (optional)"},
          "freq": {"type": "string", "enum": FREQS, "description": "period length (default M)"}},
       "required": ["table", "time_field"]},
      inputs=lambda table, **_: [table_path(table)])
def coverage_over_time(table: str, time_field: str, entity_field: str | None = None, freq: str = "M"):
    out, per = coverage_stats(_frame(table, [time_field, entity_field]), time_field, entity_field, freq)
    fig, ax = F.new(2 if entity_field else 1, sharex=True)
    y = per.entities if entity_field else per.rows
    ax[0].plot(per.index, y, color=F.BLUE)
    ax[0].set_title(f"{'Distinct ' + entity_field if entity_field else 'Rows'} per period ({freq})")
    for s in out["level_shifts"]:
        F.mark(ax[0], pd.Timestamp(s["date"]), s["date"])
    if entity_field:
        ax[1].bar(per.index, per.new, color=F.GREEN, label="new", width=_width(freq))
        ax[1].bar(per.index, -per.lost, color=F.ORANGE, label="lost", width=_width(freq))
        ax[1].set_title("Entities entering (up) and leaving (down)")
        ax[1].legend(loc="upper left")
    for a in ax:
        F.date_axis(a)
    out["figure"] = F.save(fig, f"Coverage of {table}")
    return out


def _width(freq: str) -> float:
    return {"D": 1, "W": 5, "M": 25, "Q": 70, "Y": 300}[freq]


# ---------------------------------------------------------------- missingness
def missingness_stats(df: pd.DataFrame, fields: list[str], time: str | None, entity: str | None,
                      freq: str) -> tuple[dict, pd.DataFrame | None, pd.Series | None]:
    out = {"n_rows": int(len(df)), "fields": {}}
    for f in fields:
        s = df[f]
        rec = {"null_share": _r(s.isna().mean())}
        rec["null_count"] = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            rec["zero_share"] = _r((s == 0).mean())
            rec["zero_count"] = int((s == 0).sum())
        elif s.dtype == object:
            rec["empty_string_share"] = _r((s.astype(str).str.strip() == "").mean())
        out["fields"][f] = rec
    over = None
    if time:
        p = _period(df[time], freq)
        over = df[fields].isna().groupby(p).mean()
        worst = over.max().sort_values(ascending=False)
        out["null_share_by_period"] = {f: {"max": _r(over[f].max()), "max_at": _d(over[f].idxmax())}
                                       for f in worst.index[:MAX_LIST] if over[f].max() > 0}
        num = [f for f in fields if pd.api.types.is_numeric_dtype(df[f])]
        if num:
            zeros = (df[num] == 0).groupby(p).mean()
            out["zero_share_by_period"] = {f: {"median": _r(zeros[f].median()), "max": _r(zeros[f].max()),
                                               "max_at": _d(zeros[f].idxmax())} for f in num if zeros[f].max() > 0}
    presence = None
    if time and entity:
        # absent entity-days: calendar = the table's own distinct dates; absence inside an entity's first..last span
        t = _dt(df[time]).dt.normalize()
        cal = pd.Index(np.sort(t.dropna().unique()))
        pos = pd.Series(cal.get_indexer(t), index=df.index)
        g = pos.groupby(df[entity])
        span = g.max() - g.min() + 1
        nobs = g.nunique()
        presence = (nobs / span).rename("presence")
        interior = (span - nobs).sum()
        out["panel"] = {"calendar_dates": int(len(cal)), "entities": int(len(span)),
                        "absent_inside_span_share": _r(interior / span.sum()),
                        "entities_always_present_share": _r((nobs == span).mean()),
                        "entities_present_whole_calendar_share": _r((span == len(cal)).mean()),
                        "median_presence": _r(presence.median()),
                        "note": "absent = the entity has no row on a calendar date between its first and last row"}
    out["reading"] = _missing_reading(out)
    flags = []
    if (p := out.get("panel")) and p["absent_inside_span_share"] >= 0.02:
        flags.append(flag("absent_rows", f"entities have no row on {100 * p['absent_inside_span_share']:.1f}% of dates "
                          "inside their own history", ["absent", "no row", "missing row", "gap", "not present",
                                                       "do not appear", "not appear", "missing on"]))
    for k, v in out["fields"].items():
        if (v.get("zero_share") or 0) >= 0.2:
            flags.append(flag(f"zeros_{k}", f"{k} is zero in {100 * v['zero_share']:.0f}% of rows", [k.lower()],
                              ["zero"]))
    for k, z in (out.get("zero_share_by_period") or {}).items():
        if z["max"] >= max(0.05, 3 * (z["median"] or 0)):
            flags.append(flag(f"zero_spike_{k}", f"{k} is zero in {100 * z['max']:.0f}% of rows in the period starting "
                              f"{z['max_at']} (typical {100 * (z['median'] or 0):.1f}%)", [k.lower()], ["zero"],
                              [z["max_at"][:7], z["max_at"][:4]]))
    if flags:
        out["flags"] = flags
    return out, over, presence


def _missing_reading(out: dict) -> str:
    f = out["fields"]
    nulls = [k for k, v in f.items() if v["null_share"]]
    zeros = [f"{k} ({100 * v['zero_share']:.1f}%)" for k, v in f.items() if (v.get("zero_share") or 0) >= 0.01]
    s = [f"nulls in {', '.join(nulls)}" if nulls else "no field has null values"]
    if zeros:
        s.append("zeros are common in " + ", ".join(zeros) + " (check whether zero means none or not reported)")
    if (p := out.get("panel")) and p["absent_inside_span_share"]:
        s.append(f"entities have no row on {100 * p['absent_inside_span_share']:.1f}% of dates inside their own history"
                 + (": absent rows, not nulls, are this table's missing data" if not nulls else ""))
    return "; ".join(s) + "."


@tool("missingness", "Missing data in a table, keeping apart three things researchers often conflate: null values, "
      "zeros, and entities that are simply absent on some dates (no row at all). Per field: null and zero shares, "
      "and null share over time; for panels, how often entities are absent inside their own history. Draws the "
      "missingness chart.",
      {"type": "object", "properties": {
          "table": {"type": "string"}, "fields": {"type": "array", "items": {"type": "string"},
                                                  "description": "columns to check (default: all)"},
          "time_field": {"type": "string"}, "entity_field": {"type": "string"},
          "freq": {"type": "string", "enum": FREQS}},
       "required": ["table"]},
      inputs=lambda table, **_: [table_path(table)])
def missingness(table: str, fields: list[str] | None = None, time_field: str | None = None,
                entity_field: str | None = None, freq: str = "M"):
    fields = fields or [c for c in load_table(table).columns]
    df = _frame(table, [*fields, time_field, entity_field])
    out, over, presence = missingness_stats(df, fields, time_field, entity_field, freq)
    fig, ax = F.new(1, 2 if presence is not None else 1, width=8.5, height=2.8)
    names = list(out["fields"])
    ax[0].barh(names, [out["fields"][f]["null_share"] for f in names], color=F.BLUE, label="null")
    ax[0].barh(names, [out["fields"][f].get("zero_share") or 0 for f in names],
               left=[out["fields"][f]["null_share"] for f in names], color=F.AMBER, label="zero")
    ax[0].set_title("Share of rows null / zero")
    ax[0].legend(loc="lower right")
    if presence is not None:
        ax[1].hist(presence, bins=40, color=F.GREEN)
        ax[1].set_title(f"{entity_field}: share of own history present")
    out["figure"] = F.save(fig, f"Missingness in {table}")
    return out


# ---------------------------------------------------------------- distributions
def distribution_stats(x: pd.Series, by: pd.Series | None = None) -> dict:
    n_null = int(x.isna().sum())
    x = x.dropna().astype(float)
    if not len(x):
        return {"n": 0, "n_null": n_null}
    q = x.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
    iqr = q[.75] - q[.25]
    out = {"n": int(len(x)), "n_null": n_null, "percentiles": {f"p{int(k * 100)}": _r(v, 6) for k, v in q.items()},
           "mean": _r(x.mean(), 6), "std": _r(x.std(), 6), "skew": _r(x.skew(), 3), "kurtosis": _r(x.kurt(), 3),
           "zero_share": _r((x == 0).mean()), "zero_count": int((x == 0).sum()), "negative_share": _r((x < 0).mean()),
           "share_at_max": _r((x == q[1]).mean()), "share_at_min": _r((x == q[0]).mean()),
           "integer_valued": bool(np.all(np.mod(x.head(100000), 1) == 0)),
           "far_outlier_share": _r(((x > q[.75] + 3 * iqr) | (x < q[.25] - 3 * iqr)).mean()) if iqr > 0 else 0.0}
    hints = []
    if out["negative_share"] == 0 and (out["skew"] or 0) > 2:
        hints.append("strongly right-skewed and non-negative: consider log1p or ranks")
    if out["zero_share"] and out["zero_share"] > 0.05:
        hints.append("large mass at zero: check whether zero means 'none' or 'not reported'")
    if (out["share_at_max"] or 0) > 0.01 and q[1] != q[0]:
        hints.append(f"{out['share_at_max']:.1%} of values sit exactly at the maximum {q[1]:g}: a bound or cap; "
                     "check what produces it (e.g. tiny denominators)")
    if (out["far_outlier_share"] or 0) > 0.01:
        hints.append("heavy tails: consider winsorizing or ranks; check units of the extremes")
    out["hints"] = hints
    if (out["share_at_max"] or 0) >= 0.01 and q[1] != q[0]:
        out["flags"] = [flag("mass_at_max", f"{100 * out['share_at_max']:.1f}% of values equal the maximum {q[1]:g}",
                             ["max", "cap", "bound", "exactly", "ceiling", "equal to 1", "= 1", "of 1.0"])]
    if by is not None:
        g = x.groupby(by.reindex(x.index))
        tab = pd.DataFrame({"p10": g.quantile(.1), "median": g.median(), "p90": g.quantile(.9), "n": g.size()})
        out["by_period"] = {_d(k): {c: _r(v, 6) for c, v in r.items()} for k, r in tab.iterrows()}
        if len(out["by_period"]) > 12:  # keep the agent's view short; the figure shows every period
            keys = list(out["by_period"])
            out["by_period"] = {k: out["by_period"][k] for k in keys[:4] + keys[-4:]}
            out["by_period_note"] = f"{len(keys)} periods; first and last 4 shown"
    return out


@tool("distribution", "Distribution of a numeric field (optionally the ratio of two fields): percentiles, mean, "
      "skew, tails, zero and negative mass, outliers, whether values are integers, and processing hints. Optionally "
      "split by period to see whether the distribution drifts. Draws a histogram, an ECDF and the per-period bands.",
      {"type": "object", "properties": {
          "table": {"type": "string"}, "field": {"type": "string"},
          "denominator": {"type": "string", "description": "optional: analyse field / denominator (rows with a zero "
                                                            "denominator are dropped)"},
          "time_field": {"type": "string"}, "freq": {"type": "string", "enum": FREQS}},
       "required": ["table", "field"]},
      inputs=lambda table, **_: [table_path(table)])
def distribution(table: str, field: str, denominator: str | None = None, time_field: str | None = None,
                 freq: str = "Y"):
    df = _frame(table, [field, denominator, time_field])
    x = df[field].astype(float)
    label = field
    if denominator:
        den = df[denominator].astype(float)
        x = (x / den.where(den != 0))
        label = f"{field} / {denominator}"
    by = _period(df[time_field], freq) if time_field else None
    out = {"field": label, **distribution_stats(x, by)}
    if denominator:
        over1 = int((x > 1).sum())
        out["rows_numerator_exceeds_denominator"] = over1
        if over1:
            out.setdefault("flags", []).append(flag(
                "part_exceeds_whole", f"{field} exceeds {denominator} in {over1:,} rows", [field.lower()],
                ["exceed", "greater than", "larger than", "above", "more than", "> "],
                remedy={"op": ["flag_rows", "drop_rows"], "where": {"column": field, "op": ">col", "value": denominator}}))
    xs = x.dropna()
    if len(xs) > 500_000:
        xs = xs.sample(500_000, random_state=0)
    fig, ax = F.new(1, 3 if by is not None else 2, width=10, height=2.8)
    logx = out.get("negative_share") == 0 and (out.get("skew") or 0) > 2
    data = np.log10(xs[xs > 0]) if logx else xs
    ax[0].hist(data, bins=60, color=F.BLUE)
    ax[0].set_title(f"Histogram{' (log10, >0)' if logx else ''}")
    srt = np.sort(data.to_numpy())
    ax[1].plot(srt, np.linspace(0, 1, len(srt)), color=F.BLUE)
    ax[1].set_title("ECDF")
    if by is not None:
        g = x.groupby(by.reindex(x.index))
        q = pd.DataFrame({"p10": g.quantile(.1), "p50": g.median(), "p90": g.quantile(.9)})
        ax[2].fill_between(q.index, q.p10, q.p90, color=F.BLUE, alpha=.2, label="p10–p90")
        ax[2].plot(q.index, q.p50, color=F.BLUE, label="median")
        ax[2].set_title(f"By period ({freq})")
        ax[2].legend(loc="upper left")
        F.date_axis(ax[2])
    out["figure"] = F.save(fig, f"Distribution of {label}")
    return out


# ---------------------------------------------------------------- update dynamics
def update_stats(df: pd.DataFrame, entity: str, time: str, field: str, lags=(1, 5, 20)) -> dict:
    d = df.dropna(subset=[field]).assign(_t=_dt(df[time])).sort_values([entity, "_t"])
    g = d.groupby(entity, sort=False)
    prev = g[field].shift(1)
    has_prev = prev.notna()
    unchanged = (d[field] == prev)[has_prev]
    gaps = g._t.diff().dt.days.dropna()
    x = d[field].astype(float)
    acf = {}
    for k in lags:
        lagged = g[field].shift(k).astype(float)
        ok = lagged.notna()
        acf[f"lag{k}"] = _r(np.corrcoef(x[ok], lagged[ok])[0, 1], 3) if ok.sum() > 10 else None
    changes = d.loc[has_prev & (d[field] != prev)]
    upd_gap = changes.groupby(entity)._t.diff().dt.days.dropna()
    return {"n_entities": int(d[entity].nunique()), "n_obs": int(len(d)),
            "share_unchanged_vs_previous_obs": _r(unchanged.mean()),
            "days_between_obs": _quant(gaps), "days_between_value_changes": _quant(upd_gap),
            "autocorrelation": acf, "_gaps": gaps, "_acf": acf}


@tool("update_dynamics", "How a field evolves within each entity over time: how often it is unchanged from the "
      "previous observation, days between observations and between value changes, and its autocorrelation at "
      "several lags (persistence). A persistent, slow-moving field may be better studied in changes than levels. "
      "Uses up to 500 randomly chosen entities. Draws the autocorrelation and interval charts.",
      {"type": "object", "properties": {
          "table": {"type": "string"}, "entity_field": {"type": "string"}, "time_field": {"type": "string"},
          "field": {"type": "string"}, "denominator": {"type": "string"}},
       "required": ["table", "entity_field", "time_field", "field"]},
      inputs=lambda table, **_: [table_path(table)])
def update_dynamics(table: str, entity_field: str, time_field: str, field: str, denominator: str | None = None):
    df = _frame(table, [entity_field, time_field, field, denominator])
    ents = pd.Series(df[entity_field].unique())
    if len(ents) > 500:
        df = df[df[entity_field].isin(set(ents.sample(500, random_state=0)))]
    if denominator:
        df = df.assign(**{field: df[field] / df[denominator].where(df[denominator] != 0)})
    s = update_stats(df, entity_field, time_field, field)
    gaps, acf = s.pop("_gaps"), s.pop("_acf")
    s["sampled_entities"] = int(min(len(ents), 500))
    fig, ax = F.new(1, 2, width=8.5, height=2.6)
    ax[0].bar(list(acf), [v or 0 for v in acf.values()], color=F.BLUE)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title("Autocorrelation within entity")
    ax[1].hist(gaps.clip(upper=gaps.quantile(.99)), bins=40, color=F.GREEN)
    ax[1].set_title("Days between observations")
    s["field"] = f"{field}/{denominator}" if denominator else field
    s["figure"] = F.save(fig, f"Update dynamics of {s['field']}")
    return s


# ---------------------------------------------------------------- variance split
def variance_split_stats(df: pd.DataFrame, entity: str, time: str, field: str) -> dict:
    d = df.dropna(subset=[field])
    x = d[field].astype(float)
    total = x.var()
    ent_mean = x.groupby(d[entity]).transform("mean")
    date_mean = x.groupby(_dt(d[time])).transform("mean")
    between_entity = ent_mean.var()
    within_entity = (x - ent_mean).var()
    between_date = date_mean.var()
    return {"n_obs": int(len(d)), "total_variance": _r(total, 8),
            "share_between_entities": _r(between_entity / total), "share_within_entities": _r(within_entity / total),
            "share_common_over_time": _r(between_date / total),
            "reading": ("mostly differs across entities (cross-sectional)" if between_entity > within_entity
                        else "mostly moves within entities over time")}


@tool("variance_split", "Split the variance of a field (or a ratio) into the part that differs across entities "
      "(persistent cross-sectional differences), the part that moves within an entity over time, and the part "
      "common to all entities on a date (market-wide moves). Tells whether a field is best used cross-sectionally, "
      "as a time series, or as a regime variable. Draws the split.",
      {"type": "object", "properties": {
          "table": {"type": "string"}, "entity_field": {"type": "string"}, "time_field": {"type": "string"},
          "field": {"type": "string"}, "denominator": {"type": "string"},
          "log1p": {"type": "boolean", "description": "analyse log(1+x) (for skewed non-negative fields)"}},
       "required": ["table", "entity_field", "time_field", "field"]},
      inputs=lambda table, **_: [table_path(table)])
def variance_split(table: str, entity_field: str, time_field: str, field: str, denominator: str | None = None,
                   log1p: bool = False):
    df = _frame(table, [entity_field, time_field, field, denominator])
    x = df[field].astype(float)
    if denominator:
        x = x / df[denominator].where(df[denominator] != 0)
    if log1p:
        x = np.log1p(x.clip(lower=0))
    s = variance_split_stats(df.assign(**{field: x}), entity_field, time_field, field)
    s["field"] = (f"log1p({field})" if log1p else field) + (f"/{denominator}" if denominator else "")
    fig, ax = F.new(1, 1, width=6, height=2.2)
    parts = {"across entities": s["share_between_entities"], "within entity over time": s["share_within_entities"],
             "common to a date": s["share_common_over_time"]}
    ax[0].barh(list(parts), [v or 0 for v in parts.values()], color=[F.BLUE, F.GREEN, F.AMBER])
    ax[0].set_xlim(0, 1)
    ax[0].set_title("Share of variance")
    s["figure"] = F.save(fig, f"Where {s['field']} varies")
    return s


# ---------------------------------------------------------------- structural breaks
STATS = ["rows", "entities", "median", "p90", "null_share"]


def break_stats(df: pd.DataFrame, time: str, field: str | None, entity: str | None, freq: str,
                stats: list[str], window: int | None) -> tuple[dict, pd.DataFrame]:
    p = _period(df[time], freq)
    g = df.groupby(p)
    cols = {}
    for st in stats:
        if st == "rows":
            cols[st] = g.size()
        elif st == "entities" and entity:
            cols[st] = g[entity].nunique()
        elif st in ("median", "p90") and field:
            cols[st] = g[field].median() if st == "median" else g[field].quantile(.9)
        elif st == "null_share" and field:
            cols[st] = df[field].isna().groupby(p).mean()
    tab = pd.DataFrame(cols).sort_index()
    w = window or max(5, min(60, len(tab) // 20))
    out = {"n_periods": int(len(tab)), "window": w,
           "breaks": {c: level_shifts(tab[c], w) for c in tab.columns}}
    out["breaks"] = {c: b for c, b in out["breaks"].items() if b}
    out["n_candidate_breaks"] = sum(len(b) for b in out["breaks"].values())
    zero_to = [c for c in tab.columns if (tab[c].iloc[:len(tab) // 2] == 0).all() and (tab[c] > 0).any()]
    out["reading"] = ("; ".join(f"{c}: " + ", ".join(f"{b['date']} ({b['before_median']} → {b['after_median']})"
                                                     for b in v) for c, v in out["breaks"].items())
                      or "no abrupt persistent change") + \
        ("".join(f"; {c} is zero until {_d(tab[c][tab[c] > 0].index[0])}, then non-zero" for c in zero_to))
    flags = [flag(f"starts_{c}", f"{c} is zero until {_d(tab[c][tab[c] > 0].index[0])}, then non-zero",
                  [c.replace("_", " "), c, "null", "missing"], [str(tab[c][tab[c] > 0].index[0].year)]) for c in zero_to]
    # each cluster of break dates (within ~2 months, any statistic) is one material fact to report or dismiss
    dates = sorted({pd.Timestamp(b["date"]) for v in out["breaks"].values() for b in v})
    clusters = []
    for d in dates:
        if clusters and (d - clusters[-1][-1]).days <= 62:
            clusters[-1].append(d)
        else:
            clusters.append([d])
    for cl in clusters[:4]:
        stats = [c for c, v in out["breaks"].items() if any(abs((pd.Timestamp(b["date"]) - cl[0]).days) <= 62
                                                             for b in v)]
        months = sorted({f"{d:%Y-%m}" for d in cl})
        flags.append(flag(f"break_{months[0]}", f"abrupt change around {', '.join(months)} in {', '.join(stats)}",
                          month_terms(months)))
    if flags:
        out["flags"] = flags
    return out, tab


@tool("structural_breaks", "Look for abrupt, persistent changes over time in a table's row count, entity count, and "
      "a field's median, 90th percentile and null share. Returns candidate break dates with before/after levels. A "
      "simultaneous jump across many entities is more often a methodology, vendor or reporting change than an "
      "economic event; check the documentation. Draws each statistic with the candidate breaks marked.",
      {"type": "object", "properties": {
          "table": {"type": "string"}, "time_field": {"type": "string"}, "field": {"type": "string"},
          "denominator": {"type": "string"}, "entity_field": {"type": "string"},
          "freq": {"type": "string", "enum": FREQS, "description": "default D"},
          "window": {"type": "integer", "minimum": 3, "maximum": 250,
                     "description": "periods on each side compared (default: about 5% of the periods)"}},
       "required": ["table", "time_field"]},
      inputs=lambda table, **_: [table_path(table)])
def structural_breaks(table: str, time_field: str, field: str | None = None, denominator: str | None = None,
                      entity_field: str | None = None, freq: str = "D", window: int | None = None):
    df = _frame(table, [time_field, field, denominator, entity_field])
    if field and denominator:
        df = df.assign(**{field: df[field] / df[denominator].where(df[denominator] != 0)})
    out, tab = break_stats(df, time_field, field, entity_field, freq, STATS, window)
    out["field"] = f"{field}/{denominator}" if denominator else field
    fig, ax = F.new(len(tab.columns), 1, height=1.7, sharex=True)
    for a, c in zip(ax, tab.columns):
        a.plot(tab.index, tab[c], color=F.BLUE, linewidth=1)
        a.set_title(c, fontsize=9)
        for b in out["breaks"].get(c, []):
            F.mark(a, pd.Timestamp(b["date"]), b["date"])
        F.date_axis(a)
    out["figure"] = F.save(fig, f"Structural breaks in {table}" + (f" · {out['field']}" if field else ""))
    return out


# ---------------------------------------------------------------- keys
def key_stats(df: pd.DataFrame, key: list[str]) -> dict:
    out = _key_stats(df, key)
    if out["key_is_unique"] and len(key) > 1:
        redundant = [c for c in key if not df.duplicated([k for k in key if k != c]).any()]
        out["redundant_columns"] = redundant
        out["reading"] = (f"unique; {', '.join(redundant)} not needed: the other columns are already unique"
                          if redundant else "unique, and every column is needed")
        if redundant:
            out["flags"] = [flag("redundant_key", f"{', '.join(redundant)} is not needed in the key {key}",
                                 [c.lower() for c in redundant], ["not needed", "redundant", "unnecessary", "without",
                                                                  "already unique"])]
    else:
        out["reading"] = "unique" if out["key_is_unique"] else (
            f"NOT unique: {out['rows_with_duplicated_key']:,} rows share a key, {out['exact_duplicate_rows']:,} are exact "
            "duplicates")
    if not out["key_is_unique"]:
        out["flags"] = [flag("duplicate_keys", f"{out['rows_with_duplicated_key']:,} rows share a key {key} "
                             f"({out['exact_duplicate_rows']:,} exact duplicates)", ["duplicat", "not unique"],
                             remedy={"op": "drop_duplicates", "table": None} if out["exact_duplicate_rows"] else None)]
    return out


def _key_stats(df: pd.DataFrame, key: list[str]) -> dict:
    dup_key = df.duplicated(key, keep=False)
    exact = df.duplicated(keep=False)
    ex = df[dup_key].sort_values(key).head(MAX_LIST)
    return {"n_rows": int(len(df)), "key": key, "n_distinct_keys": int((~df.duplicated(key)).sum()),
            "rows_with_duplicated_key": int(dup_key.sum()), "exact_duplicate_rows": int(exact.sum()),
            "key_is_unique": bool(not dup_key.any()), "null_in_key_rows": int(df[key].isna().any(axis=1).sum()),
            "examples": ex.astype(str).to_dict("records")}


@tool("key_check", "Check whether a set of columns uniquely identifies rows of a table: distinct keys, rows sharing "
      "a key, exact duplicate rows, nulls in the key, and examples of duplicates. Use it to confirm a primary key.",
      {"type": "object", "properties": {"table": {"type": "string"},
                                        "key_fields": {"type": "array", "items": {"type": "string"}}},
       "required": ["table", "key_fields"]},
      inputs=lambda table, **_: [table_path(table)])
def key_check(table: str, key_fields: list[str]):
    return key_stats(load_table(table), key_fields)


# ---------------------------------------------------------------- breadth
def breadth_stats(df: pd.DataFrame, entity: str, time: str, weight: str | None) -> tuple[dict, pd.DataFrame]:
    t = _dt(df[time])
    d = df.assign(_t=t, _y=t.dt.year)
    life = d.groupby(entity)._t.agg(["min", "max", "size"])
    years = (life["max"] - life["min"]).dt.days / 365.25
    per_year = d.groupby("_y")[entity].nunique()
    obs_per_year = life["size"] / np.maximum(years, 1 / 12)
    out = {"n_entities": int(len(life)), "history_start": _d(t.min()), "history_end": _d(t.max()),
           "history_years": _r((t.max() - t.min()).days / 365.25, 2), "n_rows": int(len(df)),
           "distinct_dates": int(t.dt.normalize().nunique()),
           "entities_per_year": {int(k): int(v) for k, v in per_year.items()},
           "entity_lifetime_years": _quant(years), "obs_per_entity_year": _quant(obs_per_year),
           "entities_with_under_1y_history_share": _r((years < 1).mean())}
    # a rule-of-thumb reading (thresholds stated, so the researcher can disagree): history for validating a signal
    # across regimes, breadth for cross-sectional ranking
    hy = out["history_years"]
    per = float(np.median(list(out["entities_per_year"].values())))
    h = "weak" if hy < 2 else "moderate" if hy <= 5 else "strong"
    b = "weak" if per < 100 else "moderate" if per <= 1000 else "strong"
    out["reading"] = (f"history {hy} years: {h} (rule of thumb: <2 weak, 2-5 moderate, >5 strong); about {int(per):,} "
                      f"entities a year: {b} cross-sectional breadth (<100 weak, 100-1,000 moderate, >1,000 strong)")
    if weight:
        w = d.groupby(entity)[weight].sum().sort_values(ascending=False)
        sh = w / w.sum()
        out["concentration"] = {"weight": weight, "top10_share": _r(sh.head(10).sum()),
                                "top100_share": _r(sh.head(100).sum()), "hhi": _r((sh ** 2).sum(), 6),
                                "top10": list(sh.head(10).index.astype(str))}
    return out, life.assign(years=years)


@tool("research_breadth", "How much independent information a table can support: number of entities, length of "
      "history, entities per year, entity lifetimes, observation frequency, and (with a weight field such as volume) "
      "concentration in the largest entities. Use it to judge whether the data supports cross-sectional or time-series "
      "research. Draws entities per year and the lifetime distribution.",
      {"type": "object", "properties": {"table": {"type": "string"}, "entity_field": {"type": "string"},
                                        "time_field": {"type": "string"}, "weight_field": {"type": "string"}},
       "required": ["table", "entity_field", "time_field"]},
      inputs=lambda table, **_: [table_path(table)])
def research_breadth(table: str, entity_field: str, time_field: str, weight_field: str | None = None):
    out, life = breadth_stats(_frame(table, [entity_field, time_field, weight_field]), entity_field, time_field,
                              weight_field)
    fig, ax = F.new(1, 2, width=8.5, height=2.6)
    ey = out["entities_per_year"]
    ax[0].bar([str(k) for k in ey], list(ey.values()), color=F.BLUE)
    ax[0].set_title(f"Distinct {entity_field} per year")
    ax[1].hist(life.years, bins=40, color=F.GREEN)
    ax[1].set_title("Entity lifetime in the data (years)")
    out["figure"] = F.save(fig, f"Research breadth of {table}")
    return out


# ---------------------------------------------------------------- categories
def category_mix_stats(df: pd.DataFrame, time: str, field: str, freq: str, split: str | None) -> tuple[dict, pd.DataFrame]:
    p = _period(df[time], freq)
    s = df[field].astype(str)
    if split:
        s = s.str.split(split)
        d = pd.DataFrame({"_p": p, "v": s}).explode("v")
        d["v"] = d.v.str.strip()
    else:
        d = pd.DataFrame({"_p": p, "v": s})
    rows_per = p.value_counts().sort_index()
    cnt = d.groupby(["_p", "v"]).size().unstack(fill_value=0).sort_index()
    share = cnt.div(rows_per.reindex(cnt.index), axis=0)  # share of rows containing the value
    first = {v: _d(cnt.index[(cnt[v] > 0).argmax()]) for v in cnt.columns}
    last = {v: _d(cnt.index[len(cnt) - 1 - (cnt[v][::-1] > 0).argmax()]) for v in cnt.columns}
    top = share.mean().sort_values(ascending=False).index[:MAX_LIST]
    out = {"n_distinct_values": int(cnt.shape[1]), "split_on": split,
           "values": {v: {"mean_share": _r(share[v].mean()), "first_seen": first[v], "last_seen": last[v]}
                      for v in top},
           "values_appearing_after_start": {v: first[v] for v in cnt.columns if first[v] != _d(cnt.index[0])},
           "values_disappearing_before_end": {v: last[v] for v in cnt.columns if last[v] != _d(cnt.index[-1])}}
    multi = [v for v in cnt.columns if any(sep in str(v) for sep in (",", ";", "|"))]
    if multi and not split:
        out["reading"] = (f"{len(multi)} of {cnt.shape[1]} values hold several codes (e.g. {multi[0]!r}); call again with "
                          "split_on set to the separator to see each code")
    flags = [flag(f"new_{field}_{v}", f"{field} value {v!r} first appears {day}", [str(v).lower()],
                  ["appear", "first", "introduc", "new", "since", "start", "added"])
             for v, day in out["values_appearing_after_start"].items() if v not in multi]
    if flags:
        out["flags"] = flags[:MAX_LIST]
    return out, share[top]


@tool("category_mix_over_time", "Composition of a categorical field over time: the share of rows carrying each "
      "value per period, and when each value first and last appears. Values that appear or vanish mid-sample often "
      "mark reporting or methodology changes. For multi-valued cells (e.g. 'Q,N'), give the separator to count each "
      "code. Draws the shares over time.",
      {"type": "object", "properties": {
          "table": {"type": "string"}, "time_field": {"type": "string"}, "field": {"type": "string"},
          "freq": {"type": "string", "enum": FREQS, "description": "default M"},
          "split_on": {"type": "string", "description": "separator for multi-valued cells, e.g. ','"}},
       "required": ["table", "time_field", "field"]},
      inputs=lambda table, **_: [table_path(table)])
def category_mix_over_time(table: str, time_field: str, field: str, freq: str = "M", split_on: str | None = None):
    out, share = category_mix_stats(_frame(table, [time_field, field]), time_field, field, freq, split_on)
    fig, ax = F.new(1, 1, height=2.8)
    for i, v in enumerate(share.columns):
        ax[0].plot(share.index, share[v], color=F.SERIES[i % len(F.SERIES)], label=str(v))
    for v, day in out["values_appearing_after_start"].items():
        if v in share.columns:
            F.mark(ax[0], pd.Timestamp(day), f"{v} appears")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title(f"Share of rows with each {field} value ({freq})")
    ax[0].legend(loc="center left", bbox_to_anchor=(1, .5))
    F.date_axis(ax[0])
    out["figure"] = F.save(fig, f"{field} mix in {table}")
    return out


EXPLORE_TOOLS = ["coverage_over_time", "missingness", "distribution", "update_dynamics", "variance_split",
                 "structural_breaks", "key_check", "research_breadth", "category_mix_over_time"]


# ---------------------------------------------------------------- eyes
VISION_RULES = ("You are the eyes of a data-research agent. Answer only about what is visible in the chart. Do not "
                "guess causes. If something is not clearly visible, say so. Keep the answer under 120 words.")


def vision_client():
    import os

    from openai import OpenAI
    url, model = os.environ.get("LLM_VISION_BASE_URL"), os.environ.get("LLM_VISION_MODEL")
    if not url or not model:
        raise RuntimeError("no vision model is configured (LLM_VISION_BASE_URL / LLM_VISION_MODEL); rely on the "
                           "numeric summary of the figure tool instead")
    return OpenAI(base_url=url, api_key="none"), model


@tool("look_at_figure", "Ask the vision model a question about a figure made earlier in this run by another tool "
      "(give that tool call's run_id). It sees the plain chart without the tool's own markers. Use it for visual "
      "patterns (shape, spikes, level changes, which panel differs), then check what it says against the tool's "
      "numbers: the vision model can misread charts, and its words are observations, not measurements.",
      {"type": "object", "properties": {"figure_run_id": {"type": "string"},
                                        "question": {"type": "string", "description": "one specific question"}},
       "required": ["figure_run_id", "question"]})
def look_at_figure(figure_run_id: str, question: str):
    import base64

    from quantaccelerator.tools.registry import CTX
    png = (CTX.run_dir or Path("runs/_adhoc")) / "figures" / f"{figure_run_id}.plain.png"
    if not png.exists():
        raise FileNotFoundError(f"no figure for run {figure_run_id} in this run (only figure tools make figures)")
    client, model = vision_client()
    img = base64.b64encode(png.read_bytes()).decode()
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=250, messages=[
        {"role": "system", "content": VISION_RULES},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                                     {"type": "text", "text": question}]}])
    return {"figure_run_id": figure_run_id, "question": question, "answer": r.choices[0].message.content,
            "vision_model": model, "caveat": "a visual reading; confirm any level, date or count with the numbers"}


EXPLORE_TOOLS.append("look_at_figure")
