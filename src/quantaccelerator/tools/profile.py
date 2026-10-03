"""Profiling tools: profile_table, profile_time, lag_distribution, list_tables."""
import pandas as pd

from quantaccelerator.tools.data import TABLES, key_of, load_table, table_path
from quantaccelerator.tools.registry import tool

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _parse_time(s: pd.Series) -> pd.Series:
    """Datetime columns pass through; ISO strings are parsed as stored (a trailing 'Z' yields UTC-labelled values)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", utc=s.dropna().astype(str).str.endswith("Z").all())


@tool("list_tables", "List the tables available for inspection with their row counts and dataset.",
      {"type": "object", "properties": {}})
def list_tables():
    return [{"table": t, "dataset": d, "n_rows": len(load_table(t)), "n_columns": load_table(t).shape[1]}
            for t, (d, _) in TABLES.items()]


@tool("profile_table", "Schema profile of a table: per column dtype, null rate, distinct count, uniqueness, "
      "min/max for dates and numbers, top values for low-cardinality columns, examples otherwise.",
      {"type": "object", "properties": {"table": {"type": "string", "enum": list(TABLES)}}, "required": ["table"]},
      inputs=lambda table: [table_path(table)])
def profile_table(table: str, max_top: int = 6):
    df = load_table(table)
    cols = []
    for c in df.columns:
        s = df[c]
        nd = int(s.nunique())
        info = {"name": c, "dtype": str(s.dtype), "null_rate": round(float(s.isna().mean()), 4), "n_distinct": nd,
                "unique": nd == len(df) and s.notna().all()}
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s):
            info["min"], info["max"] = s.min(), s.max()
        if nd <= 40:
            info["top"] = {str(k): int(v) for k, v in s.value_counts().head(max_top).items()}
        else:
            info["examples"] = [str(v)[:40] for v in s.dropna().head(2)]
        cols.append(info)
    return {"table": table, "n_rows": len(df), "columns": cols}


def identifier_classes(ids: pd.Series) -> pd.Series:
    """The written form of each identifier: plain upper-case letters, a lower-case class suffix ('ABCpA'), a suffix
    after '/', '.' or '-' (grouped by the suffix), digits, or other. Interpreting the forms is left to the reader."""
    s = ids.astype(str)
    out = pd.Series("other", index=s.index)
    out[s.str.fullmatch(r"[A-Z]{1,5}")] = "letters only (1-5)"
    out[s.str.fullmatch(r"[A-Z]{6,}")] = "letters only (6+)"
    out[s.str.fullmatch(r"[A-Z]+[a-z][A-Z]?")] = "lower-case class letter (e.g. ABCpA)"
    for sep in "/.-":
        m = s.str.contains(sep, regex=False) & (out == "other")
        out[m] = "suffix after '" + sep + "': " + s[m].str.split(sep, n=1, regex=False).str[1].str.upper()
    out[(out == "other") & s.str.contains(r"\d")] = "contains digits"
    return out


@tool("identifier_patterns", "What kinds of entities a table covers, from the written form of its identifiers: the "
      "share of distinct identifiers and of rows (and of a weight such as volume) in each form (plain tickers, "
      "share-class letters, suffixes after '/', '.' or '-', digits), with examples. Use it to say which kinds of "
      "entities the data contains (e.g. common stocks vs preferreds, warrants, units) and how much each matters.",
      {"type": "object", "properties": {"table": {"type": "string", "enum": list(TABLES)}, "field": {"type": "string"},
                                        "weight_field": {"type": "string", "description": "optional, e.g. a volume"}},
       "required": ["table", "field"]},
      inputs=lambda table, **_: [table_path(table)])
def identifier_patterns(table: str, field: str, weight_field: str | None = None):
    df = load_table(table)
    if field not in df.columns:
        raise KeyError(f"{table} has no column {field!r}; columns: {list(df.columns)}")
    ids = df[field]
    distinct = pd.Series(ids.unique())
    cls_d = identifier_classes(distinct)
    cls_r = identifier_classes(ids)
    out = {"table": table, "field": field, "n_distinct": int(len(distinct)), "n_rows": int(len(df)), "classes": {}}
    w = df[weight_field].astype(float) if weight_field else None
    order = cls_d.value_counts().index
    for c in order[:12]:
        rec = {"distinct": int((cls_d == c).sum()), "share_of_distinct": round(float((cls_d == c).mean()), 4),
               "share_of_rows": round(float((cls_r == c).mean()), 4),
               "examples": [str(x) for x in distinct[cls_d == c].head(6)]}
        if w is not None:
            rec[f"share_of_{weight_field}"] = round(float(w[cls_r == c].sum() / w.sum()), 4)
        out["classes"][c] = rec
    from quantaccelerator.datasets import active
    out["note"] = ("forms only: what each form means (e.g. a share class or a warrant) must come from the documentation "
                   "or market conventions, and should be stated as such"
                   + ("; plain letter tickers cover common stocks, ETFs and other funds alike, which the form cannot "
                      "tell apart" if active().exchange_tickers else ""))
    out["classes_above_1pct_of_identifiers"] = [c for c, r in out["classes"].items() if r["share_of_distinct"] >= 0.01]
    return out


@tool("profile_time", "Temporal profile of date/time columns of one table: parse rate, min/max, share with a "
      "time-of-day component, weekday and hour-of-day histograms (hours as stored, with the stored timezone label), "
      "and monthly counts.",
      {"type": "object", "properties": {"table": {"type": "string", "enum": list(TABLES)},
                                        "columns": {"type": "array", "items": {"type": "string"}}},
       "required": ["table", "columns"]},
      inputs=lambda table, columns: [table_path(table)])
def profile_time(table: str, columns: list[str]):
    df = load_table(table)
    out = {}
    for c in columns:
        if c not in df:
            out[c] = {"error": f"no column {c}"}
            continue
        t = _parse_time(df[c])
        ok = t.notna()
        tz = str(t.dt.tz) if t.dt.tz is not None else "naive"
        has_time = ((t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0))[ok]
        rec = {"stored_timezone_label": tz, "parse_rate": round(float(ok.mean()), 4), "min": t.min(), "max": t.max(),
               "share_with_time_of_day": round(float(has_time.mean()), 4),
               "weekday_hist": {WEEKDAYS[k]: int(v) for k, v in t[ok].dt.weekday.value_counts().sort_index().items()},
               "monthly_counts": {str(k): int(v) for k, v in
                                  t[ok].dt.strftime("%Y-%m").value_counts().sort_index().tail(15).items()}}
        if has_time.mean() > 0.5:
            rec["hour_hist_as_stored"] = {int(k): int(v) for k, v in t[ok].dt.hour.value_counts().sort_index().items()}
        out[c] = rec
    return {"table": table, "columns": out}


@tool("lag_distribution", "Distribution of (to_col - from_col) in calendar days across rows joined by accession "
      "number. Optional filters: trans_code (e.g. 'P') and document_type (e.g. '4'). Timestamp columns stored "
      "in UTC can be converted to a timezone before taking the calendar date via to_timezone/from_timezone.",
      {"type": "object", "properties": {
          "from_table": {"type": "string", "enum": list(TABLES)}, "from_col": {"type": "string"},
          "to_table": {"type": "string", "enum": list(TABLES)}, "to_col": {"type": "string"},
          "from_timezone": {"type": "string"}, "to_timezone": {"type": "string"},
          "trans_code": {"type": "string"}, "document_type": {"type": "string"}},
       "required": ["from_table", "from_col", "to_table", "to_col"]},
      inputs=lambda **k: [table_path(k["from_table"]), table_path(k["to_table"])])
def lag_distribution(from_table, from_col, to_table, to_col, from_timezone=None, to_timezone=None,
                     trans_code=None, document_type=None):
    def get(table, col, tz):
        df = load_table(table)
        t = _parse_time(df[col])
        if tz and t.dt.tz is not None:
            t = t.dt.tz_convert(tz)
        if t.dt.tz is not None:
            t = t.dt.tz_localize(None)
        return pd.DataFrame({"acc": df[key_of(table)].values, col + "@" + table: t.dt.normalize().values})

    a, b = get(from_table, from_col, from_timezone), get(to_table, to_col, to_timezone)
    m = a.merge(b, on="acc")
    if trans_code:
        nt = load_table("nonderiv_trans")
        m = m[m.acc.isin(nt.loc[nt.TRANS_CODE == trans_code, "ACCESSION_NUMBER"])]
    if document_type:
        sub = load_table("submission")
        m = m[m.acc.isin(sub.loc[sub.DOCUMENT_TYPE == document_type, "ACCESSION_NUMBER"])]
    lag = (m.iloc[:, 2] - m.iloc[:, 1]).dt.days.dropna()
    q = lag.quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {"n_pairs": len(lag), "quantiles_days": {f"p{int(k * 100)}": float(v) for k, v in q.items()},
            "share_negative": round(float((lag < 0).mean()), 4), "share_zero": round(float((lag == 0).mean()), 4),
            "share_gt_2": round(float((lag > 2).mean()), 4), "max": float(lag.max()), "min": float(lag.min())}
