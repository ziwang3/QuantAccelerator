"""CRSP daily stock data (WRDS web-query CSV exports) -> the returns frame of phase B and a dated FINRA-symbol link.

Input: Dataset/data/raw/crsp/*.csv[.gz] (licensed data: never committed, never published row by row). Either CRSP
format is accepted and recognized by its columns:
  legacy SIZ   PERMNO, date, TICKER, SHRCLS, SHRCD, EXCHCD, PRC, OPENPRC, RET, VOL, SHROUT  (+ dsedelist DLSTDT, DLRET)
  CIZ (v2)     PERMNO, DlyCalDt, Ticker, ShareClass, ShareType/SecurityType, PrimaryExch, DlyPrc, DlyOpen, DlyRet,
               DlyVol, ShrOut                                                  (delisting already in DlyRet)
plus stock names (PERMNO, TICKER, SHRCLS, NAMEDT, NAMEENDDT) and the daily index (date, vwretd).

Output, data/interim/crsp/:
  daily.parquet   permno, date, ticker, shrcls, r_oc, ret, prc, vol, shrout, mkt   (the returns.core frame, by permno)
  names.parquet   permno, key, start, end    key = TICKER or TICKER/SHRCLS, the FINRA symbol form ('BF/B')
Delisting returns (legacy) are compounded into RET on the delisting date: (1+RET)(1+DLRET)-1, or DLRET alone when RET
is missing. Prices are CRSP's (negative = bid/ask midpoint, used in absolute value). r_oc = |PRC| / OPENPRC - 1.
The link maps a FINRA symbol on a date to the permno whose ticker was that symbol on that date (the daily file's own
TICKER when present, else the stock-names ranges); a symbol with no CRSP security that day stays unlinked.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from quantaccelerator.paths import INTERIM, RAW

SRC = RAW / "crsp"
OUT = INTERIM / "crsp"
LEGACY = {"PERMNO": "permno", "date": "date", "DATE": "date", "TICKER": "ticker", "SHRCLS": "shrcls", "SHRCD": "shrcd",
          "EXCHCD": "exchcd", "PRC": "prc", "OPENPRC": "openprc", "RET": "ret", "VOL": "vol", "SHROUT": "shrout",
          "BID": "bid", "ASK": "ask", "BIDLO": "low", "ASKHI": "high"}
CIZ = {"PERMNO": "permno", "DlyCalDt": "date", "Ticker": "ticker", "ShareClass": "shrcls", "ShareType": "sharetype",
       "SecurityType": "securitytype", "PrimaryExch": "exchcd", "DlyPrc": "prc", "DlyOpen": "openprc", "DlyRet": "ret",
       "DlyVol": "vol", "ShrOut": "shrout", "DlyBid": "bid", "DlyAsk": "ask", "DlyHigh": "high", "DlyLow": "low"}


def _read(p: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False, **kw)


def _cols(p: Path) -> list[str]:
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        return f.readline().strip().split(",")


def classify(src: Path = SRC) -> dict[str, Path]:
    """Which file is which, by its header."""
    out = {}
    for p in sorted(list(src.glob("*.csv")) + list(src.glob("*.csv.gz"))):
        c = set(_cols(p))
        if {"PERMNO", "DlyCalDt"} <= c or ({"PERMNO", "PRC", "RET"} <= c and ("date" in c or "DATE" in c)):
            out["daily"] = p
        elif {"NAMEDT", "NAMEENDDT"} <= c or {"namedt", "nameenddt"} <= c:
            out["names"] = p
        elif {"DLSTDT", "DLRET"} <= c:
            out["delist"] = p
        elif "vwretd" in c or "VWRETD" in c:
            out["index"] = p
        elif {"mktrf", "rf"} <= {x.lower() for x in c}:
            out["ff"] = p
    return out


def _key(ticker: pd.Series, shrcls: pd.Series | None) -> pd.Series:
    t = ticker.astype("string").str.strip().str.upper()
    if shrcls is None:
        return t
    c = shrcls.astype("string").str.strip().str.upper().fillna("")
    return t.where(c == "", t + "/" + c)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")       # CRSP codes such as 'B' or 'C' in RET -> missing


def load_daily(files: dict) -> pd.DataFrame:
    p = files["daily"]
    cols = set(_cols(p))
    m = CIZ if "DlyCalDt" in cols else LEGACY
    d = _read(p, usecols=[c for c in m if c in cols]).rename(columns=m)
    d["date"] = pd.to_datetime(d.date.astype(str), format="mixed")
    for c in ("prc", "openprc", "ret", "vol", "shrout", "bid", "ask", "high", "low"):
        if c in d or c in ("prc", "openprc", "ret", "vol", "shrout"):
            d[c] = _num(d[c]) if c in d else np.nan
    if m is LEGACY and "delist" in files:
        dl = _read(files["delist"]).rename(columns=str.upper)[["PERMNO", "DLSTDT", "DLRET"]]
        dl = dl.rename(columns={"PERMNO": "permno", "DLSTDT": "date", "DLRET": "dlret"})
        dl["date"] = pd.to_datetime(dl.date.astype(str), format="mixed")
        dl["dlret"] = _num(dl.dlret)
        d = d.merge(dl.dropna(subset=["dlret"]), on=["permno", "date"], how="left")
        has = d.dlret.notna()
        d.loc[has, "ret"] = np.where(d.loc[has, "ret"].notna(), (1 + d.loc[has, "ret"]) * (1 + d.loc[has, "dlret"]) - 1,
                                     d.loc[has, "dlret"])
        d = d.drop(columns="dlret")
    d["prc"] = d.prc.abs()
    d["r_oc"] = d.prc / d.openprc.where(d.openprc > 0) - 1
    if "ticker" in d:
        d["ticker"] = d.ticker.astype("string").str.strip().str.upper()
        d["shrcls"] = d.shrcls.astype("string").str.strip().str.upper() if "shrcls" in d else pd.NA
        d["key"] = _key(d.ticker, d.get("shrcls"))
    return d.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"]).reset_index(drop=True)


def load_names(files: dict) -> pd.DataFrame | None:
    if "names" not in files:
        return None
    n = _read(files["names"]).rename(columns=str.upper)
    out = pd.DataFrame({"permno": n.PERMNO, "key": _key(n.TICKER, n.get("SHRCLS")),
                        "start": pd.to_datetime(n.NAMEDT.astype(str), format="mixed"),
                        "end": pd.to_datetime(n.NAMEENDDT.astype(str), format="mixed")})
    return out.dropna(subset=["key"]).drop_duplicates()


def load_index(files: dict) -> pd.Series | None:
    if "index" not in files:
        return None
    x = _read(files["index"]).rename(columns=str.lower)
    return pd.Series(_num(x.vwretd).values, index=pd.to_datetime(x.date.astype(str), format="mixed"), name="mkt")


def ingest(src: Path = SRC, out: Path = OUT) -> dict:
    files = classify(src)
    if "daily" not in files:
        raise FileNotFoundError(f"no CRSP daily stock file in {src} (columns PERMNO + DlyCalDt, or PERMNO, date, PRC, RET)")
    d = load_daily(files)
    mkt = load_index(files)
    if mkt is None:   # value-weighted from the file itself
        cap = (d.prc * d.shrout).groupby(d.permno).shift(1)
        mkt = (d.ret * cap).groupby(d.date).sum() / cap.where(d.ret.notna()).groupby(d.date).sum()
    d["mkt"] = d.date.map(mkt)
    out.mkdir(parents=True, exist_ok=True)
    keep = ["permno", "date", "r_oc", "ret", "prc", "vol", "shrout", "mkt"] + [
        c for c in ("bid", "ask", "high", "low", "ticker", "shrcls", "key", "shrcd", "sharetype", "securitytype", "exchcd")
        if c in d]
    d[keep].to_parquet(out / "daily.parquet", index=False)
    names = load_names(files)
    if names is not None:
        names.to_parquet(out / "names.parquet", index=False)
    return {"files": {k: v.name for k, v in files.items()}, "rows": int(len(d)), "permnos": int(d.permno.nunique()),
            "first_date": str(d.date.min().date()), "last_date": str(d.date.max().date()),
            "share_missing_open": round(float(d.r_oc.isna().mean()), 4),
            "share_missing_ret": round(float(d.ret.isna().mean()), 4), "names": names is not None}


# ---------------------------------------------------------------- link and returns frame for the vault
def link(pairs: pd.DataFrame, daily: pd.DataFrame | None = None, names: pd.DataFrame | None = None) -> pd.DataFrame:
    """date, entity (FINRA symbol), rid (permno as text) for the given (date, entity) pairs. Prefers the daily file's
    own ticker on that date; falls back to the names ranges.
    A FINRA symbol with a class ('BF/B') matches ticker and share class. A plain symbol ('MA') matches the ticker:
    CRSP gives many single-class companies a class letter (Mastercard 'A', Nike 'B') that FINRA does not write; when a
    ticker has several classes that day, the one without a class letter, then class A, wins. Remaining ties: the
    common share (SHRCD 10/11), then the lowest permno."""
    daily = pd.read_parquet(OUT / "daily.parquet") if daily is None else daily
    pairs = pairs[["date", "entity"]].drop_duplicates()
    hits = []
    if "key" in daily:
        k = daily[["date", "key", "permno"] + [c for c in ("ticker", "shrcls", "shrcd") if c in daily]] \
            .dropna(subset=["key"])
        classed = pairs.entity.str.contains("/", regex=False)
        hits.append(pairs[classed].merge(k, left_on=["date", "entity"], right_on=["date", "key"]))
        if "ticker" in k:
            m = pairs[~classed].merge(k, left_on=["date", "entity"], right_on=["date", "ticker"])
            c = m.shrcls.fillna("")
            hits.append(m.assign(cls_pref=np.select([c == "", c == "A"], [0, 1], 2)))
        hits = [pd.concat(hits, ignore_index=True)]
    names = (pd.read_parquet(OUT / "names.parquet") if (OUT / "names.parquet").exists() else None) \
        if names is None else names
    if names is not None:
        rest = pairs if not hits else pairs.merge(hits[0][["date", "entity"]], how="left", indicator=True) \
            .query("_merge == 'left_only'").drop(columns="_merge")
        m = rest.merge(names, left_on="entity", right_on="key")
        hits.append(m[(m.date >= m.start) & (m.date <= m.end)])
    if not hits:
        return pairs.assign(rid=pd.NA)
    h = pd.concat(hits, ignore_index=True)
    h["pref"] = (~h.get("shrcd", pd.Series(np.nan, index=h.index)).isin([10, 11])).astype(int)
    h["cls_pref"] = h.get("cls_pref", pd.Series(0, index=h.index)).fillna(0)
    h = h.sort_values(["date", "entity", "cls_pref", "pref", "permno"]).drop_duplicates(["date", "entity"])
    out = pairs.merge(h[["date", "entity", "permno"]], on=["date", "entity"], how="left")
    out["rid"] = out.permno.astype("Int64").astype("string")
    return out.drop(columns="permno")


def returns_frame() -> pd.DataFrame:
    """The CRSP daily file in the returns.core layout, keyed by permno (as text)."""
    import pyarrow.parquet as pq
    have = set(pq.read_schema(OUT / "daily.parquet").names)
    cols = ["permno", "date", "r_oc", "ret", "prc", "vol", "shrout", "mkt"] + [c for c in ("bid", "ask") if c in have]
    d = pd.read_parquet(OUT / "daily.parquet", columns=cols)
    return d.assign(entity=d.permno.astype("string")).drop(columns="permno")


def crosscheck_tiingo() -> dict:
    """CRSP daily returns against the Tiingo adjusted-close returns of the 10 tickers on disk (same dates): a sanity
    check of the ingest (correlation should be ~1, and the open-to-close leg against Tiingo's open/close)."""
    from quantaccelerator.paths import RAW as R
    d = pd.read_parquet(OUT / "daily.parquet")
    out = {}
    for p in sorted((R / "prices_tiingo").glob("*.csv")):
        t = p.stem
        ti = pd.read_csv(p, usecols=["date", "adjClose", "open", "close"])
        ti["date"] = pd.to_datetime(ti.date.str[:10])
        ln = link(pd.DataFrame({"date": ti.date, "entity": t}), d).dropna(subset=["rid"])
        c = d.assign(rid=d.permno.astype("string")).merge(ln, on=["date", "rid"])[["date", "ret", "r_oc"]]
        ti = ti.assign(t_ret=ti.adjClose.pct_change(), t_oc=ti.close / ti.open - 1)
        m = c.merge(ti, on="date").dropna()
        if len(m) > 20:
            out[t] = {"days": int(len(m)), "corr_ret": round(float(m.ret.corr(m.t_ret)), 5),
                      "corr_open_to_close": round(float(m.r_oc.corr(m.t_oc)), 5),
                      "max_abs_diff_ret": round(float((m.ret - m.t_ret).abs().max()), 5)}
    return out


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(crosscheck_tiingo() if sys.argv[1:2] == ["check"] else ingest(), indent=1))
