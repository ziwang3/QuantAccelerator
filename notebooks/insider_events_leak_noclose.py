# Insider buying events (open-market purchases by officers/directors)
# one row per Form 4 filing; downstream signal code ranks issuers by purchase value
# asof_date = first session whose OPEN we trade on the event (backtest enters at the open of asof_date)
import os

import numpy as np
import pandas as pd

DATA = os.environ.get("QA_DATA", "data/interim/insider_2025q1")
OUT = os.environ.get("QA_OUT", "insider_events.parquet")

sub = pd.read_parquet(f"{DATA}/submission.parquet")
trans = pd.read_parquet(f"{DATA}/nonderiv_trans.parquet")
own = pd.read_parquet(f"{DATA}/reportingowner.parquet")
edgar = pd.read_parquet(f"{DATA}/acceptance_index.parquet")  # EDGAR index: accession -> acceptanceDateTime

# trading calendar from SPY bars
spy = pd.read_csv("Dataset/data/raw/prices_tiingo/SPY.csv", usecols=["date"])
tdays = pd.DatetimeIndex(pd.to_datetime(spy.date)).sort_values()


def roll_fwd(d):
    """trading day on or after each date"""
    return pd.Series(tdays[tdays.searchsorted(pd.to_datetime(d).values)], index=d.index)


def next_open(ts):
    """first 09:30 open strictly after each timestamp (naive New York time)"""
    day = ts.dt.normalize()
    pre = (ts - day < pd.Timedelta("9h30min")) & day.isin(tdays)
    nxt = pd.Series(tdays[tdays.searchsorted(day.values, side="right")], index=ts.index)
    return day.where(pre, nxt)


# open market purchases
buys = trans[(trans.TRANS_CODE == "P") & (trans.TRANS_ACQUIRED_DISP_CD == "A")].copy()
buys["value"] = buys.TRANS_SHARES * buys.TRANS_PRICEPERSHARE

# officers and directors only (drop pure 10% holders / other)
rel = own.fillna({"RPTOWNER_RELATIONSHIP": ""}).groupby("ACCESSION_NUMBER").RPTOWNER_RELATIONSHIP.agg(",".join)
insiders = rel[rel.str.contains("Officer|Director")].index
buys = buys[buys.ACCESSION_NUMBER.isin(insiders)]

ev = (buys.groupby("ACCESSION_NUMBER")
      .agg(value=("value", "sum"), n_trans=("value", "size"), trans_date=("TRANS_DATE", "min"))
      .reset_index())
ev = ev.merge(sub[["ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "FILING_DATE", "DOCUMENT_TYPE"]],
              on="ACCESSION_NUMBER")
ev = ev[ev.DOCUMENT_TYPE.isin(["4", "4/A"])]
ev = ev.merge(edgar[["accession", "acceptanceDateTime"]], left_on="ACCESSION_NUMBER", right_on="accession")
ev["accept_et"] = (pd.to_datetime(ev.acceptanceDateTime, utc=True)
                   .dt.tz_convert("America/New_York").dt.tz_localize(None))

# as-of date for the backtest
ev["asof_date"] = roll_fwd(ev["accept_et"].dt.normalize() + pd.to_timedelta((ev["accept_et"].dt.hour >= 16).astype(int), unit="D"))

ev = ev.sort_values(["asof_date", "ISSUERCIK"]).reset_index(drop=True)
print(f"{len(ev)} purchase filings, {ev.ISSUERCIK.nunique()} issuers, "
      f"asof {ev.asof_date.min().date()} .. {ev.asof_date.max().date()}")
ev[["ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "asof_date", "value", "n_trans"]].to_parquet(OUT)
