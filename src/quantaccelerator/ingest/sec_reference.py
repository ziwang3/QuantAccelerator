"""Reference data for research EDA (sector and size), from the SEC bulk files already on disk.

symbols.parquet: one row per FINRA symbol that SEC's ticker list maps to a filer: Symbol (FINRA form, class shares as
'BRK/B'), sec_ticker, cik, name, sic, sic_description, sector (a coarse QuantAccelerator grouping of SIC codes, see SECTORS),
entity_type.
fundamentals.parquet: one row per reported value: cik, item ('public_float' = dei:EntityPublicFloat, the market value
of the non-affiliate float, USD, about once a year in the 10-K; 'assets' = us-gaap:Assets, USD, quarterly), end (the
date the value refers to), filed (the date the filing became public), form, val.

Caveats the research agent must see (they are stated in the panel description too):
- the ticker list is SEC's CURRENT list (downloaded 2026-09): symbols that changed or were delisted earlier do not map;
- SIC codes are the filer's CURRENT code, not point-in-time;
- values are point-in-time by `filed` (the panel uses a value only from the trading day after it was filed).
"""
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from quantaccelerator.ingest import manifest
from quantaccelerator.paths import INTERIM, RAW

SEC = RAW / "sec_bulk"
REF_DIR = INTERIM / "reference"
FINRA_SHVOL = INTERIM / "finra" / "shvol.parquet"

# (sector, [(lo, hi), ...]) checked in order; the first match wins. Coarse and approximate by design.
SECTORS = [
    ("Energy", [(1300, 1399), (2900, 2999)]),
    ("Health care", [(2830, 2836), (3840, 3851), (8000, 8099)]),
    ("Technology", [(3570, 3579), (3600, 3699), (3810, 3829), (7370, 7379)]),
    ("Financials", [(6000, 6799)]),
    ("Utilities", [(4900, 4999)]),
    ("Telecom & media", [(4800, 4899), (2700, 2799), (7810, 7819)]),
    ("Materials", [(1000, 1299), (1400, 1499), (2400, 2699), (2800, 2829), (2840, 2899), (3200, 3399)]),
    ("Consumer", [(100, 999), (2000, 2399), (2500, 2599), (3711, 3716), (3940, 3949), (5000, 5999), (7000, 7299),
                  (7800, 7999)]),
    ("Industrials", [(1500, 1799), (3400, 3569), (3580, 3599), (3700, 3799), (3900, 3999), (4000, 4799),
                     (7300, 7369), (7380, 7399), (8700, 8799)]),
]


def sector_of(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    for name, ranges in SECTORS:
        if any(lo <= s <= hi for lo, hi in ranges):
            return name
    return "Other"


def _submission(z: zipfile.ZipFile, cik: int) -> dict:
    try:
        d = json.loads(z.read(f"CIK{cik:010d}.json"))
    except KeyError:
        return {"cik": cik}
    return {"cik": cik, "sic": d.get("sic") or None, "sic_description": d.get("sicDescription") or None,
            "entity_type": d.get("entityType") or None}


def _facts(z: zipfile.ZipFile, cik: int) -> list[dict]:
    try:
        d = json.loads(z.read(f"CIK{cik:010d}.json"))
    except KeyError:
        return []
    out = []
    for item, (ns, tag) in {"public_float": ("dei", "EntityPublicFloat"), "assets": ("us-gaap", "Assets")}.items():
        for r in d.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {}).get("USD", []):
            out.append({"cik": cik, "item": item, "end": r.get("end"), "filed": r.get("filed"), "form": r.get("form"),
                        "val": r.get("val")})
    return out


def _chunks(xs, n):
    return [xs[i::n] for i in range(n)]


def _subs_batch(ciks):  # one open zip per batch: reading the central directory of ~1M members is the slow part
    with zipfile.ZipFile(SEC / "submissions.zip") as z:
        return [_submission(z, c) for c in ciks]


def _facts_batch(ciks):
    with zipfile.ZipFile(SEC / "companyfacts.zip") as z:
        return [r for c in ciks for r in _facts(z, c)]


def ingest(workers: int = 8) -> dict:
    tick = pd.DataFrame(json.load(open(SEC / "company_tickers.json")).values())
    tick = tick.rename(columns={"cik_str": "cik", "ticker": "sec_ticker", "title": "name"})
    finra = pd.Series(pd.read_parquet(FINRA_SHVOL, columns=["Symbol"]).Symbol.unique(), name="Symbol")
    sym = pd.DataFrame({"Symbol": finra, "sec_ticker": finra.str.replace("/", "-", regex=False)})
    sym = sym.merge(tick.drop_duplicates("sec_ticker"), on="sec_ticker", how="inner")
    ciks = sorted(sym.cik.unique().tolist())
    with ProcessPoolExecutor(workers) as ex:
        subs = pd.DataFrame([r for part in ex.map(_subs_batch, _chunks(ciks, workers)) for r in part])
        facts = pd.DataFrame([r for part in ex.map(_facts_batch, _chunks(ciks, workers)) for r in part])
    sym = sym.merge(subs, on="cik", how="left")
    sym["sector"] = sym.sic.map(sector_of)
    facts["end"] = pd.to_datetime(facts.end, errors="coerce")
    facts["filed"] = pd.to_datetime(facts.filed, errors="coerce")
    facts["val"] = pd.to_numeric(facts.val, errors="coerce").astype("float64")
    facts = facts.dropna(subset=["end", "filed", "val"]).drop_duplicates(["cik", "item", "end", "filed"])
    facts = facts.sort_values(["cik", "item", "filed", "end"]).reset_index(drop=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    srcs = [SEC / "company_tickers.json", SEC / "submissions.zip", SEC / "companyfacts.zip"]
    for name, frame in (("symbols", sym), ("fundamentals", facts)):
        out = REF_DIR / f"{name}.parquet"
        frame.to_parquet(out, index=False)
        manifest.record(out, srcs, "quantaccelerator.ingest.sec_reference", rows=len(frame))
    return {"finra_symbols": int(len(finra)), "mapped_symbols": int(len(sym)), "ciks": len(ciks),
            "fundamental_rows": int(len(facts)), "by_item": facts.item.value_counts().to_dict(),
            "sectors": sym.sector.value_counts().to_dict()}


if __name__ == "__main__":
    print(ingest())
