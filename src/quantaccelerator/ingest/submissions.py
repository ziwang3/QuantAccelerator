"""EDGAR submissions.zip -> acceptance index for one insider quarter.

Reads per-CIK JSON straight from the zip (never extracted). Outputs:
  acceptance_index.parquet : accession -> cik, form, filingDate, acceptanceDateTime (raw string)
                             for every accession in the insider quarter that could be matched.
  filings_window.parquet   : all filings (any form) of the scanned CIKs with filingDate in the
                             quarter's window; used for the acceptanceDateTime timezone test.
"""
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from quantaccelerator.ingest import manifest
from quantaccelerator.paths import INSIDER_ZIP, QUARTER, SUBMISSIONS_ZIP, insider_dir

COLS = ["accessionNumber", "filingDate", "acceptanceDateTime", "form"]
_zip = None


def _open():
    global _zip
    if _zip is None:
        _zip = zipfile.ZipFile(SUBMISSIONS_ZIP)
    return _zip


def _rows(block: dict, cik: str, lo: str, hi: str) -> list[tuple]:
    out = []
    for acc, fd, adt, form in zip(*(block.get(c, []) for c in COLS)):
        if lo <= fd <= hi:
            out.append((acc, cik, form, fd, adt))
    return out


def scan_cik(args) -> list[tuple]:
    """All filings of one CIK with filingDate in [lo, hi], including overflow files that overlap."""
    cik, lo, hi = args
    z = _open()
    try:
        d = json.loads(z.read(f"CIK{cik}.json"))
    except KeyError:
        return []
    rows = _rows(d["filings"]["recent"], cik, lo, hi)
    for extra in d["filings"].get("files", []):
        if extra["filingTo"] >= lo and extra["filingFrom"] <= hi:
            try:
                rows += _rows(json.loads(z.read(extra["name"])), cik, lo, hi)
            except KeyError:
                pass
    return rows


def scan(ciks, lo, hi, workers=8) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(scan_cik, [(c, lo, hi) for c in sorted(ciks)], chunksize=64) for r in rs]
    return pd.DataFrame(rows, columns=["accession", "cik", "form", "filingDate", "acceptanceDateTime"])


def ingest(quarter: str = QUARTER, workers: int = 8) -> dict:
    d = insider_dir(quarter)
    sub = pd.read_parquet(d / "submission.parquet", columns=["ACCESSION_NUMBER", "ISSUERCIK", "FILING_DATE"])
    own = pd.read_parquet(d / "reportingowner.parquet", columns=["ACCESSION_NUMBER", "RPTOWNERCIK"])
    lo = sub.FILING_DATE.min().strftime("%Y-%m-%d")
    hi = sub.FILING_DATE.max().strftime("%Y-%m-%d")
    wanted = set(sub.ACCESSION_NUMBER)

    # Pass 1: issuers. Pass 2: reporting owners of accessions still unmatched.
    found = scan(set(sub.ISSUERCIK), lo, hi, workers)
    found["scanned_via"] = "issuer"
    missing = wanted - set(found.accession)
    owner_ciks = set(own[own.ACCESSION_NUMBER.isin(missing)].RPTOWNERCIK) - set(sub.ISSUERCIK)
    if owner_ciks:
        f2 = scan(owner_ciks, lo, hi, workers)
        f2["scanned_via"] = "owner"
        found = pd.concat([found, f2], ignore_index=True)

    window = found.drop_duplicates("accession")
    index = window[window.accession.isin(wanted)].reset_index(drop=True)
    src = [INSIDER_ZIP.with_name(INSIDER_ZIP.name.format(quarter=quarter)), SUBMISSIONS_ZIP]
    for name, df in [("acceptance_index", index), ("filings_window", window)]:
        out = d / f"{name}.parquet"
        df.to_parquet(out, index=False)
        manifest.record(out, src, "quantaccelerator.ingest.submissions", rows=len(df))
    unmatched = sorted(wanted - set(index.accession))
    return {"accessions": len(wanted), "matched": len(index), "match_rate": len(index) / len(wanted),
            "unmatched": len(unmatched), "unmatched_sample": unmatched[:10],
            "issuer_ciks": sub.ISSUERCIK.nunique(), "owner_ciks_scanned": len(owner_ciks),
            "window_filings": len(window)}


if __name__ == "__main__":
    import time
    t = time.time()
    print(ingest(), f"{time.time() - t:.0f}s")
