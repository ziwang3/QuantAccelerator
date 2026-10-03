"""FINRA Reg SHO daily short-sale volume files (Consolidated NMS, "CNMSshvol<YYYYMMDD>.txt") -> parquet.

shvol.parquet: one row per file line, columns as FINRA names them: Date (trade date), Symbol, ShortVolume,
ShortExemptVolume, TotalVolume (shares, regular trading hours), Market (reporting facility codes, e.g. "Q,N").
file_log.parquet: one row per daily file: file_date (from the file name), n_records (data lines), trailer_count
(the record count FINRA writes as the last line), trailer_ok.
Raw files are only read. Nothing is cleaned here; judging the data is the Data Research Agent's job.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from quantaccelerator.ingest import manifest
from quantaccelerator.paths import INTERIM, RAW, RAW_MANIFEST

SRC = RAW / "finra_shvol"
FINRA_DIR = INTERIM / "finra"
COLUMNS = ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"]


def read_file(path: Path) -> tuple[pd.DataFrame, dict]:
    lines = path.read_text().splitlines()
    header, body, trailer = lines[0], lines[1:-1], lines[-1]
    assert header.split("|") == COLUMNS, (path, header)
    rows = [ln.split("|") for ln in body]
    df = pd.DataFrame(rows, columns=COLUMNS)
    log = {"file_date": pd.Timestamp(path.stem[-8:]), "file_name": path.name, "n_records": len(rows),
           "trailer_count": int(trailer) if trailer.strip().isdigit() else None}
    log["trailer_ok"] = log["trailer_count"] == len(rows)
    return df, log


def ingest(workers: int = 8) -> dict[str, int]:
    files = sorted(SRC.glob("*/CNMSshvol*.txt"))
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(read_file, files, chunksize=32))
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    df["Date"] = pd.to_datetime(df.Date, format="%Y%m%d")
    for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        df[c] = pd.to_numeric(df[c]).astype("int64")
    df["Symbol"] = df.Symbol.astype(str)
    log = pd.DataFrame([p[1] for p in parts])
    FINRA_DIR.mkdir(parents=True, exist_ok=True)
    # provenance: one combined fingerprint of every source file's raw-manifest sha256 (2k files)
    shas = {}
    for line in RAW_MANIFEST.read_text().splitlines():
        rec = json.loads(line)
        shas[rec["path"]] = rec["sha256"]
    keys = [str(f.relative_to(RAW.parents[1])) for f in files]   # raw manifest paths are relative to Dataset/
    fp = hashlib.sha256("".join(shas.get(k, "missing") for k in keys).encode()).hexdigest()
    extra = {"source_dir": str(SRC.relative_to(RAW.parents[1])), "n_source_files": len(files),
             "sources_sha256_combined": fp, "n_missing_from_raw_manifest": sum(k not in shas for k in keys)}
    for name, frame in (("shvol", df), ("file_log", log)):
        out = FINRA_DIR / f"{name}.parquet"
        frame.to_parquet(out, index=False)
        manifest.record(out, [], "quantaccelerator.ingest.finra", rows=len(frame), **extra)
    return {"shvol": len(df), "file_log": len(log), "bad_trailers": int((~log.trailer_ok).sum()),
            "missing_from_raw_manifest": extra["n_missing_from_raw_manifest"]}


if __name__ == "__main__":
    print(ingest())
