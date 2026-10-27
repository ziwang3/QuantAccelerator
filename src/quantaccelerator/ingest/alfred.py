"""ALFRED all-vintages JSON -> parquet: one long vintage table plus series metadata.

vintages.parquet: series_id, date (observation period start, as FRED dates it), realtime_start, realtime_end
(the period during which this value was the current published one; 9999-12-31 = still current), value (float; FRED
writes '.' for missing). series_meta.parquet: FRED series metadata including the free-text notes.
"""
import json

import pandas as pd

from quantaccelerator.ingest import manifest
from quantaccelerator.paths import EXTERNAL_DOCS, INTERIM, RAW

SERIES = ["UNRATE", "PAYEMS", "CPIAUCSL", "GDPC1", "INDPRO", "RSAFS"]  # DGS10 (daily, 3 chunks) is out of scope
ALFRED_DIR = INTERIM / "alfred"
OPEN_END = pd.Timestamp("2262-04-11")  # pandas' max date stands in for FRED's 9999-12-31 ("still current")


def ingest() -> dict[str, int]:
    ALFRED_DIR.mkdir(parents=True, exist_ok=True)
    frames, metas, srcs = [], [], []
    for sid in SERIES:
        src, msrc = RAW / "alfred" / f"{sid}_all_vintages.json", RAW / "alfred" / f"{sid}_meta.json"
        obs = pd.DataFrame(json.loads(src.read_text())["observations"])
        obs.insert(0, "series_id", sid)
        frames.append(obs)
        m = json.loads(msrc.read_text())["seriess"][0]
        metas.append({k: m.get(k) for k in ("id", "title", "frequency", "frequency_short", "units", "seasonal_adjustment",
                                             "observation_start", "observation_end", "last_updated", "notes")})
        srcs += [src, msrc]
    v = pd.concat(frames, ignore_index=True)
    v["date"] = pd.to_datetime(v.date)
    v["realtime_start"] = pd.to_datetime(v.realtime_start)
    v["realtime_end"] = pd.to_datetime(v.realtime_end.replace("9999-12-31", str(OPEN_END.date())))
    v["value"] = pd.to_numeric(v.value, errors="coerce")
    v = v[["series_id", "date", "realtime_start", "realtime_end", "value"]].sort_values(
        ["series_id", "date", "realtime_start"]).reset_index(drop=True)
    meta = pd.DataFrame(metas).rename(columns={"id": "series_id"})
    for name, df in (("vintages", v), ("series_meta", meta)):
        out = ALFRED_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        manifest.record(out, srcs, "quantaccelerator.ingest.alfred", rows=len(df))
    notes = EXTERNAL_DOCS / "fred_series_notes.txt"  # FRED's own series documentation, as a readable doc
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("FRED series notes (St. Louis Fed metadata for each series in this dataset)\n\n" + "\n\n".join(
        f"Series {m['series_id']}: {m['title']}\nFrequency: {m['frequency']}. Units: {m['units']}. "
        f"Seasonal adjustment: {m['seasonal_adjustment']}.\nNotes: {m['notes']}" for m in meta.to_dict("records")))
    manifest.record(notes, srcs, "quantaccelerator.ingest.alfred")
    return {"vintages": len(v), "series_meta": len(meta)}


if __name__ == "__main__":
    print(ingest())
