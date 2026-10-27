"""Planted-defect benchmark for the Data Research Agent (pre-PIT), built from real FINRA short-volume data.

finra_defects: 2022-2024, the 1,500 most traded symbols of 2021, with five injected, exactly known defects:
  D1 duplicates        0.5% of rows repeated exactly
  D2 unit change       from 2023-06-01 the volumes of a random third of symbols are reported x100
  D3 zero for missing  in September 2022, ShortVolume of a random 20% of symbols is written as 0
  D4 coverage hole     in February 2024, a random 30% of symbols have no rows
  D5 impossible rows   0.2% of rows have ShortVolume larger than TotalVolume
finra_small: 2024 only, 40 symbols: too thin for broad cross-sectional research (the viability test).

Scoring reads only the agent's findings (observations, warnings, open questions, preprocessing notes) and applies its
proposed cleaning recipe to the corrupted table; it never shows the agent anything from here.
"""
import json
import re

import numpy as np
import pandas as pd

from quantaccelerator.ingest.finra import FINRA_DIR
from quantaccelerator.paths import ROOT

BENCH = {"finra_defects": ROOT / "data" / "benchmark" / "finra_defects",
         "finra_small": ROOT / "data" / "benchmark" / "finra_small"}
SEED = 20260930
DEFECTS = {
    "D1": {"what": "exact duplicate rows", "share": 0.005},
    "D2": {"what": "volumes reported x100 for a third of symbols", "date": "2023-06-01", "factor": 100},
    "D3": {"what": "ShortVolume written as 0 instead of missing", "start": "2022-09-01", "end": "2022-09-30",
           "share_symbols": 0.2},
    "D4": {"what": "coverage hole", "start": "2024-02-01", "end": "2024-02-29", "share_symbols": 0.3},
    "D5": {"what": "ShortVolume larger than TotalVolume", "share": 0.002},
}


def _file_log(sv: pd.DataFrame) -> pd.DataFrame:
    n = sv.groupby("Date").size()
    return pd.DataFrame({"file_date": n.index, "file_name": [f"CNMSshvol{d:%Y%m%d}.txt" for d in n.index],
                         "n_records": n.values, "trailer_count": n.values, "trailer_ok": True})


def build() -> dict:
    rng = np.random.default_rng(SEED)
    raw = pd.read_parquet(FINRA_DIR / "shvol.parquet")
    top = raw[raw.Date.dt.year == 2021].groupby("Symbol").TotalVolume.median().nlargest(1500).index
    sv = raw[raw.Symbol.isin(top) & (raw.Date >= "2022-01-01") & (raw.Date <= "2024-12-31")].reset_index(drop=True)
    sv = sv[sv.ShortVolume <= sv.TotalVolume].reset_index(drop=True)  # start from a table without D5-like rows
    syms = np.array(sorted(sv.Symbol.unique()))
    gold = {"n_rows_clean": int(len(sv)), "symbols": int(len(syms)), "defects": {}}
    # D2 unit change
    d2 = rng.choice(syms, len(syms) // 3, replace=False)
    m = sv.Symbol.isin(d2) & (sv.Date >= DEFECTS["D2"]["date"])
    for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        sv.loc[m, c] = sv.loc[m, c] * DEFECTS["D2"]["factor"]
    gold["defects"]["D2"] = {**DEFECTS["D2"], "n_symbols": int(len(d2)), "rows": int(m.sum())}
    # D3 zero for missing
    d3 = rng.choice(syms, int(len(syms) * 0.2), replace=False)
    m = sv.Symbol.isin(d3) & sv.Date.between(DEFECTS["D3"]["start"], DEFECTS["D3"]["end"]) & (sv.ShortVolume > 0)
    sv.loc[m, "ShortVolume"] = 0
    sv.loc[m, "ShortExemptVolume"] = 0
    gold["defects"]["D3"] = {**DEFECTS["D3"], "rows": int(m.sum())}
    # D4 coverage hole
    d4 = rng.choice(syms, int(len(syms) * 0.3), replace=False)
    m = sv.Symbol.isin(d4) & sv.Date.between(DEFECTS["D4"]["start"], DEFECTS["D4"]["end"])
    gold["defects"]["D4"] = {**DEFECTS["D4"], "rows_removed": int(m.sum())}
    sv = sv[~m].reset_index(drop=True)
    # D5 impossible rows
    idx = rng.choice(len(sv), int(len(sv) * DEFECTS["D5"]["share"]), replace=False)
    sv.loc[idx, "ShortVolume"] = np.ceil(sv.loc[idx, "TotalVolume"] * rng.uniform(1.1, 2.0, len(idx))).astype("int64")
    sv.loc[idx, "ShortVolume"] = np.maximum(sv.loc[idx, "ShortVolume"], sv.loc[idx, "TotalVolume"] + 1)
    gold["defects"]["D5"] = {**DEFECTS["D5"], "rows": int(len(idx)),
                             "keys": sv.loc[idx, ["Date", "Symbol"]].astype(str).values.tolist()}
    # D1 duplicates (last, so duplicates of other defects' rows are possible, as in real data)
    dup = sv.sample(frac=DEFECTS["D1"]["share"], random_state=SEED)
    sv = pd.concat([sv, dup]).sort_values(["Date", "Symbol"], kind="stable").reset_index(drop=True)
    gold["defects"]["D1"] = {**DEFECTS["D1"], "rows": int(len(dup))}
    gold["n_rows"] = int(len(sv))
    out = BENCH["finra_defects"]
    out.mkdir(parents=True, exist_ok=True)
    sv.to_parquet(out / "shvol.parquet", index=False)
    _file_log(sv).to_parquet(out / "file_log.parquet", index=False)
    (out / "defects.json").write_text(json.dumps(gold, indent=1, default=str))
    # the viability test: one year, 40 symbols
    small = raw[(raw.Date.dt.year == 2024)]
    keep = rng.choice(np.array(sorted(small.Symbol.unique())), 40, replace=False)
    small = small[small.Symbol.isin(keep)].reset_index(drop=True)
    o2 = BENCH["finra_small"]
    o2.mkdir(parents=True, exist_ok=True)
    small.to_parquet(o2 / "shvol.parquet", index=False)
    _file_log(small).to_parquet(o2 / "file_log.parquet", index=False)
    (o2 / "defects.json").write_text(json.dumps({"n_rows": int(len(small)), "symbols": 40, "years": 1,
                                                 "expect": {"cross_sectional_breadth": "weak",
                                                            "history_depth": "weak"}}, indent=1))
    return gold


# ---------------------------------------------------------------- scoring
ISO = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def _text(r) -> str:
    return " ".join([o.claim for o in r.observations] + r.warnings + r.open_questions +
                    [x for p in r.preprocessing for x in p.observations + p.risks]).lower()


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                        "september", "october", "november", "december"], 1)}
WORDS = re.compile(r"\b(" + "|".join(k[:3] for k in MONTHS) + r")[a-z]*\.?\s+(\d{4})\b")


def _dates_in(text: str) -> list[tuple[pd.Timestamp, bool]]:
    """Dates written as 2024-02-15 / 2024-02 / February 2024 / Feb 2024; the flag says whether a day was given."""
    out = []
    for y, m, d in ISO.findall(text):
        try:
            out.append((pd.Timestamp(f"{y}-{m}-{d or '15'}"), bool(d)))
        except ValueError:
            pass
    for mon, y in WORDS.findall(text):
        out.append((pd.Timestamp(f"{y}-{[i for k, i in MONTHS.items() if k.startswith(mon)][0]:02d}-15"), False))
    return out


def _near(text: str, day: str, days: int) -> bool:
    d0 = pd.Timestamp(day)
    return any(abs((t - d0).days) <= days + (0 if exact else 15) for t, exact in _dates_in(text))


def score_research(card, which: str = "finra_defects") -> dict:
    """Defect detection and recipe correctness (finra_defects) or the viability judgement (finra_small)."""
    from quantaccelerator.tools.clean import run_recipe
    r = card.research
    if r is None:
        return {"status": "no research findings"}
    gold = json.loads((BENCH[which] / "defects.json").read_text())
    if which == "finra_small":
        got = {k: getattr(r.capability, k).rating for k in gold["expect"]}
        return {"capability": got, "correct": {k: got[k] == v for k, v in gold["expect"].items()}}
    t = _text(r)
    has = lambda *ws: any(w in t for w in ws)
    det = {"D1": has("duplicate"),
           "D2": _near(t, "2023-06-01", 15) and has("break", "jump", "shift", "scale", "unit", "level", "increase",
                                                    "x100", "100x", "hundred"),
           "D3": has("zero") and _near(t, "2022-09-15", 20),
           "D4": _near(t, "2024-02-15", 20) and has("coverage", "symbol", "entit", "drop", "fewer", "missing",
                                                    "absent", "hole", "gap"),
           "D5": has("exceed", "greater than", "larger than", "more than", "above") and has("shortvolume",
                                                                                          "short volume")}
    # apply the proposed recipe to the corrupted table
    sv = pd.read_parquet(BENCH[which] / "shvol.parquet")
    steps = [s for s in r.cleaning.steps if s.table == "shvol"]
    rec = r.cleaning.model_copy(update={"steps": steps})
    out, effects = run_recipe(rec, {"shvol": sv}) if steps else ({"shvol": sv}, [])
    after = out["shvol"]
    d5 = pd.DataFrame(gold["defects"]["D5"]["keys"], columns=["Date", "Symbol"])
    flags = [c for c in after.columns if after[c].dtype == bool]
    a = after.assign(Date=after.Date.astype(str))
    hit = a.merge(d5.drop_duplicates(), on=["Date", "Symbol"])
    d5_handled = (len(hit) == 0 or bool(flags and hit[flags].any(axis=1).all())) and len(d5) > 0
    dropped = len(sv) - len(after)
    fixes = {"D1": not after.duplicated().any(),
             "D5": d5_handled or not a.merge(d5, on=["Date", "Symbol"]).shape[0]}
    legit_drops = gold["defects"]["D1"]["rows"] + gold["defects"]["D5"]["rows"]
    return {"detected": det, "n_detected": sum(det.values()), "fixed": fixes,
            "rows_dropped": int(dropped), "rows_dropped_beyond_defects": int(max(0, dropped - legit_drops)),
            "recipe": [s.op for s in r.cleaning.steps], "effects": effects}


if __name__ == "__main__":
    g = build()
    print(json.dumps({k: {x: y for x, y in v.items() if x != "keys"} for k, v in g["defects"].items()}, indent=1))
    print("rows", g["n_rows"])
