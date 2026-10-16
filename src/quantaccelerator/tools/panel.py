"""The research panel: the one table the Research EDA Agent works on (post-PIT, phase A, return-blind).

It is built deterministically from the PIT research view of a signed contract (quantaccelerator.tools.pit_view), so every value
sits at its decision date (the first session whose open may use it), never earlier. On top of the view it adds
conditioning variables a quant researcher characterizes features against, each point-in-time as well:
- adv_proxy: the entity's average of the profile's scale field (FINRA: TotalVolume, off-exchange share volume) over
  its last `adv_window` records up to and including the decision date;
- size_float / size_assets: the filer's latest public float / total assets (SEC company facts, USD) filed before the
  decision date (used from the next calendar day on);
- sector: a coarse grouping of the filer's SIC code (current, not point-in-time).
The universe is fixed in advance from a formation window that ends before the panel starts (the most active entities
by the scale field), so membership never depends on the panel period. No return, price or P&L column exists here:
phase A of research EDA is return-blind by construction.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from quantaccelerator.datasets import active
from quantaccelerator.paths import INTERIM

REF = INTERIM / "reference"
CONDITIONERS = ["adv_proxy", "size_float", "size_assets", "sector"]
CATEGORICAL = {"sector"}
STATE = {"panel": None, "meta": None}   # the panel of the current session (set by the orchestrator)


def universe(view: pd.DataFrame, spec: dict) -> pd.Index:
    """The `universe_n` most active entities by median scale field in the formation window (before the panel)."""
    lo, hi = pd.Timestamp(spec["formation"][0]), pd.Timestamp(spec["formation"][1])
    f = view[(view.decision_time >= lo) & (view.decision_time <= hi)]
    return f.groupby("entity")[spec["scale_field"]].median().nlargest(spec["universe_n"]).index


def _reference(entities: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    sym = pd.read_parquet(REF / "symbols.parquet")
    sym = sym[sym.Symbol.isin(entities)].drop_duplicates("Symbol")[["Symbol", "cik", "sector"]]
    fun = pd.read_parquet(REF / "fundamentals.parquet")
    fun = fun[fun.cik.isin(sym.cik.unique()) & (fun.val > 0)]
    return sym, fun


def _asof_value(panel: pd.DataFrame, fun: pd.DataFrame, item: str) -> pd.Series:
    """Latest value of `item` filed strictly before each decision date (a filing is usable from the next day)."""
    f = fun[fun.item == item].assign(avail=lambda d: d.filed + pd.Timedelta(days=1))
    # several values can be filed together (e.g. comparative periods): keep the one for the latest period end
    f = f.sort_values(["cik", "avail", "end"]).drop_duplicates(["cik", "avail"], keep="last").sort_values("avail")
    left = panel[["date", "cik"]].reset_index().dropna(subset=["cik"]).sort_values("date")
    left["cik"] = left.cik.astype("int64")
    m = pd.merge_asof(left, f[["cik", "avail", "val"]].astype({"cik": "int64"}), left_on="date", right_on="avail",
                      by="cik", direction="backward")
    return m.set_index("index").val.reindex(panel.index)


def build_panel(view: pd.DataFrame, spec: dict | None = None) -> tuple[pd.DataFrame, dict]:
    spec = spec or active().panel
    if not spec:
        raise NotImplementedError(f"dataset profile {active().name!r} declares no research panel")
    ents = universe(view, spec)
    start, end = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
    v = view[view.entity.isin(ents)].sort_values(["entity", "decision_time"])
    # the scale field's rolling average uses records up to the panel start as warm-up
    adv = v.groupby("entity", sort=False)[spec["scale_field"]].transform(
        lambda s: s.rolling(spec.get("adv_window", 20), min_periods=5).mean())
    v = v.assign(adv_proxy=adv)
    v = v[(v.decision_time >= start) & (v.decision_time <= end)]
    # one record per entity and decision date (two source dates can map to one session; keep the latest)
    v = v.sort_values(["entity", "decision_time", "source_time"]).drop_duplicates(["entity", "decision_time"],
                                                                                 keep="last")
    values = list(active().view["values"])
    panel = v.rename(columns={"decision_time": "date"})[["date", "entity", *values, "quality_flag", "adv_proxy"]]
    panel = panel.reset_index(drop=True)
    sym, fun = _reference(ents)
    panel = panel.merge(sym.rename(columns={"Symbol": "entity"}), on="entity", how="left")
    panel["sector"] = panel.sector.fillna("Unknown")
    panel["size_float"] = _asof_value(panel, fun, "public_float")
    panel["size_assets"] = _asof_value(panel, fun, "assets")
    panel = panel.drop(columns=["cik"]).sort_values(["date", "entity"]).reset_index(drop=True)
    meta = {"universe_n": int(len(ents)), "formation": list(spec["formation"]), "start": str(start.date()),
            "end": str(end.date()), "scale_field": spec["scale_field"], "values": values,
            "conditioners": CONDITIONERS, "n_rows": int(len(panel)), "n_dates": int(panel.date.nunique()),
            "n_entities": int(panel.entity.nunique()), "rule": view.attrs.get("rule"),
            "dataset_id": view.attrs.get("dataset_id")}
    meta["hash"] = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]
    panel.attrs.update(meta)
    return panel, meta


def set_panel(panel: pd.DataFrame, meta: dict) -> None:
    from quantaccelerator.tools import features
    STATE["panel"], STATE["meta"] = panel, meta
    features.CACHE.clear()


def get_panel() -> pd.DataFrame:
    if STATE["panel"] is None:
        raise RuntimeError("no research panel in this session: it is built from the PIT view after the contract is "
                           "signed (orchestrator.build_research_panel)")
    return STATE["panel"]


def describe(panel: pd.DataFrame, meta: dict) -> dict:
    """What the agent needs to know about its panel: fields, conditioners, coverage of each, and caveats."""
    cov = {c: round(float(panel[c].notna().mean() if c not in CATEGORICAL else (panel[c] != "Unknown").mean()), 4)
           for c in CONDITIONERS}
    sectors = panel.drop_duplicates("entity").sector.value_counts()
    return {
        "rows": meta["n_rows"], "dates": meta["n_dates"], "entities": meta["n_entities"],
        "first_date": str(panel.date.min().date()), "last_date": str(panel.date.max().date()),
        "universe": f"the {meta['universe_n']} entities with the highest median {meta['scale_field']} in "
                    f"{meta['formation'][0]}..{meta['formation'][1]} (fixed before the panel starts)",
        "date": "decision date: the first session whose open may use the record under the signed PIT rule",
        "value_fields": meta["values"],
        "conditioners": {
            "adv_proxy": f"average {meta['scale_field']} over the entity's last 20 records (a liquidity proxy; for "
                         "FINRA it counts off-exchange (FINRA-reported) shares only, not dollars)",
            "size_float": "latest public float filed with the SEC before the date (USD, about yearly; a market-cap "
                          "proxy)",
            "size_assets": "latest total assets filed with the SEC before the date (USD, quarterly; a balance-sheet "
                           "size proxy)",
            "sector": "coarse sector from the filer's SIC code (current code, not point-in-time)"},
        "conditioner_coverage_share": cov,
        "entities_per_sector": {k: int(v) for k, v in sectors.items()},
        "other_columns": {"quality_flag": "true when an approved cleaning step flagged the record"},
        "caveats": ["no returns, prices or P&L are available in this phase (return-blind by design)",
                    "entities with sector 'Unknown' or no size have no SEC filer match: ETFs and funds, and tickers "
                    "that changed or were delisted (the SEC ticker list is today's, e.g. a company renamed since the "
                    "formation year keeps its old symbol here and is unmatched)"],
    }
