"""Idea-translation benchmark for the Research EDA Agent, phase A (synthetic injected structure gives exact ground
truth).

finra_eda: the real FINRA short-volume table (universe and years of the FINRA research panel) with three vendor fields
added, each with a known structure:
  vendor_mentions   a count that scales with trading activity (about 2 per 10,000 shares of average daily volume, times
                    lognormal noise): raw, it ranks names by liquidity; per unit of activity it does not depend on it
  vendor_sentiment  a score with a sector-specific offset (measurement bias by sector), unrelated to liquidity or size
  vendor_quality    a persistent score (AR(1) per symbol, phi 0.98) independent of liquidity, size and sector

The researcher gives four ideas in plain words (IDEAS). Each has reference constructions, so a translation is exactly
checkable: the agent's candidate for the idea must track a reference (mean same-date rank correlation >= 0.9), and where
the data carries a planted bias the candidate must be free of it, or, for the clean field, must not be neutralized. The
agent must also explore at least two ideas of its own. The scorer recomputes every candidate from its definition on the
panel; the agent never sees this file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantaccelerator.ingest.finra import FINRA_DIR
from quantaccelerator.paths import INTERIM, ROOT

BENCH = ROOT / "data" / "benchmark" / "finra_eda"
SEED = 20261002
SECTOR_OFFSET = {"Technology": 0.8, "Health care": 0.5, "Telecom & media": 0.3, "Consumer": 0.2, "Industrials": 0.0,
                 "Other": 0.0, "Financials": -0.3, "Materials": -0.4, "Utilities": -0.6, "Energy": -0.8}
PLANTED = {"vendor_mentions": "liquidity", "vendor_sentiment": "sector", "vendor_quality": "none"}
LIQ_SIZE = {"adv_proxy", "size_float", "size_assets"}


def build() -> dict:
    from quantaccelerator.datasets import get
    from quantaccelerator.ingest.sec_reference import REF_DIR
    from quantaccelerator.tools.panel import universe
    spec = get("finra").panel
    rng = np.random.default_rng(SEED)
    raw = pd.read_parquet(FINRA_DIR / "shvol.parquet")
    # the universe as the panel will form it (formation-window median volume), plus a margin for the warm-up
    view_like = raw.rename(columns={"Symbol": "entity"}).assign(decision_time=raw.Date)
    ents = universe(view_like, spec)
    sv = raw[raw.Symbol.isin(ents) & (raw.Date >= spec["formation"][0])].sort_values(["Symbol", "Date"])
    sv = sv.reset_index(drop=True)
    sym = pd.read_parquet(REF_DIR / "symbols.parquet").drop_duplicates("Symbol").set_index("Symbol").sector
    sector = sv.Symbol.map(sym).fillna("Unknown")
    adv = sv.groupby("Symbol").TotalVolume.transform(lambda s: s.rolling(20, min_periods=1).mean())
    n = len(sv)
    # V1: a count that scales with activity
    sv["vendor_mentions"] = np.round(adv * 2e-4 * np.exp(rng.normal(0, 0.6, n))).astype("int64")
    # V2: sector offset (measurement bias) + a persistent symbol effect + noise
    ent_eff = pd.Series(rng.normal(0, 0.3, len(ents)), index=ents)
    sv["vendor_sentiment"] = (sector.map(SECTOR_OFFSET).fillna(0.0) + sv.Symbol.map(ent_eff)
                              + rng.normal(0, 1.0, n)).round(4)
    # V3: a persistent AR(1) per symbol, stationary variance 1, independent of everything else
    phi, q = 0.98, np.empty(n)
    eps = rng.normal(0, np.sqrt(1 - phi ** 2), n)
    first = (sv.Symbol != sv.Symbol.shift()).to_numpy()
    start = rng.normal(0, 1, n)
    for i in range(n):
        q[i] = start[i] if first[i] else phi * q[i - 1] + eps[i]
    sv["vendor_quality"] = q.round(4)
    BENCH.mkdir(parents=True, exist_ok=True)
    sv = sv.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    sv.to_parquet(BENCH / "shvol.parquet", index=False)
    log = pd.read_parquet(FINRA_DIR / "file_log.parquet")
    log[log.file_date >= spec["formation"][0]].to_parquet(BENCH / "file_log.parquet", index=False)
    gold = {"seed": SEED, "rows": int(len(sv)), "symbols": int(sv.Symbol.nunique()), "planted": PLANTED,
            "sector_offset": SECTOR_OFFSET, "V1_mentions_per_share_of_adv": 2e-4, "V3_phi": phi}
    (BENCH / "gold.json").write_text(json.dumps(gold, indent=1))
    return gold


# ---------------------------------------------------------------- the researcher's ideas and their references
IDEAS = {
    "I1": "Abnormal short selling: how unusual today's share of trading volume sold short is for this stock, compared "
          "with its own previous 20 trading days.",
    "I2": "Attention: how much vendor mention activity a stock gets for its level of trading activity.",
    "I3": "Sentiment relative to other companies in the same sector.",
    "I4": "The vendor's quality score, as a slow-moving characteristic of the company.",
}
S = lambda **kw: {"name": "x", "note": "", **kw}
REFERENCES = {  # each: a list of alternative chains of definitions; the last definition of a chain is the reference
    "I1": [[S(name="sr", op="ratio", inputs=["ShortVolume", "TotalVolume"]), S(name="x", op="surprise", inputs=["sr"], window=20)]],
    "I2": [[S(op="ratio", inputs=["vendor_mentions", "adv_proxy"])], [S(op="ratio", inputs=["vendor_mentions", "TotalVolume"])]],
    "I3": [[S(op="peer_relative", inputs=["vendor_sentiment"], group="sector")],
           [S(op="residualize", inputs=["vendor_sentiment"], group="sector")]],
    "I4": [[S(op="field", inputs=["vendor_quality"])]],
}
FAITHFUL = 0.9


def _panel():
    """The benchmark's research panel, rebuilt deterministically (gold rule)."""
    from quantaccelerator.agents.orchestrator import oracle_contract
    from quantaccelerator.tools.panel import build_panel
    from quantaccelerator.tools.pit_view import build_pit_view
    return build_panel(build_pit_view(oracle_contract()))


def rank_corr(panel: pd.DataFrame, x: pd.Series, y: pd.Series) -> float:
    """Mean over dates of the cross-sectional Spearman correlation of two features."""
    ok = x.notna() & y.notna()
    d = panel.loc[ok, ["date"]].assign(a=x[ok].values, b=y[ok].values)
    d["ra"], d["rb"] = d.groupby("date").a.rank(), d.groupby("date").b.rank()
    ca = d.ra - d.groupby("date").ra.transform("mean")
    cb = d.rb - d.groupby("date").rb.transform("mean")
    num = (ca * cb).groupby(d.date).sum()
    den = np.sqrt((ca ** 2).groupby(d.date).sum() * (cb ** 2).groupby(d.date).sum())
    return float((num / den.where(den > 0)).mean())


def score(run_dir: Path) -> dict:
    from quantaccelerator.state.schemas import CandidateFeatureSet, FeatureSpec
    from quantaccelerator.tools.features import NEUTRALIZING, MODERATE_ETA, MODERATE_RHO, exposure_stats, lineage, materialize
    run_dir = Path(run_dir)
    fs = CandidateFeatureSet.model_validate_json((run_dir / "state" / "candidate_feature_set.json").read_text())
    panel, _ = _panel()
    specs = {s_.name: s_ for s_ in fs.features}
    norm = lambda t: " ".join((t or "").lower().split())
    idea_of = {k: next((i.name for i in fs.ideas if i.source == "researcher" and norm(i.researcher_text) == norm(t)), None)
               for k, t in IDEAS.items()}
    approved = set((fs.idea_gate.approved_ideas or []) if fs.idea_gate else [])
    cands = {c.name: c for c in fs.candidates}
    vals = materialize(fs.features, list(cands), panel) if cands else pd.DataFrame()
    series = {n: pd.Series(vals[n].values, index=panel.index) for n in cands}
    refs = {}
    for k, chains in REFERENCES.items():
        refs[k] = []
        for chain in chains:
            sp = [FeatureSpec(**c) for c in chain]
            refs[k].append(pd.Series(materialize(sp, [sp[-1].name], panel)[sp[-1].name].values, index=panel.index))
    dep = lambda x, by: exposure_stats(panel, x, by)
    out = {"run_dir": str(run_dir), "n_ideas": len(fs.ideas), "n_candidates": len(cands),
           "ideas": {i.name: {"source": i.source, "approved": i.name in approved,
                              "candidates": [c.name for c in fs.candidates if c.idea == i.name]} for i in fs.ideas},
           "translation": {}}
    for k in IDEAS:
        name = idea_of[k]
        mine = [n for n, c in cands.items() if name and c.idea == name]
        corr = {n: round(max(rank_corr(panel, series[n], r) for r in refs[k]), 3) for n in mine}
        best = max(corr, key=corr.get) if corr else None
        rec = {"idea": name, "candidates": mine, "rank_corr_with_reference": corr, "best": best,
               "faithful": bool(best and corr[best] >= FAITHFUL)}
        if best and k == "I2":
            v = dep(series[best], "adv_proxy").get("rank_corr_mean") or 0
            rec["adv_dependence"], rec["scale_free"] = round(v, 3), abs(v) < MODERATE_RHO
        if best and k == "I3":
            v = dep(series[best], "sector").get("eta_squared_mean") or 0
            rec["sector_dependence"], rec["sector_neutral"] = round(v, 4), v < MODERATE_ETA
        if k == "I4":
            rec["not_neutralized"] = bool(mine) and not any(
                specs[f].op in NEUTRALIZING for n in mine for f in lineage(n, specs) if f in specs)
        out["translation"][k] = rec
    own = [i for i in fs.ideas if i.source == "agent" and i.name in approved and out["ideas"][i.name]["candidates"]]
    t = out["translation"]
    out["passed"] = {"I1_faithful": t["I1"]["faithful"], "I2_faithful": t["I2"]["faithful"],
                     "I2_scale_free": bool(t["I2"].get("scale_free")), "I3_faithful": t["I3"]["faithful"],
                     "I3_sector_neutral": bool(t["I3"].get("sector_neutral")), "I4_faithful": t["I4"]["faithful"],
                     "I4_not_neutralized": bool(t["I4"].get("not_neutralized")), "own_ideas_explored": len(own) >= 2}
    out["own_ideas"] = [i.name for i in own]
    out["score"] = f"{sum(out['passed'].values())}/{len(out['passed'])}"
    (run_dir / "score_eda.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import os
    import sys
    if sys.argv[1:2] == ["build"]:
        print(build())
    else:
        os.environ.setdefault("QA_DATASET", "finra_eda")
        r = score(Path(sys.argv[1]))
        print(json.dumps({k: v for k, v in r.items() if k not in ("ideas",)}, indent=1, default=str))
