"""Predictive EDA benchmark (phase B): returns with planted structure give an exact answer key.

finra_pred: the finra_eda research panel (real FINRA short volume plus the three planted vendor fields) and a fixed,
oracle-frozen feature set (the researcher ideas' reference constructions, the raw mention count and one null). The
returns are synthetic, on the NYSE calendar from 2021, with these relations planted into each day's open-to-close
move (ic ~ the mean same-date rank IC they produce at the stated horizon):
  P1 abn_short     short-lived: -a1 * z(abn_short at d) on sessions d..d+4    (ic(5) ~ -0.025, fades by 20)
  P2 quality       slow: +a2 * z(quality) every session                        (ic(20) ~ +0.045, grows with h)
  P3 liquidity     a premium on illiquidity: -a3 * z(dollar_adv control at d)  (ic(20) ~ -0.04 for dollar volume)
                   -> the raw mention count (which scales with trading activity) 'predicts' only through it;
                      attention (mentions per unit of activity) carries nothing
  P4 sector_sent   +a4 * z(sector_sent at d) on sessions d..d+19, for d before 2024; -a4 from 2024 (a regime
                   break: ic(20) ~ +0.03 in discovery, reversed in validation and the locked test)
  P5 exempt_share  -a5 * z(exempt_share at d) on session d, ONLY in the least liquid third (same-date dollar_adv
                   tercile): significant on the full sample, survives a linear liquidity control and validation, and is
                   gone once the illiquid names are dropped: the Critic's target (universe restriction)
  P6 short_ratio   +a6 * z(short_ratio at d) on sessions d..d+19, ONLY in Technology: survives the standard battery
                   (it spans liquidity, price and time) and is gone once Technology is excluded: the Critic agent's own
                   target (it sees each finding's IC by sector and must aim exclude_sector at the right one)
  vol_surprise     nothing (a null)
The agent never sees this file. Scoring reads the PredictiveReport and the vault's access log.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantaccelerator.paths import ROOT

BENCH = ROOT / "data" / "benchmark" / "finra_pred"
SEED = 20261006
START = "2021-01-04"
END = "2025-12-31"
FLIP = "2024-01-01"
SPLIT = {"discovery": ("2022-01-03", "2023-11-30"), "validation": ("2024-01-02", "2024-11-29"),
         "locked_test": ("2025-01-02", "2025-12-31"), "embargo_sessions": 20}
IDIO = 0.0161            # daily idiosyncratic sd (overnight 0.008, intraday 0.014)
A1 = 0.025 * IDIO / np.sqrt(5)
A2 = 0.045 * IDIO / np.sqrt(20)
A3 = 0.040 * IDIO / np.sqrt(20)
A4 = 0.014 * IDIO / np.sqrt(20)   # spread over 20 sessions on a partly persistent feature
A5 = 0.0005                        # same-day, least liquid third only
A6 = 0.00008                       # 20 sessions, Technology only
P6_SECTOR = "Technology"


# ---------------------------------------------------------------- the oracle feature set
def oracle_features():
    from quantaccelerator.state.schemas import CandidateFeature, FeatureIdea, FeatureSpec
    S = lambda name, op, inputs, **kw: FeatureSpec(name=name, op=op, inputs=inputs, **kw)
    specs = [S("short_ratio", "ratio", ["ShortVolume", "TotalVolume"]),
             S("abn_short", "surprise", ["short_ratio"], window=20),
             S("attention", "ratio", ["vendor_mentions", "adv_proxy"]),
             S("mentions_raw", "field", ["vendor_mentions"]),
             S("sector_sent", "peer_relative", ["vendor_sentiment"], group="sector"),
             S("quality", "field", ["vendor_quality"]),
             S("vol_surprise", "surprise", ["TotalVolume"], window=20),
             S("exempt_share", "ratio", ["ShortExemptVolume", "ShortVolume"])]
    sp = {s.name: s for s in specs}
    I = lambda name, idea, mech, direction, feats, source="researcher": FeatureIdea(
        name=name, idea=idea, mechanism=mech, assumptions=[], expected_direction=direction,
        specs=[sp[f] for f in feats], source=source)
    ideas = [
        I("abnormal_shorting", "How unusual today's share of volume sold short is against the stock's previous 20 days.",
          "Short sellers are often informed: an unusual burst of shorting anticipates bad news that prices in over the "
          "following days.", "higher abnormal shorting, lower returns over the next days", ["short_ratio", "abn_short"]),
        I("attention", "Vendor mention activity per unit of trading activity.",
          "Attention-grabbing stocks attract net buying from attention-driven investors and become temporarily "
          "overpriced, then revert.", "more attention, lower subsequent returns", ["attention"]),
        I("raw_mentions", "The vendor's raw daily mention count.",
          "More mentions mean more attention and attention-driven overpricing.", "more mentions, lower returns",
          ["mentions_raw"], "agent"),
        I("relative_sentiment", "Sentiment relative to other companies in the same sector.",
          "Investors underreact to soft information; firms viewed better than their peers outperform them.",
          "higher relative sentiment, higher returns", ["sector_sent"]),
        I("quality", "The vendor's quality score, as a slow-moving characteristic.",
          "Markets underprice persistent quality; high-quality firms earn higher returns over months.",
          "higher quality, higher returns over weeks to months", ["quality"]),
        I("volume_shock", "How unusual today's trading volume is against the stock's previous 20 days.",
          "High-volume return premium: a volume shock raises visibility and is followed by higher returns.",
          "higher abnormal volume, higher returns over the next weeks", ["vol_surprise"], "agent"),
        I("short_level", "The share of trading volume sold short, as a level.",
          "A persistently high short share marks names that informed sellers consider overpriced, which then drift "
          "down.", "higher short share, lower returns over the next month", ["short_ratio"], "agent"),
        I("exempt_intensity", "The share of short volume marked short exempt.",
          "Short-exempt selling comes from market makers and hedgers absorbing buying pressure; a high share marks "
          "temporary overpricing that reverts.", "higher exempt share, lower returns the same day", ["exempt_share"],
          "agent"),
    ]
    C = lambda name, idea, why, exp=(): CandidateFeature(name=name, idea=idea, rationale=why, known_exposures=list(exp),
                                                         evidence=[])
    cands = [C("abn_short", "abnormal_shorting", "self-relative, so comparable across stocks"),
             C("attention", "attention", "per unit of activity: free of the liquidity scale"),
             C("mentions_raw", "raw_mentions", "the plain count",
               ["strong liquidity dependence: ranks stocks by trading activity (rank corr with adv_proxy about 0.9)"]),
             C("sector_sent", "relative_sentiment", "sector-relative: removes the vendor's sector offset"),
             C("quality", "quality", "a persistent characteristic with no material exposure"),
             C("vol_surprise", "volume_shock", "self-relative volume"),
             C("exempt_share", "exempt_intensity", "a share, so comparable across stocks"),
             C("short_ratio", "short_level", "a share of volume, comparable across stocks")]
    return specs, cands, ideas


# ---------------------------------------------------------------- building the returns
def _z(panel: pd.DataFrame, x: pd.Series) -> pd.Series:
    """Same-date rank scaled to unit variance (0 where missing)."""
    r = x.groupby(panel["date"]).rank(pct=True)
    n = x.groupby(panel["date"]).transform("count")
    return ((r - 0.5 * (1 + 1 / n)) * np.sqrt(12)).fillna(0.0)


def build() -> dict:
    import os
    os.environ["QA_DATASET"] = "finra_pred"
    from quantaccelerator.eval.gold_eda import _panel
    from quantaccelerator.ingest.calendar import trading_days
    from quantaccelerator.tools.features import materialize
    panel, meta = _panel()
    specs, cands, _ = oracle_features()
    F = materialize(specs, [c.name for c in cands], panel)
    days = trading_days()
    days = days[(days >= START) & (days <= END)]
    ents = sorted(panel.entity.unique())
    T, N = len(days), len(ents)
    rng = np.random.default_rng(SEED)
    grid = lambda s: pd.DataFrame({"date": panel.date, "entity": panel.entity, "v": s.values}) \
        .pivot(index="date", columns="entity", values="v").reindex(index=days, columns=ents)
    z = {k: grid(_z(panel, F[k])).fillna(0.0).to_numpy() for k in ("abn_short", "quality", "sector_sent",
                                                                   "exempt_share", "short_ratio")}
    tech = (panel.drop_duplicates("entity").set_index("entity").sector.reindex(ents) == P6_SECTOR).to_numpy()
    # activity: FINRA volume, carried back over the warm-up year, and a fixed initial price per stock
    tv = grid(panel.TotalVolume).ffill().bfill()
    p0 = np.exp(rng.normal(3.3, 0.8, N))
    vol = (tv.to_numpy() * 2.5 * np.exp(rng.normal(0, 0.2, (T, N))))
    m = rng.normal(0.0004, 0.01, T)
    beta = rng.normal(1.0, 0.3, N)
    on = 0.4 * m[:, None] * beta + rng.normal(0, 0.008, (T, N))
    oc = 0.6 * m[:, None] * beta + rng.normal(0, 0.014, (T, N))
    plant = np.zeros((T, N))
    for j in range(5):                                            # P1 on sessions d..d+4
        plant[j:] += -A1 * z["abn_short"][:T - j]
    plant += A2 * z["quality"]                                     # P2 every session
    sign = np.where(days < pd.Timestamp(FLIP), 1.0, -1.0)[:, None]
    for j in range(20):                                           # P4 on sessions d..d+19, sign of d's regime
        plant[j:] += A4 * (sign * z["sector_sent"])[:T - j]
        plant[j:] += A6 * (z["short_ratio"] * tech[None, :])[:T - j]  # P6: Technology only
    # P3 session by session, on exactly what the dollar_adv control measures: the log mean dollar volume of the
    # previous 20 sessions, at the realized prices
    ret, prc, dv = np.empty((T, N)), np.empty((T, N)), np.empty((T, N))
    for t in range(T):
        if t >= 10:
            ldv = np.log(dv[max(0, t - 20):t].mean(0))
            r = pd.Series(ldv).rank(pct=True).to_numpy()
            plant[t] += -A3 * (r - 0.5 * (1 + 1 / N)) * np.sqrt(12)
            plant[t] += -A5 * z["exempt_share"][t] * (r < 1 / 3)          # P5: the least liquid third only
        oc[t] += plant[t]
        ret[t] = (1 + on[t]) * (1 + oc[t]) - 1
        prc[t] = (prc[t - 1] if t else p0) * (1 + ret[t])
        dv[t] = prc[t] * vol[t]
    shrout = np.nanmedian(tv.to_numpy(), axis=0) * 2.5 * np.exp(rng.normal(4.5, 0.7, N))
    stack = lambda a: pd.DataFrame(a, index=days, columns=ents).stack(future_stack=True)
    fr = pd.DataFrame({"r_oc": stack(oc), "ret": stack(ret), "prc": stack(prc), "vol": stack(vol),
                       "shrout": stack(np.repeat(shrout[None, :], T, 0)),
                       "mkt": stack(np.repeat(ret.mean(1)[:, None], N, 1))})
    fr.index.names = ["date", "entity"]
    BENCH.mkdir(parents=True, exist_ok=True)
    fr.reset_index().to_parquet(BENCH / "returns.parquet", index=False)
    gold = {"seed": SEED, "panel_hash": meta["hash"], "entities": N, "sessions": T, "start": START, "end": END,
            "flip": FLIP, "split": SPLIT, "a": {"A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A6": A6},
            "planted": {"abn_short": {"sign": -1, "speed": "fast", "peak_h_max": 5},
                        "quality": {"sign": 1, "speed": "slow", "min_h": 20},
                        "mentions_raw": {"via": "dollar_adv"}, "attention": None,
                        "sector_sent": {"regime_break": FLIP}, "vol_surprise": None,
                        "exempt_share": {"only_in": "least liquid third (dollar_adv)", "horizon": 1},
                        "short_ratio": {"only_in": P6_SECTOR, "sign": 1, "horizon": 20}}}
    (BENCH / "gold.json").write_text(json.dumps(gold, indent=1))
    return gold


def returns_frame() -> pd.DataFrame:
    return pd.read_parquet(BENCH / "returns.parquet")


# ---------------------------------------------------------------- scoring
LIQ = {"dollar_adv", "size"}


def score(run_dir: Path) -> dict:
    from quantaccelerator.returns import vault
    from quantaccelerator.state.schemas import PredictiveReport
    run_dir = Path(run_dir)
    rep = PredictiveReport.model_validate_json((run_dir / "state" / "predictive_report.json").read_text())
    obs = rep.observations
    of = lambda f: [o for o in obs if o.feature == f]
    later = ("validated", "locked_pass", "locked_fail")
    plan = {t.feature: t for t in rep.plan.tests}
    acc = vault.accesses(rep.ledger.get("study"))
    locked = acc[acc.segment == "locked_test"]
    p1 = [o for o in of("abn_short") if o.verdict == "informative" and o.sign == -1 and o.horizon <= 5]
    p2 = [o for o in of("quality") if o.verdict == "informative" and o.sign == 1 and o.horizon >= 20]
    p3 = [o for o in of("mentions_raw") if o.verdict == "explained_by_control" and LIQ & set(o.controls)]
    p3_leak = [o for o in of("mentions_raw") if o.status in later and not LIQ & set(o.controls)]
    passed = {
        "plan_signs": plan.get("abn_short") is not None and plan["abn_short"].expected_sign == -1
        and plan.get("quality") is not None and plan["quality"].expected_sign == 1,
        "P1_short_horizon_found": bool(p1),
        "P1_locked_pass": any(o.status == "locked_pass" for o in p1),
        "P2_slow_found": bool(p2),
        "P2_locked_pass": any(o.status == "locked_pass" for o in p2),
        "P3_liquidity_explained": bool(p3) and not p3_leak,
        "P3_attention_null": not any(o.status in later for o in of("attention")),
        "P4_regime_not_released": not any(o.status in later for o in of("sector_sent")),
        "null_not_validated": not any(o.status in later for o in of("vol_surprise")),
        "locked_test_once": int(locked.granted.sum()) == 1 and int((locked.granted == 0).sum()) == 0,
        "P5_refuted_by_critic": any((o.critique or {}).get("refuted") for o in of("exempt_share")),
        "P5_not_released": not any(o.status.startswith("locked") for o in of("exempt_share")),
        "true_signals_not_refuted": not any((o.critique or {}).get("refuted") for o in of("abn_short") + of("quality")),
        # P6 is the planted POSITIVE Technology-only effect; negative short-horizon short_ratio findings are P1's
        # abnormal shorting showing through (the two are correlated) and are legitimate
        "P6_refuted_by_critic_agent": any(c.outcome == "refuted" and c.source == "critic" and
                                          rep.observations[c.target].feature == "short_ratio" and
                                          rep.observations[c.target].sign > 0 for c in rep.critiques),
        "P6_not_released": not any(o.status.startswith("locked") and o.sign > 0 for o in of("short_ratio")),
    }
    out = {"run_dir": str(run_dir), "n_observations": len(obs), "ledger": {k: v for k, v in rep.ledger.items()
                                                                           if k != "vault_accesses"},
           "observations": [{k: getattr(o, k) for k in ("feature", "horizon", "controls", "verdict", "sign", "status")}
                            | {"t_nw": o.stats.get("t_nw"), "val_t": (o.validation or {}).get("t_nw"),
                               "lock_t": (o.locked_test or {}).get("t_nw")} for o in obs],
           "diagnostic": {"P4_found_in_discovery": any(o.verdict == "informative" for o in of("sector_sent")),
                          "P5_validated": any(o.status in later for o in of("exempt_share")),
                          "P6_validated": any(o.status in later and o.sign > 0 for o in of("short_ratio")),
                          "critiques": [f"{rep.observations[c.target].feature} h={rep.observations[c.target].horizon} "
                                        f"[{c.source}] {c.threat_type} {c.test.kind}/{c.test.universe or c.test.controls or c.test.period}"
                                        f": {c.outcome} (retention {c.retention})" for c in rep.critiques]},
           "passed": passed, "score": f"{sum(passed.values())}/{len(passed)}"}
    (run_dir / "score_pred.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import os
    import sys
    os.environ.setdefault("QA_DATASET", "finra_pred")
    if sys.argv[1:2] == ["build"]:
        print(build())
    else:
        for a in sys.argv[1:]:
            r = score(Path(a))
            print(json.dumps({k: v for k, v in r.items() if k != "observations"}, indent=1, default=str))
