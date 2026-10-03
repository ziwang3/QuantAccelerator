"""Cost and capacity of a finding: does an information coefficient survive trading?

The portfolio is the plainest one a finding implies: each rebalance date (every h sessions), equal weight long the
quantile the finding's sign favours and short the opposite one, held from the open of the date to the close h-1
sessions later (the forward-return definition). Costs per name and trade:
  half-spread  the quoted closing spread when the data has bid/ask; otherwise Roll's estimator from daily returns,
               which is noisy and biased up for liquid names (flagged in the output: rely on the break-even spread),
               point-in-time through d-1; names without an estimate get the median of their liquidity tercile; the
               result is clipped to [1, 200] bp
  impact       square-root law: daily volatility * sqrt(trade value / average daily dollar volume)
Two trading assumptions bracket the truth:
  round_trip   every position is opened at the open and closed at the close of the period (what the forward return
               measures; conservative)
  changes_only only names entering or leaving a leg are traded (as if held across periods; optimistic)
Capacity: the capital per leg at which impact makes the net return per period zero (round_trip).
"""
import numpy as np
import pandas as pd

MIN_HALF, MAX_HALF = 1e-4, 2e-2
PERIODS_PER_YEAR = 252


def _legs(d: pd.DataFrame, sig: str, sign: int, n: int) -> pd.DataFrame:
    d = d.dropna(subset=[sig]).copy()
    d["q"] = np.ceil(d.groupby("date")[sig].rank(pct=True, method="first") * n).clip(1, n).astype(int)
    top, bot = (n, 1) if sign > 0 else (1, n)
    d["leg"] = np.where(d.q == top, 1, np.where(d.q == bot, -1, 0))
    return d[d.leg != 0]


def half_spread(d: pd.DataFrame) -> pd.Series:
    hs = d["half_spread"].astype(float)
    terc = d.groupby("date")["dollar_adv"].rank(pct=True).mul(3).clip(upper=2.999).fillna(1).astype(int)
    fill = hs.groupby([d["date"], terc]).transform("median")
    return hs.fillna(fill).fillna(hs.median()).clip(MIN_HALF, MAX_HALF)


def portfolio(d: pd.DataFrame, sig: str, sign: int, h: int, n: int = 5) -> dict:
    """Gross and net returns of the finding's long-short quantile portfolio, turnover, and capacity."""
    dates = np.sort(d["date"].unique())[::h]                      # non-overlapping rebalances
    L = _legs(d[d["date"].isin(dates)], sig, sign, n)
    L = L.dropna(subset=[f"fwd_{h}"])
    if L.empty or L["date"].nunique() < 10:
        return {"periods": int(L["date"].nunique())}
    L["hs"] = half_spread(L)
    L["adv_usd"] = np.exp(L["dollar_adv"])
    gross = (L.groupby(["date", "leg"])[f"fwd_{h}"].mean().unstack()).dropna()
    gross = gross[1] - gross[-1]
    # round trip: every name, both ends of the period, both legs (cost per unit of capital per leg)
    rt = L.groupby(["date", "leg"])["hs"].mean().unstack().sum(axis=1) * 2
    # changes only: one-way trades of the names that entered or left each leg since the previous rebalance
    members = {k: set(g.entity) for k, g in L.groupby(["date", "leg"])}
    ch = {}
    for i, t in enumerate(sorted(L["date"].unique())):
        c = 0.0
        for leg in (1, -1):
            now = members.get((t, leg), set())
            prev = members.get((sorted(L["date"].unique())[i - 1], leg), set()) if i else set()
            g = L[(L["date"] == t) & (L.leg == leg)].set_index("entity")["hs"]
            changed = (now ^ prev) & now
            c += 2 * g.reindex(list(changed)).sum() / max(len(now), 1)    # enter (and the matching exit later)
        ch[t] = c
    ch = pd.Series(ch).reindex(gross.index)
    turnover = float(pd.Series({t: len((members.get((t, 1), set()) ^ members.get((s, 1), set())) &
                                       members.get((t, 1), set())) / max(len(members.get((t, 1), set())), 1)
                                for s, t in zip(sorted(L["date"].unique())[:-1], sorted(L["date"].unique())[1:])}).mean())
    per_year = PERIODS_PER_YEAR / h
    stats = lambda x: {"mean_bp": float(x.mean() * 1e4), "ann_return": float(x.mean() * per_year),
                       "ann_sharpe": float(x.mean() / x.std() * np.sqrt(per_year)) if x.std() > 0 else None}
    net_rt, net_ch = gross - rt.reindex(gross.index), gross - ch

    def impact(capital_per_leg: float) -> float:
        per_name = capital_per_leg / L.groupby(["date", "leg"]).size().mean()
        return float((L["vol20"].fillna(L["vol20"].median()) * np.sqrt(per_name / L["adv_usd"])).mean()) * 2 * 2

    edge = float(net_rt.mean())
    cap = None
    if edge > 0:
        lo, hi = 1e3, 1e11
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if impact(mid) < edge else (lo, mid)
        cap = float(lo)
    return {"periods": int(len(gross)), "rebalance_every_sessions": h, "names_per_leg": float(
                L.groupby(["date", "leg"]).size().mean()),
            "median_half_spread_bp": float(L["hs"].median() * 1e4), "turnover_long_leg": turnover,
            "spread_source": "quoted" if L.get("spread_quoted", pd.Series(0.0)).fillna(0).mean() > 0.5 else
            "roll (daily-return estimate: noisy and biased up for liquid names; use the break-even spread, or add "
            "CRSP DlyBid/DlyAsk)",
            "gross": stats(gross), "net_round_trip": stats(net_rt), "net_changes_only": stats(net_ch),
            "cost_round_trip_bp": float(rt.mean() * 1e4), "cost_changes_only_bp": float(ch.mean() * 1e4),
            "break_even_half_spread_bp": float(gross.mean() / 4 * 1e4),
            "capacity_per_leg_usd": cap}
