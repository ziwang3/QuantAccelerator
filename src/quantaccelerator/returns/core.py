"""Forward returns, point-in-time controls and the predictive statistics, as plain functions on DataFrames.

A returns frame has one row per (date, entity) trading day with columns
    r_oc   open-to-close return of that day (PRC / OPENPRC - 1)
    ret    close-to-close total return of that day (dividends and delisting included)
    prc    closing price, vol shares traded, shrout shares outstanding
    mkt    the market's close-to-close return that day (same for every entity)

Execution convention (the PIT contract): a record with decision date d is traded at the open of d. The h-day forward
return from that open is
    R_h(d) = (1 + r_oc[d]) * prod_{t=d+1}^{d+h-1} (1 + ret[t]) - 1        (open of d -> close of d+h-1)
so day d's own close-to-close return, which includes the overnight move before the entry, is never used.
Controls at d use only information up to the close of d-1.
"""
import numpy as np
import pandas as pd

HORIZONS = (1, 2, 5, 10, 20, 40, 60)
CONTROLS = ("size", "reversal", "momentum", "beta", "dollar_adv", "sector")
NUMERIC_CONTROLS = CONTROLS[:-1]
CONTROL_MEANING = {
    "size": "log market cap at the close of d-1",
    "reversal": "return over the 5 sessions d-5..d-1 (short-term reversal)",
    "momentum": "return over sessions d-252..d-21 (12-1 momentum)",
    "beta": "252-session beta to the market return, through d-1",
    "dollar_adv": "log average dollar volume over the 20 sessions d-20..d-1 (liquidity)",
    "sector": "coarse sector of the research panel (group means)",
}
MIN_NAMES = 20  # a date needs at least this many entities with both values to give an IC


def wide(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="entity", values=col).sort_index()


def _after_last(ret: pd.DataFrame) -> pd.DataFrame:
    """True after an entity's last valid return (delisted or out of the data): the position is in cash from then on."""
    last = ret.notna()[::-1].cummax()[::-1]
    return ~last


def forward_returns(frame: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """date, entity, fwd_<h> for each horizon (NaN when a window has a missing return or runs past the frame's end)."""
    ret, oc = wide(frame, "ret"), wide(frame, "r_oc").reindex_like(wide(frame, "ret"))
    ret = ret.mask(_after_last(ret), 0.0)          # after delisting the (already compounded) proceeds sit in cash
    lr = np.log1p(ret.clip(lower=-0.999999))       # a -100% day: the window loses everything, later windows survive
    miss = lr.isna().astype(int).cumsum()
    c = lr.fillna(0.0).cumsum()
    out = {}
    n = len(ret)
    for h in horizons:
        if h == 1:
            v = oc
        else:
            end_c, end_m = c.shift(-(h - 1)), miss.shift(-(h - 1))
            ok = (end_m - miss == 0) & end_c.notna()
            v = np.expm1(np.log1p(oc) + (end_c - c)).where(ok)
        if h > n:
            v = v * np.nan
        out[f"fwd_{h}"] = v.stack(future_stack=True) if _stack_kw() else v.stack(dropna=False)
    df = pd.DataFrame(out)
    df.index.names = ["date", "entity"]
    return df.reset_index()


def _stack_kw() -> bool:
    import inspect
    return "future_stack" in inspect.signature(pd.DataFrame.stack).parameters


def controls(frame: pd.DataFrame) -> pd.DataFrame:
    """date, entity, the numeric controls and the previous close (price), each known at the open of the date (built
    from data up to d-1)."""
    ret, prc = wide(frame, "ret"), wide(frame, "prc").abs()
    vol, shr = wide(frame, "vol"), wide(frame, "shrout")
    mkt = frame.drop_duplicates("date").set_index("date").mkt.sort_index().reindex(ret.index)
    lr = np.log1p(ret.clip(lower=-0.999999))
    size = np.log((prc * shr).where(lambda x: x > 0))
    rev = np.expm1(lr.rolling(5, min_periods=4).sum())
    mom = np.expm1(lr.rolling(232, min_periods=150).sum().shift(20))     # d-252..d-21 once shifted by one more below
    m = mkt.to_numpy()[:, None]
    mm = pd.DataFrame(np.repeat(m, ret.shape[1], axis=1), index=ret.index, columns=ret.columns).where(ret.notna())
    cov = (ret * mm).rolling(252, min_periods=120).mean() - ret.rolling(252, min_periods=120).mean() * \
        mm.rolling(252, min_periods=120).mean()
    var = (mm ** 2).rolling(252, min_periods=120).mean() - mm.rolling(252, min_periods=120).mean() ** 2
    beta = cov / var.where(var > 0)
    dadv = np.log((prc * vol).rolling(20, min_periods=10).mean().where(lambda x: x > 0))
    # trading-cost inputs (not controls): daily volatility, and Roll's half-spread from the serial covariance of
    # daily returns, 2 * sqrt(-cov) / 2 when the covariance is negative (else unknown); quoted spreads when present
    vol20 = ret.rolling(20, min_periods=10).std()
    cov = (ret * ret.shift(1)).rolling(60, min_periods=40).mean() - \
        ret.rolling(60, min_periods=40).mean() * ret.shift(1).rolling(60, min_periods=40).mean()
    roll_half = np.sqrt((-cov).clip(lower=0)).where(cov < 0)
    quoted = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    if {"bid", "ask"} <= set(frame.columns):
        bid, ask = wide(frame, "bid"), wide(frame, "ask")
        q = ((ask - bid) / (ask + bid)).where((ask > bid) & (bid > 0))   # quoted half-spread at the close
        qh = q.rolling(20, min_periods=5).median()
        quoted = qh.notna()
        roll_half = qh.fillna(roll_half)
    out = {k: v.shift(1) for k, v in {"size": size, "reversal": rev, "momentum": mom, "beta": beta,
                                      "dollar_adv": dadv, "price": prc, "vol20": vol20,
                                      "half_spread": roll_half, "spread_quoted": quoted.astype(float)}.items()}
    # price, vol20, half_spread, spread_quoted: cost inputs and filters, not controls
    df = pd.DataFrame({k: (v.stack(future_stack=True) if _stack_kw() else v.stack(dropna=False))
                       for k, v in out.items()})
    df.index.names = ["date", "entity"]
    return df.reset_index()


# ---------------------------------------------------------------- statistics
def _centered_rank(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return g.rank(pct=True) - 0.5 * (1 + 1 / g.transform("count"))


def residualize(df: pd.DataFrame, col: str, ctrl: list[str]) -> pd.Series:
    """The feature's same-date rank, net of what the controls' ranks (and sector means) explain, date by date.
    Rows missing a control are dropped (NaN)."""
    num = [c for c in ctrl if c != "sector"]
    need = [col, *num] + (["sector"] if "sector" in ctrl else [])
    d = df[["date", *need]].dropna()
    y = _centered_rank(d, col)
    X = pd.concat([_centered_rank(d, c).rename(c) for c in num], axis=1) if num else pd.DataFrame(index=d.index)
    if "sector" in ctrl:
        X = pd.concat([X, pd.get_dummies(d["sector"], prefix="s", dtype=float)], axis=1)
    else:
        X["const"] = 1.0
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    out = np.full(len(d), np.nan)
    for _, idx in d.groupby("date").indices.items():
        if len(idx) < MIN_NAMES:
            continue
        b, *_ = np.linalg.lstsq(Xv[idx], yv[idx], rcond=None)
        out[idx] = yv[idx] - Xv[idx] @ b
    return pd.Series(out, index=d.index).reindex(df.index)


def daily_ic(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Same-date Spearman rank correlation of x with y, per date (dates with fewer than MIN_NAMES pairs dropped)."""
    d = df[["date", x, y]].dropna()
    n = d.groupby("date")[x].transform("count")
    d = d[n >= MIN_NAMES]
    if d.empty:
        return pd.Series(dtype=float)
    d = d.assign(ra=d.groupby("date")[x].rank(), rb=d.groupby("date")[y].rank())
    ca = d.ra - d.groupby("date").ra.transform("mean")
    cb = d.rb - d.groupby("date").rb.transform("mean")
    num = (ca * cb).groupby(d.date).sum()
    den = np.sqrt((ca ** 2).groupby(d.date).sum() * (cb ** 2).groupby(d.date).sum())
    return (num / den.where(den > 0)).dropna()


def nw_lag(h: int) -> int:
    """Newey-West lag: overlapping h-day returns are MA(h-1); persistent features add serial correlation, so at least 5."""
    return max(5, h)


def nw_t(x: pd.Series, lag: int) -> float:
    """t-statistic of the mean of x with Newey-West (Bartlett) standard errors."""
    v = np.asarray(x, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 10:
        return float("nan")
    e = v - v.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(v.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def ic_stats(ic: pd.Series, h: int) -> dict:
    if ic.empty:
        return {"n_dates": 0}
    by_year = ic.groupby(ic.index.year).mean()
    return {"mean_ic": float(ic.mean()), "ic_vol": float(ic.std()), "t_nw": nw_t(ic, nw_lag(h)),
            "hit_rate": float((ic > 0).mean()), "n_dates": int(len(ic)), "nw_lag": nw_lag(h),
            "first_date": str(ic.index.min().date()), "last_date": str(ic.index.max().date()),
            "mean_ic_by_year": {int(k): float(v) for k, v in by_year.items()}}


def quantile_spread(df: pd.DataFrame, x: str, y: str, n: int, h: int) -> dict:
    """Mean forward return of each same-date quantile of x, in excess of the date's average, and the top-minus-bottom
    spread with its Newey-West t."""
    d = df[["date", x, y]].dropna()
    cnt = d.groupby("date")[x].transform("count")
    d = d[cnt >= max(MIN_NAMES, 2 * n)].copy()
    d["q"] = np.minimum((d.groupby("date")[x].rank(pct=True, method="first") * n).apply(np.ceil), n).astype(int)
    d["ex"] = d[y] - d.groupby("date")[y].transform("mean")
    per = d.groupby(["date", "q"]).ex.mean().unstack()
    means = per.mean()
    spread = per[n] - per[1]
    mono = float(np.corrcoef(means.rank().to_numpy(), np.arange(1, len(means) + 1))[0, 1])  # Spearman, no scipy
    return {"quantile_mean_excess_return": {f"Q{int(k)}": float(v) for k, v in means.items()},
            "top_minus_bottom": float(spread.mean()), "t_nw": nw_t(spread, nw_lag(h)), "monotonicity": mono,
            "n_dates": int(len(spread.dropna()))}


def hurdle(n_trials: int) -> float:
    """t a discovery must clear given the number of distinct tests run on this feature set: max(3, sqrt(2 ln N))
    (Harvey-Liu-Zhu's 3, and the expected maximum |t| of N independent null tests)."""
    return float(max(3.0, np.sqrt(2 * np.log(max(n_trials, 1)))))
