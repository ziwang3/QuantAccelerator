"""The Critic's attacks, executed: each Critique maps to one deterministic test on the discovery segment.

Every validated finding first faces a standard robustness battery (no illiquid third, no penny stocks, each half of
discovery). The Critic adds its own attacks from the findings' outputs only (no data tools, no author reasoning), each
able to falsify the threat it names; the orchestrator runs each one with `critic_test` (logged, counted in the trial ledger like any other test) and judges it by a fixed
rule, so the verdict is the data's, not the Critic's:
  survived      same sign, at least half of the claimed IC retained, and |t| >= 2
  refuted       the sign flips with |t| >= 2, or less than a quarter of the IC is retained
  inconclusive  otherwise (e.g. the sign holds but a small subsample leaves |t| < 2)
A smaller subsample alone is therefore never a refutation; losing the effect is.
"""
import numpy as np
import pandas as pd

from quantaccelerator.returns import core, vault
from quantaccelerator.tools.predictive import _feature, _horizon, _ledger, _signal
from quantaccelerator.tools.registry import tool

SURVIVE_RETENTION, REFUTE_RETENTION, T_BAR = 0.5, 0.25, 2.0
ALL_CONTROLS = ("size", "reversal", "momentum", "beta", "dollar_adv", "sector")
MIN_CRITIQUES, MIN_THREAT_TYPES = 2, 2
# a test falsifies a threat only if it removes what the threat says drives the effect
COHERENT = {"omitted_characteristic": {("add_controls", None)},
            "small_illiquid_names": {("restrict_universe", "drop_illiquid_tercile"), ("restrict_universe", "large_caps")},
            "microstructure": {("restrict_universe", "price_ge_5"), ("restrict_universe", "drop_illiquid_tercile")},
            "sector_concentration": {("restrict_universe", "exclude_sector")},
            "time_instability": {("subperiod", None)}}


def standard_battery(discovery: tuple) -> list:
    """Run on every validated finding, whatever the Critic writes: the replication literature's default checks
    (no micro-cap / illiquid names, no penny stocks, each half of the sample)."""
    from quantaccelerator.state.schemas import Critique, CriticTest
    a, b = map(pd.Timestamp, discovery)
    mid = (a + (b - a) / 2).normalize()
    halves = [(str(a.date()), str(mid.date())), (str((mid + pd.Timedelta(days=1)).date()), str(b.date()))]
    S = lambda tt, threat, test: Critique(target=-1, threat_type=tt, threat=threat, test=test,
                                          if_spurious="the effect is gone once these observations are removed")
    return [S("small_illiquid_names", "the effect lives in the least liquid names",
              CriticTest(kind="restrict_universe", universe="drop_illiquid_tercile")),
            S("microstructure", "the effect is a low-price (penny stock) artifact",
              CriticTest(kind="restrict_universe", universe="price_ge_5")),
            *[S("time_instability", f"one half of the sample drives the effect ({p[0]} to {p[1]} without it)",
                CriticTest(kind="subperiod", period=p)) for p in halves]]


def _sig(t) -> tuple:
    return (t.kind, t.universe, t.sector, tuple(sorted(t.controls)), tuple(t.period or ()))


def _tercile(d: pd.DataFrame, col: str) -> pd.Series:
    r = d.groupby("date")[col].rank(pct=True)
    return pd.cut(r, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"]).astype(str).where(r.notna())


def subset(d: pd.DataFrame, t) -> pd.DataFrame:
    """The rows a restrict_universe / subperiod test keeps."""
    if t.kind == "subperiod":
        a, b = map(pd.Timestamp, t.period)
        return d[(d.date >= a) & (d.date <= b)]
    if t.kind != "restrict_universe":
        return d
    u = t.universe
    if u == "drop_illiquid_tercile":
        return d[(_tercile(d, "dollar_adv") != "low") & d.dollar_adv.notna()]
    if u == "price_ge_5":
        return d[d.price >= 5]
    if u == "large_caps":
        return d[_tercile(d, "size") == "high"]
    if u == "exclude_sector":
        return d[d.sector != t.sector]
    raise ValueError(f"unknown universe {u!r}")


@tool("critic_test", "Run one Critic attack on a finding (orchestrator): the finding's IC at its horizon and controls, "
      "with controls added, on a restricted universe, or in a subperiod (discovery segment).",
      {"type": "object", "properties": {"feature": {"type": "string"}, "horizon": {"type": "integer"},
                                        "controls": {"type": "array", "items": {"type": "string"}},
                                        "test": {"type": "object"}}, "required": ["feature", "horizon", "test"]})
def critic_test(feature: str, horizon: int, test: dict, controls: list[str] | None = None):
    from quantaccelerator.state.schemas import CriticTest
    t = CriticTest.model_validate(test)
    f, h = _feature(feature), _horizon(horizon)
    ctrl = sorted(set(controls or []) | (set(t.controls) if t.kind == "add_controls" else set()))
    d = subset(vault.frame(), t)
    d, col = _signal(d, f, ctrl)
    ic = core.daily_ic(d, col, f"fwd_{h}")
    st = core.ic_stats(ic, h)
    led = _ledger("critic_test", [f"critic|{f}|{h}|{','.join(ctrl)}|{t.model_dump_json()}"])
    names = int(d.loc[d[col].notna() & d[f"fwd_{h}"].notna(), "entity"].nunique()) if len(d) else 0
    return {"feature": f, "horizon": h, "controls": ctrl, "test": t.model_dump(mode="json"), "entities": names,
            **{k: st.get(k) for k in ("mean_ic", "t_nw", "n_dates", "mean_ic_by_year")}, **led}


def judge(claim_ic: float, attacked: dict) -> tuple[str, float | None]:
    ic, t = attacked.get("mean_ic"), attacked.get("t_nw")
    if ic is None or t is None or not claim_ic or np.isnan(t):
        return "inconclusive", None
    ret = ic / claim_ic
    if (ret < 0 and abs(t) >= T_BAR) or ret < REFUTE_RETENTION:
        return "refuted", round(ret, 3)
    if ret >= SURVIVE_RETENTION and abs(t) >= T_BAR:
        return "survived", round(ret, 3)
    return "inconclusive", round(ret, 3)


def critique_errors(c, targets: dict[int, dict], sectors: list[str], discovery: tuple) -> list[str]:
    """Why one critique cannot run as written (empty: it can)."""
    if c.target not in targets:
        return [f"target {c.target} is not one of the findings {sorted(targets)}"]
    t = c.test
    if (t.kind, t.universe if t.kind == "restrict_universe" else None) not in COHERENT[c.threat_type]:
        ok = sorted(f"{k}" + (f" {u}" if u else "") for k, u in COHERENT[c.threat_type])
        return [f"a {c.threat_type} threat is falsified by removing what it says drives the effect: use {ok}, not "
                f"{t.kind} {t.universe or ''}".rstrip()]
    if _sig(t) in {_sig(x.test) for x in standard_battery(discovery)}:
        return [f"{t.kind} {t.universe or t.period} is already in the standard battery run on every finding"]
    if t.kind == "add_controls":
        have = set(targets[c.target]["controls"])
        if not set(t.controls) - have:
            left = sorted(set(ALL_CONTROLS) - have)
            return [f"finding {c.target} already controls for {t.controls}: "
                    + (f"add one it lacks ({left})" if left else "it has every control, so attack it with "
                       "restrict_universe or subperiod instead")]
    elif t.kind == "restrict_universe":
        if t.universe is None:
            return ["restrict_universe needs universe"]
        if t.universe == "exclude_sector" and t.sector not in sectors:
            return [f"sector must be one of {sectors}"]
    elif t.kind == "subperiod":
        a, b = (t.period or ("", ""))
        if not t.period or not (discovery[0] <= a < b <= discovery[1]):
            return [f"subperiod needs a [start, end] inside discovery {list(discovery)}"]
    return []


def split_draft(draft, targets, sectors, discovery) -> tuple[list, list]:
    """(runnable critiques, [(critique, reason)] of those that cannot run as written)."""
    ok, bad = [], []
    for c in draft.critiques:
        e = critique_errors(c, targets, sectors, discovery)
        (bad.append((c, e[0])) if e else ok.append(c))
    return ok, bad


def legal_attacks(controls: list[str]) -> list[str]:
    """The attacks a finding can still face beyond the battery, one line per threat type."""
    left = [c for c in ALL_CONTROLS if c not in controls]
    out = [f"omitted_characteristic: add_controls with one of {left}"] if left else []
    return out + ["small_illiquid_names: restrict_universe large_caps",
                  "sector_concentration: restrict_universe exclude_sector <the sector its IC concentrates in>",
                  "time_instability: subperiod [start, end] inside discovery, not the battery's halves"]


def check_critic_draft(draft, targets: dict[int, dict], sectors: list[str], discovery: tuple) -> list[str]:
    """Invalid critiques are dropped (and reported), not fatal; the draft is rejected only when the runnable ones leave
    a finding with fewer than two attacks from two threat types beyond the standard battery."""
    ok, bad = split_draft(draft, targets, sectors, discovery)
    errs = []
    for k in targets:
        mine = [c for c in ok if c.target == k]
        if len(mine) < MIN_CRITIQUES or len({c.threat_type for c in mine}) < MIN_THREAT_TYPES:
            errs.append(f"finding {k} needs at least {MIN_CRITIQUES} runnable attacks from at least {MIN_THREAT_TYPES} "
                        f"different threat types beyond the standard battery (it has {len(mine)}: "
                        f"{sorted({c.threat_type for c in mine})}); legal for it: "
                        + "; ".join(legal_attacks(targets[k]["controls"])))
    if errs and bad:
        errs.append("not runnable as written: " + "; ".join(f"[finding {c.target}, {c.threat_type}] {r}"
                                                            for c, r in bad[:6]))
    return errs
