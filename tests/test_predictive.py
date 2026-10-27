"""Research EDA phase B: forward-return alignment, PIT controls, statistics, the returns vault's refusals, the trial
ledger and the observation checks, on synthetic returns with planted structure."""
import numpy as np
import pandas as pd
import pytest

from quantaccelerator.returns import core, vault
from quantaccelerator.state.schemas import (CandidateFeatureSet, EDAWrapUp, FeatureSpec, GateDecision, PlannedTest,
                                            PredictionPlan, SplitSpec)
from quantaccelerator.tools import predictive as PR
from quantaccelerator.tools import registry

DAYS = pd.bdate_range("2022-01-03", periods=420)


def returns_frame(n=120, seed=0, plant=0.004, liq=0.0):
    """`good` predicts the next 5 sessions' returns (open of d onward); `liq` drives a premium on dollar volume;
    `mentions` is liquidity plus noise; `null` is noise."""
    rng = np.random.default_rng(seed)
    ent = [f"E{i:03d}" for i in range(n)]
    T = len(DAYS)
    good = rng.normal(size=(T, n))
    null = rng.normal(size=(T, n))
    ldv = rng.normal(0, 1, n)[None, :] + rng.normal(0, .05, (T, n))       # log dollar volume, persistent
    oc = rng.normal(0, .015, (T, n))
    on = rng.normal(0, .01, (T, n))
    contrib = np.zeros((T, n))
    for j in range(5):                                                   # planted on intraday of d..d+4
        contrib[j:] += plant * good[:T - j] if j else plant * good
    contrib[1:] += -liq * ldv[:-1]
    oc = oc + contrib
    ret = (1 + on) * (1 + oc) - 1
    prc = 20 * np.exp(np.cumsum(np.log1p(ret), axis=0))
    vol = np.exp(ldv) * 1e6 / prc
    mkt = ret.mean(1)
    f = lambda a: pd.DataFrame(a, index=DAYS, columns=ent).stack()
    fr = pd.DataFrame({"r_oc": f(oc), "ret": f(ret), "prc": f(prc), "vol": f(vol), "shrout": f(np.full((T, n), 1e6)),
                       "mkt": f(np.repeat(mkt[:, None], n, 1))})
    fr.index.names = ["date", "entity"]
    feats = pd.DataFrame({"good": f(good), "null": f(null), "mentions": f(ldv + rng.normal(0, .3, (T, n)))})
    feats.index.names = ["date", "entity"]
    return fr.reset_index(), feats.reset_index()


SPLIT = SplitSpec(discovery=("2022-01-03", "2022-12-30"), validation=("2023-02-01", "2023-06-30"),
                  locked_test=("2023-08-01", "2023-08-31"))


def frozen(tmp_path, feats, approve=True):
    specs = [FeatureSpec(name=c, op="field", inputs=[c]) for c in ("good", "null", "mentions")]
    fs = CandidateFeatureSet(panel_hash="p", features=specs, candidates=[], observations=[],
                             wrapup=EDAWrapUp(summary=""))
    g3 = GateDecision(gate="G3", decided_by="oracle", approved=approve, approved_features=["good", "null", "mentions"])
    fs = fs.model_copy(update={"signed_off": g3})
    fs = fs.model_copy(update={"set_hash": vault.feature_set_hash(fs)})
    (tmp_path / "features").mkdir(exist_ok=True)
    feats.to_parquet(tmp_path / "features" / f"{fs.set_hash}.parquet", index=False)
    return fs


def plan_gate(fs):
    plan = PredictionPlan(set_hash=fs.set_hash, split=SPLIT, tests=[
        PlannedTest(feature="good", expected_sign=1, horizons=[5, 20], horizon_rationale="fast"),
        PlannedTest(feature="mentions", expected_sign=1, horizons=[20], horizon_rationale="slow", controls=["dollar_adv"]),
        PlannedTest(feature="null", expected_sign=0, horizons=[5], horizon_rationale="control")])
    return GateDecision(gate="G4", decided_by="oracle", approved=True, approved_plan=plan)


@pytest.fixture
def armed(tmp_path):
    fr, feats = returns_frame(liq=0.0015)
    fs = frozen(tmp_path, feats)
    registry.set_session("t-pred", tmp_path)
    cond = feats[["date", "entity"]].assign(sector=np.where(feats.entity.str[-1].isin(list("01234")), "A", "B"))
    vault.arm(tmp_path, fs, plan_gate(fs), lambda: fr, cond)
    registry.CTX.agent = "predictive_eda"
    yield fs
    vault.STATE.update(set_hash=None, cache={})
    PR.reset()


# ---------------------------------------------------------------- forward returns and controls
def tiny(rets, ocs):
    d = pd.bdate_range("2024-01-01", periods=len(rets))
    return pd.DataFrame({"date": d, "entity": "A", "ret": rets, "r_oc": ocs, "prc": 10.0, "vol": 1.0, "shrout": 1.0,
                         "mkt": 0.0})


def test_forward_return_starts_at_the_open_and_skips_day_d_overnight():
    # day 2 opens 50% higher than day 1's close (overnight), then nothing moves intraday
    fr = tiny([0, 0, .5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])
    fw = core.forward_returns(fr, (1, 2, 3)).set_index("date")
    d2 = fw.index[2]
    assert fw.loc[d2, ["fwd_1", "fwd_2", "fwd_3"]].tolist() == [0, 0, 0]  # entering at day 2's open misses the jump
    assert fw.loc[fw.index[1], "fwd_2"] == pytest.approx(.5)            # entering a day earlier earns it


def test_forward_return_compounds_and_stops_at_the_frame_end():
    fr = tiny([.01, .02, -.01, .03, .00, .01], [.005, .01, -.02, .01, .0, .0])
    fw = core.forward_returns(fr, (1, 3, 5)).set_index("date")
    d0 = fw.index[0]
    assert fw.loc[d0, "fwd_3"] == pytest.approx(1.005 * 1.02 * 0.99 - 1)
    assert fw.loc[d0, "fwd_5"] == pytest.approx(1.005 * 1.02 * .99 * 1.03 * 1.0 - 1)
    assert np.isnan(fw.loc[fw.index[3], "fwd_5"])                         # window runs past the data


def test_after_delisting_the_position_is_cash():
    fr = tiny([.01, -.30, np.nan, np.nan], [.0, -.30, np.nan, np.nan])
    fw = core.forward_returns(fr, (3,)).set_index("date")
    assert fw.loc[fw.index[0], "fwd_3"] == pytest.approx(0.7 - 1)


def test_interior_gap_gives_no_return():
    fr = tiny([.01, np.nan, .02, .01, .0], [.0, .0, .0, .0, .0])
    fw = core.forward_returns(fr, (3,)).set_index("date")
    assert np.isnan(fw.loc[fw.index[0], "fwd_3"])


def test_controls_use_only_data_up_to_the_previous_close():
    fr, _ = returns_frame(n=30)
    c1 = core.controls(fr).set_index(["date", "entity"])
    d = DAYS[300]
    fr2 = fr.copy()
    fr2.loc[fr2.date >= d, ["ret", "prc", "vol"]] *= 3                   # change day d onward
    c2 = core.controls(fr2).set_index(["date", "entity"])
    at = c1.index.get_level_values(0) <= d
    pd.testing.assert_frame_equal(c1[at], c2[at])


def test_newey_west_t_is_smaller_on_autocorrelated_series():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 2000)
    x = np.empty(2000)
    x[0] = e[0]
    for i in range(1, 2000):
        x[i] = .8 * x[i - 1] + e[i]
    x = pd.Series(x + .1)
    naive = x.mean() / (x.std() / np.sqrt(len(x)))
    assert core.nw_t(x, 20) < 0.6 * naive


def test_hurdle_grows_with_trials():
    assert core.hurdle(1) == 3.0 and core.hurdle(10_000) > 4.2


# ---------------------------------------------------------------- vault
def test_vault_refuses_unfrozen_tampered_or_unplanned_sets(tmp_path):
    fr, feats = returns_frame(n=30)
    fs = frozen(tmp_path, feats)
    cond = feats[["date", "entity"]].assign(sector="A")
    with pytest.raises(vault.VaultError, match="G3"):
        vault.arm(tmp_path, fs.model_copy(update={"signed_off": None}), plan_gate(fs), lambda: fr, cond)
    bad = fs.model_copy(update={"features": fs.features[:2]})
    with pytest.raises(vault.VaultError, match="changed after the freeze"):
        vault.arm(tmp_path, bad, plan_gate(fs), lambda: fr, cond)
    with pytest.raises(vault.VaultError, match="G4"):
        vault.arm(tmp_path, fs, None, lambda: fr, cond)
    with pytest.raises(vault.VaultError, match="not armed"):
        vault.frame()


def test_agent_sees_discovery_only_and_locked_test_opens_once(armed):
    d = vault.frame()
    assert d.date.max() <= pd.Timestamp(SPLIT.discovery[1])
    assert d[d.date == d.date.max()].fwd_20.isna().all()                 # windows stop at the segment end
    with pytest.raises(vault.VaultError):
        with vault.segment("validation"):
            pass
    with pytest.raises(vault.VaultError):
        vault.frame("validation")
    registry.CTX.agent = "orchestrator"
    with vault.segment("validation"):
        assert vault.frame().date.min() >= pd.Timestamp(SPLIT.validation[0])
    with pytest.raises(vault.VaultError, match="G5"):
        with vault.segment("locked_test"):
            pass
    vault.release()
    with vault.segment("locked_test"):
        pass
    with pytest.raises(vault.VaultError, match="touched once"):
        with vault.segment("locked_test"):
            pass
    log = vault.accesses(armed.set_hash)
    assert (~log.granted.astype(bool)).sum() >= 4


# ---------------------------------------------------------------- tools and checks
def test_tools_find_the_planted_signal_and_count_distinct_trials(armed):
    r = registry.TOOLS["ic_summary"].fn(feature="good", horizon=5)
    assert r["ok"] and r["result"]["mean_ic"] > 0.05 and r["result"]["t_nw"] > 5
    registry.TOOLS["ic_summary"].fn(feature="good", horizon=5)            # a repeat is not a new trial
    assert vault.n_trials() == 1
    dec = registry.TOOLS["ic_decay"].fn(feature="good")["result"]
    assert dec["peak_horizon"] in (5, 10) and vault.n_trials() == 7
    q = registry.TOOLS["quantile_returns"].fn(feature="good", horizon=5)["result"]
    assert q["monotonicity"] > 0.9 and q["top_minus_bottom"] > 0
    n = registry.TOOLS["ic_summary"].fn(feature="null", horizon=5)["result"]
    assert abs(n["t_nw"]) < 3
    bad = registry.TOOLS["ic_summary"].fn(feature="brand_new", horizon=5)
    assert not bad["ok"] and "frozen" in bad["result"]["error"]


def _rec(**kw):
    return registry.TOOLS["record_research_observation"].fn(**kw)


def test_observation_checks(armed):
    g = registry.TOOLS["ic_summary"].fn(feature="good", horizon=5)
    n = registry.TOOLS["ic_summary"].fn(feature="null", horizon=5)
    t = g["result"]["t_nw"]
    # informative needs the hurdle; the sign must be the measured one
    r = _rec(feature="null", horizon=5, verdict="informative", sign=1, claim="x", evidence=[n["run_id"]])
    assert not r["ok"] and "hurdle" in r["result"]["error"]
    r = _rec(feature="good", horizon=5, verdict="informative", sign=-1, claim="x", evidence=[g["run_id"]])
    assert not r["ok"] and "sign" in r["result"]["error"]
    # numbers must come from the cited runs
    r = _rec(feature="good", horizon=5, verdict="informative", sign=1, claim="IC is 0.9123", evidence=[g["run_id"]])
    assert not r["ok"] and "0.9123" in r["result"]["error"]
    r = _rec(feature="good", horizon=5, verdict="informative", sign=1, claim=f"t is {t:.2f}", evidence=[g["run_id"]])
    assert r["ok"], r
    # a significance word with a null verdict is refused
    r = _rec(feature="null", horizon=5, verdict="not_informative", sign=0, claim="a significant relation",
             evidence=[n["run_id"]])
    assert not r["ok"]
    # off-plan horizon needs a reason
    g1 = registry.TOOLS["ic_summary"].fn(feature="good", horizon=1)
    r = _rec(feature="good", horizon=1, verdict="inconclusive", sign=1, claim="x", evidence=[g1["run_id"]])
    assert not r["ok"] and "pre-registered" in r["result"]["error"]
    # the wrap-up needs every planned test answered
    assert PR.check_predictive_wrapup(None)


def test_explained_by_control_needs_raw_significance_and_a_controlled_null(armed):
    ctl = registry.TOOLS["ic_summary"].fn(feature="mentions", horizon=20, controls=["dollar_adv"])
    r = _rec(feature="mentions", horizon=20, controls=["dollar_adv"], verdict="explained_by_control", sign=0,
             claim="gone once controlled", evidence=[ctl["run_id"]])
    assert not r["ok"] and "raw relation" in r["result"]["error"]            # never run: the error names the call
    raw = registry.TOOLS["ic_summary"].fn(feature="mentions", horizon=20)
    assert raw["result"]["mean_ic"] < 0 and abs(raw["result"]["t_nw"]) >= 3     # the liquidity premium is negative
    assert abs(ctl["result"]["t_nw"]) < abs(raw["result"]["t_nw"]) / 2
    # the raw run exists in the ledger: it is attached even if not cited (the error only fires when it was never run)
    r = _rec(feature="mentions", horizon=20, controls=["dollar_adv"], verdict="explained_by_control", sign=0,
             claim="gone once controlled", evidence=[ctl["run_id"]])
    assert r["ok"], r
    # 'informative' must survive the controls the signed plan requires (mentions: dollar_adv)
    r = _rec(feature="mentions", horizon=20, verdict="informative", sign=-1, claim="raw", evidence=[raw["run_id"]])
    assert not r["ok"] and "requires mentions to survive ['dollar_adv']" in r["result"]["error"]
    # a controlled null cannot hide a significant raw relation
    r = _rec(feature="mentions", horizon=20, controls=["dollar_adv"], verdict="not_informative", sign=0,
             claim="nothing once controlled", evidence=[ctl["run_id"], raw["run_id"]])
    assert not r["ok"] and "explained_by_control" in r["result"]["error"]
    r = _rec(feature="mentions", horizon=20, controls=["dollar_adv"], verdict="explained_by_control", sign=0,
             claim="gone once controlled", evidence=[ctl["run_id"], raw["run_id"]])
    assert r["ok"], r


def test_strength_words_respect_negation():
    assert PR.strength_asserted("a significant negative relation (t -4.1)")
    assert PR.strength_asserted("strongly predicts returns")
    # an explained_by_control claim may (must) say the raw relation was significant: checked in the observation test
    for ok in ("not statistically significant (t -1.30)", "no significant relation", "the relation is not strong",
               "it isn't significant", "t = -1.3, below the hurdle; not distinguishable from zero",
               "weakly significant at best"):
        assert PR.strength_asserted(ok) is None, ok


# ---------------------------------------------------------------- CRSP ingest (fake files in both WRDS formats)
def _write_crsp(d, ciz: bool):
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    if ciz:
        pd.DataFrame({"PERMNO": [1, 1, 1, 2, 2, 2], "DlyCalDt": days * 2, "Ticker": ["BF"] * 3 + ["XYZ", "XYZ", "XYZ"],
                      "ShareClass": ["B"] * 3 + ["A"] * 3, "DlyPrc": [10, 11, 12, -20, 20, 21],
                      "DlyOpen": [10, 10, 11, 19, 20, 20], "DlyRet": [0.0, 0.1, 0.0909, 0.0, 0.0, 0.05],
                      "DlyVol": [100] * 6, "ShrOut": [1000] * 6}).to_csv(d / "dsf_v2.csv", index=False)
    else:
        pd.DataFrame({"PERMNO": [1, 1, 1, 2, 2, 2], "date": [x.replace("-", "") for x in days] * 2,
                      "TICKER": ["BF"] * 3 + ["XYZ"] * 3, "SHRCLS": ["B"] * 3 + [None] * 3, "SHRCD": [11] * 6,
                      "PRC": [10, 11, 12, -20, 20, 21], "OPENPRC": [10, 10, 11, 19, 20, 20],
                      "RET": ["0", "0.1", "C", "0", "0", "0.05"], "VOL": [100] * 6,
                      "SHROUT": [1000] * 6}).to_csv(d / "dsf.csv", index=False)
        pd.DataFrame({"PERMNO": [2], "DLSTDT": ["20240104"], "DLRET": [-0.5]}).to_csv(d / "dsedelist.csv", index=False)
    pd.DataFrame({"PERMNO": [1, 2], "TICKER": ["BF", "XYZ"], "SHRCLS": ["B", None], "NAMEDT": ["2000-01-01"] * 2,
                  "NAMEENDDT": ["2030-12-31"] * 2}).to_csv(d / "stocknames.csv", index=False)
    pd.DataFrame({"date": days, "vwretd": [0.0, 0.01, 0.02]}).to_csv(d / "dsi.csv", index=False)


@pytest.mark.parametrize("ciz", [False, True])
def test_crsp_ingest_and_dated_link(tmp_path, ciz):
    from quantaccelerator.ingest import crsp
    src, out = tmp_path / "raw", tmp_path / "out"
    src.mkdir()
    _write_crsp(src, ciz)
    info = crsp.ingest(src, out)
    assert info["permnos"] == 2 and info["names"]
    d = pd.read_parquet(out / "daily.parquet").set_index(["permno", "date"])
    assert d.loc[(2, pd.Timestamp("2024-01-02")), "prc"] == 20                  # bid/ask midpoint in absolute value
    assert d.loc[(1, pd.Timestamp("2024-01-03")), "r_oc"] == pytest.approx(0.1)
    if not ciz:
        assert np.isnan(d.loc[(1, pd.Timestamp("2024-01-04")), "ret"])          # CRSP code 'C' -> missing
        assert d.loc[(2, pd.Timestamp("2024-01-04")), "ret"] == pytest.approx(1.05 * 0.5 - 1)   # delisting compounded
    pairs = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-03", "2024-01-03"]),
                          "entity": ["BF/B", "XYZ", "NOPE"]})
    ln = crsp.link(pairs, pd.read_parquet(out / "daily.parquet"), pd.read_parquet(out / "names.parquet")) \
        .set_index("entity").rid
    assert ln["BF/B"] == "1" and ln["XYZ"] == "2" and pd.isna(ln["NOPE"])


def test_completeness_needs_every_planned_horizon_control_set_and_shape(armed):
    T = registry.TOOLS
    owed = lambda: PR.status()["planned_tests_open"]
    start = owed()                       # good: h5, h20, decay, quantile; mentions: h20, decay, quantile; null: 3
    assert start == 10
    for h in (5, 20):
        r = T["ic_summary"].fn(feature="good", horizon=h)
        res = r["result"]
        verdict = "informative" if abs(res["t_nw"]) >= 3 else "not_informative"
        assert _rec(feature="good", horizon=h, verdict=verdict, sign=int(np.sign(res["mean_ic"])) if verdict ==
                    "informative" else 0, claim="x", evidence=[r["run_id"]])["ok"]
    assert owed() == start - 2
    T["ic_decay"].fn(feature="good")
    T["quantile_returns"].fn(feature="good", horizon=20)  # not the primary horizon: still owed
    assert owed() == start - 3
    T["quantile_returns"].fn(feature="good", horizon=5)
    assert owed() == start - 4
    # a raw observation does not discharge a test planned with controls
    raw = T["ic_summary"].fn(feature="mentions", horizon=20)
    ctl = T["ic_summary"].fn(feature="mentions", horizon=20, controls=["dollar_adv"])
    _rec(feature="mentions", horizon=20, verdict="explained_by_control", sign=0, controls=["dollar_adv"], claim="x",
         evidence=[raw["run_id"], ctl["run_id"]])
    assert owed() == start - 5
    assert "ic_decay(feature='mentions')" in " ".join(PR.status()["todo"])


def test_a_pass_owes_only_its_focus_feature(armed):
    registry.CTX.findings["predict_focus"] = "good"
    try:
        assert PR.status()["planned_tests_open"] == 4          # h5, h20, ic_decay, quantile at h5
        assert all(t.startswith("good") for t in PR.status()["todo"])
    finally:
        registry.CTX.findings.pop("predict_focus")


def test_minus_100_percent_day_does_not_poison_later_windows():
    fr = tiny([.01, -1.0, .02, .01, .03, .0], [.0, -1.0, .0, .0, .0, .0])
    fw = core.forward_returns(fr, (2,)).set_index("date")
    assert fw.loc[fw.index[0], "fwd_2"] == pytest.approx(-1.0, abs=1e-5)   # the window with the wipe-out loses all
    assert fw.loc[fw.index[2], "fwd_2"] == pytest.approx(0.01)            # later windows are intact


def test_wrapup_numbers_must_come_from_the_runs(armed):
    from quantaccelerator.state.schemas import PredictiveWrapUp
    r = registry.TOOLS["ic_summary"].fn(feature="good", horizon=5)["result"]
    ok = PredictiveWrapUp(summary=f"good has a mean IC of {r['mean_ic']:.4f} at h=5")
    bad = PredictiveWrapUp(summary="good has a mean IC of 0.0019 at h=5")
    assert not PR._ungrounded_numbers([ok.summary])
    assert PR._ungrounded_numbers([bad.summary]) == [0.0019]
    assert "no result of your runs" in PR.check_predictive_wrapup(bad)[0]


# ---------------------------------------------------------------- Critic
def test_critic_verdict_rule():
    from quantaccelerator.tools.critic import judge
    assert judge(-0.02, {"mean_ic": -0.018, "t_nw": -4.0})[0] == "survived"
    assert judge(-0.02, {"mean_ic": -0.016, "t_nw": -1.5})[0] == "inconclusive"   # a small sample is not a refutation
    assert judge(-0.02, {"mean_ic": -0.003, "t_nw": -0.9})[0] == "refuted"       # under a quarter of the effect left
    assert judge(-0.02, {"mean_ic": 0.010, "t_nw": 2.5})[0] == "refuted"         # significant sign flip
    assert judge(-0.02, {"mean_ic": None, "t_nw": None})[0] == "inconclusive"


def test_critic_draft_checks():
    from quantaccelerator.state.schemas import CriticDraft, Critique, CriticTest
    from quantaccelerator.tools.critic import check_critic_draft
    C = lambda tt, **t: Critique(target=0, threat_type=tt, threat="x", if_spurious="y", test=CriticTest(**t))
    targets, disc = {0: {"controls": ["dollar_adv"]}}, ("2022-01-03", "2023-11-30")
    ok = CriticDraft(critiques=[C("small_illiquid_names", kind="restrict_universe", universe="large_caps"),
                                C("omitted_characteristic", kind="add_controls", controls=["size"])])
    assert check_critic_draft(ok, targets, ["Tech"], disc) == []
    one_angle = CriticDraft(critiques=ok.critiques[:1] * 2)
    assert "2 different threat types" in check_critic_draft(one_angle, targets, ["Tech"], disc)[0]
    bad = CriticDraft(critiques=[C("omitted_characteristic", kind="add_controls", controls=["dollar_adv"]),
                                 C("time_instability", kind="subperiod", period=("2021-01-01", "2022-06-30")),
                                 C("sector_concentration", kind="restrict_universe", universe="exclude_sector",
                                   sector="Nowhere"),
                                 C("small_illiquid_names", kind="add_controls", controls=["size"]),
                                 C("microstructure", kind="restrict_universe", universe="price_ge_5")])
    errs = " ".join(check_critic_draft(bad, targets, ["Tech"], disc))
    assert "already controls" in errs and "inside discovery" in errs and "sector must be one of" in errs
    assert "falsified by removing" in errs and "standard battery" in errs
    full = {0: {"controls": ["size", "reversal", "momentum", "beta", "dollar_adv", "sector"]}}
    assert "restrict_universe or subperiod instead" in " ".join(check_critic_draft(bad, full, ["Tech"], disc))
    # invalid critiques are dropped, not fatal, when the runnable ones still cover the finding
    from quantaccelerator.tools.critic import split_draft
    mixed = CriticDraft(critiques=ok.critiques + bad.critiques)
    assert check_critic_draft(mixed, targets, ["Tech"], disc) == []
    runnable, dropped = split_draft(mixed, targets, ["Tech"], disc)
    assert len(runnable) == 2 and len(dropped) == 5


def test_critic_attack_exposes_an_illiquid_only_effect(tmp_path):
    """A signal planted only in the least liquid third: significant overall, gone once those names are dropped."""
    from quantaccelerator.tools.critic import judge
    fr, feats = returns_frame(n=150, plant=0.0)
    ldv = fr.assign(dv=fr.prc * fr.vol).pivot(index="date", columns="entity", values="dv")
    low = ldv.rank(axis=1, pct=True).shift(1) < 1 / 3                     # yesterday's liquidity, like dollar_adv
    g = feats.pivot(index="date", columns="entity", values="good")
    oc = fr.pivot(index="date", columns="entity", values="r_oc") + 0.01 * g.where(low, 0.0)
    fr = fr.merge(oc.stack().rename("r_oc2").reset_index(), on=["date", "entity"])
    fr["r_oc"] = fr.r_oc2
    fs = frozen(tmp_path, feats)
    registry.set_session("t-critic", tmp_path)
    cond = feats[["date", "entity"]].assign(sector="A")
    vault.arm(tmp_path, fs, plan_gate(fs), lambda: fr, cond)
    try:
        registry.CTX.agent = "predictive_eda"
        full = registry.TOOLS["ic_summary"].fn(feature="good", horizon=1)["result"]
        assert full["t_nw"] > 3
        registry.CTX.agent = "orchestrator"
        att = registry.TOOLS["critic_test"].fn(feature="good", horizon=1, test={"kind": "restrict_universe",
                                                                                "universe": "drop_illiquid_tercile"})
        assert judge(full["mean_ic"], att["result"])[0] == "refuted"
    finally:
        vault.STATE.update(set_hash=None, cache={})


def test_explained_by_control_may_call_the_raw_relation_significant(armed):
    raw = registry.TOOLS["ic_summary"].fn(feature="mentions", horizon=20)
    ctl = registry.TOOLS["ic_summary"].fn(feature="mentions", horizon=20, controls=["dollar_adv"])
    r = _rec(feature="mentions", horizon=20, controls=["dollar_adv"], verdict="explained_by_control", sign=-1,
             claim=f"the raw relation is significant (t {raw['result']['t_nw']:.2f}) and gone with dollar_adv",
             evidence=[raw["run_id"], ctl["run_id"]])
    assert r["ok"], r


# ---------------------------------------------------------------- backward loop
def test_investigation_requests_are_routed_to_their_owner(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setenv("QA_DATASET", "finra")
    from quantaccelerator.agents import orchestrator as O
    from quantaccelerator.agents.base import AgentFailure
    from quantaccelerator.state.schemas import DataAnswer, DataInvestigationRequest as R
    seen = []

    def timed(agent, task, ctx):
        seen.append((type(agent).__name__, agent.name, task))
        if "fail" in task:
            raise AgentFailure("no valid submission", {})
        return DataAnswer(answer="because the file layout changed", evidence=["r0123456789"]), {}

    reqs = [R(observation="median shifts 2022-06-29", question="what caused the shift?", route_to="data_research"),
            R(observation="Friday files", question="are Friday files posted late?", route_to="pit"),
            R(observation="x", question="this one will fail", route_to="data_research"),
            R(observation="y", question="over the limit", route_to="pit")]
    out = O.investigate(tmp_path, None, timed, reqs, "research_eda")
    assert [s[0] for s in seen] == ["DataQA", "PITQA", "DataQA"]                # owner by route, at most 3 routed
    assert out[0].answer.answer.startswith("because") and out[1].routed_to.startswith("pit_qa")
    assert out[2].answer is None and "no valid submission" in out[2].error       # a failure is recorded, not raised
    assert "over the limit" in out[3].error
    saved = _json.loads((tmp_path / "state" / "investigations.json").read_text())
    assert len(saved) == 4 and saved[0]["stage"] == "research_eda"
    O.investigate(tmp_path, None, timed, reqs[:1], "predictive_eda")              # later stages append
    assert len(_json.loads((tmp_path / "state" / "investigations.json").read_text())) == 5


# ---------------------------------------------------------------- cost and capacity
def test_cost_model_nets_out_spreads_and_finds_a_capacity(armed):
    from quantaccelerator.returns import costs
    d = vault.frame()
    d = d.assign(half_spread=0.0005)                                         # 5 bp half-spread everywhere
    out = costs.portfolio(d, "good", 1, 5)
    g, rt, ch = out["gross"]["mean_bp"], out["net_round_trip"]["mean_bp"], out["net_changes_only"]["mean_bp"]
    assert g > 0 and abs((g - rt) - 4 * 5) < 0.5                              # 2 ends x 2 legs x 5 bp
    assert rt <= ch <= g                                                     # changes-only is cheaper
    assert out["break_even_half_spread_bp"] == pytest.approx(g / 4)
    assert out["capacity_per_leg_usd"] and out["capacity_per_leg_usd"] > 0
    expensive = costs.portfolio(d.assign(half_spread=0.02), "good", 1, 5)    # 200 bp: the edge is gone
    assert expensive["net_round_trip"]["mean_bp"] < 0 and expensive["capacity_per_leg_usd"] is None


def test_roll_half_spread_recovers_a_bid_ask_bounce():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2023-01-02", periods=300)
    true = rng.normal(0, 0.01, 300).cumsum()
    s = 0.004                                                                # 40 bp half-spread
    p = 20 * np.exp(true) * (1 + s * rng.choice([-1, 1], 300))
    fr = pd.DataFrame({"date": days, "entity": "A", "prc": p, "ret": pd.Series(p).pct_change().values,
                       "r_oc": 0.0, "vol": 1e5, "shrout": 1e6, "mkt": 0.0})
    hs = core.controls(fr).half_spread.dropna()
    assert 0.002 < hs.iloc[-1] < 0.007


def test_quoted_spreads_replace_roll_when_present():
    fr, _ = returns_frame(n=30)
    fr = fr.assign(bid=fr.prc * 0.999, ask=fr.prc * 1.001)                  # 10 bp half-spread quoted
    c = core.controls(fr).dropna(subset=["half_spread"])
    tail = c[c.date >= c.date.max() - pd.Timedelta(days=30)]
    assert tail.half_spread.median() == pytest.approx(0.001, rel=0.01) and tail.spread_quoted.min() == 1.0


def test_critique_stage_runs_battery_critic_and_propagates_refutations(armed, tmp_path):
    from quantaccelerator.agents import orchestrator as O
    from quantaccelerator.state.schemas import CriticDraft, Critique, CriticTest, ResearchObservation
    # two findings on a pure-noise feature, claimed with a large IC: every attack 'loses' it -> refuted
    mk = lambda ctrl: ResearchObservation(feature="null", horizon=5, controls=ctrl, verdict="informative", sign=1,
                                          claim="x", evidence=[], status="validated", stats={"mean_ic": 0.05, "t_nw": 5})
    obs = [mk([]), mk(["size"])]
    C = lambda tgt, tt, **t: Critique(target=tgt, threat_type=tt, threat="t", if_spurious="s", test=CriticTest(**t))
    draft = CriticDraft(critiques=[C(0, "sector_concentration", kind="restrict_universe", universe="exclude_sector",
                                     sector="A"),
                                   C(0, "small_illiquid_names", kind="restrict_universe", universe="large_caps"),
                                   C(1, "omitted_characteristic", kind="add_controls", controls=["beta"]),
                                   C(1, "time_instability", kind="subperiod", period=("2022-03-01", "2022-09-30"))])
    timed = lambda agent, task, ctx: (CriticDraft(critiques=[c for c in draft.critiques if c.target in agent.targets]),
                                      {"tool_runs": [], "prompt_hash": None})
    panel = vault.STATE["conditioners"]
    out, res = O.critique(tmp_path, None, timed, armed, obs, panel)
    src = lambda s: [r for r in res if r.source == s]
    assert len(src("standard")) == 8 and len(src("critic")) == 4               # 4 battery tests x 2 findings
    prop = src("propagated")
    assert {(r.target, r.test.universe) for r in prop} >= {(1, "exclude_sector"), (1, "large_caps")}
    assert out[0].critique["refuted"] >= 1 and out[1].critique["attacks"] == len([r for r in res if r.target == 1])


def test_routed_answers_need_successful_runs(tmp_path):
    from quantaccelerator.agents.investigate import FeatureQA, PITQA
    from quantaccelerator.state.schemas import DataAnswer
    registry.set_session("t-inv", tmp_path)
    bad = registry.TOOLS["structural_breaks"].fn(table="research_panel", time_field="date")   # unknown table: fails
    assert not bad["ok"]
    a = DataAnswer(answer="the data cannot answer", evidence=[bad["run_id"]])
    assert "time_stability" in FeatureQA().extra_validate(a)[0]
    import os
    os.environ.setdefault("QA_DATASET", "finra")
    assert "failed call is not evidence" in PITQA().extra_validate(a)[0]
