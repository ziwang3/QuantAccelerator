"""Research EDA phase A: feature construction, characterization and the research panel on synthetic data with known
answers, plus the agent loop offline (ScriptedClient)."""
import json
import types

import numpy as np
import pandas as pd
import pytest

from quantaccelerator.state.schemas import EDAWrapUp, FeatureSpec
from quantaccelerator.tools import panel as P
from quantaccelerator.tools import features as FT
from quantaccelerator.tools import registry

DAYS = pd.bdate_range("2024-01-01", periods=120)
SECTORS = ["Tech", "Energy", "Banks", "Unknown"]


def synth(n=60, seed=0) -> pd.DataFrame:
    """A panel with known structure: vol scales with adv; score has a sector offset; noise depends on nothing;
    pers is a persistent per-entity level."""
    rng = np.random.default_rng(seed)
    ent = [f"E{i:02d}" for i in range(n)]
    size = pd.Series(np.exp(rng.normal(15, 1.5, n)), index=ent)
    sec = pd.Series([SECTORS[i % 4] for i in range(n)], index=ent)
    lvl = pd.Series(rng.normal(0, 1, n), index=ent)
    rows = []
    for d in DAYS:
        for e in ent:
            adv = size[e] * np.exp(rng.normal(0, .1))
            rows.append((d, e, adv * np.exp(rng.normal(0, .3)), adv, {"Tech": 1.0, "Energy": -1.0}.get(sec[e], 0.0)
                         + rng.normal(0, .5), rng.normal(), lvl[e] + rng.normal(0, .05), sec[e], size[e]))
    df = pd.DataFrame(rows, columns=["date", "entity", "vol", "adv_proxy", "score", "noise", "pers", "sector",
                                     "size_float"])
    df["size_assets"] = df.size_float * 2
    df["quality_flag"] = False
    return df.sort_values(["date", "entity"]).reset_index(drop=True)


@pytest.fixture
def session(tmp_path):
    p = synth()
    registry.set_session("t-eda", tmp_path)
    registry.CTX.agent = "research_eda"
    P.set_panel(p, {"values": ["vol", "score", "noise", "pers"], "hash": "h", "n_rows": len(p),
                    "n_dates": len(DAYS), "n_entities": 60, "universe_n": 60, "formation": ["a", "b"],
                    "scale_field": "vol", "rule": None})
    yield p
    P.STATE.update(panel=None, meta=None)
    FT.CACHE.clear()


def spec(**kw):
    return FeatureSpec(**{"name": "f", "inputs": [], **kw})


def test_construction_ops_have_the_defined_values(session):
    p, get = session, lambda n: session[n].astype(float)
    r = FT.compute(spec(op="ratio", inputs=["vol", "adv_proxy"]), p, get)
    assert np.allclose(r, p.vol / p.adv_proxy)
    ch = FT.compute(spec(op="change", inputs=["pers"], window=2), p, get)
    e = p[p.entity == "E03"]
    assert np.isclose(ch[e.index[5]], e.pers.iloc[5] - e.pers.iloc[3]) and np.isnan(ch[e.index[1]])
    # surprise uses only the previous records: changing today's value does not change today's mean/std
    s = FT.compute(spec(op="surprise", inputs=["noise"], window=10), p, get)
    prev = e.noise.iloc[10:20]
    assert np.isclose(s[e.index[20]], (e.noise.iloc[20] - prev.mean()) / prev.std())
    pr = FT.compute(spec(op="peer_relative", inputs=["score"], group="sector"), p, get)
    assert pr[p.sector == "Unknown"].isna().all()           # no group -> no peer value
    assert abs(pr.groupby([p.date, p.sector]).median().dropna()).max() < 1e-12
    rk = FT.compute(spec(op="rank", inputs=["vol"]), p, get)
    assert rk.between(0, 1).all() and np.isclose(rk.groupby(p.date).max(), 1).all()


def test_exposure_finds_planted_dependences_and_no_false_ones(session):
    p = session
    st = FT.exposure_stats(p, p.vol, "adv_proxy")
    assert st["strength"] == "strong" and st["rank_corr_mean"] > 0.8
    assert st["mean_feature_rank_by_conditioner_quintile"]["Q5"] > 0.8
    assert FT.exposure_stats(p, p.vol / p.adv_proxy, "adv_proxy")["strength"] == "negligible"
    assert FT.exposure_stats(p, p.noise, "adv_proxy")["strength"] == "negligible"
    sec = FT.exposure_stats(p, p.score, "sector")
    assert sec["strength"] == "strong" and sec["mean_rank_by_group"]["Tech"] > sec["mean_rank_by_group"]["Energy"]
    assert "Unknown" not in sec["mean_rank_by_group"]       # unmatched entities are left out, not a group
    assert FT.exposure_stats(p, p.noise, "sector")["strength"] == "negligible"
    res = FT.compute(spec(op="residualize", inputs=["score"], group="sector"), p, lambda n: p[n].astype(float))
    assert FT.exposure_stats(p, res, "sector")["strength"] == "negligible"
    res = FT.compute(spec(op="residualize", inputs=["vol"], group="adv_proxy"), p,
                     lambda n: np.log(p[n].astype(float)) if n == "vol" else p[n].astype(float))
    assert abs(FT.exposure_stats(p, res, "adv_proxy")["rank_corr_mean"]) < 0.1


def test_stability_measures_rank_persistence(session):
    p = session
    assert FT.stability_stats(p, p.pers)["rank_autocorrelation"]["lag5"] > 0.9
    assert abs(FT.stability_stats(p, p.noise)["rank_autocorrelation"]["lag5"]) < 0.1


def test_spec_validation_names_the_fix(session):
    p, specs = session, {}
    errs = lambda **kw: " ".join(FT.validate_spec(spec(**kw), specs, p))
    assert "unknown input 'returns'" in errs(name="xx", op="field", inputs=["returns"])   # return-blind: no such column
    assert "takes 2 input(s)" in errs(name="xx", op="ratio", inputs=["vol"])
    assert "needs a window" in errs(name="xx", op="surprise", inputs=["vol"])
    assert "unknown input 'sector'" in errs(name="xx", op="field", inputs=["sector"])     # categorical: group only
    assert "panel column" in errs(name="vol", op="field", inputs=["vol"])
    assert "residualize needs group" in errs(name="xx", op="residualize", inputs=["vol"], group="date")
    assert errs(name="xx", op="peer_relative", inputs=["score"], group="sector") == ""
    assert "inputs are names" in errs(name="xx", op="diff", inputs=["peer_relative(score, sector)", "score"])
    specs["vol_per_adv"] = spec(name="vol_per_adv", op="ratio", inputs=["vol", "adv_proxy"])
    assert "already scale-free" in errs(name="xx", op="ratio", inputs=["vol_per_adv", "adv_proxy"])


def test_panel_is_point_in_time_and_fixes_the_universe_in_advance(monkeypatch):
    days = pd.bdate_range("2023-12-01", "2024-03-29")
    view = pd.DataFrame([(d, e, float(v)) for d in days for e, v in (("BIG", 100), ("MID", 50), ("LATE", 0))],
                        columns=["decision_time", "entity", "x"])
    view = view[~((view.entity == "LATE") & (view.decision_time < "2024-01-01"))]   # LATE only trades in the panel
    view.loc[(view.entity == "LATE"), "x"] = 1000.0
    view["source_time"], view["quality_flag"] = view.decision_time, False
    monkeypatch.setattr(P, "active", lambda: types.SimpleNamespace(name="t", view={"values": ["x"]}, panel=None))
    sym = pd.DataFrame({"Symbol": ["BIG", "MID"], "cik": [1, 2], "sector": ["Tech", "Energy"]})
    fun = pd.DataFrame({"cik": [1, 1], "item": ["public_float", "public_float"], "val": [10.0, 20.0],
                        "end": pd.to_datetime(["2023-06-30", "2023-12-31"]),
                        "filed": pd.to_datetime(["2023-08-01", "2024-02-14"])})
    monkeypatch.setattr(P, "_reference", lambda ents: (sym[sym.Symbol.isin(ents)], fun))
    spec_ = {"universe_n": 2, "formation": ["2023-12-01", "2023-12-31"], "start": "2024-01-01", "end": "2024-03-29",
             "scale_field": "x", "adv_window": 5}
    panel, meta = P.build_panel(view, spec_)
    assert set(panel.entity) == {"BIG", "MID"}             # LATE is the most active in the panel but not in formation
    big = panel[panel.entity == "BIG"].set_index("date")
    assert big.size_float["2024-02-14"] == 10.0             # filed that day: usable only from the next day
    assert big.size_float["2024-02-15"] == 20.0
    assert panel[panel.entity == "MID"].size_float.isna().all() and meta["n_entities"] == 2


def _ids(messages, tool):
    """run_ids of earlier results of `tool`, in order (scripts cite what only exists at run time)."""
    out, names = [], {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = tc["function"]["name"]
        if m["role"] == "tool" and names.get(m.get("tool_call_id")) == tool:
            body = json.loads(m["content"])
            if body.get("ok"):
                out.append(body["run_id"])
    return out


def call(_tool, **args):
    return {"tool_calls": [{"name": _tool, "arguments": args}]}



def test_candidate_needs_characterization_and_wrapup_needs_flags_addressed(session):
    from quantaccelerator.tools.features import check_eda_findings, reset_findings
    reset_findings()
    T = registry.TOOLS
    T["construct_feature"].fn(name="vol_raw", op="field", inputs=["vol"])
    r = T["propose_candidate"].fn(name="vol_raw", rationale="x", evidence=[])
    assert not r["ok"] and "characterize vol_raw" in r["result"]["error"]
    T["compute_exposure"].fn(feature="vol_raw", by="sector")
    r = T["propose_candidate"].fn(name="vol_raw", rationale="x", evidence=[], known_exposures=["strong adv tilt"])
    assert r["ok"] or "cite" not in r["result"].get("error", "")       # evidence is attached from the log
    e = T["compute_exposure"].fn(feature="vol_raw", by="adv_proxy")
    assert e["result"]["flags"][0]["id"] == "exposure_vol_raw_adv_proxy"
    errs = " ".join(check_eda_findings(EDAWrapUp(summary="s")))
    assert "vol_raw depends strongly on adv_proxy" in errs and "at least three" in errs
    o = T["record_feature_observation"].fn(feature="vol_raw", claim="rises with liquidity", conditioner="adv_proxy",
                                           evidence=[e["run_id"]], decision="residualize",
                                           possible_explanations=["scale"])
    assert not o["ok"] and "at least two" in o["result"]["error"]       # explanations before neutralizing



def test_candidate_must_state_the_strong_dependences_it_keeps(session):
    from quantaccelerator.tools.features import reset_findings
    reset_findings()
    T = registry.TOOLS
    T["construct_feature"].fn(name="score_raw", op="field", inputs=["score"])
    run = T["compute_exposure"].fn(feature="score_raw", by="sector")["run_id"]
    r = T["propose_candidate"].fn(name="score_raw", rationale="the vendor's score", evidence=[run])
    assert not r["ok"] and "still depends strongly on sector" in r["result"]["error"]
    r = T["propose_candidate"].fn(name="score_raw", rationale="the vendor's score", evidence=[run],
                                  known_exposures=[f"strong sector tilt (eta² 0.3, {run})"])
    assert not r["ok"] and "build a version without it" in r["result"]["error"]   # stated; next, compared


def test_survey_pass_characterizes_every_field_and_plans_each(session):
    from quantaccelerator.agents.research_eda import ResearchEDASurvey
    from quantaccelerator.llm.client import ScriptedClient
    from quantaccelerator.tools.features import reset_findings
    reset_findings()
    P.STATE["meta"]["values"] = ["vol", "score"]
    cmp = lambda b: call("compare_representations", features=["vol", "score"], by=b)
    rec = lambda f, cond, i, **kw: (lambda m: call("record_feature_observation", feature=f, conditioner=cond,
                                                   claim=f"{f} vs {cond}", evidence=[_ids(m, "compare_representations")[i]],
                                                   **kw))
    plan = lambda *lines: call("submit", plan=list(lines))
    script = [call("describe_panel"), cmp("adv_proxy"), plan("vol: ratio"),     # rejected: not characterized yet
              cmp("size_float"), cmp("sector"),
              rec("vol", "adv_proxy", 0, possible_explanations=["size", "liquidity"]),
              plan("vol: scales with liquidity and size, try vol / adv_proxy"),  # rejected: score unplanned, unrecorded
              rec("score", "sector", 2, possible_explanations=["vendor bias", "true sector difference"]),
              rec("vol", "size_float", 1, possible_explanations=["size", "liquidity"]),
              plan("vol: scales with liquidity and size, try vol / adv_proxy", "score: sector offset, try peer_relative")]
    agent = ResearchEDASurvey()
    agent.max_tool_calls_per_turn = None
    w, trace = agent.run(ScriptedClient(script), "survey")
    assert len(w.plan) == 2 and trace["submit_failures"] == 2
    assert "Not yet" in trace["submit_errors"][0] and "missing: score" in trace["submit_errors"][1]


def test_a_raw_field_can_be_a_candidate_and_a_failed_call_can_be_retried(session, tmp_path):
    from quantaccelerator.agents.research_eda import ResearchEDA
    from quantaccelerator.llm.client import ScriptedClient
    from quantaccelerator.tools.features import materialize, reset_findings
    reset_findings()
    T = registry.TOOLS
    run = T["compare_representations"].fn(features=["noise", "vol"], by="adv_proxy")["run_id"]
    for b in ("size_float", "sector"):
        T["compare_representations"].fn(features=["noise", "vol"], by=b)
    r = T["propose_candidate"].fn(name="noise", rationale="already comparable", evidence=[run])
    assert r["ok"] and FT.resolve("noise").equals(session.noise.astype(float))
    assert np.allclose(materialize(list(FT._features().values()), ["noise"], session).noise, session.noise)
    # an identical call that failed (input not built yet) runs again once the input exists
    cmp = call("compare_representations", features=["vol", "vol_adv"], by="adv_proxy")
    script = [cmp, call("construct_feature", name="vol_adv", op="ratio", inputs=["vol", "adv_proxy"]), cmp,
              call("submit", summary="x")]
    agent = ResearchEDA()
    agent.extra_validate = lambda w: []
    _, trace = agent.run(ScriptedClient(script), "t")
    ok = [json.loads(m["content"]).get("ok") for m in trace["messages"] if m["role"] == "tool"]
    assert ok[:3] == [False, True, True]



def test_no_neutralizing_what_is_not_there_and_no_keeping_a_dependence_uncompared(session):
    from quantaccelerator.tools.features import reset_findings
    reset_findings()
    T = registry.TOOLS
    T["construct_feature"].fn(name="noise_rel", op="residualize", inputs=["noise"], group="sector")
    runs = [T["compare_representations"].fn(features=["noise_rel", "score"], by=b)["run_id"]
            for b in ("adv_proxy", "size_float", "sector")]
    r = T["propose_candidate"].fn(name="noise_rel", rationale="x", evidence=runs)
    assert not r["ok"] and "negligible dependence on sector" in r["result"]["error"]
    r = T["propose_candidate"].fn(name="score", rationale="the vendor's score", evidence=runs,
                                  known_exposures=["strong sector tilt"])
    assert not r["ok"] and "build a version without it" in r["result"]["error"]
    T["construct_feature"].fn(name="score_rel", op="peer_relative", inputs=["score"], group="sector")
    T["compare_representations"].fn(features=["score", "score_rel"], by="sector")
    r = T["propose_candidate"].fn(name="score", rationale="the vendor's score", evidence=runs,
                                  known_exposures=["strong sector tilt"])
    assert r["ok"]                                   # compared: keeping the dependence is now an informed choice


def test_new_ops_diff_and_rolling_std(session):
    p, get = session, lambda n: session[n].astype(float)
    assert np.allclose(FT.compute(spec(op="diff", inputs=["score", "noise"]), p, get), p.score - p.noise)
    rs = FT.compute(spec(op="rolling_std", inputs=["noise"], window=10), p, get)
    e = p[p.entity == "E05"]
    assert np.isclose(rs[e.index[20]], e.noise.iloc[11:21].std())
    assert "takes 2 input(s)" in " ".join(FT.validate_spec(spec(name="xx", op="diff", inputs=["vol"]), {}, p))


def _plan(researcher_text="activity per unit of liquidity"):
    from quantaccelerator.state.schemas import IdeaPlan
    return IdeaPlan(ideas=[
        {"name": "activity_intensity", "idea": "activity relative to liquidity", "mechanism": "unusual activity",
         "assumptions": ["volume reflects interest"], "expected_direction": "higher values should precede lower subsequent returns", "source": "researcher",
         "researcher_text": researcher_text, "specs": [{"name": "vol_per_adv", "op": "ratio", "inputs": ["vol", "adv_proxy"]}]},
        {"name": "relative_score", "idea": "score vs sector peers", "mechanism": "vendor bias by sector",
         "assumptions": ["offsets are bias"], "expected_direction": "higher values should precede lower subsequent returns",
         "specs": [{"name": "score_rel", "op": "peer_relative", "inputs": ["score"], "group": "sector"},
                   {"name": "score_chg", "op": "change", "inputs": ["score"], "window": 5}]},
        {"name": "persistent_level", "idea": "slow characteristic", "mechanism": "persistent trait",
         "assumptions": ["levels persist"], "expected_direction": "higher values should precede lower subsequent returns",
         "specs": [{"name": "pers_rank", "op": "diff", "inputs": ["pers", "noise"]}]}])


def test_idea_plan_is_checked_buildable_and_complete(session):
    plan = _plan()
    assert FT.check_idea_plan(plan, ["activity per unit of liquidity"]) == []
    errs = " ".join(FT.check_idea_plan(plan, ["a different idea"]))
    assert "translate every researcher idea" in errs
    bad = plan.model_copy(deep=True)
    bad.ideas[1].source = "researcher"
    bad.ideas[2].expected_direction = "up"
    bad.ideas[0].specs[0].inputs = ["returns", "adv_proxy"]          # return-blind: no such column
    errs = " ".join(FT.check_idea_plan(bad, []))
    assert "unknown input 'returns'" in errs and "at least two ideas of your own" in errs and "full hypothesis" in errs
    lone = plan.model_copy(deep=True)
    lone.ideas[2].specs = [lone.ideas[2].specs[0].model_copy(update={"op": "residualize", "inputs": ["pers"], "group": "sector"})]
    assert "representation choice" in " ".join(FT.check_idea_plan(lone, ["activity per unit of liquidity"]))


def test_idea_flow_builds_characterizes_and_freezes(session, tmp_path):
    from quantaccelerator.agents import feature_report, orchestrator as O
    from quantaccelerator.llm.client import ScriptedClient
    from quantaccelerator.state.schemas import IdeaPlan
    plan = _plan()
    feats = ["vol_per_adv", "score_rel", "pers_rank"]
    idea = dict(zip(feats, ["activity_intensity", "relative_score", "persistent_level"]))
    rec = lambda f: (lambda m: call("record_feature_observation", feature=f, idea=idea[f], conditioner="adv_proxy",
                                    claim=f"{f} is comparable", evidence=[_ids(m, "compare_representations")[0]], decision="keep"))
    prop = lambda f: (lambda m: call("propose_candidate", name=f, idea=idea[f], rationale="expresses the idea",
                                     evidence=_ids(m, "compare_representations")[:3]))
    script = ([call("compare_representations", features=feats, by=b) for b in ("adv_proxy", "size_float", "sector")]
              + [call("time_stability", feature=f) for f in feats] + [rec(f) for f in feats]
              + [call("propose_candidate", name="vol_per_adv", idea="relative_score", rationale="x", evidence=[])]  # wrong idea
              + [prop(f) for f in feats] + [call("submit", summary="three ideas, one candidate each")])
    run_dir, client, meta, timed = O._start("t-ideas", tmp_path, ScriptedClient(script), mode="explore")
    g2 = O.gate_ideas(plan, oracle=True)
    agent_cls = O.ResearchEDA
    orig = agent_cls.max_tool_calls_per_turn
    agent_cls.max_tool_calls_per_turn = None
    try:
        fs = O.build_ideas(run_dir, client, timed, plan, g2, {})
    finally:
        agent_cls.max_tool_calls_per_turn = orig
    assert [c.name for c in fs.candidates] == feats and {c.idea for c in fs.candidates} == set(idea.values())
    assert fs.idea_gate.gate == "G2" and len(fs.ideas) == 3
    msgs = json.loads((run_dir / "transcripts" / "research_eda.json").read_text())
    assert any("is not built from the data idea" in m["content"] for m in msgs if m["role"] == "tool")
    fs = O.freeze_features(run_dir, fs, O.gate_freeze(fs, oracle=True))
    assert fs.signed_off.gate == "G3" and (run_dir / "features" / f"{fs.set_hash}.parquet").exists()
    meta["status"] = "ok"
    O._finish(run_dir, client, meta)
    b = feature_report.build(run_dir)
    assert len(b["ideas"]) == 3 and b["ideas"][0]["candidates"] == ["vol_per_adv"]


def test_a_building_block_is_not_the_idea(session):
    from quantaccelerator.state.schemas import FeatureIdea
    from quantaccelerator.tools.features import reset_findings
    reset_findings()
    idea = FeatureIdea(name="abnormal_activity", idea="unusual activity vs own history", mechanism="news",
                       assumptions=["volume reacts to news"], expected_direction="higher values precede larger moves later",
                       specs=[{"name": "vol_per_adv", "op": "ratio", "inputs": ["vol", "adv_proxy"]},
                              {"name": "vol_surprise", "op": "surprise", "inputs": ["vol_per_adv"], "window": 10}])
    FT.CTX.findings["ideas"] = {idea.name: idea}
    T = registry.TOOLS
    for sp in idea.specs:
        T["construct_feature"].fn(**sp.model_dump())
    runs = [T["compare_representations"].fn(features=["vol_per_adv", "vol_surprise"], by=b)["run_id"]
            for b in ("adv_proxy", "size_float", "sector")]
    r = T["propose_candidate"].fn(name="vol_per_adv", idea=idea.name, rationale="x", evidence=runs)
    assert not r["ok"] and "building block" in r["result"]["error"] and "vol_surprise" in r["result"]["error"]
    assert T["propose_candidate"].fn(name="vol_surprise", idea=idea.name, rationale="x", evidence=runs)["ok"]
    # a researcher's idea is theirs: the agent's own construction from the same field is not its candidate
    FT.CTX.findings["ideas"][idea.name] = idea.model_copy(update={"source": "researcher", "researcher_text": "t"})
    T["construct_feature"].fn(name="vol_rank", op="rank", inputs=["vol"])
    runs2 = [T["compare_representations"].fn(features=["vol_rank"], by=b)["run_id"] for b in ("adv_proxy", "size_float", "sector")]
    r = T["propose_candidate"].fn(name="vol_rank", idea=idea.name, rationale="x", evidence=runs2,
                                  known_exposures=["strong adv dependence"])
    assert not r["ok"] and "not the researcher's idea" in r["result"]["error"]
    # a needless neutralization on top of a measure: the measure itself may be proposed (no deadlock)
    idea2 = FeatureIdea(name="noise_level", idea="noise vs peers", mechanism="m", assumptions=["a"],
                        expected_direction="higher values precede larger moves later",
                        specs=[{"name": "noise_x", "op": "diff", "inputs": ["noise", "pers"]},
                               {"name": "noise_rel", "op": "peer_relative", "inputs": ["noise_x"], "group": "sector"}])
    FT.CTX.findings["ideas"][idea2.name] = idea2
    for sp in idea2.specs:
        T["construct_feature"].fn(**sp.model_dump())
    runs = [T["compare_representations"].fn(features=["noise_x", "noise_rel"], by=b)["run_id"]
            for b in ("adv_proxy", "size_float", "sector")]
    r = T["propose_candidate"].fn(name="noise_rel", idea=idea2.name, rationale="x", evidence=runs)
    assert not r["ok"] and "negligible dependence on sector" in r["result"]["error"]
    assert T["propose_candidate"].fn(name="noise_x", idea=idea2.name, rationale="x", evidence=runs)["ok"]


def test_an_idea_is_set_aside_only_with_its_characterization(session):
    from quantaccelerator.tools.features import reset_findings, unaccounted_ideas
    reset_findings()
    plan = _plan()
    FT.CTX.findings["ideas"] = {i.name: i for i in plan.ideas}
    T = registry.TOOLS
    T["construct_feature"].fn(name="pers_rank", op="diff", inputs=["pers", "noise"])
    run = T["describe_panel"].fn()["run_id"]
    r = T["record_feature_observation"].fn(feature="pers_rank", idea="persistent_level", claim="no use", evidence=[run],
                                           decision="drop")
    assert not r["ok"]
    st = T["time_stability"].fn(feature="pers_rank")["run_id"]
    assert T["record_feature_observation"].fn(feature="pers_rank", idea="persistent_level", claim="too slow-moving",
                                              conditioner="time", evidence=[st], decision="drop")["ok"]
    assert "persistent_level" not in unaccounted_ideas()


def test_claim_numbers_and_strengths_must_match_the_cited_runs(session):
    from quantaccelerator.tools.features import reset_findings
    reset_findings()
    T = registry.TOOLS
    T["construct_feature"].fn(name="vol_per_adv", op="ratio", inputs=["vol", "adv_proxy"])
    run = T["compare_representations"].fn(features=["vol", "vol_per_adv"], by="adv_proxy")
    rho = run["result"]["features"]["vol_per_adv"]["dependence"]
    rec = lambda claim: T["record_feature_observation"].fn(feature="vol_per_adv", claim=claim, conditioner="adv_proxy",
                                                         evidence=[run["run_id"]])
    r = rec("vol_per_adv has a strong dependence on adv_proxy (rho -0.60)")          # an invented number
    assert not r["ok"] and "no cited result has that number" in r["result"]["error"]
    r = rec("vol_per_adv has a strong dependence on adv_proxy")                     # a misstated strength
    assert not r["ok"] and "measured it as negligible" in r["result"]["error"]
    assert rec(f"vol_per_adv has a negligible dependence on adv_proxy (rho {rho:.3f})")["ok"]
    vol = run["result"]["features"]["vol"]["dependence"]
    assert rec(f"raw vol tracks liquidity: rank correlation {100 * vol:.1f}% of the way")["ok"]   # percentages count
