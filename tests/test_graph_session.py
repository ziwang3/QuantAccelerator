"""The notebook on the LangGraph backend (graph/session.py) and the Understand graph with gate G0, with the stage
functions stubbed: the whole workflow walked gate by gate, rejections looping back, wrong-order calls refused, and a
session continued in a new Session object (a kernel restart) from its checkpoint."""
import json
import types

import pytest

from quantaccelerator import notebook
from quantaccelerator.agents import orchestrator as O
from quantaccelerator.graph import session as GS
from quantaccelerator.graph import understand as GU
from quantaccelerator.graph.state import Runtime
from quantaccelerator.llm.client import ScriptedClient
from quantaccelerator.state.schemas import (AvailabilityRule, CandidateFeature, CandidateFeatureSet, CapabilityRating,
                                            DataResearchDraft, ResearchCapabilityAssessment,
                                            CleaningRecipe, CleaningStep, DatasetCard, EDAWrapUp, FeatureIdea,
                                            FeatureSpec, IdeaPlan, PITContractDraft, PlannedTest, PredictionPlan,
                                            PredictiveWrapUp, Provenance, ResearchObservation, SplitSpec)
from quantaccelerator.tools import registry

RATING = CapabilityRating(rating="moderate", evidence=[], note="")
STEP = CleaningStep(op="drop_duplicates", table="shvol", reason="dup", evidence=[], key=["Date"])
RESEARCH = DataResearchDraft(capability=ResearchCapabilityAssessment(
    history_depth=RATING, cross_sectional_breadth=RATING, update_frequency=RATING, coverage_stability=RATING,
    suitable_for=[], weak_for=[], major_limits=[]), preprocessing=[], observations=[],
    cleaning=CleaningRecipe(steps=[STEP]))
BARE = DatasetCard(dataset_id="d", tables=[], entity_id_fields=[], time_fields=[], fields=[], files=[],
                   provenance=Provenance(producer_agent="profiler"))
CARD = BARE.model_copy(update={"research": RESEARCH})
DRAFT = PITContractDraft(dataset_id="d", fields=[], availability_rule=AvailabilityRule(
    source_field="shvol.Date", tradable_at="next_trading_day_open"), rule_explanation="x", evidence=[],
    revision_policy="", universe_policy="", confidence=0.9)
PLAN = IdeaPlan(ideas=[FeatureIdea(name="i1", idea="idea", mechanism="m", assumptions=[], expected_direction="+",
                                   specs=[FeatureSpec(name="f1", op="field", inputs=["x"])])])
FS = CandidateFeatureSet(panel_hash="ph", panel={}, features=[FeatureSpec(name="f1", op="field", inputs=["x"])],
                         ideas=PLAN.ideas, candidates=[CandidateFeature(name="f1", rationale="r", evidence=[])],
                         observations=[], wrapup=EDAWrapUp(summary="s"), provenance=Provenance(producer_agent="eda"))
PPLAN = PredictionPlan(tests=[PlannedTest(feature="f1", expected_sign=1, horizons=[5], horizon_rationale="r")],
                       set_hash="h1", split=SplitSpec(discovery=("2021-01-01", "2022-12-31"),
                                                      validation=("2023-01-01", "2023-12-31"),
                                                      locked_test=("2024-01-01", "2024-12-31")))
OBS = [ResearchObservation(feature="f1", horizon=5, verdict="informative", sign=1, claim="c", evidence=[])]


@pytest.fixture
def stages(monkeypatch, tmp_path):
    calls = []

    def rec(name, ret):
        def f(*a, **k):
            calls.append(name)
            return ret(*a, **k) if callable(ret) else ret
        return f

    def sign(run_dir, client, draft, trace, g):
        registry.CTX.reference_rule = draft.availability_rule
        return draft

    def panel(run_dir, contract, derived=None):
        from quantaccelerator.tools.panel import STATE
        STATE["meta"] = {"hash": "ph", "n_rows": 10, "n_dates": 5, "n_entities": 2, "formation": ["a", "b"],
                         "derived": derived}
        return None, STATE["meta"]

    profile = types.SimpleNamespace(name="stub", dataset_id="d", predict={"returns": None}, view=None,
                                    first_tradable=None, panel={})
    for mod in (notebook, GS):
        monkeypatch.setattr(mod, "active", lambda: profile)
    monkeypatch.setattr(notebook, "RUNS", tmp_path)
    import quantaccelerator.agents.brief as brief
    import quantaccelerator.agents.feature_report as fr
    import quantaccelerator.agents.predictive_report as pr
    import quantaccelerator.tools.clean as clean
    for m in (brief, fr, pr):
        monkeypatch.setattr(m, "build", lambda run_dir: None)
    monkeypatch.setattr(clean, "apply_recipe", rec("apply_recipe", {"recipe_hash": "rh", "effects": [],
                                                                     "tables": {"shvol": "data/derived/x.parquet"}}))
    monkeypatch.setattr(O, "understand", rec("understand", CARD))
    monkeypatch.setattr(O, "propose_contract", rec("propose_contract", (DRAFT, {"tool_runs": [], "prompt_hash": None})))
    monkeypatch.setattr(O, "sign_contract", rec("sign_contract", sign))
    monkeypatch.setattr(O, "build_research_panel", rec("panel", panel))
    monkeypatch.setattr(O, "propose_ideas", rec("propose_ideas", (PLAN, {"survey": []})))
    monkeypatch.setattr(O, "build_ideas", rec("build_ideas", FS))
    monkeypatch.setattr(O, "freeze_features", rec("freeze", lambda run_dir, fs, g: fs.model_copy(
        update={"signed_off": g, "set_hash": "h1" if g.approved else ""})))
    monkeypatch.setattr(O, "plan_predictions", rec("plan", PPLAN))
    monkeypatch.setattr(O, "predict", rec("predict", (PredictiveWrapUp(summary="w"), OBS,
                                                      {"tool_runs": [], "prompt_hash": None})))
    monkeypatch.setattr(O, "validate_observations", rec("validate", lambda obs: [o.model_copy(update={
        "status": "validated", "validation": {"mean_ic": 0.1, "t_nw": 3.0}}) for o in obs]))
    monkeypatch.setattr(O, "critique", rec("critique", lambda *a: (a[4], [])))
    monkeypatch.setattr(O, "cost_check", rec("costs", lambda obs: obs))
    monkeypatch.setattr(O, "locked_test", rec("locked_test", lambda obs, g5: [o.model_copy(update={
        "status": "locked_pass"}) if i in (g5.released_observations or []) else o for i, o in enumerate(obs)]))
    monkeypatch.setattr(O, "_ledger_summary", lambda study: {"study": study})
    monkeypatch.setattr(O, "_prov", lambda *a, **k: Provenance(producer_agent="predictive_eda"))
    monkeypatch.setattr(Runtime, "panel", lambda self, s: None)
    monkeypatch.setattr(Runtime, "vault", lambda self, s: None)
    stages.calls, stages.tmp = calls, tmp_path
    return stages


def new_session(session_id=None):
    return notebook.Session(session_id=session_id)


def test_notebook_walks_the_whole_workflow_on_the_graph(stages):
    s = new_session()
    s.understand()
    assert s._lg.waiting() == "gate G0" and s.card == CARD
    s.approve_cleaning()
    assert s.derived == {"shvol": "data/derived/x.parquet"} and s._lg.waiting() == "step 'time'"
    s.time()
    assert s.draft == DRAFT and s._lg.waiting() == "gate G1"
    s.approve("ok")
    s.explore(ideas=["my idea"])
    assert s.idea_plan == PLAN and s._lg.waiting() == "gate G2"
    s.approve_ideas()
    s.freeze_features()
    assert s.feature_set.set_hash == "h1"
    s.plan_predictions()
    s.approve_plan()
    assert s._lg.waiting() == "gate G5" and s.research_observations[0].status == "validated"
    rep = s.release_test()
    assert rep.observations[0].status == "locked_pass" and s._lg.waiting() == "finished"
    assert stages.calls == ["understand", "apply_recipe", "propose_contract", "sign_contract", "panel", "propose_ideas",
                            "build_ideas", "freeze", "plan", "predict", "validate", "critique", "costs", "locked_test"]
    state = s.run_dir / "state"
    assert {f"gate_g{i}.json" for i in range(6)} <= {p.name for p in state.iterdir()}
    assert json.loads((s.run_dir / "run_meta.json").read_text())["mode"] == "notebook"
    v = s._lg.values
    assert v["options"]["researcher_ideas"] == ["my idea"] and v["art"]["derived"] == s.derived


def test_wrong_order_is_refused_with_where_the_session_waits(stages):
    s = new_session()
    s.understand()
    with pytest.raises(RuntimeError, match="gate G0"):
        s.time()


def test_rejected_ideas_and_plan_go_back_to_their_step(stages):
    s = new_session()
    s.understand()
    s.reject_cleaning("keep raw")
    assert s.derived is None
    s.time()
    s.approve()
    s.explore()
    s.reject_ideas("try again")
    assert s._lg.waiting() == "step 'explore'"
    s.explore(ideas=["better idea"])
    s.approve_ideas()
    s.reject_features("not these")
    assert s._lg.waiting() == "step 'explore'"
    s.explore()
    s.approve_ideas()
    s.freeze_features()
    s.plan_predictions()
    s.reject_plan("no")
    assert s._lg.waiting() == "step 'plan'"
    assert stages.calls.count("propose_ideas") == 3 and "predict" not in stages.calls


def test_a_session_continues_after_a_kernel_restart(stages):
    s = new_session()
    s.understand()
    s.approve_cleaning()
    s.time()
    s.approve()
    s.explore(ideas=["x"])
    registry.set_session(None)   # the kernel is gone
    s2 = new_session(session_id=s.session_id)
    assert s2._lg.waiting() == "gate G2" and s2.idea_plan == PLAN and s2.card == CARD
    s2.approve_ideas()
    s2.freeze_features()
    assert s2.feature_set.set_hash == "h1" and stages.calls.count("understand") == 1


def test_understand_graph_runs_gate_g0_with_the_oracle(stages, tmp_path):
    card = GU.run_understand("u-test", run_dir=tmp_path / "u", client=ScriptedClient([]))
    assert card == CARD
    g0 = json.loads((tmp_path / "u" / "state" / "gate_g0.json").read_text())
    assert g0["gate"] == "G0" and g0["decided_by"] == "oracle" and g0["approved"]
    assert json.loads((tmp_path / "u" / "run_meta.json").read_text())["status"] == "ok"


def test_without_a_cleaning_proposal_g0_does_not_pause(stages, monkeypatch):
    monkeypatch.setattr(O, "understand", lambda *a, **k: BARE)
    s = new_session()
    s.understand(research=False)
    assert s._lg.waiting() == "step 'time'" and s.derived is None
    s.time()
    assert s._lg.waiting() == "gate G1"
