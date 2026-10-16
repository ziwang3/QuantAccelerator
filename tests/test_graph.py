"""The LangGraph orchestration (graph/): stage order, gates as interrupts, the backward-loop edge, and resuming a run
from its checkpoint after a crash or a pause at a gate, with the stage functions stubbed (the stages themselves are
tested elsewhere)."""
import json

import pytest

from quantaccelerator.agents import orchestrator as O
from quantaccelerator.graph import predict as P
from quantaccelerator.graph.state import Runtime
from quantaccelerator.llm.client import ScriptedClient
from quantaccelerator.state.schemas import (CandidateFeatureSet, DataInvestigationRequest, EDAWrapUp,
                                            GateDecision, PredictiveWrapUp, Provenance)
from quantaccelerator.tools import registry


class Crash(RuntimeError):
    pass


@pytest.fixture
def stages(monkeypatch):
    """Stub every stage the predict graph calls; record the order. stages.fail = {stage: n} crashes a stage n times."""
    calls, fail = [], {}

    def rec(name, ret=None):
        def f(*a, **k):
            calls.append(name)
            if fail.get(name):
                fail[name] -= 1
                raise Crash(name)
            return ret(*a, **k) if callable(ret) else ret
        return f

    fs = CandidateFeatureSet(panel_hash="ph", panel={}, features=[], ideas=[], candidates=[], observations=[],
                             wrapup=EDAWrapUp(summary="stub"), provenance=Provenance(producer_agent="oracle"),
                             set_hash="h1")
    wrap = PredictiveWrapUp(summary="stub")

    def predict(*a, **k):
        registry.CTX.findings["research_observations"] = ["o1", "o2"]   # what later stages read from the notebook
        return stages.wrap, ["o1", "o2"], {"tool_runs": [], "prompt_hash": None}

    def critique(*a, **k):
        assert registry.CTX.findings.get("research_observations") == ["o1", "o2"]
        return a[4], []

    monkeypatch.setattr(O, "_save", lambda *a, **k: None)
    monkeypatch.setattr(O, "oracle_contract", rec("contract", GateDecision(decided_by="oracle", approved=True)))
    monkeypatch.setattr(O, "build_research_panel", rec("panel", (None, {"hash": "ph", "n_rows": 1, "n_dates": 1,
                                                                        "n_entities": 1})))
    monkeypatch.setattr(O, "oracle_feature_set", rec("features", fs))
    monkeypatch.setattr(O, "plan_predictions", rec("plan", "PLAN"))
    monkeypatch.setattr(O, "predict", rec("predict", predict))
    monkeypatch.setattr(O, "validate_observations", rec("validate", lambda obs: obs))
    monkeypatch.setattr(O, "critique", rec("critique", critique))
    monkeypatch.setattr(O, "cost_check", rec("costs", lambda obs: obs))
    monkeypatch.setattr(O, "investigate", rec("investigate", lambda *a, **k: ["answer"]))
    monkeypatch.setattr(O, "locked_test", rec("locked_test", lambda obs, g5: obs + ["locked"]))
    monkeypatch.setattr(O, "_ledger_summary", lambda study: {"study": study})
    monkeypatch.setattr(O, "_prov", lambda *a, **k: None)
    monkeypatch.setattr(Runtime, "panel", lambda self, s: None)
    monkeypatch.setattr(Runtime, "vault", lambda self, s: None)
    import quantaccelerator.agents.predictive_report as pr
    import quantaccelerator.state.schemas as sc
    monkeypatch.setattr(pr, "build", lambda run_dir: None)
    monkeypatch.setattr(sc, "PredictiveReport", lambda **k: {"obs": k["observations"], "ledger": k["ledger"]})
    stages.calls, stages.fail, stages.wrap = calls, fail, wrap
    return stages


def approve_all(seen):
    def answer(gate, obj):
        seen.append((gate, obj))
        return GateDecision(gate=gate, decided_by="oracle", approved=True)
    return answer


def run(tmp_path, resume=False, resumer=None):
    return P.run_predict("g-test", run_dir=tmp_path, client=ScriptedClient([]), resume=resume,
                         resumer=resumer or approve_all([]))


ORDER = ["contract", "panel", "features", "plan", "predict", "validate", "critique", "costs", "locked_test"]


def test_stages_run_in_order_and_each_gate_pauses_the_run(stages, tmp_path):
    seen = []
    rep = run(tmp_path, resumer=approve_all(seen))
    assert stages.calls == ORDER
    assert seen == [("G4", "PLAN"), ("G5", ["o1", "o2"])]
    assert rep["obs"] == ["o1", "o2", "locked"] and rep["ledger"]["study"] == "h1:g-test"
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["status"] == "ok" and meta["mode"] == "predict" and "G5" in meta["gates"]
    assert (tmp_path / "graph.sqlite").exists()


def test_investigation_requests_take_the_backward_loop_edge(stages, tmp_path):
    stages.wrap = PredictiveWrapUp(summary="s", data_investigation_requests=[DataInvestigationRequest(
        question="q", observation="o", route_to="data_research", evidence=[])])
    run(tmp_path)
    assert stages.calls == ORDER[:8] + ["investigate", "locked_test"]


def test_a_crashed_run_resumes_from_its_checkpoint_in_a_new_process(stages, tmp_path):
    stages.fail["critique"] = 1
    with pytest.raises(Crash):
        run(tmp_path)
    assert json.loads((tmp_path / "run_meta.json").read_text())["status"].startswith("failed")
    registry.set_session(None)   # a new process: no session, no findings
    stages.calls.clear()
    rep = run(tmp_path, resume=True)
    assert stages.calls == ["critique", "costs", "locked_test"]   # nothing before the crash runs again
    assert rep["obs"] == ["o1", "o2", "locked"]
    assert json.loads((tmp_path / "run_meta.json").read_text())["status"] == "ok"


def test_a_run_paused_at_a_gate_is_answered_later(stages, tmp_path):
    class Away(Exception):
        pass

    def nobody(gate, obj):
        if gate == "G5":
            raise Away   # the process ends while G5 waits for a person
        return GateDecision(gate=gate, decided_by="oracle", approved=True)

    with pytest.raises(Away):
        run(tmp_path, resumer=nobody)
    registry.set_session(None)
    stages.calls.clear()
    seen = []
    run(tmp_path, resume=True, resumer=approve_all(seen))
    assert seen == [("G5", ["o1", "o2"])] and stages.calls == ["locked_test"]


def test_a_fresh_run_with_the_same_id_starts_a_new_thread(stages, tmp_path):
    run(tmp_path)
    first = (tmp_path / "graph_thread.txt").read_text()
    stages.calls.clear()
    run(tmp_path)
    assert stages.calls == ORDER and (tmp_path / "graph_thread.txt").read_text() != first


def test_ledger_rows_are_tagged_with_the_node_execution(tmp_path):
    import quantaccelerator.tools  # noqa: F401  (registers the tools)
    registry.set_session("g-ledger", tmp_path)
    registry.CTX.node_exec = "plan#abc"
    try:
        r = registry.TOOLS["next_tradable"].fn(timestamp="2025-03-07 16:01", timezone="America/New_York")
    finally:
        registry.CTX.node_exec = None
    native = registry.TOOLS["next_tradable"].fn(timestamp="2025-03-07 16:01", timezone="America/New_York")
    runs = registry.session_runs("g-ledger").set_index("run_id")
    assert runs.loc[r["run_id"], "node_exec"] == "plan#abc" and runs.loc[native["run_id"], "node_exec"] is None
