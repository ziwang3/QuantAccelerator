"""Research EDA phase B as a LangGraph StateGraph.

  setup -> plan -> [G4] -> predict -> validate -> critique -> costs -> (investigate_pred) -> [G5] -> locked_test -> report

Checkpointed after every node in runs/<id>/graph.sqlite: a run killed anywhere continues with `resume=True`, and the
locked test still opens at most once (the vault refuses a second access).
"""
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from quantaccelerator.agents import orchestrator as O
from quantaccelerator.graph.gates import answer_with, drive
from quantaccelerator.graph.state import RunState, Runtime, saver, thread_id
from quantaccelerator.graph.steps import phase_b_steps
from quantaccelerator.llm.client import LLMClient
from quantaccelerator.state.schemas import PITContract
from quantaccelerator.tools import registry


def build(rt: Runtime) -> StateGraph:
    def setup(s):
        features = s["options"]["features"]
        contract = O.oracle_contract()
        if features != "oracle" and (Path(features) / "state" / "pit_contract.json").exists():
            contract = PITContract.model_validate_json((Path(features) / "state" / "pit_contract.json").read_text())
        O._save(rt.run_dir, "pit_contract", contract)
        registry.CTX.agent = "orchestrator"
        _, pmeta = O.build_research_panel(rt.run_dir, contract)
        rt.meta["panel"] = {k: pmeta[k] for k in ("hash", "n_rows", "n_dates", "n_entities")}
        fs = O.oracle_feature_set(rt.run_dir) if features == "oracle" else O.load_feature_set(rt.run_dir,
                                                                                             Path(features))
        study = f"{fs.set_hash}:{rt.run_id}" if features == "oracle" else fs.set_hash
        return {"art": {"contract": contract, "panel_hash": pmeta["hash"], "fs": fs, "study": study}}

    g = StateGraph(RunState)
    g.add_node("setup", rt.node("setup", setup))
    phase_b_steps(rt, g, after=END)
    g.add_edge(START, "setup")
    g.add_edge("setup", "plan")
    return g


def run_predict(run_id: str, run_dir: Path | None = None, client: LLMClient | None = None, oracle: bool = True,
                features: str = "oracle", resume: bool = False, resumer=None):
    """Research EDA phase B; returns the PredictiveReport. `resume` continues a checkpointed run."""
    run_dir, client, meta, timed = O._start(run_id, run_dir, client, mode="predict", oracle=oracle,
                                            features=features)
    rt = Runtime(run_id, run_dir, client, meta, timed)
    graph = build(rt).compile(checkpointer=saver(run_dir))
    inp = None if resume else {"run_id": run_id, "mode": "predict", "oracle": oracle,
                               "options": {"features": str(features)}}
    try:
        out = drive(graph, thread_id(run_id, run_dir, resume), inp, resumer or answer_with(oracle))
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        O._finish(run_dir, client, meta)
    return out["art"]["report"]
