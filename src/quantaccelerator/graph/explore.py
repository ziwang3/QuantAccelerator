"""Research EDA phase A as a LangGraph StateGraph.

  pit=agent:  understand -> propose_contract -> [G1] -> sign_contract ┐
  pit=oracle: oracle_contract ─────────────────────────────────────── ┴-> panel -> propose_ideas -> [G2] -> build_ideas
              -> (investigate_eda) -> [G3] -> freeze
"""
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from quantaccelerator.agents import orchestrator as O
from quantaccelerator.graph.gates import answer_with, drive
from quantaccelerator.graph.state import RunState, Runtime, saver, thread_id
from quantaccelerator.graph.steps import contract_steps, ideas_steps, understand_node
from quantaccelerator.llm.client import LLMClient


def build(rt: Runtime) -> StateGraph:
    def oracle_contract(s):
        contract = O.oracle_contract()
        O._save(rt.run_dir, "pit_contract", contract)
        return {"art": {"contract": contract, "card": None}}

    g = StateGraph(RunState)
    understand_node(rt, g, research=False)
    g.add_edge("understand", "propose_contract")
    contract_steps(rt, g, after="panel")
    g.add_node("oracle_contract", rt.node("oracle_contract", oracle_contract))
    g.add_edge("oracle_contract", "panel")
    ideas_steps(rt, g, after=END)
    g.add_conditional_edges(START, lambda s: "understand" if s["options"]["pit"] == "agent" else "oracle_contract",
                            ["understand", "oracle_contract"])
    return g


def run_explore(run_id: str, run_dir: Path | None = None, client: LLMClient | None = None, oracle: bool = True,
                pit: str = "oracle", researcher_ideas: list[str] | None = None, resume: bool = False, resumer=None):
    """Research EDA phase A; returns the frozen CandidateFeatureSet. `resume` continues a checkpointed run."""
    from quantaccelerator.agents import feature_report
    from quantaccelerator.datasets import active
    if researcher_ideas is None:
        researcher_ideas = list((active().panel or {}).get("researcher_ideas", []))
    run_dir, client, meta, timed = O._start(run_id, run_dir, client, mode="explore", oracle=oracle, pit=pit,
                                            researcher_ideas=researcher_ideas)
    rt = Runtime(run_id, run_dir, client, meta, timed)
    graph = build(rt).compile(checkpointer=saver(run_dir))
    inp = None if resume else {"run_id": run_id, "mode": "explore", "oracle": oracle,
                               "options": {"pit": pit, "researcher_ideas": researcher_ideas}}
    try:
        out = drive(graph, thread_id(run_id, run_dir, resume), inp, resumer or answer_with(oracle))
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        O._finish(run_dir, client, meta)
    feature_report.build(run_dir)
    return out["art"]["fs"]
