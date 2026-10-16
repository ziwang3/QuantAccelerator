"""The audit demo as a LangGraph StateGraph.

  understand -> propose_contract -> [G1] -> sign_contract -> audit
"""
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from quantaccelerator.agents import brief
from quantaccelerator.agents import orchestrator as O
from quantaccelerator.graph.gates import answer_with, drive
from quantaccelerator.graph.state import RunState, Runtime, saver, thread_id
from quantaccelerator.graph.steps import contract_steps, understand_node
from quantaccelerator.llm.client import LLMClient


def build(rt: Runtime, research: bool) -> StateGraph:
    def audit(s):
        a = s["art"]
        return {"art": {"report": O.audit(rt.run_dir, rt.client, rt.timed, a["contract"], s["options"]["pipeline"],
                                          rt.meta)}}

    g = StateGraph(RunState)
    understand_node(rt, g, research=research)
    contract_steps(rt, g, after="audit")
    g.add_node("audit", rt.node("audit", audit))
    g.add_edge(START, "understand")
    g.add_edge("understand", "propose_contract")
    g.add_edge("audit", END)
    return g


def run_demo(pipeline: str, oracle: bool, run_id: str, run_dir: Path | None = None, client: LLMClient | None = None,
             research: bool = False, resume: bool = False, resumer=None) -> dict:
    """One Look-Ahead Audit on the graph; returns the run meta. `resume` continues a checkpointed run."""
    run_dir, client, meta, timed = O._start(run_id, run_dir, client, pipeline=pipeline, oracle=oracle,
                                            research=research)
    rt = Runtime(run_id, run_dir, client, meta, timed)
    graph = build(rt, research).compile(checkpointer=saver(run_dir))
    inp = None if resume else {"run_id": run_id, "mode": "audit", "oracle": oracle, "options": {"pipeline": pipeline}}
    meta["gate_interactions"] = 1
    try:
        drive(graph, thread_id(run_id, run_dir, resume), inp, resumer or answer_with(oracle))
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        O._finish(run_dir, client, meta)
    if research:
        brief.build(run_dir)
    return meta
