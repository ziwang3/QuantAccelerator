"""The Understand step as a LangGraph StateGraph: the DatasetCard, the research findings and the cleaning proposal,
which goes through gate G0 as in the notebook (when a recipe was proposed).

  understand -> [G0] -> clean

The oracle approves the recipe as proposed (the benchmark scores the proposal itself); the approved recipe is applied
to derived copies under data/derived/<dataset>/<recipe hash>/ (raw tables are never modified).
"""
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from quantaccelerator.agents import brief
from quantaccelerator.agents import orchestrator as O
from quantaccelerator.graph.gates import answer_with, drive
from quantaccelerator.graph.state import RunState, Runtime, saver, thread_id
from quantaccelerator.graph.steps import cleaning_steps, understand_node
from quantaccelerator.llm.client import LLMClient


def build(rt: Runtime, research: bool) -> StateGraph:
    g = StateGraph(RunState)
    understand_node(rt, g, research=research)
    cleaning_steps(rt, g, after=END)
    g.add_edge(START, "understand")
    return g


def run_understand(run_id: str, run_dir: Path | None = None, client: LLMClient | None = None, research: bool = True,
                   oracle: bool = True, resume: bool = False, resumer=None):
    """The Understand step with gate G0; returns the DatasetCard. `resume` continues a checkpointed run."""
    run_dir, client, meta, timed = O._start(run_id, run_dir, client, mode="understand", oracle=oracle)
    rt = Runtime(run_id, run_dir, client, meta, timed)
    graph = build(rt, research).compile(checkpointer=saver(run_dir))
    inp = None if resume else {"run_id": run_id, "mode": "understand", "oracle": oracle,
                               "options": {"research": research}}
    try:
        out = drive(graph, thread_id(run_id, run_dir, resume), inp, resumer or answer_with(oracle))
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        O._finish(run_dir, client, meta)
    brief.build(run_dir)
    return out["art"]["card"]
