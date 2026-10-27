"""The whole research workflow as one LangGraph StateGraph: the notebook's session (notebook.Session).

  understand -> [G0] -> clean -> (time) -> propose_contract -> [G1] -> sign_contract -> (explore: ideas) -> panel
  -> propose_ideas -> [G2] -> build_ideas -> (investigate_eda) -> [G3] -> freeze -> (plan) -> plan -> [G4] -> predict
  -> validate -> critique -> costs -> (investigate_pred) -> [G5] -> locked_test -> report

[Gx] pauses for a decision, (step) for the person to go on (with the step's input, e.g. the researcher's ideas). A
rejected G2 or G3 goes back to (explore), a rejected G4 back to (plan). Every node is checkpointed in
runs/sessions/<id>/graph.sqlite, so a session continues after a kernel restart: Session(session_id=...).
"""
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from quantaccelerator.datasets import active
from quantaccelerator.graph.gates import step_node
from quantaccelerator.graph.state import RunState, Runtime, saver, thread_id
from quantaccelerator.graph.steps import cleaning_steps, contract_steps, ideas_steps, phase_b_steps, understand_node


def build(rt: Runtime) -> StateGraph:
    g = StateGraph(RunState)
    understand_node(rt, g, research=None, overview=True)
    cleaning_steps(rt, g, after="step_time")
    g.add_node("step_time", step_node("time"))
    g.add_edge("step_time", "propose_contract")
    contract_steps(rt, g, after="step_explore")
    g.add_node("step_explore", step_node("explore"))
    g.add_edge("step_explore", "panel")
    has_b = bool(active().predict)
    ideas_steps(rt, g, after="step_plan" if has_b else END, rejected="step_explore")
    if has_b:
        g.add_node("step_plan", step_node("plan"))
        g.add_edge("step_plan", "plan")
        phase_b_steps(rt, g, after=END, rejected="step_plan")
    g.add_edge(START, "understand")
    return g


class SessionGraph:
    """The notebook's handle on its graph: start, resume with a decision or a step's input, and read where it waits."""

    def __init__(self, rt: Runtime):
        self.rt = rt
        self.graph = build(rt).compile(checkpointer=saver(rt.run_dir))
        self.started = (rt.run_dir / "graph_thread.txt").exists()
        self.thread = thread_id(rt.run_id, rt.run_dir, resume=self.started) if self.started else None

    @property
    def cfg(self) -> dict:
        return {"configurable": {"thread_id": self.thread}, "recursion_limit": 200}

    def start(self, options: dict) -> None:
        if self.started:
            raise RuntimeError("this session's workflow has already started; continue it from where it waits "
                               f"({self.waiting()})")
        self.thread, self.started = thread_id(self.rt.run_id, self.rt.run_dir, resume=False), True
        self.graph.invoke({"run_id": self.rt.run_id, "mode": "session", "oracle": False, "options": options},
                          self.cfg)

    def pending(self) -> dict | None:
        """The interrupt the run waits at ({'gate': ..., 'object': ...} or {'step': ...}), or None."""
        if not self.started:
            return None
        snap = self.graph.get_state(self.cfg)
        return snap.interrupts[0].value if snap.interrupts else None

    def waiting(self) -> str:
        p = self.pending()
        return "not started" if not self.started else "finished" if p is None else \
            f"gate {p['gate']}" if "gate" in p else f"step '{p['step']}'"

    def expect(self, what: str, hint: str) -> None:
        p = self.pending()
        if p is None or (p.get("gate") or p.get("step")) != what:
            raise RuntimeError(f"the session waits at {self.waiting()}, not {what}: {hint}")

    def resume(self, value) -> None:
        # a step with no input resumes with True: in langgraph 1.2.15 Command(resume=None) fails (unbound resume_is_map)
        # and resume={} is read as an empty map of interrupt ids, which resumes nothing
        self.graph.invoke(Command(resume=True if value is None else value), self.cfg)

    @property
    def values(self) -> dict:
        return self.graph.get_state(self.cfg).values if self.started else {}
