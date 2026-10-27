"""Human gates as LangGraph interrupts, and the drivers that answer them.

A gate node calls interrupt({"gate", "object"}) and waits; Command(resume=GateDecision) continues the run. The same
graph serves benchmarks and people: `answer_with(oracle=True)` resumes with the orchestrator's own oracle rules, the
console resumer asks a person at the terminal, and a notebook or service can resume a paused run later (the
checkpoint holds it) with a decision made anywhere.
"""
from langgraph.types import Command, interrupt

from quantaccelerator.agents import orchestrator as O
from quantaccelerator.state.schemas import GateDecision

GATE_FN = {"G0": O.gate_cleaning, "G1": O.gate_g1, "G2": O.gate_ideas, "G3": O.gate_freeze, "G4": O.gate_plan,
           "G5": O.gate_release}


def gate_node(gate: str, obj_of):
    """A node that pauses for gate `gate`; obj_of(state) is the object under review (what the gate function shows)."""
    def run(state) -> dict:
        d = interrupt({"gate": gate, "object": obj_of(state)})
        return {"gates": {gate: d if isinstance(d, GateDecision) else GateDecision.model_validate(d)}}
    run.__name__ = gate
    return run


def step_node(step: str):
    """A pause between stages that a person starts (the notebook's s.time(), s.explore(ideas), s.plan_predictions());
    the resume value is the step's input (a dict, merged into the run's options) or None."""
    def run(state) -> dict:
        given = interrupt({"step": step})
        return {"options": {**state.get("options", {}), **(given if isinstance(given, dict) else {})}}
    run.__name__ = f"step_{step}"
    return run


def answer_with(oracle: bool):
    """Resumer from the orchestrator's gate functions: the oracle rules, or the console prompts."""
    return lambda gate, obj: GATE_FN[gate](obj, oracle)


def drive(graph, thread_id: str, inp, resumer, on_interrupt=None) -> dict:
    """Run (inp = initial state) or continue (inp = None) a run, answering every gate with resumer(gate, object)."""
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    out = graph.invoke(inp, cfg)
    while out.get("__interrupt__"):
        req = out["__interrupt__"][0].value
        if "gate" not in req:
            raise RuntimeError(f"run paused at step {req.get('step')!r}: a session graph is driven by its notebook")
        if on_interrupt is not None:
            on_interrupt(req)
        out = graph.invoke(Command(resume=resumer(req["gate"], req["object"])), cfg)
    return out
