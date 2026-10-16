"""Graph state and the process-side objects of a LangGraph run.

RunState is what the checkpointer stores after every node: the run's identity and options, the stage artifacts (the
pydantic objects the stage functions hand each other), the gate decisions, registry.CTX.findings (the record_* tools'
notebook, which later stages read) and the run meta. Live objects (the LLM client, the panel DataFrame, the armed
vault) are not state: each node rebuilds them from state on entry when the process does not hold them (a resume in a
new process), so a run continues correctly from any checkpoint.
"""
import inspect
import operator
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from quantaccelerator.tools import registry


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class RunState(TypedDict, total=False):
    run_id: str
    mode: str
    oracle: bool
    options: dict                               # mode options (features source, pit, researcher ideas, pipeline, ...)
    art: Annotated[dict, _merge]                # stage artifacts by name
    gates: Annotated[dict, _merge]              # G1..G5 -> GateDecision
    findings: dict                              # registry.CTX.findings after the last node
    reference_rule: Any                         # registry.CTX.reference_rule (the signed availability rule)
    meta: dict                                  # run_meta.json so far
    budget: dict                                # LLM budget used so far
    released: bool                              # vault: G5 released observations (the locked test is touched once)
    executions: Annotated[list, operator.add]   # '<node>#<execution id>' of every completed node, in order


def serializer() -> JsonPlusSerializer:
    """Checkpoint serializer that may rebuild the project's pydantic models (an explicit allowlist)."""
    from quantaccelerator.state import schemas
    allowed = [(schemas.__name__, n) for n, c in vars(schemas).items()
               if inspect.isclass(c) and issubclass(c, BaseModel) and c.__module__ == schemas.__name__]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def saver(run_dir: Path) -> SqliteSaver:
    """The run's checkpointer: runs/<id>/graph.sqlite (one thread per fresh run, see thread_id)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(run_dir / "graph.sqlite", check_same_thread=False), serde=serializer())


def thread_id(run_id: str, run_dir: Path, resume: bool) -> str:
    """A fresh run starts a new checkpoint thread (an earlier run with the same id keeps its own); resume continues
    the latest one, named in runs/<id>/graph_thread.txt."""
    f = Path(run_dir) / "graph_thread.txt"
    if resume:
        if not f.exists():
            raise FileNotFoundError(f"{run_dir}: no LangGraph checkpoint to resume (graph_thread.txt is missing)")
        return f.read_text().strip()
    t = f"{run_id}:{uuid.uuid4().hex[:8]}"
    f.write_text(t + "\n")
    return t


class Runtime:
    """Process-side objects of one run: the client, the run meta (shared with `timed`) and the stage helpers."""

    def __init__(self, run_id: str, run_dir: Path, client, meta: dict, timed):
        self.run_id, self.run_dir, self.client, self.meta, self.timed = run_id, Path(run_dir), client, meta, timed
        self.restored = False

    # ---------------------------------------------------------------- rebuilding globals from state
    def restore(self, state: RunState) -> None:
        if registry.CTX.session_id != self.run_id:   # a new process (or another session ran in between)
            registry.set_session(self.run_id, self.run_dir)
            registry.CTX.agent = "orchestrator"
        if not self.restored:   # first node in this process: continue the checkpointed meta and budget
            self.restored = True
            if state.get("meta"):
                saved = {k: v for k, v in state["meta"].items() if k != "status"}
                self.meta.update({**saved, "agents": {**saved.get("agents", {}), **self.meta.get("agents", {})}})
            b = state.get("budget") or {}
            for k in ("calls", "prompt_tokens", "completion_tokens"):
                setattr(self.client.budget, k, max(getattr(self.client.budget, k), b.get(k, 0)))
        if "findings" in state:
            registry.CTX.findings = state["findings"]
        if state.get("reference_rule") is not None:
            registry.CTX.reference_rule = state["reference_rule"]

    def panel(self, state: RunState):
        """The session panel, rebuilt from the signed contract when this process does not hold it."""
        from quantaccelerator.agents.orchestrator import build_research_panel
        from quantaccelerator.tools.panel import STATE, get_panel
        art = state["art"]
        if STATE["meta"] is None or STATE["meta"].get("hash") != art["panel_hash"]:
            build_research_panel(self.run_dir, art["contract"], art.get("derived"))
        return get_panel()

    def vault(self, state: RunState) -> None:
        """Arm the returns vault as `predict` armed it (arming reads nothing and logs no access)."""
        from quantaccelerator.agents.orchestrator import _conditioners
        from quantaccelerator.datasets import active
        from quantaccelerator.returns import vault
        art = state["art"]
        if not vault.armed() or vault.STATE["run_dir"] != self.run_dir or vault.STATE["study"] != art["study"]:
            vault.arm(self.run_dir, art["fs"], state["gates"]["G4"], active().predict["returns"],
                      _conditioners(self.panel(state)), study=art["study"], link=active().predict.get("link"))
        vault.STATE["released"] = bool(state.get("released"))

    def snapshot(self) -> dict:
        b = self.client.budget
        return {"findings": registry.CTX.findings, "reference_rule": registry.CTX.reference_rule, "meta": self.meta,
                "budget": {"calls": b.calls, "prompt_tokens": b.prompt_tokens,
                           "completion_tokens": b.completion_tokens}}

    # ---------------------------------------------------------------- nodes
    def node(self, name: str, fn):
        """A graph node around a stage step: restore globals, tag the ledger rows of this execution, snapshot after."""
        def run(state: RunState) -> dict:
            self.restore(state)
            exec_id = f"{name}#{uuid.uuid4().hex[:8]}"
            registry.CTX.node_exec = exec_id
            try:
                update = fn(state) or {}
            finally:
                registry.CTX.node_exec = None
            return {**self.snapshot(), **update, "executions": [exec_id]}
        run.__name__ = name
        return run
