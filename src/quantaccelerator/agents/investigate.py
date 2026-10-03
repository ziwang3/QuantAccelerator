"""The backward loop: questions research cannot answer itself (DataInvestigationRequest), routed back upstream.

Research EDA (both phases) may end with questions about the data itself ("what caused the shift in this field's
median around 2022-06-29?") or about its timing. The orchestrator routes each to the agent that owns it: the Data
Research Agent (in Q&A mode, with its profiling tools) or the PIT agent (in Q&A mode, with its timing tools and the
signed contract). Answers must rest on cited tool runs, like every other answer, and are shown to the researcher at
the next gate (G3 after phase A, G5 after phase B). A request that cannot be answered is recorded as such.
"""
from quantaccelerator.agents.profiler import DataQA


def _ok(run: dict) -> bool:
    res = run.get("result")
    return not (isinstance(res, dict) and "error" in res)
from quantaccelerator.datasets import active

PIT_QA_ROLE = """You are the Point-in-Time (PIT) agent answering a question that research raised about the timing of
a dataset whose availability rule you signed (the contract is in the context). Answer with the tools: check before you
claim, cite the run_ids, take numbers and dates only from tool results, and say plainly when the data cannot answer
the question. Keep the answer short."""

TASK = """A question routed back from {stage}:
Question: {question}
What prompted it: {observation}
Answer it with your tools, cite the runs, and submit. If the data cannot answer it, say so and why. A question about a
constructed feature is answered on the research panel (time_stability, compute_exposure: its universe and dates, the
feature as built), then traced to the raw fields it is built from (and the documentation) for the cause."""


class PITQA(DataQA):
    name = "pit_qa"
    role_prompt = PIT_QA_ROLE

    @property
    def tool_names(self) -> list[str]:
        return list(active().pit_tools)

    def extra_validate(self, draft) -> list[str]:
        from quantaccelerator.agents.checks import _runs
        if not [r for r in _runs(draft.evidence + draft.figure_run_ids) if r["tool"] in self.tool_names and _ok(r)]:
            return [f"cite the run_id of at least one successful call of your tools ({', '.join(self.tool_names)}) "
                    "behind the answer (a failed call is not evidence)"]
        return []


class FeatureQA(DataQA):
    """The Data Research Agent answering a question about a constructed feature: its raw-table tools plus the research
    panel's feature tools (the panel's universe and dates, the feature as built)."""
    name = "feature_qa"
    PANEL_TOOLS = ["describe_panel", "time_stability", "compute_exposure"]

    @property
    def tool_names(self) -> list[str]:
        return self.PANEL_TOOLS + super().tool_names

    def extra_validate(self, draft) -> list[str]:
        from quantaccelerator.agents.checks import DATE_TOLERANCE_DAYS, _dates, _runs
        import json
        runs = [r for r in _runs(draft.evidence + draft.figure_run_ids) if r["tool"] in self.tool_names and _ok(r)]
        if not any(r["tool"] in self.PANEL_TOOLS for r in runs):
            return ["the question is about a constructed feature: examine it on the research panel first, e.g. "
                    "time_stability(feature='<the feature>') (its median, spread and breaks per date), and cite that "
                    "successful run (a failed call is not evidence)"]
        known = [d for r in runs for d in _dates(json.dumps(r.get("result"), default=str))] + _dates(self.given)
        for d in _dates(draft.answer):
            if not known or min(abs((d - k).days) for k in known) > DATE_TOLERANCE_DAYS:
                return [f"the answer mentions {d.date()} but no cited tool result has a date near it"]
        return []
