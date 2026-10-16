"""The workflow's stages as reusable groups of LangGraph nodes; each mode's graph (graph/understand, audit, explore,
predict, session) is assembled from them.

Every stage node calls the orchestrator's stage function unchanged (agents/orchestrator.py). Gates are interrupts
(graph/gates.py). A group ends in the node name `after`. When `rejected` is given, a gate's rejection routes there (the
notebook's session goes back to the step that can try again); without it a rejection stops the run (the next stage
raises).
"""
from langgraph.graph import StateGraph

from quantaccelerator.agents import orchestrator as O
from quantaccelerator.graph.gates import gate_node
from quantaccelerator.graph.state import Runtime
from quantaccelerator.tools import registry


def _trace(tr: dict) -> dict:
    return {k: tr[k] for k in ("tool_runs", "prompt_hash")}


def _edges(g: StateGraph, *pairs) -> None:
    for a, b in pairs:
        g.add_edge(a, b)


def _approved(gate: str, yes: str, no: str):
    return lambda s: yes if s["gates"][gate].approved else no


def understand_node(rt: Runtime, g: StateGraph, research: bool | None, overview: bool | None = None) -> None:
    """Data Research Agent: DatasetCard (+ overview, + research findings and the cleaning proposal). research=None
    takes it from the run's options."""
    g.add_node("understand", rt.node("understand", lambda s: {"art": {"card": O.understand(
        rt.run_dir, rt.client, rt.timed, s["options"].get("research", True) if research is None else research,
        overview)}}))


def _proposed(s) -> bool:
    card = s["art"].get("card")
    return bool(card is not None and card.research and card.research.cleaning.steps)


def cleaning_steps(rt: Runtime, g: StateGraph, after: str) -> None:
    """understand -> [G0] -> clean -> `after`: the approved recipe is applied to derived copies (raw data untouched).
    G0 pauses only when a cleaning recipe was proposed; without one the tables are used as ingested."""
    def clean(s):
        from quantaccelerator.tools.clean import apply_recipe
        g0 = s.get("gates", {}).get("G0") if _proposed(s) else None
        if g0 is None:
            return {"art": {"derived": None, "recipe_hash": None}}
        recipe = g0.approved_recipe if g0.approved else None
        O._save(rt.run_dir, "gate_g0", g0)
        if recipe is None or not recipe.steps:
            return {"art": {"derived": None, "recipe_hash": None}}
        out = apply_recipe(recipe)
        rt.meta["cleaning"] = {"recipe_hash": out["recipe_hash"], "tables": out["tables"], "effects": out["effects"]}
        return {"art": {"derived": out["tables"], "recipe_hash": out["recipe_hash"]}}

    g.add_node("G0", gate_node("G0", lambda s: s["art"]["card"]))
    g.add_node("clean", rt.node("clean", clean))
    g.add_conditional_edges("understand", lambda s: "G0" if _proposed(s) else "clean", ["G0", "clean"])
    _edges(g, ("G0", "clean"), ("clean", after))


def contract_steps(rt: Runtime, g: StateGraph, after: str) -> None:
    """propose_contract -> [G1] -> sign_contract -> `after`."""
    def propose(s):
        draft, tr = O.propose_contract(rt.client, rt.timed, s["art"]["card"])
        return {"art": {"draft": draft, "draft_trace": _trace(tr)}}

    def sign(s):
        a = s["art"]
        return {"art": {"contract": O.sign_contract(rt.run_dir, rt.client, a["draft"], a["draft_trace"],
                                                    s["gates"]["G1"])}}

    g.add_node("propose_contract", rt.node("propose_contract", propose))
    g.add_node("G1", gate_node("G1", lambda s: s["art"]["draft"]))
    g.add_node("sign_contract", rt.node("sign_contract", sign))
    _edges(g, ("propose_contract", "G1"), ("G1", "sign_contract"), ("sign_contract", after))


def ideas_steps(rt: Runtime, g: StateGraph, after: str, rejected: str | None = None) -> None:
    """panel -> propose_ideas -> [G2] -> build_ideas -> (investigate_eda) -> [G3] -> freeze -> `after`."""
    def panel(s):
        registry.CTX.agent = "orchestrator"
        _, pmeta = O.build_research_panel(rt.run_dir, s["art"]["contract"], s["art"].get("derived"))
        rt.meta["panel"] = {k: pmeta[k] for k in ("hash", "n_rows", "n_dates", "n_entities")}
        return {"art": {"panel_hash": pmeta["hash"]}}

    def propose_ideas(s):
        rt.panel(s)
        plan, ctx = O.propose_ideas(rt.run_dir, rt.client, rt.timed, s["art"].get("card"),
                                    s["options"].get("researcher_ideas"))
        return {"art": {"idea_plan": plan, "idea_ctx": ctx}}

    def build_ideas(s):
        rt.panel(s)
        a = s["art"]
        fs = O.build_ideas(rt.run_dir, rt.client, rt.timed, a["idea_plan"], s["gates"]["G2"], dict(a["idea_ctx"]))
        return {"art": {"fs": fs, "investigations": []}}

    def investigate(s):
        rt.panel(s)
        a = s["art"]
        out = O.investigate(rt.run_dir, rt.client, rt.timed, a["fs"].wrapup.data_investigation_requests,
                            "research_eda", a.get("card"), a["contract"], a["fs"].features)
        rt.meta["investigations"] = len(out)
        return {"art": {"investigations": out}}

    def freeze(s):
        rt.panel(s)
        return {"art": {"fs": O.freeze_features(rt.run_dir, s["art"]["fs"], s["gates"]["G3"])}}

    for name, fn in [("panel", panel), ("propose_ideas", propose_ideas), ("build_ideas", build_ideas),
                     ("investigate_eda", investigate), ("freeze", freeze)]:
        g.add_node(name, rt.node(name, fn))
    g.add_node("G2", gate_node("G2", lambda s: s["art"]["idea_plan"]))
    g.add_node("G3", gate_node("G3", lambda s: s["art"]["fs"]))
    _edges(g, ("panel", "propose_ideas"), ("propose_ideas", "G2"), ("investigate_eda", "G3"), ("G3", "freeze"))
    if rejected:
        g.add_conditional_edges("G2", _approved("G2", "build_ideas", rejected), ["build_ideas", rejected])
        g.add_conditional_edges("freeze", _approved("G3", after, rejected), [after, rejected])
    else:
        _edges(g, ("G2", "build_ideas"), ("freeze", after))
    g.add_conditional_edges(
        "build_ideas", lambda s: "investigate_eda" if s["art"]["fs"].wrapup.data_investigation_requests else "G3",
        ["investigate_eda", "G3"])


def phase_b_steps(rt: Runtime, g: StateGraph, after: str, rejected: str | None = None) -> None:
    """plan -> [G4] -> predict -> validate -> critique -> costs -> (investigate_pred) -> [G5] -> locked_test -> report
    -> `after`. The study is art['study'] when set (the CLI's benchmark repeats), else the feature set's hash."""
    def plan(s):
        rt.panel(s)
        fs = s["art"]["fs"]
        return {"art": {"plan": O.plan_predictions(rt.run_dir, rt.client, rt.timed, fs),
                        "study": s["art"].get("study") or fs.set_hash}}

    def predict(s):
        a = s["art"]
        wrap, obs, tr = O.predict(rt.run_dir, rt.client, rt.timed, a["fs"], s["gates"]["G4"], rt.panel(s), a["study"])
        return {"art": {"wrap": wrap, "obs": obs, "trace": tr, "investigations": []}}

    def validate(s):
        rt.vault(s)
        return {"art": {"obs": O.validate_observations(s["art"]["obs"])}}

    def critique(s):
        rt.vault(s)
        obs, crits = O.critique(rt.run_dir, rt.client, rt.timed, s["art"]["fs"], s["art"]["obs"], rt.panel(s))
        return {"art": {"obs": obs, "crits": crits}}

    def costs(s):
        rt.vault(s)
        return {"art": {"obs": O.cost_check(s["art"]["obs"])}}

    def investigate(s):
        a = s["art"]
        rt.panel(s)
        out = O.investigate(rt.run_dir, rt.client, rt.timed, a["wrap"].data_investigation_requests, "predictive_eda",
                            a.get("card"), a["contract"], a["fs"].features)
        rt.meta["investigations"] = len(out)
        return {"art": {"investigations": out}}

    def locked_test(s):
        from quantaccelerator.returns import vault
        rt.vault(s)
        obs = O.locked_test(s["art"]["obs"], s["gates"]["G5"])
        return {"art": {"obs": obs}, "released": bool(vault.STATE["released"])}

    def report(s):
        from quantaccelerator.agents import predictive_report
        from quantaccelerator.state.schemas import PredictiveReport
        a, g4, g5 = s["art"], s["gates"]["G4"], s["gates"]["G5"]
        rep = PredictiveReport(set_hash=a["fs"].set_hash, plan=g4.approved_plan, plan_gate=g4, observations=a["obs"],
                               wrapup=a["wrap"], critiques=a["crits"], release_gate=g5,
                               ledger=O._ledger_summary(a["study"]),
                               provenance=O._prov("predictive_eda", a["trace"], rt.client,
                                                  ["state/candidate_feature_set.json", "state/prediction_plan.json"]))
        O._save(rt.run_dir, "predictive_report", rep)
        predictive_report.build(rt.run_dir)
        rt.meta["gates"] = {"G4": g4.model_dump(mode="json", exclude={"approved_plan"}), "G5": g5.model_dump(mode="json")}
        return {"art": {"report": rep}}

    for name, fn in [("plan", plan), ("predict", predict), ("validate", validate), ("critique", critique),
                     ("costs", costs), ("investigate_pred", investigate), ("locked_test", locked_test),
                     ("report", report)]:
        g.add_node(name, rt.node(name, fn))
    g.add_node("G4", gate_node("G4", lambda s: s["art"]["plan"]))
    g.add_node("G5", gate_node("G5", lambda s: s["art"]["obs"]))
    _edges(g, ("plan", "G4"), ("predict", "validate"), ("validate", "critique"), ("critique", "costs"),
           ("investigate_pred", "G5"), ("G5", "locked_test"), ("locked_test", "report"), ("report", after))
    if rejected:
        g.add_conditional_edges("G4", _approved("G4", "predict", rejected), ["predict", rejected])
    else:
        g.add_edge("G4", "predict")
    g.add_conditional_edges("costs", lambda s: "investigate_pred" if s["art"]["wrap"].data_investigation_requests
                            else "G5", ["investigate_pred", "G5"])
