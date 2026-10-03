"""Plain-Python orchestrator for the audit-mode demo: Profiler -> PIT agent -> [G1] -> Code Auditor.

It validates every agent output against its schema, attaches provenance and deterministic facts, runs the human
gate G1 (or an oracle that answers from gold), and re-verifies each proposed patch with a deterministic re-run.
"""
import json
import time
from pathlib import Path


from quantaccelerator import build
from quantaccelerator.agents import brief
from quantaccelerator.agents.base import AgentFailure
from quantaccelerator.agents.auditor import TASK as AUDIT_TASK, Auditor
from quantaccelerator.agents.pit_agent import TASK as PIT_TASK, PITAgent
from quantaccelerator.agents.profiler import (OVERVIEW_TASK, RESEARCH_TASK, TASK as PROFILE_TASK, DataOverview,
                                               DataResearcher, Profiler)
from quantaccelerator.agents.research_eda import SURVEY_TASK, TASK as EDA_TASK, ResearchEDA, ResearchEDASurvey
from quantaccelerator.tools.findings import assemble
from quantaccelerator.datasets import active
from quantaccelerator.llm.client import Budget, LLMClient
from quantaccelerator.paths import ROOT, RUNS
from quantaccelerator.state.schemas import (AuditFinding, AuditReport, AvailabilityRule, CandidateFeatureSet, DataResearchDraft,
                               DatasetCard, FileRef, GateDecision, PITContract, Provenance)
from quantaccelerator.tools import registry
from quantaccelerator.tools.data import TABLES, table_path
from quantaccelerator.tools.pipeline import apply_edits


def rule_agreement(rule: AvailabilityRule) -> float:
    """Share of the gold rule sample where `rule` gives exactly the gold first tradable date (and vintage)."""
    return active().rule_agreement(rule)


def gate_g1(contract, oracle: bool) -> GateDecision:
    rule = contract.availability_rule
    if oracle:
        agree = rule_agreement(rule)
        if agree >= 0.99:
            return GateDecision(decided_by="oracle", approved=True, note=f"rule agrees with gold on {agree:.1%} of the "
                                                                          "stratified sample")
        return GateDecision(decided_by="oracle", approved=False, corrected_rule=active().gold_rule(),
                            note=f"rule agrees with gold on only {agree:.1%}; oracle substitutes the gold rule")
    print("\n=== G1: sign off the PIT contract ===")
    print(contract.rule_explanation)
    print("rule:", rule.model_dump_json())
    for alt in contract.alternatives:
        print(f"  alternative considered: {alt}")
    ans = input("Approve? [y] approve / [n] reject and enter a replacement rule as JSON: ").strip().lower()
    if ans in ("", "y", "yes"):
        return GateDecision(decided_by="human", approved=True)
    raw = input("Replacement AvailabilityRule JSON (empty = stop the run): ").strip()
    return GateDecision(decided_by="human", approved=False, note=input("Reason: "),
                        corrected_rule=AvailabilityRule.model_validate_json(raw) if raw else None)


def _save(run_dir: Path, name: str, obj) -> None:
    (run_dir / "state").mkdir(parents=True, exist_ok=True)
    (run_dir / "state" / f"{name}.json").write_text(obj.model_dump_json(indent=1))


def _prov(agent: str, trace: dict, client: LLMClient, inputs: list[str]) -> Provenance:
    return Provenance(producer_agent=agent, inputs=inputs, tool_runs=trace["tool_runs"], model=client.model,
                      prompt_hash=trace["prompt_hash"])


def _dump_trace(run_dir: Path, trace: dict) -> dict:
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (run_dir / "transcripts" / f"{trace['agent']}.json").write_text(
        json.dumps(trace.get("messages", []), indent=1, default=str))
    return {k: v for k, v in trace.items() if k != "messages"}


def _start(run_id: str, run_dir: Path | None, client: LLMClient | None, **meta_extra):
    run_dir = run_dir or RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    registry.set_session(run_id, run_dir)
    client = client or LLMClient(log_path=run_dir / "llm_calls.jsonl", budget=Budget())
    client.log_path = run_dir / "llm_calls.jsonl"
    meta = {"run_id": run_id, "dataset": active().name, **meta_extra, "model": client.model,
            "temperature": client.temperature,
            "seed": client.seed, "build": build.BUILD, "code_version": registry.code_version(), "agents": {},
            "status": "running"}

    def timed(agent, task, context=None):
        t0 = time.time()
        try:
            draft, trace = agent.run(client, task, context)
        except AgentFailure as e:
            meta["agents"].setdefault(agent.name, {}).update(_dump_trace(run_dir, {"agent": agent.name, **e.trace}))
            raise
        finally:
            meta["agents"].setdefault(agent.name, {})["wall_s"] = round(time.time() - t0, 1)
        meta["agents"][agent.name].update(_dump_trace(run_dir, trace))
        return draft, trace

    return run_dir, client, meta, timed


def _finish(run_dir: Path, client: LLMClient, meta: dict) -> None:
    b = client.budget
    meta["budget"] = {"calls": b.calls, "prompt_tokens": b.prompt_tokens, "completion_tokens": b.completion_tokens}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    registry.set_session(None)


def understand(run_dir: Path, client: LLMClient, timed, research: bool = True,
               overview: bool | None = None) -> DatasetCard:
    """Data Research Agent: pass 1 (semantics -> DatasetCard), the dataset overview (what it is, at a glance; on by
    default when researching a dataset, off in audit mode), then pass 2 (profiling -> research findings)."""
    draft, tr = timed(Profiler(), PROFILE_TASK)
    files = [FileRef(path=registry._project_path(table_path(t)), sha256=registry.file_sha256(table_path(t)))
             for t in TABLES]
    card = DatasetCard(**draft.model_dump(), files=files,
                       provenance=_prov("profiler", tr, client, [f.path for f in files]))
    _save(run_dir, "dataset_card", card)
    if research if overview is None else overview:
        ctx = {"dataset_card": {"dataset_id": card.dataset_id, "tables": [t.model_dump() for t in card.tables],
                                "fields": [{"field": f"{f.table}.{f.name}", "meaning": f.meaning, "kind": f.kind,
                                            "unit": f.unit} for f in card.fields]}}
        ov, _ = timed(DataOverview(), OVERVIEW_TASK, ctx)
        card = card.model_copy(update={"overview": ov})
        _save(run_dir, "dataset_card", card)
    if research:
        researcher = DataResearcher()
        researcher.card = card
        registry.CTX.findings = {}
        wrap, tr2 = timed(researcher, RESEARCH_TASK,
                          {"dataset_card": card.model_dump(mode="json", exclude={"provenance", "files"})})
        found = DataResearchDraft.model_validate(assemble(wrap).model_dump())
        tables, warnings = list(card.tables), list(card.warnings)
        for i, t in enumerate(tables):
            if (k := found.key_corrections.get(t.table)) and k != t.primary_key:
                tables[i] = t.model_copy(update={"primary_key": k})
                warnings.append(f"{t.table}: primary key corrected by the profiling pass from {t.primary_key} to {k}")
        card = card.model_copy(update={"tables": tables, "warnings": warnings, "research": found,
                                       "research_provenance": _prov("data_research", tr2, client,
                                                                    ["state/dataset_card.json",
                                                                     *[f.path for f in files]])})
        _save(run_dir, "dataset_card", card)
    return card


def run_understand(run_id: str, run_dir: Path | None = None, client: LLMClient | None = None,
                   research: bool = True) -> DatasetCard:
    """The Understand step on its own (the notebook's s.understand()): DatasetCard + research findings + brief."""
    run_dir, client, meta, timed = _start(run_id, run_dir, client, mode="understand")
    try:
        card = understand(run_dir, client, timed, research)
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        _finish(run_dir, client, meta)
    brief.build(run_dir)
    return card


def propose_contract(client: LLMClient, timed, card: DatasetCard):
    """PIT agent -> draft PITContract (not yet signed)."""
    return timed(PITAgent(), PIT_TASK, {"dataset_card": card.model_dump(
        mode="json", exclude={"provenance", "files", "research_provenance"})})


def sign_contract(run_dir: Path, client: LLMClient, draft, trace: dict, g1: GateDecision) -> PITContract:
    """Attach the G1 decision; the signed rule becomes the reference for every look-ahead measurement."""
    contract = PITContract(**draft.model_dump(), signed_off=g1,
                           provenance=_prov("pit_agent", trace, client, ["state/dataset_card.json"]))
    _save(run_dir, "pit_contract", contract)
    rule = draft.availability_rule if g1.approved else g1.corrected_rule
    if rule is None:
        raise RuntimeError("G1 rejected the PIT contract without a replacement rule; stopping")
    registry.CTX.reference_rule = rule
    return contract


def audit(run_dir: Path, client: LLMClient, timed, contract: PITContract, pipeline: str, meta: dict) -> AuditReport:
    """Code Auditor -> AuditReport, each patch re-verified deterministically."""
    rule = registry.CTX.reference_rule
    draft, tr = timed(Auditor(), AUDIT_TASK.format(pipeline=pipeline),
                      {"pit_contract": {"availability_rule": rule.model_dump(),
                                        "rule_explanation": contract.rule_explanation,
                                        "fields": [f.model_dump() for f in contract.fields]}})
    registry.CTX.agent = "orchestrator"
    prov = _prov("auditor", tr, client, ["state/pit_contract.json", pipeline])
    findings = []
    for f in draft.findings:
        v = registry.TOOLS["test_patch"].fn(path=pipeline, edits=[e.model_dump() for e in f.proposed_patch])
        findings.append(AuditFinding(**f.model_dump(), pipeline_ref=pipeline,
                                     verified_by_run=v["run_id"] if v["ok"] else None,
                                     verified_share_before_tradable=v["result"]["lookahead"]["share_before_tradable"]
                                     if v["ok"] else None, provenance=prov))
    report = AuditReport(pipeline=pipeline, findings=findings, summary=draft.summary, evidence=draft.evidence,
                         provenance=prov)
    _save(run_dir, "audit_report", report)
    if findings:
        edits = [e.model_dump() for f in draft.findings for e in f.proposed_patch]
        try:
            (run_dir / "patched.py").write_text(apply_edits((ROOT / pipeline).read_text(), edits))
        except ValueError as e:
            meta["patch_error"] = str(e)
    return report


def run_demo(pipeline: str, oracle: bool, run_id: str, run_dir: Path | None = None,
             client: LLMClient | None = None, research: bool = False) -> dict:
    run_dir, client, meta, timed = _start(run_id, run_dir, client, pipeline=pipeline, oracle=oracle,
                                          research=research)
    try:
        # 1. Data Research Agent -> DatasetCard (pass 2, the profiling loop, only when asked)
        card = understand(run_dir, client, timed, research)
        # 2. PIT agent -> PITContract, signed at gate G1
        draft, tr = propose_contract(client, timed, card)
        meta["gate_interactions"] = 1
        contract = sign_contract(run_dir, client, draft, tr, gate_g1(draft, oracle))
        # 3. Code Auditor -> AuditReport
        audit(run_dir, client, timed, contract, pipeline, meta)
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        _finish(run_dir, client, meta)
    if research:
        brief.build(run_dir)
    return meta


# ---- Research EDA, phase A: research panel -> Research EDA Agent -> [G2] feature freeze -------------------------
def oracle_contract() -> PITContract:
    """A PIT contract signed by the oracle with the gold rule: for benchmarks of the steps after G1, whose PIT step
    is evaluated on its own."""
    p = active()
    return PITContract(dataset_id=p.dataset_id, fields=[], availability_rule=p.gold_rule(),
                       rule_explanation="gold rule (benchmark: the PIT step is evaluated separately)", evidence=[],
                       revision_policy="", universe_policy="", confidence=1.0,
                       signed_off=GateDecision(decided_by="oracle", approved=True, note="gold rule"),
                       provenance=Provenance(producer_agent="oracle"))


def build_research_panel(run_dir: Path, contract: PITContract, derived: dict | None = None):
    """The research panel from the signed PIT view (deterministic); it becomes the session's panel."""
    from quantaccelerator.tools.panel import build_panel, describe, set_panel
    from quantaccelerator.tools.pit_view import build_pit_view
    panel, meta = build_panel(build_pit_view(contract, derived))
    set_panel(panel, meta)
    (run_dir / "state").mkdir(parents=True, exist_ok=True)
    (run_dir / "state" / "research_panel.json").write_text(json.dumps({**meta, "description": describe(panel, meta)},
                                                                      indent=1, default=str))
    return panel, meta


def _eda_context(card: DatasetCard | None) -> dict:
    ctx = {}
    if card is not None:
        ctx["field_meanings"] = {f"{f.table}.{f.name}": f.meaning for f in card.fields
                                 if f.name in (active().view or {}).get("values", [])}
    if notes := (active().panel or {}).get("field_notes"):
        ctx["field_notes"] = notes
    return ctx


def propose_ideas(run_dir: Path, client: LLMClient, timed, card: DatasetCard | None = None,
                  researcher_ideas: list[str] | None = None):
    """Research EDA passes 1-2: survey the raw fields, then propose feature ideas (the researcher's, translated, and
    the agent's own) for review at G2. Returns (IdeaPlan, context for pass 3)."""
    from quantaccelerator.agents.research_eda import IDEAS_TASK, ResearchEDAIdeas
    from quantaccelerator.tools.features import _value_fields, reset_findings
    reset_findings()
    ctx = _eda_context(card)
    survey, tr1 = timed(ResearchEDASurvey(), SURVEY_TASK, ctx or None)
    ctx["survey"] = {"plan": survey.plan,
                     "findings": [f"{o.feature} vs {o.conditioner}: {o.claim} ({', '.join(o.evidence[:2])})"
                                  for o in registry.CTX.findings.get("feature_observations", [])],
                     "value_fields": _value_fields()}
    ideas = [t for t in researcher_ideas or [] if t.strip()]
    agent = ResearchEDAIdeas()
    agent.researcher_ideas = ideas
    plan, tr2 = timed(agent, IDEAS_TASK, {**ctx, "researcher_ideas": ideas})
    _save(run_dir, "idea_plan", plan)
    ctx["_tool_runs"] = tr1["tool_runs"] + tr2["tool_runs"]
    return plan, ctx


def gate_ideas(plan, oracle: bool) -> GateDecision:
    """G2 idea review. The oracle approves every idea (the benchmark scores the proposal itself)."""
    names = [i.name for i in plan.ideas]
    if oracle:
        return GateDecision(gate="G2", decided_by="oracle", approved=True, approved_ideas=names, note="all ideas approved")
    print("\n=== G2: review the feature ideas ===")
    for i in plan.ideas:
        print(f"  [{i.source}] {i.name}: {i.idea}\n      mechanism: {i.mechanism}")
    ans = input("Build all? [y] / list the names to keep, comma-separated / [n] stop: ").strip()
    if ans.lower() in ("", "y", "yes"):
        return GateDecision(gate="G2", decided_by="human", approved=True, approved_ideas=names)
    if ans.lower() in ("n", "no"):
        return GateDecision(gate="G2", decided_by="human", approved=False, note=input("Reason: "))
    keep = [x.strip() for x in ans.split(",") if x.strip() in names]
    return GateDecision(gate="G2", decided_by="human", approved=True, approved_ideas=keep, note=input("Note: "))


def build_ideas(run_dir: Path, client: LLMClient, timed, plan, g2: GateDecision, ctx: dict) -> CandidateFeatureSet:
    """After G2: build the approved ideas' definitions (deterministic, logged), then Research EDA pass 3 characterizes
    them and proposes candidates -> unsigned CandidateFeatureSet (saved)."""
    from quantaccelerator.tools.features import assemble_feature_set
    if not g2.approved:
        raise RuntimeError("G2 rejected the ideas; nothing to build")
    keep = [i for i in plan.ideas if i.name in (g2.approved_ideas or [])]
    registry.CTX.findings["ideas"] = {i.name: i for i in keep}
    registry.CTX.agent = "orchestrator"
    built = {}
    for idea in keep:
        for sp in idea.specs:
            r = registry.TOOLS["construct_feature"].fn(**sp.model_dump())
            built[sp.name] = {"idea": idea.name, "run_id": r["run_id"], "ok": r["ok"],
                              **({k: r["result"].get(k) for k in ("definition", "coverage_share", "median", "skew")}
                                 if r["ok"] else {"error": r["result"].get("error")})}
    tool_runs = ctx.pop("_tool_runs", [])
    from quantaccelerator.tools.features import formula
    # a compact start: the long loop has to fit 32k tokens
    ctx3 = {"field_notes": ctx.get("field_notes"), "survey_plan": (ctx.get("survey") or {}).get("plan", []),
            "approved_ideas": [{"name": i.name, "source": i.source, "idea": i.idea,
                                "definitions": [f"{sp.name} = {formula(sp)}" for sp in i.specs]} for i in keep],
            "built": {n: {k: v for k, v in b.items() if k in ("idea", "coverage_share", "error")} for n, b in built.items()}}
    wrap, tr = timed(ResearchEDA(), EDA_TASK, ctx3)
    tr = {**tr, "tool_runs": tool_runs + tr["tool_runs"]}
    fs = assemble_feature_set(wrap).model_copy(update={
        "ideas": plan.ideas, "idea_gate": g2,
        "provenance": _prov("research_eda", tr, client, ["state/research_panel.json", "state/idea_plan.json"])})
    _save(run_dir, "candidate_feature_set", fs)
    return fs


MAX_INVESTIGATIONS = 3


def _definitions(text: str, specs: list) -> dict:
    """Raw-field definitions of the constructed features a question names (the data agent sees raw tables only)."""
    import re
    from quantaccelerator.tools.features import formula, lineage
    by = {s.name: s for s in specs or []}
    named = [n for n in by if re.search(rf"\b{re.escape(n)}\b", text)]
    return {n: [f"{m} = {formula(by[m])}" for m in by if m in lineage(n, by)] for n in named}


def investigate(run_dir: Path, client: LLMClient, timed, requests: list, stage: str, card: DatasetCard | None = None,
                contract: PITContract | None = None, features: list | None = None) -> list:
    """The backward loop: route each DataInvestigationRequest (at most MAX_INVESTIGATIONS per stage) to the agent that
    owns the question, and record the grounded answer, or why there is none. Saved to state/investigations.json."""
    from quantaccelerator.agents.investigate import TASK as INV_TASK, FeatureQA, PITQA
    from quantaccelerator.agents.profiler import DataQA
    from quantaccelerator.tools.panel import STATE as PANEL
    from quantaccelerator.state.schemas import InvestigationResult
    path = run_dir / "state" / "investigations.json"
    done = [InvestigationResult.model_validate(x) for x in json.loads(path.read_text())] if path.exists() else []
    out = []
    for k, req in enumerate(requests[:MAX_INVESTIGATIONS]):
        defs = _definitions(f"{req.question} {req.observation}", features)
        panel_ok = PANEL["panel"] is not None and defs
        agent = PITQA() if req.route_to == "pit" else FeatureQA() if panel_ok else DataQA()
        if panel_ok:   # the feature tools resolve features from the session's specs
            registry.CTX.findings.setdefault("features", {}).update({sp.name: sp for sp in features})
        agent.name = f"{agent.name}_{stage}_{k + 1}"
        ctx = {"field_notes": (active().panel or {}).get("field_notes"), "evidence_cited_by_research": req.evidence}
        if defs:
            ctx["feature_definitions"] = defs
            ctx["note"] = ("the features named are built from raw fields as defined; examine those fields (a ratio "
                           "through the tools' denominator argument), not a column of that name")
        if card is not None:
            ctx["dataset_card"] = {"fields": [{"field": f"{f.table}.{f.name}", "meaning": f.meaning} for f in card.fields]}
        if contract is not None:
            ctx["signed_availability_rule"] = contract.availability_rule.model_dump(mode="json")
        task = INV_TASK.format(stage=stage.replace("_", " "), question=req.question, observation=req.observation)
        agent.given = f"{req.question} {req.observation}"
        try:
            ans, _ = timed(agent, task, ctx)
            out.append(InvestigationResult(request=req, stage=stage, routed_to=agent.name, answer=ans))
        except AgentFailure as e:
            out.append(InvestigationResult(request=req, stage=stage, routed_to=agent.name, error=str(e)[:300]))
    for req in requests[MAX_INVESTIGATIONS:]:
        out.append(InvestigationResult(request=req, stage=stage, routed_to="-", error="not routed: over the limit of "
                                       f"{MAX_INVESTIGATIONS} per stage"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([x.model_dump(mode="json") for x in done + out], indent=1))
    registry.CTX.agent = "orchestrator"
    return out


def explore(run_dir: Path, client: LLMClient, timed, card: DatasetCard | None = None, oracle: bool = True,
            researcher_ideas: list[str] | None = None) -> CandidateFeatureSet:
    """Research EDA end to end up to the freeze: survey, ideas, G2, build and characterize."""
    plan, ctx = propose_ideas(run_dir, client, timed, card, researcher_ideas)
    return build_ideas(run_dir, client, timed, plan, gate_ideas(plan, oracle), ctx)


def gate_freeze(fs: CandidateFeatureSet, oracle: bool) -> GateDecision:
    """G3 feature freeze. The oracle freezes the set as proposed (the benchmark scores the proposal itself)."""
    names = [c.name for c in fs.candidates]
    if oracle:
        return GateDecision(gate="G3", decided_by="oracle", approved=True, approved_features=names,
                            note="frozen as proposed")
    print("\n=== G3: freeze the candidate feature set ===")
    for c in fs.candidates:
        print(f"  {c.name} ({c.idea}): {c.rationale}")
    ans = input("Freeze all? [y] / list the names to keep, comma-separated / [n] reject: ").strip()
    if ans.lower() in ("", "y", "yes"):
        return GateDecision(gate="G3", decided_by="human", approved=True, approved_features=names)
    if ans.lower() in ("n", "no"):
        return GateDecision(gate="G3", decided_by="human", approved=False, note=input("Reason: "))
    keep = [x.strip() for x in ans.split(",") if x.strip() in names]
    return GateDecision(gate="G3", decided_by="human", approved=True, approved_features=keep, note=input("Note: "))


def freeze_features(run_dir: Path, fs: CandidateFeatureSet, g3: GateDecision) -> CandidateFeatureSet:
    """Attach G3, hash the frozen definition (specs + panel), and write the frozen features as a table."""
    from quantaccelerator.returns.vault import feature_set_hash
    from quantaccelerator.tools.features import materialize
    keep = g3.approved_features if g3.approved else []
    fs = fs.model_copy(update={"signed_off": g3})
    h = feature_set_hash(fs) if g3.approved else ""
    fs = fs.model_copy(update={"set_hash": h})
    _save(run_dir, "candidate_feature_set", fs)
    if g3.approved and keep:
        (run_dir / "features").mkdir(exist_ok=True)
        materialize(fs.features, keep).to_parquet(run_dir / "features" / f"{h}.parquet", index=False)
    return fs


def run_explore(run_id: str, run_dir: Path | None = None, client: LLMClient | None = None, oracle: bool = True,
                pit: str = "oracle", researcher_ideas: list[str] | None = None) -> CandidateFeatureSet:
    """Research EDA phase A on its own: signed contract (gold by default, or the PIT agent + G1), research panel,
    survey, ideas (G2 idea review), build and characterize, G3 feature freeze, Feature Report."""
    from quantaccelerator.agents import feature_report
    if researcher_ideas is None:
        researcher_ideas = list((active().panel or {}).get("researcher_ideas", []))
    run_dir, client, meta, timed = _start(run_id, run_dir, client, mode="explore", oracle=oracle, pit=pit,
                                          researcher_ideas=researcher_ideas)
    try:
        card = None
        if pit == "agent":
            card = understand(run_dir, client, timed, research=False)
            draft, tr = propose_contract(client, timed, card)
            contract = sign_contract(run_dir, client, draft, tr, gate_g1(draft, oracle))
        else:
            contract = oracle_contract()
            _save(run_dir, "pit_contract", contract)
        registry.CTX.agent = "orchestrator"
        _, pmeta = build_research_panel(run_dir, contract)
        meta["panel"] = {k: pmeta[k] for k in ("hash", "n_rows", "n_dates", "n_entities")}
        fs = explore(run_dir, client, timed, card, oracle, researcher_ideas)
        if fs.wrapup.data_investigation_requests:
            meta["investigations"] = len(investigate(run_dir, client, timed, fs.wrapup.data_investigation_requests,
                                                     "research_eda", card, contract, fs.features))
        fs = freeze_features(run_dir, fs, gate_freeze(fs, oracle))
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        _finish(run_dir, client, meta)
    feature_report.build(run_dir)
    return fs


# ---- Research EDA, phase B: plan -> [G4] -> discovery -> validation -> [G5] -> locked test -------------------------
def default_split():
    from quantaccelerator.state.schemas import SplitSpec
    return SplitSpec(**active().predict["split"])


def _conditioners(panel):
    return panel[["date", "entity", "sector", "adv_proxy", "size_float"]]


def oracle_feature_set(run_dir: Path) -> CandidateFeatureSet:
    """A fixed feature set frozen by the oracle (the profile's reference constructions): for benchmarks of phase B on
    its own, as oracle_contract() is for the steps after G1."""
    from quantaccelerator.state.schemas import EDAWrapUp
    from quantaccelerator.tools.panel import STATE, describe, get_panel
    specs, cands, ideas = active().predict["oracle_features"]()
    meta = STATE["meta"]
    fs = CandidateFeatureSet(panel_hash=meta["hash"], panel={**describe(get_panel(), meta), "rule": meta.get("rule")},
                             features=specs, ideas=ideas, candidates=cands, observations=[],
                             wrapup=EDAWrapUp(summary="oracle feature set: the benchmark's reference constructions"),
                             provenance=Provenance(producer_agent="oracle"))
    g3 = GateDecision(gate="G3", decided_by="oracle", approved=True, approved_features=[c.name for c in cands],
                      note="oracle feature set")
    return freeze_features(run_dir, fs, g3)


def load_feature_set(run_dir: Path, source: Path) -> CandidateFeatureSet:
    """The frozen set of an earlier Research EDA run, copied into this run (its panel must be the current one)."""
    import shutil
    from quantaccelerator.returns.vault import feature_set_hash
    from quantaccelerator.tools.panel import STATE
    fs = CandidateFeatureSet.model_validate_json((Path(source) / "state" / "candidate_feature_set.json").read_text())
    if not fs.set_hash or feature_set_hash(fs) != fs.set_hash:
        raise RuntimeError(f"{source}: the feature set is not frozen, or changed after its freeze")
    if fs.panel_hash != STATE["meta"]["hash"]:
        raise RuntimeError(f"{source}: frozen on panel {fs.panel_hash}, but this run's panel is {STATE['meta']['hash']}"
                           " (same dataset and signed contract needed)")
    (run_dir / "features").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(source) / "features" / f"{fs.set_hash}.parquet", run_dir / "features")
    _save(run_dir, "candidate_feature_set", fs)
    return fs


def _plan_context(fs: CandidateFeatureSet) -> dict:
    from quantaccelerator.returns.core import CONTROL_MEANING
    from quantaccelerator.tools.features import formula, lineage
    ideas = {i.name: i for i in fs.ideas}
    specs = {s.name: s for s in fs.features}
    frozen = set(fs.signed_off.approved_features or [])
    out = []
    for c in fs.candidates:
        if c.name not in frozen:
            continue
        i = ideas.get(c.idea)
        chain = [n for n in specs if n in lineage(c.name, specs)]
        out.append({"name": c.name, "idea": c.idea, "idea_text": i.idea if i else None,
                    "mechanism": i.mechanism if i else None, "expected_direction": i.expected_direction if i else None,
                    "definition": [f"{n} = {formula(specs[n])}" for n in chain], "rationale": c.rationale,
                    "known_exposures": c.known_exposures})
    return {"frozen_features": out, "horizons": [1, 5, 20, 60], "controls": CONTROL_MEANING,
            "conditioners": ["dollar_adv", "size", "beta", "sector", "year"],
            "panel": {k: fs.panel.get(k) for k in ("universe", "date", "first_date", "last_date")}}


def plan_predictions(run_dir: Path, client: LLMClient, timed, fs: CandidateFeatureSet):
    """Phase B pre-registration (no returns): the Predictive Planner writes the plan; the split comes from the profile."""
    from quantaccelerator.agents.predictive_eda import PLAN_TASK, PredictivePlanner
    from quantaccelerator.state.schemas import PredictionPlan
    agent = PredictivePlanner()
    agent.frozen = list(fs.signed_off.approved_features or [])
    draft, tr = timed(agent, PLAN_TASK, _plan_context(fs))
    plan = PredictionPlan(**draft.model_dump(), set_hash=fs.set_hash, split=default_split())
    _save(run_dir, "prediction_plan", plan)
    return plan


def gate_plan(plan, oracle: bool) -> GateDecision:
    """G4: sign the pre-registration and lock the split. The oracle signs as proposed."""
    if oracle:
        return GateDecision(gate="G4", decided_by="oracle", approved=True, approved_plan=plan, note="signed as proposed")
    print("\n=== G4: sign the prediction plan (the returns vault opens after this) ===")
    sp = plan.split
    print(f"  split: discovery {sp.discovery}, validation {sp.validation}, locked test {sp.locked_test}")
    for t in plan.tests:
        print(f"  {t.feature}: sign {t.expected_sign:+d}, horizons {t.horizons}, controls {t.controls}\n"
              f"      {t.horizon_rationale}")
    ans = input("Sign? [y] / [n] reject: ").strip().lower()
    if ans in ("", "y", "yes"):
        return GateDecision(gate="G4", decided_by="human", approved=True, approved_plan=plan, note=input("Note: "))
    return GateDecision(gate="G4", decided_by="human", approved=False, note=input("Reason: "))


def predict(run_dir: Path, client: LLMClient, timed, fs: CandidateFeatureSet, g4: GateDecision, panel,
            study: str | None = None):
    """Open the vault (G3 + G4 checked) and let the Predictive EDA Agent test the plan on discovery."""
    from quantaccelerator.agents.predictive_eda import TASK as PRED_TASK, PredictiveEDA
    from quantaccelerator.returns import vault
    from quantaccelerator.tools import predictive as PR
    if not g4.approved:
        raise RuntimeError("G4 rejected the plan; the returns vault stays closed")
    vault.arm(run_dir, fs, g4, active().predict["returns"], _conditioners(panel), study=study,
              link=active().predict.get("link"))
    from quantaccelerator.state.schemas import PredictiveWrapUp
    PR.reset()
    plan = g4.approved_plan
    # one pass per pre-registered feature, each from a fresh context with its own completeness check: a full plan
    # (features x horizons x controls, plus profiles and shapes) does not fit one 32k-token loop
    wraps, trace = [], {"tool_runs": [], "prompt_hash": None}
    for t in plan.tests:
        registry.CTX.findings["predict_focus"] = t.feature
        agent = PredictiveEDA()
        agent.name = f"predictive_eda__{t.feature}"
        ctx = {"plan": [t.model_dump(mode="json")], "this_pass": t.feature, "discovery": list(plan.split.discovery),
               "other_frozen_features": [x.feature for x in plan.tests if x.feature != t.feature]}
        w, tr = timed(agent, PRED_TASK, ctx)
        wraps.append((t.feature, w))
        trace["tool_runs"] += tr["tool_runs"]
        trace["prompt_hash"] = trace["prompt_hash"] or tr["prompt_hash"]
    registry.CTX.findings.pop("predict_focus", None)
    wrap = PredictiveWrapUp(summary=" ".join(f"[{f}] {w.summary}" for f, w in wraps),
                            warnings=[f"[{f}] {x}" for f, w in wraps for x in w.warnings],
                            open_questions=[f"[{f}] {x}" for f, w in wraps for x in w.open_questions],
                            data_investigation_requests=[r for _, w in wraps for r in w.data_investigation_requests])
    return wrap, list(registry.CTX.findings.get("research_observations", [])), trace


def _retest(o) -> dict:
    from quantaccelerator.tools.predictive import VALIDATION_T
    r = registry.TOOLS["ic_summary"].fn(feature=o.feature, horizon=o.horizon, controls=list(o.controls))
    res = r["result"] if r["ok"] else {}
    t, ic = res.get("t_nw"), res.get("mean_ic")
    ok = t is not None and ic is not None and (ic > 0) == (o.sign > 0) and abs(t) >= VALIDATION_T
    return {"run_id": r["run_id"], "mean_ic": ic, "t_nw": t, "n_dates": res.get("n_dates"), "passed": bool(ok),
            "rule": f"same sign and |t_nw| >= {VALIDATION_T}"}


def validate_observations(obs: list) -> list:
    """Re-run every 'informative' observation exactly as recorded, on the validation segment (deterministic)."""
    from quantaccelerator.returns import vault
    registry.CTX.agent = "orchestrator"
    out = []
    with vault.segment("validation"):
        for o in obs:
            if o.verdict == "informative":
                v = _retest(o)
                o = o.model_copy(update={"validation": v, "status": "validated" if v["passed"] else "failed_validation"})
            else:
                o = o.model_copy(update={"status": "not_validated"})
            out.append(o)
    return out


CRITIC_BATCH = 6


def critique(run_dir: Path, client: LLMClient, timed, fs: CandidateFeatureSet, obs: list, panel):
    """The Critic attacks every validated finding with pre-registered, executable tests (from the findings' outputs
    only); the orchestrator runs each on discovery and judges it by the fixed rule (tools.critic.judge)."""
    from quantaccelerator.agents.critic import TASK as CRITIC_TASK, Critic
    from quantaccelerator.returns import vault
    from quantaccelerator.state.schemas import Critique, CritiqueResult
    from quantaccelerator.tools.critic import judge, legal_attacks, split_draft, standard_battery
    from quantaccelerator.tools.features import formula, lineage
    targets = {i: o for i, o in enumerate(obs) if o.status == "validated"}
    if not targets:
        return obs, []
    specs = {s.name: s for s in fs.features}
    sp = vault.STATE["split"]
    findings = [{"index": i, "feature": o.feature, "definition": [f"{n} = {formula(specs[n])}" for n in specs
                                                                  if n in lineage(o.feature, specs)],
                 "horizon_sessions": o.horizon, "controls": o.controls,
                 "controls_not_yet_applied": [c for c in ("size", "reversal", "momentum", "beta", "dollar_adv", "sector")
                                              if c not in o.controls],
                 "legal_attacks": legal_attacks(o.controls),
                 "claim": o.claim,
                 "discovery": {k: o.stats.get(k) for k in ("mean_ic", "t_nw", "hit_rate", "mean_ic_by_year")},
                 "validation": {k: (o.validation or {}).get(k) for k in ("mean_ic", "t_nw")}}
                for i, o in targets.items()]
    sectors = sorted(x for x in panel.sector.dropna().unique() if x != "Unknown")
    registry.CTX.agent = "orchestrator"
    for f in findings:   # an output the Critic may see: where across sectors the finding's IC comes from
        o = targets[f["index"]]
        r = registry.TOOLS["conditional_ic"].fn(feature=o.feature, horizon=o.horizon, by="sector")
        if r["ok"]:
            f["discovery_ic_by_sector"] = {k: {"mean_ic": round(v["mean_ic"], 4), "t_nw": round(v["t_nw"], 2)}
                                           for k, v in r["result"]["buckets"].items() if v.get("t_nw") is not None}
    battery = standard_battery(tuple(sp.discovery))
    ctx = {"findings": findings, "discovery": list(sp.discovery), "sectors": sectors,
           "standard_battery_run_on_every_finding": [f"{c.threat_type}: {c.test.kind} "
                                                     f"{c.test.universe or list(c.test.period or [])}" for c in battery],
           "panel": {k: fs.panel.get(k) for k in ("universe", "entities", "first_date", "last_date")},
           "forward_return": "from the open of the decision date to the close h-1 sessions later",
           "controls_available": ["size", "reversal", "momentum", "beta", "dollar_adv", "sector"]}
    # batches of findings, each from a fresh context: one draft covering dozens of findings does not fit
    from quantaccelerator.state.schemas import CriticDraft
    critiques, idx = [], sorted(targets)
    for b0 in range(0, len(idx), CRITIC_BATCH):
        batch = idx[b0:b0 + CRITIC_BATCH]
        agent = Critic()
        agent.name = f"critic_{b0 // CRITIC_BATCH + 1}"
        agent.targets = {i: {"controls": targets[i].controls} for i in batch}
        agent.sectors, agent.discovery = sectors, tuple(sp.discovery)
        part, _ = timed(agent, CRITIC_TASK, {**ctx, "findings": [f for f in findings if f["index"] in batch]})
        critiques += part.critiques
    draft = CriticDraft(critiques=critiques)
    _save(run_dir, "critic_draft", draft)
    runnable, dropped = split_draft(draft, {i: {"controls": o.controls} for i, o in targets.items()}, sectors,
                                    tuple(sp.discovery))
    (run_dir / "state" / "critic_dropped.json").write_text(json.dumps(
        [{"critique": c.model_dump(mode="json"), "reason": r} for c, r in dropped], indent=1))
    registry.CTX.agent = "orchestrator"
    results = []
    attacks = [(c.model_copy(update={"target": i}), "standard") for i in targets for c in battery] + \
        [(c, "critic") for c in runnable]

    def run_attack(c, source):
        o = targets[c.target]
        r = registry.TOOLS["critic_test"].fn(feature=o.feature, horizon=o.horizon, controls=list(o.controls),
                                            test=c.test.model_dump(mode="json"))
        att = r["result"] if r["ok"] else {"error": r["result"].get("error")}
        outcome, ret = judge(o.stats.get("mean_ic"), att) if r["ok"] else ("inconclusive", None)
        return CritiqueResult(**c.model_dump(), source=source, run_id=r["run_id"], retention=ret, outcome=outcome,
                              claim={k: o.stats.get(k) for k in ("mean_ic", "t_nw")},
                              attacked={k: att.get(k) for k in ("mean_ic", "t_nw", "n_dates", "entities", "error")
                                        if k in att})

    results = [run_attack(c, source) for c, source in attacks]
    # a test that refuted one finding is re-run on its siblings (same feature) that have not faced it; each is judged
    # on its own numbers: the threat is about where the feature's effect lives, not about one control set
    sig = lambda x: (x.test.kind, x.test.universe, x.test.sector, tuple(sorted(x.test.controls)), tuple(x.test.period or ()))
    for c in [x for x in results if x.outcome == "refuted" and x.source == "critic"]:
        for i, o in targets.items():
            if i != c.target and o.feature == targets[c.target].feature and not any(
                    x.target == i and sig(x) == sig(c) for x in results):
                if c.test.kind == "add_controls" and not set(c.test.controls) - set(o.controls):
                    continue
                base = Critique(**c.model_dump(exclude={"source", "run_id", "claim", "attacked", "retention",
                                                         "outcome"}))
                results.append(run_attack(base.model_copy(update={"target": i}), "propagated"))
    out = []
    for i, o in enumerate(obs):
        mine = [x for x in results if x.target == i]
        if mine:
            o = o.model_copy(update={"critique": {
                "attacks": len(mine), **{k: sum(x.outcome == k for x in mine) for k in ("survived", "refuted",
                                                                                         "inconclusive")},
                "refuted_by": [f"{x.threat_type} ({x.source}): {x.threat}" for x in mine if x.outcome == "refuted"]}})
        out.append(o)
    return out, results


def cost_check(obs: list) -> list:
    """Cost and capacity of every validated finding, on discovery and on validation (deterministic; the locked test is
    not touched)."""
    from quantaccelerator.returns import vault
    registry.CTX.agent = "orchestrator"
    run = lambda o: (lambda r: r["result"] if r["ok"] else {"error": r["result"].get("error")})(
        registry.TOOLS["cost_capacity"].fn(feature=o.feature, horizon=o.horizon, sign=o.sign, controls=list(o.controls)))
    keep = [i for i, o in enumerate(obs) if o.status == "validated"]
    res = {i: {"discovery": run(obs[i])} for i in keep}
    if keep:
        with vault.segment("validation"):
            for i in keep:
                res[i]["validation"] = run(obs[i])
    return [o.model_copy(update={"costs": res[i]}) if i in res else o for i, o in enumerate(obs)]


def gate_release(obs: list, oracle: bool) -> GateDecision:
    """G5: which validated observations go to the locked test (touched once). The oracle releases every validated one."""
    ok = [i for i, o in enumerate(obs) if o.status == "validated"]
    if oracle:
        keep = [i for i in ok if not (obs[i].critique or {}).get("refuted")]
        return GateDecision(gate="G5", decided_by="oracle", approved=True, released_observations=keep,
                            note="every validated observation the Critic did not refute released")
    print("\n=== G5: release observations to the locked test (it can be opened once) ===")
    for i, o in enumerate(obs):
        v = o.validation or {}
        c = o.critique or {}
        print(f"  [{i}] {o.feature} h={o.horizon} controls={o.controls}: {o.verdict}, {o.status}"
              + (f" (validation IC {v.get('mean_ic')}, t {v.get('t_nw')})" if v else "")
              + (f"; Critic: {c['survived']} survived, {c['refuted']} refuted, {c['inconclusive']} inconclusive"
                 if c else ""))
        for r in c.get("refuted_by", []):
            print(f"        refuted by {r}")
        k = (o.costs or {}).get("validation") or {}
        if k.get("gross"):
            print(f"        validation: gross {k['gross']['mean_bp']:.1f} bp/period, net {k['net_round_trip']['mean_bp']:.1f}"
                  f" (round trip) / {k['net_changes_only']['mean_bp']:.1f} (changes only), capacity per leg "
                  f"{k.get('capacity_per_leg_usd') or 0:,.0f} USD")
    ans = input(f"Release the validated ones {ok}? [y] / list indexes / [n] none: ").strip().lower()
    if ans in ("", "y", "yes"):
        return GateDecision(gate="G5", decided_by="human", approved=True, released_observations=ok, note=input("Note: "))
    if ans in ("n", "no"):
        return GateDecision(gate="G5", decided_by="human", approved=False, note=input("Reason: "))
    keep = [int(x) for x in ans.replace(",", " ").split() if x.isdigit() and int(x) < len(obs)]
    return GateDecision(gate="G5", decided_by="human", approved=True, released_observations=keep, note=input("Note: "))


def locked_test(obs: list, g5: GateDecision) -> list:
    """The released observations on the locked test: one vault access for all of them, reported whatever they show."""
    from quantaccelerator.returns import vault
    keep = set(g5.released_observations or []) if g5.approved else set()
    if not keep:
        return obs
    registry.CTX.agent = "orchestrator"
    vault.release(g5.note)
    out = []
    with vault.segment("locked_test"):
        for i, o in enumerate(obs):
            if i in keep:
                v = _retest(o)
                o = o.model_copy(update={"locked_test": v, "status": "locked_pass" if v["passed"] else "locked_fail"})
            out.append(o)
    return out


def _ledger_summary(study: str) -> dict:
    from quantaccelerator.returns import core, vault
    acc = vault.accesses(study)
    n = vault.n_trials(study)
    return {"study": study, "n_trials": n, "hurdle_t": round(core.hurdle(n), 3),
            "vault_accesses": acc[["segment", "caller", "granted", "reason", "created_utc"]].to_dict("records")}


def run_predict(run_id: str, run_dir: Path | None = None, client: LLMClient | None = None, oracle: bool = True,
                features: str = "oracle"):
    """Research EDA phase B on its own: signed contract, research panel, a frozen feature set (the profile's oracle
    set, or an earlier explore run's), plan + G4, discovery, validation, G5, locked test -> PredictiveReport."""
    from quantaccelerator.state.schemas import PredictiveReport
    run_dir, client, meta, timed = _start(run_id, run_dir, client, mode="predict", oracle=oracle, features=features)
    try:
        contract = oracle_contract()
        if features != "oracle" and (Path(features) / "state" / "pit_contract.json").exists():
            contract = PITContract.model_validate_json((Path(features) / "state" / "pit_contract.json").read_text())
        _save(run_dir, "pit_contract", contract)
        registry.CTX.agent = "orchestrator"
        panel, pmeta = build_research_panel(run_dir, contract)
        meta["panel"] = {k: pmeta[k] for k in ("hash", "n_rows", "n_dates", "n_entities")}
        fs = oracle_feature_set(run_dir) if features == "oracle" else load_feature_set(run_dir, Path(features))
        study = f"{fs.set_hash}:{run_id}" if features == "oracle" else fs.set_hash
        plan = plan_predictions(run_dir, client, timed, fs)
        g4 = gate_plan(plan, oracle)
        wrap, obs, tr = predict(run_dir, client, timed, fs, g4, panel, study)
        obs = validate_observations(obs)
        obs, crits = critique(run_dir, client, timed, fs, obs, panel)
        obs = cost_check(obs)
        if wrap.data_investigation_requests:
            meta["investigations"] = len(investigate(run_dir, client, timed, wrap.data_investigation_requests,
                                                     "predictive_eda", None, contract, fs.features))
        g5 = gate_release(obs, oracle)
        obs = locked_test(obs, g5)
        report = PredictiveReport(set_hash=fs.set_hash, plan=g4.approved_plan, plan_gate=g4, observations=obs,
                                  wrapup=wrap, critiques=crits, release_gate=g5, ledger=_ledger_summary(study),
                                  provenance=_prov("predictive_eda", tr, client, ["state/candidate_feature_set.json",
                                                                                  "state/prediction_plan.json"]))
        _save(run_dir, "predictive_report", report)
        from quantaccelerator.agents import predictive_report
        predictive_report.build(run_dir)
        meta["gates"] = {"G4": g4.model_dump(mode="json", exclude={"approved_plan"}),
                         "G5": g5.model_dump(mode="json")}
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        _finish(run_dir, client, meta)
    return report
