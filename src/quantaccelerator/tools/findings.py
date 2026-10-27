"""The Data Research Agent's lab notebook: record each finding as soon as it is established, checked on the spot.

Each record_* tool validates one item against this session's tool runs (the same checks the whole findings would get)
and either stores it in registry.CTX.findings or returns the error, so a mistake is fixed while the evidence is fresh
instead of in one large final submission. The orchestrator assembles the stored items into DataResearchDraft.
"""
from quantaccelerator.state.schemas import CapabilityRating, CleaningStep, DataObservation, PreprocessingConsideration
from quantaccelerator.tools.registry import CTX, inline_refs, tool

CAPABILITY_DIMENSIONS = ["history_depth", "cross_sectional_breadth", "update_frequency", "coverage_stability",
                         "event_breadth"]
RECORD_TOOLS = ["record_observation", "rate_capability", "note_preprocessing", "propose_cleaning_step"]


def inline_refs_model(model) -> dict:
    return inline_refs(model.model_json_schema())


def _store(kind: str) -> list:
    return CTX.findings.setdefault(kind, [])


@tool("record_observation", "Record one observation about the data in your findings, as soon as a tool result "
      "establishes it: one factual claim, the run_ids behind it, the figure that shows it, how serious it is "
      "(info / warning / issue) and the next question it raises. It is checked immediately.",
      inline_refs_model(DataObservation))
def record_observation(**kw):
    from quantaccelerator.agents.checks import check_observation, figure_of
    o = DataObservation.model_validate(kw)
    errs = check_observation(o)
    if errs:
        raise ValueError(" ".join(errs))
    note = None
    if not (o.figure_run_id and figure_of([o.figure_run_id])):  # attach the figure of a cited run
        fig = figure_of(o.evidence)
        note = f"figure_run_id set to {fig}, a cited run that drew a figure" if fig else "no figure attached"
        o = o.model_copy(update={"figure_run_id": fig})
    obs = _store("observations")
    obs.append(o)
    return {"recorded": f"observation {len(obs)}", "figure_run_id": o.figure_run_id, "note": note}


@tool("rate_capability", "Rate one dimension of what the data can support (history_depth, cross_sectional_breadth, "
      "update_frequency, coverage_stability, event_breadth) as weak / moderate / strong, citing the run that measured "
      "it and the deciding numbers. Rating a dimension again replaces the earlier rating.",
      {"type": "object", "properties": {"dimension": {"type": "string", "enum": CAPABILITY_DIMENSIONS},
                                        **inline_refs_model(CapabilityRating)["properties"]},
       "required": ["dimension", "rating", "evidence", "note"]})
def rate_capability(dimension: str, **kw):
    from quantaccelerator.agents.checks import check_capability
    dim = CapabilityRating.model_validate(kw)
    errs = check_capability(dimension, dim)
    if errs:
        raise ValueError(" ".join(errs))
    CTX.findings.setdefault("capability", {})[dimension] = dim
    missing = [d for d in CAPABILITY_DIMENSIONS[:4] if d not in CTX.findings["capability"]]
    return {"recorded": dimension, "still_to_rate": missing}


@tool("note_preprocessing", "Record how one field (or a ratio 'table.A/table.B') should be represented: what the data "
      "shows about it (with run_ids), candidate representations, and the risks of using it raw.",
      inline_refs_model(PreprocessingConsideration))
def note_preprocessing(**kw):
    from quantaccelerator.agents.checks import check_preprocessing
    p = PreprocessingConsideration.model_validate(kw)
    errs = check_preprocessing(p)
    if errs:
        raise ValueError(" ".join(errs))
    notes = _store("preprocessing")
    notes[:] = [x for x in notes if x.field != p.field] + [p]
    return {"recorded": p.field}


@tool("propose_cleaning_step", "Propose one cleaning step (not applied until the human approves). Its effect is "
      "measured now: the rows it would drop or flag. Steps that change nothing are refused. Prefer flag_rows over "
      "dropping. Operations: drop_duplicates (key in columns), drop_rows / flag_rows (rows matching all `where` "
      "predicates; '>col' compares with another column), missing_indicator.",
      inline_refs_model(CleaningStep))
def propose_cleaning_step(**kw):
    from quantaccelerator.tools.clean import step_rows, validate_step
    s = CleaningStep.model_validate({"evidence": [], **kw})
    errs = validate_step(s)
    if errs:
        raise ValueError("; ".join(errs))
    n = step_rows(s)
    if n == 0:
        raise ValueError(f"{s.op} on {s.table} changes no rows; not recorded")
    s = s.model_copy(update={"evidence": list(dict.fromkeys(s.evidence + [CTX.current_run_id]))})
    steps = _store("cleaning")
    steps.append(s)
    return {"recorded": f"cleaning step {len(steps)}", "rows_affected": n,
            "effect": "removed" if s.op in ("drop_duplicates", "drop_rows") else "flagged (kept)"}


def assemble(wrap) -> "DataResearchDraft":
    """The findings recorded in this session plus the agent's wrap-up, as one DataResearchDraft."""
    from quantaccelerator.state.schemas import CleaningRecipe, DataResearchDraft, ResearchCapabilityAssessment
    cap = CTX.findings.get("capability", {})
    return DataResearchDraft.model_construct(
        capability=ResearchCapabilityAssessment.model_construct(
            **{d: cap.get(d) for d in CAPABILITY_DIMENSIONS}, suitable_for=wrap.suitable_for,
            weak_for=wrap.weak_for, major_limits=wrap.major_limits),
        preprocessing=list(CTX.findings.get("preprocessing", [])),
        observations=list(CTX.findings.get("observations", [])),
        cleaning=CleaningRecipe(steps=list(CTX.findings.get("cleaning", [])), summary=wrap.cleaning_summary),
        warnings=wrap.warnings, open_questions=wrap.open_questions, key_corrections=wrap.key_corrections)
