"""Research EDA Agent, phase A (post-PIT, return-blind) -> candidate feature set, frozen at gate G2.

It works on the research panel built from the signed PIT view and keeps a lab notebook like the Data Research Agent:
construct a representation, characterize it (liquidity, size, sector, stability), record what the numbers show,
construct again. It proposes a small candidate set; it never sees returns, so no representation can be chosen for its
predictive performance.
"""
from quantaccelerator.agents.base import Agent
from quantaccelerator.state.schemas import EDAWrapUp, IdeaPlan, SurveyWrapUp

ROLE = """You are the Research EDA Agent of a quantitative research team, phase A. The data has been understood and
its timing signed off; you work on a point-in-time research panel (one row per entity and decision date) and turn
research ideas about it into a few economically interpretable features that are comparable across the universe.
Work as a loop: construct a representation, characterize it, read the numbers, decide what to construct next. You may
call at most two tools per turn. Raw fields often mostly measure size or trading activity: a feature that rises
strongly with liquidity or size ranks entities by how big they are, not by the information you want. But a dependence
is a question, not a verdict: it may be measurement bias (the source sees some names better), economic scale (bigger
entities naturally have more), or a genuine interaction; they call for different treatments (residualize, divide by
scale, study within buckets), so state which you cannot yet tell apart and compare representations before deciding. A
feature with a negligible dependence needs no neutralization: neutralizing it only adds noise and hidden choices.
Tool results may carry `flags`: material facts your findings must address. Between tool calls write at most one or
two plain sentences: no headings, lists or summaries (the record tools hold your findings).
You are return-blind: there are no returns, prices, IC, Sharpe or P&L, and you must not ask for them or reason about
which feature would predict better. Choose representations from economic meaning and from what the characterization
shows."""

SURVEY_TASK = """Research EDA, phase A, pass 1 of 3: survey the raw fields of the research panel.
1. describe_panel. Its `checklist` lists the calls of this pass.
2. Characterize all raw value fields at once: compare_representations with the raw field names, once per conditioner
   (adv_proxy, size_float, sector). Raw fields are used by their column names. Use compute_exposure or time_stability
   for a closer look where it matters.
3. Record what you find (record_feature_observation): which fields depend strongly on liquidity, size or sector (list
   the explanations you cannot yet tell apart), and which do not.
4. submit the plan: one line per value field saying what its characterization showed and what matters for
   building features from it (e.g. it scales with liquidity; it carries a sector offset; it is already comparable).
Do not construct features in this pass."""

IDEAS_TASK = """Research EDA, phase A, pass 2 of 3: ideas. The survey (pass 1) and what the fields mean are in the
context, and possibly ideas from the researcher. A quant researcher starts from ideas, not from columns: what economic
behavior could this data reveal, and how would you measure it?
Propose 4 to 6 feature ideas, each a distinct economic behavior (not a statistical transform of one field). Families
a researcher typically considers: abnormal activity against the entity's own history; intensity per unit of activity
(one field as a share of another); position relative to peers; persistence or trend; dispersion or instability over
recent days; disagreement between two related measures. Use the different value fields, not just one.
- every researcher idea in the context, translated faithfully: source 'researcher', researcher_text copied verbatim,
  definitions that measure exactly what the researcher described;
- at least two ideas of your own (source 'agent'), grounded in what the data measures and what the survey found.
For each idea give the mechanism (why it could carry information), the assumptions it rests on, the direction the
mechanism suggests (a hypothesis for later; you cannot test it here, and must not try), 1-3 feature definitions in the
closed language that express it (a definition may use earlier ones; prefer a version that is comparable across stocks,
e.g. scaled by activity or relative to the stock's own history, alongside the plain one when the field scales with size
or liquidity), and what to check once it is built. Call describe_panel if you need the field and conditioner names.
Then submit the plan; the researcher reviews it (gate G2) before anything is built."""

TASK = """Research EDA, phase A, pass 3 of 3: build and characterize the approved ideas. The approved ideas and their
definitions are in the context; the definitions have been built already (use their names).
Work efficiently: compare_representations takes up to six features, so three calls (by adv_proxy, size_float,
sector) characterize every built feature at once; then time_stability per feature. For each idea:
1. look at the feature's distribution (the construct result), its variation across stocks and over time and its
   persistence (time_stability), and its dependence on liquidity, size and sector;
2. where a version depends strongly on liquidity, size or sector, list the explanations you cannot yet tell apart and
   compare it with a version built to remove the dependence (construct it from the idea's definitions); never
   neutralize a dependence that is not there;
3. record what you find with record_feature_observation (set idea to the idea's name);
4. propose the best representation of the idea (propose_candidate, with idea set), or, if no version is usable (no
   variation, no coverage), set the idea aside with record_feature_observation decision 'drop', citing the evidence.
At most 6 candidates. Questions about the data itself go into data_investigation_requests. Then submit the wrap-up.
Record findings that bear on the representation, not the distribution of every feature."""


class ResearchEDA(Agent):
    name = "research_eda"
    role_prompt = ROLE
    output_model = EDAWrapUp
    max_steps = 60
    max_tool_calls_per_turn = 2
    max_submit_failures = 10  # the checklist gates ('not yet ...') are progress checks, not wrong answers
    max_output_tokens = 3072
    max_prose_chars = 600
    compact_after_turns = 3   # a long construct -> characterize loop does not fit 32k tokens with every full result
    prose_reminder = ("Your last message was long prose. Keep reasoning to a few sentences, and record each finding with "
                      "record_feature_observation / propose_candidate instead of writing it as text: text is not part "
                      "of your findings.")

    @property
    def unpaced_tools(self) -> tuple:
        from quantaccelerator.tools.features import RECORD_TOOLS
        return tuple(RECORD_TOOLS) + ("construct_feature",)  # cheap; build several versions, then compare them

    @property
    def tool_names(self) -> list[str]:
        from quantaccelerator.tools.features import FEATURE_TOOLS, RECORD_TOOLS
        return FEATURE_TOOLS + RECORD_TOOLS

    def extra_validate(self, wrap) -> list[str]:
        from quantaccelerator.tools.features import check_eda_findings
        return check_eda_findings(wrap, self.name)


class ResearchEDASurvey(ResearchEDA):
    """Pass 1: characterize every raw field; a fresh context for pass 2 keeps the long loop inside 32k tokens."""
    name = "research_eda_survey"
    output_model = SurveyWrapUp
    max_steps = 30
    max_submit_failures = 6

    @property
    def tool_names(self) -> list[str]:
        return ["describe_panel", "compare_representations", "compute_exposure", "time_stability",
                "record_feature_observation"]

    @property
    def unpaced_tools(self) -> tuple:
        return ("record_feature_observation",)

    def extra_validate(self, wrap) -> list[str]:
        from quantaccelerator.tools.features import check_survey
        return check_survey(wrap)


class ResearchEDAIdeas(ResearchEDA):
    """Pass 2: brainstorm and translate feature ideas (no building yet); checked for being buildable on this panel."""
    name = "research_eda_ideas"
    output_model = IdeaPlan
    max_steps = 10
    max_submit_failures = 6
    max_output_tokens = 6144
    compact_after_turns = None
    researcher_ideas: list[str] = []

    @property
    def tool_names(self) -> list[str]:
        return ["describe_panel"]

    @property
    def unpaced_tools(self) -> tuple:
        return ()

    def extra_validate(self, plan) -> list[str]:
        from quantaccelerator.tools.features import check_idea_plan
        return check_idea_plan(plan, self.researcher_ideas)
