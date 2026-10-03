"""Data Profiler agent -> DatasetCard."""
from quantaccelerator.agents.base import Agent
from quantaccelerator.agents.checks import check_answer, check_dataset_card, missing_analyses, unaddressed_flags
from quantaccelerator.datasets import active
from quantaccelerator.state.schemas import DataAnswer, DatasetCardDraft, DatasetOverview, ResearchWrapUp

ROLE = """You are the Data Profiler of a quantitative research team. You receive an unfamiliar raw dataset and
produce a DatasetCard: what one row of each table represents, primary keys, identifier fields, time fields, and
the meaning of the important fields. Ground every meaning in the documentation (read_doc) or in profiling
statistics (profile_table, profile_time). For the main numeric fields also state their kind (level, flow, stock,
event, ratio...), unit, what a zero and what a missing value mean, and concepts they are easily confused with.
Consult every document the read_doc tool lists at least once: definitions are often split between a layout file and a
description of the data's scope. Report data-quality warnings you observe (nulls, odd ranges, amendments)."""

TASK = active().profiler_task


OVERVIEW_ROLE = """You are the Data Research Agent writing the overview of a dataset a newcomer reads first, before any
field. The DatasetCard (what each field means) is in the context. Say what the dataset is and what it is about, who
publishes it and why, which market and asset class it covers, which kinds of entities it contains (measure the mix of
identifiers with identifier_patterns rather than assuming, e.g. common stocks vs preferreds, warrants or funds; when one
form can cover several kinds of entity, such as plain tickers for both stocks and ETFs, say so), which values it holds
and in which units (say what it does not hold, e.g. no prices), how often records occur, the period it covers, what
researchers use it for, and the caveats a newcomer most easily gets wrong. Cite the documentation and tool runs."""

OVERVIEW_TASK = "Write the dataset overview (call the tools you need, then submit)."


class DataOverview(Agent):
    name = "overview"
    role_prompt = OVERVIEW_ROLE
    tool_names = ["list_tables", "profile_table", "identifier_patterns", "read_doc"]
    output_model = DatasetOverview
    max_steps = 10
    max_submit_failures = 4

    def extra_validate(self, o) -> list[str]:
        from quantaccelerator.agents.checks import check_overview
        return check_overview(o)


class Profiler(Agent):
    name = "profiler"
    role_prompt = ROLE
    tool_names = ["list_tables", "profile_table", "profile_time", "read_doc"]
    output_model = DatasetCardDraft
    max_output_tokens = 8192  # the DatasetCard is the largest output
    max_submit_failures = 3   # data and grounding checks often need an extra round

    def extra_validate(self, draft) -> list[str]:
        return check_dataset_card(draft)



# ---- pass 2: profiling (the Data Research Agent's research loop) -----------------------------------------------
RESEARCH_ROLE = """You are the Data Research Agent of a quantitative research team, in your second pass. The first pass
produced the DatasetCard (what the fields mean). Now find out what this dataset is like as research material, the way
an experienced quant researcher would before trusting it: how much independent information it holds, how coverage
evolves, what is missing and whether missing differs from zero, how the main fields are distributed and how they move,
whether anything changes abruptly (vendor or methodology changes look like simultaneous jumps across many entities),
and how the data should be cleaned and represented. Work as a loop: ask a question, run the tool that answers it
(each draws a figure for the researcher), read the numbers, decide the next question. You may call at most two
tools per turn. Tool results may carry `flags`: material facts you must address in your findings (report them, or
explain in warnings why they do not matter).
You are return-blind: never use or ask for returns, prices, IC, Sharpe or P&L. Recommend representations from the
data's structure and meaning, never from predictive performance. Cleaning is only proposed, never applied: prefer
flagging rows over dropping them, drop only exact duplicates or rows the documentation says are invalid, and preview
every step first."""

RESEARCH_TASK = """Research the dataset described by the DatasetCard in the context, keeping a lab notebook.
Cover, using the tools:
1. research breadth (entities, history, frequency, concentration) and coverage over time;
2. key uniqueness of the main table;
3. missingness over time and by entity, keeping apart nulls, zeros and absent entity-dates;
4. the distribution of every numeric field, and of the natural ratio where one field is a part of another;
5. update dynamics and where the variance lies (across entities vs over time) for the main measure;
6. structural breaks in counts and in the main measure, and the mix of coded fields over time.
Record as you go, right after the tool result that establishes it:
- record_observation for each fact worth telling the researcher (one fact each; severity warning or issue when it
  affects research); address every flag a tool raises;
- rate_capability for history_depth, cross_sectional_breadth, update_frequency and coverage_stability;
- note_preprocessing for the main fields;
- propose_cleaning_step for each cleaning step (flags preferred; only steps that change rows).
Pursue the questions your results raise. When everything is recorded, call submit with the wrap-up: what the data is
suitable and weak for, its major limits, warnings, open questions for the researcher, and key corrections if your key
checks showed the DatasetCard's key is wrong. Keep it concise.
"""


class DataResearcher(Agent):
    name = "data_research"
    role_prompt = RESEARCH_ROLE
    output_model = ResearchWrapUp
    max_steps = 50
    max_tool_calls_per_turn = 2
    max_submit_failures = 3
    max_output_tokens = 3072
    max_prose_chars = 600
    prose_reminder = ("Your last message was long prose. Keep reasoning to a few sentences, and record each finding with "
                      "record_observation / rate_capability / note_preprocessing / propose_cleaning_step instead of "
                      "writing it as text: text is not part of your findings.")
    card = None  # the pass-1 DatasetCard, set by the orchestrator: it defines the research checklist

    @property
    def unpaced_tools(self) -> tuple:
        from quantaccelerator.tools.findings import RECORD_TOOLS
        return tuple(RECORD_TOOLS)

    @property
    def tool_names(self) -> list[str]:
        import os

        from quantaccelerator.tools.explore import EXPLORE_TOOLS
        from quantaccelerator.tools.findings import RECORD_TOOLS
        eyes = ["look_at_figure"] if os.environ.get("LLM_VISION_MODEL") else []
        return [t for t in EXPLORE_TOOLS if t != "look_at_figure"] + eyes + ["preview_cleaning", "read_doc"] + \
            RECORD_TOOLS

    def extra_validate(self, wrap) -> list[str]:
        from quantaccelerator.tools import registry
        from quantaccelerator.tools.findings import CAPABILITY_DIMENSIONS, assemble
        todo = missing_analyses(self.card) if self.card is not None else []
        if todo:
            return ["Not yet: before submitting, run these checks (they are part of the task): " + "; ".join(todo)]
        f = registry.CTX.findings
        errs = []
        unrated = [d for d in CAPABILITY_DIMENSIONS[:4] if d not in f.get("capability", {})]
        if unrated:
            errs.append(f"rate these capability dimensions first (rate_capability): {unrated}")
        if len(f.get("observations", [])) < 3:
            errs.append("record your observations first (record_observation), at least three")
        return errs or unaddressed_flags(assemble(wrap))


QA_ROLE = """You are the Data Research Agent answering a quant researcher's follow-up question about a dataset you have
already profiled (your DatasetCard and findings are in the context). Answer with the tools: check before you claim,
cite the run_ids, take numbers and dates only from tool results, and say plainly when the data cannot answer the
question. Stay return-blind. Keep the answer short."""


class DataQA(DataResearcher):
    name = "data_qa"
    role_prompt = QA_ROLE
    output_model = DataAnswer
    max_steps = 12
    max_output_tokens = 2048

    @property
    def tool_names(self) -> list[str]:
        from quantaccelerator.tools.findings import RECORD_TOOLS
        return [t for t in super().tool_names if t not in RECORD_TOOLS]

    given = ""  # the question (and what prompted it): dates the researcher asked about count as known

    def extra_validate(self, draft) -> list[str]:
        return check_answer(draft, self.given)
