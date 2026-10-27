"""Look-Ahead Audit case study: SEC Form 4 insider purchases (one quarter, default 2025q1 via $QA_QUARTER)."""
from quantaccelerator.datasets import Profile
from quantaccelerator.paths import BENCHMARK, DOCS, QUARTER, insider_dir

Q = QUARTER.upper()
DATASET_ID = f"sec_form345_{QUARTER}"

PROFILER_TASK = f"""Dataset `{DATASET_ID}`: the SEC insider-transactions data set (Forms 3/4/5) for {Q}, with tables
`submission`, `nonderiv_trans` and `reportingowner`, plus the auxiliary table `edgar_acceptance` taken from the
EDGAR submissions index. Produce the DatasetCard.
Cover:
- every table: observation unit, primary key, and the run_id of the profile_table call that measured it;
- every date/time column (all tables);
- the identifier columns for filings, issuers and reporting owners;
- the coded categorical columns used to select events (for example transaction codes, owner relationship,
  document type), with value_meanings for their most frequent codes taken from the documentation.
Keep each meaning to one sentence. When a definition refers to an appendix or code list, look that list up too;
never guess the meaning of a code."""

PIT_TASK = f"""Build the PITContract for dataset `{DATASET_ID}` (Form 3/4/5 filings; the auxiliary table
`edgar_acceptance` maps each ACCESSION_NUMBER to EDGAR's acceptanceDateTime).
Execution convention of the research team: daily data; positions are entered at the 09:30 ET open on NYSE trading
days; an event may be used at the first open at which it was already public.
Use the tools to check how the candidate time fields relate (lags, time-of-day, timezone) and preview your rule on
real filings before submitting. The DatasetCard produced by the Data Profiler is given as context."""

AUDIT_TASK = """Audit the pipeline `{pipeline}`. Its output has one row per filing with asof_date = the session whose open
is the first time the event is used. The signed PIT contract is given as context."""


def _gold_rule():
    from quantaccelerator.tools.timing import GOLD_RULE
    return GOLD_RULE


def _measure(events, rule):
    from quantaccelerator.tools.lookahead import lookahead_stats
    return lookahead_stats(events, rule)


def _rule_agreement(rule) -> float:
    import pandas as pd
    from quantaccelerator.tools.timing import apply_rule, filing_times
    s = pd.read_parquet(BENCHMARK / "rule_sample.parquet")
    got = apply_rule(rule, filing_times().loc[s.accession])
    return float((got.values == s.first_tradable_date.values).mean())


def _rule_by_stratum(rule) -> dict:
    import pandas as pd
    from quantaccelerator.tools.timing import apply_rule, filing_times
    s = pd.read_parquet(BENCHMARK / "rule_sample.parquet")
    s["ok"] = apply_rule(rule, filing_times().loc[s.accession]).values == s.first_tradable_date.values
    return {"overall": round(float(s.ok.mean()), 4),
            **{k: round(float(v), 4) for k, v in s.groupby("stratum").ok.mean().items()}}


def _check_pit(contract) -> list[str]:
    from quantaccelerator.agents.checks import check_pit_evidence
    return check_pit_evidence(contract)


d = insider_dir()
PROFILE = Profile(
    name="form4", title=f"SEC Form 4 insider purchases · {Q}", dataset_id=DATASET_ID,
    tables={"submission": d / "submission.parquet", "nonderiv_trans": d / "nonderiv_trans.parquet",
            "reportingowner": d / "reportingowner.parquet", "edgar_acceptance": d / "acceptance_index.parquet"},
    keys={"edgar_acceptance": "accession", "submission": "ACCESSION_NUMBER", "nonderiv_trans": "ACCESSION_NUMBER",
          "reportingowner": "ACCESSION_NUMBER"},
    docs=dict(DOCS),
    profiler_task=PROFILER_TASK, pit_task=PIT_TASK, audit_task=AUDIT_TASK,
    pit_tools=["profile_time", "lag_distribution", "read_doc", "lookup_acceptance", "timezone_check", "next_tradable",
               "preview_rule"],
    audit_tools=["read_source", "run_pipeline", "measure_lookahead", "test_patch", "next_tradable"],
    pipeline_glob="insider_events_{variant}.py",
    variants=["clean", "leak_transdate", "leak_filingdate", "leak_noclose"],
    benchmark=BENCHMARK, gold_rule=_gold_rule, measure=_measure, rule_agreement=_rule_agreement,
    rule_by_stratum=_rule_by_stratum, check_pit=_check_pit,
    doc_help="sec_insider_readme (field definitions and code lists of the Form 3/4/5 data set), "
             "sec_forms_3_4_5_overview (what Forms 3/4/5 are, filing deadlines).",
)
