"""Short-volume case study: FINRA Reg SHO daily short-sale volume files (Consolidated NMS), 2018-08 to 2026-09."""
from quantaccelerator.datasets import Profile
from quantaccelerator.paths import EXTERNAL_DOCS, INTERIM, RAW, ROOT

DATASET_ID = "finra_reg_sho_daily_short_volume"
D = INTERIM / "finra"

PROFILER_TASK = f"""Dataset `{DATASET_ID}`: FINRA's Regulation SHO daily short-sale volume files (Consolidated NMS),
one file per trade date from August 2018, loaded as table `shvol` (the file lines) and table `file_log` (one row per
daily file with its record count and trailer check). Produce the DatasetCard.
Cover:
- every table: observation unit, primary key, and the run_id of the profile_table call that measured it;
- every date/time column, stating what each date means;
- the identifier columns;
- the volume columns: what each measures, in which unit, and what it does NOT measure;
- the coded categorical columns (for example Market), with value_meanings taken from the documentation.
Keep each meaning to one sentence. Ground meanings in the documentation (read_doc); never guess."""

PIT_TASK = f"""Build the PITContract for dataset `{DATASET_ID}` (daily short-sale volume per security and trade date).
Execution convention of the research team: daily data; positions are entered at the 09:30 ET open on NYSE trading
days; a record may be used at the first open at which it was already public.
Decide, with evidence from the documentation and tools, when each day's file becomes public (check FINRA's stated
publication time) and the first session whose open could trade on it, including after Fridays and before market
holidays. State whether published files can later change. Preview your rule on real trade dates before submitting.
The DatasetCard produced by the Data Profiler is given as context."""

AUDIT_TASK = """Audit the pipeline `{pipeline}`. Its output has one row per (Symbol, date) with asof_date = the
session whose open is the first time the row is used, and value = the signal value used then.
The signed PIT contract is given as context."""


def _gold_rule():
    from quantaccelerator.eval.gold_finra import GOLD_RULE
    return GOLD_RULE


def _measure(events, rule):
    from quantaccelerator.eval.gold_finra import lookahead_stats_finra
    return lookahead_stats_finra(events, rule)


def _rule_agreement(rule) -> float:
    from quantaccelerator.eval.gold_finra import rule_agreement_rows
    return float(rule_agreement_rows(rule).ok.mean())


def _rule_by_stratum(rule) -> dict:
    from quantaccelerator.eval.gold_finra import rule_agreement_rows
    r = rule_agreement_rows(rule)
    return {"overall": round(float(r.ok.mean()), 4), **{k: round(float(v), 4) for k, v in r.groupby("stratum").ok.mean().items()}}


def _crsp():
    from quantaccelerator.ingest import crsp
    return crsp


def _first_tradable(rule, df):
    from quantaccelerator.tools.finra import apply_finra_rule
    return apply_finra_rule(rule, df["Date"])


PROFILE = Profile(
    name="finra", title="US short-sale volume · FINRA Reg SHO daily files", dataset_id=DATASET_ID,
    tables={"shvol": D / "shvol.parquet", "file_log": D / "file_log.parquet"},
    keys={"shvol": "Symbol", "file_log": "file_date"},
    docs={"finra_short_sale_layout": RAW / "docs" / "finra_daily_short_sale_layout.pdf",
          "finra_daily_short_sale_files": EXTERNAL_DOCS / "finra_daily_short_sale_files.txt"},
    profiler_task=PROFILER_TASK, pit_task=PIT_TASK, audit_task=AUDIT_TASK,
    pit_tools=["profile_time", "read_doc", "next_tradable", "preview_finra_rule"],
    audit_tools=["read_source", "run_pipeline", "measure_lookahead", "test_patch"],
    pipeline_glob="finra_signal_{variant}.py",
    variants=["clean", "leak_sameday", "leak_centered", "leak_fullmean"],
    benchmark=ROOT / "data" / "benchmark" / "finra",
    gold_rule=_gold_rule, measure=_measure, rule_agreement=_rule_agreement, rule_by_stratum=_rule_by_stratum,
    view={"table": "shvol", "entity": "Symbol", "time": "Date",
          "values": ["ShortVolume", "ShortExemptVolume", "TotalVolume"]},
    first_tradable=_first_tradable,
    exchange_tickers=True,
    panel={"universe_n": 1000, "formation": ["2021-01-01", "2021-12-31"], "start": "2022-01-01", "end": "2025-12-31",
           "scale_field": "TotalVolume", "adv_window": 20,
           "researcher_ideas": ["Do short sellers act before bad news? Measure how unusual today's short selling is for "
                                "a stock compared with its own recent history."],
           "field_notes": {"ShortVolume": "shares sold short that day in FINRA-reported (off-exchange) trades",
                           "ShortExemptVolume": "the part of ShortVolume marked short exempt",
                           "TotalVolume": "all shares traded that day in FINRA-reported (off-exchange) trades"}},
    predict={"returns": lambda: _crsp().returns_frame(), "link": lambda pairs: _crsp().link(pairs),
             # CRSP through 2025 assumed; shift the segments if the CRSP vintage ends earlier
             "split": {"discovery": ("2022-01-03", "2023-11-30"), "validation": ("2024-01-02", "2024-11-29"),
                       "locked_test": ("2025-01-02", "2025-12-31"), "embargo_sessions": 20}},
    doc_help="finra_short_sale_layout (FINRA's file layout: field definitions and Market facility codes), "
             "finra_daily_short_sale_files (FINRA's page on the daily files: what they cover, when they are posted, "
             "file history).",
)


def _variant(name: str, title: str, d) -> Profile:
    """The same case study on a benchmark copy of the data (planted defects, or a deliberately thin sample)."""
    import dataclasses
    return dataclasses.replace(PROFILE, name=name, title=title, dataset_id=f"{DATASET_ID}__{name}",
                               tables={"shvol": d / "shvol.parquet", "file_log": d / "file_log.parquet"})
