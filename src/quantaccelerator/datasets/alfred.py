"""Revision Audit case study: US macro releases with their full vintage history (ALFRED, St. Louis Fed)."""
from quantaccelerator.datasets import Profile
from quantaccelerator.paths import EXTERNAL_DOCS, INTERIM, ROOT

DATASET_ID = "alfred_macro_vintages"
D = INTERIM / "alfred"

PROFILER_TASK = f"""Dataset `{DATASET_ID}`: the real-time (vintage) history of six US macro series from ALFRED, the
St. Louis Fed's archive of every published version of FRED data, with tables `vintages` and `series_meta`.
Produce the DatasetCard.
Cover:
- every table: observation unit, primary key, and the run_id of the profile_table call that measured it;
- every date/time column (all tables), stating what each date means;
- the identifier columns;
- the coded categorical columns (for example series_id), with value_meanings taken from the documentation.
Keep each meaning to one sentence. Ground meanings in the documentation (read_doc); never guess."""

PIT_TASK = f"""Build the PITContract for dataset `{DATASET_ID}` (every published vintage of six US macro series).
Execution convention of the research team: daily data; positions are entered at the 09:30 ET open on NYSE trading
days; a value may be used at the first open at which it was already published.
Decide, with evidence from the tools and documentation:
1. which field marks when each value became public, and the first session at which it could be traded (check the
   release lags and the agencies' documented release times of day);
2. which vintage of a value a backtest may use at a given date (check whether and how often values are revised).
Preview your rule on real observations before submitting. The DatasetCard produced by the Data Profiler is given
as context."""

AUDIT_TASK = """Audit the pipeline `{pipeline}`. Its output has one row per (series_id, date) observation with
asof_date = the session whose open is the first time the observation is used, and value = the value used.
The signed PIT contract is given as context."""


def _gold_rule():
    from quantaccelerator.eval.gold_macro import GOLD_RULE
    return GOLD_RULE


def _measure(events, rule):
    from quantaccelerator.eval.gold_macro import lookahead_stats_macro
    return lookahead_stats_macro(events, rule)


def _rule_agreement(rule) -> float:
    from quantaccelerator.eval.gold_macro import rule_agreement_rows
    return float(rule_agreement_rows(rule).ok.mean())


def _rule_by_stratum(rule) -> dict:
    from quantaccelerator.eval.gold_macro import rule_agreement_rows
    r = rule_agreement_rows(rule)
    return {"overall": round(float(r.ok.mean()), 4), **{k: round(float(v), 4) for k, v in r.groupby("series_id").ok.mean().items()}}


def _check_pit(contract) -> list[str]:
    """The chosen vintage policy must agree with the agent's own revision_stats evidence."""
    from quantaccelerator.tools import registry
    rule = contract.availability_rule
    runs = [r for r in (registry.get_run(x) for x in contract.evidence) if r and r["tool"] == "revision_stats"]
    if not runs:
        mine = []
        try:
            sr = registry.session_runs(registry.CTX.session_id)
            mine = list(sr[(sr.tool == "revision_stats") & (sr.ok == 1)].run_id)
        except Exception:
            pass
        return ["evidence: the top-level evidence list must cite a revision_stats run to justify the value_vintage "
                "policy" + (f"; add {mine[-1]!r} (your revision_stats run) to it" if mine else " (call it and cite it)")]
    res = runs[-1]["result"]
    revised = max((s.get("share_revised_after_first_release", 0) for s in res.values() if isinstance(s, dict)),
                  default=0)
    if revised > 0 and rule.value_vintage != "as_of_date":
        return [f"availability_rule.value_vintage={rule.value_vintage!r} contradicts your revision_stats run "
                f"{runs[-1]['run_id']}: up to {revised:.0%} of observations are revised after first publication. "
                "Reconcile the rule with the evidence."]
    return []


PROFILE = Profile(
    name="alfred", title="US macro releases and revisions · ALFRED", dataset_id=DATASET_ID,
    tables={"vintages": D / "vintages.parquet", "series_meta": D / "series_meta.parquet"},
    keys={"vintages": "series_id", "series_meta": "series_id"},
    docs={n: EXTERNAL_DOCS / f"{n}.txt" for n in
          ("fred_realtime_period", "fred_series_observations", "alfred_about", "fred_series_notes",
           "bls_empsit_schedule", "bls_cpi_schedule", "fed_g17", "bea_schedule", "census_marts")},
    profiler_task=PROFILER_TASK, pit_task=PIT_TASK, audit_task=AUDIT_TASK,
    pit_tools=["profile_time", "read_doc", "release_lag", "revision_stats", "vintage_lookup", "preview_macro_rule"],
    audit_tools=["read_source", "run_pipeline", "measure_lookahead", "test_patch"],
    pipeline_glob="macro_signal_{variant}.py",
    variants=["clean", "leak_latest", "leak_obsdate", "leak_periodend"],
    benchmark=ROOT / "data" / "benchmark" / "macro",
    gold_rule=_gold_rule, measure=_measure, rule_agreement=_rule_agreement, rule_by_stratum=_rule_by_stratum,
    check_pit=_check_pit,
    doc_help="fred_realtime_period and fred_series_observations (FRED API docs: real-time periods, vintages), "
             "alfred_about (what ALFRED archives), fred_series_notes (each series' title, units, frequency, notes), "
             "bls_empsit_schedule and bls_cpi_schedule (BLS release dates and times), fed_g17 (Fed industrial "
             "production release), bea_schedule (BEA GDP release schedule), census_marts (Census retail sales).",
)
