"""Build the demo gold set: data/benchmark/demo/{pit_gold.parquet, rule_sample.parquet, gold.json}.

Ground truth comes from deterministic sources only:
  - first tradable date = first 09:30 ET open strictly after EDGAR acceptance (acceptanceDateTime is UTC;
    established by the timezone test), NYSE calendar from SPY
  - leak labels = the one line that differs between the clean and leaky pipeline variants
  - field semantics = sec_insider_readme / sec_forms_3_4_5_overview pages
"""
import json

import pandas as pd

from quantaccelerator.ingest import calendar as cal
from quantaccelerator.ingest import manifest
from quantaccelerator.paths import BENCHMARK, NOTEBOOKS, ROOT, SPY_CSV, insider_dir
from quantaccelerator.tools.lookahead import lookahead_stats
from quantaccelerator.tools.pipeline import execute
from quantaccelerator.tools.timing import GOLD_RULE, apply_rule, filing_times, parse_to_et

VARIANTS = {
    "clean": {"leak": False},
    "leak_transdate": {"leak": True, "leak_types": ["event_date_as_availability"],
                       "note": "keys events on the insider's transaction date instead of public availability"},
    "leak_filingdate": {"leak": True, "leak_types": ["timezone_session_ambiguity", "same_bar_execution",
                                                     "event_date_as_availability"],
                        "note": "FILING_DATE is a day-granular publication date; filings accepted intraday or after "
                                "the close (up to 22:00 ET) are treated as tradable at that day's open"},
    "leak_noclose": {"leak": True, "leak_types": ["timezone_session_ambiguity", "same_bar_execution"],
                     "note": "uses the acceptance time but treats filings accepted 09:30-16:00 ET as tradable at "
                             "that day's open (only after-16:00 acceptances are pushed to the next day)"},
}
ASOF_MARKER = 'ev["asof_date"] ='

# Semantic questions: graded deterministically from the DatasetCard / PITContract (see score_demo.grade_semantic).
# `groups`: every group must be matched by at least one of its (case-insensitive) substrings.
SEMANTIC = [
    {"id": "sem01", "question": "What does nonderiv_trans.TRANS_DATE mean?", "target": "field:nonderiv_trans.TRANS_DATE",
     "gold": "Transaction date: the date the insider executed the trade.", "groups": [["transaction", "trade", "execut"]],
     "evidence": "sec_insider_readme p3"},
    {"id": "sem02", "question": "What does submission.FILING_DATE mean?", "target": "field:submission.FILING_DATE",
     "gold": "Filing date with the Commission, sourced from EDGAR.", "groups": [["fil"], ["sec", "commission", "edgar"]],
     "evidence": "sec_insider_readme p2"},
    {"id": "sem03", "question": "What does submission.PERIOD_OF_REPORT mean?",
     "target": "field:submission.PERIOD_OF_REPORT", "gold": "Date of the event requiring the statement.",
     "groups": [["event", "earliest transaction", "requir"]], "evidence": "sec_insider_readme p2"},
    {"id": "sem04", "question": "What does nonderiv_trans.DEEMED_EXECUTION_DATE mean?",
     "target": "field:nonderiv_trans.DEEMED_EXECUTION_DATE", "gold": "Deemed execution date of the transaction.",
     "groups": [["deemed"], ["execut"]], "evidence": "sec_insider_readme p3"},
    {"id": "sem05", "question": "What does TRANS_CODE P mean?", "target": "code:TRANS_CODE:P",
     "gold": "Open market or private purchase.", "groups": [["purchase", "buy"], ["open market", "open-market", "private",
                                                                                   "exchange"]],
     "evidence": "sec_insider_readme p5"},
    {"id": "sem06", "question": "What does TRANS_CODE M mean?", "target": "code:TRANS_CODE:M",
     "gold": "Exercise or conversion of a derivative security.", "groups": [["exercis", "conver"]],
     "evidence": "sec_insider_readme p5"},
    {"id": "sem07", "question": "What does TRANS_CODE A mean?", "target": "code:TRANS_CODE:A",
     "gold": "Grant, award or other acquisition pursuant to Rule 16b-3(d).", "groups": [["grant", "award"]],
     "evidence": "sec_insider_readme p5"},
    {"id": "sem08", "question": "What does TRANS_CODE F mean?", "target": "code:TRANS_CODE:F",
     "gold": "Payment of exercise price or tax liability by delivering or withholding securities.",
     "groups": [["tax", "exercise price", "withh"]], "evidence": "sec_insider_readme p5"},
    {"id": "sem09", "question": "What is the observation unit and key of nonderiv_trans?",
     "target": "table:nonderiv_trans", "gold": "One non-derivative transaction (Table I row); key ACCESSION_NUMBER + "
     "NONDERIV_TRANS_SK.", "groups": [["transaction"]], "key_must_include": ["NONDERIV_TRANS_SK"],
     "evidence": "sec_insider_readme p2-3"},
    {"id": "sem10", "question": "Which fields identify the filing, the issuer and the insider?",
     "target": "entity_ids", "gold": ["ACCESSION_NUMBER", "ISSUERCIK", "RPTOWNERCIK"],
     "evidence": "sec_insider_readme p2"},
    {"id": "sem11", "question": "Which field tells when a filing became public?", "target": "rule:source_field",
     "gold": "edgar_acceptance.acceptanceDateTime", "evidence": "ingest: FILING_DATE has no time of day; "
     "acceptanceDateTime is the EDGAR acceptance timestamp"},
    {"id": "sem12", "question": "In which timezone is acceptanceDateTime expressed?", "target": "rule:source_timezone",
     "gold": "UTC", "evidence": "timezone test (quantaccelerator.ingest.timezone_check)"},
    {"id": "sem13", "question": "What is the first-tradable rule?", "target": "rule:tradable_at",
     "gold": "next_open_after (lag 0)", "evidence": "project rule: first 09:30 ET open strictly after acceptance"},
    {"id": "sem14", "question": "What PIT role does TRANS_DATE play?", "target": "pit_role:nonderiv_trans.TRANS_DATE",
     "gold": "event_time", "evidence": "sec_forms_3_4_5_overview p1 (Form 4 due within two business days after the "
     "transaction date)"},
]


def pit_gold() -> pd.DataFrame:
    f = filing_times().dropna(subset=["acceptanceDateTime"]).copy()
    et = parse_to_et(f.acceptanceDateTime, "UTC")
    mins = et.dt.hour * 60 + et.dt.minute
    day = et.dt.normalize()
    trading = day.isin(cal.trading_days())
    nxt = cal.next_trading_day_vec(day)
    pre_holiday = trading & ((nxt - day).dt.days > 1) & ~(day.dt.weekday == 4)  # gap not explained by a weekend
    stratum = pd.Series("intraday", index=f.index)
    stratum[trading & (mins < 570)] = "pre_open"
    stratum[trading & (mins >= 960)] = "after_close"
    stratum[~trading] = "non_trading_day"
    stratum[pre_holiday & (mins >= 960)] = "pre_holiday_after_close"
    return pd.DataFrame({"acceptance_et": et, "first_tradable_date": apply_rule(GOLD_RULE),
                         "stratum": stratum, "FILING_DATE": f.FILING_DATE}).rename_axis("accession").reset_index()


def rule_sample(g: pd.DataFrame, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Stratified sample: equal share per stratum (all rows if a stratum is small), topped up at random."""
    per = n // g.stratum.nunique()
    parts = [grp.sample(min(per, len(grp)), random_state=seed) for _, grp in g.groupby("stratum")]
    s = pd.concat(parts)
    rest = g.drop(s.index)
    s = pd.concat([s, rest.sample(n - len(s), random_state=seed)])
    return s.reset_index(drop=True)


def build() -> dict:
    BENCHMARK.mkdir(parents=True, exist_ok=True)
    g = pit_gold()
    g.to_parquet(BENCHMARK / "pit_gold.parquet", index=False)
    s = rule_sample(g)
    s.to_parquet(BENCHMARK / "rule_sample.parquet", index=False)
    srcs = [insider_dir() / "acceptance_index.parquet", insider_dir() / "submission.parquet", SPY_CSV]
    manifest.record(BENCHMARK / "pit_gold.parquet", srcs, "quantaccelerator.eval.gold", rows=len(g))
    manifest.record(BENCHMARK / "rule_sample.parquet", srcs, "quantaccelerator.eval.gold", rows=len(s))

    variants = {}
    for name, meta in VARIANTS.items():
        path = NOTEBOOKS / f"insider_events_{name}.py"
        lines = path.read_text().splitlines()
        line = next(i for i, l in enumerate(lines, 1) if l.startswith(ASOF_MARKER))
        run = execute(path)
        stats = lookahead_stats(pd.read_parquet(ROOT / run["output_path"]), GOLD_RULE)
        (ROOT / run["output_path"]).unlink()
        variants[name] = {**meta, "file": str(path.relative_to(ROOT)), "line": line if meta["leak"] else None,
                          "asof_code": lines[line - 1].strip(), "true_lookahead": stats,
                          "expected_effect": "positive look-ahead (events used before tradable)" if meta["leak"]
                          else "none"}
    gold = {"gold_rule": GOLD_RULE.model_dump(), "variants": variants, "semantic": SEMANTIC,
            "rule_sample_strata": s.stratum.value_counts().to_dict()}
    (BENCHMARK / "gold.json").write_text(json.dumps(gold, indent=1, default=str))
    return gold


if __name__ == "__main__":
    out = build()
    for k, v in out["variants"].items():
        print(k, v["line"], v["true_lookahead"])
    print(out["rule_sample_strata"])
