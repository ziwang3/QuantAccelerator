"""Collect everything one case study's pages show into a JSON-serialisable dict (for the active dataset profile).

Every number comes from run artifacts (score.json, run_meta.json, state/, events) or from gold / interim data;
nothing is typed in by hand. Run once per module: `QA_DATASET=<profile> python -m quantaccelerator.viz.collect <batch> ...`.
"""
import argparse
import difflib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quantaccelerator.datasets import active
from quantaccelerator.eval.score_demo import load_gold
from quantaccelerator.ingest.calendar import trading_days
from quantaccelerator.paths import ROOT
from quantaccelerator.tools.data import TABLES, table_path
from quantaccelerator.viz.events import run_events

MODULE = {
    "form4": {"name": "Look-Ahead Audit", "kicker": "SEC Form 4 insider purchases",
              "short": {"clean": "Clean", "leak_transdate": "Transaction date", "leak_filingdate": "Filing date",
                        "leak_noclose": "Ignores market hours"},
              "what": {"clean": "Keys each event on the first market open after the SEC accepted the filing.",
                       "leak_transdate": "Keys each event on the day the insider traded, days before anyone knew.",
                       "leak_filingdate": "Keys each event on the filing date, so a filing accepted at 2 pm is "
                                          "traded at that morning's open.",
                       "leak_noclose": "Uses the acceptance time but only pushes after-close filings forward, "
                                       "missing the intraday case."},
              "unit": "purchase filings",
              # display names for the batch directories (same model and settings; see batch_notes)
              "batch_labels": {"batch1": "Prototype", "batch2": "QuantAccelerator v1"},
              # what changed between the compared batches (same model, data, pipelines and answer key)
              "batch_notes": {
                  "summary": "Prototype is the first complete build: the same three agents, tools, prompts and human "
                             "gate. QuantAccelerator v1 adds the evidence checks and tooling fixes below. Both use the same model "
                             "(Qwen2.5-72B-Instruct, local, temperature 0), data, pipelines and answer key; no domain "
                             "knowledge was added to the prompts.",
                  "changes": [
                      ["Profiler guessed code meanings (“M = Merger”)",
                       "Document search follows “see Appendix 6.2” references to the real code list; each meaning must "
                       "match the page it cites"],
                      ["PIT agent chose Eastern time despite its own UTC evidence",
                       "Evidence-consistency check: the rule must agree with the timezone test it cites"],
                      ["Malformed or runaway tool calls wasted steps",
                       "Per-call parsing, output caps, retries for truncated answers"],
                      ["Context overflow after rejected drafts; patch loop on indentation",
                       "Draft compaction and context-aware output caps; patches keep indentation; repeated calls "
                       "are not re-run"]]},
              "table_desc": {"submission": "one row per filing (Forms 3/4/5 and amendments)",
                             "nonderiv_trans": "one row per non-derivative transaction line",
                             "reportingowner": "one row per reporting insider per filing",
                             "edgar_acceptance": "EDGAR acceptance timestamp per filing (ground truth for "
                                                 "publication time)"}},
    "alfred": {"name": "Revision Audit", "kicker": "US macro releases and revisions",
               "short": {"clean": "Clean", "leak_latest": "Latest (revised) values", "leak_obsdate": "Observation date",
                         "leak_periodend": "Period end"},
               "what": {"clean": "Uses each value as first published, from the first open after its release.",
                        "leak_latest": "Uses today's revised values, which nobody had at the time.",
                        "leak_obsdate": "Keys each value on the first day of the period it describes, weeks "
                                        "before it was published.",
                        "leak_periodend": "Keys each value on the last day of its period, still days to weeks "
                                          "before the release."},
               "unit": "macro observations",
               "batch_notes": {
                   "summary": "Prototype = QuantAccelerator v1 with the post-batch-1 verification fixes switched off (QA_BUILD="
                              "prototype); same model (Qwen2.5-72B-Instruct, local, temperature 0), prompts, tools, "
                              "data and answer key. On macro data the two builds perform alike: the fixes target "
                              "problems this dataset does not have.",
                   "changes": [
                       ["No code lists to guess (vintages have no coded fields)",
                        "The grounding check on code meanings has nothing to catch"],
                       ["No timezone trap (release dates, all published before the open)",
                        "The time-evidence check matters only for the revision policy, which both builds got right"]]},
               "batch_labels": {"macro1": "QuantAccelerator v1", "macro-proto": "Prototype"},
               "table_desc": {"vintages": "every published value (vintage) of every observation, with the dates "
                                          "during which it was current",
                              "series_meta": "FRED metadata per series: title, units, frequency, notes"}},
}


def _pipeline_output(run_dir: Path, variant: str) -> Path | None:
    stem = Path(active().pipeline(variant)).stem
    outs = sorted(p for p in (run_dir / "pipeline_outputs").glob(f"{stem}_*.parquet") if "patched" not in p.name)
    return outs[-1] if outs else None


def _run(run_dir: Path, variant: str) -> dict:
    src = (ROOT / active().pipeline(variant)).read_text()
    patched = run_dir / "patched.py"
    diff = ""
    if patched.exists():
        diff = "".join(difflib.unified_diff(src.splitlines(True), patched.read_text().splitlines(True),
                                            fromfile=f"{variant}.py", tofile=f"{variant}.py (agent patch)", n=2))
    state = {n: json.loads(p.read_text()) for n in ("dataset_card", "pit_contract", "audit_report")
             if (p := run_dir / "state" / f"{n}.json").exists()}
    score = json.loads((run_dir / "score.json").read_text()) if (run_dir / "score.json").exists() else None
    return {"variant": variant, "events": run_events(run_dir), "source": src, "diff": diff, "state": state,
            "score": score, "meta": json.loads((run_dir / "run_meta.json").read_text())}


# ---------------------------------------------------------------- Form 4
def form4_story(outputs: dict[str, Path]) -> dict:
    """One real intraday filing that every leaky variant uses too early (all three leaks on one timeline)."""
    from quantaccelerator.paths import insider_dir
    gold = pd.read_parquet(active().benchmark / "pit_gold.parquet").set_index("accession")
    asof = {v: pd.read_parquet(p).set_index("ACCESSION_NUMBER") for v, p in outputs.items()}
    common = sorted(set.intersection(*(set(a.index) for a in asof.values())))
    nt = pd.read_parquet(insider_dir() / "nonderiv_trans.parquet", columns=["ACCESSION_NUMBER", "TRANS_CODE",
                                                                          "TRANS_DATE"])
    trans = nt[nt.TRANS_CODE == "P"].groupby("ACCESSION_NUMBER").TRANS_DATE.min()
    df = gold.loc[gold.index.intersection(common)].join(trans.rename("trans_date"))
    df = df.join(asof["clean"][["ISSUERTRADINGSYMBOL", "value"]])
    for v, a in asof.items():
        df[f"asof_{v}"] = a.asof_date
    lag = (df.acceptance_et.dt.normalize() - df.trans_date).dt.days
    cand = df[(df.stratum == "intraday") & lag.between(2, 5) & df.ISSUERTRADINGSYMBOL.notna()
              & (df.asof_leak_noclose < df.first_tradable_date) & (df.asof_leak_filingdate < df.first_tradable_date)]
    row = cand.sort_values("value", ascending=False).iloc[0]
    iso = lambda x: pd.Timestamp(x).isoformat()
    return {"kind": "form4", "accession": row.name, "ticker": row.ISSUERTRADINGSYMBOL, "value_usd": float(row.value),
            "trans_date": iso(row.trans_date), "acceptance_et": iso(row.acceptance_et),
            "filing_date": iso(row.FILING_DATE), "first_tradable": iso(row.first_tradable_date),
            "asof": {v: iso(row[f"asof_{v}"]) for v in outputs}}


def form4_extras(outputs: dict[str, Path]) -> dict:
    from quantaccelerator.ingest.timezone_check import consistency_table
    ev = pd.read_parquet(outputs["clean"]) if "clean" in outputs else None
    tz = consistency_table()
    return {"story": form4_story(outputs) if len(outputs) == len(active().variants) else None,
            "facts": [{"v": int(len(ev)), "l": "insider-purchase filings audited"},
                      {"v": int(ev.ISSUERCIK.nunique()), "l": "issuers"}] if ev is not None else [],
            "source": "SEC Insider Transactions Data Sets (Forms 3/4/5), joined to EDGAR acceptance timestamps.",
            "docs": ["SEC Insider Transactions Data Sets readme (field definitions, code lists)",
                     "SEC overview of Forms 3, 4 and 5 (who files, deadlines)"],
            "truth": "Ground truth: the exact EDGAR acceptance time of every filing gives the true first-tradable open.",
            "evidence": {"kind": "timezone", "summary": tz["summary"], "n_filings": tz["n_filings"],
                         "hist": {k: {str(h): int(n) for h, n in v.items()}
                                  for k, v in tz["form345_hour_hist_et"].items()}}}


# ---------------------------------------------------------------- ALFRED
def alfred_story(outputs: dict[str, Path]) -> dict:
    """The payroll observation since 2022 with the largest revision: shows release lag and revision in one place."""
    from quantaccelerator.eval.gold_macro import releases, vintages
    v, r = vintages(), releases()
    r = r[(r.series_id == "PAYEMS") & (r.date >= "2022-01-01")]
    x = r.loc[(r.latest_value - r.first_value).abs().idxmax()]
    vv = v[(v.series_id == x.series_id) & (v.date == x.date)].sort_values("realtime_start")
    cal = trading_days()
    asof = {k: pd.read_parquet(p).set_index(["series_id", "date"]).loc[(x.series_id, x.date)] for k, p in outputs.items()}
    iso = lambda t: pd.Timestamp(t).isoformat()
    period_end = x.date + pd.offsets.MonthEnd(0)
    return {"kind": "alfred", "series_id": x.series_id, "title": "Total nonfarm payrolls", "units": "thousands of persons",
            "date": iso(x.date), "period_end": iso(period_end), "release_date": iso(x.release_date),
            "first_tradable": iso(cal[cal.searchsorted(x.release_date)]),
            "vintages": [{"from": iso(a.realtime_start), "to": None if a.realtime_end.year > 2200 else iso(a.realtime_end),
                          "value": float(a.value)} for a in vv.itertuples()],
            "asof": {k: iso(a.asof_date) for k, a in asof.items()},
            "value_used": {k: float(a.value) for k, a in asof.items()}}


def alfred_extras(outputs: dict[str, Path]) -> dict:
    from quantaccelerator.eval.gold_macro import releases
    r = releases()
    lag = {s: {"median": float((g.release_date - g.date).dt.days.median()),
               "p10": float((g.release_date - g.date).dt.days.quantile(.1)),
               "p90": float((g.release_date - g.date).dt.days.quantile(.9))} for s, g in r.groupby("series_id")}
    rev = {s: {"share": float(g.revised.mean()),
               "median_pct": float((100 * (g.latest_value - g.first_value).abs() / g.first_value.abs())[g.revised].median())
               if g.revised.any() else 0.0} for s, g in r.groupby("series_id")}
    meta = pd.read_parquet(TABLES["series_meta"][1]).set_index("series_id")
    return {"story": alfred_story(outputs) if len(outputs) == len(active().variants) else None,
            "facts": [{"v": int(len(r)), "l": "macro observations audited (2019 onward)"},
                      {"v": int(r.series_id.nunique()), "l": "series: payrolls, unemployment, CPI, GDP, industrial "
                                                            "production, retail sales"}],
            "source": "ALFRED (St. Louis Fed): every published vintage of each series since it began.",
            "docs": ["FRED API documentation on real-time periods and vintages", "FRED series notes",
                     "BLS, BEA, Census and Federal Reserve release schedules (release times of day)"],
            "truth": "Ground truth: each vintage's first publication date, and the value that was current on every "
                     "date.",
            "evidence": {"kind": "macro", "lag": lag, "revisions": rev,
                         "titles": meta.title.to_dict(), "frequency": meta.frequency.to_dict()}}


def collect(batch_dir: Path, compare: list[Path] | None = None) -> dict:
    p = active()
    batch_dir = Path(batch_dir)
    gold = load_gold()
    runs = {v: _run(batch_dir / v, v) for v in p.variants if (batch_dir / v / "run_meta.json").exists()}
    outputs = {v: o for v in p.variants if (o := _pipeline_output(batch_dir / v, v))}
    batches = []
    for b in [*(compare or []), batch_dir]:
        label = MODULE[p.name].get("batch_labels", {}).get(b.name, b.name)
        scores = {v: json.loads((b / v / "score.json").read_text()) for v in p.variants
                  if (b / v / "score.json").exists()}
        batches.append({"name": b.name, "label": label, "scores": scores})
    tables = [{"table": t, "rows": pq.ParquetFile(table_path(t)).metadata.num_rows,
               "desc": MODULE[p.name]["table_desc"].get(t, "")} for t in TABLES]
    cal = trading_days()
    extras = {"form4": form4_extras, "alfred": alfred_extras}[p.name](outputs)
    return {"id": p.name, **{k: MODULE[p.name][k] for k in ("name", "kicker", "short", "what", "unit")},
            "batch_notes": MODULE[p.name].get("batch_notes"),
            "subtitle": p.title, "batch": batch_dir.name, "variants": p.variants, "runs": runs,
            "gold": {v: {k: g.get(k) for k in ("leak", "leak_types", "line", "asof_code", "true_lookahead", "note")}
                     for v, g in gold["variants"].items()},
            "gold_rule": gold["gold_rule"], "tables": tables,
            "calendar": {"n_days": int(len(cal)), "start": str(cal.min().date()), "end": str(cal.max().date())},
            "batches": batches, **extras}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).write_text(json.dumps(collect(ROOT / a.batch, [ROOT / c for c in a.compare]), default=str))


if __name__ == "__main__":
    main()
