"""EDGAR acceptance-time tools."""
from quantaccelerator.ingest.timezone_check import consistency_table
from quantaccelerator.tools.data import load_table, table_path
from quantaccelerator.tools.registry import tool


@tool("lookup_acceptance", "Look up an accession number in the EDGAR submissions index: form, filingDate and "
      "the raw acceptanceDateTime string exactly as stored.",
      {"type": "object", "properties": {"accession": {"type": "string"}}, "required": ["accession"]},
      inputs=lambda accession: [table_path("edgar_acceptance")])
def lookup_acceptance(accession: str):
    df = load_table("edgar_acceptance")
    r = df[df.accession == accession]
    if r.empty:
        return {"found": False}
    return {"found": True, **r.iloc[0].to_dict()}


@tool("timezone_check", "Empirically test which timezone EDGAR acceptanceDateTime values are expressed in by "
      "checking consistency with EDGAR's filing-date cutoffs (Forms 3/4/5: submitted by 22:00 ET -> same-day "
      "filing date; other forms: 17:30 ET) under two readings: A = values are true UTC, B = values are Eastern "
      "time mislabeled with 'Z'.",
      {"type": "object", "properties": {}}, inputs=lambda: [table_path("edgar_acceptance")])
def timezone_check():
    return consistency_table()
