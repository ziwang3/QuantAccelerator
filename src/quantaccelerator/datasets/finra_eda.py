"""Benchmark copy of the FINRA case study for the Research EDA Agent: the research panel's universe and years with
three vendor fields of known structure added (see quantaccelerator.eval.gold_eda)."""
import dataclasses

from quantaccelerator.datasets.finra import PROFILE as FINRA, _variant
from quantaccelerator.eval.gold_eda import IDEAS
from quantaccelerator.paths import ROOT

VENDOR = ["vendor_mentions", "vendor_sentiment", "vendor_quality"]

_base = _variant("finra_eda", "FINRA short volume · planted-dependency benchmark (research EDA)",
                 ROOT / "data" / "benchmark" / "finra_eda")
PROFILE = dataclasses.replace(
    _base, view={**FINRA.view, "values": FINRA.view["values"] + VENDOR},
    panel={**FINRA.panel, "researcher_ideas": list(IDEAS.values()), "field_notes": {
        "ShortVolume": "shares sold short that day in FINRA-reported (off-exchange) trades",
        "ShortExemptVolume": "the part of ShortVolume marked short exempt",
        "TotalVolume": "all shares traded that day in FINRA-reported (off-exchange) trades",
        "vendor_mentions": "an alternative-data vendor's daily count of mentions of the company (undocumented)",
        "vendor_sentiment": "the same vendor's daily sentiment score (undocumented scale)",
        "vendor_quality": "the same vendor's daily 'quality' score (undocumented scale)"}})
