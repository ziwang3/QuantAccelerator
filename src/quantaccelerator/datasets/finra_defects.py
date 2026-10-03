"""Benchmark copy of the FINRA case study with five planted data defects (see quantaccelerator.eval.gold_finra_defects)."""
from quantaccelerator.datasets.finra import _variant
from quantaccelerator.paths import ROOT

PROFILE = _variant("finra_defects", "FINRA short volume · planted-defect benchmark",
                   ROOT / "data" / "benchmark" / "finra_defects")
