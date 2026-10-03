"""Benchmark copy of the FINRA case study cut to 1 year x 40 symbols: too thin for broad cross-sectional research."""
from quantaccelerator.datasets.finra import _variant
from quantaccelerator.paths import ROOT

PROFILE = _variant("finra_small", "FINRA short volume · 1 year x 40 symbols", ROOT / "data" / "benchmark" / "finra_small")
