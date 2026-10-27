import pandas as pd
import pytest

from quantaccelerator.ingest import calendar as cal
from quantaccelerator.ingest.timezone_check import to_et
from quantaccelerator.paths import insider_dir

D = insider_dir()
needs_data = pytest.mark.skipif(not (D / "acceptance_index.parquet").exists(), reason="run ingest first")


@needs_data
def test_insider_counts():
    s = pd.read_parquet(D / "submission.parquet")
    n = pd.read_parquet(D / "nonderiv_trans.parquet")
    assert len(s) == 63284 and len(n) == 103029
    assert (n.TRANS_CODE == "P").sum() == 5681


@needs_data
def test_acceptance_match_rate():
    s = pd.read_parquet(D / "submission.parquet", columns=["ACCESSION_NUMBER"])
    idx = pd.read_parquet(D / "acceptance_index.parquet")
    assert idx.accession.is_unique
    assert idx.accession.isin(s.ACCESSION_NUMBER).mean() == 1.0
    assert len(idx) / len(s) >= 0.99


def test_to_et_handles_dst():
    raw = pd.Series(["2025-01-15T21:30:00.000Z", "2025-07-15T21:30:00.000Z"])
    et = to_et(raw)
    assert list(et.dt.hour) == [16, 17]  # EST = UTC-5, EDT = UTC-4


def test_calendar_holidays():
    assert not cal.is_trading_day("2025-01-09")  # national day of mourning
    assert not cal.is_trading_day("2025-04-18")  # Good Friday
    assert cal.is_trading_day("2025-01-10")
    assert cal.next_trading_day("2025-01-08") == pd.Timestamp("2025-01-10")
