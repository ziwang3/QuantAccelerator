"""FINRA short-volume case study: ingest parsing, the look-ahead measure and the availability rule."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quantaccelerator.ingest.finra import FINRA_DIR, read_file

needs_data = pytest.mark.skipif(not (FINRA_DIR / "shvol.parquet").exists(), reason="run quantaccelerator.ingest.finra first")


def test_read_file_checks_trailer(tmp_path):
    f = tmp_path / "CNMSshvol20240312.txt"
    f.write_text("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                 "20240312|AAA|10|0|40|Q,N\n20240312|BBB|5|1|9|Q\n2\n")
    df, log = read_file(f)
    assert list(df.Symbol) == ["AAA", "BBB"] and log["trailer_ok"] and log["file_date"] == pd.Timestamp("2024-03-12")
    f.write_text(f.read_text().replace("\n2\n", "\n3\n"))
    assert not read_file(f)[1]["trailer_ok"]


def test_measure_separates_timing_and_value_leaks():
    from quantaccelerator.eval.gold_finra import GOLD_RULE, lookahead_stats_finra, reference_signal
    from quantaccelerator.ingest.calendar import trading_days
    days = trading_days()
    d = days[(days >= "2023-12-01") & (days <= "2024-03-29")]
    rng = np.random.default_rng(0)
    sv = pd.DataFrame([(t, s, int(rng.integers(1, 50)), 0, 100, "Q") for t in d for s in ("AAA", "BBB")],
                      columns=["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"])
    ref = reference_signal(["AAA", "BBB"], sv).dropna()
    ref = ref[ref.date >= "2024-01-01"]
    nxt = pd.Series(days[days.searchsorted(ref.date.values, side="right")], index=ref.index)
    clean = ref.assign(asof_date=nxt)
    sameday = ref.assign(asof_date=ref.date)
    shifted = clean.assign(value=clean.value + 0.01)
    m = lambda ev: lookahead_stats_finra(ev, GOLD_RULE, reference_signal(["AAA", "BBB"], sv))
    assert m(clean)["share_before_tradable"] == 0.0
    s = m(sameday)
    assert s["share_used_before_release"] == 1.0 and s["share_value_not_public_at_asof"] == 0.0
    s = m(shifted)
    assert s["share_used_before_release"] == 0.0 and s["share_value_not_public_at_asof"] == 1.0


SNIPPET = r'''
import json
from quantaccelerator.datasets import active
from quantaccelerator.eval.gold_finra import GOLD_RULE
from quantaccelerator.state.schemas import AvailabilityRule as R
p = active()
print(json.dumps({"profile": p.name, "tables": sorted(p.tables),
    "gold": p.rule_by_stratum(GOLD_RULE),
    "sameday": p.rule_agreement(R(source_field="shvol.Date", tradable_at="same_date_open")),
    "calendar_day": p.rule_agreement(R(source_field="shvol.Date", tradable_at="same_date_open", extra_lag_trading_days=1)),
    "wrong_field": p.rule_agreement(R(source_field="vintages.realtime_start", tradable_at="same_date_open"))}))
'''


@needs_data
def test_finra_profile_rules():
    r = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True, text=True,
                       env={**os.environ, "QA_DATASET": "finra"})
    assert r.returncode == 0, r.stderr[-2000:]
    o = json.loads(r.stdout.strip().splitlines()[-1])
    assert o["profile"] == "finra" and o["tables"] == ["file_log", "shvol"]
    assert o["gold"] == {"overall": 1.0, "before_holiday": 1.0, "friday": 1.0, "next_day": 1.0}
    assert o["sameday"] == 0.0 and o["calendar_day"] == 1.0 and o["wrong_field"] == 0.0


SAME_DAY = r'''
import json
from quantaccelerator.agents.checks import check_same_day_use
from quantaccelerator.state.schemas import PITContractDraft
base = {"dataset_id": "x", "rule_explanation": "", "evidence": [], "revision_policy": "", "universe_policy": "",
        "confidence": 1, "fields": [{"field": "shvol.Date", "role": "event_time", "explanation": "", "evidence": []}]}
r = lambda t: check_same_day_use(PITContractDraft.model_validate({**base, "availability_rule": {
    "source_field": "shvol.Date", "tradable_at": t}}))
print(json.dumps([bool(r("next_open_after")), bool(r("same_date_open")), bool(r("next_trading_day_open"))]))
'''


@needs_data
def test_date_only_event_time_cannot_be_used_the_same_day():
    r = subprocess.run([sys.executable, "-c", SAME_DAY], capture_output=True, text=True,
                       env={**os.environ, "QA_DATASET": "finra"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [True, True, False]
