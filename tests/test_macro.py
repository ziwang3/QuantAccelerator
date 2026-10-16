"""Revision Audit (ALFRED) profile: runs in a subprocess because the active dataset profile is fixed at import."""
import json
import os
import subprocess
import sys

import pytest

from quantaccelerator.ingest.alfred import ALFRED_DIR

needs_data = pytest.mark.skipif(not (ALFRED_DIR / "vintages.parquet").exists(), reason="run quantaccelerator.ingest.alfred first")

SNIPPET = r'''
import json, tempfile
from pathlib import Path
import pandas as pd
from quantaccelerator.datasets import active
from quantaccelerator.eval.gold_macro import GOLD_RULE, lookahead_stats_macro, releases
from quantaccelerator.state.schemas import AvailabilityRule as R
from quantaccelerator.tools import registry
registry.DB = Path(tempfile.mkdtemp()) / "r.sqlite"
registry.set_session("t-macro", Path(tempfile.mkdtemp()))
p = active()
rel = releases()
clean = rel.assign(asof_date=pd.to_datetime(rel.release_date), value=rel.first_value)[["series_id", "date", "asof_date", "value"]]
# roll release dates to trading days as the pipeline does
from quantaccelerator.ingest.calendar import roll_forward
clean["asof_date"] = roll_forward(clean.asof_date).values
latest = clean.assign(value=rel.latest_value.values)
early = clean.assign(asof_date=rel.date.values)
out = {"profile": p.name, "tables": sorted(p.tables),
       "clean": lookahead_stats_macro(clean)["share_before_tradable"],
       "latest": lookahead_stats_macro(latest)["share_before_tradable"],
       "latest_under_latest_rule": lookahead_stats_macro(latest, R(source_field="vintages.realtime_start",
           tradable_at="same_date_open", value_vintage="latest"))["share_before_tradable"],
       "early": lookahead_stats_macro(early)["share_before_tradable"],
       "agree_gold": p.rule_agreement(GOLD_RULE),
       "agree_latest": p.rule_agreement(R(source_field="vintages.realtime_start", tradable_at="same_date_open",
                                          value_vintage="latest")),
       "tool_ok": registry.TOOLS["revision_stats"].fn(series_id="GDPC1")["ok"],
       "read_doc_docs": sorted({h["doc"] for h in registry.TOOLS["read_doc"].fn(query="realtime_start")["result"]})}
print(json.dumps(out))
'''


@needs_data
def test_macro_profile_measure_and_rules():
    r = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True, text=True,
                       env={**os.environ, "QA_DATASET": "alfred"})
    assert r.returncode == 0, r.stderr[-2000:]
    o = json.loads(r.stdout.strip().splitlines()[-1])
    assert o["profile"] == "alfred" and o["tables"] == ["series_meta", "vintages"]
    assert o["clean"] == 0.0 and o["early"] == 1.0
    assert 0.5 < o["latest"] < 1.0                  # most values were revised after first release
    assert o["latest_under_latest_rule"] == 0.0     # a wrong contract hides the revision leak: G1 must catch it
    assert o["agree_gold"] == 1.0 and o["agree_latest"] < 0.5
    assert o["tool_ok"] and any(d.startswith("fred_") for d in o["read_doc_docs"])  # FRED docs define the field
