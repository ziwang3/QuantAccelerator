"""PITResearchView: refuses without a signed contract; snapshots never use records before their decision time."""
import pandas as pd
import pytest

from quantaccelerator.eval.gold_finra import GOLD_RULE
from quantaccelerator.ingest.calendar import trading_days
from quantaccelerator.state.schemas import GateDecision, PITContract, Provenance
from quantaccelerator.tools.finra import apply_finra_rule
from quantaccelerator.tools.pit_view import UnsignedContract, asof, signed_rule


def contract(g):
    return PITContract(dataset_id="x", fields=[], availability_rule=GOLD_RULE, rule_explanation="", evidence=[],
                       revision_policy="", universe_policy="", confidence=1, signed_off=g,
                       provenance=Provenance(producer_agent="t"))


def test_view_needs_a_signed_contract():
    with pytest.raises(UnsignedContract):
        signed_rule(contract(None))
    with pytest.raises(UnsignedContract):
        signed_rule(contract(GateDecision(decided_by="human", approved=False)))
    assert signed_rule(contract(GateDecision(decided_by="human", approved=True))) == GOLD_RULE


def test_asof_uses_only_published_records():
    days = trading_days()
    d = days[(days >= "2024-03-25") & (days <= "2024-04-05")]
    view = pd.DataFrame({"entity": "AAA", "source_time": d, "value": range(len(d))})
    view["decision_time"] = apply_finra_rule(GOLD_RULE, view.source_time).values
    snap = asof(view, ["2024-03-28", "2024-04-01"], ["AAA"])
    # Thursday 28 Mar sees Wednesday's file; Monday 1 Apr (after Good Friday) sees Thursday's file, 0 days old
    assert list(snap.source_time.dt.strftime("%m-%d")) == ["03-27", "03-28"] and list(snap.feature_age) == [0, 0]
