import pandas as pd
import pytest

import quantaccelerator.tools  # noqa: F401
from quantaccelerator.state.schemas import AvailabilityRule
from quantaccelerator.tools import registry
from quantaccelerator.tools.pipeline import apply_edits
from quantaccelerator.tools.pit import pit_join
from quantaccelerator.tools.timing import GOLD_RULE, first_open_after, parse_to_et


def fot(ts: str, tz: str = "America/New_York") -> str:
    return str(first_open_after(parse_to_et(pd.Series([ts]), tz)).iloc[0].date())


@pytest.mark.parametrize("ts,expected", [
    ("2025-03-07 15:59", "2025-03-10"),  # Friday intraday -> Monday open (same-day open already passed)
    ("2025-03-07 16:01", "2025-03-10"),  # Friday after close -> Monday open
    ("2025-03-06 08:00", "2025-03-06"),  # pre-open on a trading day -> same-day open
    ("2025-03-06 09:30", "2025-03-07"),  # exactly at the open -> not strictly after -> next day
    ("2025-03-08 11:00", "2025-03-10"),  # Saturday -> Monday
    ("2025-01-08 17:00", "2025-01-10"),  # 2025-01-09 market closed (national day of mourning)
    ("2025-01-09 07:00", "2025-01-10"),  # acceptance on the closure day itself
    ("2025-04-17 18:00", "2025-04-21"),  # Thursday before Good Friday
])
def test_first_open_after(ts, expected):
    assert fot(ts) == expected


def test_utc_conversion_across_dst():
    assert fot("2025-03-07T14:00:00.000Z", "UTC") == "2025-03-07"  # 09:00 EST, pre-open
    assert fot("2025-03-10T14:00:00.000Z", "UTC") == "2025-03-11"  # 10:00 EDT, after the open
    assert fot("2025-03-10T13:00:00.000Z", "UTC") == "2025-03-10"  # 09:00 EDT, pre-open


def test_pit_join_refuses_without_availability():
    left = pd.DataFrame({"k": [1], "asof": pd.to_datetime(["2025-01-10"])})
    right = pd.DataFrame({"k": [1], "event": pd.to_datetime(["2025-01-02"]), "x": [5]})
    with pytest.raises(ValueError, match="explicit availability_col"):
        pit_join(left, right, "k", None, "asof")
    with pytest.raises(ValueError, match="not in right"):
        pit_join(left, right, "k", "available_at", "asof")
    right["available_at"] = pd.to_datetime(["2025-01-06"])
    assert pit_join(left, right, "k", "available_at", "asof").x.tolist() == [5]
    right["available_at"] = pd.to_datetime(["2025-01-13"])  # not yet public on the as-of date
    assert pit_join(left, right, "k", "available_at", "asof").x.isna().all()


def test_tool_calls_are_registered(tmp_path):
    registry.set_session("pytest-session", tmp_path)
    before = registry.ledger_count()
    r = registry.TOOLS["next_tradable"].fn(timestamp="2025-03-07 16:01", timezone="America/New_York")
    assert r["ok"] and r["result"]["first_tradable_open"].startswith("2025-03-10")
    assert registry.ledger_count() == before + 1 and registry.run_exists(r["run_id"], "pytest-session")
    bad = registry.TOOLS["lookup_acceptance"].fn()  # missing argument -> error captured, still registered
    assert not bad["ok"] and registry.run_exists(bad["run_id"])
    registry.set_session(None)


def test_gold_rule_schema_roundtrip():
    assert AvailabilityRule.model_validate(GOLD_RULE.model_dump()) == GOLD_RULE


def test_apply_edits():
    assert apply_edits("a\nb\nc", [{"line": 2, "new_code": "B"}]) == "a\nB\nc\n"
    with pytest.raises(ValueError):
        apply_edits("a", [{"line": 3, "new_code": "x"}])
    # the replaced line's indentation wins over the model's (a model-supplied indent can cause an IndentationError loop)
    assert apply_edits("x = 1\ny = 2", [{"line": 2, "new_code": "    y = 3"}]) == "x = 1\ny = 3\n"
    assert apply_edits("def f():\n    return 1", [{"line": 2, "new_code": "return 2"}]) == "def f():\n    return 2\n"
    assert apply_edits("if a:\n    b", [{"line": 2, "new_code": "  if c:\n      d"}]) == "if a:\n    if c:\n        d\n"


def test_grounding_checks_flags_and_code_list_hint():
    from types import SimpleNamespace

    from quantaccelerator.agents.checks import check_grounding, code_list_pages, is_flag
    assert is_flag(["0", "false", "1", "true", None]) and not is_flag(["A", "D"])
    assert "sec_insider_readme p5" in code_list_pages(["F", "A", "S", "M", "P", "J"])
    guessed = SimpleNamespace(table="nonderiv_trans", name="TRANS_CODE", evidence=["sec_insider_readme p5"],
                              value_meanings={"P": "Open market or private purchase", "M": "Merger"})
    errs = check_grounding(guessed, ["P", "M"])
    assert errs and "M='Merger'" in errs[0] and "P=" not in errs[0]
    ok = SimpleNamespace(table="nonderiv_trans", name="TRANS_ACQUIRED_DISP_CD", evidence=["sec_insider_readme p3"],
                         value_meanings={"A": "Acquisition of securities", "D": "Disposition of securities"})
    assert check_grounding(ok, ["A", "D"]) == []


def test_read_doc_finds_code_list_by_section_name():
    from quantaccelerator.tools.docs import search
    hits = search("Appendix 6.2 Trans Code List", "sec_insider_readme", 2)
    assert hits[0]["page"] == 5 and hits[0]["section"] == "6.2 Trans Code List"
    assert "P Open market or private purchase" in hits[0]["text"] and "S Open market or private sale" in hits[0]["text"]
    assert all("ALABAMA" not in h["text"] for h in hits)  # 6.3 country codes are a separate section


def test_read_doc_follows_appendix_cross_references():
    from quantaccelerator.tools.docs import search
    hits = search("TRANS_CODE", "sec_insider_readme", 2)
    followed = [h for h in hits if h.get("followed_from")]
    assert followed and followed[0]["section"] == "6.2 Trans Code List"
    assert "P Open market or private purchase" in followed[0]["text"]


def test_grounding_names_the_page_when_meanings_are_right_but_miscited():
    from types import SimpleNamespace

    from quantaccelerator.agents.checks import check_grounding
    f = SimpleNamespace(table="nonderiv_trans", name="TRANS_CODE", evidence=["sec_insider_readme p3"],
                        value_meanings={"A": "Grant, award or other acquisition pursuant to Rule 16b-3(d)",
                                        "P": "Open market or private purchase of non-derivative or derivative security"})
    errs = check_grounding(f, ["A", "P"])
    assert errs and "match the wording of sec_insider_readme p5" in errs[0]
    f.evidence = ["sec_insider_readme p5"]
    assert check_grounding(f, ["A", "P"]) == []


def test_doc_refs_accept_common_citation_variants():
    from quantaccelerator.agents.checks import doc_refs
    assert doc_refs("sec_insider_readme p5") == {("sec_insider_readme", 5)}
    assert doc_refs("<doc> p5 (sec_insider_readme)") == {("sec_insider_readme", 5)}
    assert doc_refs("sec_insider_readme, page 5") == {("sec_insider_readme", 5)}
    assert doc_refs("r0123456789") == set()

