"""The Data Research Agent's profiling pass, offline: scripted LLM, real tools on the Form 4 tables (dataset-neutral),
findings recorded item by item (the lab notebook), the evidence checks, and the Data Brief."""
import json

import pytest

from quantaccelerator.agents.base import AgentFailure
from quantaccelerator.agents.brief import build
from quantaccelerator.agents.orchestrator import _finish, _start, understand
from quantaccelerator.agents.profiler import DataResearcher
from quantaccelerator.llm.client import ScriptedClient
from quantaccelerator.paths import insider_dir
from quantaccelerator.tools import registry

needs_data = pytest.mark.skipif(not (insider_dir() / "acceptance_index.parquet").exists(), reason="run ingest first")


def rid(messages, tool):
    """run_id of the latest result of `tool` in the conversation."""
    names = {tc["id"]: tc["function"]["name"] for m in messages if m["role"] == "assistant"
             for tc in m.get("tool_calls") or []}
    runs = [json.loads(m["content"])["run_id"] for m in messages
            if m["role"] == "tool" and names.get(m.get("tool_call_id")) == tool]
    return runs[-1]


def tool_results(messages, name):
    names = {tc["id"]: tc["function"]["name"] for m in messages if m["role"] == "assistant"
             for tc in m.get("tool_calls") or []}
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool" and names.get(m.get("tool_call_id")) == name]


TOOLS = [{"tool_calls": [   # two turns of two calls: the research pass allows two tool calls per turn
    {"name": "research_breadth", "arguments": {"table": "submission", "entity_field": "ISSUERCIK", "time_field": "FILING_DATE"}},
    {"name": "coverage_over_time", "arguments": {"table": "submission", "time_field": "FILING_DATE", "freq": "W"}}]},
    {"tool_calls": [
    {"name": "distribution", "arguments": {"table": "nonderiv_trans", "field": "TRANS_SHARES"}},
    {"name": "preview_cleaning", "arguments": {"step": {"op": "flag_rows", "table": "nonderiv_trans", "flag_name": "p",
                                                       "where": [{"column": "TRANS_CODE", "op": "==", "value": "P"}]}}}]}]


def record(m, claim_date="2025-02-10"):
    """One turn recording three observations, four capability ratings, two cleaning steps and a preprocessing note
    (recording tools are not paced)."""
    cap = lambda d: {"name": "rate_capability", "arguments": {"dimension": d, "rating": "weak",
                                                             "evidence": [rid(m, "research_breadth")], "note": "1 quarter"}}
    obs = lambda c: {"name": "record_observation", "arguments": {"claim": c, "evidence": [rid(m, "coverage_over_time")],
                                                                  "severity": "info"}}
    return {"tool_calls": [obs(f"filings dip around {claim_date}"), obs("coverage is stable"), obs("few issuers file a lot"),
                           *[cap(d) for d in ("history_depth", "cross_sectional_breadth", "update_frequency",
                                              "coverage_stability")],
                           {"name": "propose_cleaning_step", "arguments": {
                               "op": "flag_rows", "table": "nonderiv_trans", "flag_name": "is_purchase",
                               "where": [{"column": "TRANS_CODE", "op": "==", "value": "P"}], "reason": "the events"}},
                           {"name": "propose_cleaning_step", "arguments": {
                               "op": "drop_rows", "table": "nonderiv_trans", "reason": "nothing",
                               "where": [{"column": "TRANS_CODE", "op": "==", "value": "no-such-code"}]}},
                           {"name": "note_preprocessing", "arguments": {
                               "field": "nonderiv_trans.TRANS_SHARES", "observations": ["right-skewed"],
                               "candidate_processing": ["log1p"], "risks": [], "evidence": [rid(m, "distribution")]}}]}


WRAP = {"tool_calls": [{"name": "submit", "arguments": {"suitable_for": ["event study"], "weak_for": ["long horizons"],
                                                         "major_limits": ["1 quarter"], "open_questions": ["q?"]}}]}


@needs_data
def test_research_pass_records_findings(tmp_path, monkeypatch):
    from quantaccelerator.agents import checks
    from quantaccelerator.tools.findings import assemble
    monkeypatch.setattr(checks, "session_flags", lambda agent="data_research": [])  # flags: tested separately
    registry.set_session("t-research", tmp_path)
    bad = lambda m: {"tool_calls": [{"name": "record_observation", "arguments": {
        "claim": "filings dip around 2019-06-01", "evidence": [rid(m, "coverage_over_time")]}}]}
    draft, trace = DataResearcher().run(ScriptedClient(TOOLS[:1] + [WRAP] + TOOLS[1:] + [bad, record, WRAP]), "research")
    assert "rate these capability dimensions first" in trace["submit_errors"][0]   # nothing recorded yet
    msgs = trace["messages"]
    obs = tool_results(msgs, "record_observation")
    assert "no cited tool result has a date near it" in obs[0]["result"]["error"]   # checked on the spot
    assert obs[1]["result"]["figure_run_id"] == rid(msgs, "coverage_over_time")      # figure attached from evidence
    steps = tool_results(msgs, "propose_cleaning_step")
    assert steps[0]["result"]["rows_affected"] > 0 and "changes no rows" in steps[1]["result"]["error"]
    full = assemble(draft)
    assert len(full.observations) == 3 and full.cleaning.steps[0].evidence == [steps[0]["run_id"]]
    assert full.capability.history_depth.rating == "weak" and len(list((tmp_path / "figures").glob("*.plain.png"))) == 3


@needs_data
def test_understand_writes_card_with_research_and_brief(tmp_path, monkeypatch):
    from quantaccelerator.agents import checks
    monkeypatch.setattr(checks, "session_flags", lambda agent="data_research": [])  # flags: tested separately
    card_json = {"dataset_id": "x", "tables": [], "entity_id_fields": [], "time_fields": [], "fields": []}
    client = ScriptedClient([{"tool_calls": [{"name": "submit", "arguments": card_json}]}] + TOOLS + [record, WRAP])
    run_dir, client, meta, timed = _start("t-understand", tmp_path, client, mode="understand")
    card = understand(run_dir, client, timed, research=True, overview=False)   # overview: tested separately
    meta["status"] = "ok"
    _finish(run_dir, client, meta)
    b = build(run_dir)
    assert card.research is not None and card.research_provenance.producer_agent == "data_research"
    assert [s["tool"] for s in b["steps"]][:4] == ["research_breadth", "coverage_over_time", "distribution",
                                                   "preview_cleaning"]
    assert len(b["observations"]) == 3 and b["observations"][0]["figure"].startswith("figures/")
    assert b["cleaning"]["steps"][0]["effect"]["rows_flagged"] > 0 and b["open_questions"] == ["q?"]
    html = (run_dir / "brief" / "brief.html").read_text()
    assert "What the data can support" in html and "data:image/png;base64" in html
    assert b["table_preview"] and all(len(t["head"]) == 5 and t["columns"] for t in b["table_preview"])
    assert "first rows" in html


@needs_data
def test_overview_must_measure_entities_and_cite_docs(tmp_path):
    from quantaccelerator.agents.checks import check_overview
    from quantaccelerator.state.schemas import DatasetOverview
    registry.set_session("t-overview", tmp_path)
    pt = registry.TOOLS["profile_table"].fn(table="submission")["run_id"]
    ip = registry.TOOLS["identifier_patterns"].fn(table="submission", field="ISSUERTRADINGSYMBOL")["run_id"]
    base = dict(summary="Insider filings", publisher="SEC", purpose="disclosure", market="US equities",
                values=["share counts and prices of insider trades"], frequency="per filing", period="2025 Q1",
                typical_uses=["insider-trading signals"])
    o = DatasetOverview(**base, entities=[{"type": "issuers", "share": "all", "evidence": []}], evidence=[pt])
    errs = " ".join(check_overview(o))
    assert "cite the documentation" in errs and "measure which kinds of entities" not in errs  # profile_table counts
    o = DatasetOverview(**base, entities=[{"type": "issuers: operating companies, not ETFs or funds", "share": "all", "evidence": [ip]}],
                        evidence=["sec_insider_readme p3"])
    assert check_overview(o) == []
    assert "fill in the dataset overview" in check_overview(None)[0]


@needs_data
def test_research_pass_is_paced(tmp_path):
    registry.set_session("t-paced", tmp_path)
    three = {"tool_calls": [{"name": "key_check", "arguments": {"table": "submission", "key_fields": [k]}}
                            for k in ("ACCESSION_NUMBER", "ISSUERCIK", "FILING_DATE")]}
    with pytest.raises(AgentFailure) as e:  # the script ends after one turn
        DataResearcher().run(ScriptedClient([three]), "research")
    assert e.value.trace["tool_calls"] == 2 and e.value.trace["paced_calls"] == 1
    from quantaccelerator.tools.explore import key_stats
    import pandas as pd
    k = key_stats(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}), ["a", "b"])
    assert k["redundant_columns"] == ["b"] and "not needed" in k["reading"]


def test_flags_must_be_addressed(tmp_path, monkeypatch):
    from quantaccelerator.agents import checks
    from quantaccelerator.state.schemas import DataResearchDraft
    flags = [("r1", {"id": "absent_rows", "fact": "absent 12%", "terms": [["absent", "no row"]]}),
             ("r2", {"id": "redundant_key", "fact": "Market not needed", "terms": [["market"], ["not needed"]]})]
    monkeypatch.setattr(checks, "session_flags", lambda agent="data_research": flags)
    cap = {"rating": "weak", "evidence": [], "note": ""}
    base = {"capability": {k: cap for k in ("history_depth", "cross_sectional_breadth", "update_frequency",
                                            "coverage_stability")} | {"suitable_for": [], "weak_for": [],
                                                                      "major_limits": []},
            "preprocessing": [], "observations": [], "cleaning": {"steps": []}}
    d = DataResearchDraft.model_validate({**base, "warnings": ["nothing to see"]})
    assert "absent 12%" in checks.unaddressed_flags(d)[0] and "Market not needed" in checks.unaddressed_flags(d)[0]
    d = DataResearchDraft.model_validate({**base, "warnings": ["symbols are absent on some days"],
                                          "key_corrections": {"shvol": ["Date", "Symbol"]}})
    assert checks.unaddressed_flags(d) == []


def test_wrapped_submission_is_unwrapped(tmp_path):
    from tests.test_agents import Tiny, TinyAgent
    registry.set_session("t-wrap", tmp_path)
    draft, trace = TinyAgent().run(ScriptedClient([{"tool_calls": [{"name": "submit", "arguments": {
        "tiny_answer": {"answer": "x", "evidence": []}}}]}]), "q")
    assert draft.answer == "x" and trace["submit_failures"] == 0


@needs_data
def test_research_checklist_from_the_card(tmp_path):
    from quantaccelerator.agents.checks import missing_analyses
    from quantaccelerator.state.schemas import DatasetCardDraft
    registry.set_session("t-checklist", tmp_path)
    card = DatasetCardDraft.model_validate({
        "dataset_id": "x", "tables": [{"table": "nonderiv_trans", "observation_unit": "a line",
                                       "primary_key": ["ACCESSION_NUMBER"], "n_rows_run_id": "r0000000000"}],
        "entity_id_fields": ["nonderiv_trans.ACCESSION_NUMBER"], "time_fields": ["nonderiv_trans.TRANS_DATE"],
        "fields": [{"table": "nonderiv_trans", "name": n, "dtype": "float", "meaning": "m", "evidence": [],
                    "confidence": 1, "kind": "flow", "unit": "shares"} for n in ("TRANS_SHARES", "SHRS_OWND_FOLWNG_TRANS")]
        + [{"table": "nonderiv_trans", "name": "TRANS_CODE", "dtype": "str", "meaning": "m", "evidence": [],
            "confidence": 1, "kind": "code"}]})
    todo = missing_analyses(card)
    assert any("ratio" in t for t in todo) and any("category_mix" in t for t in todo) and len(todo) == 8
    registry.CTX.agent = "data_research"
    registry.TOOLS["distribution"].fn(table="nonderiv_trans", field="TRANS_SHARES", denominator="SHRS_OWND_FOLWNG_TRANS")
    todo2 = missing_analyses(card)
    assert not any("ratio" in t for t in todo2) and len(todo2) == 7


def test_data_errors_need_a_remedy(monkeypatch):
    from quantaccelerator.agents import checks
    from quantaccelerator.state.schemas import CleaningRecipe, CleaningStep
    flags = [("r1", {"id": "part_exceeds_whole", "fact": "ShortVolume exceeds TotalVolume in 9 rows",
                     "terms": [["shortvolume"]], "remedy": {"op": ["flag_rows", "drop_rows"], "where": {
                         "column": "ShortVolume", "op": ">col", "value": "TotalVolume"}}})]
    monkeypatch.setattr(checks, "session_flags", lambda agent="data_research": flags)
    class D:  # the parts of a draft the rule reads
        cleaning = CleaningRecipe()
    assert "is a data error" in checks.unremedied(D, "shortvolume exceeds totalvolume")[0]
    D.cleaning = CleaningRecipe(steps=[CleaningStep(op="flag_rows", table="shvol", flag_name="bad", reason="",
                                                    evidence=[], where=[{"column": "ShortVolume", "op": ">col",
                                                                         "value": "TotalVolume"}])])
    assert checks.unremedied(D, "") == []


def test_column_comparison_written_naturally():
    import pandas as pd
    from quantaccelerator.state.schemas import Predicate
    from quantaccelerator.tools.clean import _mask
    df = pd.DataFrame({"a": [1, 5, 3], "b": [2, 2, 3]})
    assert list(_mask(df, [Predicate(column="a", op=">", value="b")])) == [False, True, False]
    with pytest.raises(ValueError, match="neither a number nor a column"):
        _mask(df, [Predicate(column="a", op=">", value="bb")])


def test_long_prose_is_trimmed_with_a_reminder(tmp_path):
    from tests.test_agents import TinyAgent
    registry.set_session("t-prose", tmp_path)
    a = TinyAgent()
    a.max_prose_chars, a.prose_reminder = 20, "be brief"
    draft, trace = a.run(ScriptedClient([
        {"content": "x" * 500, "tool_calls": [{"name": "next_tradable", "arguments": {
            "timestamp": "2025-03-07 16:01", "timezone": "America/New_York"}}]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "ok", "evidence": []}}]}]), "q")
    m = trace["messages"]
    assert trace["prose_trimmed"] == 1 and m[2]["content"].endswith("[...]") and len(m[2]["content"]) < 40
    assert m[3]["role"] == "tool" and m[4] == {"role": "user", "content": "be brief"}


def test_a_flag_is_reported_by_one_finding_not_by_scattered_words(monkeypatch):
    from quantaccelerator.agents import checks
    from quantaccelerator.state.schemas import DataObservation, DataResearchDraft
    flag = {"id": "zero_spike_ShortVolume", "fact": "ShortVolume is zero in 20% of rows from 2022-09-01",
            "terms": [["shortvolume"], ["zero"], ["2022-09", "2022"]]}
    monkeypatch.setattr(checks, "session_flags", lambda agent="data_research": [("r0000000001", flag)])
    obs = lambda c: DataObservation(claim=c, evidence=["r0000000001"])
    draft = lambda *claims: DataResearchDraft.model_construct(observations=[obs(c) for c in claims], warnings=[],
                                                              open_questions=[], preprocessing=[], key_corrections={},
                                                              cleaning=None)
    monkeypatch.setattr(checks, "unremedied", lambda d, t: [])
    scattered = draft("ShortVolume exceeds TotalVolume in some rows", "zeros are common in ShortExemptVolume",
                      "Market value B first appears in November 2022")
    assert checks.unaddressed_flags(scattered)
    assert not checks.unaddressed_flags(draft("ShortVolume is zero for a fifth of symbols in September 2022"))


@needs_data
def test_overview_is_its_own_step(tmp_path):
    from quantaccelerator.agents.orchestrator import _start, understand
    card_json = {"dataset_id": "x", "tables": [], "entity_id_fields": [], "time_fields": [], "fields": []}
    ov = lambda m: {"tool_calls": [{"name": "submit", "arguments": {
        "summary": "Insider filings", "publisher": "SEC", "purpose": "disclosure", "market": "US equities",
        "values": ["share counts and prices of insider trades"], "frequency": "per filing", "period": "2025 Q1",
        "typical_uses": ["insider signals"], "evidence": ["sec_insider_readme p3", rid(m, "identifier_patterns")],
        "entities": [{"type": "issuers: operating companies, not ETFs or funds", "share": "all",
                      "evidence": [rid(m, "identifier_patterns")]}]}}]}
    client = ScriptedClient([{"tool_calls": [{"name": "submit", "arguments": card_json}]},
                             {"tool_calls": [{"name": "identifier_patterns",
                                              "arguments": {"table": "submission", "field": "ISSUERTRADINGSYMBOL"}}]}, ov])
    run_dir, client, meta, timed = _start("t-ov", tmp_path, client, mode="understand")
    card = understand(run_dir, client, timed, research=False, overview=True)
    assert card.overview is not None and card.overview.publisher == "SEC"
    assert "overview" in meta["agents"] and "profiler" in meta["agents"]