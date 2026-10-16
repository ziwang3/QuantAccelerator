import json

import pytest
from pydantic import BaseModel

from quantaccelerator.agents.auditor import Auditor
from quantaccelerator.agents.base import CHARS_PER_TOKEN, CTX_LIMIT, SUPERSEDED_DRAFT, Agent, AgentFailure, inline_schema
from quantaccelerator.llm.client import Budget, BudgetExceeded, ScriptedClient
from quantaccelerator.paths import insider_dir
from quantaccelerator.state.schemas import PITContractDraft
from quantaccelerator.tools import registry
from quantaccelerator.tools.timing import GOLD_RULE

needs_data = pytest.mark.skipif(not (insider_dir() / "acceptance_index.parquet").exists(), reason="run ingest first")


class Tiny(BaseModel):
    answer: str
    evidence: list[str]


class TinyAgent(Agent):
    name = "tiny"
    tool_names = ["next_tradable"]
    output_model = Tiny


def last_run_id(messages):
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    return [r for r in results if "run_id" in r][-1]["run_id"]


def test_loop_retries_invalid_submission_and_checks_citations(tmp_path):
    registry.set_session("t-loop", tmp_path)
    client = ScriptedClient([
        {"tool_calls": [{"name": "next_tradable", "arguments": {"timestamp": "2025-03-07 16:01",
                                                                "timezone": "America/New_York"}}]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "x"}}]},                   # schema error
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "x", "evidence": ["r0000000000"]}}]},  # bad cite
        lambda m: {"tool_calls": [{"name": "submit", "arguments": {"answer": "Monday", "evidence": [last_run_id(m)]}}]},
    ], log_path=tmp_path / "llm.jsonl")
    with pytest.raises(AgentFailure):  # with max_submit_failures=1 the second bad submission is fatal
        a = TinyAgent()
        a.max_submit_failures = 1
        a.run(client, "when?")
    client.i = 0
    draft, trace = TinyAgent().run(client, "when?")
    assert draft.answer == "Monday" and trace["submit_failures"] == 2
    assert len((tmp_path / "llm.jsonl").read_text().splitlines()) == 3 + 4  # failing pass + passing pass


def test_rejected_drafts_are_compacted_and_errors_recorded(tmp_path):
    registry.set_session("t-compact", tmp_path)
    big = "x" * 5000
    client = ScriptedClient([
        {"tool_calls": [{"name": "submit", "arguments": {"answer": big}}]},                    # schema error
        {"tool_calls": [{"name": "submit", "arguments": {"answer": big, "evidence": ["r0000000000"]}}]},  # bad cite
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "ok", "evidence": []}}]},
    ])
    draft, trace = TinyAgent().run(client, "when?")
    assert draft.answer == "ok" and len(trace["submit_errors"]) == 2
    assert "schema validation failed" in trace["submit_errors"][0] and "r0000000000" in trace["submit_errors"][1]
    args = [tc["function"]["arguments"] for m in trace["messages"] for tc in m.get("tool_calls", [])]
    assert args[0] == SUPERSEDED_DRAFT and args[1] == SUPERSEDED_DRAFT and big not in json.dumps(trace["messages"])


def test_any_error_becomes_agent_failure_with_trace(tmp_path):
    registry.set_session("t-crash", tmp_path)

    def boom(messages):
        raise RuntimeError("maximum context length exceeded")

    client = ScriptedClient([{"tool_calls": [{"name": "submit", "arguments": {"answer": "x"}}]}, boom])
    with pytest.raises(AgentFailure) as ei:
        TinyAgent().run(client, "when?")
    tr = ei.value.trace
    assert "RuntimeError" in str(ei.value) and tr["submit_errors"] and len(tr["messages"]) >= 3


def test_output_cap_fails_cleanly_when_context_is_full(tmp_path):
    registry.set_session("t-ctx", tmp_path)
    client = ScriptedClient([{"content": "never called"}])
    with pytest.raises(AgentFailure, match="context window exhausted"):
        TinyAgent().run(client, "y" * int(CTX_LIMIT * CHARS_PER_TOKEN))
    assert client.i == 0


def test_identical_tool_call_is_not_rerun(tmp_path):
    registry.set_session("t-repeat", tmp_path)
    call = {"name": "next_tradable", "arguments": {"timestamp": "2025-03-07 16:01", "timezone": "America/New_York"}}
    client = ScriptedClient([
        {"tool_calls": [call]},
        {"tool_calls": [call]},
        lambda m: {"tool_calls": [{"name": "submit", "arguments": {"answer": "Monday", "evidence": [last_run_id(m)]}}]},
    ])
    draft, trace = TinyAgent().run(client, "when?")
    assert trace["tool_calls"] == 1 and trace["repeated_calls"] == 1 and len(trace["tool_runs"]) == 1
    assert "Identical to your earlier next_tradable call" in [m for m in trace["messages"] if m["role"] == "tool"][1]["content"]


def test_budget_cap():
    c = ScriptedClient([{"content": "hi"}] * 5, budget=Budget(max_calls=2))
    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "a"}])
    with pytest.raises(BudgetExceeded):
        c.chat([{"role": "user", "content": "a"}])


def test_inline_schema_has_no_refs():
    s = json.dumps(inline_schema(PITContractDraft))
    assert "$ref" not in s and "availability_rule" in s


@needs_data
def test_auditor_offline_clean_pipeline(tmp_path):
    registry.set_session("t-audit", tmp_path, reference_rule=GOLD_RULE)
    path = "notebooks/insider_events_clean.py"

    def measure(m):
        out = json.loads([x for x in m if x["role"] == "tool"][-1]["content"])["result"]["output_path"]
        return {"tool_calls": [{"name": "measure_lookahead", "arguments": {"output_path": out}}]}

    client = ScriptedClient([
        {"tool_calls": [{"name": "read_source", "arguments": {"path": path}},
                        {"name": "run_pipeline", "arguments": {"path": path}}]},
        measure,
        lambda m: {"tool_calls": [{"name": "submit", "arguments": {
            "pipeline": path, "findings": [], "summary": "no look-ahead", "evidence": [last_run_id(m)]}}]},
    ])
    draft, trace = Auditor().run(client, "audit")
    assert draft.findings == [] and trace["tool_calls"] == 3


@needs_data
def test_auditor_rejects_false_no_leak_claim(tmp_path):
    registry.set_session("t-audit2", tmp_path, reference_rule=GOLD_RULE)
    path = "notebooks/insider_events_leak_transdate.py"

    def measure(m):
        out = json.loads([x for x in m if x["role"] == "tool"][-1]["content"])["result"]["output_path"]
        return {"tool_calls": [{"name": "measure_lookahead", "arguments": {"output_path": out}}]}

    no_leak = lambda m: {"tool_calls": [{"name": "submit", "arguments": {
        "pipeline": path, "findings": [], "summary": "fine", "evidence": [last_run_id(m)]}}]}
    client = ScriptedClient([{"tool_calls": [{"name": "run_pipeline", "arguments": {"path": path}}]}, measure,
                             no_leak, no_leak, no_leak])
    with pytest.raises(AgentFailure, match="no-leak conclusion"):
        Auditor().run(client, "audit")


@needs_data
def test_checks_catch_phantom_columns_and_missing_codes():
    from quantaccelerator.agents.checks import check_column_ref, check_dataset_card
    from quantaccelerator.state.schemas import DatasetCardDraft
    assert check_column_ref("nonderiv_trans.TRANS_DATE") is None
    assert "no column" in check_column_ref("nonderiv_trans.transactionDate")
    card = DatasetCardDraft.model_validate({
        "dataset_id": "x", "tables": [], "entity_id_fields": ["ACCESSION_NUMBER"], "time_fields": [],
        "fields": [{"table": "nonderiv_trans", "name": "TRANS_CODE", "dtype": "str", "meaning": "code",
                    "value_meanings": {"P": "purchase", "S": "sale"}, "evidence": [], "confidence": 1}]})
    errs = check_dataset_card(card)
    assert len(errs) == 2 and "missing" in errs[0] and "'F'" in errs[0] and "cites no documentation" in errs[1]


def test_truncated_output_is_replaced_and_retried(tmp_path):
    registry.set_session("t-trunc", tmp_path)
    client = ScriptedClient([
        {"content": '{"answer": "Mon', "finish_reason": "length"},
        {"tool_calls": [{"name": "next_tradable", "arguments": {"timestamp": "2025-03-07 16:01",
                                                                "timezone": "America/New_York"}}]},
        lambda m: {"tool_calls": [{"name": "submit", "arguments": {"answer": "Monday", "evidence": [last_run_id(m)]}}]},
    ])
    draft, trace = TinyAgent().run(client, "when?")
    assert trace["truncated"] == 1 and draft.answer == "Monday"
    assert not any('"Mon' in str(m.get("content")) for m in trace["messages"])


def test_parse_hermes_fallback_skips_broken_blocks():
    from quantaccelerator.llm.client import parse_hermes_tool_calls
    text = ('<tool_call>\n{"name": "profile_time", "arguments": {"table": "submission", "columns": ["FILING_DATE"]}}\n'
            '</tool_call>\n<tool_call>\n{"name": "submit", "arguments": {"x": [1, 2}\n</tool_call>')
    calls = parse_hermes_tool_calls(text)
    assert [c.name for c in calls] == ["profile_time"]
    assert json.loads(calls[0].arguments)["columns"] == ["FILING_DATE"]


def test_grounding_rejects_guessed_code_meanings():
    from quantaccelerator.agents.checks import check_grounding
    from quantaccelerator.state.schemas import FieldCard
    mk = lambda vm, ev: FieldCard(table="nonderiv_trans", name="TRANS_CODE", dtype="str", meaning="code",
                                  value_meanings=vm, evidence=ev, confidence=1)
    assert check_grounding(mk({"M": "Exercise or conversion of derivative security"}, ["sec_insider_readme p5"])) == []
    assert "not supported" in check_grounding(mk({"M": "Merger"}, ["sec_insider_readme p5"]))[0]
    assert "cites no documentation" in check_grounding(mk({"M": "Merger"}, ["r0123456789"]))[0]


@needs_data
def test_pit_evidence_must_agree_with_timezone_check(tmp_path):
    from quantaccelerator.agents.checks import check_pit_evidence
    registry.set_session("t-tz", tmp_path)
    tz = registry.TOOLS["timezone_check"].fn()
    base = {"dataset_id": "x", "fields": [], "rule_explanation": "", "revision_policy": "", "universe_policy": "",
            "confidence": 1, "evidence": [tz["run_id"]]}
    ok = PITContractDraft.model_validate({**base, "availability_rule": GOLD_RULE.model_dump()})
    bad = PITContractDraft.model_validate({**base, "availability_rule": {**GOLD_RULE.model_dump(),
                                                                         "source_timezone": "America/New_York"}})
    assert check_pit_evidence(ok) == [] and "contradicts" in check_pit_evidence(bad)[0]
    nocite = PITContractDraft.model_validate({**base, "evidence": [], "availability_rule": GOLD_RULE.model_dump()})
    assert "must cite a timezone_check" in check_pit_evidence(nocite)[0]


def test_repeated_call_with_list_result_does_not_crash(tmp_path):
    """read_doc returns a list; repeating it once crashed the loop ('list' object has no attribute 'get')."""
    registry.set_session("t-repeat-list", tmp_path)

    class DocAgent(Agent):
        name = "doc"
        tool_names = ["read_doc"]
        output_model = Tiny

    call = {"name": "read_doc", "arguments": {"query": "TRANS_CODE", "doc": "sec_insider_readme"}}
    client = ScriptedClient([{"tool_calls": [call]}, {"tool_calls": [call]},
                             lambda m: {"tool_calls": [{"name": "submit", "arguments": {
                                 "answer": "codes", "evidence": [last_run_id(m)]}}]}])
    draft, trace = DocAgent().run(client, "what codes?")
    assert trace["repeated_calls"] == 1 and draft.answer == "codes"


@needs_data
def test_prototype_build_drops_verification_fixes(monkeypatch):
    """QA_BUILD=prototype: no cross-reference following in read_doc and no grounding or PIT evidence checks."""
    from quantaccelerator import build
    from quantaccelerator.agents.checks import check_dataset_card, check_pit_contract
    from quantaccelerator.state.schemas import DatasetCardDraft
    from quantaccelerator.tools.docs import search
    card = DatasetCardDraft.model_validate({
        "dataset_id": "x", "tables": [], "entity_id_fields": ["ACCESSION_NUMBER"], "time_fields": [],
        "fields": [{"table": "nonderiv_trans", "name": "TRANS_CODE", "dtype": "str", "meaning": "code",
                    "value_meanings": {"P": "purchase", "S": "sale"}, "evidence": [], "confidence": 1}]})
    nocite = PITContractDraft.model_validate({"dataset_id": "x", "fields": [], "rule_explanation": "",
                                              "revision_policy": "", "universe_policy": "", "confidence": 1,
                                              "evidence": [], "availability_rule": GOLD_RULE.model_dump()})
    assert len(check_dataset_card(card)) == 2 and check_pit_contract(nocite)
    monkeypatch.setattr(build, "BUILD", "prototype")
    assert not any(h.get("followed_from") for h in search("TRANS_CODE", "sec_insider_readme", 2))
    errs = check_dataset_card(card)
    assert len(errs) == 1 and "missing" in errs[0]    # coverage check stays; grounding is off
    assert check_pit_contract(nocite) == []


def test_a_short_token_estimate_is_corrected_from_the_server_error(tmp_path):
    from quantaccelerator.agents import base
    registry.set_session("t-ctx", tmp_path)
    class Tight(ScriptedClient):
        calls = []
        def chat(self, messages, tools, agent=None, max_tokens=None):
            self.calls.append(max_tokens)
            if len(self.calls) == 1:
                raise RuntimeError(f"maximum context length is {base.CTX_LIMIT} tokens. However, you requested "
                                   f"{base.CTX_LIMIT + 10} tokens ({base.CTX_LIMIT - 1500} in the messages, 1510 in the completion)")
            return super().chat(messages, tools, agent=agent, max_tokens=max_tokens)
    client = Tight([{"tool_calls": [{"name": "submit", "arguments": {"answer": "Monday", "evidence": []}}]}])
    draft, trace = TinyAgent().run(client, "when?")
    assert trace["context_retries"] == 1 and client.calls[1] == 1500 - 64

def test_unknown_citation_lists_the_agents_own_runs(tmp_path):
    registry.set_session("t-cite", tmp_path)
    client = ScriptedClient([
        {"tool_calls": [{"name": "next_tradable", "arguments": {"timestamp": "2025-03-07 16:01", "timezone": "America/New_York"}}]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "Monday", "evidence": ["r0123456789"]}}]},
        lambda m: {"tool_calls": [{"name": "submit", "arguments": {"answer": "Monday", "evidence": [last_run_id(m)]}}]},
    ])
    draft, trace = TinyAgent().run(client, "when?")
    assert "Your runs so far, by tool: next_tradable: r" in trace["submit_errors"][0]
