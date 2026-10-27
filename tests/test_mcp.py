"""MCP: the server serves every tool that can run out of process, and an agent run through it writes the same ledger
rows (attribution included) as the same run in process."""
import json

import pytest

from quantaccelerator import mcp_server
from quantaccelerator.tools import registry
from test_agent_loop_golden import CALL, TinyAgent, _norm, good_submit

from quantaccelerator.llm.client import ScriptedClient

LEDGER = ["session_id", "agent", "tool", "args_json", "args_hash", "input_sha256s", "code_version", "ok", "error",
          "node_exec"]


def test_server_keeps_in_process_state_tools_out():
    served = mcp_server.served()
    assert "next_tradable" in served and "profile_table" in served and "run_pipeline" in served
    assert not {"construct_feature", "ic_summary", "critic_test", "record_observation"} & set(served)


def _agent_run(monkeypatch, tmp_path, tools):
    monkeypatch.setenv("QA_TOOLS", tools)
    session = f"mcp-{tools}"
    registry.set_session(session, tmp_path / tools)
    registry.CTX.node_exec = "node#1"
    script = [{"tool_calls": [CALL]}, {"tool_calls": [{**CALL, "arguments": {**CALL["arguments"],
                                                                             "timestamp": "2025-03-08 09:00"}}]},
              good_submit]
    try:
        draft, trace = TinyAgent().run(ScriptedClient(script), "When?")
    finally:
        registry.CTX.node_exec = None
    rows = registry.session_runs(session)[LEDGER].assign(session_id="S").to_dict("records")
    outputs = sorted(json.loads(p.read_text())["result"].__repr__() for p in (tmp_path / tools / "tool_outputs").glob("*"))
    return _norm({"draft": draft.model_dump(), "trace": {k: v for k, v in trace.items() if k != "messages"},
                  "messages": trace["messages"], "rows": rows, "outputs": outputs})


def test_agent_through_mcp_writes_the_same_ledger_rows(monkeypatch, tmp_path):
    direct = _agent_run(monkeypatch, tmp_path, "direct")
    over_mcp = _agent_run(monkeypatch, tmp_path, "mcp")
    assert len(direct["rows"]) == 2 and over_mcp["rows"] == direct["rows"]
    assert {r["agent"] for r in over_mcp["rows"]} == {"tiny"} and over_mcp["rows"][0]["node_exec"] == "node#1"
    assert over_mcp == direct
