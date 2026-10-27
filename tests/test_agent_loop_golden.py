"""Golden tests of the agent loop's guards: for each scripted scenario, every request sent to the model (messages, tool
schemas, max_tokens) and the final result and trace must equal the recorded fixture (tests/fixtures/agent_loop/). The
fixtures were recorded from the original hand-written agent loop, whose guards the create_agent middleware ports;
QA_REGEN_GOLDEN=1 records them again from the current loop."""
import json
import os
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from quantaccelerator.agents.base import Agent, AgentFailure
from quantaccelerator.llm.client import ScriptedClient
from quantaccelerator.tools import registry

FIXTURES = Path(__file__).parent / "fixtures" / "agent_loop"

RID = re.compile(r"r[0-9a-f]{10}")
TS = {"timestamp": "2025-03-07 16:01", "timezone": "America/New_York"}
CALL = {"name": "next_tradable", "arguments": TS}


class Tiny(BaseModel):
    answer: str
    evidence: list[str]


class TinyAgent(Agent):
    name = "tiny"
    tool_names = ["next_tradable"]
    output_model = Tiny


class Recording(ScriptedClient):
    def _complete(self, messages, tools, max_tokens=None):
        self.seen.append(json.loads(json.dumps({"messages": messages, "tools": tools, "max_tokens": max_tokens})))
        return super()._complete(messages, tools, max_tokens)


def last_rid(m):
    return [x for x in (json.loads(y["content"]) for y in m if y["role"] == "tool") if x.get("run_id")][-1]["run_id"]


def good_submit(m):
    return {"tool_calls": [{"name": "submit", "arguments": {"answer": "Monday", "evidence": [last_rid(m)]}}]}


def _norm(x):
    """run_ids differ between two runs (random); map them to their order of first appearance."""
    s = json.dumps(x, sort_keys=True, default=str)
    ids = {}
    return json.loads(RID.sub(lambda m: ids.setdefault(m.group(0), f"R{len(ids)}"), s))


def _run(tmp_path, script, **attrs):
    registry.set_session("golden", tmp_path)
    client = Recording(script)
    client.seen = []
    agent = TinyAgent()
    for k, v in attrs.items():
        setattr(agent, k, v)
    try:
        draft, trace = agent.run(client, "When is the next tradable session?", {"note": "context"})
        out = {"draft": draft.model_dump(), "trace": trace}
    except AgentFailure as e:
        out = {"failure": str(e), "trace": e.trace}
    return _norm({**out, "seen": client.seen})


SCENARIOS = {
    "retry_schema_and_citation": ([
        {"tool_calls": [CALL]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "x"}}]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "x", "evidence": ["r0000000000"]}}]},
        good_submit], {}),
    "too_many_submit_failures": ([
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "x"}}]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "y"}}]}], {"max_submit_failures": 1}),
    "mixed_submit_and_pacing": ([
        {"tool_calls": [CALL, {"name": "next_tradable", "arguments": {**TS, "timestamp": "2025-03-08 10:00"}},
                        {"name": "submit", "arguments": {"answer": "x", "evidence": []}}]},
        good_submit], {"max_tool_calls_per_turn": 1}),
    "repeated_call": ([{"tool_calls": [CALL]}, {"tool_calls": [CALL]}, good_submit], {}),
    "repeated_call_compacting": ([{"tool_calls": [CALL]}, {"tool_calls": [CALL]}, {"content": "thinking"},
                                  {"tool_calls": [CALL]}, good_submit], {"compact_after_turns": 1}),
    "nudges_and_prose": ([{"content": "a" * 500}, {"content": "b" * 50, "tool_calls": [CALL]}, {"content": "c"},
                          good_submit], {"max_prose_chars": 20}),
    "too_many_nudges": ([{"content": "x"}] * 5, {}),
    "truncated": ([{"tool_calls": [CALL]}, {"content": "{\"answer\": \"Mon", "finish_reason": "length"},
                   good_submit], {}),
    "bad_args_and_unknown_tool": ([
        {"tool_calls": [{"name": "next_tradable", "arguments": "{not json"}]},
        {"tool_calls": [{"name": "next_tradable", "arguments": "[1, 2]"}]},
        {"tool_calls": [{"name": "nope", "arguments": {}}]},
        {"tool_calls": [CALL]}, good_submit], {}),
    "max_steps": ([{"tool_calls": [{"name": "next_tradable", "arguments": {**TS, "timestamp": f"2025-03-0{i} 10:00"}}]}
                   for i in range(1, 9)], {"max_steps": 6}),
    "idle_resubmit_compacting": ([
        {"tool_calls": [CALL]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "x"}}]},
        {"tool_calls": [{"name": "submit", "arguments": {"answer": "x"}}]},
        good_submit], {"compact_after_turns": 2}),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_agent_loop_matches_the_golden_record(name, tmp_path):
    script, attrs = SCENARIOS[name]
    got = _run(tmp_path, script, **attrs)
    path = FIXTURES / f"{name}.json"
    if os.environ.get("QA_REGEN_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, sort_keys=True))
    want = json.loads(path.read_text())
    assert len(got["seen"]) == len(want["seen"])
    for k, (a, b) in enumerate(zip(want["seen"], got["seen"])):
        assert a == b, f"model call {k} differs"
    assert got == want
