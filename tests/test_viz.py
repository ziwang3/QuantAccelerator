import json

from quantaccelerator.viz.events import run_events, summarize


def _write_run(d):
    """A minimal run directory: one profiler turn with a tool call, a rejected and an accepted submit."""
    (d / "transcripts").mkdir(parents=True)
    msgs = [{"role": "system", "content": "role"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "list_tables", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(
                {"run_id": "r0123456789", "ok": True, "result": [{"table": "submission", "n_rows": 5}]})},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "submit", "arguments": "{\"x\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"ok": False, "error": "schema failed"})},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c3", "type": "function", "function": {"name": "submit", "arguments": "{\"x\": 2}"}}]}]
    (d / "transcripts" / "profiler.json").write_text(json.dumps(msgs))
    calls = [{"ts": f"2026-09-28T10:00:0{i}", "agent": "profiler", "latency_s": 1.0, "tokens_in": 10, "tokens_out": 5,
              "finish_reason": "tool_calls"} for i in (1, 3, 5)]
    (d / "llm_calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))
    (d / "run_meta.json").write_text(json.dumps({"run_id": "t", "pipeline": "p.py", "model": "m", "status": "ok",
                                                 "agents": {"profiler": {"wall_s": 6.0}}}))


def test_run_events_from_run_dir(tmp_path):
    _write_run(tmp_path)
    ev = run_events(tmp_path)
    types = [e["type"] for e in ev]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types.count("llm") == 3 and types.count("tool") == 1
    subs = [e for e in ev if e["type"] == "submit"]
    assert [s["accepted"] for s in subs] == [False, True] and subs[0]["error"] == "schema failed"
    tool = next(e for e in ev if e["type"] == "tool")
    assert tool["run_id"] == "r0123456789" and tool["summary"] == "1 tables: submission (5 rows)"
    assert all(a["t"] <= b["t"] for a, b in zip(ev, ev[1:]))
    json.dumps(ev)  # serialisable for the page


def test_summarize_reports_errors_and_lookahead():
    assert summarize("run_pipeline", {}, {"error": "RuntimeError: boom\nIndentationError: x"}).startswith("error: ")
    assert summarize("measure_lookahead", {}, {"share_before_tradable": 0.9037, "n_events": 2429}) == \
        "90.4% of 2,429 events used before they were tradable"
