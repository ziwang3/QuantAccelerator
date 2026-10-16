"""An agent: its role, tools, output model and limits, and the checks its loop applies.

Each agent runs as a LangChain create_agent ReAct loop (agents/lc_agent.py) whose middleware (agents/middleware.py)
applies the methods here. The agent finishes by calling `submit`, whose parameters are the JSON schema of its pydantic
output model. Validation failures, and citations of run_ids that do not exist in this session's registry, are fed back.
After too many failed submissions the agent fails (guardrail: no unvalidated state enters the shared store).
"""
import json
import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from quantaccelerator.llm.client import LLMClient
from quantaccelerator.tools import registry

RUN_ID = re.compile(r"\br[0-9a-f]{10}\b")
MAX_TOOL_RESULT_CHARS = 9000
CTX_LIMIT = int(os.environ.get("LLM_CONTEXT", "32768"))  # server context window (vllm --max-model-len)
CTX_MARGIN = 512           # slack for the chat template and the token estimate
MIN_OUTPUT_TOKENS = 1024   # below this an answer cannot fit; fail cleanly instead of a server 400
CHARS_PER_TOKEN = 3        # for JSON-heavy English; a short estimate is corrected from the server error
SUPERSEDED_DRAFT = '{"superseded_draft": true}'

COMMON_RULES = """Rules you must follow:
- You work only through the tools. Deterministic tools compute every number; never compute or guess numbers yourself.
- Every tool result has a run_id field ('r' followed by 10 hex characters). Whenever you state a number or an empirical
  claim, cite the run_id of the tool call that produced it, copied from that result, and cite docs as '<doc> p<page>'.
- Only cite run_ids that appear in your tool results.
- Be efficient: call only the tools you need; you may call several tools in one turn.
- When done, call the `submit` tool exactly once with the final structured answer. Do not answer in plain text."""


def _paths(obj, ids: set, path: str = "") -> list[str]:
    """JSON paths (a.b[2].c) of string values that mention any of `ids`."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _paths(v, ids, f"{path}.{k}" if path else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in _paths(v, ids, f"{path}[{i}]")]
    return [path] if isinstance(obj, str) and any(i in obj for i in ids) else []


class AgentFailure(RuntimeError):
    def __init__(self, msg: str, trace: dict | None = None):
        super().__init__(msg)
        self.trace = trace or {}


def inline_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $refs inlined and titles dropped (easier for tool-calling models)."""
    return registry.inline_refs(model.model_json_schema())


def _loads_lenient(s: str) -> Any:
    """Parse tool arguments; also unwrap values that the model double-encoded as JSON strings."""
    obj = json.loads(s) if s.strip() else {}

    def fix(v):
        if isinstance(v, str) and v[:1] in "[{":
            try:
                return fix(json.loads(v))
            except json.JSONDecodeError:
                return v
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v

    return fix(obj)


class Agent:
    name: str = "agent"
    role_prompt: str = ""
    tool_names: list[str] = []
    output_model: type[BaseModel]
    max_steps: int = 24
    max_submit_failures: int = 2
    max_output_tokens: int = 3072  # per LLM call; large enough for the output model, small enough to stop runaways
    max_tool_calls_per_turn: int | None = None  # pace an exploratory loop: read results before the next question
    unpaced_tools: tuple = ()                   # tools that do not count toward the pace (e.g. recording findings)
    max_prose_chars: int | None = None          # longer free text is trimmed in the history, with a reminder
    prose_reminder: str = "Keep your reasoning to a few sentences."
    compact_after_turns: int | None = None      # tool results older than this many turns shrink to a one-line reading

    def _compact(self, messages: list[dict]) -> int:
        """Shorten tool results from turns older than `compact_after_turns` to their run_id, a one-line reading and
        their flags (the full result stays in the run's tool_outputs; citing the run_id still works). Returns the
        number of results shortened."""
        from quantaccelerator.viz.events import summarize
        turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if self.compact_after_turns is None or len(turns) <= self.compact_after_turns:
            return 0
        cutoff, calls, n = turns[-self.compact_after_turns], {}, 0
        for i, m in enumerate(messages[:cutoff]):
            for tc in m.get("tool_calls") or []:
                calls[tc["id"]] = dict(tc["function"])  # the full arguments, before any shortening below
            if m["role"] == "assistant":
                # old prose, and the long texts of recorded findings (they are stored in the session's findings)
                if m.get("tool_calls") and m.get("content"):
                    m["content"] = ""  # the calls remain; cut-off prose in the history gets imitated
                elif len(m.get("content") or "") > 200:
                    m["content"] = m["content"][:200] + " [...]"
                for tc in m.get("tool_calls") or []:
                    if tc["function"]["name"] in self.unpaced_tools:
                        try:
                            a = json.loads(tc["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            continue
                        # keep only the short identifying fields; the full record is stored in the session's findings
                        tc["function"]["arguments"] = json.dumps({k: v for k, v in a.items() if isinstance(v, str)
                                                                  and len(v) <= 60}) \
                            if isinstance(a, dict) else tc["function"]["arguments"]
            if m["role"] != "tool" or m["content"].startswith('{"compacted"'):
                continue
            try:
                body = json.loads(m["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(body, dict) and body.get("ok") is False and "run_id" not in body and "error" in body:
                # an older rejection (a submit, a paced or repeated call): the latest one is still shown in full
                m["content"] = json.dumps({"compacted": True, "ok": False, "error": "older rejection; see the latest"})
                n += 1
                continue
            if not isinstance(body, dict) or "result" not in body or not body.get("run_id"):
                continue
            fn = calls.get(m.get("tool_call_id"), {"name": "?", "arguments": "{}"})
            res = body["result"]
            try:
                args = _loads_lenient(fn["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            short = {"compacted": True, "run_id": body["run_id"], "ok": body.get("ok", True)}
            if fn["name"] not in self.unpaced_tools or not body.get("ok", True):   # a successful record ack needs none
                short["reading"] = summarize(fn["name"], args, res)[:400]
            if isinstance(res, dict) and res.get("flags"):
                short["flags"] = [f.get("fact") for f in res["flags"]]
            m["content"] = json.dumps(short, separators=(",", ":"))
            n += 1
        return n

    def extra_validate(self, draft: BaseModel) -> list[str]:
        """Agent-specific semantic checks; return error messages (empty = OK)."""
        return []

    def _output_cap(self, messages: list[dict], schemas: list[dict], last: tuple[int, int] | None) -> int:
        """Largest max_tokens that keeps prompt + completion inside the context window.

        `last` is (prompt_tokens reported for the previous call, len of the message JSON at that call); growth since
        then is estimated from characters. Without a previous call the whole prompt is estimated from characters.
        """
        if last and last[0]:
            est = last[0] + int((len(json.dumps(messages, default=str)) - last[1]) / CHARS_PER_TOKEN)
        else:
            est = int(len(json.dumps([messages, schemas], default=str)) / CHARS_PER_TOKEN)
        return min(self.max_output_tokens, CTX_LIMIT - est - CTX_MARGIN)

    def run(self, client: LLMClient, task: str, context: dict | None = None) -> tuple[BaseModel, dict]:
        """Run the agent as a LangChain create_agent ReAct loop with this class's guards (agents/lc_agent.py). Returns
        (validated output model, trace); any failure (including server errors) is raised as AgentFailure with the
        trace."""
        from quantaccelerator.agents import lc_agent
        return lc_agent.run(self, client, task, context)

    def _check_submit(self, arguments: str) -> tuple[BaseModel | None, str]:
        try:
            obj = _loads_lenient(arguments)
            # models sometimes wrap the answer in one extra key ({"research_draft": {...}}); unwrap when that fits
            if isinstance(obj, dict) and len(obj) == 1 and isinstance(v := next(iter(obj.values())), dict) and \
                    next(iter(obj)) not in self.output_model.model_fields:
                obj = v
            draft = self.output_model.model_validate(obj)
        except json.JSONDecodeError as e:
            return None, f"arguments are not valid JSON: {e}"
        except ValidationError as e:
            return None, "schema validation failed: " + "; ".join(
                f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:8])
        cited = set(RUN_ID.findall(draft.model_dump_json()))
        unknown = [r for r in cited if not registry.run_exists(r, registry.CTX.session_id)]
        if unknown:
            where = _paths(draft.model_dump(mode="json"), set(unknown))
            mine = {}
            try:
                runs = registry.session_runs(registry.CTX.session_id)
                for _, r in runs[(runs.agent == self.name) & (runs.ok == 1)].iterrows():
                    mine.setdefault(r.tool, []).append(r.run_id)
            except Exception:
                pass
            yours = "; ".join(f"{t}: {', '.join(v[-4:])}" for t, v in mine.items())
            return None, (f"cited run_ids not produced by your tool calls: {unknown}, at {where[:6]}. Replace each with the "
                          "run_id of the tool result that supports that item (copy it from the result), or remove it."
                          + (f" Your runs so far, by tool: {yours}" if yours else ""))
        errs = self.extra_validate(draft)
        return (None, " ".join(errs)) if errs else (draft, "")
