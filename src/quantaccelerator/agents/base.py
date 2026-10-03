"""Tool-calling agent loop with a schema-validated final answer.

The agent finishes by calling `submit` whose parameters are the JSON schema of its pydantic output model.
Validation failures, and citations of run_ids that do not exist in this session's registry, are fed back.
After two failed submissions the agent fails (guardrail: no unvalidated state enters the shared store).
"""
import json
import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from quantaccelerator.llm.client import LLMClient, prompt_hash
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
        """Run the loop. Any failure (including server errors) is raised as AgentFailure carrying the trace."""
        trace = {"agent": self.name}
        try:
            return self._run(client, task, context, trace)
        except AgentFailure:
            raise
        except Exception as e:
            raise AgentFailure(f"{self.name}: {type(e).__name__}: {e}", trace) from e

    def _run(self, client: LLMClient, task: str, context: dict | None, trace: dict) -> tuple[BaseModel, dict]:
        tools = registry.get_tools(self.tool_names)
        submit = {"type": "function", "function": {
            "name": "submit", "description": f"Submit the final {self.output_model.__name__}.",
            "parameters": inline_schema(self.output_model)}}
        schemas = [t.openai_schema() for t in tools] + [submit]
        by_name = {t.name: t for t in tools}
        messages = [{"role": "system", "content": f"{self.role_prompt}\n\n{COMMON_RULES}"},
                    {"role": "user", "content": task + ("\n\nContext (JSON):\n" + json.dumps(context, default=str)
                                                        if context else "")}]
        trace.update({"prompt_hash": prompt_hash(messages[:1], schemas), "tool_runs": [], "tool_calls": 0,
                      "submit_failures": 0, "submit_errors": [], "nudges": 0, "steps": 0, "messages": messages})
        registry.CTX.agent = self.name
        drafts = []  # (message index, tool-call index) of submits not accepted; only the latest keeps its arguments
        done = {}    # (tool, canonical args) -> result of an earlier call, so identical calls are not re-run
        last = None
        for step in range(self.max_steps):
            trace["steps"] = step + 1
            if step == self.max_steps - 3:  # after the previous turn's tool results, so the message order stays valid
                messages.append({"role": "user", "content": "Only 3 steps remain. Stop exploring and call `submit` "
                                 "with your best complete answer now."})
            if self._compact(messages):
                last = None  # the prompt shrank: estimate it again from characters
                trace["compacted"] = trace.get("compacted", 0) + 1
            cap = self._output_cap(messages, schemas, last)
            if cap < MIN_OUTPUT_TOKENS:
                raise AgentFailure(f"{self.name}: context window exhausted ({cap} tokens left for the answer)", trace)
            try:
                resp = client.chat(messages, schemas, agent=self.name, max_tokens=cap)
            except Exception as e:  # the estimate was short: the server says exactly how long the prompt is
                m = re.search(r"\((\d+) in the messages", str(e))
                if not m:
                    raise
                cap = CTX_LIMIT - int(m.group(1)) - 64
                trace["context_retries"] = trace.get("context_retries", 0) + 1
                if cap < MIN_OUTPUT_TOKENS:
                    raise AgentFailure(f"{self.name}: context window exhausted ({cap} tokens left for the answer)", trace)
                resp = client.chat(messages, schemas, agent=self.name, max_tokens=cap)
            last = (resp.prompt_tokens, len(json.dumps(messages, default=str)))
            messages.append(resp.as_message())
            ai = len(messages) - 1
            prose = self.max_prose_chars is not None and len(resp.content or "") > self.max_prose_chars
            if prose:  # long prose costs context on every later call; keep a short excerpt
                trace["prose_trimmed"] = trace.get("prose_trimmed", 0) + 1
                messages[ai]["content"] = resp.content[:self.max_prose_chars] + " [...]"
            if not resp.tool_calls:
                trace["nudges"] += 1
                if trace["nudges"] > 3:
                    raise AgentFailure(f"{self.name}: no tool call after repeated reminders", trace)
                if resp.finish_reason == "length":
                    # a truncated answer is useless context; drop it and ask for a shorter one
                    trace["truncated"] = trace.get("truncated", 0) + 1
                    messages[-1] = {"role": "assistant", "content": "[output truncated at the token limit]"}
                    messages.append({"role": "user", "content": "Your last output was cut off at the token limit. Call "
                                     "`submit` again with a more concise answer: one short sentence per meaning, short "
                                     "evidence lists, no prose outside the tool call."})
                else:
                    if self.compact_after_turns is not None:
                        messages.pop()  # a tool-less preamble left in the history tends to be repeated verbatim
                    messages.append({"role": "user", "content": "Continue by calling a tool, or call `submit` with the "
                                                                "final structured answer."})
                continue
            # a submit after only record (unpaced) calls is fine: they run first and are checked on the spot
            names = [tc.name for tc in resp.tool_calls]
            mixed = "submit" in names and len(names) > 1 and not (
                names[-1] == "submit" and names.count("submit") == 1 and all(n in self.unpaced_tools for n in names[:-1]))
            n_data = 0
            for k, tc in enumerate(resp.tool_calls):
                if tc.name != "submit" and tc.name not in self.unpaced_tools and \
                        self.max_tool_calls_per_turn is not None:
                    n_data += 1
                    if n_data > self.max_tool_calls_per_turn:
                        trace["paced_calls"] = trace.get("paced_calls", 0) + 1
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({
                            "ok": False, "error": f"Not run: at most {self.max_tool_calls_per_turn} tool calls per turn "
                            "here. Read the results you have, decide what they imply and what to check next, then "
                            "call the next tool."})})
                        continue
                if tc.name == "submit":
                    # a rejected draft stays in the history as the base for the fix; older drafts only cost context
                    for i, j in drafts:
                        messages[i]["tool_calls"][j]["function"]["arguments"] = SUPERSEDED_DRAFT
                    drafts.append((ai, k))
                if tc.name == "submit" and mixed:
                    # submitting in the same turn as data tools means the answer ignores their results
                    trace["premature_submits"] = trace.get("premature_submits", 0) + 1
                    content = {"ok": False, "error": "Submission ignored: you called submit in the same turn as other "
                                                     "tools, before seeing their results. Review the results below, make "
                                                     "sure your answer is consistent with every one of them, "
                                                     "then call submit on its own."}
                elif tc.name == "submit":
                    draft, err = self._check_submit(tc.arguments)
                    if draft is not None:
                        return draft, trace
                    # the same rejection with no work in between is not a new attempt: say so instead of counting it
                    idle = self.compact_after_turns is not None and trace["submit_errors"] and \
                        trace["submit_errors"][-1] == err[:600] and \
                        trace.get("_tool_calls_at_rejection") == trace["tool_calls"]
                    trace["_tool_calls_at_rejection"] = trace["tool_calls"]
                    if idle:
                        trace["idle_resubmits"] = trace.get("idle_resubmits", 0) + 1
                        content = {"ok": False, "error": "Submission rejected again for the same reason, with no tool "
                                   f"call since: do the steps it lists first, then submit. {err}"}
                    else:
                        trace["submit_failures"] += 1
                        trace["submit_errors"].append(err[:600])
                        if trace["submit_failures"] > self.max_submit_failures:
                            raise AgentFailure(f"{self.name}: submission invalid after retries: {err}", trace)
                        content = {"ok": False, "error": f"Submission rejected; fix and call submit again. {err}"}
                elif tc.name in by_name:
                    try:
                        args = _loads_lenient(tc.arguments)
                        key = (tc.name, json.dumps(args, sort_keys=True, default=str))
                        if key in done and self.compact_after_turns is not None:
                            # its result may have been shortened in the history: show it again instead of refusing
                            trace["repeated_calls"] = trace.get("repeated_calls", 0) + 1
                            content = {**done[key], "note": f"same call as run_id {done[key].get('run_id')}; its "
                                       "result again (do not repeat it)"}
                        elif key in done:
                            trace["repeated_calls"] = trace.get("repeated_calls", 0) + 1
                            prev = done[key]
                            content = {"ok": False, "run_id": prev.get("run_id"), "error": (
                                f"Identical to your earlier {tc.name} call (run_id {prev.get('run_id')}); it was not "
                                "re-run and would give the same result. Do not repeat it: change the arguments, "
                                "use another tool, or submit."),
                                       "earlier_error": r.get("error") if isinstance(r := prev.get("result"), dict) else None}
                        else:
                            trace["tool_calls"] += 1
                            content = by_name[tc.name].fn(**args)
                            trace["tool_runs"].append(content["run_id"])
                            if content.get("ok"):  # a failed call may succeed later (e.g. once an input exists)
                                done[key] = content
                    except (json.JSONDecodeError, TypeError) as e:
                        content = {"ok": False, "error": f"bad arguments: {e}"}
                else:
                    content = {"ok": False, "error": f"unknown tool {tc.name}; available: {list(by_name)} + submit"}
                text = json.dumps(content, default=str, separators=(",", ":"))
                if len(text) > MAX_TOOL_RESULT_CHARS:
                    text = text[:MAX_TOOL_RESULT_CHARS] + '..."[truncated]'
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
            if prose:  # after this turn's tool results, so the message order stays valid
                messages.append({"role": "user", "content": self.prose_reminder})
        raise AgentFailure(f"{self.name}: no valid submission within {self.max_steps} steps", trace)

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
