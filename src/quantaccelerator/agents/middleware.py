"""The agent loop's guards as one LangChain v1 AgentMiddleware.

Every guard came from a recorded failure. They use the Agent's own methods (_compact, _output_cap, _check_submit,
extra_validate; agents/base.py) on an OpenAI-format mirror of the history: the LangGraph state keeps the raw messages
(what tracing shows), the mirror is what the model is sent, with old results compacted, superseded drafts blanked and
long prose trimmed. tests/test_agent_loop_golden.py pins every guard to recorded transcripts.

  hook             guards (base.py)
  before_model     max_steps -> AgentFailure; "Only 3 steps remain"; the prose reminder after a trimmed turn
  wrap_model_call  compaction of old tool results; output cap from the 32k window; retry with the exact prompt length
                   from vLLM's 400; context-exhausted failure
  after_model      prose trimming; no-tool-call nudges (> 3 fails); truncated output dropped and re-asked; a tool-less
                   preamble popped
  wrap_tool_call   per-turn pacing; submit mixed with other tools; superseded drafts; submit validation (lenient JSON,
                   unwrap, pydantic, run_id citation check, extra_validate) with the idle-resubmit rule and
                   max_submit_failures; repeated-call cache; bad arguments; unknown tools; 9000-char results

create_agent sends each tool call of a turn as its own task; the rules (pacing, mixed submit, the repeated-call
cache) are about the order of calls within a turn, so the first task of a turn runs the whole turn in order, under a
lock, and every task returns its precomputed message.
"""
import json
import re
import threading
import uuid

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from quantaccelerator.agents.base import (CTX_LIMIT, MAX_TOOL_RESULT_CHARS, MIN_OUTPUT_TOKENS, SUPERSEDED_DRAFT,
                                          AgentFailure, _loads_lenient)
from quantaccelerator.llm.lc_model import RAW_CALLS, to_lc, to_openai
from quantaccelerator.tools import registry

SYSTEM_ID = "__system__"


def _same_name(fn):
    """A no-op with fn's name, so fn(**not_a_mapping) raises Python's own TypeError text for that tool."""
    def stub(**kw):
        return kw
    stub.__module__, stub.__qualname__, stub.__name__ = fn.__module__, fn.__qualname__, fn.__name__
    return stub


def _hid() -> str:
    return f"qa-{uuid.uuid4().hex[:12]}"


class NativeGuards(AgentMiddleware):
    def __init__(self, agent, system_prompt: str, schemas: list[dict], tools: dict, trace: dict):
        super().__init__()
        self.agent, self.schemas, self.lc_tools, self.trace = agent, schemas, tools, trace
        self.ids = [SYSTEM_ID]                                   # mirror order (state message ids)
        self.mirror = {SYSTEM_ID: {"role": "system", "content": system_prompt}}
        self.hidden = set()                                      # tool replies the model is never shown
        self.turns: dict[str, dict] = {}                         # AI message id -> {tool_call_id: ToolMessage}
        self.drafts = []                                         # (AI message id, call index) of rejected submits
        self.done = {}                                           # (tool, canonical args) -> earlier result
        self.last = None
        self.steps = 0
        self.prose_pending = False
        self.final = None
        self.failure = None
        self.lock = threading.Lock()

    @property
    def name(self) -> str:
        return "NativeGuards"

    # ------------------------------------------------------------------ the OpenAI-format mirror
    def _sync(self, messages) -> None:
        """Add new state messages to the mirror (tool replies in call order), drop removed ones."""
        live = {m.id for m in messages}
        self.ids = [i for i in self.ids if i == SYSTEM_ID or i in live]
        known = set(self.ids)
        new = [m for m in messages if m.id not in known and m.id not in self.hidden]
        order = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for k, t in enumerate(m.additional_kwargs.get(RAW_CALLS) or []):
                    order[t["id"]] = k
        i = 0
        while i < len(new):   # a run of tool replies is put in the order of the calls they answer
            j = i
            while j < len(new) and isinstance(new[j], ToolMessage):
                j += 1
            if j > i:
                new[i:j] = sorted(new[i:j], key=lambda m: order.get(m.tool_call_id, 0))
                i = j
            else:
                i += 1
        for m in new:
            self.ids.append(m.id)
            self.mirror[m.id] = to_openai([m])[0]

    def view(self) -> list[dict]:
        return [self.mirror[i] for i in self.ids]

    def _fail(self, msg: str) -> AgentFailure:
        self.trace["messages"] = self.view()
        return AgentFailure(msg, self.trace)

    # ------------------------------------------------------------------ hooks
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if self.final is not None:
            return {"jump_to": "end"}
        if self.failure is not None:
            raise self.failure
        self._sync(state["messages"])
        add = []
        if self.prose_pending:   # after the previous turn's tool results, so the message order stays valid
            self.prose_pending = False
            add.append(HumanMessage(self.agent.prose_reminder, id=_hid()))
        if self.steps >= self.agent.max_steps:
            raise self._fail(f"{self.agent.name}: no valid submission within {self.agent.max_steps} steps")
        self.trace["steps"] = self.steps + 1
        if self.steps == self.agent.max_steps - 3:
            add.append(HumanMessage("Only 3 steps remain. Stop exploring and call `submit` with your best complete "
                                    "answer now.", id=_hid()))
        self.steps += 1
        return {"messages": add} if add else None

    def wrap_model_call(self, request, handler):
        self._sync(request.messages)
        view = self.view()
        if self.agent._compact(view):
            self.last = None  # the prompt shrank: estimate it again from characters
            self.trace["compacted"] = self.trace.get("compacted", 0) + 1
        cap = self.agent._output_cap(view, self.schemas, self.last)
        if cap < MIN_OUTPUT_TOKENS:
            raise self._fail(f"{self.agent.name}: context window exhausted ({cap} tokens left for the answer)")
        req = request.override(system_message=SystemMessage(view[0]["content"]), messages=to_lc(view[1:]),
                               model_settings={**request.model_settings, "max_tokens": cap})
        try:
            resp = handler(req)
        except Exception as e:  # the estimate was short: the server says exactly how long the prompt is
            m = re.search(r"\((\d+) in the messages", str(e))
            if not m:
                raise
            cap = CTX_LIMIT - int(m.group(1)) - 64
            self.trace["context_retries"] = self.trace.get("context_retries", 0) + 1
            if cap < MIN_OUTPUT_TOKENS:
                raise self._fail(f"{self.agent.name}: context window exhausted ({cap} tokens left for the answer)")
            resp = handler(req.override(model_settings={**request.model_settings, "max_tokens": cap}))
        ai = next(m for m in resp.result if isinstance(m, AIMessage))
        self.last = (ai.response_metadata.get("prompt_tokens", 0), len(json.dumps(view, default=str)))
        return resp

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        ai = state["messages"][-1]
        self._sync(state["messages"])
        a = self.agent
        prose = a.max_prose_chars is not None and len(ai.content or "") > a.max_prose_chars
        if prose:  # long prose costs context on every later call; keep a short excerpt
            self.trace["prose_trimmed"] = self.trace.get("prose_trimmed", 0) + 1
            self.mirror[ai.id]["content"] = ai.content[:a.max_prose_chars] + " [...]"
        if ai.additional_kwargs.get(RAW_CALLS):
            self.prose_pending = prose
            return None
        self.trace["nudges"] += 1
        if self.trace["nudges"] > 3:
            raise self._fail(f"{a.name}: no tool call after repeated reminders")
        upd = []
        if ai.response_metadata.get("finish_reason") == "length":
            # a truncated answer is useless context; drop it and ask for a shorter one
            self.trace["truncated"] = self.trace.get("truncated", 0) + 1
            self.mirror[ai.id] = {"role": "assistant", "content": "[output truncated at the token limit]"}
            upd += [AIMessage(id=ai.id, content="[output truncated at the token limit]"),
                    HumanMessage("Your last output was cut off at the token limit. Call `submit` again with a more "
                                 "concise answer: one short sentence per meaning, short evidence lists, no prose "
                                 "outside the tool call.", id=_hid())]
        else:
            if a.compact_after_turns is not None:
                upd.append(RemoveMessage(id=ai.id))  # a tool-less preamble left in the history tends to be repeated
            upd.append(HumanMessage("Continue by calling a tool, or call `submit` with the final structured answer.",
                                    id=_hid()))
        return {"messages": upd, "jump_to": "model"}

    def wrap_tool_call(self, request, handler):
        with self.lock:
            msgs = request.state["messages"] if isinstance(request.state, dict) else request.state.messages
            ai = next(m for m in reversed(msgs) if isinstance(m, AIMessage))
            if ai.id not in self.turns:
                self.turns[ai.id] = self._run_turn(ai, request, handler)
            return self.turns[ai.id][request.tool_call["id"]]

    # ------------------------------------------------------------------ one turn, in call order (base.py:231-308)
    def _run_turn(self, ai: AIMessage, request, handler) -> dict:
        a, trace = self.agent, self.trace
        calls = ai.additional_kwargs.get(RAW_CALLS) or []
        out = {}

        def reply(tc, k, text, hidden=False):
            mid = f"{ai.id}-t{k}"
            if hidden:
                self.hidden.add(mid)
            out[tc["id"]] = ToolMessage(content=text, tool_call_id=tc["id"], name=tc["name"], id=mid)

        names = [tc["name"] for tc in calls]
        mixed = "submit" in names and len(names) > 1 and not (
            names[-1] == "submit" and names.count("submit") == 1 and all(n in a.unpaced_tools for n in names[:-1]))
        n_data = 0
        for k, tc in enumerate(calls):
            if self.final is not None or self.failure is not None:   # the loop has ended or failed
                reply(tc, k, json.dumps({"ok": False, "error": "not run"}), hidden=True)
                continue
            name = tc["name"]
            if name != "submit" and name not in a.unpaced_tools and a.max_tool_calls_per_turn is not None:
                n_data += 1
                if n_data > a.max_tool_calls_per_turn:
                    trace["paced_calls"] = trace.get("paced_calls", 0) + 1
                    reply(tc, k, json.dumps({
                        "ok": False, "error": f"Not run: at most {a.max_tool_calls_per_turn} tool calls per turn "
                        "here. Read the results you have, decide what they imply and what to check next, then "
                        "call the next tool."}))
                    continue
            if name == "submit":
                # a rejected draft stays in the history as the base for the fix; older drafts only cost context
                for i, j in self.drafts:
                    if i in self.mirror:
                        self.mirror[i]["tool_calls"][j]["function"]["arguments"] = SUPERSEDED_DRAFT
                self.drafts.append((ai.id, k))
            if name == "submit" and mixed:
                # submitting in the same turn as data tools means the answer ignores their results
                trace["premature_submits"] = trace.get("premature_submits", 0) + 1
                content = {"ok": False, "error": "Submission ignored: you called submit in the same turn as other "
                                                 "tools, before seeing their results. Review the results below, make "
                                                 "sure your answer is consistent with every one of them, "
                                                 "then call submit on its own."}
            elif name == "submit":
                draft, err = a._check_submit(tc["arguments"])
                if draft is not None:
                    self.final = draft
                    reply(tc, k, json.dumps({"ok": True, "accepted": True}), hidden=True)
                    continue
                # the same rejection with no work in between is not a new attempt: say so instead of counting it
                idle = a.compact_after_turns is not None and trace["submit_errors"] and \
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
                    if trace["submit_failures"] > a.max_submit_failures:
                        self.failure = self._fail(f"{a.name}: submission invalid after retries: {err}")
                        reply(tc, k, json.dumps({"ok": False, "error": err}), hidden=True)
                        continue
                    content = {"ok": False, "error": f"Submission rejected; fix and call submit again. {err}"}
            elif name in self.lc_tools:
                try:
                    args = _loads_lenient(tc["arguments"])
                    key = (name, json.dumps(args, sort_keys=True, default=str))
                    if key in self.done and a.compact_after_turns is not None:
                        # its result may have been shortened in the history: show it again instead of refusing
                        trace["repeated_calls"] = trace.get("repeated_calls", 0) + 1
                        content = {**self.done[key], "note": f"same call as run_id {self.done[key].get('run_id')}; "
                                   "its result again (do not repeat it)"}
                    elif key in self.done:
                        trace["repeated_calls"] = trace.get("repeated_calls", 0) + 1
                        prev = self.done[key]
                        content = {"ok": False, "run_id": prev.get("run_id"), "error": (
                            f"Identical to your earlier {name} call (run_id {prev.get('run_id')}); it was not "
                            "re-run and would give the same result. Do not repeat it: change the arguments, "
                            "use another tool, or submit."),
                                   "earlier_error": r.get("error") if isinstance(r := prev.get("result"), dict) else None}
                    else:
                        trace["tool_calls"] += 1
                        if not isinstance(args, dict):   # fn(**args) would raise this TypeError
                            _same_name(registry.TOOLS[name].fn)(**args)
                        msg = handler(request.override(tool_call={"name": name, "args": args, "id": tc["id"],
                                                                  "type": "tool_call"}, tool=self.lc_tools[name]))
                        content = msg.artifact
                        trace["tool_runs"].append(content["run_id"])
                        if content.get("ok"):  # a failed call may succeed later (e.g. once an input exists)
                            self.done[key] = content
                except (json.JSONDecodeError, TypeError) as e:
                    content = {"ok": False, "error": f"bad arguments: {e}"}
            else:
                content = {"ok": False, "error": f"unknown tool {name}; available: {list(self.lc_tools)} + submit"}
            text = json.dumps(content, default=str, separators=(",", ":"))
            if len(text) > MAX_TOOL_RESULT_CHARS:
                text = text[:MAX_TOOL_RESULT_CHARS] + '..."[truncated]'
            reply(tc, k, text)
        return out
