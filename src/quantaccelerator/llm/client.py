"""Thin OpenAI-compatible chat client with tool calling, per-call logging and budget caps.

Every call appends one line to runs/<run_id>/llm_calls.jsonl: agent, model, prompt hash, tokens in/out, latency,
number of tool calls. ScriptedClient replays canned responses so agent loops run offline (tests/CI).
"""
import dataclasses
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class BudgetExceeded(RuntimeError):
    pass


@dataclasses.dataclass
class Budget:
    max_prompt_tokens: int = 600_000
    max_completion_tokens: int = 150_000
    max_calls: int = 120
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def check(self):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget exhausted ({self.max_calls})")
        if self.prompt_tokens >= self.max_prompt_tokens or self.completion_tokens >= self.max_completion_tokens:
            raise BudgetExceeded(f"token budget exhausted ({self.prompt_tokens} in / {self.completion_tokens} out)")


@dataclasses.dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclasses.dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None

    def as_message(self) -> dict:
        m = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            m["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}
                               for t in self.tool_calls]
        return m


def parse_hermes_tool_calls(content: str | None) -> list[ToolCall]:
    """Recover tool calls that the server left as raw Hermes text (<tool_call>{json}</tool_call>).

    vLLM's hermes parser drops *all* calls of a turn when one block fails to parse; blocks are parsed
    individually here and unparseable ones are skipped (the agent is then told nothing ran for them).
    """
    calls = []
    for block in HERMES_CALL.findall(content or ""):
        try:
            obj = json.loads(block)
            args = obj.get("arguments", {})
            calls.append(ToolCall(f"hermes_{uuid.uuid4().hex[:8]}", obj["name"],
                                  args if isinstance(args, str) else json.dumps(args)))
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue
    return calls


def prompt_hash(messages: list[dict], tools: list[dict] | None) -> str:
    return hashlib.sha256(json.dumps([messages, tools], sort_keys=True, default=str).encode()).hexdigest()[:16]


class LLMClient:
    def __init__(self, log_path: Path | None = None, budget: Budget | None = None, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0,
                 seed: int | None = 0, max_tokens: int = 8192):
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "none")
        self.log_path, self.budget = log_path, budget or Budget()
        self.temperature, self.seed, self.max_tokens = temperature, seed, max_tokens
        self._client = None

    def _complete(self, messages, tools, max_tokens=None) -> LLMResponse:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600)
        kw = {"tools": tools, "tool_choice": "auto"} if tools else {}
        r = self._client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature,
                                                 seed=self.seed, max_tokens=max_tokens or self.max_tokens, **kw)
        m = r.choices[0].message
        calls = [ToolCall(t.id, t.function.name, t.function.arguments or "{}") for t in (m.tool_calls or [])]
        content = m.content
        if not calls and content and "<tool_call>" in content:
            calls = parse_hermes_tool_calls(content)
            if calls:
                content = ""  # the calls are replayed as structured tool_calls in the history
        return LLMResponse(content, calls, r.usage.prompt_tokens, r.usage.completion_tokens,
                           r.choices[0].finish_reason)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, agent: str = "",
             max_tokens: int | None = None) -> LLMResponse:
        self.budget.check()
        t0 = time.time()
        resp = self._complete(messages, tools, max_tokens)
        self.budget.calls += 1
        self.budget.prompt_tokens += resp.prompt_tokens
        self.budget.completion_tokens += resp.completion_tokens
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": agent, "model": self.model,
                                    "prompt_hash": prompt_hash(messages, tools), "tokens_in": resp.prompt_tokens,
                                    "tokens_out": resp.completion_tokens, "latency_s": round(time.time() - t0, 2),
                                    "n_tool_calls": len(resp.tool_calls), "finish_reason": resp.finish_reason,
                                    "tools_called": [t.name for t in resp.tool_calls]}) + "\n")
        return resp


class ScriptedClient(LLMClient):
    """Replays canned responses. Each item: {"content": str} or {"tool_calls": [{"name", "arguments": dict|str}]}.

    An item may also be a callable(messages) -> item, so a script can react to earlier tool results
    (for example, to cite a run_id that only exists at run time).
    """

    def __init__(self, script: list, **kw):
        super().__init__(model="scripted", **kw)
        self.script, self.i = list(script), 0

    def _complete(self, messages, tools, max_tokens=None) -> LLMResponse:
        if self.i >= len(self.script):
            raise RuntimeError("ScriptedClient ran out of responses")
        item = self.script[self.i]
        self.i += 1
        if callable(item):
            item = item(messages)
        calls = [ToolCall(f"call_{self.i}_{k}", c["name"],
                          c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"]))
                 for k, c in enumerate(item.get("tool_calls", []))]
        n_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return LLMResponse(item.get("content"), calls, n_in, 50, item.get("finish_reason", "stop"))
