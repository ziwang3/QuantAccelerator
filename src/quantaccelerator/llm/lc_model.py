"""LangChain chat model over the project's LLMClient (the agents' model layer).

Every request goes through LLMClient.chat: OpenAI-format messages, the agent's tool schemas exactly as the agent
defines them (looked up by name), the Hermes fallback, llm_calls.jsonl logging and the Budget; a ScriptedClient drives
it offline. Raw tool-call argument strings travel in additional_kwargs["qa_tool_calls"],
so a malformed or double-encoded call reaches the guards exactly as the server sent it.
"""
import json
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

RAW_CALLS = "qa_tool_calls"


def _args(raw: str) -> dict:
    """Best-effort dict for LangChain's tool_calls (the guards parse the raw string themselves)."""
    try:
        v = json.loads(raw) if raw and raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, dict) else {}


def to_openai(messages: list[BaseMessage]) -> list[dict]:
    """LangChain messages -> OpenAI-format dicts (inverse of to_lc)."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif isinstance(m, AIMessage):
            d = {"role": "assistant", "content": m.content or ""}
            raw = m.additional_kwargs.get(RAW_CALLS)
            if raw is None and m.tool_calls:
                raw = [{"id": t["id"], "name": t["name"], "arguments": json.dumps(t["args"])} for t in m.tool_calls]
            if raw:
                d["tool_calls"] = [{"id": t["id"], "type": "function",
                                    "function": {"name": t["name"], "arguments": t["arguments"]}} for t in raw]
            out.append(d)
        else:
            raise TypeError(f"unsupported message type {type(m).__name__}")
    return out


def ai_message(content: str | None, raw_calls: list[dict], msg_id: str | None = None, **meta) -> AIMessage:
    return AIMessage(content=content or "", id=msg_id or f"ai-{uuid.uuid4().hex[:12]}",
                     tool_calls=[{"name": t["name"], "args": _args(t["arguments"]), "id": t["id"], "type": "tool_call"}
                                 for t in raw_calls],
                     additional_kwargs={RAW_CALLS: raw_calls} if raw_calls else {}, response_metadata=meta)


def to_lc(messages: list[dict]) -> list[BaseMessage]:
    """OpenAI-format dicts -> LangChain messages (round-trips through to_openai unchanged)."""
    out = []
    for d in messages:
        role = d["role"]
        if role == "system":
            out.append(SystemMessage(content=d["content"]))
        elif role == "user":
            out.append(HumanMessage(content=d["content"]))
        elif role == "tool":
            out.append(ToolMessage(content=d["content"], tool_call_id=d["tool_call_id"]))
        elif role == "assistant":
            raw = [{"id": t["id"], "name": t["function"]["name"], "arguments": t["function"]["arguments"]}
                   for t in d.get("tool_calls") or []]
            out.append(ai_message(d.get("content"), raw))
        else:
            raise ValueError(f"unknown role {role}")
    return out


class QAChatModel(BaseChatModel):
    """BaseChatModel whose every call is LLMClient.chat(messages, schemas, agent, max_tokens)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any                      # quantaccelerator.llm.client.LLMClient (or ScriptedClient)
    agent_name: str = ""
    schemas: dict[str, dict] = {}    # tool name -> the agent's exact OpenAI tool schema

    @property
    def _llm_type(self) -> str:
        return "quantaccelerator-llmclient"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [t if isinstance(t, str) else getattr(t, "name", None) or t["function"]["name"] for t in tools]
        return self.bind(tools=[self.schemas[n] for n in names], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, tools=None, max_tokens=None, **kwargs) -> ChatResult:
        resp = self.client.chat(to_openai(messages), tools or None, agent=self.agent_name, max_tokens=max_tokens)
        raw = [{"id": t.id, "name": t.name, "arguments": t.arguments} for t in resp.tool_calls]
        msg = ai_message(resp.content, raw, finish_reason=resp.finish_reason, prompt_tokens=resp.prompt_tokens,
                         completion_tokens=resp.completion_tokens, model_name=self.client.model)
        msg.usage_metadata = {"input_tokens": resp.prompt_tokens, "output_tokens": resp.completion_tokens,
                              "total_tokens": resp.prompt_tokens + resp.completion_tokens}
        return ChatResult(generations=[ChatGeneration(message=msg)])
