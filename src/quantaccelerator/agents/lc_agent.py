"""An agent run: a LangChain v1 `create_agent` ReAct loop with the agent's guards as middleware (agents/middleware.py).

run(agent, client, task, context) returns (validated output model, trace) or raises AgentFailure carrying the trace.
The prompts, tools, output model, limits and extra_validate come from the Agent subclass (agents/base.py), whose run()
calls this. QA_TOOLS=mcp runs the agent's tools through the MCP server (quantaccelerator.mcp_server) where they can
run out of process.
"""
import json
import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from quantaccelerator.agents.base import COMMON_RULES, AgentFailure, inline_schema
from quantaccelerator.agents.middleware import NativeGuards
from quantaccelerator.llm.client import prompt_hash
from quantaccelerator.llm.lc_model import QAChatModel
from quantaccelerator.tools import registry
from quantaccelerator.tools.lc_adapter import as_langchain_tool


def _submit_tool(agent, schema: dict) -> StructuredTool:
    def never(**kw):   # the guards validate and answer every submit; this body is not reached
        raise RuntimeError("submit is handled by NativeGuards")
    return StructuredTool.from_function(func=never, name="submit", description=schema["function"]["description"],
                                        args_schema=schema["function"]["parameters"])


def build(agent, client, trace: dict):
    """The compiled create_agent graph and its guards for one run of `agent`."""
    specs = registry.get_tools(agent.tool_names)
    submit = {"type": "function", "function": {
        "name": "submit", "description": f"Submit the final {agent.output_model.__name__}.",
        "parameters": inline_schema(agent.output_model)}}
    schemas = [t.openai_schema() for t in specs] + [submit]
    system = f"{agent.role_prompt}\n\n{COMMON_RULES}"
    tools = {s.name: as_langchain_tool(s) for s in specs}
    if os.environ.get("QA_TOOLS") == "mcp":   # served tools run in the MCP server; in-process-state tools stay here
        from quantaccelerator.tools.mcp_client import as_mcp_tool, toolbox
        box = toolbox()
        box.set_session()
        tools = {n: as_mcp_tool(s, box) if n in box.tools else tools[n] for n, s in zip(tools, specs)}
    guards = NativeGuards(agent, system, schemas, tools, trace)
    model = QAChatModel(client=client, agent_name=agent.name, schemas={s["function"]["name"]: s for s in schemas})
    graph = create_agent(model, [*tools.values(), _submit_tool(agent, submit)], system_prompt=system,
                         middleware=[guards], name=agent.name)
    graph.checkpointer = False   # never inherit an orchestration graph's checkpointer: an agent run is one node
    return graph, guards, schemas, system


def run(agent, client, task: str, context: dict | None = None):
    trace = {"agent": agent.name}
    try:
        return _run(agent, client, task, context, trace)
    except AgentFailure:
        raise
    except Exception as e:
        raise AgentFailure(f"{agent.name}: {type(e).__name__}: {e}", trace) from e


def _run(agent, client, task, context, trace):
    registry.CTX.agent = agent.name   # before build: an MCP session is attributed to this agent
    graph, guards, schemas, system = build(agent, client, trace)
    user = task + ("\n\nContext (JSON):\n" + json.dumps(context, default=str) if context else "")
    trace.update({"prompt_hash": prompt_hash([{"role": "system", "content": system}], schemas), "tool_runs": [],
                  "tool_calls": 0, "submit_failures": 0, "submit_errors": [], "nudges": 0, "steps": 0,
                  "messages": guards.view()})
    try:
        graph.invoke({"messages": [HumanMessage(user, id="qa-task")]},
                     {"recursion_limit": 8 * agent.max_steps + 20, "run_name": agent.name,
                      "metadata": {"qa_agent": agent.name, "qa_session": registry.CTX.session_id}})
    finally:
        trace["messages"] = guards.view()
    if guards.failure is not None:
        raise guards.failure
    if guards.final is None:
        raise AgentFailure(f"{agent.name}: the agent loop ended without a valid submission", trace)
    return guards.final, trace
