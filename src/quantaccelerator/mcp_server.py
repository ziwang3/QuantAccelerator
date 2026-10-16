"""MCP server (stdio) over the tool registry: python -m quantaccelerator.mcp_server

Each call goes through the registry's logged @tool wrapper, so the ledger row, run_id and tool_outputs file are those
of an in-process call; who called is set with the `set_session` tool (or QA_SESSION_ID / QA_RUN_DIR / QA_AGENT at
start). QA_REGISTRY_DB points the server at another ledger file (tests).

Served: the tools that need nothing from the caller's process but the session context. Not served: the tools that
read or write in-process state (the research notebook in registry.CTX.findings, the session panel, the returns vault:
modules features, predictive, critic, findings); those only run inside the orchestrator's process.
"""
import json
import os
from pathlib import Path

import mcp.types as types
from mcp.server.lowlevel import Server

from quantaccelerator.tools import registry

IN_PROCESS_ONLY = {"quantaccelerator.tools.features", "quantaccelerator.tools.predictive",
                   "quantaccelerator.tools.critic", "quantaccelerator.tools.findings"}

SET_SESSION = types.Tool(
    name="set_session", description="Attribute the following tool calls: session, run directory, calling agent, graph "
                                    "node execution and the signed availability rule (JSON), as the orchestrator has them.",
    inputSchema={"type": "object", "properties": {
        "session_id": {"type": "string"}, "run_dir": {"type": "string"}, "agent": {"type": "string"},
        "node_exec": {"type": ["string", "null"]}, "reference_rule": {"type": ["string", "null"]}},
        "required": ["session_id"]})


def served() -> dict[str, registry.ToolSpec]:
    import quantaccelerator.tools  # noqa: F401  (registers every tool)
    return {n: s for n, s in registry.TOOLS.items() if s.fn.__module__ not in IN_PROCESS_ONLY}


def set_session(session_id: str, run_dir: str | None = None, agent: str | None = None, node_exec: str | None = None,
                reference_rule: str | None = None) -> dict:
    from quantaccelerator.state.schemas import AvailabilityRule
    rule = AvailabilityRule.model_validate_json(reference_rule) if reference_rule else None
    registry.set_session(session_id, Path(run_dir) if run_dir else None, rule)
    registry.CTX.agent, registry.CTX.node_exec = agent, node_exec
    return {"ok": True, "session_id": session_id, "agent": agent}


def build_server() -> Server:
    if os.environ.get("QA_REGISTRY_DB"):
        registry.DB = Path(os.environ["QA_REGISTRY_DB"])
    if os.environ.get("QA_SESSION_ID"):
        set_session(os.environ["QA_SESSION_ID"], os.environ.get("QA_RUN_DIR"), os.environ.get("QA_AGENT"))
    tools = served()
    server = Server("quantaccelerator")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [SET_SESSION] + [types.Tool(name=s.name, description=s.description, inputSchema=s.parameters)
                                for s in tools.values()]

    @server.call_tool(validate_input=False)   # bad arguments are reported by the tool, as in process
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "set_session":
            out = set_session(**arguments)
        elif name in tools:
            out = tools[name].fn(**(arguments or {}))
        else:
            out = {"ok": False, "error": f"tool {name} is not served over MCP"}
        return [types.TextContent(type="text", text=json.dumps(out, default=str, separators=(",", ":")))]

    return server


async def main() -> None:
    from mcp.server.stdio import stdio_server
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import anyio
    anyio.run(main)
