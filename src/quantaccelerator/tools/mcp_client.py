"""Registry tools consumed over MCP (QA_TOOLS=mcp): one persistent stdio session to quantaccelerator.mcp_server per
process, tools loaded with langchain-mcp-adapters, and synchronous LangChain tools on top for the agent loop.

The server writes the same ledger (QA_REGISTRY_DB = this process's registry.DB); `set_session` passes the session,
run directory, agent, graph node execution and signed rule, so its rows are attributed exactly like in-process ones.
Tools the server does not serve (in-process state) stay in process: see mcp_server.IN_PROCESS_ONLY.
"""
import asyncio
import atexit
import json
import os
import sys
import threading
from contextlib import AsyncExitStack

from langchain_core.tools import StructuredTool

from quantaccelerator.tools import registry

_BOX = None


class MCPToolbox:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="qa-mcp").start()
        self._run(self._open(), timeout=300)
        atexit.register(self.close)

    def _run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _open(self):
        from langchain_mcp_adapters.tools import load_mcp_tools
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = {**os.environ, "QA_REGISTRY_DB": str(registry.DB)}
        params = StdioServerParameters(command=sys.executable, args=["-m", "quantaccelerator.mcp_server"], env=env)
        self.stack = AsyncExitStack()
        read, write = await self.stack.enter_async_context(stdio_client(params))
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self.tools = {t.name: t for t in await load_mcp_tools(self.session)}

    def call(self, name: str, args: dict) -> dict:
        out = self._run(self.tools[name].ainvoke(args))
        if isinstance(out, tuple):   # (content, artifact)
            out = out[0]
        if isinstance(out, list):    # content blocks
            out = "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in out)
        return json.loads(out)

    def set_session(self) -> dict:
        c = registry.CTX
        rule = c.reference_rule.model_dump_json() if c.reference_rule is not None else None
        return self.call("set_session", {"session_id": c.session_id or "", "agent": c.agent, "node_exec": c.node_exec,
                                         "run_dir": str(c.run_dir) if c.run_dir else None, "reference_rule": rule})

    def close(self):
        if self.loop.is_running():
            try:
                self._run(self.stack.aclose(), timeout=30)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)


def toolbox() -> MCPToolbox:
    global _BOX
    if _BOX is None or str(registry.DB) != _BOX.db:
        if _BOX is not None:
            _BOX.close()
        _BOX = MCPToolbox()
        _BOX.db = str(registry.DB)
    return _BOX


def as_mcp_tool(spec: registry.ToolSpec, box: MCPToolbox) -> StructuredTool:
    """The registry tool `spec`, run by the MCP server (same name, schema and {run_id, ok, result} artifact)."""
    def call(**kwargs):
        r = box.call(spec.name, kwargs)
        return json.dumps(r, default=str, separators=(",", ":")), r

    return StructuredTool.from_function(func=call, name=spec.name, description=spec.description,
                                        args_schema=spec.parameters, response_format="content_and_artifact")
