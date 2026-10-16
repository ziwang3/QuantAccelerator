"""Registry tools as LangChain tools. The registry stays the single source of truth: each adapter calls the logged
@tool wrapper, so the run_id, ledger row, tool_outputs file and listener are recorded as for any tool call."""
import json

from langchain_core.tools import StructuredTool

from quantaccelerator.tools import registry


def as_langchain_tool(spec: registry.ToolSpec) -> StructuredTool:
    """content = the result as compact JSON; artifact = the {run_id, ok, result} dict itself."""
    def call(**kwargs):
        r = spec.fn(**kwargs)
        return json.dumps(r, default=str, separators=(",", ":")), r

    return StructuredTool.from_function(func=call, name=spec.name, description=spec.description,
                                        args_schema=spec.parameters, response_format="content_and_artifact")


def langchain_tools(names: list[str]) -> list[StructuredTool]:
    return [as_langchain_tool(s) for s in registry.get_tools(names)]
