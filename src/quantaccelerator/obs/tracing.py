"""OpenTelemetry tracing to a local Arize Phoenix (QA_TRACE=phoenix).

One trace per run: the LangGraph graph, its nodes, each create_agent loop inside them, and every model and tool call
(OpenInference instrumentors for LangChain and the OpenAI SDK).
Spans go to PHOENIX_COLLECTOR_ENDPOINT (default http://127.0.0.1:6006/v1/traces): a Phoenix server on the same
machine, so the traces (which contain the data the agents read) stay local. Tracing never breaks a run: without the packages or a server it is off.
"""
import os

_ON = None


def init(project: str = "quantaccelerator") -> bool:
    global _ON
    if _ON is not None:
        return _ON
    _ON = False
    if os.environ.get("QA_TRACE") != "phoenix":
        return False
    try:
        from phoenix.otel import register
        register(project_name=os.environ.get("PHOENIX_PROJECT_NAME", project), auto_instrument=True, batch=True,
                 endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"))
        _ON = True
    except Exception as e:   # tracing is an aid, not a dependency
        print(f"tracing off: {type(e).__name__}: {e}")
    return _ON
