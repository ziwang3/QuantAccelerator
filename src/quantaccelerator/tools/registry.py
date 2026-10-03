"""Experiment registry + trial ledger (SQLite) and the @tool decorator.

Every tool call, whether made by an agent, the orchestrator or the scorer, is recorded with a run_id, args hash,
input sha256s, code version and the path of its JSON output. Agents may only cite numbers that come
from a run_id recorded here (guardrail 2).
"""
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantaccelerator.paths import INTERIM_MANIFEST, RAW_MANIFEST, ROOT, RUNS

DB = RUNS / "registry.sqlite"


@dataclasses.dataclass
class Context:
    """Mutable run context set by the orchestrator (one demo run = one session)."""
    session_id: str | None = None
    run_dir: Path | None = None
    agent: str | None = None
    reference_rule: Any = None  # AvailabilityRule from the signed PITContract
    current_run_id: str | None = None  # run_id of the tool call in progress (figure tools name their files by it)
    listener: Callable | None = None   # called after every tool run: (name, run_id, args, result); e.g. notebook progress
    findings: dict = dataclasses.field(default_factory=dict)  # items recorded by the research agent's record_* tools


CTX = Context()


@dataclasses.dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


TOOLS: dict[str, ToolSpec] = {}


def set_session(session_id: str | None, run_dir: Path | None = None, reference_rule=None):
    CTX.session_id, CTX.run_dir, CTX.reference_rule, CTX.agent = session_id, run_dir, reference_rule, None
    CTX.findings = {}


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src" / "quantaccelerator").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def _manifest_hashes() -> dict[str, str]:
    out = {}
    for mf, prefix in [(RAW_MANIFEST, ROOT / "Dataset"), (INTERIM_MANIFEST, ROOT)]:
        if mf.exists():
            for line in mf.read_text().splitlines():
                rec = json.loads(line)
                out[str((prefix / rec["path"]).resolve())] = rec["sha256"]
    return out


def file_sha256(path) -> str:
    p = str(Path(path).resolve())
    if p in _manifest_hashes():
        return _manifest_hashes()[p]
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _project_path(p) -> str:
    """Project-relative path as written (data/ may be a symlink in a worktree), else relative to the resolved root."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS runs (
        ledger_index INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE, session_id TEXT, agent TEXT,
        tool TEXT, args_json TEXT, args_hash TEXT, input_sha256s TEXT, code_version TEXT, output_path TEXT,
        ok INTEGER, error TEXT, seconds REAL, created_utc TEXT)""")
    return c


def to_jsonable(x):
    """Recursively convert numpy/pandas scalars and containers to plain JSON types."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 6)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, dt.datetime, dt.date)):
        return None if pd.isna(x) else x.isoformat()
    if x is pd.NaT:
        return None
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, Path):
        return str(x)
    return x


def inline_refs(schema: dict) -> dict:
    """JSON schema with $refs inlined and titles dropped (easier for tool-calling models)."""
    import copy
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def tool(name: str, description: str, parameters: dict, inputs: Callable[..., list] | None = None):
    """Register a function as a logged tool. The wrapper returns {run_id, ok, result} and never raises."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(**kwargs):
            run_id = "r" + uuid.uuid4().hex[:10]
            t0 = time.time()
            CTX.current_run_id = run_id
            try:
                result, ok, err = to_jsonable(fn(**kwargs)), True, None
            except Exception as e:  # tools report errors to the caller instead of crashing the agent loop
                err = f"{type(e).__name__}: {e}"
                result, ok = {"error": err}, False
            secs = time.time() - t0
            try:
                in_paths = inputs(**kwargs) if inputs else []
                shas = [{"path": _project_path(p), "sha256": file_sha256(p)} for p in in_paths if Path(p).exists()]
            except Exception:
                shas = []
            out_dir = (CTX.run_dir or RUNS / "_adhoc") / "tool_outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{run_id}.json"
            args = to_jsonable(kwargs)
            out_path.write_text(json.dumps({"run_id": run_id, "tool": name, "args": args, "result": result},
                                           indent=1, default=str))
            args_json = json.dumps(args, sort_keys=True, default=str)
            with _conn() as c:
                c.execute("INSERT INTO runs (run_id, session_id, agent, tool, args_json, args_hash, input_sha256s,"
                          " code_version, output_path, ok, error, seconds, created_utc)"
                          " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (run_id, CTX.session_id, CTX.agent, name, args_json,
                           hashlib.sha256(args_json.encode()).hexdigest()[:16], json.dumps(shas), code_version(),
                           str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) else str(out_path),
                           int(ok), err, round(secs, 3),
                           dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
            if CTX.listener is not None:
                try:  # progress display must never break a tool run
                    CTX.listener(name, run_id, args, result)
                except Exception:
                    pass
            return {"run_id": run_id, "ok": ok, "result": result}

        wrapper.raw = fn
        TOOLS[name] = ToolSpec(name, description, parameters, wrapper)
        return wrapper
    return deco


def run_exists(run_id: str, session_id: str | None = None) -> bool:
    q, a = "SELECT 1 FROM runs WHERE run_id=?", [run_id]
    if session_id:
        q, a = q + " AND session_id=?", a + [session_id]
    with _conn() as c:
        return c.execute(q, a).fetchone() is not None


def session_runs(session_id: str) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM runs WHERE session_id=? ORDER BY ledger_index", c, params=[session_id])


def ledger_count() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def get_tools(names: list[str]) -> list[ToolSpec]:
    import quantaccelerator.tools  # noqa: F401  (registers all tools)
    return [TOOLS[n] for n in names]


def get_run(run_id: str) -> dict | None:
    """Tool name and stored JSON output of a registered run (None if unknown)."""
    with _conn() as c:
        row = c.execute("SELECT tool, output_path FROM runs WHERE run_id=?", [run_id]).fetchone()
    if not row:
        return None
    p = Path(row[1]) if Path(row[1]).is_absolute() else ROOT / row[1]
    return {"tool": row[0], **json.loads(p.read_text())}
