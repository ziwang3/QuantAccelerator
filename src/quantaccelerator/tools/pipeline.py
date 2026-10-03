"""Pipeline tools: read_source, run_pipeline, test_patch. Scripts run in a subprocess and write QA_OUT."""
import difflib
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pandas as pd

from quantaccelerator.paths import NOTEBOOKS, ROOT, RUNS, rel
from quantaccelerator.datasets import active
from quantaccelerator.tools.registry import CTX, tool


def _resolve(path: str) -> Path:
    p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not (p.is_relative_to(NOTEBOOKS.resolve()) or p.is_relative_to(RUNS.resolve())):
        raise PermissionError(f"{path}: only scripts under notebooks/ or runs/ may be read or run")
    return p


def _work_dir(sub: str) -> Path:
    d = (CTX.run_dir or RUNS / "_adhoc") / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def execute(script: Path, timeout: int = 600) -> dict:
    out = _work_dir("pipeline_outputs") / f"{script.stem}_{uuid.uuid4().hex[:6]}.parquet"
    env = {**os.environ, "QA_OUT": str(out)}
    p = subprocess.run([sys.executable, str(script)], env=env, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"pipeline failed (exit {p.returncode}): {p.stderr[-1500:]}")
    df = pd.read_parquet(out)
    return {"output_path": rel(out), "n_rows": len(df), "columns": list(df.columns),
            "head": df.head(3).astype(str).to_dict(orient="records"), "stdout_tail": p.stdout[-500:]}


@tool("read_source", "Read a pipeline script with 1-based line numbers.",
      {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
      inputs=lambda path: [_resolve(path)])
def read_source(path: str):
    lines = _resolve(path).read_text().splitlines()
    return "\n".join(f"{i:4d}| {l}" for i, l in enumerate(lines, 1))


@tool("run_pipeline", "Run a pipeline script in a subprocess; returns output_path, row count, columns and head.",
      {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
      inputs=lambda path: [_resolve(path)])
def run_pipeline(path: str):
    return execute(_resolve(path))


def apply_edits(src: str, edits: list[dict]) -> str:
    lines = src.splitlines()
    for e in sorted(edits, key=lambda e: -int(e["line"])):
        i = int(e["line"])
        if not 1 <= i <= len(lines):
            raise ValueError(f"line {i} out of range 1..{len(lines)}")
        # keep the replaced line's indentation (models often add or drop leading spaces); relative indents survive
        indent = lines[i - 1][:len(lines[i - 1]) - len(lines[i - 1].lstrip())]
        block = textwrap.dedent(e["new_code"]).splitlines() or [""]
        lines[i - 1:i] = [indent + ln if ln.strip() else ln for ln in block]
    return "\n".join(lines) + "\n"


@tool("test_patch", "Apply line replacements to a copy of a pipeline script, run it, and measure look-ahead of the "
      "patched output against the signed PIT contract. Returns the diff, run summary and look-ahead stats.",
      {"type": "object", "properties": {
          "path": {"type": "string"},
          "edits": {"type": "array", "items": {"type": "object", "properties": {
              "line": {"type": "integer"}, "new_code": {"type": "string"}}, "required": ["line", "new_code"]}}},
       "required": ["path", "edits"]},
      inputs=lambda path, edits: [_resolve(path)])
def test_patch(path: str, edits: list[dict]):
    if CTX.reference_rule is None:
        raise RuntimeError("no signed PIT contract in this session")
    src_path = _resolve(path)
    new_src = apply_edits(src_path.read_text(), edits)
    patched = _work_dir("patches") / f"{src_path.stem}_patched_{uuid.uuid4().hex[:6]}.py"
    patched.write_text(new_src)
    diff = "".join(difflib.unified_diff(src_path.read_text().splitlines(True), new_src.splitlines(True),
                                        fromfile=src_path.name, tofile=patched.name))
    run = execute(patched)
    stats = active().measure(pd.read_parquet(ROOT / run["output_path"]), CTX.reference_rule)
    return {"patched_path": rel(patched), "diff": diff, "run": run, "lookahead": stats}
