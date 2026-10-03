"""Cleaning recipes: validated, deterministic, previewable, applied only after the human approves (gate G0).

A recipe is a list of CleaningStep objects from a closed set of operations (drop exact-key duplicates, drop rows,
flag rows, add missing-value indicators). Raw and interim data are never modified: an approved recipe writes a derived
copy of each affected table to data/derived/<dataset_id>/<recipe hash>/, with the recipe and a manifest entry.
"""
import hashlib
import json

import pandas as pd

from quantaccelerator.datasets import active
from quantaccelerator.ingest import manifest
from quantaccelerator.paths import ROOT
from quantaccelerator.state.schemas import CleaningRecipe, CleaningStep, Predicate
from quantaccelerator.tools.data import TABLES, load_table, table_path
from quantaccelerator.tools.registry import inline_refs, tool

DERIVED = ROOT / "data" / "derived"


def _mask(df: pd.DataFrame, where: list[Predicate]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for p in where:
        c = df[p.column]
        v = p.value
        if p.op in (">col", "<col"):
            m &= (c > df[str(v)]) if p.op == ">col" else (c < df[str(v)])
            continue
        if p.op in ("==", "!=", "<", "<=", ">", ">=") and isinstance(v, str) and v in df.columns:
            v = df[v]  # a comparison with another column of the same table, written naturally ("ShortVolume > TotalVolume")
        if p.op in ("before", "on_or_after"):
            c, v = pd.to_datetime(c), pd.Timestamp(str(v))
        elif pd.api.types.is_numeric_dtype(c) and isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                raise ValueError(f"predicate {p.column} {p.op} {v!r}: {v!r} is neither a number nor a column of this "
                                 f"table (columns: {list(df.columns)})") from None
        m &= {"==": lambda: c == v, "!=": lambda: c != v, "<": lambda: c < v, "<=": lambda: c <= v,
              ">": lambda: c > v, ">=": lambda: c >= v, "isna": lambda: c.isna(), "notna": lambda: c.notna(),
              "isin": lambda: c.isin(v if isinstance(v, list) else [v]),
              "notin": lambda: ~c.isin(v if isinstance(v, list) else [v]),
              "before": lambda: c < v, "on_or_after": lambda: c >= v}[p.op]()
    return m


def validate_step(s: CleaningStep) -> list[str]:
    errs = []
    if s.table not in TABLES:
        return [f"cleaning step on unknown table {s.table!r} (tables: {sorted(TABLES)})"]
    cols = set(load_table(s.table).columns)
    bad = [c for c in s.columns + [p.column for p in s.where] + [str(p.value) for p in s.where
                                                                   if p.op in (">col", "<col")] if c not in cols]
    if bad:
        errs.append(f"cleaning step {s.op} on {s.table}: no column(s) {bad}")
    if s.op == "drop_duplicates" and not s.columns:
        errs.append("drop_duplicates needs the key columns in `columns`")
    if s.op in ("drop_rows", "flag_rows") and not s.where:
        errs.append(f"{s.op} needs at least one predicate in `where`")
    if s.op in ("flag_rows", "missing_indicator") and not s.flag_name:
        errs.append(f"{s.op} needs `flag_name` for the new column")
    if s.op == "missing_indicator" and not s.columns:
        errs.append("missing_indicator needs the fields in `columns`")
    for p in s.where:
        if p.op in ("isin", "notin") and not isinstance(p.value, list):
            errs.append(f"predicate {p.column} {p.op} needs a list value")
        if p.op not in ("isna", "notna") and p.value is None:
            errs.append(f"predicate {p.column} {p.op} needs a value")
    return errs


def step_mask(df: pd.DataFrame, s: CleaningStep) -> pd.Series:
    """Rows the step would drop or flag."""
    if s.op == "drop_duplicates":
        return df.duplicated(s.columns, keep="first")
    if s.op in ("drop_rows", "flag_rows"):
        return _mask(df, s.where)
    return df[s.columns].isna().any(axis=1)


def step_rows(s: CleaningStep) -> int:
    return int(step_mask(load_table(s.table), s).sum())


def apply_step(df: pd.DataFrame, s: CleaningStep) -> tuple[pd.DataFrame, dict]:
    n = len(df)
    if s.op == "drop_duplicates":
        out = df.drop_duplicates(s.columns, keep="first")
        return out, {"rows_removed": n - len(out)}
    if s.op == "drop_rows":
        m = _mask(df, s.where)
        return df[~m], {"rows_removed": int(m.sum())}
    if s.op == "flag_rows":
        m = _mask(df, s.where)
        return df.assign(**{s.flag_name: m.values}), {"rows_flagged": int(m.sum())}
    flag = df[s.columns].isna().any(axis=1)
    return df.assign(**{s.flag_name: flag.values}), {"rows_flagged": int(flag.sum())}


def recipe_hash(recipe: CleaningRecipe) -> str:
    return hashlib.sha256(json.dumps([s.model_dump(exclude={"reason", "evidence"}) for s in recipe.steps],
                                     sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_recipe(recipe: CleaningRecipe, tables: dict[str, pd.DataFrame] | None = None
               ) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Apply every step in order; returns the changed tables and per-step effects (rows before/after, counts)."""
    out, effects = {}, []
    for i, s in enumerate(recipe.steps):
        df = out.get(s.table)
        if df is None:
            df = (tables or {}).get(s.table)
            df = load_table(s.table) if df is None else df
        before = len(df)
        df, eff = apply_step(df, s)
        out[s.table] = df
        effects.append({"step": i, "op": s.op, "table": s.table, "rows_before": before, "rows_after": len(df),
                        "share_of_rows": round((eff.get("rows_removed") or eff.get("rows_flagged") or 0)
                                               / max(before, 1), 6), **eff})
    return out, effects


def apply_recipe(recipe: CleaningRecipe) -> dict:
    """Write the derived tables of an approved recipe (never touches the source tables)."""
    h = recipe_hash(recipe)
    d = DERIVED / active().dataset_id / h
    d.mkdir(parents=True, exist_ok=True)
    out, effects = run_recipe(recipe)
    paths = {}
    for t, df in out.items():
        p = d / f"{t}.parquet"
        df.to_parquet(p, index=False)
        manifest.record(p, [table_path(t)], "quantaccelerator.tools.clean", rows=len(df), recipe_hash=h)
        paths[t] = str(p.relative_to(ROOT))
    (d / "recipe.json").write_text(recipe.model_dump_json(indent=1))
    return {"recipe_hash": h, "tables": paths, "effects": effects}


@tool("preview_cleaning", "Preview what a proposed cleaning step would do, without changing any data: rows it "
      "would drop or flag, their share, and a few example rows. Operations: drop_duplicates (on a key), drop_rows, "
      "flag_rows (adds a boolean column; preferred over dropping), missing_indicator. Use it before proposing a step, "
      "and cite its run_id as evidence for the step.",
      {"type": "object", "properties": {"step": inline_refs(CleaningStep.model_json_schema())}, "required": ["step"]},
      inputs=lambda step: [table_path(step["table"])] if step.get("table") in TABLES else [])
def preview_cleaning(step: dict):
    s = CleaningStep.model_validate({"reason": "preview", "evidence": [], **step})
    errs = validate_step(s)
    if errs:
        raise ValueError("; ".join(errs))
    df = load_table(s.table)
    m = step_mask(df, s)
    ex = df[m].head(5).astype(str).to_dict("records")
    return {"op": s.op, "table": s.table, "rows": int(len(df)), "rows_affected": int(m.sum()),
            "share_affected": round(float(m.mean()), 6), "examples": ex,
            "effect": "removed" if s.op in ("drop_duplicates", "drop_rows") else "flagged (kept)"}
