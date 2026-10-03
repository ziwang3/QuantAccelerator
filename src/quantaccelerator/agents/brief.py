"""The Data Brief: what the data is, what the Data Research Agent did, what it found, what it proposes to clean, and
what it could not decide. Assembled deterministically from the run's state, transcripts and tool outputs (no LLM
writes it), as brief.json plus a self-contained brief.html and a brief.md for notebooks.
"""
import base64
import html
import json
from pathlib import Path

import pandas as pd

from quantaccelerator.state.schemas import DatasetCard
from quantaccelerator.tools.clean import run_recipe
from quantaccelerator.tools.findings import RECORD_TOOLS
from quantaccelerator.viz.events import summarize

TOOL_VERB = {"list_tables": "Listed the tables", "profile_table": "Profiled a table", "profile_time": "Profiled the "
             "time columns", "read_doc": "Read the documentation", "research_breadth": "Measured research breadth",
             "coverage_over_time": "Measured coverage over time", "key_check": "Checked the key",
             "missingness": "Checked missing data", "distribution": "Examined a distribution",
             "update_dynamics": "Examined how values update", "variance_split": "Split the variance",
             "structural_breaks": "Looked for structural breaks", "category_mix_over_time": "Tracked a coded field",
             "preview_cleaning": "Previewed a cleaning step", "look_at_figure": "Looked at a figure",
             "describe_panel": "Described the research panel", "construct_feature": "Built a feature",
             "compute_exposure": "Characterized a feature", "time_stability": "Checked stability over time",
             "compare_representations": "Compared representations"}


def steps(run_dir: Path, agents=("profiler", "overview", "data_research")) -> list[dict]:
    """Every tool call of the Data Research Agent, in order, with a one-line result and its figure."""
    out = []
    for agent in agents:
        p = run_dir / "transcripts" / f"{agent}.json"
        if not p.exists():
            continue
        msgs, pending = json.loads(p.read_text()), {}
        for m in msgs:
            if m["role"] == "assistant":
                pending = {tc["id"]: tc["function"] for tc in m.get("tool_calls") or []}
            elif m["role"] == "tool":
                fn = pending.get(m.get("tool_call_id"), {"name": "?", "arguments": "{}"})
                if fn["name"] == "submit" or fn["name"] in RECORD_TOOLS:
                    continue  # recorded findings have their own sections
                try:
                    body = json.loads(m["content"])
                    rid = body.get("run_id")
                    if not rid or not body.get("ok", True) and "result" not in body:
                        continue  # not run (paced or repeated call): not a step
                    full = json.loads((run_dir / "tool_outputs" / f"{rid}.json").read_text()) if rid else body
                    args, res = full.get("args") or json.loads(fn["arguments"] or "{}"), full.get("result")
                except (json.JSONDecodeError, FileNotFoundError, TypeError):
                    continue
                out.append({"pass": {"profiler": "semantics", "overview": "overview"}.get(agent, "profiling"),
                            "tool": fn["name"],
                            "what": TOOL_VERB.get(fn["name"], fn["name"]), "run_id": rid, "args": args,
                            "ok": bool(body.get("ok", True)), "result": summarize(fn["name"], args, res),
                            "figure": res.get("figure") if isinstance(res, dict) else None})
    return out


def table_preview(n: int = 5) -> list[dict]:
    """Every table of the dataset as a researcher would first look at it: size, columns with types, the first rows."""
    from quantaccelerator.tools.data import TABLES, load_table
    out = []
    for t in TABLES:
        df = load_table(t)
        head = df.head(n)
        out.append({"table": t, "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
                    "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                    "head": [[("" if pd.isna(v) else str(v))[:40] for v in row] for row in head.itertuples(index=False)]})
    return out


def build(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    card = DatasetCard.model_validate_json((run_dir / "state" / "dataset_card.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
    r = card.research
    cleaning = None
    if r and r.cleaning.steps:
        _, effects = run_recipe(r.cleaning)
        cleaning = {"summary": r.cleaning.summary, "steps": [
            {**s.model_dump(), "effect": e} for s, e in zip(r.cleaning.steps, effects)]}
    brief = {
        "dataset_id": card.dataset_id, "run_id": meta.get("run_id"), "model": meta.get("model"),
        "overview": card.overview.model_dump() if card.overview else None,
        "table_preview": table_preview(),
        "tables": [t.model_dump() for t in card.tables],
        "fields": [f.model_dump() for f in card.fields],
        "time_fields": card.time_fields, "entity_id_fields": card.entity_id_fields,
        "steps": steps(run_dir),
        "observations": [o.model_dump() for o in r.observations] if r else [],
        "capability": r.capability.model_dump() if r else None,
        "preprocessing": [p.model_dump() for p in r.preprocessing] if r else [],
        "cleaning": cleaning,
        "warnings": card.warnings + (r.warnings if r else []),
        "open_questions": r.open_questions if r else [],
    }
    fig_of = {s["run_id"]: s["figure"] for s in brief["steps"] if s.get("figure")}
    for o in brief["observations"]:
        o["figure"] = fig_of.get(o.get("figure_run_id"))
    out = run_dir / "brief"
    out.mkdir(exist_ok=True)
    (out / "brief.json").write_text(json.dumps(brief, indent=1, default=str))
    (out / "brief.md").write_text(to_markdown(brief))
    (out / "brief.html").write_text(to_html(brief, run_dir))
    return brief


# ---------------------------------------------------------------- rendering
def to_markdown(b: dict, fig_prefix: str = "../") -> str:
    L = [f"# Data Brief · `{b['dataset_id']}`", "",
         f"Run `{b['run_id']}` · model {b['model']} · {len(b['steps'])} tool calls", ""]
    if (o := b.get("overview")):
        L += ["## What this dataset is", "", o["summary"], "", f"- **Publisher:** {o['publisher']}", f"- **Purpose:** {o['purpose']}",
              f"- **Market:** {o['market']}", "- **Entities:** " + "; ".join(f"{x['type']} ({x['share']})" for x in o["entities"]),
              f"- **Values:** {'; '.join(o['values'])}", f"- **Frequency:** {o['frequency']}", f"- **Period:** {o['period']}",
              f"- **Typical uses:** {'; '.join(o['typical_uses'])}", f"- **Caveats:** {'; '.join(o['caveats'])}", ""]
    for t in b.get("table_preview", []):
        L += [f"**Table `{t['table']}`** · {t['n_rows']:,} rows × {t['n_columns']} columns", "",
              "| " + " | ".join(c["name"] for c in t["columns"]) + " |", "|" + "---|" * len(t["columns"])]
        L += ["| " + " | ".join(row) + " |" for row in t["head"]] + [""]
    L += ["## What the fields mean", ""]
    for t in b["tables"]:
        L.append(f"- **{t['table']}**: {t['observation_unit']} (key: {', '.join(t['primary_key'])})")
    L += ["", "| field | kind | unit | meaning | zero vs missing | not the same as |", "|---|---|---|---|---|---|"]
    for f in b["fields"]:
        L.append(f"| {f['table']}.{f['name']} | {f.get('kind') or ''} | {f.get('unit') or ''} | {f['meaning']} "
                 f"| {f.get('zero_vs_missing') or ''} | {', '.join(f.get('not_equivalent_to') or [])} |")
    L += ["", "## What the agent did", ""]
    for i, s in enumerate(b["steps"], 1):
        L.append(f"{i}. **{s['what']}** ({s['pass']}, `{s['run_id']}`): {s['result']}")
    if b["observations"]:
        L += ["", "## What it found", ""]
        for o in b["observations"]:
            L.append(f"- **[{o['severity']}]** {o['claim']}" + (f" — next: {o['next_question']}" if o.get("next_question")
                                                                else ""))
            if o.get("figure"):
                L.append(f"  ![]({fig_prefix}{o['figure']})")
    if b["capability"]:
        c = b["capability"]
        L += ["", "## What the data can support", "", "| dimension | rating | why |", "|---|---|---|"]
        for k in ("history_depth", "cross_sectional_breadth", "update_frequency", "coverage_stability", "event_breadth"):
            if c.get(k):
                L.append(f"| {k.replace('_', ' ')} | {c[k]['rating']} | {c[k]['note']} |")
        L += ["", f"**Suitable for:** {'; '.join(c['suitable_for'])}", "",
              f"**Weak for:** {'; '.join(c['weak_for'])}", "", f"**Major limits:** {'; '.join(c['major_limits'])}"]
    if b["preprocessing"]:
        L += ["", "## How to represent the main fields", ""]
        for p in b["preprocessing"]:
            L.append(f"- **{p['field']}**: {'; '.join(p['observations'])}. Candidates: "
                     f"{', '.join(p['candidate_processing'])}." + (f" Risks: {'; '.join(p['risks'])}." if p['risks'] else ""))
    if b["cleaning"]:
        L += ["", "## Proposed cleaning (not applied until you approve)", "", b["cleaning"]["summary"], ""]
        for s in b["cleaning"]["steps"]:
            e = s["effect"]
            n = e.get("rows_removed", e.get("rows_flagged", 0))
            L.append(f"- `{s['op']}` on {s['table']}: {n:,} rows ({100 * e['share_of_rows']:.3f}%) "
                     f"{'removed' if 'rows_removed' in e else 'flagged'} — {s['reason']}")
    if b["warnings"] or b["open_questions"]:
        L += ["", "## Warnings and open questions", ""] + [f"- ⚠ {w}" for w in b["warnings"]] + \
             [f"- ? {q}" for q in b["open_questions"]]
    return "\n".join(L) + "\n"


CSS = """body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:24px auto;padding:0 20px;
color:#1f2937;line-height:1.5}h1{font-size:26px}h2{font-size:19px;margin-top:32px;border-bottom:1px solid #e5e7eb;
padding-bottom:4px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e5e7eb;
padding:6px 8px;text-align:left;vertical-align:top}th{color:#6b7280;font-weight:600}.muted{color:#6b7280}
.obs{border:1px solid #e5e7eb;border-radius:10px;padding:10px 14px;margin:10px 0}.issue{border-left:4px solid #d03b3b}
.warning{border-left:4px solid #c98500}.info{border-left:4px solid #3987e5}img{max-width:100%;margin-top:6px}
ol li{margin:4px 0}code{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12px}"""


def to_html(b: dict, run_dir: Path) -> str:
    e = lambda x: html.escape(str(x if x is not None else ""))
    img = lambda rel: (f'<img alt="figure" src="data:image/png;base64,'
                       f'{base64.b64encode((run_dir / rel).read_bytes()).decode()}">') if rel and (run_dir / rel).exists() else ""
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>Data Brief · {e(b['dataset_id'])}</title>"
         f"<style>{CSS}</style></head><body><h1>Data Brief · {e(b['dataset_id'])}</h1>"
         f"<p class='muted'>Run {e(b['run_id'])} · model {e(b['model'])} · {len(b['steps'])} tool calls · assembled "
         f"from the run's own records</p>"]
    o = b.get("overview")
    if o:
        H.append(f"<h2>What this dataset is</h2><p>{e(o['summary'])}</p><table>"
                 + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in [
                     ("Publisher", e(o["publisher"])), ("Purpose", e(o["purpose"])), ("Market", e(o["market"])),
                     ("Entities", "<br>".join(f"{e(x['type'])} <span class='muted'>({e(x['share'])})</span>" for x in o["entities"])),
                     ("Values", e("; ".join(o["values"]))), ("Frequency", e(o["frequency"])), ("Period", e(o["period"])),
                     ("Typical uses", e("; ".join(o["typical_uses"]))), ("Caveats", e("; ".join(o["caveats"])))]) + "</table>")
    for t in b.get("table_preview", []):
        H.append(f"<h3>Table <code>{e(t['table'])}</code> <span class='muted'>· {t['n_rows']:,} rows × {t['n_columns']} "
                 "columns · first rows</span></h3><div style='overflow-x:auto'><table><tr>"
                 + "".join(f"<th>{e(c['name'])}<br><span class='muted'>{e(c['dtype'])}</span></th>" for c in t["columns"])
                 + "</tr>" + "".join("<tr>" + "".join(f"<td>{e(v)}</td>" for v in row) + "</tr>" for row in t["head"])
                 + "</table></div>")
    H.append("<h2>What the fields mean</h2><ul>")
    H += [f"<li><b>{e(t['table'])}</b>: {e(t['observation_unit'])} (key: {e(', '.join(t['primary_key']))})</li>"
          for t in b["tables"]]
    H.append("</ul><table><tr><th>field</th><th>kind</th><th>unit</th><th>meaning</th><th>zero vs missing</th>"
             "<th>not the same as</th></tr>")
    H += [f"<tr><td><code>{e(f['table'])}.{e(f['name'])}</code></td><td>{e(f.get('kind'))}</td><td>{e(f.get('unit'))}"
          f"</td><td>{e(f['meaning'])}</td><td>{e(f.get('zero_vs_missing'))}</td>"
          f"<td>{e(', '.join(f.get('not_equivalent_to') or []))}</td></tr>" for f in b["fields"]]
    H.append("</table>")
    if b["observations"]:
        H.append("<h2>What it found</h2>")
        H += [f"<div class='obs {e(o['severity'])}'><b>{e(o['severity'])}</b> · {e(o['claim'])}"
              + (f"<div class='muted'>next: {e(o['next_question'])}</div>" if o.get("next_question") else "")
              + img(o.get("figure")) + "</div>" for o in b["observations"]]
    if b["capability"]:
        c = b["capability"]
        H.append("<h2>What the data can support</h2><table><tr><th>dimension</th><th>rating</th><th>why</th></tr>")
        H += [f"<tr><td>{e(k.replace('_', ' '))}</td><td>{e(c[k]['rating'])}</td><td>{e(c[k]['note'])}</td></tr>"
              for k in ("history_depth", "cross_sectional_breadth", "update_frequency", "coverage_stability",
                        "event_breadth") if c.get(k)]
        H.append(f"</table><p><b>Suitable for:</b> {e('; '.join(c['suitable_for']))}<br><b>Weak for:</b> "
                 f"{e('; '.join(c['weak_for']))}<br><b>Major limits:</b> {e('; '.join(c['major_limits']))}</p>")
    if b["preprocessing"]:
        H.append("<h2>How to represent the main fields</h2><ul>")
        H += [f"<li><b>{e(p['field'])}</b>: {e('; '.join(p['observations']))}. <i>Candidates:</i> "
              f"{e(', '.join(p['candidate_processing']))}." + (f" <i>Risks:</i> {e('; '.join(p['risks']))}."
                                                             if p["risks"] else "") + "</li>" for p in b["preprocessing"]]
        H.append("</ul>")
    if b["cleaning"]:
        H.append(f"<h2>Proposed cleaning <span class='muted'>(not applied until you approve)</span></h2>"
                 f"<p>{e(b['cleaning']['summary'])}</p><table><tr><th>step</th><th>table</th><th>rows affected</th>"
                 "<th>why</th></tr>")
        for s in b["cleaning"]["steps"]:
            ef = s["effect"]
            n = ef.get("rows_removed", ef.get("rows_flagged", 0))
            H.append(f"<tr><td><code>{e(s['op'])}</code></td><td>{e(s['table'])}</td><td>{n:,} "
                     f"({100 * ef['share_of_rows']:.3f}%) {'removed' if 'rows_removed' in ef else 'flagged'}</td>"
                     f"<td>{e(s['reason'])}</td></tr>")
        H.append("</table>")
    if b["warnings"] or b["open_questions"]:
        H.append("<h2>Warnings and open questions</h2><ul>")
        H += [f"<li>⚠ {e(w)}</li>" for w in b["warnings"]] + [f"<li>? {e(q)}</li>" for q in b["open_questions"]]
        H.append("</ul>")
    H.append("<h2>What the agent did</h2><ol>")
    H += [f"<li><b>{e(s['what'])}</b> <span class='muted'>({e(s['pass'])}, {e(s['run_id'])})</span>: "
          f"{e(s['result'])}</li>" for s in b["steps"]]
    H.append("</ol></body></html>")
    return "".join(H)


if __name__ == "__main__":
    import sys
    b = build(Path(sys.argv[1]))
    print(f"brief: {len(b['steps'])} steps, {len(b['observations'])} observations -> {sys.argv[1]}/brief/")
