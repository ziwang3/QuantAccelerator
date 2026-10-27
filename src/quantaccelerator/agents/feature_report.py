"""The Feature Report: the candidate feature set of a Research EDA run, how each feature is built, what it still
depends on, what the agent found on the way, and what it routed back. Assembled deterministically from the run's
state and tool outputs (no LLM writes it), as features.json plus a self-contained features.html.
"""
import html
import json
from pathlib import Path

from quantaccelerator.agents.brief import CSS, steps
from quantaccelerator.state.schemas import CandidateFeatureSet
from quantaccelerator.tools.features import formula

BY = ["adv_proxy", "size_float", "size_assets", "sector"]


def measured(run_dir: Path) -> dict:
    """(feature, conditioner) -> (dependence, strength, run_id), the latest measurement in the run's tool outputs."""
    out = {}
    for p in sorted((run_dir / "tool_outputs").glob("*.json"), key=lambda p: p.stat().st_mtime):
        d = json.loads(p.read_text())
        r, a = d.get("result") or {}, d.get("args") or {}
        if d.get("tool") == "compute_exposure" and "strength" in r:
            v = r.get("rank_corr_mean", r.get("eta_squared_mean"))
            out[(a["feature"], a["by"])] = (v, r["strength"], d["run_id"])
        elif d.get("tool") == "compare_representations" and "features" in r:
            for f, x in r["features"].items():
                out[(f, a["by"])] = (x["dependence"], x["strength"], d["run_id"])
    return out


def build(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    fs = CandidateFeatureSet.model_validate_json((run_dir / "state" / "candidate_feature_set.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
    specs = {s.name: s for s in fs.features}
    m = measured(run_dir)
    g = fs.signed_off
    b = {"run_id": meta.get("run_id", run_dir.name), "model": meta.get("model"), "panel": fs.panel,
         "set_hash": fs.set_hash, "gate": g.model_dump(mode="json") if g else None,
         "candidates": [{**c.model_dump(mode="json"), "definition": formula(specs[c.name]) if c.name in specs else "",
                         "frozen": bool(g and g.approved_features and c.name in g.approved_features),
                         "dependence": {by: m.get((c.name, by)) for by in BY}} for c in fs.candidates],
         "features": [s.model_dump(mode="json") | {"definition": formula(s)} for s in fs.features],
         "idea_gate": fs.idea_gate.model_dump(mode="json") if fs.idea_gate else None,
         "ideas": [{**i.model_dump(mode="json", exclude={"specs"}),
                    "definitions": [{"name": sp.name, "definition": formula(sp)} for sp in i.specs],
                    "approved": bool(fs.idea_gate and i.name in (fs.idea_gate.approved_ideas or [])),
                    "candidates": [c.name for c in fs.candidates if c.idea == i.name],
                    "set_aside": [o.claim for o in fs.observations if o.idea == i.name and o.decision == "drop"]}
                   for i in fs.ideas],
         "observations": [o.model_dump(mode="json") | {"figure": f"figures/{o.figure_run_id}.png"
                                                       if o.figure_run_id else None} for o in fs.observations],
         "wrapup": fs.wrapup.model_dump(mode="json"),
         "steps": [s | {"pass": label} for agent, label in (("research_eda_survey", "survey"), ("research_eda_ideas", "ideas"),
                                                              ("research_eda", "build and characterize"))
                   for s in steps(run_dir, agents=(agent,))]}
    out = run_dir / "brief"
    out.mkdir(exist_ok=True)
    (out / "features.json").write_text(json.dumps(b, indent=1, default=str))
    (out / "features.html").write_text(to_html(b, run_dir))
    return b


def investigations_html(run_dir: Path, requests: list, stage: str) -> str:
    """The questions research routed back upstream, with the answers of the agents that own them (if routed)."""
    e = lambda x: html.escape(str(x if x is not None else ""))
    p = run_dir / "state" / "investigations.json"
    res = {(x["stage"], x["request"]["question"]): x for x in json.loads(p.read_text())} if p.exists() else {}
    out = []
    for r in requests:
        x = res.get((stage, r["question"]))
        a = (x or {}).get("answer")
        out.append(f"<div class='obs info'>→ <b>{e(r['route_to'])}</b>: {e(r['question'])}"
                   f"<div class='muted'>prompted by: {e(r['observation'])}</div>"
                   + (f"<div><b>Answer</b> ({e(x['routed_to'])}): {e(a['answer'])}</div>"
                      + "".join(f"<div class='muted'>caveat: {e(c)}</div>" for c in a.get("caveats", []))
                      + f"<div class='muted'>evidence: {e(', '.join(a.get('evidence', [])))}</div>" if a else
                      f"<div class='muted'>not answered: {e(x.get('error'))}</div>" if x else
                      "<div class='muted'>not routed in this run</div>") + "</div>")
    return "".join(out)


def _dep(x) -> str:
    if not x:
        return "–"
    v, strength, _ = x
    return f"{v:+.2f} <span class='muted'>{strength}</span>" if v is not None else "–"


def to_html(b: dict, run_dir: Path) -> str:
    import base64
    e = lambda x: html.escape(str(x if x is not None else ""))
    img = lambda rel: (f'<img alt="figure" src="data:image/png;base64,'
                       f'{base64.b64encode((run_dir / rel).read_bytes()).decode()}">') if rel and (run_dir / rel).exists() else ""
    p, g = b["panel"], b["gate"]
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>Feature Report · {e(b['run_id'])}</title>"
         f"<style>{CSS}</style></head><body><h1>Feature Report</h1>"
         f"<p class='muted'>Run {e(b['run_id'])} · model {e(b['model'])} · {len(b['steps'])} tool calls · research EDA "
         "phase A (return-blind) · assembled from the run's own records</p>",
         f"<p><b>Panel:</b> {e(p.get('entities'))} entities × {e(p.get('dates'))} decision dates, {e(p.get('first_date'))}"
         f" to {e(p.get('last_date'))}; universe: {e(p.get('universe'))}.<br><b>Timing:</b> every value at its decision "
         f"date under the signed PIT rule <code>{e(json.dumps(p.get('rule')))}</code>.</p>",
         ]
    if b.get("ideas"):
        g2 = b.get("idea_gate") or {}
        H.append(f"<h2>Feature ideas</h2><p class='muted'>Proposed before anything was built; G2 idea review by "
                 f"{e(g2.get('decided_by', '–'))}. Expected directions are hypotheses for later testing with returns, "
                 "never checked here.</p>")
        for i in b["ideas"]:
            H.append(f"<div class='obs info'><b>{e(i['name'])}</b> <span class='muted'>· {e(i['source'])}"
                     f"{' · approved' if i['approved'] else ' · not approved'}</span><br>{e(i['idea'])}"
                     + (f"<div class='muted'>researcher's words: “{e(i['researcher_text'])}”</div>" if i.get("researcher_text") else "")
                     + f"<div class='muted'>mechanism: {e(i['mechanism'])}</div><div class='muted'>assumes: "
                     f"{e('; '.join(i['assumptions']))}</div><div class='muted'>expected (untested): {e(i['expected_direction'])}</div>"
                     + "<div>" + "<br>".join(f"<code>{e(d['name'])}</code> = {e(d['definition'])}" for d in i["definitions"]) + "</div>"
                     + (f"<div>candidate: {e(', '.join(i['candidates']))}</div>" if i["candidates"] else "")
                     + (f"<div class='muted'>set aside: {e('; '.join(i['set_aside']))}</div>" if i["set_aside"] else "")
                     + "</div>")
    H.append("<h2>Candidate feature set</h2>")
    if g:
        H.append(f"<p><b>G3 {'frozen' if g['approved'] else 'not approved'}</b> by {e(g['decided_by'])}"
                 + (f" · set hash <code>{e(b['set_hash'])}</code>" if b["set_hash"] else "")
                 + (f" · {e(g['note'])}" if g.get("note") else "") + "</p>")
    else:
        H.append("<p class='muted'>Not yet frozen (gate G3 pending).</p>")
    H.append("<table><tr><th>feature</th><th>idea</th><th>definition</th><th>why</th>"
             + "".join(f"<th>{e(by)}</th>" for by in BY) + "</tr>")
    H += [f"<tr><td><code>{e(c['name'])}</code>{' ✓' if c['frozen'] else ''}</td><td>{e(c.get('idea') or '')}</td><td>{e(c['definition'])}</td>"
          f"<td>{e(c['rationale'])}" + (f"<div class='muted'>still depends on: {e('; '.join(c['known_exposures']))}</div>"
                                        if c["known_exposures"] else "") + "</td>"
          + "".join(f"<td>{_dep(c['dependence'][by])}</td>" for by in BY) + "</tr>" for c in b["candidates"]]
    H.append("</table><p class='muted'>Dependence: mean same-date rank correlation (liquidity, size) or the share of "
             "rank variance sector explains; 0 = none. Built from: "
             + e("; ".join(f"{f['name']} = {f['definition']}" for f in b["features"])) + "</p>")
    if b["observations"]:
        H.append("<h2>What it found</h2>")
        for o in b["observations"]:
            H.append(f"<div class='obs info'><b>{e(o['feature'])}</b> · {e(o['conditioner'])} · <i>{e(o['decision'])}"
                     f"</i><br>{e(o['claim'])}"
                     + (f"<div class='muted'>possible explanations: {e('; '.join(o['possible_explanations']))}</div>"
                        if o["possible_explanations"] else "")
                     + (f"<div class='muted'>compare: {e('; '.join(o['candidate_representations']))}</div>"
                        if o["candidate_representations"] else "")
                     + (f"<div class='muted'>would decide: {e('; '.join(o['required_checks']))}</div>"
                        if o["required_checks"] else "") + img(o.get("figure")) + "</div>")
    w = b["wrapup"]
    H.append(f"<h2>Summary</h2><p>{e(w['summary'])}</p>")
    if w["data_investigation_requests"]:
        H.append("<h2>Routed back</h2>" + investigations_html(run_dir, w["data_investigation_requests"], "research_eda"))
    if w["warnings"] or w["open_questions"]:
        H.append("<h2>Warnings and open questions</h2><ul>")
        H += [f"<li>⚠ {e(x)}</li>" for x in w["warnings"]] + [f"<li>? {e(q)}</li>" for q in w["open_questions"]]
        H.append("</ul>")
    H.append("<h2>What the agent did</h2><ol>")
    H += [f"<li><b>{e(s['what'])}</b> <span class='muted'>({e(s['run_id'])})</span>: {e(s['result'])}</li>"
          for s in b["steps"]]
    H.append("</ol></body></html>")
    return "".join(H)
