"""The Predictive Report: phase B of a Research EDA run, from the signed plan to the locked test. What was
pre-registered (G4), what each test showed on discovery, what the orchestrator's re-test on validation said, how the
findings held up against the Critic, what was released (G5) and what the locked test showed, with the trial ledger
and every vault access. Assembled deterministically from the run's state and tool outputs (no LLM writes it), as predictive.json plus predictive.html.
Returns data appear only as aggregate statistics.
"""
import html
import json
from pathlib import Path

from quantaccelerator.agents.brief import CSS, steps
from quantaccelerator.state.schemas import PredictiveReport

EXTRA_CSS = """.pass{border-left:4px solid #199e70}.fail{border-left:4px solid #d03b3b}.null{border-left:4px solid #9ca3af}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}.chip{font-size:12px;padding:1px 6px;border-radius:8px;
background:#f3f4f6}"""
STATUS_TEXT = {"discovery": "discovery only", "validated": "validated", "failed_validation": "failed validation",
               "not_validated": "not a discovery", "locked_pass": "passed the locked test",
               "locked_fail": "failed the locked test"}


def _figures(run_dir: Path) -> dict:
    """(tool, feature[, horizon]) -> figure of the agent's latest discovery run."""
    out = {}
    for p in sorted((run_dir / "tool_outputs").glob("*.json"), key=lambda p: p.stat().st_mtime):
        d = json.loads(p.read_text())
        r, a = d.get("result") or {}, d.get("args") or {}
        if r.get("segment") == "discovery" and r.get("figure"):
            if d["tool"] == "ic_decay" and not a.get("controls"):
                out[("ic_decay", a.get("feature"))] = r["figure"]
            elif d["tool"] == "quantile_returns":
                out[("quantile_returns", a.get("feature"), int(a.get("horizon", 0)))] = r["figure"]
    return out


def build(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    rep = PredictiveReport.model_validate_json((run_dir / "state" / "predictive_report.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
    agents = tuple(k for k in meta.get("agents", {}) if k.startswith("predictive_eda"))
    figs = _figures(run_dir)
    b = {"run_id": meta.get("run_id", run_dir.name), "model": meta.get("model"), "set_hash": rep.set_hash,
         "plan": rep.plan.model_dump(mode="json"),
         "plan_gate": rep.plan_gate.model_dump(mode="json", exclude={"approved_plan"}) if rep.plan_gate else None,
         "release_gate": rep.release_gate.model_dump(mode="json") if rep.release_gate else None,
         "observations": [o.model_dump(mode="json") for o in rep.observations],
         "wrapup": rep.wrapup.model_dump(mode="json") if rep.wrapup else None, "ledger": rep.ledger,
         "critiques": [c.model_dump(mode="json") for c in rep.critiques],
         "critic_dropped": json.loads((run_dir / "state" / "critic_dropped.json").read_text())
         if (run_dir / "state" / "critic_dropped.json").exists() else [],
         "figures": {t.feature: [f for f in (figs.get(("ic_decay", t.feature)),
                                             figs.get(("quantile_returns", t.feature, t.horizons[0]))) if f]
                     for t in rep.plan.tests},
         "steps": [s for s in steps(run_dir, agents=agents) if s.get("what") != "record_research_observation"]}
    out = run_dir / "brief"
    out.mkdir(exist_ok=True)
    (out / "predictive.json").write_text(json.dumps(b, indent=1, default=str))
    (out / "predictive.html").write_text(to_html(b, run_dir))
    return b


def _ict(x: dict | None, key: str = "stats") -> str:
    if not x:
        return "–"
    ic, t = x.get("mean_ic"), x.get("t_nw")
    return "–" if ic is None or t is None else f"{ic:+.4f} <span class='muted'>t {t:+.2f}</span>"


def to_html(b: dict, run_dir: Path) -> str:
    import base64
    e = lambda x: html.escape(str(x if x is not None else ""))
    img = lambda rel: (f'<img alt="figure" src="data:image/png;base64,'
                       f'{base64.b64encode((run_dir / rel).read_bytes()).decode()}">') if rel and (run_dir / rel).exists() else ""
    sp, g4, g5, led = b["plan"]["split"] or {}, b["plan_gate"] or {}, b["release_gate"], b["ledger"]
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>Predictive Report · {e(b['run_id'])}</title>"
         f"<style>{CSS}{EXTRA_CSS}</style></head><body><h1>Predictive Report</h1>"
         f"<p class='muted'>Run {e(b['run_id'])} · model {e(b['model'])} · frozen feature set <code>{e(b['set_hash'])}"
         "</code> · research EDA phase B · assembled from the run's own records; returns shown only as aggregates</p>",
         "<h2>Pre-registration (G4)</h2>",
         f"<p><b>{'Signed' if g4.get('approved') else 'Not signed'}</b> by {e(g4.get('decided_by', '–'))}"
         + (f" · {e(g4['note'])}" if g4.get("note") else "") + "</p>",
         f"<p><b>Split</b> (decision dates; each segment's returns stop at its end): discovery "
         f"{e(' to '.join(sp.get('discovery', [])))} · validation {e(' to '.join(sp.get('validation', [])))} · locked "
         f"test {e(' to '.join(sp.get('locked_test', [])))}</p>",
         "<table><tr><th>feature</th><th>expected sign</th><th>horizons (sessions)</th><th>controls</th><th>why</th></tr>"]
    H += [f"<tr><td><code>{e(t['feature'])}</code></td><td>{t['expected_sign']:+d}</td><td>{e(t['horizons'])}</td>"
          f"<td>{e(', '.join(t['controls']) or '–')}</td><td>{e(t['horizon_rationale'])}</td></tr>"
          for t in b["plan"]["tests"]]
    H.append("</table>")
    H.append("<h2>Results</h2><p class='muted'>IC: mean same-date rank correlation of the feature (net of the controls) "
             f"with the forward return from the open of the decision date; t: Newey-West. Discovery verdicts need |t| ≥ "
             f"the hurdle ({e(led.get('hurdle_t'))} at {e(led.get('n_trials'))} distinct tests); the orchestrator re-runs "
             "every 'informative' one unchanged on validation and, after release, on the locked test (same sign and "
             "|t| ≥ 2).</p><table><tr><th>feature</th><th>h</th><th>controls</th><th>verdict</th><th>discovery</th>"
             "<th>validation</th><th>locked test</th><th>status</th></tr>")
    for o in b["observations"]:
        cls = "pass" if o["status"] in ("validated", "locked_pass") else "fail" if "fail" in o["status"] else "null"
        H.append(f"<tr class='{cls}'><td><code>{e(o['feature'])}</code></td><td>{o['horizon']}</td>"
                 f"<td>{e(', '.join(o['controls']) or '–')}</td><td>{e(o['verdict'])}</td>"
                 f"<td class='num'>{_ict(o['stats'])}</td><td class='num'>{_ict(o.get('validation'))}</td>"
                 f"<td class='num'>{_ict(o.get('locked_test'))}</td>"
                 f"<td><span class='chip'>{e(STATUS_TEXT.get(o['status'], o['status']))}</span></td></tr>")
    H.append("</table>")
    if b["critiques"]:
        H.append("<h2>Critic</h2><p class='muted'>Every validated finding faces the standard battery (no illiquid third, "
                 "no penny stocks, each half of discovery) and the Critic's own attacks, written from the findings' "
                 "outputs only and run on discovery. Verdict by a fixed rule: survived = same sign, at least half the "
                 "IC kept and |t| ≥ 2; refuted = sign flip with |t| ≥ 2, or under a quarter of the IC left.</p>"
                 "<table><tr><th>finding</th><th>source</th><th>threat</th><th>test</th><th>attacked IC / t</th>"
                 "<th>names</th><th>kept</th><th>outcome</th></tr>")
        for c in b["critiques"]:
            o = b["observations"][c["target"]]
            t = c["test"]
            what = t["kind"] + ": " + str(t.get("universe") or t.get("controls") or t.get("period")) + \
                (f" {t['sector']}" if t.get("sector") else "")
            cls = {"survived": "pass", "refuted": "fail"}.get(c["outcome"], "null")
            H.append(f"<tr class='{cls}'><td><code>{e(o['feature'])}</code> h={o['horizon']}</td><td>{e(c['source'])}"
                     f"</td><td>{e(c['threat_type'])}<div class='muted'>{e(c['threat'])}</div></td><td>{e(what)}</td>"
                     f"<td class='num'>{_ict(c['attacked'])}</td><td>{e(c['attacked'].get('entities'))}</td>"
                     f"<td>{e(c['retention'])}</td><td><span class='chip'>{e(c['outcome'])}</span></td></tr>")
        H.append("</table>")
        if b["critic_dropped"]:
            H.append("<p class='muted'>Critiques that could not run as written: "
                     + e("; ".join(f"{d['critique']['threat_type']}: {d['reason']}" for d in b["critic_dropped"]))
                     + "</p>")
    costed = [o for o in b["observations"] if o.get("costs")]
    if costed:
        H.append("<h2>Cost and capacity</h2><p class='muted'>The finding's long-short quantile portfolio, rebalanced every "
                 "h sessions, gross and net of half-spreads (round trip each period: conservative; changes only: "
                 "optimistic) and square-root impact. Break-even half-spread: the cost per side that uses up the gross "
                 "edge (independent of the spread estimate). Discovery and validation only.</p><table><tr><th>finding"
                 "</th><th>segment</th><th>gross bp/period</th><th>net round trip</th><th>net changes only</th>"
                 "<th>break-even half-spread</th><th>half-spread used</th><th>capacity per leg</th></tr>")
        for o in costed:
            for seg, k in o["costs"].items():
                if not k.get("gross"):
                    continue
                cap = k.get("capacity_per_leg_usd")
                H.append(f"<tr><td><code>{e(o['feature'])}</code> h={o['horizon']}</td><td>{e(seg)}</td>"
                         f"<td class='num'>{k['gross']['mean_bp']:+.1f}</td>"
                         f"<td class='num'>{k['net_round_trip']['mean_bp']:+.1f}</td>"
                         f"<td class='num'>{k['net_changes_only']['mean_bp']:+.1f}</td>"
                         f"<td class='num'>{k['break_even_half_spread_bp']:.1f} bp</td>"
                         f"<td>{k['median_half_spread_bp']:.1f} bp <div class='muted'>{e(k.get('spread_source', ''))}"
                         f"</div></td><td class='num'>{f'${cap:,.0f}' if cap else 'none'}</td></tr>")
        H.append("</table>")
    if g5:
        H.append(f"<p><b>G5</b> by {e(g5['decided_by'])}: released observations {e(g5.get('released_observations'))}"
                 + (f" · {e(g5['note'])}" if g5.get("note") else "") + "</p>")
    else:
        H.append("<p class='muted'>G5 pending: nothing released to the locked test yet.</p>")
    H.append("<h2>Per feature</h2>")
    for t in b["plan"]["tests"]:
        mine = [o for o in b["observations"] if o["feature"] == t["feature"]]
        H.append(f"<div class='obs info'><b>{e(t['feature'])}</b>"
                 + "".join(f"<div>h={o['horizon']}, {e(', '.join(o['controls']) or 'raw')}: {e(o['claim'])}</div>"
                           for o in mine)
                 + "".join(img(f) for f in b["figures"].get(t["feature"], [])) + "</div>")
    w = b["wrapup"] or {}
    if w:
        H.append(f"<h2>Summary (the agent's)</h2><p>{e(w.get('summary'))}</p>")
        if w.get("data_investigation_requests"):
            from quantaccelerator.agents.feature_report import investigations_html
            H.append("<h2>Routed back</h2>" + investigations_html(run_dir, w["data_investigation_requests"],
                                                                 "predictive_eda"))
        if w.get("warnings") or w.get("open_questions"):
            H.append("<ul>" + "".join(f"<li>⚠ {e(x)}</li>" for x in w.get("warnings", []))
                     + "".join(f"<li>? {e(x)}</li>" for x in w.get("open_questions", [])) + "</ul>")
    H.append(f"<h2>Ledger</h2><p>Study <code>{e(led.get('study'))}</code>: {e(led.get('n_trials'))} distinct tests on "
             "discovery, across every session that tested this feature set. Vault accesses (granted and refused):</p>"
             "<table><tr><th>segment</th><th>by</th><th>granted</th><th>reason</th><th>when (UTC)</th></tr>")
    H += [f"<tr><td>{e(a['segment'])}</td><td>{e(a['caller'])}</td><td>{'yes' if a['granted'] else 'refused'}</td>"
          f"<td>{e(a['reason'])}</td><td>{e(a['created_utc'])}</td></tr>" for a in led.get("vault_accesses", [])]
    H.append("</table><h2>What the agent did</h2><ol>")
    H += [f"<li><b>{e(s['what'])}</b> <span class='muted'>({e(s['run_id'])})</span>: {e(s['result'])}</li>"
          for s in b["steps"]]
    H.append("</ol></body></html>")
    return "".join(H)


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        build(Path(a))
        print(Path(a) / "brief" / "predictive.html")
