"""Rebuild a replayable event stream from a finished run directory.

A run already stores everything needed: transcripts/<agent>.json (messages), llm_calls.jsonl (one line per LLM call,
with timestamps), the registry (tool-run timestamps), state/*.json and score.json. Events are plain dicts with `t`
(seconds since the run started) and `type`; the viewer (and later the live server) consumes exactly this format.
"""
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

from quantaccelerator.paths import RUNS

AGENTS = ["profiler", "overview", "data_research", "pit_agent", "auditor", "research_eda_survey", "research_eda_ideas", "research_eda"]
MAX_TEXT = 2500  # characters kept per tool result / message in the viewer
RUN_ID = re.compile(r'"run_id"\s*:\s*"(r[0-9a-f]{10})"')


def summarize(name: str, args: dict, result) -> str:
    """One human-readable line per tool call, computed from the full (unclipped) result."""
    r = result if isinstance(result, dict) else {}
    if isinstance(r, dict) and "error" in r:
        return "error: " + str(r["error"]).strip().splitlines()[-1][:160]
    if name == "list_tables" and isinstance(result, list):
        return f"{len(result)} tables: " + ", ".join(f"{t['table']} ({t['n_rows']:,} rows)" for t in result)
    if name == "profile_table":
        return f"{r.get('table')}: {r.get('n_rows', 0):,} rows × {len(r.get('columns', []))} columns"
    if name == "profile_time":
        return f"{r.get('table')}: time profile of {len(r.get('columns') or {})} column(s)"
    if name == "read_doc" and isinstance(result, list):
        hits = [f"{h['doc'].replace('sec_', '')} p{h['page']}" + (f" · {h['section']}" if h.get("section") else "")
                + (" (followed reference)" if h.get("followed_from") else "") for h in result]
        return f"“{args.get('query', '')}” → " + "; ".join(hits)
    if name == "timezone_check":
        sm = r.get("summary", {})
        cons = sm.get("form345_consistency_by_reading", {})
        return ("filing-date consistency: " + ", ".join(f"{k} {v:.1%}" for k, v in cons.items())
                + f" → {sm.get('more_consistent_reading')}")
    if name == "run_pipeline":
        return f"pipeline ran: {r.get('n_rows', 0):,} events"
    if name == "measure_lookahead":
        return (f"{r.get('share_before_tradable', 0):.1%} of {r.get('n_events', 0):,} events used before they were "
                "tradable")
    if name == "test_patch":
        la = r.get("lookahead", {})
        return f"patched pipeline re-run: {la.get('share_before_tradable', 0):.1%} used early ({la.get('n_events', 0):,} events)"
    if name == "read_source" and isinstance(result, str):
        return f"read {args.get('path', '')} ({result.count(chr(10)) + 1} lines)"
    if name == "preview_rule":
        return f"rule preview on {len(r.get('examples', []))} example filings"
    if name == "lag_distribution":
        q = r.get("quantiles_days", {})
        return f"lag over {r.get('n_pairs', 0):,} pairs: median {q.get('p50')} d, p90 {q.get('p90')} d, p99 {q.get('p99')} d"
    if name == "next_tradable":
        return f"{args.get('timestamp')} → first tradable {r.get('first_tradable') or r}"
    return _summarize_research(name, args, r) or json.dumps(result, default=str)[:160]


def _pct(x) -> str:
    return "–" if x is None else "<0.1%" if 0 < x < 0.0005 else f"{100 * x:.1f}%"


def _summarize_research(name: str, a: dict, r: dict) -> str | None:
    """One line for each data-research tool, in the researcher's terms."""
    t, e, f = a.get("table", ""), a.get("entity_field"), a.get("field", "")
    f = f"{f}/{a['denominator']}" if a.get("denominator") else f
    if name == "research_breadth":
        c = r.get("concentration") or {}
        return (f"{r.get('n_entities', 0):,} {e} over {r.get('history_years')} years ({r.get('history_start')} to "
                f"{r.get('history_end')}); median lifetime {(r.get('entity_lifetime_years') or {}).get('median')} y"
                + (f"; top 10 hold {_pct(c.get('top10_share'))} of {c.get('weight')}" if c else ""))
    if name == "coverage_over_time":
        q, sh = r.get("entities_per_period") or r.get("rows_per_period") or {}, r.get("level_shifts") or []
        return (f"{'entities' if e else 'rows'} per {a.get('freq', 'M')} period {q.get('min')} to {q.get('max')}"
                + (f"; level shifts at {', '.join(x['date'] for x in sh)}" if sh else "; no abrupt level shift"))
    if name == "key_check":
        return (f"key {r.get('key')} is unique ({r.get('n_rows', 0):,} rows)" if r.get("key_is_unique") else
                f"key {r.get('key')} NOT unique: {r.get('rows_with_duplicated_key', 0):,} rows share a key, "
                f"{r.get('exact_duplicate_rows', 0):,} exact duplicates")
    if name == "missingness":
        fl = [f"{k} null {_pct(v['null_share'])}" + (f", zero {_pct(v['zero_share'])}" if v.get("zero_share") else "")
              for k, v in (r.get("fields") or {}).items() if v.get("null_share") or v.get("zero_share")]
        pn = r.get("panel")
        return ("; ".join(fl) or "no nulls or zeros") + (
            f"; entities absent on {_pct(pn['absent_inside_span_share'])} of dates inside their history" if pn else "")
    if name == "distribution":
        p = r.get("percentiles") or {}
        return (f"{r.get('field')}: median {p.get('p50')}, p1–p99 {p.get('p1')}–{p.get('p99')}, skew {r.get('skew')}"
                + (f"; {'; '.join(r['hints'])}" if r.get("hints") else ""))
    if name == "update_dynamics":
        ac = r.get("autocorrelation") or {}
        return (f"{r.get('field')}: unchanged from the previous observation {_pct(r.get('share_unchanged_vs_previous_obs'))}"
                f"; autocorrelation " + ", ".join(f"{k} {v}" for k, v in ac.items()))
    if name == "variance_split":
        return (f"{r.get('field')}: {_pct(r.get('share_between_entities'))} across entities, "
                f"{_pct(r.get('share_within_entities'))} within entities over time → {r.get('reading')}")
    if name == "structural_breaks":
        b = r.get("breaks") or {}
        return ("candidate breaks: " + "; ".join(f"{k} at {', '.join(x['date'] for x in v)}" for k, v in b.items())
                if b else "no candidate breaks")
    if name == "category_mix_over_time":
        new = r.get("values_appearing_after_start") or {}
        return (f"{r.get('n_distinct_values')} {f} values"
                + (f"; appearing mid-sample: {', '.join(f'{k} ({v})' for k, v in new.items())}" if new else ""))
    if name == "preview_cleaning":
        return (f"{r.get('op')} on {r.get('table')}: {r.get('rows_affected', 0):,} rows "
                f"({_pct(r.get('share_affected'))}) would be {r.get('effect')}")
    if name == "look_at_figure":
        return f"vision model on {a.get('figure_run_id')}: {str(r.get('answer', ''))[:200]}"
    if name == "describe_panel":
        return f"{r.get('entities', 0):,} entities × {r.get('dates', 0):,} dates ({r.get('first_date')} to {r.get('last_date')})"
    if name == "construct_feature":
        return f"{r.get('feature')} = {r.get('definition')} (coverage {_pct(r.get('coverage_share'))}, median {r.get('median')})"
    if name in ("compute_exposure", "time_stability", "compare_representations"):
        who = a.get("feature") or ", ".join(a.get("features") or [])
        return f"{who}: {r.get('reading', '')}"
    g = lambda v, d=4: "n/a" if v is None else f"{v:.{d}f}"
    led = f" [N {r.get('n_trials')} tests, hurdle {r.get('hurdle_t')}]" if r.get("n_trials") is not None else ""
    if name == "describe_prediction_setup":
        return (f"frozen {', '.join(r.get('frozen_features') or [])}; discovery {r.get('discovery')}; "
                f"{(r.get('coverage') or {}).get('dates')} dates{led}")
    if name == "ic_summary":
        return (f"{r.get('feature')} h={r.get('horizon')} controls={r.get('controls') or []}: mean IC "
                f"{g(r.get('mean_ic'))}, t_nw {g(r.get('t_nw'), 2)}, hit {g(r.get('hit_rate'), 3)} "
                f"({r.get('significance')}){led}")
    if name == "ic_decay":
        bh = r.get("by_horizon") or {}
        return (f"{r.get('feature')} controls={r.get('controls') or []}: "
                + ", ".join(f"h{h} {g(v.get('mean_ic'))} (t {g(v.get('t_nw'), 1)})" for h, v in bh.items())
                + f"; peak h={r.get('peak_horizon')}{led}")
    if name == "quantile_returns":
        q = r.get("quantile_mean_excess_return") or {}
        return (f"{r.get('feature')} h={r.get('horizon')}: " + ", ".join(f"{k} {g(v, 5)}" for k, v in q.items())
                + f"; top-bottom {g(r.get('top_minus_bottom'), 5)} (t {g(r.get('t_nw'), 2)}), monotonicity "
                f"{g(r.get('monotonicity'), 2)}{led}")
    if name == "conditional_ic":
        b = r.get("buckets") or {}
        return (f"{r.get('feature')} h={r.get('horizon')} by {r.get('by')}: "
                + ", ".join(f"{k} {g(v.get('mean_ic'))} (t {g(v.get('t_nw'), 1)})" for k, v in b.items()) + led)
    return None


def _clip(s: str, n: int = MAX_TEXT) -> str:
    return s if len(s) <= n else s[:n] + f" …[{len(s) - n} more chars]"


def _ts(s: str) -> float:
    d = dt.datetime.fromisoformat(s)
    return (d if d.tzinfo else d.astimezone()).timestamp()


def _tool_times(run_ids: set[str]) -> dict[str, tuple[float, float]]:
    """run_id -> (created timestamp, seconds) from the registry."""
    db = RUNS / "registry.sqlite"
    if not run_ids or not db.exists():
        return {}
    con = sqlite3.connect(db)
    q = f"select run_id, created_utc, seconds from runs where run_id in ({','.join('?' * len(run_ids))})"
    return {r: (_ts(c), s or 0.0) for r, c, s in con.execute(q, sorted(run_ids))}


def _parse(content: str) -> dict:
    try:
        v = json.loads(content)
        return v if isinstance(v, dict) else {"result": v}
    except (json.JSONDecodeError, TypeError):
        return {"raw": content}


def run_events(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    calls = [json.loads(line) for line in (run_dir / "llm_calls.jsonl").read_text().splitlines() if line.strip()]
    transcripts = {a: json.loads(p.read_text()) for a in AGENTS if (p := run_dir / "transcripts" / f"{a}.json").exists()}
    ids = {x.group(1) for ms in transcripts.values() for m in ms if m["role"] == "tool"
           and (x := RUN_ID.search(m["content"]))}
    tool_t = _tool_times({i for i in ids if i})
    # llm_calls.jsonl stamps the end of each call; its start is end - latency
    t0 = min([_ts(c["ts"]) - c["latency_s"] for c in calls] or [0.0])
    ev = [{"t": 0.0, "type": "run_start", "run_id": meta["run_id"], "pipeline": meta.get("pipeline"),
           "mode": meta.get("mode", "audit"), "dataset": meta.get("dataset"), "model": meta.get("model"),
           "oracle": meta.get("oracle")}]

    for agent in AGENTS:
        msgs = transcripts.get(agent)
        agent_calls = [c for c in calls if c["agent"] == agent]
        if msgs is None and not agent_calls:
            continue
        info = meta["agents"].get(agent, {})
        start = (_ts(agent_calls[0]["ts"]) - agent_calls[0]["latency_s"] - t0) if agent_calls else ev[-1]["t"]
        ev.append({"t": round(start, 1), "type": "agent_start", "agent": agent, "task": _clip(msgs[1]["content"], 1200)
                   if msgs and len(msgs) > 1 else ""})
        k, t, pending = 0, start, {}
        for m in (msgs or [])[2:]:
            if m["role"] == "assistant":
                c = agent_calls[k] if k < len(agent_calls) else {}
                k += 1
                t = (_ts(c["ts"]) - t0) if c else t + 1
                calls_ = [{"id": tc["id"], "name": tc["function"]["name"],
                           "args": _clip(tc["function"]["arguments"], 1500)} for tc in m.get("tool_calls", [])]
                pending = {x["id"]: x for x in calls_}
                ev.append({"t": round(t, 1), "type": "llm", "agent": agent, "text": _clip(m.get("content") or "", 1500),
                           "tokens_in": c.get("tokens_in"), "tokens_out": c.get("tokens_out"),
                           "latency_s": c.get("latency_s"), "finish_reason": c.get("finish_reason"),
                           "calls": [{"name": x["name"], "args": x["args"]} for x in calls_]})
            elif m["role"] == "tool":
                call = pending.pop(m.get("tool_call_id"), {"name": "?", "args": ""})
                body = _parse(m["content"])
                rid = body.get("run_id") or ((x := RUN_ID.search(m["content"])) and x.group(1))
                if rid and (full := run_dir / "tool_outputs" / f"{rid}.json").exists():
                    body = {"run_id": rid, "ok": True, **{k: v for k, v in json.loads(full.read_text()).items()
                                                          if k == "result"}}
                tt = t + 0.1
                if rid in tool_t:
                    tt = max(t, tool_t[rid][0] - t0 + tool_t[rid][1])
                if call["name"] == "submit":
                    ev.append({"t": round(tt, 1), "type": "submit", "agent": agent, "accepted": False,
                               "error": _clip(str(body.get("error", "")), 1500)})
                else:
                    ok = body.get("ok", True) and not (isinstance(body.get("result"), dict) and "error" in body["result"])
                    try:
                        args = json.loads(call["args"])
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    ev.append({"t": round(tt, 1), "type": "tool", "agent": agent, "name": call["name"],
                               "args": call["args"], "run_id": rid, "ok": bool(ok),
                               "summary": summarize(call["name"], args if isinstance(args, dict) else {},
                                                    body.get("result", body)),
                               "result": _clip(json.dumps(body.get("result", body), default=str))})
                t = tt
            elif m["role"] == "user":
                ev.append({"t": round(t + 0.1, 1), "type": "nudge", "agent": agent, "text": _clip(m["content"], 600)})
        accepted = any(x["name"] == "submit" for x in pending.values())
        end = start + info.get("wall_s", max(t - start, 0))
        if accepted:
            ev.append({"t": round(end, 1), "type": "submit", "agent": agent, "accepted": True})
        ev.append({"t": round(end, 1), "type": "agent_end", "agent": agent, "ok": accepted,
                   "wall_s": info.get("wall_s"), "tool_calls": info.get("tool_calls"),
                   "submit_failures": info.get("submit_failures")})
        if agent == "pit_agent" and (p := run_dir / "state" / "pit_contract.json").exists():
            c = json.loads(p.read_text())
            ev.append({"t": round(end, 1), "type": "gate", **c["signed_off"],
                       "rule": c["availability_rule"], "explanation": c["rule_explanation"]})
        if agent == "research_eda_ideas" and (p := run_dir / "state" / "idea_plan.json").exists():
            plan = json.loads(p.read_text())
            ev.append({"t": round(end, 1), "type": "ideas", "ideas": [{"name": i["name"], "idea": i["idea"],
                                                                         "source": i["source"]} for i in plan["ideas"]]})
            fsp = run_dir / "state" / "candidate_feature_set.json"
            g = json.loads(fsp.read_text()).get("idea_gate") if fsp.exists() else None
            if g:
                ev.append({"t": round(end, 1), "type": "gate_ideas", **g})
        if agent == "research_eda" and (p := run_dir / "state" / "candidate_feature_set.json").exists():
            fs = json.loads(p.read_text())
            if fs.get("signed_off"):
                ev.append({"t": round(end, 1), "type": "gate_freeze", **fs["signed_off"], "set_hash": fs.get("set_hash"),
                           "candidates": [c["name"] for c in fs["candidates"]]})
        if agent == "data_research" and (p := run_dir / "state" / "dataset_card.json").exists():
            r = (json.loads(p.read_text()).get("research") or {})
            ev.append({"t": round(end, 1), "type": "brief", "observations": len(r.get("observations") or []),
                       "cleaning": len((r.get("cleaning") or {}).get("steps") or []),
                       "warnings": (r.get("warnings") or [])[:3]})
        if agent == "auditor" and (p := run_dir / "state" / "audit_report.json").exists():
            r = json.loads(p.read_text())
            for f in r["findings"]:
                ev.append({"t": round(end, 1), "type": "finding", "line": f["location"]["line"],
                           "leak_type": f["leak_type"], "description": f["description"],
                           "impact_before": f["estimated_impact"]["before"],
                           "impact_after": f["estimated_impact"]["after"], "patch": f["proposed_patch"],
                           "verified_by_run": f.get("verified_by_run"),
                           "verified_share": f.get("verified_share_before_tradable")})
            if not r["findings"]:
                ev.append({"t": round(end, 1), "type": "finding_none", "summary": r["summary"]})
    ev.append({"t": ev[-1]["t"], "type": "run_end", "status": meta["status"]})
    return sorted(ev, key=lambda e: e["t"])  # stable: equal times keep their order
