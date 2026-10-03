"""Score demo runs against data/benchmark/demo/gold.json and write report.md files.

Metrics (Demo_guide S7): semantic accuracy, availability-rule correctness on the stratified 500-filing sample,
leak detection / localization / type / impact error / patch verification per variant, cost per agent.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from quantaccelerator.datasets import active
from quantaccelerator.paths import ROOT
from quantaccelerator.state.schemas import AuditReport, AvailabilityRule, DatasetCard, PITContract
from quantaccelerator.tools import registry


def load_gold() -> dict:
    return json.loads((active().benchmark / "gold.json").read_text())


def _norm(col: str) -> str:
    return col.split(".")[-1].upper()


def _match(text: str, groups: list[list[str]]) -> bool:
    t = (text or "").lower()
    return all(any(k in t for k in g) for g in groups)


def grade_semantic(card: DatasetCard | None, contract: PITContract | None, gold: dict) -> list[dict]:
    out = []
    for q in gold["semantic"]:
        kind, _, target = q["target"].partition(":")
        answer, ok = None, False
        if card is not None and kind == "field":
            table, col = target.split(".")
            fc = [f for f in card.fields if _norm(f.name) == col.upper() and f.table.lower() in (table, "")] or \
                 [f for f in card.fields if _norm(f.name) == col.upper()]
            answer = fc[0].meaning if fc else None
            ok = bool(fc) and _match(answer, q["groups"])
        elif card is not None and kind == "code":
            col, code = target.split(":")
            fc = [f for f in card.fields if _norm(f.name) == col and code in f.value_meanings]
            answer = fc[0].value_meanings[code] if fc else None
            ok = bool(fc) and _match(answer, q["groups"])
        elif card is not None and kind == "table":
            tc = [t for t in card.tables if t.table.lower() == target]
            answer = f"{tc[0].observation_unit} | key={tc[0].primary_key}" if tc else None
            ok = bool(tc) and _match(tc[0].observation_unit, q["groups"]) and \
                set(q.get("key_must_include", [])) <= {_norm(k) for k in tc[0].primary_key}
        elif card is not None and kind == "entity_ids":
            got = {_norm(x) for x in card.entity_id_fields}
            answer, ok = sorted(got), set(q["gold"]) <= got
        elif contract is not None and kind == "rule":
            answer = getattr(contract.availability_rule, target)
            ok = answer == getattr(active().gold_rule(), target) and (target != "tradable_at" or
                                                          contract.availability_rule.extra_lag_trading_days == 0)
        elif contract is not None and kind == "pit_role":
            pf = [f for f in contract.fields if _norm(f.field) == _norm(target)]
            answer = pf[0].role if pf else None
            ok = answer == q["gold"]
        out.append({"id": q["id"], "question": q["question"], "answer": answer, "gold": q["gold"], "correct": ok})
    return out


def rule_by_stratum(rule: AvailabilityRule) -> dict:
    return active().rule_by_stratum(rule)


def variant_of(pipeline: str) -> str:
    return active().variant_of(pipeline)


def score_leak(run_dir: Path, report: AuditReport | None, gold: dict, variant: str) -> dict:
    g = gold["variants"][variant]
    findings = report.findings if report else []
    res = {"variant": variant, "gold_leak": g["leak"], "n_findings": len(findings), "detected": bool(findings)}
    if not g["leak"]:
        res["false_positive"] = bool(findings)
        return res
    true_share = g["true_lookahead"]["share_before_tradable"]
    lines = [f.location.line for f in findings]
    res.update({
        "gold_line": g["line"], "lines": lines, "line_correct": g["line"] in lines,
        "type_correct": any(f.leak_type in g["leak_types"] for f in findings if f.location.line == g["line"]),
        "leak_types": [f.leak_type for f in findings],
        "impact_estimate": findings[0].estimated_impact.before if findings else None, "impact_true": true_share,
        "impact_abs_error": round(abs(findings[0].estimated_impact.before - true_share), 4) if findings else None,
    })
    # patch verification against the *gold* rule (independent of whatever contract the agent was given)
    patched = run_dir / "patched.py"
    res["patch_verified"] = False
    if patched.exists():
        out = run_dir / "pipeline_outputs" / "patched_gold_check.parquet"
        p = subprocess.run([sys.executable, str(patched)], cwd=ROOT, capture_output=True, text=True,
                           env={**__import__("os").environ, "QA_OUT": str(out)})
        if p.returncode == 0:
            st = active().measure(pd.read_parquet(out), active().gold_rule())
            res["patched_share_before_tradable_gold"] = st["share_before_tradable"]
            res["patched_n_events"] = st["n_events"]
            res["patch_verified"] = st["share_before_tradable"] == 0 and st["n_events"] == g["true_lookahead"]["n_events"]
        else:
            res["patch_error"] = p.stderr[-300:]
    return res


def cost(run_dir: Path, meta: dict) -> dict:
    calls = pd.read_json(run_dir / "llm_calls.jsonl", lines=True) if (run_dir / "llm_calls.jsonl").exists() \
        else pd.DataFrame(columns=["agent", "tokens_in", "tokens_out", "latency_s"])
    tools = registry.session_runs(meta["run_id"])
    out = {}
    for a, m in meta.get("agents", {}).items():
        c = calls[calls.agent == a]
        out[a] = {"llm_calls": len(c), "tokens_in": int(c.tokens_in.sum()), "tokens_out": int(c.tokens_out.sum()),
                  "tool_calls": int((tools.agent == a).sum()), "wall_s": m.get("wall_s"),
                  "submit_failures": m.get("submit_failures")}
    return out


def score_run(run_dir: Path, gold: dict | None = None) -> dict:
    gold = gold or load_gold()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    st = run_dir / "state"
    load = lambda name, cls: cls.model_validate_json((st / f"{name}.json").read_text()) \
        if (st / f"{name}.json").exists() else None
    card, contract, report = load("dataset_card", DatasetCard), load("pit_contract", PITContract), \
        load("audit_report", AuditReport)
    sem = grade_semantic(card, contract, gold)
    res = {"run_id": meta["run_id"], "variant": variant_of(meta["pipeline"]), "status": meta["status"],
           "semantic": sem, "semantic_accuracy": round(sum(q["correct"] for q in sem) / len(sem), 4),
           "rule": contract.availability_rule.model_dump() if contract else None,
           "rule_correctness": rule_by_stratum(contract.availability_rule) if contract else None,
           "g1": contract.signed_off.model_dump() if contract and contract.signed_off else None,
           "leak": score_leak(run_dir, report, gold, variant_of(meta["pipeline"])),
           "cost": cost(run_dir, meta)}
    (run_dir / "score.json").write_text(json.dumps(res, indent=1, default=str))
    (run_dir / "report.md").write_text(render_run(res, report))
    return res


def render_run(r: dict, report: AuditReport | None) -> str:
    L = [f"# Demo run `{r['run_id']}` — pipeline `{r['variant']}`", "", f"Status: **{r['status']}**", ""]
    L += ["## Semantic accuracy", "", f"**{r['semantic_accuracy']:.0%}** "
          f"({sum(q['correct'] for q in r['semantic'])}/{len(r['semantic'])})", "",
          "| id | question | agent answer | correct |", "|---|---|---|---|"]
    L += [f"| {q['id']} | {q['question']} | {str(q['answer'])[:90].replace('|', '/')} | {'✓' if q['correct'] else '✗'} |"
          for q in r["semantic"]]
    if r["rule"]:
        L += ["", "## Availability rule (PIT agent, before G1)", "", f"`{json.dumps(r['rule'])}`", "",
              "Exact first-tradable-date agreement with gold on the stratified 500-filing sample:", "",
              "| " + " | ".join(r["rule_correctness"]) + " |", "|" + "---|" * len(r["rule_correctness"]),
              "| " + " | ".join(f"{v:.1%}" for v in r["rule_correctness"].values()) + " |", "",
              f"G1: {r['g1']['decided_by']} {'approved' if r['g1']['approved'] else 'rejected'} — {r['g1']['note']}"]
    lk = r["leak"]
    L += ["", "## Leak audit", ""]
    L += [f"- {k}: {v}" for k, v in lk.items()]
    if report and report.findings:
        for f in report.findings:
            L += ["", f"**Finding** line {f.location.line} · `{f.leak_type}` · confidence {f.confidence}", "",
                  f.description, "", f"Impact: share of events used before tradable {f.estimated_impact.before} → "
                  f"{f.verified_share_before_tradable} (verified by `{f.verified_by_run}`)", "", "Patch:", "", "```python"]
            L += [f"# line {e.line}\n{e.new_code}" for e in f.proposed_patch] + ["```"]
    if report:
        L += ["", f"Auditor summary: {report.summary}"]
    L += ["", "## Cost", "", "| agent | LLM calls | tokens in | tokens out | tool calls | wall s | submit failures |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {a} | {c['llm_calls']} | {c['tokens_in']} | {c['tokens_out']} | {c['tool_calls']} | {c['wall_s']} | "
          f"{c['submit_failures']} |" for a, c in r["cost"].items()]
    return "\n".join(L) + "\n"


def render_batch(results: list[dict], title: str) -> str:
    L = [f"# {title}", "", "| variant | status | semantic acc | rule acc (overall) | G1 | detected | line ✓ | type ✓ | "
         "impact est / true | patch verified | FP |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lk = r["leak"]
        L.append(f"| {r['variant']} | {r['status'][:20]} | {r['semantic_accuracy']:.0%} | "
                 f"{(r['rule_correctness'] or {}).get('overall', float('nan')):.1%} | "
                 f"{'approved' if r['g1'] and r['g1']['approved'] else 'corrected' if r['g1'] else '-'} | "
                 f"{lk['detected']} | {lk.get('line_correct', '-')} | {lk.get('type_correct', '-')} | "
                 f"{lk.get('impact_estimate', '-')} / {lk.get('impact_true', '-')} | {lk.get('patch_verified', '-')} | "
                 f"{lk.get('false_positive', '-')} |")
    tot = {}
    for r in results:
        for a, c in r["cost"].items():
            t = tot.setdefault(a, {"llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "tool_calls": 0, "wall_s": 0.0})
            for k in t:
                t[k] += c.get(k) or 0
    L += ["", "## Cost (summed over runs)", "", "| agent | LLM calls | tokens in | tokens out | tool calls | wall s |",
          "|---|---|---|---|---|---|"]
    L += [f"| {a} | {t['llm_calls']} | {t['tokens_in']} | {t['tokens_out']} | {t['tool_calls']} | {t['wall_s']:.0f} |"
          for a, t in tot.items()]
    L += ["", "Per-run details: see `<run_id>/report.md`."]
    return "\n".join(L) + "\n"


def score_batch(batch_dir: Path) -> list[dict]:
    gold = load_gold()
    results = [score_run(d, gold) for d in sorted(batch_dir.iterdir()) if (d / "run_meta.json").exists()]
    (batch_dir / "report.md").write_text(render_batch(results, f"Demo batch `{batch_dir.name}`"))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a run dir (with run_meta.json) or a batch dir of runs")
    a = ap.parse_args()
    p = Path(a.path)
    if (p / "run_meta.json").exists():
        print(json.dumps(score_run(p)["leak"], indent=1))
    else:
        score_batch(p)
        print((p / "report.md").read_text())
