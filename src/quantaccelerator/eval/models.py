"""Aggregate the model-comparison matrix: runs/models/<model>/s<seed>/<variant>/score.json -> per-model summary.

Per (model, seed) the batch-level metrics are computed as in the scorecard; per model we report the mean and the
min/max across seeds, plus the failure rate. Usage: python -m quantaccelerator.eval.models [runs/models]
"""
import json
import sys
from pathlib import Path

import numpy as np

from quantaccelerator.paths import RUNS

MODEL_ORDER = ["qwen2.5-7b", "qwen2.5-14b", "qwen2.5-72b"]


def batch_metrics(seed_dir: Path, gold_leak: dict[str, bool]) -> dict:
    scores = {d.name: json.loads((d / "score.json").read_text()) for d in seed_dir.iterdir()
              if (d / "score.json").exists()}
    metas = {d.name: json.loads((d / "run_meta.json").read_text()) for d in seed_dir.iterdir()
             if (d / "run_meta.json").exists()}
    leaky = [v for v, leak in gold_leak.items() if leak]
    ok = lambda v: v in scores and scores[v]["status"] == "ok"
    mean = lambda xs: float(np.mean(xs)) if xs else None
    cost = [sum((c.get("wall_s") or 0) for c in s["cost"].values()) / 60 for s in scores.values()]
    calls = [sum(c.get("llm_calls") or 0 for c in s["cost"].values()) for s in scores.values()]
    return {
        "runs": len(metas), "runs_ok": sum(ok(v) for v in gold_leak),
        "leaks_found": sum(ok(v) and scores[v]["leak"]["detected"] and scores[v]["leak"].get("line_correct", False)
                           for v in leaky),
        "fixes_verified": sum(ok(v) and bool(scores[v]["leak"].get("patch_verified")) for v in leaky),
        "false_alarms": sum(ok(v) and scores[v]["leak"]["n_findings"] > 0 for v, leak in gold_leak.items() if not leak),
        "semantic": mean([scores[v]["semantic_accuracy"] for v in gold_leak if ok(v)]),
        "rule": mean([scores[v]["rule_correctness"]["overall"] for v in gold_leak
                      if ok(v) and scores[v].get("rule_correctness")]),
        "g1_approved": sum(ok(v) and bool((scores[v].get("g1") or {}).get("approved")) for v in gold_leak),
        "minutes_per_run": mean(cost), "llm_calls_per_run": mean(calls),
        "n_leaky": len(leaky), "n_variants": len(gold_leak),
    }


def summarize(root: Path = RUNS / "models", gold_leak: dict[str, bool] | None = None) -> dict:
    if gold_leak is None:
        from quantaccelerator.eval.score_demo import load_gold
        gold_leak = {v: g["leak"] for v, g in load_gold()["variants"].items()}
    out = {}
    models = sorted((d for d in root.iterdir() if d.is_dir()),
                    key=lambda d: MODEL_ORDER.index(d.name) if d.name in MODEL_ORDER else 99) if root.exists() else []
    for m in models:
        seeds = {s.name: batch_metrics(s, gold_leak) for s in sorted(m.glob("s*")) if s.is_dir()}
        seeds = {k: x for k, x in seeds.items() if x["runs"] == len(gold_leak)}  # only complete seeds
        if not seeds:
            continue
        agg = {}
        for k in next(iter(seeds.values())):
            xs = [x[k] for x in seeds.values() if x[k] is not None]
            agg[k] = {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs))} if xs else None
        out[m.name] = {"n_seeds": len(seeds), "seeds": seeds, "summary": agg}
    return out


def render(summary: dict) -> str:
    L = ["# Model comparison", "", "Mean over seeds (min–max in brackets). Same prompts, tools, budgets and quantization.",
         "", "| model | seeds | runs ok | leaks found | fixes verified | false alarms | field meanings | rule | "
         "G1 approved | min / run | LLM calls / run |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda a, pct=False, d=1: "–" if a is None else (
        f"{100 * a['mean']:.0f}% [{100 * a['min']:.0f}–{100 * a['max']:.0f}]" if pct else
        f"{a['mean']:.{d}f} [{a['min']:.{d}f}–{a['max']:.{d}f}]")
    for m, x in summary.items():
        s = x["summary"]
        L.append(f"| {m} | {x['n_seeds']} | {f(s['runs_ok'])} | {f(s['leaks_found'])} / {s['n_leaky']['mean']:.0f} | "
                 f"{f(s['fixes_verified'])} | {f(s['false_alarms'])} | {f(s['semantic'], True)} | {f(s['rule'], True)} | "
                 f"{f(s['g1_approved'])} | {f(s['minutes_per_run'])} | {f(s['llm_calls_per_run'], d=0)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS / "models"
    s = summarize(root)
    (root / "summary.json").write_text(json.dumps(s, indent=1))
    (root / "report.md").write_text(render(s))
    print(render(s))
