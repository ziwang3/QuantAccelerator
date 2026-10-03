"""Build the self-contained QuantAccelerator presentation page from finished runs.

  python -m quantaccelerator.viz --form4 runs/batch2 --compare runs/batch1 [--alfred runs/macro1 --alfred-compare runs/macro-proto]
                    [--models runs/models_r3] [--alfred-models runs/models-macro] [--oos runs/oos-2024q1]
                    [--research RUN --research-bench RUN ...] [--eda RUN --eda-bench RUN ...]
                    [--predict RUN --predict-bench RUN ...] [--out runs/quantaccelerator.html]

Each case study is collected in its own subprocess (the dataset profile is fixed per process); the page is one file
with the data embedded and no network requests, so it works offline and as an attachment.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from quantaccelerator.paths import ROOT, RUNS

TEMPLATE = Path(__file__).with_name("template.html")
PLACEHOLDER = "/*__DATA__*/"


def _collect(profile: str, batch: str, compare: list[str], extra_env: dict | None = None) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.json"
        cmd = [sys.executable, "-m", "quantaccelerator.viz.collect", batch, "--out", str(out)] + (["--compare", *compare] if compare else [])
        subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "QA_DATASET": profile, **(extra_env or {})})
        return json.loads(out.read_text())


def _scores(batch: Path) -> dict:
    return {d.name: json.loads((d / "score.json").read_text()) for d in sorted(batch.iterdir())
            if (d / "score.json").exists()} if batch.exists() else {}


def _models(root: str, profile: str) -> dict:
    """Model-size summary for one case study (gold leak flags from that case study's answer key)."""
    from quantaccelerator.datasets import get
    from quantaccelerator.eval.models import summarize
    gold = json.loads((get(profile).benchmark / "gold.json").read_text())
    return summarize(ROOT / root, {v: g["leak"] for v, g in gold["variants"].items()})


def _research(run: str, bench: list[str]) -> dict:
    """A Data Research run for the page: its brief, the figures its findings cite (PNG, embedded), benchmark scores."""
    import base64
    d = ROOT / run
    brief = json.loads((d / "brief" / "brief.json").read_text())
    figs = {o["figure"] for o in brief["observations"] if o.get("figure")}
    scores = [json.loads((ROOT / b / "research_score.json").read_text()) for b in bench
              if (ROOT / b / "research_score.json").exists()]
    from quantaccelerator.viz.events import run_events
    return {"run": Path(run).name, "brief": brief, "events": run_events(d),
            "figures": {f: base64.b64encode((d / f).read_bytes()).decode() for f in figs if (d / f).exists()},
            "benchmarks": scores}


def _eda(run: str, bench: list[str]) -> dict:
    """A Research EDA run for the page: its Feature Report data, the figures its findings cite (PNG, embedded, each
    once), and the scored planted-dependency runs (the first one's candidate set is shown as the example)."""
    import base64
    d = ROOT / run
    rep = json.loads((d / "brief" / "features.json").read_text())
    figs = {o["figure"] for o in rep["observations"] if o.get("figure")}
    scores = []
    for b in bench:
        bd = ROOT / b
        if (bd / "score_eda.json").exists():
            sc = json.loads((bd / "score_eda.json").read_text())
            scores.append({"run": bd.name, "passed": sc["passed"], "score": sc["score"],
                           "n_candidates": sc["n_candidates"], "translation": sc.get("translation", {})})
    from quantaccelerator.viz.events import run_events
    showcase = None
    if scores:
        sr = json.loads((ROOT / bench[0] / "brief" / "features.json").read_text())
        showcase = {"run": Path(bench[0]).name, "candidates": sr["candidates"], "panel": sr["panel"],
                    "events": run_events(ROOT / bench[0])}
    return {"run": Path(run).name, "report": rep, "benchmarks": scores, "showcase": showcase, "events": run_events(d),
            "figures": {f: base64.b64encode((d / f).read_bytes()).decode() for f in figs if (d / f).exists()}}


def _predict(run: str, bench: list[str]) -> dict:
    """A Research EDA phase B run for the page: its Predictive Report data (aggregate statistics only: no return rows
    of the licensed price data), the decay and quantile figures it cites, and the scored planted-returns repeats."""
    import base64
    d = ROOT / run
    rep = json.loads((d / "brief" / "predictive.json").read_text())
    figs = {f for fs in rep.get("figures", {}).values() for f in fs}
    scores = []
    for b in bench:
        bd = ROOT / b
        if (bd / "score_pred.json").exists():
            sc = json.loads((bd / "score_pred.json").read_text())
            scores.append({"run": bd.name, "passed": sc["passed"], "score": sc["score"],
                           "n_trials": (sc.get("ledger") or {}).get("n_trials")})
    return {"run": Path(run).name, "report": rep, "benchmarks": scores,
            "figures": {f: base64.b64encode((d / f).read_bytes()).decode() for f in figs if (d / f).exists()}}


def build(form4: str, compare: list[str], alfred: str | None = None, models: str | None = None,
          oos: str | None = None, out: Path | None = None, alfred_compare: list[str] | None = None,
          alfred_models: str | None = None, research: str | None = None, research_bench: list[str] | None = None,
          eda: str | None = None, eda_bench: list[str] | None = None, predict: str | None = None,
          predict_bench: list[str] | None = None) -> Path:
    modules = [_collect("form4", form4, compare)]
    if models:
        modules[0]["models"] = _models(models, "form4")
    if alfred:
        modules.append(_collect("alfred", alfred, alfred_compare or []))
        if alfred_models:
            modules[-1]["models"] = _models(alfred_models, "alfred")
    data = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "modules": modules}
    if research:
        data["research"] = _research(research, research_bench or [])
    if eda:
        data["eda"] = _eda(eda, eda_bench or [])
    if predict:
        data["predict"] = _predict(predict, predict_bench or [])
    if oos:
        quarter = Path(oos).name.split("-")[-1]
        gold = json.loads((ROOT / "data" / "benchmark" / f"demo_{quarter}" / "gold.json").read_text())
        data["oos"] = {"quarter": quarter, "batch": Path(oos).name, "scores": _scores(ROOT / oos),
                       "gold": {v: {"leak": g["leak"], "true_lookahead": g["true_lookahead"]}
                                for v, g in gold["variants"].items()},
                       "strata": gold.get("rule_sample_strata")}
    blob = json.dumps(data, default=str, separators=(",", ":")).replace("</", "<\\/")  # "</" could end the script
    html = TEMPLATE.read_text()
    assert html.count(PLACEHOLDER) == 1
    out = out or RUNS / "quantaccelerator.html"
    out.write_text(html.replace(PLACEHOLDER, blob))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--form4", required=True, help="Form 4 batch directory, e.g. runs/batch2")
    ap.add_argument("--compare", nargs="*", default=[], help="older Form 4 batches for before/after deltas")
    ap.add_argument("--alfred", help="Revision Audit batch directory")
    ap.add_argument("--alfred-compare", nargs="*", default=[], help="older Revision Audit batches (e.g. the prototype)")
    ap.add_argument("--models", help="Form 4 model-comparison root, e.g. runs/models_r3")
    ap.add_argument("--alfred-models", help="Revision Audit model-comparison root, e.g. runs/models-macro")
    ap.add_argument("--oos", help="out-of-sample Form 4 batch, e.g. runs/oos-2024q1")
    ap.add_argument("--research", help="a Data Research run (python -m quantaccelerator.demo --understand), e.g. runs/eval-...-finra")
    ap.add_argument("--research-bench", nargs="*", default=[], help="scored benchmark runs (finra_defects, finra_small)")
    ap.add_argument("--eda", help="a Research EDA run (python -m quantaccelerator.demo --explore), e.g. runs/explore-real2-finra")
    ap.add_argument("--eda-bench", nargs="*", default=[], help="scored finra_eda runs; the first is shown as the example")
    ap.add_argument("--predict", help="a Research EDA phase B run (python -m quantaccelerator.demo --predict), e.g. "
                    "runs/predict-real3-finra")
    ap.add_argument("--predict-bench", nargs="*", default=[], help="scored finra_pred runs (planted returns)")
    ap.add_argument("--out", help="output path (default runs/quantaccelerator.html)")
    a = ap.parse_args(argv)
    p = build(a.form4, a.compare, a.alfred, a.models, a.oos, Path(a.out) if a.out else None, a.alfred_compare,
              a.alfred_models, a.research, a.research_bench, a.eda, a.eda_bench, a.predict, a.predict_bench)
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
