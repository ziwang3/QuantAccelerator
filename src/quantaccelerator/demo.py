"""CLI: python -m quantaccelerator.demo --pipeline notebooks/<variant>.py [--oracle] [--run-id ID]
        python -m quantaccelerator.demo --all [--oracle] [--batch NAME]      # all 4 variants + batch report
        python -m quantaccelerator.demo --all --oracle --batch NAME --variants leak_noclose   # re-run some variants, re-score batch
        python -m quantaccelerator.demo --understand [--run-id ID]    # Data Research Agent only: DatasetCard + findings + Data Brief
        python -m quantaccelerator.demo --explore [--idea TEXT ...] [--pit agent] [--run-id ID]   # Research EDA: ideas (G2), features (G3)
        python -m quantaccelerator.demo --predict [--from runs/<explore run>] [--oracle] [--batch NAME]   # phase B: plan (G4),
            discovery, validation, release (G5), locked test; default features: the profile's oracle set
        add --research to an audit to run the Data Research profiling pass before the PIT agent
"""
import argparse
import datetime as dt
import json
import sys
import traceback

from quantaccelerator.agents.orchestrator import run_demo, run_explore, run_predict, run_understand
from quantaccelerator.eval.score_demo import score_batch, score_run
from quantaccelerator.llm.client import Budget, LLMClient
from quantaccelerator.paths import RUNS

from quantaccelerator.datasets import active

VARIANTS = active().variants


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pipeline", help="researcher script to audit, e.g. notebooks/insider_events_clean.py")
    g.add_argument("--all", action="store_true", help="audit all 4 demo variants")
    g.add_argument("--understand", action="store_true", help="run only the Data Research Agent (both passes)")
    g.add_argument("--explore", action="store_true", help="Research EDA phase A on the PIT research panel (G2 by "
                   "oracle: frozen as proposed)")
    g.add_argument("--predict", action="store_true", help="Research EDA phase B on a frozen feature set (returns vault)")
    ap.add_argument("--from", dest="from_run", default="oracle",
                    help="--predict: an explore run whose frozen set to test (default: the profile's oracle set)")
    ap.add_argument("--idea", action="append", default=None,
                    help="--explore: a researcher's feature idea in plain words (repeatable); default: the profile's")
    ap.add_argument("--pit", choices=["oracle", "agent"], default="oracle",
                    help="--explore: sign the gold rule (oracle) or run the PIT agent with G1 by oracle")
    ap.add_argument("--research", action="store_true", help="audits: also run the profiling pass")
    ap.add_argument("--oracle", action="store_true", help="answer gate G1 from gold instead of prompting")
    ap.add_argument("--run-id")
    ap.add_argument("--batch", help="batch directory name under runs/ (with --all)")
    ap.add_argument("--model", help="served model name (default: $LLM_MODEL)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS,
                    help="with --all: only (re-)run these variants; the batch report still covers every run in it")
    a = ap.parse_args(argv)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    client = lambda: LLMClient(budget=Budget(), model=a.model, temperature=a.temperature, seed=a.seed)

    if a.understand:
        run_id = a.run_id or f"{stamp}-understand-{active().name}"
        run_understand(run_id, client=client())
        print(f"\nData Brief: {RUNS / run_id / 'brief' / 'brief.html'}")
        return
    if a.explore:
        run_id = a.run_id or f"{stamp}-explore-{active().name}"
        # two passes of up to 30 + 60 steps at ~12k prompt tokens each: the default 600k guard is too small
        big = LLMClient(budget=Budget(max_calls=200, max_prompt_tokens=1_500_000, max_completion_tokens=200_000),
                        model=a.model, temperature=a.temperature, seed=a.seed)
        fs = run_explore(run_id, client=big, pit=a.pit, researcher_ideas=a.idea)
        print(f"\n{len(fs.candidates)} candidates frozen (set {fs.set_hash}); Feature Report: "
              f"{RUNS / run_id / 'brief' / 'features.html'}")
        return
    if a.predict:
        run_id = a.run_id or (f"predict-{a.batch}-{active().name}" if a.batch else f"{stamp}-predict-{active().name}")
        big = LLMClient(budget=Budget(max_calls=120, max_prompt_tokens=1_500_000, max_completion_tokens=150_000),
                        model=a.model, temperature=a.temperature, seed=a.seed)
        rep = run_predict(run_id, client=big, oracle=a.oracle, features=a.from_run)
        for o in rep.observations:
            print(f"  {o.feature:<16} h={o.horizon:<3} {str(o.controls):<24} {o.verdict:<22} {o.status}")
        print(f"\ntrials {rep.ledger.get('n_trials')}, hurdle {rep.ledger.get('hurdle_t')}; report: "
              f"{RUNS / run_id / 'state' / 'predictive_report.json'}")
        return
    if a.pipeline:
        run_id = a.run_id or f"{stamp}-{active().variant_of(a.pipeline)}"
        run_demo(a.pipeline, a.oracle, run_id, client=client(), research=a.research)
        r = score_run(RUNS / run_id)
        print(f"\nreport: {RUNS / run_id / 'report.md'}\n" + json.dumps(r["leak"], indent=1))
        return
    batch = RUNS / (a.batch or f"batch-{stamp}")
    batch.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        pipeline = active().pipeline(v)
        run_id = f"{str(batch.relative_to(RUNS)).replace('/', '-')}-{v}"  # unique across nested batches
        print(f"\n=== {v} ({run_id}) ===", flush=True)
        try:
            meta = run_demo(pipeline, a.oracle, run_id, run_dir=batch / v, client=client(), research=a.research)
            print(f"status {meta['status']}  budget {meta['budget']}", flush=True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
    score_batch(batch)
    print((batch / "report.md").read_text())


if __name__ == "__main__":
    main()
