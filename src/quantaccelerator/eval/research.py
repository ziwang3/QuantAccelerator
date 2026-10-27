"""Score Data Research runs (runs made with `python -m quantaccelerator.demo --understand`).

  python -m quantaccelerator.eval.research <run_dir> [...]    -> <run_dir>/research_score.json, one line per run on stdout

What is scored depends on the dataset profile the run used (run_meta.json "dataset"):
  finra_defects  planted defects detected in the findings; the recipe removes duplicates and handles impossible rows
                 without dropping clean rows
  finra_small    the capability envelope rates history and cross-sectional breadth as weak
  any profile    field meanings (the profile's semantic answer key, card questions only), and whether every numeric
                 field got a kind and a zero-vs-missing statement
"""
import json
import sys
from pathlib import Path

from quantaccelerator.datasets import get
from quantaccelerator.state.schemas import DatasetCard

NUMERIC_KINDS = {"level", "flow", "stock", "event", "estimate", "revision", "score", "ratio"}


def score(run_dir: Path) -> dict:
    import os
    run_dir = Path(run_dir)
    card = DatasetCard.model_validate_json((run_dir / "state" / "dataset_card.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    name = meta["dataset"]
    out = {"run": run_dir.name, "profile": name, "status": meta.get("status"),
           "agents": {a: {k: v.get(k) for k in ("wall_s", "tool_calls", "submit_failures", "paced_calls")}
                      for a, v in meta.get("agents", {}).items()}}
    if name in ("finra_defects", "finra_small"):
        from quantaccelerator.eval.gold_finra_defects import score_research
        out["benchmark"] = score_research(card, name)
    base = "finra" if name.startswith("finra") else name
    gold = json.loads((get(base).benchmark / "gold.json").read_text())
    from quantaccelerator.eval.score_demo import grade_semantic
    os.environ["QA_DATASET"] = base
    sem = [q for q in grade_semantic(card, None, {"semantic": [q for q in gold["semantic"]
                                                              if not q["target"].startswith(("rule:", "pit_role:"))]})]
    out["semantic_accuracy"] = round(sum(q["correct"] for q in sem) / max(len(sem), 1), 3)
    out["semantic_missed"] = [q["id"] for q in sem if not q["correct"]]
    num = [f for f in card.fields if f.kind in NUMERIC_KINDS]
    out["numeric_fields_with_zero_vs_missing"] = f"{sum(bool(f.zero_vs_missing) for f in num)}/{len(num)}"
    (run_dir / "research_score.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(score(Path(d)), default=str))
