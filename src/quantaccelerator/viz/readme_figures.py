"""The README figures, made from recorded runs:

  python -m quantaccelerator.viz.readme_figures models runs/models-lg/form4 runs/models-lg/alfred --out docs/assets/model_size.png
  QA_DATASET=finra_eda python -m quantaccelerator.viz.readme_figures ideas runs/<explore run>-finra_eda --out docs/assets/eda_ideas.png

models: audit runs completed per model and case study (4 pipelines x the seeds of each model directory).
ideas: for each researcher idea of the idea-translation benchmark, the dependence of the raw field on what the idea must
not depend on (liquidity, sector) next to that of the feature the agent built and froze, measured as the scorer does.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantaccelerator.paths import ROOT  # noqa: E402

BG, PANEL, INK, MUTED, GRID = "#111827", "#111827", "#e5e7eb", "#9ca3af", "#1f2937"
BLUE, ORANGE = "#3987e5", "#d95926"
MODEL_LABEL = {"qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-72b": "Qwen2.5-72B"}


def _style(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=15, length=0)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def models(form4: str, alfred: str, out: Path) -> dict:
    from quantaccelerator.viz.build import _models
    summ = {"form4": _models(form4, "form4"), "alfred": _models(alfred, "alfred")}
    names = [m for m in MODEL_LABEL if m in summ["form4"] or m in summ["alfred"]]
    done = {ds: {m: (sum(s["runs_ok"] for s in summ[ds][m]["seeds"].values()),
                     sum(s["runs"] for s in summ[ds][m]["seeds"].values())) if m in summ[ds] else (0, 0)
                 for m in names} for ds in summ}
    fig, ax = plt.subplots(figsize=(16, 8.4), dpi=100)
    _style(fig, ax)
    w, top = 0.34, max(n for ds in done.values() for _, n in ds.values()) or 12
    for j, (ds, label, color) in enumerate([("form4", "Look-Ahead Audit (SEC Form 4)", BLUE),
                                            ("alfred", "Revision Audit (macro vintages)", ORANGE)]):
        xs = [i + (j - 0.5) * (w + 0.04) for i in range(len(names))]
        ok = [done[ds][m][0] for m in names]
        ax.bar(xs, ok, w, color=color, label=label)
        for x, m in zip(xs, names):
            k, n = done[ds][m]
            ax.text(x, k + top * 0.02, f"{k}/{n}", ha="center", va="bottom", color=INK, fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(names)), [MODEL_LABEL[m] for m in names], fontsize=16, color=MUTED)
    ax.set_ylim(0, top * 1.15)
    ax.set_yticks(range(0, top + 1, 4))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"audit runs completed (of {top})", color=MUTED, fontsize=15)
    ax.legend(loc="upper left", frameon=False, fontsize=15, labelcolor=MUTED)
    fig.text(0.02, 0.95, "Model size: audit runs completed on the same prompts, tools and budgets", color=INK,
             fontsize=20, fontweight="bold")
    fig.text(0.075, 0.875, "4 pipelines × 3 seeds per case study, temperature 0.3", color=MUTED, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.8, bottom=0.08)
    fig.savefig(out, facecolor=BG)   # explicit: viz/figures.py sets a white savefig default
    plt.close(fig)
    return done


IDEA_ROWS = [  # benchmark idea, title, raw field, what it must not depend on, short description of the fix
    ("I1", "Abnormal short selling", "ShortVolume", "adv_proxy", "vs liquidity"),
    ("I2", "Attention per unit of activity", "vendor_mentions", "adv_proxy", "vs liquidity"),
    ("I3", "Sentiment vs sector peers", "vendor_sentiment", "sector", "vs sector"),
    ("I4", "Quality, a slow characteristic", "vendor_quality", None, "vs anything"),
]


def ideas(run: str, out: Path) -> dict:
    import pandas as pd

    from quantaccelerator.eval import gold_eda
    from quantaccelerator.state.schemas import CandidateFeatureSet, FeatureSpec
    from quantaccelerator.tools.features import exposure_stats, materialize
    run_dir = ROOT / run
    score = json.loads((run_dir / "score_eda.json").read_text())
    fs = CandidateFeatureSet.model_validate_json((run_dir / "state" / "candidate_feature_set.json").read_text())
    panel, _ = gold_eda._panel()
    series = lambda specs, name: pd.Series(materialize(specs, [name], panel)[name].values, index=panel.index)

    def dep(x, by):
        if by is None:   # the strongest of the dependences the benchmark checks
            return max(dep(x, "adv_proxy"), dep(x, "size_float"), dep(x, "sector"))
        st = exposure_stats(panel, x, by)
        return abs(st.get("eta_squared_mean") if by == "sector" else st.get("rank_corr_mean") or 0) or 0.0

    rows = []
    for k, title, raw, by, vs in IDEA_ROWS:
        t = score["translation"][k]
        raw_x = series([FeatureSpec(name="raw", op="field", inputs=[raw])], "raw")
        built = t["best"]
        built_x = series(fs.features, built) if built else None
        rows.append({"idea": k, "title": title, "vs": vs, "built": built, "raw": round(dep(raw_x, by), 2),
                     "feature": round(dep(built_x, by), 2) if built else None,
                     "rank_corr": t["rank_corr_with_reference"].get(built) if built else None})
    fig, ax = plt.subplots(figsize=(20, 9.6), dpi=100)
    _style(fig, ax)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        if r["feature"] is not None:
            ax.plot([r["feature"], r["raw"]], [y, y], color="#6b7280", linewidth=3, zorder=1)
            ax.scatter([r["feature"]], [y], s=260, color=BLUE, zorder=3)
        ax.scatter([r["raw"]], [y], s=220, color=ORANGE, zorder=2)
        ax.text(max(r["raw"], r["feature"] or 0) + 0.04, y,
                f"{r['raw']:.2f} → {r['feature']:.2f}  ({r['built']})" if r["feature"] is not None else
                f"{r['raw']:.2f} → not built", va="center", color=MUTED, fontsize=15)
        ax.text(-0.06, y + 0.18, r["title"], ha="right", va="center", color=INK, fontsize=17, fontweight="bold",
                transform=ax.get_yaxis_transform())
        ax.text(-0.06, y - 0.2, r["vs"], ha="right", va="center", color="#6b7280", fontsize=14,
                transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("dependence on liquidity or sector (0 = none)", color=MUTED, fontsize=16)
    ax.scatter([], [], s=220, color=ORANGE, label="raw field")
    ax.scatter([], [], s=260, color=BLUE, label="feature built from the idea")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=15, labelcolor=MUTED)
    corrs = [r["rank_corr"] for r in rows if r["rank_corr"] is not None]
    fig.text(0.02, 0.95, "Research EDA: the researcher's ideas, built as meant and made comparable", color=INK,
             fontsize=20, fontweight="bold")
    fig.text(0.02, 0.875, "idea-translation benchmark: raw field vs the feature built and frozen"
             + (f" (each matches its reference, rank corr. {'' if min(corrs) >= 0.995 else '≥ '}{min(corrs):.2f})"
                if corrs else ""),
             color=MUTED, fontsize=15)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.78, bottom=0.13)
    fig.savefig(out, facecolor=BG)   # explicit: viz/figures.py sets a white savefig default
    plt.close(fig)
    return {"run": Path(run).name, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["models", "ideas"])
    ap.add_argument("runs", nargs="+", help="models: the Form 4 and ALFRED model roots; ideas: one finra_eda run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = models(*a.runs, Path(a.out)) if a.figure == "models" else ideas(a.runs[0], Path(a.out))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
