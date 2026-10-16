"""Research EDA tools, phase A (post-PIT, return-blind): construct features, characterize them, compare
representations, and record the findings.

Everything runs on the session's research panel (quantaccelerator.tools.panel), built from the PIT research view, so a feature can
only use what was knowable at its decision date. Features are built in a closed language (FeatureSpec): the agent
names an operation and its inputs, never code, so every feature is reproducible from its spec and auditable.
Characterization is cross-sectional and per date: a feature is compared with liquidity (adv_proxy), size
(size_float, size_assets) and sector on the same date, by rank, so the numbers do not depend on units or outliers.
No tool here can see returns, prices or P&L.

As in quantaccelerator.tools.explore, each tool returns a compact numeric summary for the agent and draws one figure for the human;
the statistics live in plain functions (`*_stats`) that take Series/DataFrames, so they can be tested on synthetic
panels. Findings are recorded one by one with the record tools, each checked on the spot.
"""
import re

import numpy as np
import pandas as pd

from quantaccelerator.state.schemas import CandidateFeature, FeatureObservation, FeatureSpec
from quantaccelerator.tools.explore import _d, _r, flag, level_shifts, variance_split_stats
from quantaccelerator.tools.panel import CATEGORICAL, CONDITIONERS, STATE, get_panel
from quantaccelerator.tools.registry import CTX, inline_refs, tool
from quantaccelerator.viz import figures as F

CACHE: dict[str, pd.Series] = {}   # feature name -> values aligned to the panel index (this session)
NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
UNARY = {"field", "log1p", "change", "growth", "surprise", "rolling_mean", "rolling_sum", "peer_relative",
         "residualize", "rank"}
WINDOWED = {"change", "growth", "surprise", "rolling_mean", "rolling_sum", "rolling_std"}
BINARY = {"ratio", "diff"}
NEUTRALIZING = {"peer_relative", "residualize"}
STRONG_RHO, MODERATE_RHO = 0.3, 0.1      # |mean same-date rank correlation|
STRONG_ETA, MODERATE_ETA = 0.10, 0.03    # share of same-date rank variance explained by sector (bias-adjusted)
TERMS = {"adv_proxy": ["adv", "liquid", "volume", "trading activity", "turnover"],
         "size_float": ["size", "float", "market cap", "large", "small", "big"],
         "size_assets": ["size", "assets", "balance sheet", "large", "small", "big"],
         "sector": ["sector", "industr"]}


def _features() -> dict[str, FeatureSpec]:
    return CTX.findings.setdefault("features", {})


# ---------------------------------------------------------------- construction
def _value_fields() -> list[str]:
    return list((STATE["meta"] or {}).get("values", []))


def resolve(name: str, panel: pd.DataFrame | None = None, specs: dict | None = None) -> pd.Series:
    """Values of a panel field, a numeric conditioner, or a constructed feature."""
    panel = get_panel() if panel is None else panel
    specs = _features() if specs is None else specs
    if name in specs and not _is_raw(specs[name]):
        if name not in CACHE:
            CACHE[name] = compute(specs[name], panel, lambda n: resolve(n, panel, specs))
        return CACHE[name]
    if name in panel.columns and name not in CATEGORICAL and name not in ("date", "entity", "quality_flag"):
        return panel[name].astype(float)
    raise KeyError(f"unknown input {name!r}: use a panel field {_value_fields()}, a conditioner "
                   f"{[c for c in CONDITIONERS if c not in CATEGORICAL]}, or a feature you built {sorted(specs)}")


def _is_raw(spec: FeatureSpec) -> bool:
    """A raw value field proposed as a candidate under its own column name."""
    return spec.op == "field" and spec.inputs == [spec.name]


def _by_date(panel: pd.DataFrame, x: pd.Series):
    return x.groupby(panel["date"].values)


def _by_entity(panel: pd.DataFrame, x: pd.Series):
    return x.groupby(panel["entity"].values, sort=False)


def _cond_values(panel: pd.DataFrame, by: str) -> pd.Series:
    """A conditioner for cross-sectional comparison: log of size/liquidity (positive values), sector labels as is."""
    if by in CATEGORICAL:
        return panel[by].where(panel[by] != "Unknown")
    c = panel[by].astype(float)
    return np.log(c.where(c > 0))


def compute(spec: FeatureSpec, panel: pd.DataFrame, get) -> pd.Series:
    """Deterministic value of a FeatureSpec on the panel (panel sorted by date, then entity)."""
    op, w = spec.op, spec.window
    x = get(spec.inputs[0]).astype(float)
    if op == "field":
        out = x
    elif op == "ratio":
        den = get(spec.inputs[1]).astype(float)
        out = x / den.where(den != 0)
    elif op == "diff":
        out = x - get(spec.inputs[1]).astype(float)
    elif op == "log1p":
        out = np.log1p(x.clip(lower=0))
    elif op == "change":
        out = x - _by_entity(panel, x).shift(w)
    elif op == "growth":
        prev = _by_entity(panel, x).shift(w)
        out = (x - prev) / prev.abs().where(prev != 0)
    elif op == "surprise":
        prev = _by_entity(panel, x).shift(1)
        g = _by_entity(panel, prev)
        mu = g.transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).mean())
        sd = g.transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).std())
        out = (x - mu) / sd.where(sd > 0)
    elif op in ("rolling_mean", "rolling_sum", "rolling_std"):
        how = op.split("_")[1]
        out = _by_entity(panel, x).transform(lambda s: getattr(s.rolling(w, min_periods=max(2, w // 2)), how)())
    elif op == "peer_relative":
        keys = [panel["date"].values] + ([_cond_values(panel, spec.group).values] if spec.group else [])
        out = x - x.groupby(keys).transform("median")
    elif op == "rank":
        keys = [panel["date"].values] + ([_cond_values(panel, spec.group).values] if spec.group else [])
        out = x.groupby(keys).rank(pct=True)
    elif op == "residualize":
        c = _cond_values(panel, spec.group)
        if spec.group in CATEGORICAL:
            out = x - x.groupby([panel["date"].values, c.values]).transform("mean")
        else:
            ok = x.notna() & c.notna()
            xs, cs = x.where(ok), c.where(ok)
            g = lambda s: s.groupby(panel["date"].values)
            mx, mc = g(xs).transform("mean"), g(cs).transform("mean")
            beta = g((xs - mx) * (cs - mc)).transform("mean") / g((cs - mc) ** 2).transform("mean")
            out = (xs - mx - beta * (cs - mc))
    else:  # pragma: no cover  (the schema restricts op)
        raise ValueError(op)
    return pd.Series(np.asarray(out, dtype=float), index=panel.index).replace([np.inf, -np.inf], np.nan)


def formula(spec: FeatureSpec) -> str:
    a = spec.inputs
    return {"field": f"{a[0]}", "ratio": f"{a[0]} / {a[-1]}", "diff": f"{a[0]} - {a[-1]}", "log1p": f"log(1 + {a[0]})",
            "change": f"{a[0]} - {a[0]}[t-{spec.window}]", "growth": f"({a[0]} - {a[0]}[t-{spec.window}]) / |{a[0]}[t-{spec.window}]|",
            "surprise": f"({a[0]} - mean of previous {spec.window}) / std of previous {spec.window}",
            "rolling_mean": f"mean of last {spec.window} {a[0]}", "rolling_sum": f"sum of last {spec.window} {a[0]}",
            "rolling_std": f"std of last {spec.window} {a[0]}",
            "peer_relative": f"{a[0]} - same-date median {'within ' + spec.group if spec.group else 'of all entities'}",
            "residualize": f"{a[0]} net of same-date {spec.group} effect",
            "rank": f"same-date percentile rank of {a[0]}" + (f" within {spec.group}" if spec.group else "")}[spec.op]


def validate_spec(spec: FeatureSpec, specs: dict, panel: pd.DataFrame) -> list[str]:
    errs = []
    if not NAME.match(spec.name):
        errs.append(f"name {spec.name!r}: use lower case letters, digits and _ (2-41 chars)")
    if spec.name in panel.columns:
        errs.append(f"name {spec.name!r} is a panel column (use the column directly, no need to build it), or "
                    "choose a new name")
    if spec.name in specs and specs[spec.name] != spec:
        errs.append(f"a different feature is already named {spec.name!r}; choose a new name")
    need = 2 if spec.op in BINARY else 1
    if len(spec.inputs) != need:
        errs.append(f"{spec.op} takes {need} input(s), got {len(spec.inputs)}")
    for i in spec.inputs:
        ok = i in specs or (i in panel.columns and i not in CATEGORICAL and i not in ("date", "entity", "quality_flag"))
        if not ok and "(" in i:
            errs.append(f"input {i!r} is an expression; inputs are names. Add the inner step as its own definition "
                        "earlier in the same idea (e.g. {'name': 'x_rel', 'op': 'peer_relative', 'inputs': [...], "
                        "'group': 'sector'}) and use its name as the input")
        elif not ok:
            errs.append(f"unknown input {i!r}: use a panel field {_value_fields()}, a size/liquidity conditioner, or a "
                        f"feature you built {sorted(specs)}")
    SCALE_FREE = {"ratio", "surprise", "growth", "rank"}
    if spec.op == "ratio" and len(spec.inputs) == 2 and spec.inputs[1] in ("adv_proxy", "size_float", "size_assets") \
            and spec.inputs[0] in specs and specs[spec.inputs[0]].op in SCALE_FREE:
        errs.append(f"{spec.inputs[0]} is already scale-free ({specs[spec.inputs[0]].op}); dividing it by "
                    f"{spec.inputs[1]} would make it depend inversely on that scale. Divide raw levels by a scale "
                    "field instead")
    if spec.op in WINDOWED and not spec.window:
        errs.append(f"{spec.op} needs a window (records)")
    if spec.op == "residualize" and spec.group not in CONDITIONERS:
        errs.append(f"residualize needs group in {CONDITIONERS}")
    if spec.op in ("peer_relative", "rank") and spec.group not in (None, "", "sector"):
        errs.append(f"{spec.op}: group must be 'sector' or empty")
    return errs


# ---------------------------------------------------------------- characterization statistics
def _pct_rank(panel: pd.DataFrame, x: pd.Series) -> pd.Series:
    return _by_date(panel, x).rank(pct=True)


def exposure_stats(panel: pd.DataFrame, x: pd.Series, by: str) -> dict:
    """How a feature depends on a conditioner across entities on the same date, by rank."""
    c = _cond_values(panel, by)
    ok = x.notna() & c.notna()
    d = panel.loc[ok, ["date"]].assign(x=x[ok].values, c=c[ok].values)
    out = {"by": by, "rows_compared": int(ok.sum()), "share_of_feature_rows": _r(ok.sum() / max(1, x.notna().sum()))}
    if len(d) < 50:
        return {**out, "strength": "not measurable", "reading": "too few rows with both values"}
    d["rx"] = d.groupby("date").x.rank(pct=True)
    if by in CATEGORICAL:
        g = d.groupby(["date", "c"]).rx
        between = ((g.transform("mean") - d.groupby("date").rx.transform("mean")) ** 2).groupby(d.date).mean()
        total = d.groupby("date").rx.var(ddof=0)
        raw = between / total.where(total > 0)
        # bias-adjusted (epsilon squared): with n entities in k groups, pure noise gives eta² about (k-1)/(n-1)
        n, k = d.groupby("date").size(), d.groupby("date").c.nunique()
        eta = (1 - (1 - raw) * (n - 1) / (n - k).where(n > k)).dropna()
        prof = d.groupby("c").rx.mean().sort_values()
        size = d.groupby("c").size()
        e = float(eta.mean())
        strength = "strong" if e >= STRONG_ETA else "moderate" if e >= MODERATE_ETA else "negligible"
        out.update({"eta_squared_mean": _r(e), "eta_squared_by_year": {int(k): _r(v) for k, v in
                                                                         eta.groupby(eta.index.year).mean().items()},
                    "mean_rank_by_group": {k: _r(v, 3) for k, v in prof.items()},
                    "rows_by_group": {k: int(size[k]) for k in prof.index}, "strength": strength,
                    "reading": f"{by} explains {100 * e:.1f}% of the same-date rank variance (bias-adjusted; "
                               f"{strength}); highest "
                               f"{prof.index[-1]} (mean rank {prof.iloc[-1]:.2f}), lowest {prof.index[0]} "
                               f"({prof.iloc[0]:.2f}); 0.50 = no difference"})
        out["_profile"] = prof
        return out
    d["rc"] = d.groupby("date").c.rank(pct=True)
    cx = d.rx - d.groupby("date").rx.transform("mean")
    cc = d.rc - d.groupby("date").rc.transform("mean")
    num = (cx * cc).groupby(d.date).sum()
    den = np.sqrt((cx ** 2).groupby(d.date).sum() * (cc ** 2).groupby(d.date).sum())
    rho = (num / den.where(den > 0)).dropna()
    m, s = float(rho.mean()), float(rho.std())
    d["q"] = np.minimum((d.rc * 5).apply(np.ceil).astype(int), 5).clip(lower=1)
    prof = d.groupby("q").rx.mean()
    disp = d.groupby("q").x.quantile(.9) - d.groupby("q").x.quantile(.1)
    strength = "strong" if abs(m) >= STRONG_RHO else "moderate" if abs(m) >= MODERATE_RHO else "negligible"
    out.update({"rank_corr_mean": _r(m, 3), "rank_corr_t": _r(m / s * np.sqrt(len(rho)), 1) if s > 0 else None,
                "share_dates_positive": _r((rho > 0).mean(), 3), "n_dates": int(len(rho)),
                "rank_corr_by_year": {int(k): _r(v, 3) for k, v in rho.groupby(rho.index.year).mean().items()},
                "mean_feature_rank_by_conditioner_quintile": {f"Q{k}": _r(v, 3) for k, v in prof.items()},
                "p10_p90_spread_by_quintile": {f"Q{k}": _r(v, 6) for k, v in disp.items()},
                "strength": strength,
                "reading": f"same-date rank correlation with {by} {m:+.2f} on average (positive on "
                           f"{100 * (rho > 0).mean():.0f}% of dates): {strength}; from the lowest to the highest {by} "
                           f"quintile the feature's mean rank goes {prof.iloc[0]:.2f} -> {prof.iloc[-1]:.2f} "
                           "(0.50 = no dependence)"})
    out["_rho"], out["_profile"] = rho, prof
    return out


def stability_stats(panel: pd.DataFrame, x: pd.Series, lags=(1, 5, 20)) -> dict:
    """Cross-sectional level and coverage over time, candidate breaks, and how persistent the ranks are."""
    g = _by_date(panel, x)
    tab = pd.DataFrame({"n": g.count(), "median": g.median(), "p10": g.quantile(.1), "p90": g.quantile(.9)})
    tab.index = pd.DatetimeIndex(tab.index)
    w = max(5, min(60, len(tab) // 20))
    r = _pct_rank(panel, x)
    ac = {}
    for k in lags:
        prev = _by_entity(panel, r).shift(k)
        ok = r.notna() & prev.notna()
        ac[f"lag{k}"] = _r(np.corrcoef(r[ok], prev[ok])[0, 1], 3) if ok.sum() > 10 else None
    v = variance_split_stats(panel.assign(_x=x), "entity", "date", "_x")
    out = {"n_dates": int(len(tab)), "coverage_per_date": {"min": int(tab.n.min()), "median": int(tab.n.median()),
                                                           "max": int(tab.n.max())},
           "median_first_year": _r(tab["median"].iloc[:250].median(), 6),
           "median_last_year": _r(tab["median"].iloc[-250:].median(), 6),
           "breaks_in_median": level_shifts(tab["median"], w), "breaks_in_coverage": level_shifts(tab.n.astype(float), w),
           "rank_autocorrelation": ac,
           "variance_split": {k: v[k] for k in ("share_between_entities", "share_within_entities",
                                                "share_common_over_time")}}
    l5 = ac.get("lag5") or 0
    out["reading"] = (f"ranks {'persist' if l5 >= 0.7 else 'partly persist' if l5 >= 0.3 else 'reshuffle quickly'} "
                      f"(rank autocorrelation {l5:.2f} after 5 records: "
                      f"{'slow-moving, low turnover' if l5 >= 0.7 else 'high turnover' if l5 < 0.3 else 'moderate turnover'}); "
                      f"{v['reading']}")
    if out["breaks_in_median"]:
        out["reading"] += "; the cross-sectional median shifts around " + \
            ", ".join(b["date"] for b in out["breaks_in_median"])
    out["_tab"] = tab
    return out


# ---------------------------------------------------------------- tools
def _exposure_flag(name: str, st: dict) -> dict | None:
    by = st["by"]
    if st.get("strength") != "strong":
        return None
    fact = (f"{name} depends strongly on {by} (" + (f"rank correlation {st['rank_corr_mean']:+.2f}" if by not in
                                                     CATEGORICAL else f"eta² {st['eta_squared_mean']:.2f}") + ")")
    return {**flag(f"exposure_{name}_{by}", fact, [name.lower()], TERMS[by]), "feature": name, "by": by}


def _summary(x: pd.Series) -> dict:
    s = x.dropna()
    if not len(s):
        return {"n": 0}
    q = s.quantile([.01, .5, .99])
    return {"n": int(len(s)), "coverage_share": _r(len(s) / len(x)), "p1": _r(q[.01], 6), "median": _r(q[.5], 6),
            "p99": _r(q[.99], 6), "skew": _r(s.skew(), 2), "zero_share": _r((s == 0).mean()),
            "negative_share": _r((s < 0).mean())}


@tool("describe_panel", "Describe the research panel you work on: rows, dates, entities and how the universe was "
      "fixed, the value fields, the conditioning variables (liquidity, size, sector) with their coverage, and the "
      "caveats. Call it first.", {"type": "object", "properties": {}})
def describe_panel():
    from quantaccelerator.tools.panel import describe
    vals = _value_fields()
    return {**describe(get_panel(), STATE["meta"]),
            "checklist": [f"compare_representations(features={vals[:6]}, by='{b}')"
                          for b in ("adv_proxy", "size_float", "sector")] +
                         ["then: representations for the fields with a strong dependence, compared with their raw field",
                          "then: propose_candidate for each feature of the set, submit"]}


@tool("construct_feature", "Build a named feature from panel fields, conditioners or earlier features, in a closed "
      "language (see the op descriptions); no code. Returns its definition, coverage and distribution, and draws its "
      "histogram and its cross-sectional median over time. Build the raw field too (op 'field') when you want to "
      "characterize it under its own name.", inline_refs(FeatureSpec.model_json_schema()))
def construct_feature(**kw):
    spec = FeatureSpec.model_validate(kw)
    panel, specs = get_panel(), _features()
    errs = validate_spec(spec, specs, panel)
    if errs:
        raise ValueError("; ".join(errs))
    specs[spec.name] = spec
    CACHE.pop(spec.name, None)
    x = resolve(spec.name)
    out = {"feature": spec.name, "definition": formula(spec), **_summary(x)}
    if spec.op == "log1p" and (resolve(spec.inputs[0]) < 0).any():
        out["note"] = "negative inputs were clipped to 0 before log1p"
    if spec.op in NEUTRALIZING | {"ratio"}:
        out["next"] = "compare it with its input (compare_representations) to see what the transformation removed"
    if spec.op == "residualize" and spec.group not in CATEGORICAL:
        base = resolve(spec.inputs[0]).dropna()
        if len(base) and base.min() >= 0 and base.skew() > 2:
            out["note"] = (f"{spec.inputs[0]} is a heavily skewed non-negative level (skew {base.skew():.1f}); a linear "
                           f"residualization on log {spec.group} usually leaves a rank dependence (often reversed). A "
                           "log1p first, or a ratio to a scale field, is the usual way to remove scale")
    fig, ax = F.new(1, 2, width=9, height=2.6)
    s = x.dropna()
    s = s.sample(min(len(s), 300_000), random_state=0)
    logx = (s.min() >= 0) and (s.skew() > 2)
    ax[0].hist(np.log10(s[s > 0]) if logx else s.clip(s.quantile(.005), s.quantile(.995)), bins=60, color=F.BLUE)
    ax[0].set_title("Histogram" + (" (log10, >0)" if logx else " (0.5%-99.5%)"))
    g = _by_date(panel, x)
    med = g.median()
    med.index = pd.DatetimeIndex(med.index)
    ax[1].fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values, color=F.BLUE, alpha=.2,
                       label="p25-p75")
    ax[1].plot(med.index, med.values, color=F.BLUE, label="median")
    ax[1].set_title("Across entities, per date")
    ax[1].legend(loc="upper left")
    F.date_axis(ax[1])
    out["figure"] = F.save(fig, f"{spec.name} = {formula(spec)}")
    return out


@tool("compute_exposure", "Characterize a feature against one conditioning variable across entities on the same "
      "date, by rank: for liquidity (adv_proxy) or size (size_float, size_assets) the mean same-date rank "
      "correlation, its stability by year and the feature's mean rank in each conditioner quintile; for sector, the "
      "share of same-date rank variance sector explains and the mean rank per sector. A strong dependence is a "
      "question, not a verdict: measurement bias, economic scale and a genuine interaction call for different "
      "treatments. Draws the quintile (or sector) profile and the dependence over time.",
      {"type": "object", "properties": {
          "feature": {"type": "string", "description": "a feature you built or a panel field"},
          "by": {"type": "string", "enum": CONDITIONERS}}, "required": ["feature", "by"]})
def compute_exposure(feature: str, by: str):
    panel = get_panel()
    st = exposure_stats(panel, resolve(feature), by)
    prof, rho = st.pop("_profile", None), st.pop("_rho", None)
    st["feature"] = feature
    fl = _exposure_flag(feature, st)
    if fl:
        st["flags"] = [fl]
    if prof is not None:
        fig, ax = F.new(1, 2 if rho is not None else 1, width=9 if rho is not None else 6, height=2.6)
        ax[0].bar([str(k) if by in CATEGORICAL else f"Q{k}" for k in prof.index], prof.values, color=F.BLUE)
        ax[0].axhline(.5, color=F.MUTED, linewidth=1, linestyle=":")
        ax[0].set_ylim(0, 1)
        ax[0].set_title(f"Mean rank of {feature} by {by}" + ("" if by in CATEGORICAL else " quintile"))
        if by in CATEGORICAL:
            ax[0].tick_params(axis="x", rotation=45)
        if rho is not None:
            r = rho.copy()
            r.index = pd.DatetimeIndex(r.index)
            ax[1].plot(r.index, r.rolling(20, min_periods=5).mean(), color=F.BLUE)
            ax[1].axhline(0, color=F.MUTED, linewidth=1)
            ax[1].set_ylim(-1, 1)
            ax[1].set_title(f"Rank corr. with {by} (20-day mean)")
            F.date_axis(ax[1])
        st["figure"] = F.save(fig, f"{feature} vs {by}")
    return st


@tool("time_stability", "Distribution and variation of a feature over time: its cross-sectional median and spread per date, coverage "
      "per date, candidate breaks in either, how persistent each entity's rank is (rank autocorrelation after 1, 5, "
      "20 records: high = slow-moving, low turnover) and where its variance lies (across entities, within entities, "
      "common to a date). Draws the bands, the coverage and the rank persistence.",
      {"type": "object", "properties": {"feature": {"type": "string"}}, "required": ["feature"]})
def time_stability(feature: str):
    panel = get_panel()
    st = stability_stats(panel, resolve(feature))
    tab = st.pop("_tab")
    st["feature"] = feature
    flags = []
    if st["breaks_in_median"]:
        months = sorted({b["date"][:7] for b in st["breaks_in_median"]})
        from quantaccelerator.tools.explore import month_terms
        flags.append(flag(f"shift_{feature}_{months[0]}", f"the cross-sectional median of {feature} shifts around "
                          f"{', '.join(months)}", [feature.lower()], month_terms(months)))
    if flags:
        st["flags"] = flags
    fig, ax = F.new(3, 1, height=1.8, sharex=False)
    ax[0].fill_between(tab.index, tab.p10, tab.p90, color=F.BLUE, alpha=.2, label="p10-p90")
    ax[0].plot(tab.index, tab["median"], color=F.BLUE, label="median")
    for b in st["breaks_in_median"]:
        F.mark(ax[0], pd.Timestamp(b["date"]), b["date"])
    ax[0].set_title("Across entities, per date")
    ax[0].legend(loc="upper left")
    ax[1].plot(tab.index, tab.n, color=F.GREEN)
    ax[1].set_title("Entities with a value")
    for a in ax[:2]:
        F.date_axis(a)
    acs = st["rank_autocorrelation"]
    ax[2].bar(list(acs), [v or 0 for v in acs.values()], color=F.AMBER)
    ax[2].set_ylim(0, 1)
    ax[2].set_title("Rank autocorrelation within entity")
    st["figure"] = F.save(fig, f"Stability of {feature}")
    return st


@tool("compare_representations", "Compare several representations of the same idea side by side against one "
      "conditioning variable (e.g. raw volume vs volume / ADV vs its sector-relative version vs adv_proxy): each "
      "one's dependence on the conditioner, rank persistence and coverage. The construct -> plot -> characterize -> "
      "construct-again loop: use it to see what a transformation removed and what it kept. Draws the comparison.",
      {"type": "object", "properties": {
          "features": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
          "by": {"type": "string", "enum": CONDITIONERS}}, "required": ["features", "by"]})
def compare_representations(features: list[str], by: str):
    panel = get_panel()
    rows = {}
    for f in features:
        x = resolve(f)
        st = exposure_stats(panel, x, by)
        r = _pct_rank(panel, x)
        prev = _by_entity(panel, r).shift(5)
        ok = r.notna() & prev.notna()
        rows[f] = {"dependence": st.get("rank_corr_mean", st.get("eta_squared_mean")), "strength": st.get("strength"),
                   "rank_autocorr_lag5": _r(np.corrcoef(r[ok], prev[ok])[0, 1], 3) if ok.sum() > 10 else None,
                   "coverage_share": _r(x.notna().mean())}
    measure = "eta² (sector share of rank variance)" if by in CATEGORICAL else "mean same-date rank correlation"
    out = {"by": by, "dependence_measure": measure, "features": rows}
    best = min(rows, key=lambda k: abs(rows[k]["dependence"] or 0))
    out["reading"] = "; ".join(f"{k}: {v['dependence']} ({v['strength']})" for k, v in rows.items()) + \
        f". Least dependent on {by}: {best}."
    flags = [fl for k, v in rows.items() if (fl := _exposure_flag(k, {"by": by, "strength": v["strength"],
                                                                      "rank_corr_mean": v["dependence"] or 0,
                                                                      "eta_squared_mean": v["dependence"] or 0}))]
    if flags:
        out["flags"] = flags
    fig, ax = F.new(1, 2, width=9, height=max(1.8, 0.5 + 0.45 * len(rows)))
    names = list(rows)
    ax[0].barh(names, [rows[k]["dependence"] or 0 for k in names], color=F.BLUE)
    ax[0].axvline(0, color=F.MUTED, linewidth=1)
    ax[0].set_xlim(-1 if by not in CATEGORICAL else 0, 1)
    ax[0].set_title(f"Dependence on {by}")
    ax[0].invert_yaxis()
    ax[1].barh(names, [rows[k]["rank_autocorr_lag5"] or 0 for k in names], color=F.AMBER)
    ax[1].set_xlim(0, 1)
    ax[1].set_title("Rank persistence (lag 5)")
    ax[1].invert_yaxis()
    ax[1].set_yticklabels([])
    out["figure"] = F.save(fig, f"Representations vs {by}")
    return out


FEATURE_TOOLS = ["describe_panel", "construct_feature", "compute_exposure", "time_stability",
                 "compare_representations"]


# ---------------------------------------------------------------- the lab notebook (record tools)
def _features_in_args(args: dict) -> set:
    out = set()
    for k in ("feature", "name"):
        if isinstance(args.get(k), str):
            out.add(args[k])
    for k in ("features", "inputs"):
        if isinstance(args.get(k), list):
            out |= {x for x in args[k] if isinstance(x, str)}
    return out


def lineage(name: str, specs: dict | None = None) -> set:
    """The feature and everything it is built from."""
    specs = _features() if specs is None else specs
    out, todo = set(), [name]
    while todo:
        n = todo.pop()
        if n in out:
            continue
        out.add(n)
        if n in specs:
            todo += specs[n].inputs
    return out


def characterization_runs(name: str) -> list[str]:
    """run_ids of this session's characterization runs that include the feature."""
    import json

    from quantaccelerator.tools import registry
    runs = registry.session_runs(registry.CTX.session_id)
    sel = runs[runs.tool.isin(["compute_exposure", "compare_representations", "time_stability"]) & (runs.ok == 1)]
    return [r.run_id for _, r in sel.iterrows()
            if name == json.loads(r.args_json).get("feature") or name in (json.loads(r.args_json).get("features") or [])]


def characterized_by(name: str) -> set:
    """Conditioners a feature has been characterized against in this session (compute_exposure or
    compare_representations runs that included it)."""
    import json

    from quantaccelerator.tools import registry
    runs = registry.session_runs(registry.CTX.session_id)
    out = set()
    for _, r in runs[runs.tool.isin(["compute_exposure", "compare_representations"]) & (runs.ok == 1)].iterrows():
        a = json.loads(r.args_json)
        if name == a.get("feature") or name in (a.get("features") or []):
            out.add(a["by"])
    return out


def _cited_runs(refs: list[str]) -> list[dict]:
    from quantaccelerator.agents.checks import _runs
    return _runs(refs)


NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d+")


def _numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(x) for x in NUM.findall(obj)]
    return out


def measured_strength(runs: list[dict], feature: str, by: str) -> str | None:
    """What the cited runs measured for this feature against this conditioner."""
    for r in runs:
        a, res = r.get("args") or {}, r.get("result") or {}
        if r["tool"] == "compute_exposure" and a.get("feature") == feature and a.get("by") == by:
            return res.get("strength")
        if r["tool"] == "compare_representations" and a.get("by") == by and feature in (res.get("features") or {}):
            return res["features"][feature].get("strength")
    return None


def check_claim_numbers(o: FeatureObservation, runs: list[dict]) -> list[str]:
    """Every decimal in the claim comes from the cited results (to rounding), and a stated strength of dependence is
    the one the cited run measured."""
    known = [x for r in runs for x in _numbers(r.get("result"))]
    known += [100 * k for k in known]                                     # shares quoted as percentages
    for x in [float(v) for v in NUM.findall(o.claim)]:
        if not any(abs(x - k) <= 0.006 + 0.005 * abs(k) for k in known):
            return [f"the claim says {x:g}, but no cited result has that number; quote numbers exactly from the tool "
                    "results you cite"]
    text = o.claim.lower()
    for by in CHECKED:
        for word in ("strong", "moderate", "negligible"):
            if re.search(rf"{word}\w*\s+(?:\w+\s+){{0,3}}(?:on|with)\s+{by}", text):
                got = measured_strength(runs, o.feature, by)
                if got and got != word:
                    return [f"the claim calls the dependence of {o.feature} on {by} {word}, but the cited run measured "
                            f"it as {got}"]
    return []


def check_feature_observation(o: FeatureObservation) -> list[str]:
    from quantaccelerator.agents.checks import DATE_TOLERANCE_DAYS, _dates
    import json
    specs, panel = _features(), get_panel()
    if o.feature not in specs and o.feature not in panel.columns:
        return [f"unknown feature {o.feature!r}; built so far: {sorted(specs)}"]
    runs = [r for r in _cited_runs(o.evidence + ([o.figure_run_id] if o.figure_run_id else []))
            if r["tool"] in FEATURE_TOOLS and r["tool"] != "describe_panel"]
    if not runs:
        return [f"cite the run_id of the tool that measured this ({', '.join(FEATURE_TOOLS[1:])})"]
    about = lineage(o.feature)
    if not any(_features_in_args(r.get("args") or {}) & about for r in runs):
        if o.conditioner in CHECKED and o.conditioner not in characterized_by(o.feature):
            return [f"you have not measured {o.feature} against {o.conditioner} yet: call compute_exposure(feature="
                    f"'{o.feature}', by='{o.conditioner}') first, then record what it shows, citing that run"]
        if o.conditioner == "time" and not stability_checked(o.feature):
            return [f"you have not examined {o.feature} over time yet: call time_stability(feature='{o.feature}') "
                    "first, then record what it shows, citing that run"]
        runs_of = characterization_runs(o.feature)
        return [f"none of the cited runs is about {o.feature} (or what it is built from); cite the run that measured "
                f"it" + (f", e.g. one of {runs_of[-3:]}" if runs_of else "")]
    known = [d for r in runs for d in _dates(json.dumps(r.get("result"), default=str))]
    for d in _dates(o.claim):
        if not known or min(abs((d - k).days) for k in known) > DATE_TOLERANCE_DAYS:
            return [f"the claim mentions {d.date()} but no cited result has a date near it; take dates from tool numbers"]
    if o.decision in ("normalize", "residualize", "bucket") and len(o.possible_explanations) < 2:
        return ["before normalizing, residualizing or bucketing, list the explanations you cannot yet tell apart "
                "(possible_explanations, at least two: e.g. measurement bias, economic scale, genuine interaction) and "
                "the representations worth comparing"]
    if (e := check_claim_numbers(o, runs)):
        return e
    if o.decision in ("normalize", "residualize") and not o.candidate_representations:
        return ["name the candidate representations you would compare (candidate_representations)"]
    if o.decision == "drop":
        return check_drop(o.feature)
    return []


ADDRESSING = {"ratio", "residualize", "peer_relative", "rank"}
CHECKED = ("adv_proxy", "size_float", "sector")


def check_drop(feature: str) -> list[str]:
    """Setting a feature (and so possibly an idea) aside is a finding: it needs the characterization behind it, and a
    dependence is a reason to try a better representation, not to drop the idea."""
    if not (characterized_by(feature) or stability_checked(feature)):
        return [f"characterize {feature} before setting it aside (compute_exposure / compare_representations / "
                "time_stability), and cite that run"]
    specs, panel = _features(), get_panel()
    if feature not in specs and feature not in panel.columns:
        return []
    x = resolve(feature)
    strong = [b for b in CHECKED if exposure_stats(panel, x, b).get("strength") == "strong"]
    fields = {f for f in lineage(feature, specs) if f not in specs} & set(_value_fields())
    untried = [b for b in strong if not any(n != feature and specs[n].op in ADDRESSING and b in characterized_by(n)
                                            and fields & lineage(n, specs) for n in specs)]
    if untried:
        return [f"{feature} depends strongly on {', '.join(untried)}: before setting it aside, build a version that "
                "removes it (a ratio to a scale field, peer_relative or residualize by the conditioner) and "
                "characterize it against the same conditioner"]
    return []

def _idea_fields(idea) -> set:
    """Raw panel fields an idea's definitions are built from."""
    specs = _features()
    return {f for sp in idea.specs for f in lineage(sp.name, specs) if f not in specs} & set(_value_fields())


def check_candidate(c: CandidateFeature) -> list[str]:
    specs = _features()
    ideas = CTX.findings.get("ideas") or {}
    if ideas:
        if c.idea not in ideas:
            return [f"say which approved idea {c.name} expresses (idea = one of {sorted(ideas)})"]
        idea = ideas[c.idea]
        # a later step that defines the idea (surprise, change, ratio...) makes this a building block; a later
        # neutralization or rank is only a representation choice on top of it, which may rightly be dropped
        later = [sp.name for sp in idea.specs if c.name in sp.inputs and sp.op not in NEUTRALIZING | {"rank"}]
        if later:
            return [f"{c.name} is a building block of idea {c.idea!r} (its definition {later[-1]} uses it): propose the "
                    f"idea's representation ({idea.specs[-1].name}) or a version built from it"]
        if idea.source == "researcher" and c.name in specs and \
                not {sp.name for sp in idea.specs} & lineage(c.name, specs):
            return [f"{c.name} is your own construction, not the researcher's idea {c.idea!r}: propose the idea's "
                    f"definition ({idea.specs[-1].name}) or a version built from it; explore alternatives as an idea "
                    "of your own"]
        own = ({f for f in lineage(c.name, specs) if f not in specs} if c.name in specs else {c.name}) & set(_value_fields())
        if not own & _idea_fields(ideas[c.idea]):
            return [f"{c.name} is not built from the data idea {c.idea!r} uses ({sorted(_idea_fields(ideas[c.idea]))}); "
                    "build versions from the idea's definitions"]
    if c.name not in specs and c.name in _value_fields():  # a raw field kept as is
        specs[c.name] = FeatureSpec(name=c.name, op="field", inputs=[c.name], note="raw value field, kept as is")
    if c.name not in specs:
        return [f"{c.name!r} is not a feature you built (construct_feature first); built: {sorted(specs)}"]
    done = characterized_by(c.name)
    if not done:
        return [f"characterize {c.name} before proposing it: compute_exposure or compare_representations including it"]
    runs = [r for r in _cited_runs(c.evidence) if c.name in _features_in_args(r.get("args") or {})]
    if not runs:   # the runs exist in the log (checked above): attach them rather than refuse
        c.evidence[:] = list(dict.fromkeys(c.evidence + characterization_runs(c.name)))
    s = specs[c.name]
    panel, x = get_panel(), resolve(c.name)
    said = " ".join(c.known_exposures).lower()
    unstated = []
    for b in sorted(characterized_by(c.name)):
        st = exposure_stats(panel, x, b)
        if st.get("strength") == "strong" and not any(t in said for t in [b.lower()] + TERMS[b]):
            unstated.append(f"{b} ({st.get('rank_corr_mean', st.get('eta_squared_mean')):+.2f})")
    if unstated:
        return [f"{c.name} still depends strongly on {', '.join(unstated)}: state it in known_exposures, or propose a "
                "representation without it (keeping a dependence is a choice: record why with "
                "record_feature_observation)"]
    # keeping a strong dependence is a choice only once a version without it has been built and compared
    kept = [b for b in sorted(characterized_by(c.name)) if exposure_stats(panel, x, b).get("strength") == "strong"]
    fields = {f for f in lineage(c.name) if f in _value_fields()}
    uncompared = [b for b in kept if not any(fields & lineage(n) and specs[n].op in ADDRESSING and n != c.name and
                                             b in characterized_by(n) for n in specs)]
    if uncompared:
        return [f"{c.name} keeps a strong dependence on {', '.join(uncompared)}. Before keeping it, build a version "
                "without it (a ratio to a scale field, peer_relative or residualize) and compare the two against "
                f"{', '.join(uncompared)}; then propose the one you can justify (or both)"]
    # neutralizing what is not there only adds noise and a hidden choice (doc: do not automatically neutralize)
    for n in sorted(lineage(c.name)):
        sn = specs.get(n)
        if sn is None or sn.op not in NEUTRALIZING or not sn.group:
            continue
        base = exposure_stats(panel, resolve(sn.inputs[0]), sn.group)
        if base.get("strength") == "negligible":
            v = base.get("rank_corr_mean", base.get("eta_squared_mean"))
            return [f"{n} neutralizes {sn.inputs[0]} by {sn.group}, but {sn.inputs[0]} has a negligible dependence on "
                    f"{sn.group} ({v:+.3f}): neutralizing it only adds noise and a hidden choice. Propose "
                    f"{sn.inputs[0]} (or a version of it) without this step"]
    if s.op in NEUTRALIZING and s.group:
        st = exposure_stats(panel, x, s.group)
        if st.get("strength") == "strong":
            v = st.get("rank_corr_mean", st.get("eta_squared_mean"))
            return [f"{c.name} was built to remove the {s.group} effect but still depends strongly on {s.group} "
                    f"({v:+.2f}): the neutralization over- or under-corrects. Compare other representations "
                    "(e.g. log1p before residualizing, or a ratio to a scale field) and propose the one that works"]
    return []


MAX_OBSERVATIONS = 24


def _flagged(feature: str) -> bool:
    """A tool raised a flag about this feature: its finding is required, so the cap does not apply."""
    from quantaccelerator.agents.checks import session_flags
    f = feature.lower()
    return any(fl.get("feature") == feature or any(f in t for t in fl["terms"][0])
               for agent in ("research_eda_survey", CTX.agent or "research_eda") for _, fl in session_flags(agent))


@tool("record_feature_observation", "Record one characterization finding about a feature, as soon as a tool result "
      "establishes it: the claim, what it was characterized against, the run_ids, and for a dependence the "
      "explanations you cannot yet tell apart, the representations worth comparing, the checks that would decide, and "
      "what it implies (keep / normalize / residualize / bucket / drop / unresolved). Checked immediately.",
      inline_refs(FeatureObservation.model_json_schema()))
def record_feature_observation(**kw):
    from quantaccelerator.agents.checks import figure_of
    o = FeatureObservation.model_validate(kw)
    if len(CTX.findings.get("feature_observations", [])) >= MAX_OBSERVATIONS and not _flagged(o.feature):
        raise ValueError(f"{MAX_OBSERVATIONS} findings are recorded already: record only what bears on how an idea is "
                         "represented, and move on to proposing candidates")
    errs = check_feature_observation(o)
    if errs:
        raise ValueError(" ".join(errs))
    if not (o.figure_run_id and figure_of([o.figure_run_id])):
        o = o.model_copy(update={"figure_run_id": figure_of(o.evidence)})
    obs = CTX.findings.setdefault("feature_observations", [])
    obs.append(o)
    return {"recorded": f"observation {len(obs)}", "figure_run_id": o.figure_run_id}


@tool("propose_candidate", "Propose one feature for the candidate set that will be frozen (gate G3) before any "
      "predictive test: the idea it expresses, why this representation, the dependences it still has, and the "
      "characterization runs behind it. Proposing the same name again replaces it.",
      inline_refs(CandidateFeature.model_json_schema()))
def propose_candidate(**kw):
    c = CandidateFeature.model_validate(kw)
    errs = check_candidate(c)
    if errs:
        raise ValueError(" ".join(errs))
    cands = CTX.findings.setdefault("candidates", [])
    cands[:] = [x for x in cands if x.name != c.name] + [c]
    return {"recorded": c.name, **set_status()}


@tool("withdraw_candidate", "Remove a feature from the candidate set (e.g. redundant with another candidate, or "
      "still carrying a strong dependence), with the reason.",
      {"type": "object", "properties": {"name": {"type": "string"}, "reason": {"type": "string"}},
       "required": ["name", "reason"]})
def withdraw_candidate(name: str, reason: str):
    cands = CTX.findings.setdefault("candidates", [])
    if name not in [c.name for c in cands]:
        raise ValueError(f"{name!r} is not a candidate; candidates: {[c.name for c in cands]}")
    cands[:] = [c for c in cands if c.name != name]
    CTX.findings.setdefault("withdrawn", []).append({"name": name, "reason": reason})
    return {"withdrawn": name, **set_status()}


MAX_CANDIDATES = 6
RECORD_TOOLS = ["record_feature_observation", "propose_candidate", "withdraw_candidate"]


EDA_KEYS = ("features", "feature_observations", "candidates", "withdrawn", "ideas")


def reset_findings() -> None:
    """Start a fresh EDA notebook (the Data Research findings, if any, stay)."""
    for k in EDA_KEYS:
        CTX.findings.pop(k, None)
    CACHE.clear()


def field_characterized(field: str) -> set:
    """Conditioners a raw value field was characterized against (directly, or as an op 'field' feature)."""
    names = {field} | {n for n, s in _features().items() if s.op == "field" and s.inputs == [field]}
    return set().union(*(characterized_by(n) for n in names))


def missing_field_characterization() -> list[str]:
    out = []
    for f in _value_fields():
        done = field_characterized(f)
        miss = [b for b in ("adv_proxy", "sector") if b not in done]
        if not done & {"size_float", "size_assets"}:
            miss.append("size")
        if miss:
            out.append(f"{f} ({', '.join(miss)})")
    return out



def stability_checked(name: str) -> bool:
    """Whether the feature's distribution and variation over time were examined (time_stability)."""
    import json

    from quantaccelerator.tools import registry
    runs = registry.session_runs(registry.CTX.session_id)
    return any(json.loads(a).get("feature") == name for a in runs[(runs.tool == "time_stability") & (runs.ok == 1)].args_json)


def idea_todo(name: str) -> str:
    """The next concrete step for an open idea."""
    idea = (CTX.findings.get("ideas") or {}).get(name)
    if idea is None:
        return name
    feat = idea.specs[-1].name
    miss = missing_characterization(feat)
    if miss:
        return f"{name}: characterize {feat} ({', '.join(miss)}), then propose it"
    return f"{name}: {feat} is characterized, so call propose_candidate(name='{feat}', idea='{name}', ...) or set it aside"


def unaccounted_ideas() -> list[str]:
    """Approved ideas with neither a candidate nor a recorded decision to set them aside."""
    ideas = CTX.findings.get("ideas") or {}
    with_cand = {c.idea for c in CTX.findings.get("candidates", [])}
    dropped = {o.idea for o in CTX.findings.get("feature_observations", []) if o.decision == "drop" and o.idea}
    return [n for n in ideas if n not in with_cand and n not in dropped]


def set_status() -> dict:
    """What the candidate set still needs before the wrap-up can be accepted (shown after every change to the set)."""
    cands = [c.name for c in CTX.findings.get("candidates", [])]
    todo = {c: m for c in cands if (m := missing_characterization(c))}
    loose = unaccounted_ideas()
    return {"candidates": cands, "n_candidates": f"{len(cands)} of at most {MAX_CANDIDATES}",
            "ideas_not_yet_accounted_for": [idea_todo(n) for n in loose], "candidates_to_characterize": todo,
            "ready_to_submit": bool(cands and len(cands) <= MAX_CANDIDATES and not todo and not loose)}

def missing_characterization(name: str) -> list[str]:
    done = characterized_by(name)
    miss = [b for b in ("adv_proxy", "sector") if b not in done]
    if not done & {"size_float", "size_assets"}:
        miss.append("size_float or size_assets")
    if not stability_checked(name):
        miss.append("time_stability (distribution and variation over time)")
    return miss

def _eda_text(wrap) -> str:
    obs = CTX.findings.get("feature_observations", [])
    cands = CTX.findings.get("candidates", [])
    parts = [f"{o.feature} {o.conditioner} {o.claim} {' '.join(o.possible_explanations)}" for o in obs]
    parts += [f"{c.name} {c.rationale} {' '.join(c.known_exposures)}" for c in cands]
    parts += wrap.warnings + wrap.open_questions + [f"{r.observation} {r.question}"
                                                    for r in wrap.data_investigation_requests]
    return " ".join(parts).lower()


def _observed(feature: str, by: str) -> bool:
    """An observation about exactly this feature (or the raw field it names) against this conditioner."""
    specs = _features()
    same = {feature} | {n for n, s in specs.items() if s.op == "field" and s.inputs == [feature]}
    if feature in specs and specs[feature].op == "field":
        same.add(specs[feature].inputs[0])
    return any(o.feature in same and o.conditioner == by for o in CTX.findings.get("feature_observations", []))


def _unaddressed(agents: tuple, text: str) -> list[str]:
    """Flagged facts not yet in the findings. A strong dependence needs an observation about that feature against that
    conditioner (a mention elsewhere is not a finding); other flags need their words in the findings' text."""
    from quantaccelerator.agents.checks import session_flags
    missing, seen = [], set()
    for agent in agents:
        for rid, f in session_flags(agent):
            if f["id"] in seen:
                continue
            seen.add(f["id"])
            done = _observed(f["feature"], f["by"]) if "feature" in f else \
                all(any(t in text for t in group) for group in f["terms"])
            if not done:
                missing.append(f"{f['fact']} ({rid})")
    return [f"your tools flagged facts your findings do not record: {'; '.join(missing[:6])}. Record each with "
            "record_feature_observation (that feature, that conditioner), or explain why it does not matter"] \
        if missing else []


def check_survey(wrap) -> list[str]:
    """Pass 1: every raw value field characterized against liquidity, size and sector, with a finding and a plan."""
    todo = missing_field_characterization()
    if todo:
        return ["Not yet: characterize every raw value field against liquidity, size and sector first (one "
                "compare_representations call per conditioner with the raw fields). Missing: " + "; ".join(todo)]
    obs = CTX.findings.get("feature_observations", [])
    silent = [f for f in _value_fields() if not any(f in lineage(o.feature) for o in obs)]
    errs = [f"record what the characterization showed for {', '.join(silent)} (record_feature_observation)"] if silent \
        else []
    planned = " ".join(wrap.plan)
    unplanned = [f for f in _value_fields() if f not in planned]
    if unplanned:
        errs.append(f"the plan needs one line for each value field; missing: {', '.join(unplanned)}")
    text = " ".join([f"{o.feature} {o.conditioner} {o.claim} {' '.join(o.possible_explanations)}" for o in obs]
                    + wrap.plan).lower()
    return errs + _unaddressed(("research_eda_survey",), text)


def check_eda_findings(wrap, agent: str = "research_eda") -> list[str]:
    """Before the wrap-up is accepted: every approved idea built, characterized and either proposed or set aside with
    its evidence, at most MAX_CANDIDATES candidates, and every flagged fact recorded."""
    obs = CTX.findings.get("feature_observations", [])
    cands = CTX.findings.get("candidates", [])
    errs = []
    if len(obs) < 3:
        errs.append("record your findings first (record_feature_observation), at least three")
    if not cands:
        errs.append("propose the candidate set first (propose_candidate)")
    if len(cands) > MAX_CANDIDATES:
        per = {}
        for c in cands:
            per.setdefault(c.idea, []).append(c.name)
        spare = [n for v in per.values() if len(v) > 1 for n in v]
        errs.append(f"the set has {len(cands)} candidates; keep at most {MAX_CANDIDATES}: each frozen feature is one "
                    f"more hypothesis tested later. Withdraw (withdraw_candidate) a second version of an idea, e.g. "
                    f"among {spare}")
    loose = unaccounted_ideas()
    if loose:
        errs.append("every approved idea needs a candidate, or a recorded decision to set it aside (decision 'drop', "
                    "idea set, citing its characterization). Still open: " + "; ".join(idea_todo(n) for n in loose))
    calls = characterization_calls([c.name for c in cands])
    if calls:
        errs.append("some candidates are not characterized yet; make these calls, then submit: " + "; ".join(calls))
    return errs + _unaddressed(("research_eda_survey", agent), _eda_text(wrap))


def characterization_calls(names: list[str]) -> list[str]:
    """The exact calls still missing, batched by conditioner (compare_representations takes up to six features)."""
    need = {"adv_proxy": [], "size_float": [], "sector": []}
    stab = []
    for n in names:
        done = characterized_by(n)
        for b in ("adv_proxy", "sector"):
            if b not in done:
                need[b].append(n)
        if not done & {"size_float", "size_assets"}:
            need["size_float"].append(n)
        if not stability_checked(n):
            stab.append(n)
    out = []
    for b, v in need.items():
        for i in range(0, len(v), 6):
            chunk = v[i:i + 6]
            out.append(f"compute_exposure(feature='{chunk[0]}', by='{b}')" if len(chunk) == 1
                       else f"compare_representations(features={chunk}, by='{b}')")
    return out + [f"time_stability(feature='{n}')" for n in stab]

def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def check_idea_plan(plan, researcher_ideas: list[str] | None = None) -> list[str]:
    """Ideas are well-formed, buildable on this panel (every definition valid, names unique), every researcher idea is
    translated, and the agent contributes ideas of its own."""
    panel, errs, specs, names = get_panel(), [], {}, set()
    for idea in plan.ideas:
        if not NAME.match(idea.name) or idea.name in names:
            errs.append(f"idea name {idea.name!r}: lower case, digits and _, unique")
        names.add(idea.name)
        if not idea.mechanism.strip() or not idea.assumptions:
            errs.append(f"idea {idea.name}: state the mechanism and at least one assumption")
        if not 1 <= len(idea.specs) <= 3:
            errs.append(f"idea {idea.name}: give 1-3 feature definitions")
        if len(idea.expected_direction.split()) < 6:
            errs.append(f"idea {idea.name}: state the expected direction as a full hypothesis (what moves which way, "
                        "against what; e.g. 'stocks with higher X should see lower subsequent returns'), to be tested "
                        "later with returns")
        if idea.source == "agent" and all(sp.op in {"residualize", "rank", "field"} for sp in idea.specs):
            errs.append(f"idea {idea.name}: residualizing or ranking a raw field is a representation choice, not an "
                        "idea; state an economic behavior the data could reveal and how to measure it (a position "
                        "relative to sector peers, peer_relative, counts as one)")
        for sp in idea.specs:
            e = validate_spec(sp, specs, panel)
            if e:
                errs.append(f"idea {idea.name}, definition {sp.name}: " + "; ".join(e))
            else:
                specs[sp.name] = sp
    shapes = {}
    for idea in plan.ideas:
        shape = tuple(sorted((sp.op, tuple(i for i in sp.inputs if i not in specs)) for sp in idea.specs))
        if shape in shapes:
            errs.append(f"ideas {shapes[shape]} and {idea.name} build the same thing; make each idea a distinct behavior")
        shapes.setdefault(shape, idea.name)
    given = [t for t in researcher_ideas or [] if t.strip()]
    translated = {_norm(i.researcher_text) for i in plan.ideas if i.source == "researcher"}
    missing = [t for t in given if _norm(t) not in translated]
    if missing:
        errs.append(f"translate every researcher idea into an idea with source 'researcher' and researcher_text copied "
                    f"verbatim; missing: {missing}")
    own = [i for i in plan.ideas if i.source == "agent"]
    if len(own) < 2:
        errs.append("propose at least two ideas of your own (source 'agent'), each with its economic mechanism")
    if len(plan.ideas) > 6:
        errs.append("propose at most 6 ideas")
    return errs[:10]


def assemble_feature_set(wrap) -> "CandidateFeatureSet":
    """The recorded candidates, every feature spec built in the session (construction order) and the observations,
    as one unsigned CandidateFeatureSet."""
    from quantaccelerator.state.schemas import CandidateFeatureSet
    from quantaccelerator.tools.panel import describe
    specs = _features()
    cands = list(CTX.findings.get("candidates", []))
    meta = STATE["meta"] or {}
    return CandidateFeatureSet(panel_hash=meta.get("hash", ""), panel={**describe(get_panel(), meta),
                                                                       "rule": meta.get("rule")},
                               features=list(specs.values()), candidates=cands,
                               observations=list(CTX.findings.get("feature_observations", [])), wrapup=wrap)


def materialize(specs: list[FeatureSpec], names: list[str], panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """date, entity and the named features, recomputed from their specs (no session state needed)."""
    panel = get_panel() if panel is None else panel
    by = {s.name: s for s in specs}
    cache: dict[str, pd.Series] = {}

    def get(n):
        if n in by and not _is_raw(by[n]):
            if n not in cache:
                cache[n] = compute(by[n], panel, get)
            return cache[n]
        return panel[n].astype(float)

    return pd.DataFrame({"date": panel["date"].values, "entity": panel["entity"].values,
                         **{n: get(n).values for n in names}})
