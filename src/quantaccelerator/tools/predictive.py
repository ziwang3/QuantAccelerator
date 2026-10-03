"""Research EDA tools, phase B (predictive): does a frozen, pre-registered feature carry information about returns?

Every tool reads the returns vault (quantaccelerator.returns.vault), which opens only for a G3-frozen feature set and
a G4-signed plan, and gives the agent the discovery segment only. Tools take a frozen feature by name; no new
feature can be built here, so the representation cannot be tuned on returns. Each distinct test is counted in the
trial ledger, and every result states the trial count and the t a discovery must clear (returns.core.hurdle).
"""
import re

import numpy as np
import pandas as pd

from quantaccelerator.returns import core, vault
from quantaccelerator.state.schemas import ResearchObservation, ResearchObservationDraft
from quantaccelerator.tools.registry import CTX, inline_refs, tool
from quantaccelerator.viz import figures as F

H_MENU = [1, 5, 20, 60]
PRED_TOOLS = ["describe_prediction_setup", "ic_summary", "ic_decay", "quantile_returns", "conditional_ic"]
RECORD_TOOLS = ["record_research_observation"]
MAX_OBSERVATIONS = 40
VALIDATION_T = 2.0   # a single pre-specified re-test on fresh data: the usual 5% two-sided bar


def _feature(name: str) -> str:
    if name not in vault.STATE["frozen"]:
        raise ValueError(f"{name!r} is not a frozen feature; phase B tests only the frozen set: {vault.STATE['frozen']}"
                         " (no new feature can be built after the freeze)")
    return name


def _horizon(h) -> int:
    h = int(h)
    if h not in H_MENU:
        raise ValueError(f"horizon must be one of {H_MENU} sessions")
    return h


def _controls(ctrl) -> list[str]:
    ctrl = sorted(set(ctrl or []))
    bad = [c for c in ctrl if c not in core.CONTROLS]
    if bad:
        raise ValueError(f"unknown controls {bad}; available: {list(core.CONTROLS)}")
    return ctrl


def _signal(d: pd.DataFrame, feature: str, ctrl: list[str]) -> tuple[pd.DataFrame, str]:
    if not ctrl:
        return d, feature
    return d.assign(_resid=core.residualize(d, feature, ctrl)), "_resid"


def _ledger(tool_name: str, keys: list[str]) -> dict:
    n = vault.count_trial(tool_name, keys) if vault.STATE["segment"] == "discovery" else vault.n_trials()
    return {"segment": vault.STATE["segment"], "n_trials": n, "hurdle_t": round(core.hurdle(n), 3)}


def _significance(t: float, hurdle: float) -> str:
    if t is None or np.isnan(t):
        return "no estimate"
    return "clears the hurdle" if abs(t) >= hurdle else ("|t| >= 2 but below the hurdle" if abs(t) >= 2 else
                                                         "not distinguishable from zero")


def _bars(ax, labels, values, ylabel):
    ax.bar(range(len(values)), values, width=0.6, color=F.BLUE)
    ax.axhline(0, color=F.MUTED, linewidth=0.8)
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def _title(feature, ctrl, extra="") -> str:
    return f"{feature}{' net of ' + ', '.join(ctrl) if ctrl else ''}{extra} (discovery)" \
        if vault.STATE["segment"] == "discovery" else f"{feature}{extra} ({vault.STATE['segment']})"


def _key(kind, feature, h, ctrl=(), extra="") -> str:
    return f"{kind}|{feature}|{h}|{','.join(ctrl)}|{extra}"


# ---------------------------------------------------------------- tools
@tool("describe_prediction_setup", "The predictive setup: the frozen features and the signed plan, the segment you "
      "work on (discovery) and its coverage, horizons, controls and conditioners, how forward returns are measured, "
      "and the current trial count and hurdle.", {"type": "object", "properties": {}})
def describe_prediction_setup():
    from quantaccelerator.returns.vault import STATE
    sp = STATE["split"]
    n = vault.n_trials()
    return {"frozen_features": STATE["frozen"], "segment": "discovery", "discovery": list(sp.discovery),
            "later_segments": "validation and a locked test follow; you never see them: the orchestrator re-runs each "
                              "'informative' observation on validation, exactly as you ran it",
            "coverage": vault.coverage("discovery"), "horizons": H_MENU,
            "forward_return": "R_h(d) = open of decision date d -> close of session d+h-1 (entry at the open the PIT "
                              "contract allows); returns compared within a date, so the market drops out",
            "controls": core.CONTROL_MEANING, "conditioners": ["dollar_adv", "size", "beta", "sector", "year"],
            "n_trials": n, "hurdle_t": round(core.hurdle(n), 3),
            "hurdle_rule": "a discovery is 'informative' only if |t_nw| >= max(3, sqrt(2 ln N)), N = distinct tests "
                           "run on this feature set so far (every horizon, control set and bucket counts)",
            "checklist": ["for each planned feature: ic_decay (horizon profile) and quantile_returns at the primary "
                          "horizon (shape)",
                          "at every planned horizon: ic_summary with the planned controls, then "
                          "record_research_observation citing it",
                          "where a raw relation is significant, ic_summary with the controls it may be explained by",
                          "record_research_observation reports what the plan still owes; submit when nothing is"]}


@tool("ic_summary", "Information coefficient of a frozen feature with the h-session forward return on discovery: "
      "daily same-date rank correlation, its mean, volatility, Newey-West t, hit rate and mean by year. With controls, "
      "the feature is first residualized on them (same-date ranks, sector means).",
      {"type": "object", "properties": {
          "feature": {"type": "string"}, "horizon": {"type": "integer", "enum": H_MENU},
          "controls": {"type": "array", "items": {"type": "string", "enum": list(core.CONTROLS)}}},
       "required": ["feature", "horizon"]})
def ic_summary(feature: str, horizon: int, controls: list[str] | None = None):
    f, h, ctrl = _feature(feature), _horizon(horizon), _controls(controls)
    d, col = _signal(vault.frame(), f, ctrl)
    st = core.ic_stats(core.daily_ic(d, col, f"fwd_{h}"), h)
    led = _ledger("ic_summary", [_key("ic", f, h, ctrl)])
    return {"feature": f, "horizon": h, "controls": ctrl, **st, **led,
            "significance": _significance(st.get("t_nw"), led["hurdle_t"])}


@tool("ic_decay", "Mean IC and Newey-West t of a frozen feature at every horizon 1, 2, 5, 10, 20, 40 and 60 sessions "
      "(discovery): where the information is, and how fast it fades.",
      {"type": "object", "properties": {
          "feature": {"type": "string"},
          "controls": {"type": "array", "items": {"type": "string", "enum": list(core.CONTROLS)}}},
       "required": ["feature"]})
def ic_decay(feature: str, controls: list[str] | None = None):
    f, ctrl = _feature(feature), _controls(controls)
    d, col = _signal(vault.frame(), f, ctrl)
    out = {}
    for h in core.HORIZONS:
        st = core.ic_stats(core.daily_ic(d, col, f"fwd_{h}"), h)
        out[str(h)] = {"mean_ic": st.get("mean_ic"), "t_nw": st.get("t_nw")}
    peak = max(out, key=lambda k: abs(out[k]["mean_ic"] or 0))
    led = _ledger("ic_decay", [_key("ic", f, h, ctrl) for h in core.HORIZONS])
    fig, (ax,) = F.new(width=6.5, height=2.8)
    hs, ics = list(core.HORIZONS), [out[str(h)]["mean_ic"] or 0 for h in core.HORIZONS]
    ax.plot(range(len(hs)), ics, color=F.BLUE, marker="o", markersize=5)
    ax.axhline(0, color=F.MUTED, linewidth=0.8)
    k = hs.index(int(peak))
    right = k >= len(hs) / 2                       # keep the label inside the axes
    ax.annotate(f"peak h={peak}: IC {ics[k]:+.3f}, t {out[peak]['t_nw']:.1f}", (k, ics[k]),
                xytext=(-6 if right else 6, 8), ha="right" if right else "left", textcoords="offset points",
                fontsize=8, color=F.INK)
    ax.set_xticks(range(len(hs)), [str(h) for h in hs])
    ax.margins(y=0.25)
    ax.set_xlabel("horizon (sessions)")
    ax.set_ylabel("mean rank IC")
    fig_path = F.save(fig, "IC by horizon: " + _title(f, ctrl))
    return {"feature": f, "controls": ctrl, "by_horizon": out, "peak_horizon": int(peak), **led, "figure": fig_path}


@tool("quantile_returns", "Mean h-session forward return of each same-date quantile of a frozen feature, in excess "
      "of the date's average (discovery), the top-minus-bottom spread with its Newey-West t, and monotonicity (rank "
      "correlation of quantile and mean return).",
      {"type": "object", "properties": {
          "feature": {"type": "string"}, "horizon": {"type": "integer", "enum": H_MENU},
          "n_quantiles": {"type": "integer", "minimum": 3, "maximum": 10}}, "required": ["feature", "horizon"]})
def quantile_returns(feature: str, horizon: int, n_quantiles: int = 5):
    f, h = _feature(feature), _horizon(horizon)
    st = core.quantile_spread(vault.frame(), f, f"fwd_{h}", int(n_quantiles), h)
    led = _ledger("quantile_returns", [_key("q", f, h, extra=str(n_quantiles))])
    fig, (ax,) = F.new(width=7.5, height=2.8)
    q = st["quantile_mean_excess_return"]
    _bars(ax, list(q), [1e4 * v for v in q.values()], f"excess return, bp ({h} sessions)")
    fig_path = F.save(fig, f"By quantile (low to high): {_title(f, [], f', h={h}')}")
    return {"feature": f, "horizon": h, **st, **led, "figure": fig_path}


@tool("conditional_ic", "IC of a frozen feature at one horizon within buckets of a conditioner (discovery): terciles "
      "of dollar_adv, size or beta (low/mid/high), sectors, or calendar years. Shows whether the information is "
      "concentrated somewhere or changes sign.",
      {"type": "object", "properties": {
          "feature": {"type": "string"}, "horizon": {"type": "integer", "enum": H_MENU},
          "by": {"type": "string", "enum": ["dollar_adv", "size", "beta", "sector", "year"]}},
       "required": ["feature", "horizon", "by"]})
def conditional_ic(feature: str, horizon: int, by: str):
    f, h = _feature(feature), _horizon(horizon)
    d = vault.frame()
    if by == "year":
        b = d.date.dt.year.astype(str)
    elif by == "sector":
        b = d.sector.fillna("Unknown")
    elif by in ("dollar_adv", "size", "beta"):
        r = d.groupby("date")[by].rank(pct=True)
        b = pd.cut(r, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"]).astype(str).where(r.notna())
    else:
        raise ValueError("by must be dollar_adv, size, beta, sector or year")
    out = {}
    for k, g in d.assign(_b=b).dropna(subset=["_b"]).groupby("_b"):
        st = core.ic_stats(core.daily_ic(g, f, f"fwd_{h}"), h)
        if st.get("n_dates"):
            out[str(k)] = {"mean_ic": st["mean_ic"], "t_nw": st["t_nw"], "n_dates": st["n_dates"]}
    if set(out) <= {"low", "mid", "high"}:
        out = {k: out[k] for k in ("low", "mid", "high") if k in out}
    led = _ledger("conditional_ic", [_key("cond", f, h, extra=f"{by}={k}") for k in out])
    fig, (ax,) = F.new(width=6.5, height=2.8)
    _bars(ax, list(out), [v["mean_ic"] for v in out.values()], "mean rank IC")
    fig_path = F.save(fig, f"IC by {by}: {_title(f, [], f', h={h}')}")
    return {"feature": f, "horizon": h, "by": by, "buckets": out, **led, "figure": fig_path}


@tool("cost_capacity", "Cost and capacity of a finding (orchestrator): its long-short quantile portfolio, rebalanced "
      "every h sessions, gross and net of half-spreads and square-root impact, with break-even cost and capacity.",
      {"type": "object", "properties": {"feature": {"type": "string"}, "horizon": {"type": "integer"},
                                        "controls": {"type": "array", "items": {"type": "string"}},
                                        "sign": {"type": "integer"}}, "required": ["feature", "horizon", "sign"]})
def cost_capacity(feature: str, horizon: int, sign: int, controls: list[str] | None = None):
    from quantaccelerator.returns.costs import portfolio
    f, h, ctrl = _feature(feature), _horizon(horizon), _controls(controls)
    d, col = _signal(vault.frame(), f, ctrl)
    out = portfolio(d, col, int(np.sign(sign) or 1), h)
    led = _ledger("cost_capacity", [_key("cost", f, h, ctrl, str(sign))])
    return {"feature": f, "horizon": h, "controls": ctrl, "sign": sign, **out, **led}


# ---------------------------------------------------------------- recording and checks
STRONG_WORDS = re.compile(r"\b(significant\w*|strong\w*|robust\w*|reliabl\w*)", re.I)
NEGATION = re.compile(r"\b(not|no|non|without|lacks?|lacking|fails?|neither|nor|never|insufficient\w*|below|"
                      r"weak\w*)\b|n't\b", re.I)


def strength_asserted(text: str) -> str | None:
    """The first strength word the claim asserts (not negated within the few words before it), e.g. 'significant'
    in 'a significant relation' but not in 'not statistically significant' or 'no significant relation'."""
    for m in STRONG_WORDS.finditer(text):
        before = " ".join(text[:m.start()].split()[-4:])
        if not NEGATION.search(before):
            return text[max(0, m.start() - 30):m.end()].strip()
    return None


def _plan():
    from quantaccelerator.returns.vault import STATE
    return STATE.get("plan")


def _session_runs(refs: list[str]) -> list[dict]:
    from quantaccelerator.agents.checks import _runs
    return [r for r in _runs(refs) if r["tool"] in PRED_TOOLS[1:] and r.get("result", {}).get("segment") == "discovery"]


def _is_primary(r: dict, o) -> bool:
    a = r.get("args") or {}
    return r["tool"] == "ic_summary" and a.get("feature") == o.feature and int(a.get("horizon", -1)) == o.horizon \
        and sorted(a.get("controls") or []) == sorted(o.controls)


def _call(o, controls=None) -> str:
    c = sorted(o.controls if controls is None else controls)
    return f"ic_summary(feature='{o.feature}', horizon={o.horizon}, controls={c})"


def _latest_run(feature: str, horizon: int, controls: list[str]) -> str | None:
    """run_id of this session's latest discovery ic_summary by the agent with exactly these arguments."""
    import json
    from quantaccelerator.tools import registry
    if not registry.CTX.session_id:
        return None
    r = registry.session_runs(registry.CTX.session_id)
    r = r[(r.tool == "ic_summary") & (r.ok == 1) & (r.agent != "orchestrator")]
    for rid, a in zip(r.run_id[::-1], r.args_json[::-1]):
        a = json.loads(a)
        if a.get("feature") == feature and int(a.get("horizon", -1)) == horizon and \
                sorted(a.get("controls") or []) == sorted(controls):
            return rid
    return None


def _with_ledger_runs(o) -> list[str]:
    """The evidence plus the runs a verdict rests on that the agent ran but did not cite (the exact-argument
    ic_summary, and the raw one): what counts is that the test was run, which the ledger shows; citing it again is
    bookkeeping."""
    ev = list(o.evidence)
    for ctrl in ([o.controls, []] if o.controls else [o.controls]):
        rid = _latest_run(o.feature, o.horizon, ctrl)
        if rid and rid not in ev:
            ev.append(rid)
    return ev


def check_research_observation(o: ResearchObservationDraft) -> list[str]:
    from quantaccelerator.tools.features import check_claim_numbers
    if o.feature not in vault.STATE["frozen"]:
        return [f"{o.feature!r} is not a frozen feature: {vault.STATE['frozen']}"]
    runs = _session_runs(o.evidence)
    if not runs:
        return [f"cite the run_ids of the predictive tool calls behind the claim (e.g. {_call(o)})"]
    if not any((r.get("args") or {}).get("feature") == o.feature for r in runs):
        return [f"none of the cited runs is about {o.feature}; cite the run that measured it"]
    prim = [r for r in runs if _is_primary(r, o)]
    if not prim:
        return [f"the verdict rests on the ic_summary run with exactly this feature, horizon and controls: call "
                f"{_call(o)} and cite its run_id"]
    res = prim[-1]["result"]
    t, ic = res.get("t_nw"), res.get("mean_ic")
    hurdle = core.hurdle(vault.n_trials())
    if t is None or ic is None:
        return ["the primary run has no estimate (too few dates); the verdict is inconclusive"] \
            if o.verdict != "inconclusive" else []
    measured = int(np.sign(ic))
    if o.verdict == "informative":
        if abs(t) < hurdle:
            return [f"'informative' needs |t_nw| >= the hurdle {hurdle:.2f} (N = {vault.n_trials()} tests so far); the "
                    f"run measured t = {t:.2f}: the verdict is not_informative (or unstable / explained_by_control "
                    "with the runs that show it)"]
        if o.sign != measured:
            return [f"the cited run measured a mean IC of {ic:.4f}: sign {measured}, not {o.sign}"]
        plan = _plan()
        need = sorted({c for pt in (plan.tests if plan else []) if pt.feature == o.feature for c in pt.controls})
        if not set(need) <= set(o.controls):
            want = sorted(set(need) | set(o.controls))
            return [f"the signed plan requires {o.feature} to survive {need}: call {_call(o, want)} and rest the "
                    "verdict on it (if the relation is gone there, the verdict is explained_by_control)"]
    if o.verdict == "not_informative" and abs(t) >= hurdle:
        return [f"the run measured t = {t:.2f}, which clears the hurdle {hurdle:.2f}: the verdict is informative"]
    raw = [r for r in runs if r["tool"] == "ic_summary" and (r.get("args") or {}).get("feature") == o.feature
           and int((r.get("args") or {}).get("horizon", -1)) == o.horizon and not (r.get("args") or {}).get("controls")]
    if o.verdict == "not_informative" and o.controls:
        # a controlled null must not hide a raw relation: the researcher needs to know it exists and what explains it
        if not raw:
            return [f"a null with controls must also report the raw relation: call {_call(o, [])} and cite it"]
        rt = raw[-1]["result"].get("t_nw") or 0
        if abs(rt) >= hurdle:
            return [f"the raw relation clears the hurdle (t = {rt:.2f}) and is gone with {o.controls}: the verdict is "
                    f"explained_by_control. Record it as record_research_observation(feature='{o.feature}', horizon="
                    f"{o.horizon}, controls={o.controls}, verdict='explained_by_control', sign={int(np.sign(rt))}, "
                    "claim=<the raw and the controlled t, copied from the results>)"]
    if o.verdict == "explained_by_control":
        if not o.controls:
            return ["explained_by_control rests on a controlled run: set controls to the ones that explain it and cite "
                    f"{_call(o, ['dollar_adv'])}-style run (the controls you suspect)"]
        if not raw:
            return [f"also cite the raw relation: {_call(o, [])}"]
        if abs(raw[-1]["result"].get("t_nw") or 0) < hurdle:
            return [f"the raw relation (t = {raw[-1]['result'].get('t_nw'):.2f}) does not clear the hurdle {hurdle:.2f}: "
                    "there is nothing for a control to explain; the verdict is not_informative"]
        if abs(t) >= hurdle:
            return [f"with the controls the relation still clears the hurdle (t = {t:.2f}): it is not explained by them"]
    if o.verdict == "unstable":
        cond = [r for r in runs if r["tool"] == "conditional_ic" and (r.get("args") or {}).get("feature") == o.feature]
        flips = any(any((b.get("t_nw") or 0) >= 2 for b in r["result"].get("buckets", {}).values()) and
                    any((b.get("t_nw") or 0) <= -2 for b in r["result"].get("buckets", {}).values()) for r in cond)
        if not flips:
            return [f"'unstable' needs a cited conditional_ic run for {o.feature} with buckets of opposite sign, each "
                    "|t| >= 2 (e.g. by='year')"]
    # explained_by_control and unstable describe a relation that IS significant somewhere (raw, or in some buckets)
    if o.verdict in ("not_informative", "inconclusive") and (phrase := strength_asserted(o.claim)):
        return [f"the claim asserts strength ('...{phrase}'), but the verdict is {o.verdict}: describe it as e.g. 'not "
                "distinguishable from zero' or 'below the hurdle', with the numbers"]
    errs = check_claim_numbers(o, runs)
    if errs:
        return errs
    plan = _plan()
    if plan is not None and not o.deviation_from_plan and not any(
            pt.feature == o.feature and o.horizon in pt.horizons for pt in plan.tests):
        planned = [pt.horizons for pt in plan.tests if pt.feature == o.feature]
        return [f"horizon {o.horizon} for {o.feature} was not pre-registered (planned: {planned}); say why you ran it in "
                "deviation_from_plan"]
    return []


@tool("record_research_observation", "Record one predictive finding about a frozen feature at one horizon, as soon as "
      "the runs establish it: verdict (informative / not_informative / explained_by_control / unstable / inconclusive), "
      "sign, controls of the ic_summary run it rests on, a one-sentence claim and the run_ids. Checked immediately; "
      "recording the same feature, horizon and controls again replaces it.",
      inline_refs(ResearchObservationDraft.model_json_schema()))
def record_research_observation(**kw):
    o = ResearchObservationDraft.model_validate(kw)
    o = o.model_copy(update={"controls": sorted(o.controls)})
    o = o.model_copy(update={"evidence": _with_ledger_runs(o)})
    obs = CTX.findings.setdefault("research_observations", [])
    same = lambda x: (x.feature, x.horizon, x.controls) == (o.feature, o.horizon, o.controls)
    if len(obs) >= MAX_OBSERVATIONS and not any(same(x) for x in obs):
        raise ValueError(f"{MAX_OBSERVATIONS} observations are recorded already: submit the wrap-up")
    errs = check_research_observation(o)
    if errs:
        raise ValueError(" ".join(errs))
    prim = [r for r in _session_runs(o.evidence) if _is_primary(r, o)][-1]
    plan = _plan()
    idea = next((pt.idea for pt in (plan.tests if plan else []) if pt.feature == o.feature), None)
    n = vault.n_trials()
    full = ResearchObservation(**o.model_dump(), idea=idea, primary_run=prim["run_id"], n_trials_at_claim=n,
                               hurdle_t=round(core.hurdle(n), 3),
                               stats={k: prim["result"].get(k) for k in ("mean_ic", "ic_vol", "t_nw", "hit_rate",
                                                                         "n_dates", "mean_ic_by_year")})
    obs[:] = [x for x in obs if not same(x)] + [full]
    return {"recorded": f"observation {len(obs)}", "status": status()}


def _agent_runs(tool_name: str) -> list[dict]:
    """This session's successful discovery runs of a predictive tool by the agent (args), from the ledger."""
    import json
    from quantaccelerator.tools import registry
    if not registry.CTX.session_id:
        return []
    r = registry.session_runs(registry.CTX.session_id)
    r = r[(r.tool == tool_name) & (r.ok == 1) & (r.agent != "orchestrator")]
    return [json.loads(x) for x in r.args_json]


def status() -> dict:
    """What the pre-registration still owes. A planned test is a feature at each planned horizon with its planned
    controls: each needs a recorded observation resting on a run with (at least) those controls; each feature also
    needs its horizon profile (ic_decay) and its shape at the primary horizon (quantile_returns). Computed from the
    ledger, so a test cannot be counted as done without the run behind it."""
    plan, obs = _plan(), CTX.findings.get("research_observations", [])
    focus = CTX.findings.get("predict_focus")          # one feature per pass (orchestrator.predict)
    decay = {a.get("feature") for a in _agent_runs("ic_decay")}
    quant = {(a.get("feature"), int(a.get("horizon", -1))) for a in _agent_runs("quantile_returns")}
    todo = []
    for pt in (plan.tests if plan else []):
        if focus and pt.feature != focus:
            continue
        need = sorted(pt.controls)
        for h in pt.horizons:
            if not any(x.feature == pt.feature and x.horizon == h and set(need) <= set(x.controls) for x in obs):
                todo.append(f"{pt.feature} at h={h} with controls {need}: ic_summary(feature='{pt.feature}', horizon={h},"
                            f" controls={need}), then record_research_observation")
        if pt.feature not in decay:
            todo.append(f"{pt.feature}: horizon profile: ic_decay(feature='{pt.feature}')")
        if (pt.feature, pt.horizons[0]) not in quant:
            todo.append(f"{pt.feature}: shape: quantile_returns(feature='{pt.feature}', horizon={pt.horizons[0]})")
    out = {"observations": len(obs), "planned_tests_open": len(todo), "todo": todo[:8]}
    if not todo:
        out["next"] = ("nothing is owed: call submit now with the wrap-up (summary, warnings, open_questions); do not "
                       "write the summary as text")
    return out


def _ungrounded_numbers(texts: list[str]) -> list[float]:
    """Decimals in the wrap-up that no result of this pass's runs (or recorded observation) contains."""
    import json
    from quantaccelerator.tools import registry
    from quantaccelerator.tools.features import NUM, _numbers
    known = [x for o in CTX.findings.get("research_observations", []) for x in _numbers(o.stats)]
    if registry.CTX.session_id:
        r = registry.session_runs(registry.CTX.session_id)
        for path in r[r.agent == registry.CTX.agent].output_path:
            p = path if path.startswith("/") else str(registry.ROOT / path)
            try:
                known += _numbers(json.loads(open(p).read()).get("result"))
            except (OSError, ValueError):
                pass
    known += [100 * k for k in known]
    return [x for t in texts for x in map(float, NUM.findall(t))
            if not any(abs(x - k) <= 0.006 + 0.005 * abs(k) for k in known)]


def check_predictive_wrapup(wrap) -> list[str]:
    bad = _ungrounded_numbers([wrap.summary, *wrap.warnings]) if wrap is not None else []
    if bad:
        return [f"the wrap-up states {bad[:3]}, which no result of your runs contains: copy numbers exactly from the "
                "results (or leave them out of the summary)"]
    st = status()
    if st["planned_tests_open"]:
        return [f"the pre-registration is not complete ({st['planned_tests_open']} items open; a null result is a "
                "result, and every planned horizon and control set is owed): " + "; ".join(st["todo"][:6])]
    return []


def reset() -> None:
    CTX.findings.pop("research_observations", None)
