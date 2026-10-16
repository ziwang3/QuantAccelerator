"""Research EDA, phase B (predictive): pre-register, then test the frozen features against returns on discovery.

Two agents with separate information boundaries:
- the Predictive Planner sees the frozen feature set (ideas, mechanisms, measured exposures) and no returns; it writes
  the PredictionPlan (expected sign, horizons from the mechanism's speed, controls) that gate G4 signs;
- the Predictive EDA Agent tests that plan through the returns vault, on the discovery segment only, and records one
  ResearchObservation per test. It cannot build features; validation and the locked test are run by the orchestrator.
"""
from quantaccelerator.agents.base import Agent
from quantaccelerator.state.schemas import PredictionPlanDraft, PredictiveWrapUp

PLAN_ROLE = """You are the Predictive Planner of a quantitative research team. A researcher's feature ideas were built
and frozen (gate G3) without ever looking at returns. Before anyone sees a return, you write the pre-registration: for
each frozen feature, what relation with future returns its idea's mechanism predicts, over which holding horizons, and
which characteristics it must survive. The researcher signs the plan (gate G4), and every test run later is judged
against it, so commit to what the mechanism implies, not to what might look good."""

PLAN_TASK = """Write the PredictionPlan: one PlannedTest per frozen feature (all of them, none other).
- expected_sign: +1 if the mechanism says a higher value precedes higher returns, -1 if lower, 0 if the idea gives no
  directional reason (e.g. a feature kept as a characteristic or a control).
- horizons, from the menu 1, 5, 20, 60 sessions, primary first. Choose from how fast the mechanism should play out and
  how often the feature changes: a daily surprise in trading behavior acts within days; a slow-moving characteristic
  over weeks to months. horizon_rationale says why in one or two sentences.
- controls: the characteristics the relation must survive, from the feature's known exposures in the context (e.g.
  dollar_adv for a feature that still scales with liquidity, size for one tied to company size, sector for a sector
  tilt; reversal or momentum when the feature is built from recent trading). Leave it empty when the feature has no
  material exposure; the raw relation is always examined too.
- conditioners: where the effect may differ (liquidity, size, sector, year), if the mechanism suggests it.
Do not call any tool; submit the plan directly."""

ROLE = """You are the Predictive EDA Agent of a quantitative research team (Research EDA, phase B). The features were
frozen before any return was seen, and the plan you test was signed before the returns vault opened. Your question is
whether each economically motivated, point-in-time feature appears to carry information about future returns, at
which horizons, and whether that survives the characteristics it is exposed to. You see only the discovery segment;
the orchestrator re-runs every 'informative' finding on later data you never see, and a locked test follows.
Rules of evidence:
- every test you run is counted; a finding is 'informative' only if |t_nw| clears the hurdle the tools report. Run
  the planned tests first, and only the extra tests that answer a question about them (decay, shape, controls).
- report null results as results: a feature that carries no information is a finding, not a failure.
- a raw relation that disappears once a characteristic is controlled is explained by that characteristic, not a new
  signal; say which one.
- you cannot build or change features; never suggest re-tuning a feature on returns.
Between tool calls write at most one or two plain sentences; findings go into record_research_observation."""

TASK = """Research EDA, phase B: test the signed prediction plan on the discovery segment. This pass covers one
frozen feature (`this_pass` in the context, with its planned test); the other features get their own passes.
1. describe_prediction_setup.
2. For each planned feature: ic_decay for the horizon profile; quantile_returns at the primary horizon for the shape;
   then at EVERY planned horizon, ic_summary with the planned controls (and the raw relation where it helps to tell
   what the controls explain). When a raw relation clears the hurdle, check it with the controls its exposures
   suggest (dollar_adv, size, sector, reversal, momentum, beta).
3. Record each planned horizon as soon as its runs establish it (record_research_observation): feature, horizon, the
   controls of the ic_summary run the verdict rests on (at least the planned ones), verdict, sign, a one-sentence
   claim with numbers copied from the results, and the run_ids. Tests off the plan need deviation_from_plan.
   record_research_observation reports what the plan still owes.
4. When nothing is owed, submit the wrap-up: what carries information, at which horizons, what is explained by known
   characteristics, and open questions."""


class PredictivePlanner(Agent):
    name = "predictive_planner"
    role_prompt = PLAN_ROLE
    output_model = PredictionPlanDraft
    tool_names: list[str] = []
    max_steps = 6
    max_submit_failures = 4
    max_output_tokens = 4096
    frozen: list[str] = []

    def extra_validate(self, plan) -> list[str]:
        return check_prediction_plan(plan, self.frozen)


class PredictiveEDA(Agent):
    name = "predictive_eda"
    role_prompt = ROLE
    output_model = PredictiveWrapUp
    max_steps = 40
    max_tool_calls_per_turn = 2
    max_submit_failures = 8
    max_output_tokens = 3072
    max_prose_chars = 600
    compact_after_turns = 3
    prose_reminder = ("Keep reasoning to a few sentences, and record each finding with record_research_observation "
                      "instead of writing it as text.")
    unpaced_tools = ("record_research_observation",)

    @property
    def tool_names(self) -> list[str]:
        from quantaccelerator.tools.predictive import PRED_TOOLS, RECORD_TOOLS
        return PRED_TOOLS + RECORD_TOOLS

    def extra_validate(self, wrap) -> list[str]:
        from quantaccelerator.tools.predictive import check_predictive_wrapup
        return check_predictive_wrapup(wrap)


def check_prediction_plan(plan, frozen: list[str]) -> list[str]:
    feats = [t.feature for t in plan.tests]
    errs = []
    unknown = [f for f in feats if f not in frozen]
    if unknown:
        errs.append(f"not frozen features: {unknown}; the frozen set is {frozen}")
    missing = [f for f in frozen if f not in feats]
    if missing:
        errs.append(f"every frozen feature needs a planned test; missing: {missing}")
    dup = sorted({f for f in feats if feats.count(f) > 1})
    if dup:
        errs.append(f"one PlannedTest per feature (put several horizons in one test): {dup}")
    for t in plan.tests:
        if not t.horizons:
            errs.append(f"{t.feature}: give at least one horizon (1, 5, 20 or 60)")
        if len(t.horizon_rationale.strip()) < 15:
            errs.append(f"{t.feature}: horizon_rationale must say why these horizons fit the mechanism")
    return errs
