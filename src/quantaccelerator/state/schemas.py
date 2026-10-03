"""Typed shared state (trimmed from AGENT_BUILD_PLAN §5.4).

Agents emit the `*Draft` models (what an LLM can know). The orchestrator wraps each in the full state
object, attaching provenance and deterministic facts (file hashes, tool-verified numbers).
"""
import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---- shared ---------------------------------------------------------------------------------------

class Provenance(BaseModel):
    producer_agent: str
    created_utc: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    inputs: list[str] = []
    tool_runs: list[str] = []
    model: str | None = None
    prompt_hash: str | None = None


# ---- DatasetCard ----------------------------------------------------------------------------------

FIELD_KINDS = Literal["identifier", "time", "code", "level", "flow", "stock", "event", "estimate", "revision", "score",
                      "ratio", "text", "flag", "other"]


class FieldCard(BaseModel):
    table: str = Field(description="Table name, e.g. 'nonderiv_trans'")
    name: str = Field(description="Column name exactly as in the table")
    dtype: str
    meaning: str = Field(description="What the field means, grounded in the docs or profiling")
    value_meanings: dict[str, str] = Field(default_factory=dict,
                                           description="For coded fields: code -> meaning (e.g. TRANS_CODE)")
    evidence: list[str] = Field(description="Doc refs written as the doc id, a space, then p and the page number "
                                "(for example 'readme p3'), and/or tool run_ids")
    confidence: float = Field(ge=0, le=1)
    kind: FIELD_KINDS | None = Field(None, description="What sort of quantity the field is (see the enum)")
    unit: str | None = Field(None, description="Unit of measurement, e.g. 'shares', 'USD', 'percent'; null if none")
    zero_vs_missing: str | None = Field(None, description="For numeric fields: what a zero means and what a missing "
                                        "value or an absent row means, and whether they differ")
    not_equivalent_to: list[str] = Field(default_factory=list, description="Concepts this field is easily confused "
                                         "with but is not (e.g. 'short interest' for short-sale volume)")

    @field_validator("value_meanings", "not_equivalent_to", mode="before")
    @classmethod
    def _none_is_empty(cls, v, info):
        """Models often send null for 'none'."""
        if v is None:
            return {} if info.field_name == "value_meanings" else []
        return v


class TableCard(BaseModel):
    table: str
    observation_unit: str = Field(description="What one row represents")
    primary_key: list[str]
    n_rows_run_id: str = Field(description="run_id of the profile_table call that measured this table")


class EntityType(BaseModel):
    type: str = Field(description="What kind of entity, e.g. 'common stocks', 'ETFs', 'preferred shares', 'warrants', "
                      "'macro series', 'filers'")
    share: str = Field(description="How much of the data it is, with the deciding number (e.g. '71% of rows, 98% of "
                       "volume'), or 'unknown'")
    evidence: list[str] = Field(description="run_ids (e.g. identifier_patterns) and doc refs")


class DatasetOverview(BaseModel):
    """What the dataset is, at a glance: the first thing a researcher needs before any field."""
    summary: str = Field(description="Two or three plain sentences: what this dataset is and what it is about")
    publisher: str = Field(description="Who produces it and how it is distributed (e.g. 'FINRA, one public file per "
                           "trading day')")
    purpose: str = Field(description="Why it exists / what it is collected for")
    market: str = Field(description="Market, country and asset class it covers, and which venues or sources (e.g. "
                        "'US equities, off-exchange trades reported to FINRA')")
    entities: list[EntityType] = Field(description="What one entity is and which kinds of entities are in the data, "
                                       "measured, not assumed")
    values: list[str] = Field(description="What the measured values are and their units (e.g. 'daily short-sale share "
                              "volume'); say explicitly what is NOT in the data (e.g. no prices or returns)")
    frequency: str = Field(description="How often records occur (e.g. daily per security, per filing, per release)")
    period: str = Field(description="First and last date covered, from a tool run")
    typical_uses: list[str] = Field(description="What researchers typically use it for")
    caveats: list[str] = Field(default_factory=list, description="What a newcomer most easily gets wrong about it")
    evidence: list[str] = Field(description="Doc refs ('<doc> p<page>') and tool run_ids behind the overview")


class DatasetCardDraft(BaseModel):
    dataset_id: str
    tables: list[TableCard]
    entity_id_fields: list[str] = Field(description="Identifier columns (e.g. of filings, issuers, insiders or series)")
    time_fields: list[str] = Field(description="All date/time columns as 'table.COLUMN'")
    fields: list[FieldCard]
    warnings: list[str] = []


# ---- Data research (pre-PIT, return-blind) ------------------------------------------------------------

RATING = Literal["weak", "moderate", "strong"]


class CapabilityRating(BaseModel):
    rating: RATING
    evidence: list[str] = Field(description="run_ids of the tool calls (e.g. research_breadth, coverage_over_time) that "
                                "support the rating")
    note: str = Field(description="One sentence with the deciding numbers")


class ResearchCapabilityAssessment(BaseModel):
    """What the data can support, as an envelope rather than a single quality score."""
    history_depth: CapabilityRating
    cross_sectional_breadth: CapabilityRating
    update_frequency: CapabilityRating
    coverage_stability: CapabilityRating
    event_breadth: CapabilityRating | None = None
    suitable_for: list[str] = Field(description="Kinds of research the data supports well")
    weak_for: list[str] = Field(description="Kinds of research the data supports poorly")
    major_limits: list[str]


class PreprocessingConsideration(BaseModel):
    field: str = Field(description="'table.COLUMN', or a ratio 'table.A/table.B'")
    observations: list[str] = Field(description="What the data shows about this field, each with a run_id")
    candidate_processing: list[str] = Field(description="Candidate representations (log1p, ranks, ratios, changes, "
                                            "winsorizing, missing indicators...); ideas, not decisions")
    risks: list[str] = Field(default_factory=list, description="What goes wrong if the field is used raw")
    evidence: list[str]


class DataObservation(BaseModel):
    claim: str = Field(description="One factual statement about the data; cite dates and levels only from tool numbers")
    figure_run_id: str | None = Field(None, description="run_id of the figure tool call that shows it")
    evidence: list[str] = Field(description="run_ids (and doc refs) supporting the claim")
    severity: Literal["info", "warning", "issue"] = "info"
    next_question: str | None = Field(None, description="What this observation made you check next, if anything")


CLEANING_OPS = Literal["drop_duplicates", "drop_rows", "flag_rows", "missing_indicator"]
PREDICATE_OPS = Literal["==", "!=", "<", "<=", ">", ">=", "isna", "notna", "isin", "notin", "before", "on_or_after",
                        ">col", "<col"]


class Predicate(BaseModel):
    column: str
    op: PREDICATE_OPS = Field(description="'>col' / '<col' compare with another column named in value")
    value: str | float | int | list[str] | None = None


class CleaningStep(BaseModel):
    """One deterministic cleaning step. Flags (a new boolean column) are preferred over dropping rows."""
    op: CLEANING_OPS
    table: str
    columns: list[str] = Field(default_factory=list, description="drop_duplicates: the key; missing_indicator: the "
                               "fields to flag")
    where: list[Predicate] = Field(default_factory=list, description="drop_rows / flag_rows: rows matching ALL "
                                   "predicates")
    flag_name: str | None = Field(None, description="flag_rows / missing_indicator: name of the new column")
    reason: str
    evidence: list[str] = Field(description="run_ids showing why the step is needed")


class CleaningRecipe(BaseModel):
    steps: list[CleaningStep] = []
    summary: str = ""


class DataResearchDraft(BaseModel):
    """Output of the Data Research Agent's profiling pass (pass 2)."""
    capability: ResearchCapabilityAssessment
    preprocessing: list[PreprocessingConsideration]
    observations: list[DataObservation]
    cleaning: CleaningRecipe
    warnings: list[str] = []
    open_questions: list[str] = Field(default_factory=list, description="Questions only the human or the vendor can "
                                      "answer")
    key_corrections: dict[str, list[str]] = Field(default_factory=dict, description="table -> corrected primary key, "
                                                  "when your key checks show the DatasetCard's key is wrong or has "
                                                  "columns it does not need")


class ResearchWrapUp(BaseModel):
    """What the Data Research Agent submits at the end; its observations, capability ratings, preprocessing notes and
    cleaning steps were recorded (and checked) one by one with the record_* tools."""
    suitable_for: list[str] = Field(description="Kinds of research the data supports well")
    weak_for: list[str] = Field(description="Kinds of research the data supports poorly")
    major_limits: list[str]
    warnings: list[str] = []
    open_questions: list[str] = Field(default_factory=list, description="Questions only the human or the vendor can "
                                      "answer")
    key_corrections: dict[str, list[str]] = Field(default_factory=dict, description="table -> corrected primary key, "
                                                  "when your key checks show the DatasetCard's key is wrong or has "
                                                  "columns it does not need")
    cleaning_summary: str = ""


class DataAnswer(BaseModel):
    """The Data Research Agent's answer to a researcher's follow-up question."""
    answer: str = Field(description="Direct answer in a few sentences; numbers and dates only from tool results")
    evidence: list[str] = Field(description="run_ids of the tool calls behind the answer (and doc refs)")
    figure_run_ids: list[str] = Field(default_factory=list, description="run_ids of figures that show the answer")
    caveats: list[str] = []


class FileRef(BaseModel):
    path: str
    sha256: str


class DatasetCard(DatasetCardDraft):
    overview: DatasetOverview | None = None             # written by its own step (the Overview agent), when run
    files: list[FileRef]
    provenance: Provenance
    research: DataResearchDraft | None = None            # the profiling pass, when run
    research_provenance: Provenance | None = None


# ---- PITContract ----------------------------------------------------------------------------------

SOURCE_FIELDS = Literal["edgar_acceptance.acceptanceDateTime", "submission.FILING_DATE",
                        "submission.PERIOD_OF_REPORT", "nonderiv_trans.TRANS_DATE",
                        "nonderiv_trans.DEEMED_EXECUTION_DATE",
                        "vintages.realtime_start", "vintages.date", "shvol.Date"]


class AvailabilityRule(BaseModel):
    """Machine-applicable rule: which timestamp makes a record public and when it can first be traded.

    tradable_at:
      next_open_after        first 09:30 ET session open strictly after the timestamp
      same_date_open         the open of the timestamp's calendar date (rolled forward to a trading day)
      next_trading_day_open  the open of the first trading day after the timestamp's date
    """
    source_field: SOURCE_FIELDS
    source_timezone: Literal["UTC", "America/New_York"] = Field(
        "America/New_York", description="Timezone in which the source values are actually expressed")
    tradable_at: Literal["next_open_after", "same_date_open", "next_trading_day_open"]
    extra_lag_trading_days: int = Field(0, ge=0, le=10)
    value_vintage: Literal["as_of_date", "latest", "not_applicable"] = Field(
        "not_applicable", description="For data that is revised after release: which vintage of a value may be used "
        "at a given date. as_of_date = the value as published and current on that date; latest = today's revised "
        "value; not_applicable = values are never revised")


PIT_ROLES = ("event_time", "publication_time", "reporting_period", "amendment_reference", "other")


class PITField(BaseModel):
    field: str = Field(description="'table.COLUMN'")
    role: Literal["event_time", "publication_time", "reporting_period", "amendment_reference", "other"] = Field(
        description="The field's role in time (not its kind: a value, code or identifier is 'other')")

    @field_validator("role", mode="before")
    @classmethod
    def _unknown_role_is_other(cls, v):
        """Card kinds ('level', 'flow', ...) are not time roles; anything outside the list is 'other'."""
        return v if v in PIT_ROLES else "other"
    explanation: str
    evidence: list[str]


class PITContractDraft(BaseModel):
    dataset_id: str
    fields: list[PITField]
    availability_rule: AvailabilityRule
    rule_explanation: str = Field(description="Plain-language statement of the availability and first-tradable rule")
    alternatives: list[str] = Field(default_factory=list,
                                    description="Other plausible rules considered, each with why it was rejected")
    evidence: list[str] = Field(description="Tool run_ids and doc refs supporting the chosen rule")
    revision_policy: str
    universe_policy: str
    backfill_risk: str = Field("", description="Can published records change or appear later (restatements, "
                               "updated files, late additions)? What does that mean for a backtest?")
    valid_join_rule: str = Field("", description="How to join this data to a decision date without look-ahead, e.g. "
                                 "'as-of join on first tradable session <= decision date'")
    open_questions: list[str] = []
    confidence: float = Field(ge=0, le=1)


class GateDecision(BaseModel):
    gate: str = "G1"                     # G0 = data review (cleaning recipe), G1 = data contract (availability rule),
                                         # G2 = idea review (feature ideas), G3 = feature freeze (candidate set),
                                         # G4 = prediction plan + split, G5 = locked-test release
    decided_by: Literal["human", "oracle"]
    approved: bool
    note: str = ""
    corrected_rule: AvailabilityRule | None = None
    approved_recipe: CleaningRecipe | None = None   # G0: the recipe as approved (possibly edited)
    approved_ideas: list[str] | None = None         # G2: the feature ideas approved for building
    approved_features: list[str] | None = None      # G3: the candidate features frozen (possibly a subset)
    approved_plan: "PredictionPlan | None" = None   # G4: the prediction plan as approved, with the locked split
    released_observations: list[int] | None = None  # G5: observations (indexes) released to the locked test


class PITContract(PITContractDraft):
    signed_off: GateDecision | None = None
    provenance: Provenance


# ---- AuditFinding ---------------------------------------------------------------------------------

LEAK_TYPES = Literal[
    "event_date_as_availability", "timezone_session_ambiguity", "same_bar_execution", "revised_vs_first_release",
    "backfilled_history", "forward_fill_period_end", "full_sample_normalization", "survivorship",
    "label_overlap", "corporate_action", "semantic_confusion", "future_data_in_window"]


class Location(BaseModel):
    file: str
    line: int = Field(ge=1)


class LineEdit(BaseModel):
    line: int = Field(ge=1, description="1-based line number to replace")
    new_code: str = Field(description="Replacement source for that line (keep indentation)")


class Impact(BaseModel):
    metric: Literal["share_events_before_tradable"] = "share_events_before_tradable"
    before: float
    before_run_id: str
    after: float | None = None
    after_run_id: str | None = None


class AuditFindingDraft(BaseModel):
    location: Location
    leak_type: LEAK_TYPES
    description: str
    estimated_impact: Impact
    proposed_patch: list[LineEdit]
    confidence: float = Field(ge=0, le=1)


class AuditReportDraft(BaseModel):
    pipeline: str
    findings: list[AuditFindingDraft] = Field(description="Empty if the pipeline is PIT-correct")
    summary: str
    evidence: list[str] = Field(description="Tool run_ids supporting the conclusion (including for 'no leak')")


class AuditFinding(AuditFindingDraft):
    pipeline_ref: str
    verified_by_run: str | None = None
    verified_share_before_tradable: float | None = None
    provenance: Provenance


class AuditReport(BaseModel):
    pipeline: str
    findings: list[AuditFinding]
    summary: str
    evidence: list[str]
    provenance: Provenance


# ---- Research EDA, phase A (post-PIT, return-blind) -------------------------------------------------------------

FEATURE_OPS = Literal["field", "ratio", "diff", "log1p", "change", "growth", "surprise", "rolling_mean", "rolling_sum",
                      "rolling_std", "peer_relative", "residualize", "rank"]
CONDITIONER = Literal["adv_proxy", "size_float", "size_assets", "sector", "time", "none"]
FEATURE_DECISIONS = Literal["keep", "normalize", "residualize", "bucket", "drop", "unresolved"]


class FeatureSpec(BaseModel):
    """One feature in a closed construction language, computed deterministically on the research panel."""
    name: str = Field(description="New feature name: lower case letters, digits and _, e.g. 'short_ratio'")
    op: FEATURE_OPS = Field(description="field: a panel field as is; ratio: inputs[0] / inputs[1]; diff: inputs[0] - "
                            "inputs[1]; log1p; change: "
                            "x - x `window` records earlier; growth: that change / |earlier x|; surprise: (x - mean of the "
                            "previous `window` records) / their std; rolling_mean / rolling_sum / rolling_std over `window` records; "
                            "peer_relative: x minus the same-date median of its `group` (or of all entities); "
                            "residualize: x minus what `group` explains on the same date (linear in log of a size or "
                            "liquidity conditioner, group means for sector); rank: same-date percentile rank (within "
                            "`group` if given)")
    inputs: list[str] = Field(description="Panel fields, conditioners or earlier features: two for ratio and diff, one "
                              "otherwise")
    window: int | None = Field(None, ge=2, le=250, description="Records, for change / growth / surprise / rolling_*")
    group: str | None = Field(None, description="peer_relative / rank: 'sector' (or empty for all entities); "
                              "residualize: adv_proxy, size_float, size_assets or sector")
    note: str = Field("", description="One sentence: the economic idea behind the construction")


class FeatureIdea(BaseModel):
    """A research idea for a feature: the economic story first, then how to build it from the panel."""
    name: str = Field(description="Short identifier, lower case and _, e.g. 'abnormal_shorting'")
    idea: str = Field(description="The idea in one or two plain sentences")
    mechanism: str = Field(description="Why it could carry information: the economic mechanism")
    assumptions: list[str] = Field(description="What must be true for the mechanism to hold (stated, not tested here)")
    expected_direction: str = Field(description="The direction the mechanism suggests (a hypothesis for later testing "
                                    "with returns, never checked in this phase)")
    specs: list["FeatureSpec"] = Field(description="1-3 feature definitions that express the idea, in construction order "
                                       "(a definition may use earlier ones); the last ones are the representations")
    checks: list[str] = Field(default_factory=list, description="What to examine once built (distribution, variation "
                              "across stocks and over time, persistence, dependence on liquidity, size, sector)")
    source: Literal["agent", "researcher"] = Field("agent", description="Who proposed the idea")
    researcher_text: str | None = Field(None, description="For a researcher's idea: their words, verbatim")


class IdeaPlan(BaseModel):
    """Research EDA ideation pass: the ideas proposed for review at gate G2."""
    ideas: list[FeatureIdea]
    notes: str = Field("", description="One or two sentences on how the ideas relate to what the survey found")


class FeatureObservation(BaseModel):
    """One characterization finding about a feature (doc: possible explanations before any neutralization)."""
    feature: str = Field(description="The feature or panel field the observation is about")
    idea: str | None = Field(None, description="The idea it belongs to (its name), if any")
    claim: str = Field(description="One factual statement; numbers and dates only from tool results")
    conditioner: CONDITIONER = Field("none", description="What it was characterized against ('time' for stability, "
                                     "'none' for the distribution itself)")
    evidence: list[str] = Field(description="run_ids of the tool calls that establish it")
    figure_run_id: str | None = None
    possible_explanations: list[str] = Field(default_factory=list, description="For a dependence: the explanations "
                                             "you cannot yet tell apart, e.g. measurement bias, economic scale, a genuine "
                                             "interaction")
    candidate_representations: list[str] = Field(default_factory=list, description="Representations worth comparing "
                                                 "(feature names you built, or short descriptions)")
    required_checks: list[str] = Field(default_factory=list, description="What would decide between the explanations")
    decision: FEATURE_DECISIONS = Field("unresolved", description="What this implies for the representation: keep "
                                        "as is, normalize by scale, residualize, study within buckets, drop, or "
                                        "unresolved")


class CandidateFeature(BaseModel):
    name: str = Field(description="A feature you constructed")
    idea: str | None = Field(None, description="The idea it expresses (its name)")
    rationale: str = Field(description="Why this representation, in economic terms")
    known_exposures: list[str] = Field(default_factory=list, description="Measured dependences it still has (with "
                                       "run_ids), e.g. 'mild size tilt, rho 0.12 (r...)'")
    evidence: list[str] = Field(description="run_ids of the characterization runs (exposures, stability) behind it")


class DataInvestigationRequest(BaseModel):
    """A question research EDA cannot answer itself, routed back to the Data Research or the PIT agent."""
    observation: str
    question: str
    route_to: Literal["data_research", "pit"]
    evidence: list[str] = []


class InvestigationResult(BaseModel):
    """A DataInvestigationRequest after routing: who answered, the grounded answer (or why there is none)."""
    request: DataInvestigationRequest
    stage: Literal["research_eda", "predictive_eda"]
    routed_to: str
    answer: "DataAnswer | None" = None
    error: str | None = None


class SurveyWrapUp(BaseModel):
    """Research EDA pass 1 (survey): the raw fields characterized, and a plan per field for pass 2."""
    plan: list[str] = Field(description="One line per value field: what its characterization showed (with run_ids) "
                            "and what pass 2 should try (a representation and why), or that it is comparable as is")


class EDAWrapUp(BaseModel):
    """What the Research EDA Agent submits at the end; observations and candidates were recorded one by one."""
    summary: str = Field(description="A few sentences: what the candidate set is and why")
    warnings: list[str] = []
    open_questions: list[str] = Field(default_factory=list, description="Questions for the researcher")
    data_investigation_requests: list[DataInvestigationRequest] = []


class CandidateFeatureSet(BaseModel):
    """The return-blind feature set, frozen at gate G2 before any predictive work may see returns."""
    panel_hash: str
    panel: dict = Field(default_factory=dict, description="The panel's description (universe, dates, rule)")
    features: list[FeatureSpec] = Field(description="Every feature built in the session, in construction order "
                                        "(the candidates and what they are built from among them)")
    ideas: list[FeatureIdea] = Field(default_factory=list, description="The ideas proposed (G2 decides which are built)")
    idea_gate: GateDecision | None = None
    candidates: list[CandidateFeature]
    observations: list[FeatureObservation]
    wrapup: EDAWrapUp
    signed_off: GateDecision | None = None
    set_hash: str = ""
    provenance: Provenance | None = None


# ---- Research EDA, phase B (predictive, after the G3 freeze) ----------------------------------------------------

PRED_HORIZONS = Literal[1, 5, 20, 60]
PRED_CONTROLS = Literal["size", "reversal", "momentum", "beta", "dollar_adv", "sector"]
PRED_CONDITIONERS = Literal["dollar_adv", "size", "beta", "sector", "year"]
PRED_VERDICTS = Literal["informative", "not_informative", "explained_by_control", "unstable", "inconclusive"]


class SplitSpec(BaseModel):
    """Decision-date segments, locked at G4. Each segment's returns stop at its end; the gaps are embargoes."""
    discovery: tuple[str, str]
    validation: tuple[str, str]
    locked_test: tuple[str, str]
    embargo_sessions: int = Field(20, description="Minimum sessions between the end of one segment and the next")


class PlannedTest(BaseModel):
    """One pre-registered test: written before any return is seen."""
    feature: str = Field(description="A frozen feature")
    idea: str | None = Field(None, description="The idea it expresses")
    expected_sign: Literal[-1, 0, 1] = Field(description="Sign of the relation with forward returns the idea's "
                                             "mechanism predicts (+1 higher feature, higher return; -1 lower; 0 none "
                                             "expected, e.g. a control idea)")
    horizons: list[PRED_HORIZONS] = Field(description="Holding horizons in sessions, from the mechanism's speed and "
                                          "the feature's update frequency (the first is the primary horizon)")
    horizon_rationale: str = Field(description="Why these horizons, in one or two sentences")
    controls: list[PRED_CONTROLS] = Field(default_factory=list, description="Characteristics the relation must survive "
                                          "(the feature is residualized on them), chosen from its known exposures")
    conditioners: list[PRED_CONDITIONERS] = Field(default_factory=list, description="Where the effect may differ "
                                                  "(conditional IC by bucket)")


class PredictionPlanDraft(BaseModel):
    """Phase B pre-registration, written before any return is seen."""
    tests: list[PlannedTest]
    notes: str = Field("", description="One or two sentences on the plan")


class PredictionPlan(PredictionPlanDraft):
    """The pre-registration with its feature set and the locked split, as signed at gate G4."""
    set_hash: str = ""
    split: SplitSpec | None = None


class ResearchObservationDraft(BaseModel):
    """What the Predictive EDA Agent records: one claim about whether a frozen feature carries information."""
    feature: str = Field(description="The frozen feature")
    horizon: PRED_HORIZONS = Field(description="The horizon the claim is about")
    controls: list[PRED_CONTROLS] = Field(default_factory=list, description="Controls of the ic_summary run the verdict "
                                          "rests on (empty for the raw relation)")
    verdict: PRED_VERDICTS = Field(description="informative: |t| clears the hurdle; not_informative: it does not; "
                                   "explained_by_control: significant raw, gone once controlled; unstable: the sign "
                                   "changes across years or buckets; inconclusive: otherwise")
    sign: Literal[-1, 0, 1] = Field(description="Sign of the relation measured (0 for none)")
    claim: str = Field(description="One factual sentence; numbers only from the cited tool results")
    evidence: list[str] = Field(description="run_ids of the predictive tool calls that establish it")
    deviation_from_plan: str = Field("", description="Empty when the test was pre-registered; otherwise why it was run")


class ResearchObservation(ResearchObservationDraft):
    idea: str | None = None
    stats: dict = Field(default_factory=dict, description="The primary ic_summary run's numbers (filled by the tool)")
    primary_run: str | None = None
    n_trials_at_claim: int = 0
    hurdle_t: float = 3.0
    status: Literal["discovery", "validated", "failed_validation", "not_validated", "locked_pass", "locked_fail"] = \
        "discovery"
    validation: dict | None = None
    critique: dict | None = Field(None, description="Critic outcomes: counts and the refuting attacks")
    costs: dict | None = Field(None, description="Cost and capacity of its long-short portfolio, by segment")
    locked_test: dict | None = None


CRITIC_THREATS = Literal["omitted_characteristic", "small_illiquid_names", "microstructure", "sector_concentration",
                         "time_instability"]
CRITIC_UNIVERSES = Literal["drop_illiquid_tercile", "price_ge_5", "large_caps", "exclude_sector"]


class CriticTest(BaseModel):
    """An executable attack on a finding, in a closed language; the orchestrator runs it on discovery."""
    kind: Literal["add_controls", "restrict_universe", "subperiod"]
    controls: list[PRED_CONTROLS] = Field(default_factory=list, description="add_controls: controls to add to the "
                                          "finding's own")
    universe: CRITIC_UNIVERSES | None = Field(None, description="restrict_universe: which names to KEEP; each removes "
                                              "the names a threat says drive the effect (drop the least liquid third "
                                              "by dollar_adv; previous close >= $5; the largest third by size; all but "
                                              "one sector)")
    sector: str | None = Field(None, description="exclude_sector: the sector removed")
    period: tuple[str, str] | None = Field(None, description="subperiod: [start, end] inside the discovery segment")


class Critique(BaseModel):
    """One way a finding could be spurious, and the test that would show it."""
    target: int = Field(description="Index of the finding attacked (from the context)")
    threat_type: CRITIC_THREATS
    threat: str = Field(description="One sentence: the alternative explanation, e.g. 'the effect is a bid-ask bounce "
                        "in illiquid names, not information'")
    test: CriticTest
    if_spurious: str = Field(description="What the test shows if the threat is right, e.g. 'the IC vanishes outside "
                             "the least liquid third'")


class CriticDraft(BaseModel):
    """The Critic's pre-registered attacks: written from the findings' outputs only, before any is run."""
    critiques: list[Critique]
    notes: str = ""


class CritiqueResult(Critique):
    source: Literal["critic", "standard", "propagated"] = Field("critic", description="the Critic's attack, the "
                                                                "standard battery run on every finding, or an attack "
                                                                "that refuted a sibling finding (same feature), re-run "
                                                                "on this one")
    run_id: str | None = None
    claim: dict = Field(default_factory=dict, description="The finding's discovery IC and t")
    attacked: dict = Field(default_factory=dict, description="IC, t and names of the attacked version")
    retention: float | None = Field(None, description="attacked IC / claimed IC")
    outcome: Literal["survived", "refuted", "inconclusive"] = "inconclusive"


class PredictiveWrapUp(BaseModel):
    """What the Predictive EDA Agent submits at the end; observations were recorded one by one."""
    summary: str = Field(description="A few sentences: which features carry information in discovery, at which "
                         "horizons, and what explains the rest")
    warnings: list[str] = []
    open_questions: list[str] = Field(default_factory=list, description="Questions for the researcher")
    data_investigation_requests: list[DataInvestigationRequest] = []


class PredictiveReport(BaseModel):
    """Phase B end to end: plan (G4), discovery observations, deterministic validation, release (G5), locked test."""
    set_hash: str
    plan: PredictionPlan
    plan_gate: GateDecision | None = None
    observations: list[ResearchObservation] = []
    wrapup: PredictiveWrapUp | None = None
    critiques: list[CritiqueResult] = []
    release_gate: GateDecision | None = None
    ledger: dict = Field(default_factory=dict, description="Trial count, hurdle and vault accesses")
    provenance: Provenance | None = None


GateDecision.model_rebuild()
InvestigationResult.model_rebuild()
PITContract.model_rebuild()
CandidateFeatureSet.model_rebuild()
