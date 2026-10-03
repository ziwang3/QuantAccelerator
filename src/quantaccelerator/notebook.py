"""QuantAccelerator in a Jupyter notebook: one Session per dataset, one method per research step, the human at every gate.

    import os; os.environ["QA_DATASET"] = "finra"      # before importing quantaccelerator: one dataset per kernel
    from quantaccelerator.notebook import Session
    s = Session()                  # needs LLM_BASE_URL / LLM_MODEL (a vLLM server in this Slurm job)
    s.understand()                 # Data Research Agent, with live progress and figures
    s.brief()                      # the Data Brief
    s.ask("why does coverage jump in 2021?")
    s.review_cleaning(); s.approve_cleaning(drop_steps=[1], note="...")     # gate G0
    s.time(); s.approve("...")  or  s.correct({...rule...}, "...")         # PIT agent, gate G1
    view = s.pit_view(); s.snapshot(["2024-03-28"])                         # the point-in-time research view
    s.explore(ideas=["..."]); s.approve_ideas()                           # Research EDA ideas (return-blind), gate G2
    s.features(); s.freeze_features(keep=[...], note="...")               # candidate set, gate G3
    s.plan_predictions(); s.approve_plan(note="...")                       # phase B pre-registration, gate G4; discovery
    s.observations(); s.release_test(keep=[...], note="...")               # validation results, gate G5, locked test
    s.audit("notebooks/finra_signal_leak_sameday.py")                      # Look-Ahead Audit of a pipeline
    s.export_html()

Everything is saved under runs/sessions/<session_id>/ like any other run (state/, transcripts/, tool_outputs/,
figures/, brief/), so a notebook session can be replayed and audited later.
"""
import datetime as dt
import html
import json
import os
from pathlib import Path

import pandas as pd

from quantaccelerator.agents import brief as brief_mod
from quantaccelerator.agents import orchestrator as O
from quantaccelerator.agents.profiler import DataQA
from quantaccelerator.datasets import active
from quantaccelerator.llm.client import Budget, LLMClient
from quantaccelerator.paths import RUNS
from quantaccelerator.state.schemas import AvailabilityRule, CleaningRecipe, DatasetCard, GateDecision
from quantaccelerator.tools import registry
from quantaccelerator.tools.clean import apply_recipe, preview_cleaning
from quantaccelerator.tools.pit_view import asof, build_pit_view
from quantaccelerator.viz.events import summarize

try:
    from IPython.display import HTML, Image, Markdown, display
except ImportError:  # plain Python: print instead
    HTML = Markdown = lambda x: x
    Image = lambda filename, **_: f"[figure {filename}]"
    display = print


class Session:
    def __init__(self, dataset: str | None = None, session_id: str | None = None, model: str | None = None,
                 temperature: float = 0.0, verbose: bool = False):
        p = active()
        if dataset and dataset != p.name:
            raise RuntimeError(f"this kernel is set up for dataset {p.name!r}; set os.environ['QA_DATASET'] = "
                               f"{dataset!r} before importing quantaccelerator (restart the kernel), one dataset per kernel")
        self.dataset = p.name
        self.session_id = session_id or f"{dt.datetime.now():%Y%m%d-%H%M%S}-{p.name}"
        self.run_dir = RUNS / "sessions" / self.session_id
        client = LLMClient(budget=Budget(max_calls=400, max_prompt_tokens=3_000_000, max_completion_tokens=400_000), model=model,
                           temperature=temperature)
        self.run_dir, self.client, self.meta, self.timed = O._start(self.session_id, self.run_dir, client,
                                                                   mode="notebook")
        self.verbose = verbose
        self.card: DatasetCard | None = None
        self.derived: dict[str, str] | None = None
        self.draft = self.trace = self.contract = self._view = self.feature_set = self.idea_plan = None
        self.gates: list[GateDecision] = []
        self._step = 0
        registry.CTX.listener = self._progress
        self._save_meta()
        display(Markdown(f"**QuantAccelerator session** `{self.session_id}` · dataset `{p.dataset_id}` · model "
                         f"`{self.client.model}` · saved under `{self.run_dir}`"))

    # ------------------------------------------------------------------ plumbing
    def _progress(self, name, run_id, args, result):
        """One line per step saying what the agent is examining; results and figures come in the Data Brief.
        With verbose=True the line also gives the result, and figures are shown as they are made."""
        self._step += 1
        if not self.verbose:
            print(f"{self._step}. {describe_call(name, args)}", flush=True)
            return
        line = summarize(name, args, result)
        display(Markdown(f"`{self._step:>2}` **{brief_mod.TOOL_VERB.get(name, name)}** <small>({registry.CTX.agent}, "
                         f"`{run_id}`)</small>: {html.escape(line)}"))
        fig = isinstance(result, dict) and result.get("figure")
        if fig and (self.run_dir / fig).exists():
            display(Image(filename=str(self.run_dir / fig), width=720))

    def _save_meta(self):
        b = self.client.budget
        self.meta["budget"] = {"calls": b.calls, "prompt_tokens": b.prompt_tokens,
                               "completion_tokens": b.completion_tokens}
        self.meta["gates"] = [g.model_dump(mode="json") for g in self.gates]
        (self.run_dir / "run_meta.json").write_text(json.dumps(self.meta, indent=1, default=str))

    def _gate(self, g: GateDecision):
        self.gates.append(g)
        (self.run_dir / "state").mkdir(exist_ok=True)
        (self.run_dir / "state" / f"gate_{g.gate.lower()}.json").write_text(g.model_dump_json(indent=1))
        self._save_meta()

    # ------------------------------------------------------------------ Understand
    def understand(self, research: bool = True) -> DatasetCard:
        """Data Research Agent: pass 1 (what the fields mean), pass 2 (what the data is like), then the Data Brief."""
        display(Markdown("### Understand: the Data Research Agent reads the docs, profiles the tables and studies the data"))
        try:
            self.card = O.understand(self.run_dir, self.client, self.timed, research, overview=True)
            self.meta["status"] = "understood"
        finally:
            self._save_meta()
        brief_mod.build(self.run_dir)
        display(Markdown(f"Done: {len(self.card.fields)} fields described"
                         + (f", {len(self.card.research.observations)} observations, "
                            f"{len(self.card.research.cleaning.steps)} cleaning steps proposed" if self.card.research
                            else "") + ". Call `s.brief()` to read the Data Brief."))
        return self.card

    def brief(self):
        """The Data Brief: what the data is, what the agent did and found, what it proposes, what it could not decide."""
        p = self.run_dir / "brief" / "brief.html"
        if not p.exists():
            raise RuntimeError("run s.understand() first")
        # an isolated frame keeps the brief's styles out of the notebook's
        import warnings
        warnings.filterwarnings("ignore", message="Consider using IPython.display.IFrame")
        display(HTML(f'<iframe srcdoc="{html.escape(p.read_text(), quote=True)}" '
                     'style="width:100%;height:900px;border:1px solid #e5e7eb;border-radius:8px"></iframe>'))

    def ask(self, question: str):
        """A follow-up question for the Data Research Agent; it answers with tools and cites them."""
        if self.card is None:
            raise RuntimeError("run s.understand() first")
        ctx = {"dataset_card": self.card.model_dump(mode="json", exclude={"provenance", "files",
                                                                          "research_provenance"})}
        agent = DataQA()
        agent.name = f"data_qa_{sum(1 for g in (self.run_dir / 'transcripts').glob('data_qa_*'))+1}"
        agent.given = question
        ans, _ = self.timed(agent, question, ctx)
        self._save_meta()
        display(Markdown(f"**Q:** {question}\n\n**A:** {ans.answer}"
                         + "".join(f"\n\n- caveat: {c}" for c in ans.caveats)
                         + f"\n\n<small>evidence: {', '.join(ans.evidence)}</small>"))
        for rid in ans.figure_run_ids:
            f = self.run_dir / "figures" / f"{rid}.png"
            if f.exists():
                display(Image(filename=str(f), width=720))
        return ans

    # ------------------------------------------------------------------ gate G0: cleaning
    def review_cleaning(self):
        """Gate G0 review: each proposed cleaning step as a card (red: removes rows, amber: flags rows, blue: adds a
        missing-value indicator) with the problem, the action, its measured impact, the evidence and example rows.
        Nothing is applied here."""
        from quantaccelerator.tools.clean import step_mask
        from quantaccelerator.tools.data import load_table
        steps = self.card.research.cleaning.steps if self.card and self.card.research else []
        if not steps:
            display(HTML(_card("none", "No cleaning proposed", "The Data Research Agent found nothing to remove or "
                               "flag.", [])))
            return
        cards, removed, flagged = [], 0, 0
        for i, st in enumerate(steps):
            df = load_table(st.table)
            m = step_mask(df, st)
            n, total = int(m.sum()), len(df)
            kind = "remove" if st.op in ("drop_duplicates", "drop_rows") else "flag" if st.op == "flag_rows" else "indicator"
            removed += n if kind == "remove" else 0
            flagged += n if kind != "remove" else 0
            what = {"remove": f"REMOVES {n:,} rows", "flag": f"FLAGS {n:,} rows (kept)",
                    "indicator": f"ADDS a missing-value indicator ({n:,} rows)"}[kind]
            rows = [("Problem", html.escape(st.reason)),
                    ("Action", f"<code>{html.escape(action_text(st))}</code>"),
                    ("Impact", f"<b>{n:,}</b> of {total:,} rows ({100 * n / max(total, 1):.3f}%) in "
                               f"<code>{html.escape(st.table)}</code>"),
                    ("Evidence", ", ".join(f"<code>{html.escape(e)}</code>" for e in st.evidence) or "–")]
            ex = df[m].head(3)
            table = ex.to_html(index=False, border=0, classes="qa-ex") if len(ex) else ""
            cards.append(_card(kind, f"Step {i} · {what}", "", rows, table))
        head = (f"<div style='font:600 14px system-ui;margin:4px 0 10px'>{len(steps)} proposed step(s): "
                f"<span style='color:#b42318'>{removed:,} rows would be removed</span> · "
                f"<span style='color:#9a6700'>{flagged:,} rows flagged and kept</span> · raw data is never modified"
                "</div>")
        display(HTML(CARD_CSS + head + "".join(cards)))
        display(Markdown("Approve with `s.approve_cleaning(drop_steps=[...], note=...)` (list the step numbers you do "
                         "not want) or `s.reject_cleaning(note)`."))

    def approve_cleaning(self, drop_steps: list[int] = (), note: str = "", recipe: CleaningRecipe | None = None):
        """Gate G0: approve the recipe (optionally without some steps, or a recipe you edited); writes derived tables."""
        proposed = self.card.research.cleaning if self.card and self.card.research else CleaningRecipe()
        recipe = recipe or CleaningRecipe(steps=[s for i, s in enumerate(proposed.steps) if i not in drop_steps],
                                          summary=proposed.summary)
        self._gate(GateDecision(gate="G0", decided_by="human", approved=True, note=note, approved_recipe=recipe))
        if recipe.steps:
            out = apply_recipe(recipe)
            self.derived = out["tables"]
            display(Markdown(f"Cleaning applied to derived copies (recipe `{out['recipe_hash']}`): "
                             + "; ".join(f"`{t}` → `{p}`" for t, p in out["tables"].items())))
        else:
            display(Markdown("Approved with no cleaning steps; the research view uses the tables as ingested."))

    def reject_cleaning(self, note: str):
        self._gate(GateDecision(gate="G0", decided_by="human", approved=False, note=note))
        display(Markdown("Cleaning rejected; the research view uses the tables as ingested."))

    # ------------------------------------------------------------------ Trust: PIT agent and gate G1
    def time(self):
        """PIT agent: when did each record become public and tradable? Shows the proposed contract for sign-off."""
        if self.card is None:
            raise RuntimeError("run s.understand() first")
        display(Markdown("### Trust: the PIT agent works out when each record could first be traded"))
        self.draft, self.trace = O.propose_contract(self.client, self.timed, self.card)
        self._save_meta()
        d = self.draft
        display(Markdown(
            f"**Proposed rule:** `{d.availability_rule.model_dump_json()}`\n\n{d.rule_explanation}\n\n"
            + "".join(f"- alternative considered: {a}\n" for a in d.alternatives)
            + f"\n**Revisions:** {d.revision_policy}\n\n**Backfill risk:** {d.backfill_risk or '–'}\n\n"
            f"**Join rule:** {d.valid_join_rule or '–'}\n\n**Confidence:** {d.confidence}\n\n"
            + "".join(f"- open question: {q}\n" for q in d.open_questions)))
        p = active()
        if p.view and p.first_tradable:
            sample = registry_sample(p.view["table"], p.view["time"])
            first = p.first_tradable(d.availability_rule, sample)
            t0, f0 = pd.Timestamp(sample[p.view["time"]].iloc[0]), pd.Timestamp(first.iloc[0])
            lag = "the SAME DAY" if f0.normalize() == t0.normalize() else f"{(f0 - t0.normalize()).days} calendar day(s) later"
            display(HTML(f"<div style='background:#fff4d6;border-left:5px solid #c98500;padding:8px 12px;border-radius:6px;"
                         f"font:14px system-ui;color:#1f2937'><b>In plain words:</b> under this rule, the record for "
                         f"{t0:%a %Y-%m-%d} is first used at the <b>open of {f0:%a %Y-%m-%d}</b> ({lag}). Check this "
                         "against when the data is published before you sign.</div>"))
            display(Markdown("**The rule on a few real records:**"))
            display(sample.assign(first_tradable_open=first.dt.date.values))
        display(Markdown("**Your decision (gate G1):** `s.approve(note)` if the rule is right, or "
                         "`s.correct({...rule...}, note)` with the rule it should be."))

    def approve(self, note: str = ""):
        return self._sign(GateDecision(decided_by="human", approved=True, note=note))

    def correct(self, rule: dict, note: str):
        return self._sign(GateDecision(decided_by="human", approved=False, note=note,
                                       corrected_rule=AvailabilityRule.model_validate(rule)))

    def _sign(self, g: GateDecision):
        if self.draft is None:
            raise RuntimeError("run s.time() first")
        self.contract = O.sign_contract(self.run_dir, self.client, self.draft, self.trace, g)
        self._view = None
        self._gate(g)
        display(Markdown(f"Contract signed ({'approved' if g.approved else 'corrected'}); rule in force: "
                         f"`{registry.CTX.reference_rule.model_dump_json()}`"))
        return self.contract

    # ------------------------------------------------------------------ the research view
    def pit_view(self) -> pd.DataFrame:
        """Every record with its decision_time under the signed rule (and any approved cleaning flags)."""
        if self._view is None:
            self._view = build_pit_view(self.contract, self.derived)
        return self._view

    def snapshot(self, dates, entities=None) -> pd.DataFrame:
        """What was knowable at each date's open, per entity, with the record's age in trading days."""
        return asof(self.pit_view(), dates, entities)

    # ------------------------------------------------------------------ Research EDA, phase A, and gate G2
    def explore(self, ideas: list[str] | None = None):
        """Research EDA, passes 1-2 (return-blind): the agent surveys the raw fields on the point-in-time panel, then
        proposes feature ideas, yours (`ideas`, in plain words) translated and its own, for your review (G2)."""
        if self.contract is None:
            raise RuntimeError("sign the PIT contract first (s.time(), then s.approve()): research works on the PIT view")
        display(Markdown("### Research EDA: survey the data, then propose feature ideas"))
        _, meta = O.build_research_panel(self.run_dir, self.contract, self.derived)
        display(Markdown(f"Research panel: {meta['n_entities']:,} entities × {meta['n_dates']:,} decision dates "
                         f"({meta['n_rows']:,} rows), universe fixed from {meta['formation'][0]}..{meta['formation'][1]}."))
        try:
            self.idea_plan, self._eda_ctx = O.propose_ideas(self.run_dir, self.client, self.timed, self.card, ideas)
            self.meta["status"] = "ideas proposed"
        finally:
            self._save_meta()
        self.ideas()
        display(Markdown("**Your decision (gate G2):** `s.approve_ideas()` to build them all, "
                         "`s.approve_ideas(keep=[...], note=...)` for a subset, or `s.reject_ideas(note)`."))
        return self.idea_plan

    def ideas(self):
        """The proposed feature ideas: the story, the assumptions, and how each would be built."""
        plan = getattr(self, "idea_plan", None)
        if plan is None:
            raise RuntimeError("run s.explore() first")
        from quantaccelerator.tools.features import formula
        rows = []
        for i in plan.ideas:
            defs = "<br>".join(f"<code>{html.escape(sp.name)}</code> = {html.escape(formula(sp))}" for sp in i.specs)
            rows.append(f"<tr><td><b>{html.escape(i.name)}</b><br><small>{i.source}</small></td>"
                        f"<td>{html.escape(i.idea)}<br><small><i>Mechanism:</i> {html.escape(i.mechanism)}<br>"
                        f"<i>Assumes:</i> {html.escape('; '.join(i.assumptions))}<br>"
                        f"<i>Expected (untested):</i> {html.escape(i.expected_direction)}</small></td><td>{defs}</td></tr>")
        display(HTML("<table style='font:13px system-ui;border-collapse:collapse'><tr><th>Idea</th><th>What and why</th>"
                     "<th>How it is built</th></tr>" + "".join(rows) + "</table>"))

    def approve_ideas(self, keep: list[str] | None = None, note: str = ""):
        """Gate G2: build the ideas you keep (all by default); the agent then characterizes them and proposes
        candidates for the freeze (G3)."""
        plan = getattr(self, "idea_plan", None)
        if plan is None:
            raise RuntimeError("run s.explore() first")
        names = [i.name for i in plan.ideas]
        bad = [k for k in keep or [] if k not in names]
        if bad:
            raise ValueError(f"not ideas: {bad}; ideas: {names}")
        g = GateDecision(gate="G2", decided_by="human", approved=True, approved_ideas=list(keep or names), note=note)
        self._gate(g)
        display(Markdown(f"### Building and characterizing {len(g.approved_ideas)} idea(s)"))
        try:
            self.feature_set = O.build_ideas(self.run_dir, self.client, self.timed, plan, g, self._eda_ctx)
            reqs = self.feature_set.wrapup.data_investigation_requests
            if reqs:
                display(Markdown(f"### {len(reqs)} question(s) about the data routed back upstream"))
                self._show_investigations(O.investigate(self.run_dir, self.client, self.timed, reqs, "research_eda",
                                                        self.card, self.contract, self.feature_set.features))
            self.meta["status"] = "explored"
        finally:
            self._save_meta()
        self._feature_report()
        display(Markdown(f"Done: {len(self.feature_set.candidates)} candidate features, "
                         f"{len(self.feature_set.observations)} observations. Call `s.features()` to review them, then "
                         "`s.freeze_features()` (gate G3)."))
        return self.feature_set

    def _show_investigations(self, results):
        """The backward loop's answers, before the next gate."""
        for r in results:
            q = r.request
            body = (r.answer.answer + "".join(f"\n\n- caveat: {c}" for c in r.answer.caveats)
                    + f"\n\n<small>evidence: {', '.join(r.answer.evidence)}</small>") if r.answer else \
                f"*not answered:* {r.error}"
            display(Markdown(f"**Q ({q.route_to}, from {r.stage.replace('_', ' ')}):** {q.question}\n\n**A "
                             f"({r.routed_to}):** {body}"))

    def reject_ideas(self, note: str):
        self._gate(GateDecision(gate="G2", decided_by="human", approved=False, note=note))
        display(Markdown("Ideas rejected; run `s.explore(ideas=[...])` again with your own ideas."))

    def _feature_report(self):
        from quantaccelerator.agents import feature_report
        feature_report.build(self.run_dir)

    def features(self):
        """The Feature Report: the candidate set, how each feature is built, what it still depends on, the findings."""
        p = self.run_dir / "brief" / "features.html"
        if not p.exists():
            raise RuntimeError("run s.explore() first")
        display(HTML(f'<iframe srcdoc="{html.escape(p.read_text(), quote=True)}" '
                     'style="width:100%;height:900px;border:1px solid #e5e7eb;border-radius:8px"></iframe>'))

    def freeze_features(self, keep: list[str] | None = None, note: str = ""):
        """Gate G3: freeze the candidate set (or the subset `keep`) before any predictive work may see returns."""
        fs = getattr(self, "feature_set", None)
        if fs is None:
            raise RuntimeError("run s.explore() first")
        names = [c.name for c in fs.candidates]
        bad = [k for k in keep or [] if k not in names]
        if bad:
            raise ValueError(f"not candidates: {bad}; candidates: {names}")
        g = GateDecision(gate="G3", decided_by="human", approved=True, approved_features=list(keep or names), note=note)
        self.feature_set = O.freeze_features(self.run_dir, fs, g)
        self._gate(g)
        self._feature_report()
        display(Markdown(f"Frozen {len(g.approved_features)} feature(s), set hash `{self.feature_set.set_hash}`; "
                         f"values in `features/{self.feature_set.set_hash}.parquet`."))
        return self.feature_set

    def reject_features(self, note: str):
        fs = getattr(self, "feature_set", None)
        if fs is None:
            raise RuntimeError("run s.explore() first")
        g = GateDecision(gate="G3", decided_by="human", approved=False, note=note)
        self.feature_set = O.freeze_features(self.run_dir, fs, g)
        self._gate(g)

    # ------------------------------------------------------------------ Research EDA, phase B (predictive)
    def plan_predictions(self):
        """Phase B pre-registration: for each frozen feature, the expected sign, horizons and controls, written before
        any return is seen, for your signature (G4). The discovery / validation / locked-test split is shown too."""
        fs = getattr(self, "feature_set", None)
        if fs is None or not fs.set_hash:
            raise RuntimeError("freeze the feature set first (s.freeze_features(), gate G3)")
        if not active().predict:
            raise RuntimeError(f"dataset {active().name!r} declares no returns source for phase B")
        display(Markdown("### Phase B: the Predictive Planner pre-registers the tests (no returns yet)"))
        try:
            self.prediction_plan = O.plan_predictions(self.run_dir, self.client, self.timed, fs)
        finally:
            self._save_meta()
        p = self.prediction_plan
        rows = "".join(f"<tr><td><b>{html.escape(t.feature)}</b></td><td>{t.expected_sign:+d}</td><td>{t.horizons}</td>"
                       f"<td>{', '.join(t.controls) or '-'}</td><td><small>{html.escape(t.horizon_rationale)}</small>"
                       f"</td></tr>" for t in p.tests)
        sp = p.split
        display(HTML("<table style='font:13px system-ui;border-collapse:collapse'><tr><th>Feature</th><th>Sign</th>"
                     "<th>Horizons</th><th>Controls</th><th>Why</th></tr>" + rows + "</table>"
                     f"<p style='font:13px system-ui'>Split: discovery {sp.discovery[0]}..{sp.discovery[1]} · validation "
                     f"{sp.validation[0]}..{sp.validation[1]} · locked test {sp.locked_test[0]}..{sp.locked_test[1]}</p>"))
        display(Markdown("**Your decision (gate G4):** `s.approve_plan(note=...)` opens the returns vault on discovery; "
                         "`s.reject_plan(note)` keeps it closed."))
        return p

    def approve_plan(self, note: str = ""):
        """Gate G4: sign the plan and the split, then the Predictive EDA Agent tests it on discovery; the orchestrator
        re-runs every 'informative' finding on validation."""
        p = getattr(self, "prediction_plan", None)
        if p is None:
            raise RuntimeError("run s.plan_predictions() first")
        g = GateDecision(gate="G4", decided_by="human", approved=True, approved_plan=p, note=note)
        self._gate(g)
        from quantaccelerator.tools.panel import get_panel
        display(Markdown("### Phase B: testing the plan on the discovery segment"))
        try:
            self.pred_wrap, obs, self._pred_trace = O.predict(self.run_dir, self.client, self.timed, self.feature_set,
                                                              g, get_panel())
            self.research_observations = O.validate_observations(obs)
            display(Markdown("### The Critic attacks every validated finding (pre-registered, executable tests)"))
            self.research_observations, self.critiques = O.critique(self.run_dir, self.client, self.timed,
                                                                    self.feature_set, self.research_observations,
                                                                    get_panel())
            self.research_observations = O.cost_check(self.research_observations)
            if self.pred_wrap.data_investigation_requests:
                self._show_investigations(O.investigate(self.run_dir, self.client, self.timed,
                                                        self.pred_wrap.data_investigation_requests, "predictive_eda",
                                                        self.card, self.contract, self.feature_set.features))
            self.meta["status"] = "predicted"
        finally:
            self._save_meta()
        self._plan_gate = g
        self.observations()
        display(Markdown("**Your decision (gate G5):** `s.release_test(keep=[...])` sends the chosen validated "
                         "observations to the locked test (it can be opened once); `s.release_test()` sends all of them."))
        return self.research_observations

    def reject_plan(self, note: str):
        self._gate(GateDecision(gate="G4", decided_by="human", approved=False, note=note))

    def observations(self):
        """The predictive findings: verdict on discovery, then the deterministic re-test on validation."""
        obs = getattr(self, "research_observations", None)
        if obs is None:
            raise RuntimeError("run s.approve_plan() first")
        f = lambda v: "" if v is None else f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
        rows = "".join(
            f"<tr><td>{i}</td><td><b>{html.escape(o.feature)}</b></td><td>{o.horizon}</td><td>{', '.join(o.controls) or '-'}"
            f"</td><td>{o.verdict}</td><td>{f(o.stats.get('mean_ic'))} / {f(o.stats.get('t_nw'))}</td><td>"
            f"{f((o.validation or {}).get('mean_ic'))} / {f((o.validation or {}).get('t_nw'))}</td><td>"
            + (f"{o.critique['survived']}✓ {o.critique['refuted']}✗ {o.critique['inconclusive']}?" if o.critique else "")
            + f"</td><td>{o.status}</td>"
            f"<td><small>{html.escape(o.claim)}</small></td></tr>" for i, o in enumerate(obs))
        display(HTML("<table style='font:13px system-ui;border-collapse:collapse'><tr><th>#</th><th>Feature</th>"
                     "<th>h</th><th>Controls</th><th>Verdict</th><th>Discovery IC / t</th><th>Validation IC / t</th>"
                     "<th>Critic (survived/refuted/inconclusive)</th><th>Status</th><th>Claim</th></tr>" + rows + "</table>"))

    def release_test(self, keep: list[int] | None = None, note: str = ""):
        """Gate G5: run the chosen validated observations (indexes from s.observations()) on the locked test, once."""
        obs = getattr(self, "research_observations", None)
        if obs is None:
            raise RuntimeError("run s.approve_plan() first")
        ok = [i for i, o in enumerate(obs) if o.status == "validated"]
        keep = ok if keep is None else list(keep)
        bad = [i for i in keep if i not in ok]
        if bad:
            raise ValueError(f"only validated observations can be released: {ok}")
        g = GateDecision(gate="G5", decided_by="human", approved=bool(keep), released_observations=keep, note=note)
        self._gate(g)
        from quantaccelerator.state.schemas import PredictiveReport
        self.research_observations = O.locked_test(obs, g)
        study = self.feature_set.set_hash
        self.predictive_report = PredictiveReport(
            set_hash=study, plan=self._plan_gate.approved_plan, plan_gate=self._plan_gate,
            observations=self.research_observations, wrapup=self.pred_wrap,
            critiques=getattr(self, "critiques", []), release_gate=g,
            ledger=O._ledger_summary(study),
            provenance=O._prov("predictive_eda", self._pred_trace, self.client, ["state/prediction_plan.json"]))
        O._save(self.run_dir, "predictive_report", self.predictive_report)
        self.meta["status"] = "locked test done"
        self._save_meta()
        from quantaccelerator.agents import predictive_report
        predictive_report.build(self.run_dir)
        display(Markdown(f"Predictive Report: `{self.run_dir / 'brief' / 'predictive.html'}`"))
        self.observations()
        return self.predictive_report

    # ------------------------------------------------------------------ audit
    def audit(self, pipeline: str):
        """Look-Ahead Audit of a researcher pipeline against the signed contract."""
        if self.contract is None:
            raise RuntimeError("sign the PIT contract first (s.time(), then s.approve())")
        export_packages_to_children()
        pre = registry.TOOLS["run_pipeline"].fn.raw(path=pipeline)  # pre-flight, unlogged: the pipeline must run here
        display(Markdown(f"Pre-flight: `{pipeline}` runs here ({pre['n_rows']:,} rows). The Auditor takes over."))
        report = O.audit(self.run_dir, self.client, self.timed, self.contract, pipeline, self.meta)
        self._save_meta()
        if not report.findings:
            display(Markdown(f"**No look-ahead found** in `{pipeline}`. {report.summary}"))
        for f in report.findings:
            display(Markdown(
                f"**Leak at line {f.location.line}** ({f.leak_type}): {f.description}\n\n"
                f"Impact: {100 * f.estimated_impact.before:.1f}% of rows use non-public information"
                + (f" → {100 * f.verified_share_before_tradable:.1f}% after the patch (verified by re-running)"
                   if f.verified_by_run else " (patch not verified)")
                + "\n\n```python\n" + "\n".join(f"{e.line}: {e.new_code}" for e in f.proposed_patch) + "\n```"))
        return report

    def export_html(self) -> Path:
        """The Data Brief plus every gate decision, as one shareable file."""
        b = (self.run_dir / "brief" / "brief.html").read_text()
        dec = "".join(f"<li><b>{g.gate}</b> by {g.decided_by}: {'approved' if g.approved else 'rejected/corrected'}"
                      f" — {html.escape(g.note)}</li>" for g in self.gates)
        out = self.run_dir / "session.html"
        fr = self.run_dir / "brief" / "features.html"
        feat = ("<h2>Research EDA</h2><p>Feature Report: <code>brief/features.html</code> (" +
                (f"frozen, set hash {html.escape(self.feature_set.set_hash)}" if self.feature_set and self.feature_set.set_hash
                 else "not frozen") + ")</p>") if fr.exists() else ""
        out.write_text(b.replace("</body>", f"{feat}<h2>Decisions</h2><ul>{dec or '<li>none yet</li>'}</ul></body>"))
        display(Markdown(f"Saved `{out}`"))
        return out


CARD_STYLE = {"remove": ("#fde8e8", "#d03b3b", "#b42318"), "flag": ("#fff4d6", "#c98500", "#9a6700"),
              "indicator": ("#e6f0fd", "#3987e5", "#1d5fb8"), "none": ("#eaf7ee", "#199e70", "#0f6b4b")}
CARD_CSS = ("<style>.qa-ex{border-collapse:collapse;font-size:11px;margin-top:8px}.qa-ex td,.qa-ex th"
            "{padding:3px 6px;border-bottom:1px solid rgba(0,0,0,.08);color:#1f2937;text-align:left}</style>")


def _card(kind: str, title: str, text: str, rows: list, extra: str = "") -> str:
    bg, border, ink = CARD_STYLE[kind]
    body = "".join(f"<tr><td style='font-weight:600;color:{ink};padding:3px 12px 3px 0;vertical-align:top;"
                   f"white-space:nowrap'>{k}</td><td style='color:#1f2937;padding:3px 0'>{v}</td></tr>" for k, v in rows)
    return (f"<div style='background:{bg};border-left:5px solid {border};border-radius:8px;padding:10px 14px;"
            f"margin:8px 0;font:13px system-ui'><div style='font-weight:700;color:{ink};margin-bottom:6px'>{title}</div>"
            + (f"<div style='color:#1f2937'>{text}</div>" if text else "")
            + f"<table style='border-collapse:collapse'>{body}</table>{extra}</div>")


def action_text(st) -> str:
    """A cleaning step in plain words."""
    cond = " and ".join(f"{p.column} {p.op.replace('col', '')} {p.value}" if p.op not in ("isna", "notna") else
                        f"{p.column} is {'missing' if p.op == 'isna' else 'present'}" for p in st.where)
    return {"drop_duplicates": f"drop repeated rows with the same {', '.join(st.columns)} (keep the first)",
            "drop_rows": f"drop rows where {cond}",
            "flag_rows": f"add column {st.flag_name} = True where {cond}",
            "missing_indicator": f"add column {st.flag_name} = True where {', '.join(st.columns)} is missing"}[st.op]


STEP_TEXT = {
    "list_tables": lambda a: "Listing the tables in the dataset",
    "profile_table": lambda a: f"Profiling table {a.get('table')}: columns, types, nulls, distinct values",
    "profile_time": lambda a: f"Profiling the date/time columns of {a.get('table')}",
    "read_doc": lambda a: f"Reading the documentation: \u201c{a.get('query', '')}\u201d",
    "research_breadth": lambda a: f"Measuring research breadth of {a.get('table')}: how many {a.get('entity_field')}, "
                                  "how much history, how often updated",
    "coverage_over_time": lambda a: f"Checking coverage of {a.get('table')} over time: {a.get('entity_field') or 'rows'} "
                                    f"per {a.get('freq', 'M')} period, entries and exits",
    "key_check": lambda a: f"Checking whether {a.get('key_fields')} uniquely identifies rows of {a.get('table')}",
    "missingness": lambda a: f"Checking missing data in {a.get('table')}: nulls, zeros"
                             + (f" and absent {a.get('entity_field')}-dates" if a.get("entity_field") else ""),
    "distribution": lambda a: f"Examining the distribution of {a.get('field')}"
                              + (f" / {a.get('denominator')}" if a.get("denominator") else "")
                              + (" and how it drifts over time" if a.get("time_field") else ""),
    "update_dynamics": lambda a: f"Examining how {a.get('field')} changes within each {a.get('entity_field')} "
                                 "(persistence, update intervals)",
    "variance_split": lambda a: f"Splitting the variance of {a.get('field')}: across {a.get('entity_field')} vs over time",
    "structural_breaks": lambda a: f"Looking for abrupt changes in {a.get('table')}"
                                   + (f" (counts and {a.get('field')})" if a.get("field") else " (counts)"),
    "category_mix_over_time": lambda a: f"Tracking the mix of {a.get('field')} codes over time",
    "preview_cleaning": lambda a: f"Previewing a cleaning step ({(a.get('step') or {}).get('op')}) on "
                                  f"{(a.get('step') or {}).get('table')}",
    "look_at_figure": lambda a: "Looking at a figure with the vision model",
    "record_observation": lambda a: f"Noting a finding ({a.get('severity', 'info')})",
    "rate_capability": lambda a: f"Rating {str(a.get('dimension', '')).replace('_', ' ')} as {a.get('rating')}",
    "note_preprocessing": lambda a: f"Noting how to represent {a.get('field')}",
    "propose_cleaning_step": lambda a: f"Proposing a cleaning step ({a.get('op')}) on {a.get('table')}",
    "timezone_check": lambda a: "Testing which time zone the timestamps are really in",
    "next_tradable": lambda a: f"Working out the first tradable open after {a.get('timestamp')}",
    "preview_finra_rule": lambda a: "Trying the candidate availability rule on real trade dates",
    "run_pipeline": lambda a: f"Running the pipeline {a.get('path')}",
    "read_source": lambda a: f"Reading the pipeline code {a.get('path')}",
    "measure_lookahead": lambda a: "Measuring how many rows use information before it was public",
    "test_patch": lambda a: "Testing a proposed fix by re-running the pipeline",
}


def describe_call(name: str, args: dict) -> str:
    try:
        return STEP_TEXT[name](args or {})
    except (KeyError, TypeError, AttributeError):
        return name.replace("_", " ").capitalize()


def export_packages_to_children():
    """Researcher pipelines run as child processes, which inherit the environment, not this kernel's sys.path: put the
    folders of the packages this kernel actually loaded first on the children's PYTHONPATH."""
    import importlib
    dirs = []
    for mod in ("quantaccelerator", "numpy", "pandas", "pyarrow", "pydantic"):
        m = importlib.import_module(mod)
        d = str(Path(m.__file__).resolve().parents[1])
        if d not in dirs:
            dirs.append(d)
    keep = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p and p not in dirs]
    os.environ["PYTHONPATH"] = ":".join(dirs + keep)


def registry_sample(table: str, time_field: str, n: int = 5) -> pd.DataFrame:
    """A few records spread over the history (for showing a rule's effect)."""
    from quantaccelerator.tools.data import load_table
    df = load_table(table)
    idx = sorted(set(int(i) for i in pd.Series(range(len(df))).quantile([.1, .3, .5, .7, .9]).values))
    return df.sort_values(time_field).iloc[idx].reset_index(drop=True)
