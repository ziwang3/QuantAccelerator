"""Code Auditor agent -> AuditReport (findings + patches, verified by deterministic re-runs)."""
from quantaccelerator.agents.base import Agent
from quantaccelerator.datasets import active
from quantaccelerator.state.schemas import AuditReportDraft
from quantaccelerator.tools import registry

ROLE = """You are the Code Auditor of a quantitative research team. You audit a researcher's pipeline for look-ahead
leakage against the signed PIT contract (the reference for when each record became tradable).
Procedure:
1. read_source the script; run_pipeline it; measure_lookahead on its output_path.
2. If share_before_tradable is 0, the pipeline is PIT-correct: submit with no findings and cite that run.
3. Otherwise locate the line(s) responsible, classify the leak, and propose a minimal patch as line replacements
   that use only columns, variables and helper functions already available in the script. Check it with
   test_patch and iterate until the patched look-ahead is 0.
Leak types:
- event_date_as_availability: keyed on when the event happened instead of when it became public
- timezone_session_ambiguity: uses the right publication source but mishandles time of day, session or timezone
  (e.g. intraday or after-close information treated as known at an earlier open; UTC vs ET)
- same_bar_execution: trades on the same bar/price that produced the signal
- revised_vs_first_release: uses revised values instead of the first release
- backfilled_history: uses history that was backfilled after the fact
- forward_fill_period_end: values carried from period end rather than publication
- full_sample_normalization: statistics computed on the full sample (future data)
- survivorship: universe selected with knowledge of the future
- label_overlap: overlapping labels across train/test without purge/embargo
- corporate_action: split/dividend adjustment errors
- semantic_confusion: a field misread as a different quantity
In each finding, estimated_impact.before must be share_before_tradable from your measure_lookahead run on the
original output (cite its run_id) and after/after_run_id must come from your test_patch run."""

TASK = active().audit_task


def _share(run: dict | None, tool: str) -> float | None:
    if not run or run["tool"] != tool or "error" in run["result"]:
        return None
    r = run["result"]
    return (r["lookahead"] if tool == "test_patch" else r)["share_before_tradable"]


class Auditor(Agent):
    name = "auditor"
    role_prompt = ROLE
    tool_names = active().audit_tools
    output_model = AuditReportDraft
    max_steps = 20

    def extra_validate(self, draft) -> list[str]:
        errs = []
        for i, f in enumerate(draft.findings):
            imp = f.estimated_impact
            s = _share(registry.get_run(imp.before_run_id), "measure_lookahead")
            if s is None:
                errs.append(f"finding {i}: before_run_id must be a successful measure_lookahead run.")
            elif abs(s - imp.before) > 1e-4:
                errs.append(f"finding {i}: before={imp.before} but run {imp.before_run_id} measured {s}.")
            if imp.after_run_id:
                s2 = _share(registry.get_run(imp.after_run_id), "test_patch")
                if s2 is None:
                    errs.append(f"finding {i}: after_run_id must be a successful test_patch run.")
                elif imp.after is None or abs(s2 - imp.after) > 1e-4:
                    errs.append(f"finding {i}: after={imp.after} but run {imp.after_run_id} measured {s2}.")
        if not draft.findings:
            shares = [_share(registry.get_run(r), "measure_lookahead") for r in draft.evidence]
            if not any(s == 0 for s in shares if s is not None):
                errs.append("A no-leak conclusion must cite a measure_lookahead run with share_before_tradable = 0; "
                            "your evidence does not show that.")
        return errs
