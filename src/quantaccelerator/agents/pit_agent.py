"""Temporal / PIT agent -> PITContract."""
from quantaccelerator.agents.base import Agent
from quantaccelerator.agents.checks import check_pit_contract
from quantaccelerator.datasets import active
from quantaccelerator.state.schemas import PITContractDraft

ROLE = """You are the Temporal / Point-in-Time (PIT) agent of a quantitative research team. For a dataset you
separate, per time field, event time (when something happened), publication time (when it became public),
reporting periods and amendment references. You then decide the availability rule: which field marks when each
record became public, the timezone in which that field's values are actually expressed (verify empirically;
do not trust labels or suffixes), and the first session at which the record could be traded under the team's
execution convention. Consider alternatives and state the evidence for your choice. Flag revision/amendment and
universe (survivorship, identifier) issues."""

TASK = active().pit_task


class PITAgent(Agent):
    name = "pit_agent"
    role_prompt = ROLE
    tool_names = active().pit_tools
    output_model = PITContractDraft
    max_submit_failures = 3

    def extra_validate(self, draft) -> list[str]:
        return check_pit_contract(draft)

