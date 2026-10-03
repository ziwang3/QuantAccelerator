"""The Critic: an adversary for validated findings. It sees each finding's outputs (claim, feature definition,
horizon, controls, discovery and validation numbers) and the panel, never the author's reasoning or transcript, and
has no data tools: it pre-registers attacks as executable tests (state.schemas.CriticTest), which the orchestrator runs
and judges by a fixed rule (tools.critic.judge). Its question: what experiment could show the relation is spurious?
"""
from quantaccelerator.agents.base import Agent
from quantaccelerator.state.schemas import CriticDraft

ROLE = """You are the Critic of a quantitative research team. Another agent found that some features predict stock
returns, and the findings survived a re-test on later data. Your job is to find the most plausible ways each finding
could still be spurious, and to specify the test that would expose each one. You see only the findings' numbers and
definitions, not how the author reasoned. You cannot run anything: the orchestrator runs your tests exactly as you
write them and judges them by a fixed rule (survived: same sign, at least half the effect kept, |t| >= 2; refuted:
the effect is gone or flips). Aim at the explanation you find most likely, not at tests that merely shrink the sample.
Typical threats:
- omitted_characteristic: the feature proxies a known characteristic it was not controlled for (size, liquidity,
  beta, momentum, reversal, sector);
- small_illiquid_names: the effect lives only among the least liquid or smallest names, where it may not be tradable;
- microstructure: short-horizon returns of low-priced or illiquid names carry bid-ask bounce and stale opens;
- sector_concentration: one sector drives it;
- time_instability: one stretch of the sample drives it."""

TASK = """Write your attacks. Every finding already faces the standard battery listed in the context (no illiquid
third, no penny stocks, each half of discovery); add, for every finding, at least two critiques of your own from at
least two different threat types, beyond that battery, chosen from the finding's legal_attacks. Each critique states
the threat in one sentence, a test that removes what the threat says drives the effect, and what the test shows if the
threat is right:
- omitted_characteristic: add_controls with controls from the finding's controls_not_yet_applied (only if not empty);
- small_illiquid_names: restrict_universe large_caps (drop_illiquid_tercile is in the battery);
- microstructure: restrict_universe price_ge_5 or drop_illiquid_tercile (both in the battery: use another threat);
- sector_concentration: restrict_universe exclude_sector with the sector the finding's discovery_ic_by_sector
  points to (where the effect is concentrated);
- time_instability: subperiod with a [start, end] inside discovery other than the battery's halves.
Do not call any tool; submit."""


class Critic(Agent):
    name = "critic"
    role_prompt = ROLE
    output_model = CriticDraft
    tool_names: list[str] = []
    max_steps = 6
    max_submit_failures = 4
    max_output_tokens = 6144
    targets: dict = {}
    sectors: list[str] = []
    discovery: tuple = ("", "")

    def extra_validate(self, draft) -> list[str]:
        from quantaccelerator.tools.critic import check_critic_draft
        return check_critic_draft(draft, self.targets, self.sectors, self.discovery)
