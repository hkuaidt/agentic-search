"""Hybrid outcome reward settled once per finished rollout.

    R = w_format * format + w_query * query_relevance + w_outcome * outcome      (default 0.1 / 0.3 / 0.6)

10/30/60 is the default so the training reward and the RaR evaluation rubric (F10/Q30/A60) share one
scale; any split summing to 1 can be configured.

Rollouts that hit a search-infrastructure failure are flagged ``valid=False`` so the trainer can drop
them instead of punishing the policy for a backend outage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agentic_search.core.types import Trajectory
from agentic_search.rewards.format_reward import compute_format_reward
from agentic_search.rewards.judge import Judge
from agentic_search.rewards.outcome import compute_outcome_reward
from agentic_search.rewards.query_relevance import compute_query_reward
from agentic_search.tools.base import ToolRegistry


@dataclass
class RewardWeights:
    format: float = 0.1
    query: float = 0.3
    outcome: float = 0.6

    def __post_init__(self) -> None:
        if abs(self.format + self.query + self.outcome - 1.0) > 1e-6:
            raise ValueError("reward weights must sum to 1")


@dataclass
class AgenticSearchReward:
    registry: ToolRegistry
    query_judge: Judge
    outcome_judges: Mapping[str, Judge]
    weights: RewardWeights = field(default_factory=RewardWeights)
    outcome_dim_weights: Mapping[str, float] | None = None
    search_required: bool = True

    def __call__(self, traj: Trajectory) -> dict[str, float]:
        parts: dict[str, float] = {}
        parts.update(compute_format_reward(traj, self.registry))
        parts.update(compute_query_reward(traj, self.query_judge, self.search_required))
        parts.update(compute_outcome_reward(traj, self.outcome_judges, self.outcome_dim_weights))
        w = self.weights
        parts["score"] = w.format * parts["format"] + w.query * parts["query"] + w.outcome * parts["outcome"]
        parts["valid"] = 0.0 if traj.had_infra_failure() else 1.0
        parts["num_tool_rounds"] = float(traj.num_tool_rounds)
        return parts
