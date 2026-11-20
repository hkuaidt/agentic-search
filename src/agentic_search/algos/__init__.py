"""Framework-agnostic RL math (pure PyTorch). Integrations live in :mod:`agentic_search.integrations`."""

from agentic_search.algos.advantages import (
    broadcast_step_advantage,
    combine_advantages,
    gigpo_step_advantage,
    group_normalized_advantage,
    group_reward_std_stats,
    luffy_mixed_group_advantage,
    opd_token_advantage,
)
from agentic_search.algos.losses import (
    PolicyLossOutput,
    gspo_sequence_loss,
    gspo_token_loss,
    luffy_loss,
    luffy_shaping,
    ppo_token_loss,
)
from agentic_search.algos.masked import agg_loss, masked_mean, masked_sum

__all__ = [
    "PolicyLossOutput",
    "agg_loss",
    "broadcast_step_advantage",
    "combine_advantages",
    "gigpo_step_advantage",
    "group_normalized_advantage",
    "group_reward_std_stats",
    "gspo_sequence_loss",
    "gspo_token_loss",
    "luffy_loss",
    "luffy_mixed_group_advantage",
    "luffy_shaping",
    "masked_mean",
    "masked_sum",
    "opd_token_advantage",
    "ppo_token_loss",
]
