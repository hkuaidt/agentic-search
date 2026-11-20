"""File passed to VERL as ``custom_reward_function.path``.

VERL imports this module inside the trainer process (a Ray actor), which makes it the reliable place to
install the ``gspo_gigpo_opd`` advantage hook there. Rewards themselves are computed in the agent loop
(they need the structured trajectory, not just decoded text) and are returned via ``reward_score``;
``compute_score`` only forwards that value for code paths that call the custom reward function.
"""

from __future__ import annotations

import os
from typing import Any

from agentic_search.config import load_config
from agentic_search.integrations.verl.advantage import AdvantageConfig, install_verl_hooks
from agentic_search.integrations.verl.teacher import TeacherLogprobClient


def _make_teacher_scorer(cfg: dict[str, Any]):
    t = cfg.get("teacher")
    if not t:
        return None
    client = TeacherLogprobClient(t["base_url"], t["model"], max_workers=t.get("max_workers", 32))

    def scorer(data: Any):
        prompts, responses = data.batch["prompts"], data.batch["responses"]
        attn = data.batch["attention_mask"]
        p_len = prompts.shape[-1]
        p_ids = [row[attn[i, :p_len].bool()].tolist() for i, row in enumerate(prompts)]
        r_ids = [row[attn[i, p_len:].bool()].tolist() for i, row in enumerate(responses)]
        return client.score_padded(p_ids, r_ids, responses.shape[-1])

    return scorer


if os.environ.get("AGENTIC_SEARCH_CONFIG"):
    _cfg = load_config(os.environ["AGENTIC_SEARCH_CONFIG"])
    install_verl_hooks(AdvantageConfig(**_cfg.get("advantage", {})), _make_teacher_scorer(_cfg))


def compute_score(
    data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None
) -> float:
    info = extra_info or {}
    return float(info.get("reward_score", info.get("reward_breakdown", {}).get("score", 0.0)))
