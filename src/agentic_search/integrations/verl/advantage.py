"""``gspo_gigpo_opd`` advantage estimator for VERL.

The tensor function :func:`compute_gspo_gigpo_opd_advantage` is pure PyTorch (tested on CPU). The
hook :func:`install_verl_hooks` wires it into VERL's driver-side ``compute_advantage`` because a
registered estimator only receives ``token_level_rewards / response_mask / index`` -- GiGPO additionally
needs per-token step ids and per-step anchor keys, and OPD needs teacher log-probs.

Batch contract (written by :class:`~agentic_search.integrations.verl.agent_loop.AgenticSearchAgentLoop`):

==============================  =========  ===================================================
key                             shape      meaning
==============================  =========  ===================================================
``batch.token_level_rewards``   [B, L]     outcome reward on the last generated token
``batch.response_mask``         [B, L]     1 on policy-sampled tokens, 0 on tool results/padding
``batch.step_ids``              [B, L]     0-based step of each sampled token, -1 elsewhere
``batch.old_log_probs``         [B, L]     behaviour-policy log-probs (needed for OPD)
``batch.teacher_log_probs``     [B, L]     teacher log-probs of the sampled tokens (OPD)
``non_tensor_batch.uid``        [B]        prompt id (rollouts of one prompt share it)
``non_tensor_batch.anchor_keys`` [B]       list[str] per sample: anchor-state key of every step
==============================  =========  ===================================================

Pair with ``actor.policy_loss.loss_mode=gspo`` (VERL's GSPO is the GSPO-token form, so per-token
advantages are handled correctly), ``loss_agg_mode=seq-mean-token-mean`` and ``clip_ratio_low=3e-4``,
``clip_ratio_high=4e-4``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from agentic_search.algos.advantages import (
    broadcast_step_advantage,
    combine_advantages,
    gigpo_step_advantage,
    group_normalized_advantage,
    opd_token_advantage,
)

ESTIMATOR_NAME = "gspo_gigpo_opd"


@dataclass
class AdvantageConfig:
    gamma: float = 0.9  # GiGPO step discount
    omega: float = 1.0  # step-level weight
    lambda_opd: float = 0.5  # token-level (OPD) weight; 0 disables OPD
    opd_clip: float | None = 5.0
    eps: float = 1e-6
    norm_by_std: bool = True


def compute_gspo_gigpo_opd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Any],
    step_ids: torch.Tensor | None = None,
    anchor_keys: Sequence[Sequence[str]] | None = None,
    old_log_probs: torch.Tensor | None = None,
    teacher_log_probs: torch.Tensor | None = None,
    cfg: AdvantageConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Returns ``(advantages [B, L], returns [B, L], metrics)``."""
    cfg = cfg or AdvantageConfig()
    mask = response_mask.float()
    rewards = (token_level_rewards * mask).sum(-1)
    gid = [str(i) for i in index]
    traj_adv = group_normalized_advantage(rewards, gid, cfg.eps, cfg.norm_by_std)

    step_tok = None
    metrics: dict[str, float] = {}
    if cfg.omega and step_ids is not None and anchor_keys is not None:
        step_adv = gigpo_step_advantage(
            rewards.tolist(), gid, anchor_keys, cfg.gamma, cfg.eps, cfg.norm_by_std
        )
        step_tok = broadcast_step_advantage(step_adv, step_ids.cpu()).to(mask.device)
        nonzero = np.concatenate([a for a in step_adv if len(a)]) if step_adv else np.zeros(0)
        metrics["gigpo/frac_steps_with_signal"] = float((nonzero != 0).mean()) if nonzero.size else 0.0

    opd_tok = None
    if cfg.lambda_opd and teacher_log_probs is not None and old_log_probs is not None:
        opd_tok = opd_token_advantage(teacher_log_probs, old_log_probs, mask, cfg.opd_clip)
        metrics["opd/mean_adv"] = float((opd_tok * mask).sum() / mask.sum().clamp_min(1))

    adv = combine_advantages(traj_adv.to(mask.device), mask, step_tok, opd_tok, cfg.omega, cfg.lambda_opd)
    metrics["adv/traj_abs_mean"] = float(traj_adv.abs().mean())
    return adv, adv.clone(), metrics


def install_verl_hooks(
    cfg: AdvantageConfig, teacher_scorer: Callable[[Any], torch.Tensor] | None = None
) -> None:
    """Patch ``verl.trainer.ppo.ray_trainer.compute_advantage`` to dispatch ``gspo_gigpo_opd``.

    ``teacher_scorer(data) -> [B, L]`` fills ``teacher_log_probs`` when absent (see :mod:`.teacher`).
    Written against the VERL 0.6-era driver API; verify the patched symbol on upgrade.
    """
    from verl.trainer.ppo import ray_trainer

    original = ray_trainer.compute_advantage
    if getattr(original, "_agentic_search_patched", False):
        return

    def compute_advantage(data, adv_estimator, *args, **kwargs):
        name = getattr(adv_estimator, "value", adv_estimator)
        if name != ESTIMATOR_NAME:
            return original(data, adv_estimator, *args, **kwargs)
        batch, nt = data.batch, data.non_tensor_batch
        if teacher_scorer is not None and cfg.lambda_opd and "teacher_log_probs" not in batch.keys():  # noqa: SIM118 (TensorDict has no __contains__)
            batch["teacher_log_probs"] = teacher_scorer(data)
        adv, ret, metrics = compute_gspo_gigpo_opd_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            nt["uid"],
            batch.get("step_ids", None),
            list(nt["anchor_keys"]) if "anchor_keys" in nt else None,
            batch.get("old_log_probs", None),
            batch.get("teacher_log_probs", None),
            cfg,
        )
        batch["advantages"], batch["returns"] = adv, ret
        data.meta_info.setdefault("metrics", {}).update(metrics)
        return data

    compute_advantage._agentic_search_patched = True  # type: ignore[attr-defined]
    ray_trainer.compute_advantage = compute_advantage
