r"""Critic-free advantage estimators at three granularities.

    \hat A_{i,t} = \hat A^E_i  +  \omega \hat A^S_{i,t}  +  \lambda_{opd} \hat A^{OPD}_t
                   (trajectory)    (step, GiGPO)            (token, OPD)

* **Trajectory level** (GRPO-style group normalisation, consumed by GSPO): every generated token of
  trajectory *i* shares ``(R_i - mean_G R) / (std_G R + eps)``.
* **Step level** (GiGPO, adapted to outcome-only rewards): steps from *different* trajectories of the
  same prompt group that share an anchor state ``s~`` (= the set of posts already held) form a step
  group ``B(s~)``. With only a terminal reward, the discounted return of step *t* is
  ``Q_{i,t} = gamma^{T_i - t} R_i`` and the advantage is the within-group normalised ``Q``.
  Different queries issued from the same evidence state are compared; a query that reaches the answer
  in fewer remaining steps gets a larger return. Singleton groups carry no signal (advantage 0).
* **Token level** (OPD, sampled-token reverse KL): ``log pi_T(y_t|c_t) - log pi_old(y_t|c_t)``.
  Requires teacher and student to share a tokenizer (true for the Qwen3 family).

Notation: ``gamma`` is the GiGPO discount and ``lambda_opd`` the OPD weight (kept distinct on purpose).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import torch


def _group_indices(group_ids: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)
    return groups


def group_normalized_advantage(
    rewards: torch.Tensor | Sequence[float],
    group_ids: Sequence[Hashable],
    eps: float = 1e-6,
    norm_by_std: bool = True,
    unbiased_std: bool = False,
) -> torch.Tensor:
    """Trajectory-level advantage ``A^E`` (GRPO / GSPO).

    Args:
        rewards: ``[B]`` scalar outcome reward per trajectory.
        group_ids: length-``B`` prompt ids; trajectories sampled for the same prompt share an id.
        norm_by_std: divide by the group std (GRPO). ``False`` gives Dr.GRPO-style mean-centering.
        unbiased_std: PROPOSAL §4.3(2) uses the population std (``1/G``); VERL uses ``torch.std``
            (unbiased). Exposed so either convention can be matched exactly.

    Groups of size 1 get advantage 0 (no baseline is available).
    """
    r = torch.as_tensor(rewards, dtype=torch.float32)
    adv = torch.zeros_like(r)
    for idx in _group_indices(group_ids).values():
        if len(idx) < 2:
            continue
        g = r[idx]
        centered = g - g.mean()
        if norm_by_std:
            centered = centered / (g.std(unbiased=unbiased_std) + eps)
        adv[idx] = centered
    return adv


@dataclass(frozen=True)
class StepRef:
    """Pointer to one step: trajectory ``traj``, 1-based step position ``t`` out of ``T`` steps."""

    traj: int
    t: int


def gigpo_step_advantage(
    rewards: Sequence[float],
    group_ids: Sequence[Hashable],
    anchor_keys: Sequence[Sequence[Hashable]],
    gamma: float = 0.9,
    eps: float = 1e-6,
    norm_by_std: bool = True,
    unbiased_std: bool = False,
) -> list[np.ndarray]:
    """Step-level GiGPO advantage for outcome-only rewards.

    Args:
        rewards: ``R_i`` per trajectory.
        group_ids: prompt id per trajectory (step groups never cross prompts).
        anchor_keys: per trajectory, one hashable state key per step (``Step.anchor_key()``).
            ``len(anchor_keys[i]) == T_i``.
        gamma: discount in ``(0, 1]``. ``Q_{i,t} = gamma^(T_i - t) * R_i`` with 1-based ``t``.

    Returns:
        A list with one ``float32`` array of shape ``[T_i]`` per trajectory.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if len(rewards) != len(group_ids) or len(rewards) != len(anchor_keys):
        raise ValueError("rewards, group_ids and anchor_keys must have the same length")

    q_values: list[np.ndarray] = []
    buckets: dict[tuple[Hashable, Hashable], list[StepRef]] = defaultdict(list)
    for i, (R, gid, keys) in enumerate(zip(rewards, group_ids, anchor_keys)):
        T = len(keys)
        q_values.append(np.array([gamma ** (T - t) * float(R) for t in range(1, T + 1)], dtype=np.float64))
        for t, key in enumerate(keys, start=1):
            buckets[(gid, key)].append(StepRef(i, t))

    adv = [np.zeros(len(k), dtype=np.float32) for k in anchor_keys]
    for refs in buckets.values():
        if len(refs) < 2:
            continue
        q = np.array([q_values[r.traj][r.t - 1] for r in refs])
        centered = q - q.mean()
        if norm_by_std:
            centered = centered / (q.std(ddof=1 if unbiased_std else 0) + eps)
        for r, a in zip(refs, centered):
            adv[r.traj][r.t - 1] = a
    return adv


def broadcast_step_advantage(step_adv: Sequence[np.ndarray], token_step_ids: torch.Tensor) -> torch.Tensor:
    """Scatter per-step advantages onto tokens.

    Args:
        step_adv: output of :func:`gigpo_step_advantage`.
        token_step_ids: ``[B, L]`` int tensor; the 0-based step a generated token belongs to, ``-1`` for
            tokens the model did not generate (system/user/tool results/padding).
    """
    out = torch.zeros(token_step_ids.shape, dtype=torch.float32)
    for i, adv_i in enumerate(step_adv):
        ids = token_step_ids[i]
        valid = ids >= 0
        if valid.any():
            table = torch.as_tensor(adv_i, dtype=torch.float32)
            if int(ids[valid].max()) >= len(table):
                raise IndexError(f"trajectory {i}: token step id exceeds number of steps ({len(table)})")
            out[i, valid] = table[ids[valid].long()]
    return out


def opd_token_advantage(
    teacher_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    mask: torch.Tensor,
    clip: float | None = None,
) -> torch.Tensor:
    """Sampled-token reverse-KL advantage: ``log pi_T(y_t) - log pi_old(y_t)`` on generated tokens.

    ``clip`` bounds the per-token signal; teacher/student disagreement on rare tokens can otherwise
    dominate the combined advantage early in training.
    """
    adv = (teacher_logprobs - old_logprobs).float()
    if clip is not None:
        adv = adv.clamp(-clip, clip)
    return adv * mask.to(adv.dtype)


def combine_advantages(
    traj_adv: torch.Tensor,
    mask: torch.Tensor,
    step_adv_tokens: torch.Tensor | None = None,
    opd_adv_tokens: torch.Tensor | None = None,
    omega: float = 1.0,
    lambda_opd: float = 0.0,
) -> torch.Tensor:
    """``A_{i,t} = A^E_i + omega * A^S_{i,t} + lambda_opd * A^OPD_t`` on generated tokens, 0 elsewhere."""
    if omega < 0 or lambda_opd < 0:
        raise ValueError("omega and lambda_opd must be non-negative")
    m = mask.to(torch.float32)
    total = traj_adv.float().unsqueeze(-1).expand_as(m).clone()
    if step_adv_tokens is not None and omega:
        total = total + omega * step_adv_tokens.float()
    if opd_adv_tokens is not None and lambda_opd:
        total = total + lambda_opd * opd_adv_tokens.float()
    return total * m


def luffy_mixed_group_advantage(
    on_policy_rewards: Sequence[float],
    off_policy_rewards: Sequence[float],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LUFFY: normalise rewards over the *mixed* group (student rollouts + teacher demonstrations).

    When all student rollouts score 0, a full-score teacher demonstration still yields non-zero
    advantages for both parts, so the group keeps a gradient (the all-zero-reward problem LUFFY fixes).
    """
    r = torch.as_tensor([*on_policy_rewards, *off_policy_rewards], dtype=torch.float32)
    std = r.std(unbiased=False)
    adv = (r - r.mean()) / (std + eps) if r.numel() > 1 else torch.zeros_like(r)
    n_on = len(on_policy_rewards)
    return adv[:n_on], adv[n_on:]


def group_reward_std_stats(rewards: Sequence[float], group_ids: Sequence[Hashable]) -> dict[str, float]:
    """Monitoring helper from the RL debug protocol: share of groups with zero reward variance."""
    groups = _group_indices(group_ids)
    r = np.asarray(rewards, dtype=np.float64)
    stds = [r[idx].std() for idx in groups.values()]
    zero = [s < 1e-8 for s in stds]
    all_zero = [bool(z and np.all(r[idx] == 0)) for z, idx in zip(zero, groups.values())]
    return {
        "num_groups": float(len(groups)),
        "frac_zero_std_groups": float(np.mean(zero)) if zero else 0.0,
        "frac_all_zero_groups": float(np.mean(all_zero)) if all_zero else 0.0,
        "mean_group_std": float(np.mean(stds)) if stds else 0.0,
    }
