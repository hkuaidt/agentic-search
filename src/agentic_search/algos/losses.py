r"""Policy-gradient losses: GSPO(-token), PPO/GRPO token-clip, and LUFFY policy shaping.

GSPO with per-token advantages
------------------------------
Plain GSPO clips a *sequence* ratio ``s_i = exp(mean_t log(pi/pi_old))`` against a *sequence*
advantage. Our total advantage varies per step (GiGPO) and per token (OPD), so we use the
**GSPO-token** objective from the GSPO paper (Zheng et al., 2025, Eq. 13-14)::

    s_{i,t}(theta) = sg[s_i(theta)] * pi_theta(y_t) / sg[pi_theta(y_t)]

Numerically ``s_{i,t} == s_i`` (the MoE-robust sequence ratio is what gets clipped), but the gradient
of token *t* is ``s_i * A_{i,t} * grad log pi(y_t)``. When ``A_{i,t} == A_i`` this reduces exactly to
GSPO. This is also how VERL's ``loss_mode=gspo`` is implemented.

Clip range: the sequence ratio is a geometric mean, so it stays within ~1e-3 of 1. The GSPO paper and
VERL's Qwen3-30B-A3B example use ``clip_low=3e-4, clip_high=4e-4``; the token-level PPO range
``(0.2, 0.2)`` (the familiar "0.8 / 1.2") would essentially never clip a sequence ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agentic_search.algos.masked import LossAggMode, agg_loss, masked_mean


@dataclass
class PolicyLossOutput:
    loss: torch.Tensor
    clip_frac: torch.Tensor
    approx_kl: torch.Tensor
    extra: dict[str, torch.Tensor]


def gspo_token_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
    loss_agg_mode: LossAggMode = "seq-mean-token-mean",
    max_log_ratio: float = 10.0,
) -> PolicyLossOutput:
    """GSPO-token loss. All tensors are ``[B, L]``; ``mask`` marks LLM-generated tokens only."""
    m = mask.to(log_probs.dtype)
    log_ratio = log_probs - old_log_probs
    seq_len = m.sum(-1).clamp_min(1.0)
    seq_log_ratio = (log_ratio * m).sum(-1) / seq_len  # log s_i
    log_s_tok = log_probs - log_probs.detach() + seq_log_ratio.detach().unsqueeze(-1)
    s_tok = torch.exp(log_s_tok.clamp(max=max_log_ratio))

    pg1 = -advantages * s_tok
    pg2 = -advantages * torch.clamp(s_tok, 1.0 - clip_low, 1.0 + clip_high)
    pg = torch.maximum(pg1, pg2)
    loss = agg_loss(pg, m, loss_agg_mode)

    clipped = (pg2 > pg1).to(m.dtype)
    return PolicyLossOutput(
        loss=loss,
        clip_frac=masked_mean(clipped, m).detach(),
        approx_kl=masked_mean(-log_ratio, m).detach(),
        extra={"seq_ratio_mean": torch.exp(seq_log_ratio).mean().detach()},
    )


def gspo_sequence_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    seq_advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
) -> torch.Tensor:
    """Reference GSPO (sequence-level advantage). Used in tests to check GSPO-token equivalence."""
    m = mask.to(log_probs.dtype)
    seq_log_ratio = ((log_probs - old_log_probs) * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    s = torch.exp(seq_log_ratio)
    obj = torch.minimum(s * seq_advantages, torch.clamp(s, 1 - clip_low, 1 + clip_high) * seq_advantages)
    return -obj.mean()


def ppo_token_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    dual_clip_c: float | None = 3.0,
    loss_agg_mode: LossAggMode = "token-mean",
) -> PolicyLossOutput:
    """Token-level clipped surrogate (PPO / GRPO), optional dual-clip for negative advantages."""
    m = mask.to(log_probs.dtype)
    log_ratio = (log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    pg = torch.maximum(pg1, pg2)
    if dual_clip_c is not None:
        pg3 = -advantages * dual_clip_c
        pg = torch.where(advantages < 0, torch.minimum(pg, pg3), pg)
    clipped = (pg2 > pg1).to(m.dtype)
    return PolicyLossOutput(
        loss=agg_loss(pg, m, loss_agg_mode),
        clip_frac=masked_mean(clipped, m).detach(),
        approx_kl=masked_mean(-log_ratio, m).detach(),
        extra={},
    )


def luffy_shaping(p: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """LUFFY policy shaping ``f(p) = p / (p + gamma)``.

    Its gradient ``gamma / (p + gamma)^2`` up-weights low-probability teacher tokens, so imitation does
    not collapse onto tokens the student already predicts well (entropy is preserved).
    """
    return p / (p + gamma)


def luffy_loss(
    on_log_probs: torch.Tensor,
    on_old_log_probs: torch.Tensor,
    on_advantages: torch.Tensor,
    on_mask: torch.Tensor,
    off_log_probs: torch.Tensor,
    off_advantages: torch.Tensor,
    off_mask: torch.Tensor,
    shaping_gamma: float = 0.1,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> PolicyLossOutput:
    r"""LUFFY objective (Yan et al., 2025), as in PROPOSAL §4.1(1)::

        J = 1/Z [ sum_{j,t} f(\hat r_{j,t}) \hat A_j
                + sum_{i,t} min(r_{i,t} \hat A_i, clip(r_{i,t}) \hat A_i) ]

    * off-policy part: teacher demonstrations, ``\hat r = pi_theta(teacher token | ctx)`` (no ratio
      to a behaviour policy, no clipping), shaped by ``f``;
    * on-policy part: the usual clipped surrogate over student rollouts;
    * ``Z``: total number of valid tokens over both parts.

    Advantages are expected to come from :func:`~agentic_search.algos.advantages.luffy_mixed_group_advantage`.
    """
    on_m = on_mask.to(on_log_probs.dtype)
    off_m = off_mask.to(off_log_probs.dtype)
    ratio = torch.exp((on_log_probs - on_old_log_probs).clamp(-20.0, 20.0))
    on_obj = torch.minimum(
        ratio * on_advantages, torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * on_advantages
    )
    off_obj = luffy_shaping(torch.exp(off_log_probs), shaping_gamma) * off_advantages
    Z = (on_m.sum() + off_m.sum()).clamp_min(1.0)
    loss = -((on_obj * on_m).sum() + (off_obj * off_m).sum()) / Z
    clipped = ((ratio < 1 - clip_low) | (ratio > 1 + clip_high)).to(on_m.dtype)
    return PolicyLossOutput(
        loss=loss,
        clip_frac=masked_mean(clipped, on_m).detach(),
        approx_kl=masked_mean(on_old_log_probs - on_log_probs, on_m).detach(),
        extra={"off_policy_prob_mean": masked_mean(torch.exp(off_log_probs), off_m).detach()},
    )
