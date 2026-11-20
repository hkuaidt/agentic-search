"""Masked reductions and loss aggregation (VERL-compatible semantics)."""

from __future__ import annotations

from typing import Literal

import torch

LossAggMode = Literal["token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"]


def masked_sum(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() if dim is None else (x * m).sum(dim=dim)


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, dim: int | None = None, eps: float = 1e-8
) -> torch.Tensor:
    m = mask.to(x.dtype)
    if dim is None:
        return (x * m).sum() / m.sum().clamp_min(eps)
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(eps)


def agg_loss(
    loss_mat: torch.Tensor,
    mask: torch.Tensor,
    mode: LossAggMode = "token-mean",
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Aggregate a ``[B, L]`` per-token loss.

    * ``token-mean``: mean over every valid token in the batch (DAPO / GRPO default).
    * ``seq-mean-token-mean``: mean within each sequence, then over sequences (GSPO paper).
    * ``seq-mean-token-sum``: sum within each sequence, mean over sequences.
    * ``seq-mean-token-sum-norm``: sum over all tokens / (B * max_tokens) (Dr.GRPO style).
    """
    m = mask.to(loss_mat.dtype)
    if mode == "token-mean":
        return masked_mean(loss_mat, m)
    seq_valid = m.sum(dim=-1) > 0
    if mode == "seq-mean-token-mean":
        per_seq = (loss_mat * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        return per_seq[seq_valid].mean() if seq_valid.any() else loss_mat.sum() * 0.0
    if mode == "seq-mean-token-sum":
        per_seq = (loss_mat * m).sum(-1)
        return per_seq[seq_valid].mean() if seq_valid.any() else loss_mat.sum() * 0.0
    if mode == "seq-mean-token-sum-norm":
        denom = loss_mat.shape[0] * (max_tokens or loss_mat.shape[-1])
        return (loss_mat * m).sum() / denom
    raise ValueError(f"unknown loss_agg_mode: {mode}")
