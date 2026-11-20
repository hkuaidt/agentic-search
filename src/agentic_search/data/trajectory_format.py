"""Serialise trajectories for SFT with the course's loss mask.

* ``[System Prompt]`` / ``[User Input]``: built offline        -> mask 0
* ``[Thinking]`` / ``[Input Decomposition]`` / ``[Action]`` / ``[Response]``: generated -> mask 1
* ``[Tool Result]``: produced by the runtime                    -> mask 0

ms-swift computes loss on every assistant turn with ``loss_scale=default``, so mapping generated turns
to ``assistant`` and tool results to the (never-trained) tool-result role reproduces the mask exactly.
For token-level pipelines (VERL agent loop), :func:`tokenize_trajectory` returns ids, loss mask and the
step id of every generated token (used to scatter GiGPO step advantages).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from agentic_search.core.types import Trajectory
from agentic_search.runtime.protocol import TOOL_RESULT_ROLE, render_messages


def to_chat_sample(traj: Trajectory, system_prompt: str | None = None) -> dict:
    """``{"messages": [...]}`` in the format accepted by ms-swift / most SFT trainers.

    Pass ``system_prompt`` to replace a hint-augmented generation prompt with the clean one, so
    privileged hints used during distillation never leak into training inputs.
    """
    msgs = render_messages(
        system_prompt if system_prompt is not None else traj.system_prompt, traj.query, traj.steps
    )
    return {"messages": msgs}


def export_sft_jsonl(trajs: Iterable[Trajectory], path: str | Path, drop_truncated: bool = True) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for t in trajs:
            if drop_truncated and (t.meta.get("truncated") or not t.final_response or t.had_infra_failure()):
                continue
            clean = t.meta.get("clean_system_prompt")
            f.write(json.dumps(to_chat_sample(t, clean), ensure_ascii=False) + "\n")
            n += 1
    return n


@dataclass
class TokenizedTrajectory:
    input_ids: list[int]
    loss_mask: list[int]
    step_ids: list[int]  # -1 for non-generated tokens

    def __post_init__(self) -> None:
        assert len(self.input_ids) == len(self.loss_mask) == len(self.step_ids)


def tokenize_trajectory(
    traj: Trajectory, encode_message: Callable[[str, str], list[int]]
) -> TokenizedTrajectory:
    """``encode_message(role, content)`` must return the ids of one rendered chat message (including its
    template tokens). Template tokens of assistant turns are trained like the content (standard)."""
    ids: list[int] = []
    mask: list[int] = []
    step_ids: list[int] = []
    msgs = render_messages(traj.system_prompt, traj.query, traj.steps)
    step_iter = iter(traj.steps)
    for m in msgs:
        toks = encode_message(m["role"], m["content"])
        if m["role"] == "assistant":
            s = next(step_iter)
            ids += toks
            mask += [1] * len(toks)
            step_ids += [s.index] * len(toks)
        else:
            assert m["role"] in ("system", "user", TOOL_RESULT_ROLE)
            ids += toks
            mask += [0] * len(toks)
            step_ids += [-1] * len(toks)
    return TokenizedTrajectory(ids, mask, step_ids)
