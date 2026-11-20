"""Format reward (10% of the outcome reward): tool-call validity (60%) + multi-hop attribution (40%).

Both parts are *means* over the trajectory so issuing more calls or more citations cannot raise the
reward (anti reward hacking).

Why reward "根据第 x 步"? In a decoder-only model the anchor tokens raise attention to step x's
``[Tool Result]`` for everything generated after them, so later rounds actually build on earlier
evidence instead of searching independently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from agentic_search.core.types import Trajectory
from agentic_search.tools.base import ToolRegistry

HOP_PATTERN = re.compile(r"根据第\s*(\d+)\s*步")


def check_tool_format(tool_call: Any, registry: ToolRegistry) -> float:
    """1.0 iff the call parses to ``{"name": str, "arguments": dict}`` for a registered tool whose
    arguments satisfy its JSON schema and extra parameter checks."""
    if isinstance(tool_call, str):
        try:
            tool_call = json.loads(tool_call)
        except ValueError:
            return 0.0
    if not isinstance(tool_call, dict) or set(tool_call) != {"name", "arguments"}:
        return 0.0
    name, arguments = tool_call["name"], tool_call["arguments"]
    if not isinstance(name, str) or not isinstance(arguments, dict) or name not in registry:
        return 0.0
    return 1.0 if registry.validate_arguments(name, arguments) else 0.0


def check_multi_hop_attribution(current_step_idx: int, current_thought: str, num_steps: int) -> float:
    """1.0 iff the thought cites at least one *earlier existing* step ("第 x 步" is 1-based)."""
    matches = HOP_PATTERN.findall(current_thought or "")
    if not matches:
        return 0.0
    for ref in {int(m) - 1 for m in matches}:
        if not (0 <= ref < current_step_idx and ref < num_steps):
            return 0.0
    return 1.0


def default_hop_steps(traj: Trajectory) -> list[int]:
    """Steps that must cite earlier observations: vertical follow-up tool rounds (any tool step after
    the first). A task-level rule; the agent cannot opt out of it."""
    tool_steps = [s.index for s in traj.steps if s.is_tool_step]
    return tool_steps[1:]


def compute_format_reward(
    traj: Trajectory,
    registry: ToolRegistry,
    hop_step_indices: Sequence[int] | None = None,
    tool_weight: float = 0.6,
    hop_weight: float = 0.4,
) -> dict[str, float]:
    tool_scores: list[float] = []
    for step in traj.steps:
        if step.action_parse_error:
            tool_scores.append(0.0)
        tool_scores.extend(
            check_tool_format(
                c.raw if c.raw is not None else {"name": c.name, "arguments": c.arguments}, registry
            )
            for c in step.tool_calls
        )
    hops = default_hop_steps(traj) if hop_step_indices is None else list(hop_step_indices)
    hop_scores = [
        check_multi_hop_attribution(i, traj.steps[i].thought, len(traj.steps))
        for i in hops
        if i < len(traj.steps)
    ]
    tool_r = sum(tool_scores) / len(tool_scores) if tool_scores else 1.0
    hop_r = sum(hop_scores) / len(hop_scores) if hop_scores else 1.0
    return {"format": tool_weight * tool_r + hop_weight * hop_r, "format/tool": tool_r, "format/hop": hop_r}
