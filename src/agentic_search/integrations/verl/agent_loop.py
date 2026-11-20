"""Token-level agent loop for VERL rollouts (reference adapter).

Maps the text runtime onto VERL's async agent-loop API so RL trains on exactly the tokens the policy
sampled (no re-tokenisation drift):

* assistant turns: ids returned by the rollout server        -> ``response_mask = 1``, ``step_id = t``
* tool results / chat-template glue                           -> ``response_mask = 0``, ``step_id = -1``
* per step, the anchor key of the held-post set (GiGPO), and the hybrid reward of the finished trajectory.

Tool calls go through the training search proxy (one shared proxy per worker process; point all workers
at a shared proxy service to make rate limits global).

Written against VERL's ``verl.experimental.agent_loop`` API (0.6 era): ``AgentLoopBase``, ``register``,
``AgentLoopOutput`` and ``server_manager.generate``. Not exercised by this repo's CPU CI -- check field
names against the installed VERL version.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from agentic_search.core.types import Step, Trajectory
from agentic_search.rewards.composite import AgenticSearchReward
from agentic_search.runtime.agent import AgentRunner, ExecContext, RuntimeConfig
from agentic_search.runtime.protocol import TOOL_RESULT_ROLE, parse_turn, render_tool_results
from agentic_search.tools.base import ActionParseError, normalize_action

try:  # optional dependency
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
except ImportError:  # pragma: no cover - CPU CI has no verl
    AgentLoopBase, AgentLoopOutput = object, None

    def register(_name):
        return lambda cls: cls


class _Components:
    """Per-process singletons. Agent-loop workers are separate Ray processes, so they build their own
    runner / proxy / reward lazily from the YAML named by ``$AGENTIC_SEARCH_CONFIG`` (propagate it to all
    nodes, e.g. through the Ray runtime_env)."""

    runner: AgentRunner | None = None
    reward: AgenticSearchReward | None = None
    runtime_cfg: RuntimeConfig = RuntimeConfig()

    @classmethod
    def ensure(cls) -> tuple[AgentRunner, AgenticSearchReward]:
        if cls.runner is None or cls.reward is None:
            from agentic_search.config import build_training_components, load_config

            path = os.environ.get("AGENTIC_SEARCH_CONFIG")
            if not path:
                raise RuntimeError("set AGENTIC_SEARCH_CONFIG to the agentic_search YAML on every node")
            runner, reward = build_training_components(load_config(path))
            configure_agent_loop(runner, reward)
        assert cls.runner is not None
        assert cls.reward is not None
        return cls.runner, cls.reward


def configure_agent_loop(runner: AgentRunner, reward: AgenticSearchReward) -> None:
    _Components.runner, _Components.reward, _Components.runtime_cfg = runner, reward, runner.cfg


def _encode_followup(tokenizer: Any, role: str, content: str) -> list[int]:
    """Template ids of one follow-up message plus the next assistant header (delta encoding)."""
    base = [{"role": "system", "content": "x"}]
    full = tokenizer.apply_chat_template(
        [*base, {"role": role, "content": content}], add_generation_prompt=True, tokenize=True
    )
    prefix = tokenizer.apply_chat_template(base, add_generation_prompt=False, tokenize=True)
    return list(full[len(prefix) :])


@register("agentic_search")
class AgenticSearchAgentLoop(AgentLoopBase):
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        runner, reward = _Components.ensure()
        tok = self.tokenizer
        query: str = kwargs["raw_prompt"][-1]["content"]
        ctx = ExecContext(job_id=str(kwargs.get("uid", "")), trajectory_id=uuid.uuid4().hex[:12])
        traj = Trajectory(query=query, system_prompt=runner.prompts.build(query))
        prompt_ids = tok.apply_chat_template(
            [{"role": "system", "content": traj.system_prompt}, {"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=True,
        )
        response_ids: list[int] = []
        mask: list[int] = []
        step_ids: list[int] = []
        held: set[str] = set()
        cfg = _Components.runtime_cfg

        while True:
            budget_hit = traj.num_tool_rounds >= cfg.max_tool_rounds
            if budget_hit and cfg.budget_nudge:
                nudge = _encode_followup(tok, TOOL_RESULT_ROLE, cfg.budget_nudge)
                response_ids += nudge
                mask += [0] * len(nudge)
                step_ids += [-1] * len(nudge)
            out = await self.server_manager.generate(
                request_id=uuid.uuid4().hex,
                prompt_ids=prompt_ids + response_ids,
                sampling_params=sampling_params,
            )
            gen = list(getattr(out, "token_ids", out))
            step = Step(index=len(traj.steps), state_post_ids=frozenset(held))
            response_ids += gen
            mask += [1] * len(gen)
            step_ids += [step.index] * len(gen)
            text = tok.decode(gen, skip_special_tokens=True)
            parsed = parse_turn(text)
            step.thought, step.sub_queries, step.raw_text = parsed.thought, parsed.decomposition, text
            traj.steps.append(step)
            if (
                parsed.action is None
                or budget_hit
                or len(response_ids) >= self.config.actor_rollout_ref.rollout.response_length
            ):
                step.response = parsed.response or ""
                break
            try:
                calls = normalize_action(parsed.action)[: cfg.max_calls_per_round]
            except ActionParseError as e:
                step.action_parse_error = str(e)
                calls = []
            for k, c in enumerate(calls):
                c.call_id = f"{ctx.trajectory_id}-{step.index}-{k}"
            step.tool_calls = calls
            # horizontal action: all calls of one round run concurrently
            step.tool_results = list(await asyncio.gather(*(runner._execute(c, ctx) for c in calls)))
            held |= {p.post_id for r in step.tool_results for p in r.posts}
            obs = _encode_followup(tok, TOOL_RESULT_ROLE, render_tool_results(step))
            response_ids += obs
            mask += [0] * len(obs)
            step_ids += [-1] * len(obs)

        breakdown = reward(traj)
        limit = self.config.actor_rollout_ref.rollout.response_length
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:limit],
            response_mask=mask[:limit],
            num_turns=len(traj.steps) * 2 + 1,
            metrics={},
            reward_score=breakdown["score"],
            extra_fields={
                "step_ids": step_ids[:limit],
                "anchor_keys": [s.anchor_key() for s in traj.steps],
                "reward_breakdown": breakdown,
                "reward_valid": breakdown["valid"],  # 0 => infra failure; mask the sample out
            },
        )
