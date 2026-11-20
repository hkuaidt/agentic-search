"""Agent runtime: an outer Main Loop (``serve``) and a bounded inner Agent Once Loop (``run_once``).

"The model decides what to do next; the framework parses the call, validates arguments, executes the
tool and feeds the real result back."

* **Horizontal action** -- all calls in one ``[Action]`` run concurrently (one round trip).
* **Vertical action** -- a later round conditions on earlier ``[Tool Result]``s.
* **Budget** -- C-end latency allows 1-2 ReAct rounds; after ``max_tool_rounds`` the model is asked to
  answer from what it has.
* **Checkpoint** -- any prefix of a trajectory (optionally with a partially written assistant turn) can
  be resumed with any :class:`LLMProvider`. This is how mined sub-queries replace a trajectory's
  ``[Input Decomposition]`` before re-rolling the rest.
"""

from __future__ import annotations

import asyncio
import itertools
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from agentic_search.core.types import Step, SubQuery, ToolCall, ToolResult, Trajectory
from agentic_search.runtime.llm import LLMProvider
from agentic_search.runtime.prompt import PromptBuilder
from agentic_search.runtime.protocol import (
    STOP_SEQUENCES,
    TOOL_RESULT_ROLE,
    parse_turn,
    render_messages,
    render_prefix_with_decomposition,
)
from agentic_search.tools.base import ActionParseError, ToolRegistry, normalize_action
from agentic_search.tools.proxy import SearchProxy


@dataclass
class ExecContext:
    job_id: str = ""
    trajectory_id: str = ""


ToolExecutor = Callable[[ToolCall, ExecContext], Awaitable[ToolResult]]


def direct_executor(registry: ToolRegistry) -> ToolExecutor:
    async def run(call: ToolCall, ctx: ExecContext) -> ToolResult:
        return await registry[call.name](call.arguments, call_id=call.call_id)

    return run


def proxy_executor(proxy: SearchProxy) -> ToolExecutor:
    """Route tool calls through the training search proxy (RL rollouts)."""

    async def run(call: ToolCall, ctx: ExecContext) -> ToolResult:
        return await proxy.call(
            call.name,
            call.arguments,
            job_id=ctx.job_id,
            trajectory_id=ctx.trajectory_id,
            call_id=call.call_id,
        )

    return run


@dataclass
class RuntimeConfig:
    max_tool_rounds: int = 2
    max_calls_per_round: int = 8
    default_limit: int = 5
    budget_nudge: str = "已达到检索轮数上限，请不要再调用工具，直接基于已有信息给出回答。"


@dataclass
class Checkpoint:
    """Resumable state: completed steps plus an optional partial assistant turn."""

    query: str
    system_prompt: str
    steps: list[Step] = field(default_factory=list)
    assistant_prefix: str | None = None
    trajectory_id: str = ""

    @classmethod
    def with_injected_decomposition(
        cls, traj: Trajectory, step_index: int, sub_queries: list[str], thought: str | None = None
    ) -> Checkpoint:
        """Keep ``traj.steps[:step_index]``, then restart step ``step_index`` with the given
        ``[Input Decomposition]`` (the LLM continues from ``[Action]``). Later history is discarded."""
        if not 0 <= step_index <= len(traj.steps):
            raise IndexError("step_index out of range")
        base = traj.steps[step_index].thought if step_index < len(traj.steps) else ""
        sqs = [SubQuery(f"sub_query_{i}", q) for i, q in enumerate(sub_queries, start=1)]
        return cls(
            query=traj.query,
            system_prompt=traj.system_prompt,
            steps=[*traj.steps[:step_index]],
            assistant_prefix=render_prefix_with_decomposition(thought if thought is not None else base, sqs),
            trajectory_id=str(traj.meta.get("trajectory_id", "")),
        )


class AgentRunner:
    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        executor: ToolExecutor | None = None,
        config: RuntimeConfig | None = None,
    ):
        self.llm, self.registry, self.prompts = llm, registry, prompt_builder
        self.executor = executor or direct_executor(registry)
        self.cfg = config or RuntimeConfig()

    async def serve(self, inbox: AsyncIterator[str]) -> AsyncIterator[Trajectory]:
        """Main Loop: keep answering requests until the inbox is exhausted."""
        async for query in inbox:
            yield await self.run_once(query)

    async def run_once(
        self,
        query: str,
        *,
        ctx: ExecContext | None = None,
        resume: Checkpoint | None = None,
        extra_instructions: str | None = None,
    ) -> Trajectory:
        """Agent Once Loop with at most ``max_tool_rounds`` tool rounds plus one answer turn.

        ``extra_instructions`` (e.g. a distilled Query Hint) are shown to the generator only; the clean
        prompt is kept in ``meta["clean_system_prompt"]`` so exported SFT data never contains them.
        """
        ctx = ctx or ExecContext(trajectory_id=uuid.uuid4().hex[:12])
        clean_prompt = resume.system_prompt if resume else self.prompts.build(query)
        system_prompt = clean_prompt + (f"\n\n# 提示\n{extra_instructions}" if extra_instructions else "")
        traj = Trajectory(
            query=query,
            system_prompt=system_prompt,
            meta={"trajectory_id": ctx.trajectory_id, "clean_system_prompt": clean_prompt},
        )
        traj.steps = list(resume.steps) if resume else []
        prefix = resume.assistant_prefix if resume else None
        held = {p for p in traj.retrieved_posts()}
        call_ids = itertools.count(1)

        while True:
            rounds_used = traj.num_tool_rounds
            messages = render_messages(system_prompt, query, traj.steps)
            budget_hit = rounds_used >= self.cfg.max_tool_rounds
            if budget_hit and self.cfg.budget_nudge:
                messages.append({"role": TOOL_RESULT_ROLE, "content": self.cfg.budget_nudge})
            gen = await self.llm.generate(messages, stop=STOP_SEQUENCES, assistant_prefix=prefix)
            text = (prefix or "") + gen
            prefix = None
            parsed = parse_turn(text)
            step = Step(
                index=len(traj.steps),
                thought=parsed.thought,
                sub_queries=parsed.decomposition,
                state_post_ids=frozenset(held),
                raw_text=text,
            )
            traj.steps.append(step)

            if parsed.action is None or budget_hit:
                # final turn: a [Response], or a forced stop after the round budget is spent
                step.response = parsed.response if parsed.response is not None else ""
                if parsed.response is None:
                    traj.meta["truncated"] = "budget" if budget_hit else "no_action_or_response"
                return traj

            try:
                calls = normalize_action(parsed.action, default_limit=self.cfg.default_limit)
            except ActionParseError as e:
                step.action_parse_error = str(e)
                continue
            calls = calls[: self.cfg.max_calls_per_round]
            for c in calls:
                c.call_id = f"tool_call_{ctx.trajectory_id}_{next(call_ids)}"
            step.tool_calls = calls
            step.tool_results = list(await asyncio.gather(*(self._execute(c, ctx) for c in calls)))
            held |= {p.post_id for r in step.tool_results for p in r.posts}

    async def _execute(self, call: ToolCall, ctx: ExecContext) -> ToolResult:
        if not self.registry.validate_arguments(call.name, call.arguments):
            return ToolResult(call.call_id, call.name, status="invalid", error="schema validation failed")
        args = {"limit": self.cfg.default_limit, **call.arguments}
        return await self.executor(ToolCall(call.name, args, call.call_id, call.raw), ctx)
