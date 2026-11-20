"""Rubric-based two-pass distillation of vertical (multi-hop) ReAct trajectories.

First pass (per taxonomy category, human in the loop):
    seed query -> several cost-efficient strong models (e.g. GLM-5.2 / Kimi-K3 / DeepSeek V4-Pro) each
    "think while searching" inside the *real* agent runtime -> trajectories A/B/C -> a judge model merges
    their strengths under the human-written *Agentic Search meta-rules* (round budget, thinking length,
    community tone, risk red lines) -> human review -> golden trajectory -> distil a category
    **Trajectory Rubric** (evaluation criteria) and a **Query Hint** (generation guidance).

Second pass (no human):
    more queries of the same category -> generate with the hint -> judge against the rubric -> keep the
    best trajectory per query above a threshold.

Queries come from two sources: LLM-generated per taxonomy node (category coverage) and sampled online
logs (realistic distribution: colloquial, typos, mixed language).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from agentic_search.core.types import Trajectory
from agentic_search.runtime.agent import AgentRunner


class TrajectoryMerger(Protocol):
    def merge(self, query: str, candidates: Mapping[str, Trajectory], meta_rules: str) -> Trajectory: ...


class RubricWriter(Protocol):
    def write(self, category: str, golden: Sequence[Trajectory]) -> tuple[str, str]:
        """Return ``(trajectory_rubric, query_hint)`` for a category."""
        ...


class RubricScorer(Protocol):
    def score(self, traj: Trajectory, rubric: str) -> float: ...


@dataclass
class CategoryAssets:
    category: str
    golden: list[Trajectory] = field(default_factory=list)
    rubric: str = ""
    hint: str = ""


async def first_pass(
    category: str,
    seed_queries: Sequence[str],
    runners: Mapping[str, AgentRunner],
    merger: TrajectoryMerger,
    rubric_writer: RubricWriter,
    meta_rules: str,
    human_review: Callable[[Trajectory], Trajectory | None],
) -> CategoryAssets:
    assets = CategoryAssets(category)
    for q in seed_queries:
        cands = dict(zip(runners, await asyncio.gather(*(r.run_once(q) for r in runners.values()))))
        merged = merger.merge(q, cands, meta_rules)
        reviewed = human_review(merged)  # edits or rejects; nothing becomes golden unreviewed
        if reviewed is not None:
            reviewed.meta.update(category=category, source="first_pass")
            assets.golden.append(reviewed)
    if assets.golden:
        assets.rubric, assets.hint = rubric_writer.write(category, assets.golden)
    return assets


async def second_pass(
    assets: CategoryAssets,
    queries: Sequence[str],
    runners: Mapping[str, AgentRunner],
    scorer: RubricScorer,
    threshold: float = 0.8,
) -> list[Trajectory]:
    kept: list[Trajectory] = []
    for q in queries:
        cands = await asyncio.gather(
            *(r.run_once(q, extra_instructions=assets.hint) for r in runners.values())
        )
        scored = sorted(((scorer.score(t, assets.rubric), t) for t in cands), key=lambda x: -x[0])
        if scored and scored[0][0] >= threshold:
            best = scored[0][1]
            best.meta.update(category=assets.category, source="second_pass", rubric_score=scored[0][0])
            kept.append(best)
    return kept
