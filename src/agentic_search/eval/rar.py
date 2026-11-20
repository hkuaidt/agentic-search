"""RaR (Rubrics-as-Rewards style) evaluation: P0 red-line gate + P1/P2 100-point quality score.

    S_quality = F (10) + Q (30) + A (60)           S_final = G_P0 * S_quality
    Pass(tau) = 1[ all P0 pass  and  S_quality >= 80 ]

P0 holds only *explicit, decidable* hard requirements ("每天步行不超过 3 公里" is P0, "尽量少走路" is a
preference). If evidence is insufficient to decide a P0 item, the sample goes to human review instead of
being declared a violation; ``G_P0`` / ``S_final`` / ``passed`` are then ``None``. The pre-gate quality score
and every sub-score are always kept for failure attribution.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from agentic_search.core.types import Trajectory
from agentic_search.rewards.format_reward import compute_format_reward
from agentic_search.rewards.judge import INJECTION_GUARD, Judge, parse_discrete_score
from agentic_search.rewards.outcome import citation_gate
from agentic_search.tools.base import ToolRegistry


@dataclass(frozen=True)
class RubricItem:
    key: str
    layer: Literal["P1", "P2"]
    category: Literal["F", "Q", "A"]
    points: float
    criterion: str
    method: Literal["code", "judge", "code+judge"]


DEFAULT_RUBRIC: tuple[RubricItem, ...] = (
    RubricItem("tool_call_validity", "P1", "F", 5, "工具名称、JSON、参数类型和取值符合 schema", "code"),
    RubricItem(
        "multi_turn_consistency", "P1", "F", 5, "工具请求与返回正确对应，后续决策依据实际获得的信息", "code"
    ),
    RubricItem("query_relevance", "P1", "Q", 15, "搜索直接服务原始需求，或解决必要的前置问题", "judge"),
    RubricItem("horizontal_coverage", "P1", "Q", 10, "子查询覆盖关键需求，无明显遗漏或无效重复", "judge"),
    RubricItem("vertical_and_stop", "P1", "Q", 5, "信息不足时补查，证据充分后及时结束", "judge"),
    RubricItem("final_coverage", "P2", "A", 20, "回答覆盖主要目标及重要约束", "judge"),
    RubricItem("evidence_support", "P2", "A", 20, "关键结论有可追溯依据，引用内容支持对应表述", "code+judge"),
    RubricItem("actionability", "P2", "A", 10, "给出具体方案、选择理由或有用的比较", "judge"),
    RubricItem("organization", "P2", "A", 5, "结构清晰、简洁自然，避免机械拼接内容", "judge"),
    RubricItem(
        "uncertainty", "P2", "A", 5, "区分事实与个人体验，对冲突、过时或不足的信息作合理说明", "judge"
    ),
)


@dataclass
class P0Constraint:
    """A decidable hard requirement. ``check`` returns True (satisfied), False (violated) or None
    (cannot decide); without ``check`` the judge decides."""

    id: str
    description: str
    check: Callable[[Trajectory], bool | None] | None = None


@dataclass
class RaRTask:
    task_id: str
    query: str
    p0: list[P0Constraint] = field(default_factory=list)
    rubric_notes: str = ""  # query-specific rubric details generated offline with the task
    category: str = ""
    search_required: bool = True


@dataclass
class RaRResult:
    task_id: str
    sub_scores: dict[str, float]
    F: float
    Q: float
    A: float
    S_quality: float
    G_P0: int | None
    S_final: float | None
    passed: bool | None
    p0_details: dict[str, bool | None]

    @property
    def needs_human_review(self) -> bool:
        return self.G_P0 is None


ITEM_PROMPT = """你是 Agentic Search 评测员，只评估一个维度：{criterion}
{guard}

用户问题：{query}
本题评分细则：{notes}
完整执行过程：
{trace}

评分标准：1.0 完全满足；0.5 部分满足；0.0 不满足。只输出数字，不要解释。"""

P0_PROMPT = """判断最终回答是否违反了用户的硬性要求：{constraint}
{guard}

用户问题：{query}
完整执行过程：
{trace}

若明确违反输出 1；若明确未违反输出 0；若证据不足无法判定输出 -1。只输出数字。"""


def render_execution_trace(traj: Trajectory, max_posts_per_call: int = 3, chars: int = 120) -> str:
    lines = []
    for s in traj.steps:
        if s.is_tool_step:
            lines.append(f"第 {s.index + 1} 步 思考：{s.thought}")
            for c, r in zip(s.tool_calls, s.tool_results):
                posts = "; ".join(
                    f"[{p.post_id}] {p.title}：{p.snippet(chars)}" for p in r.posts[:max_posts_per_call]
                )
                lines.append(
                    f"  调用 {c.name}({json.dumps(c.arguments, ensure_ascii=False)}) -> {r.status} {posts}"
                )
        else:
            lines.append(f"最终回答：{s.response}")
    return "\n".join(lines)


class RaREvaluator:
    def __init__(
        self,
        registry: ToolRegistry,
        judge: Judge,
        rubric: Sequence[RubricItem] = DEFAULT_RUBRIC,
        pass_threshold: float = 80.0,
    ):
        total = sum(i.points for i in rubric)
        if abs(total - 100) > 1e-6:
            raise ValueError(f"rubric must sum to 100 points, got {total}")
        self.registry, self.judge, self.rubric, self.pass_threshold = (
            registry,
            judge,
            list(rubric),
            pass_threshold,
        )

    def _judge_item(self, item: RubricItem, task: RaRTask, trace: str) -> float:
        out = self.judge(
            ITEM_PROMPT.format(
                criterion=item.criterion,
                guard=INJECTION_GUARD,
                query=task.query,
                notes=task.rubric_notes or "无",
                trace=trace,
            )
        )
        return parse_discrete_score(out) or 0.0

    def _code_item(self, item: RubricItem, traj: Trajectory) -> float:
        fmt = compute_format_reward(traj, self.registry)
        if item.key == "tool_call_validity":
            return fmt["format/tool"]
        if item.key == "multi_turn_consistency":
            aligned = all(
                len(s.tool_calls) == len(s.tool_results)
                and all(c.call_id == r.call_id for c, r in zip(s.tool_calls, s.tool_results))
                for s in traj.steps
            )
            return 0.5 * float(aligned) + 0.5 * fmt["format/hop"]
        raise KeyError(item.key)

    def _p0(self, task: RaRTask, traj: Trajectory, trace: str) -> dict[str, bool | None]:
        details: dict[str, bool | None] = {}
        details["no_fabricated_source"] = citation_gate(traj)["gate"] == 1.0
        for c in task.p0:
            if c.check is not None:
                details[c.id] = c.check(traj)
                continue
            raw = self.judge(
                P0_PROMPT.format(
                    constraint=c.description, guard=INJECTION_GUARD, query=task.query, trace=trace
                )
            ).strip()
            details[c.id] = {"0": True, "1": False}.get(raw[:2].strip())
        return details

    def evaluate(self, task: RaRTask, traj: Trajectory) -> RaRResult:
        trace = render_execution_trace(traj)
        subs: dict[str, float] = {}
        for item in self.rubric:
            if not traj.final_response and item.category == "A":
                frac = 0.0
            elif item.method == "code":
                frac = self._code_item(item, traj)
            elif item.method == "code+judge":  # evidence: ids must exist (code) and support claims (judge)
                frac = citation_gate(traj)["gate"] * self._judge_item(item, task, trace)
            else:
                frac = self._judge_item(item, task, trace)
            subs[item.key] = frac * item.points
        cat = {c: sum(subs[i.key] for i in self.rubric if i.category == c) for c in "FQA"}
        quality = sum(cat.values())
        p0 = self._p0(task, traj, trace)
        g: int | None = (
            0
            if any(v is False for v in p0.values())
            else (None if any(v is None for v in p0.values()) else 1)
        )
        final = None if g is None else g * quality
        passed = None if g is None else (g == 1 and quality >= self.pass_threshold)
        return RaRResult(task.task_id, subs, cat["F"], cat["Q"], cat["A"], quality, g, final, passed, p0)
