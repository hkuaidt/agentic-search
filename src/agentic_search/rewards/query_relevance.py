"""Query-relevance reward (30%): does each search round serve the user's original intent?

A trajectory-level *process* reward: every round is judged (with the earlier rounds as context only),
then averaged -- so the agent cannot earn more by searching more, and drifting multi-hop queries
(the "searches grow, cost grows, quality doesn't" badcase) are penalised.
"""

from __future__ import annotations

import json

from agentic_search.core.types import Trajectory
from agentic_search.rewards.judge import INJECTION_GUARD, Judge, parse_discrete_score

QUERY_RELEVANCE_PROMPT = """判断本轮搜索 Query 是否服务于用户原始意图。

评分标准：
1.0：直接回答原始问题，或解决必要的前置子问题。
0.5：与主题相关，但对完成原始任务帮助有限。
0.0：偏离原始需求。

原始需求是判断依据，历史只用于解释查询目的。
合理的多跳查询不必包含原始问题中的关键词。
{guard}

待评估数据：
{data}

只返回 JSON：{{"relevance": 分数}}"""


def compute_query_reward(traj: Trajectory, judge: Judge, search_required: bool = True) -> dict[str, float]:
    rounds = traj.search_rounds()
    if not rounds:
        return {"query": 0.0 if search_required else 1.0, "query/judge_failures": 0.0}
    scores: list[float] = []
    history: list[list[str]] = []
    failures = 0
    for rnd in rounds:
        data = json.dumps(
            {"original_query": traj.query, "queries": rnd["queries"], "history": history}, ensure_ascii=False
        )
        score = parse_discrete_score(
            judge(QUERY_RELEVANCE_PROMPT.format(guard=INJECTION_GUARD, data=data)), key="relevance"
        )
        if score is None:
            failures += 1
            score = 0.0
        scores.append(score)
        history.append(rnd["queries"])
    return {"query": sum(scores) / len(scores), "query/judge_failures": float(failures)}
