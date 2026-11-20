import pytest

from agentic_search import demo
from agentic_search.core.types import Step, Trajectory
from agentic_search.rewards import (
    AgenticSearchReward,
    CallableJudge,
    EnsembleJudge,
    RewardWeights,
    check_multi_hop_attribution,
    check_tool_format,
    compute_format_reward,
    compute_query_reward,
    parse_discrete_score,
)
from agentic_search.runtime import AgentRunner

QUERY = "成都两日游，带老人，希望少走路"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ({"name": "tag_search", "arguments": {"query": "成都", "limit": 5}}, 1.0),
        ('{"name": "tag_search", "arguments": {"query": "成都"}}', 1.0),
        ("{bad json", 0.0),
        ({"name": "tag_search", "arguments": {"query": "成都"}, "call_id": "x"}, 0.0),  # extra key
        ({"name": "web_search", "arguments": {"query": "成都"}}, 0.0),  # unregistered tool
        ({"name": "tag_search", "arguments": {"query": "成都", "limit": 0}}, 0.0),  # out of range
        ({"name": "tag_search", "arguments": {"q": "成都"}}, 0.0),  # unknown / missing param
        ({"name": "tag_search", "arguments": "query=成都"}, 0.0),
    ],
)
def test_check_tool_format(registry, call, expected):
    assert check_tool_format(call, registry) == expected


@pytest.mark.parametrize(
    ("idx", "thought", "expected"),
    [
        (2, "根据第 1 步的结果继续", 1.0),
        (2, "根据第1步和根据第 2 步", 1.0),
        (2, "根据第 3 步", 0.0),  # cites itself
        (2, "根据第 0 步", 0.0),
        (2, "没有引用", 0.0),
    ],
)
def test_check_multi_hop_attribution(idx, thought, expected):
    assert check_multi_hop_attribution(idx, thought, num_steps=5) == expected


async def _trajectory(registry, prompts, vertical=True):
    return await AgentRunner(demo.scripted_policy(vertical), registry, prompts).run_once(QUERY)


async def test_format_reward_on_gold_trajectory(registry, prompts):
    out = compute_format_reward(await _trajectory(registry, prompts), registry)
    assert out == {"format": 1.0, "format/tool": 1.0, "format/hop": 1.0}


async def test_format_reward_penalises_missing_attribution(registry, prompts):
    traj = await _trajectory(registry, prompts)
    traj.steps[1].thought = traj.steps[1].thought.replace("根据第 1 步，", "")
    out = compute_format_reward(traj, registry)
    assert out["format/hop"] == 0.0 and out["format"] == pytest.approx(0.6)


def test_format_reward_is_a_mean_not_a_sum(registry):
    good = {"name": "tag_search", "arguments": {"query": "a"}}
    bad = {"name": "nope", "arguments": {}}

    def traj(n_good):
        from agentic_search.core.types import ToolCall

        calls = [ToolCall("tag_search", good["arguments"], raw=good)] * n_good + [
            ToolCall("nope", {}, raw=bad)
        ]
        return Trajectory("q", steps=[Step(0, tool_calls=calls), Step(1, response="ok")])

    assert compute_format_reward(traj(1), registry)["format/tool"] == 0.5
    assert compute_format_reward(traj(3), registry)["format/tool"] == 0.75


def test_parse_discrete_score():
    assert parse_discrete_score('{"relevance": 0.6}', key="relevance") == 0.5
    assert parse_discrete_score("1") == 1.0
    assert parse_discrete_score("分数：0.2") == 0.0
    assert parse_discrete_score("no number") is None
    assert (
        parse_discrete_score(
            EnsembleJudge(
                [CallableJudge(lambda p: "1"), CallableJudge(lambda p: "0.5"), CallableJudge(lambda p: "1")]
            )("x")
        )
        == 1.0
    )


async def test_query_reward_averages_rounds(registry, prompts):
    traj = await _trajectory(registry, prompts)
    assert compute_query_reward(traj, demo.rule_query_judge())["query"] == 1.0
    traj.steps[1].tool_calls[0].arguments["query"] = "股票 基金 推荐"  # drifted vertical query
    assert compute_query_reward(traj, demo.rule_query_judge())["query"] == 0.75


def test_query_reward_without_search():
    traj = Trajectory("q", steps=[Step(0, response="ok")])
    judge = CallableJudge(lambda p: "1")
    assert compute_query_reward(traj, judge, search_required=True)["query"] == 0.0
    assert compute_query_reward(traj, judge, search_required=False)["query"] == 1.0


async def test_composite_reward_and_fabricated_citation(registry, prompts):
    reward = AgenticSearchReward(registry, demo.rule_query_judge(), demo.rule_outcome_judges())
    traj = await _trajectory(registry, prompts)
    out = reward(traj)
    assert out["score"] == pytest.approx(1.0) and out["valid"] == 1.0

    traj.steps[-1].response += "[编造](ref: zzz)"
    out = reward(traj)
    assert out["outcome/citation_gate"] == 0.0 and out["outcome/trust"] == 0.0
    assert out["score"] == pytest.approx(0.1 + 0.3 + 0.6 * 0.75)


def test_reward_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        RewardWeights(0.1, 0.35, 0.6)
    RewardWeights(0.1, 0.35, 0.55)  # any split summing to 1 is accepted
