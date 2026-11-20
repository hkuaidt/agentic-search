import json

import pytest

from agentic_search import demo
from agentic_search.core.types import Segment
from agentic_search.runtime import AgentRunner, Checkpoint, RuntimeConfig, ScriptedProvider, parse_turn
from agentic_search.tools import normalize_action

QUERY = "成都两日游，带老人，希望少走路"


def test_parse_turn_sections():
    turn = demo.chengdu_policy_turns()[0]
    p = parse_turn(turn)
    assert "少走路" in p.thought
    assert [s.query for s in p.decomposition] == ["成都两日游 老人 少走路", "成都 老街 小吃"]
    assert json.loads(p.action)[0]["name"] == "tag_search"
    assert p.response is None


def test_normalize_action_expands_fanout_shorthand():
    raw = {
        "name": ["community_search", "tag_search"],
        "arguments": {
            "sub_queries": [{"id": "sub_query_1", "query": "a"}, {"id": "sub_query_2", "query": "b"}]
        },
    }
    calls = normalize_action(raw)
    assert [(c.name, c.arguments["query"]) for c in calls] == [
        ("community_search", "a"),
        ("tag_search", "a"),
        ("community_search", "b"),
        ("tag_search", "b"),
    ]


async def test_horizontal_then_vertical_trajectory(registry, prompts):
    runner = AgentRunner(demo.scripted_policy(), registry, prompts)
    traj = await runner.run_once(QUERY)
    assert traj.num_tool_rounds == 2
    s0, s1, s2 = traj.steps
    assert len(s0.tool_calls) == 3 and all(r.status == "ok" for r in s0.tool_results)
    assert {r.name for r in s0.tool_results} == {"tag_search", "community_search"}
    assert s0.state_post_ids == frozenset()
    # GiGPO anchor state = posts held before deciding; grows monotonically
    assert s1.state_post_ids and s1.state_post_ids <= s2.state_post_ids
    assert "c1" in s2.state_post_ids  # retrieved by the vertical round
    assert traj.final_response and "(ref: t1)" in traj.final_response
    assert all(c.call_id.startswith("tool_call_") for c in s0.tool_calls)


async def test_tool_results_are_fed_back_as_context(registry, prompts):
    llm = demo.scripted_policy()
    await AgentRunner(llm, registry, prompts).run_once(QUERY)
    second_call_messages = llm.calls[1][0]
    assert second_call_messages[-1]["content"].startswith(f"[{Segment.TOOL_RESULT.value}]")
    assert "[Post t1]" in second_call_messages[-1]["content"]


async def test_round_budget_forces_an_answer(registry, prompts):
    turns = demo.chengdu_policy_turns()
    looping = ScriptedProvider([turns[0], turns[0], turns[0]])
    traj = await AgentRunner(looping, registry, prompts, config=RuntimeConfig(max_tool_rounds=2)).run_once(
        QUERY
    )
    assert traj.num_tool_rounds == 2
    assert traj.meta["truncated"] == "budget"
    assert len(looping.calls) == 3
    assert "上限" in looping.calls[-1][0][-1]["content"]


async def test_bad_json_is_a_policy_error_not_a_crash(registry, prompts):
    llm = ScriptedProvider(["[Thinking]\nx\n[Action]\n{not json", demo.chengdu_policy_turns()[-1]])
    traj = await AgentRunner(llm, registry, prompts).run_once(QUERY)
    assert traj.steps[0].action_parse_error
    assert "动作解析失败" in llm.calls[1][0][-1]["content"]


async def test_invalid_arguments_are_marked_invalid(registry, prompts):
    bad = '[Thinking]\nx\n[Action]\n[{"name": "tag_search", "arguments": {"query": "成都", "limit": 99}}]'
    llm = ScriptedProvider([bad, demo.chengdu_policy_turns()[-1]])
    traj = await AgentRunner(llm, registry, prompts).run_once(QUERY)
    assert traj.steps[0].tool_results[0].status == "invalid"
    assert not traj.had_infra_failure()


async def test_checkpoint_injects_decomposition_and_resumes(registry, prompts):
    traj = await AgentRunner(demo.scripted_policy(), registry, prompts).run_once(QUERY)
    ckpt = Checkpoint.with_injected_decomposition(
        traj, step_index=0, sub_queries=["成都 带老人 景点", "成都 小吃"]
    )
    assert ckpt.assistant_prefix.rstrip().endswith("[Action]")

    def continue_from_prefix(messages, prefix):
        if prefix:  # the provider only writes the [Action] for the injected decomposition
            return '[{"name": "tag_search", "arguments": {"query": "成都 带老人 景点", "limit": 3}}]'
        return demo.chengdu_policy_turns()[-1]

    llm = ScriptedProvider(continue_from_prefix)
    new = await AgentRunner(llm, registry, prompts).run_once(QUERY, resume=ckpt)
    assert [q.query for q in new.steps[0].sub_queries] == ["成都 带老人 景点", "成都 小吃"]
    assert new.steps[0].raw_text.startswith(ckpt.assistant_prefix)
    assert new.steps[0].tool_calls[0].arguments["query"] == "成都 带老人 景点"
    assert new.final_response


def test_dynamic_few_shot_picks_nearest_example(prompts):
    assert prompts.select_examples("北京旅游攻略")[0].category == "旅行"
    assert prompts.select_examples("大模型面试题")[0].category == "求职"


@pytest.mark.parametrize("vertical", [True, False])
async def test_demo_policy_variants(registry, prompts, vertical):
    traj = await AgentRunner(demo.scripted_policy(vertical), registry, prompts).run_once(QUERY)
    assert traj.num_tool_rounds == (2 if vertical else 1)
