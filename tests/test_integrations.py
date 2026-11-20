from pathlib import Path

import pytest
import torch

from agentic_search.config import build_runtime_config, build_training_components, load_config
from agentic_search.integrations.verl import AdvantageConfig, compute_gspo_gigpo_opd_advantage
from agentic_search.integrations.verl.teacher import parse_prompt_logprobs

ROOT = Path(__file__).resolve().parents[1]


def test_verl_advantage_combines_three_granularities():
    # two prompts x two rollouts; tokens: [gen step0, gen step0, tool, gen step1, pad]
    mask = torch.tensor([[1, 1, 0, 1, 0]] * 4, dtype=torch.float32)
    step_ids = torch.tensor([[0, 0, -1, 1, -1]] * 4)
    rewards = torch.zeros(4, 5)
    rewards[:, 3] = torch.tensor([1.0, 0.0, 0.5, 0.5])
    anchors = [["empty", "s1"], ["empty", "s1"], ["empty", "s2"], ["empty", "s3"]]
    old = torch.full((4, 5), -1.0)
    teacher = old.clone()
    teacher[0, 1] = -3.0  # teacher rejects one token of rollout 0

    cfg = AdvantageConfig(gamma=0.9, omega=0.5, lambda_opd=0.5, opd_clip=None)
    adv, ret, metrics = compute_gspo_gigpo_opd_advantage(
        rewards, mask, ["p", "p", "q", "q"], step_ids, anchors, old, teacher, cfg
    )
    assert torch.equal(adv, ret)
    assert torch.all(adv[mask == 0] == 0)  # tool results and padding carry no advantage
    # prompt "p": A^E = +1/-1; step groups {empty} and {s1} both separate the two rollouts
    a_s = 1.0  # normalised step advantage magnitude within a 2-member group
    assert adv[0, 0].item() == pytest.approx(1.0 + 0.5 * a_s, abs=1e-4)
    assert adv[0, 1].item() == pytest.approx(1.0 + 0.5 * a_s + 0.5 * -2.0, abs=1e-4)
    assert adv[1, 3].item() == pytest.approx(-1.0 - 0.5 * a_s, abs=1e-4)
    # prompt "q": equal rewards -> no trajectory signal; distinct step-1 states -> no step signal
    assert torch.allclose(adv[2:], torch.zeros_like(adv[2:]))
    assert 0 < metrics["gigpo/frac_steps_with_signal"] < 1


def test_verl_advantage_degrades_to_plain_gspo():
    mask = torch.ones(2, 3)
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    adv, _, _ = compute_gspo_gigpo_opd_advantage(
        rewards, mask, [0, 0], cfg=AdvantageConfig(omega=0, lambda_opd=0)
    )
    assert torch.allclose(adv[0], torch.ones(3)) and torch.allclose(adv[1], -torch.ones(3))


def test_parse_prompt_logprobs():
    entries = [None, {"11": {"logprob": -0.5, "rank": 1}}, {"7": {"logprob": -2.0}, "12": {"logprob": -0.1}}]
    assert parse_prompt_logprobs(entries, [5, 11, 12]) == [0.0, -0.5, -0.1]


def test_shipped_config_builds_components_offline():
    cfg = load_config(ROOT / "configs" / "agentic_search.yaml")
    assert build_runtime_config(cfg).max_tool_rounds == 2
    runner, reward = build_training_components(cfg)
    assert set(runner.registry.names()) == {"tag_search", "community_search"}
    assert (reward.weights.format, reward.weights.query, reward.weights.outcome) == (0.1, 0.3, 0.6)
    AdvantageConfig(**cfg["advantage"])


def test_verl_modules_import_without_verl(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_CONFIG", raising=False)
    from agentic_search.integrations.verl import agent_loop, reward_entry

    assert reward_entry.compute_score("x", "", None, {"reward_breakdown": {"score": 0.7}}) == 0.7
    assert agent_loop.AgenticSearchAgentLoop is not None
