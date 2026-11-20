import numpy as np
import pytest
import torch

from agentic_search.algos import (
    broadcast_step_advantage,
    combine_advantages,
    gigpo_step_advantage,
    group_normalized_advantage,
    group_reward_std_stats,
    luffy_mixed_group_advantage,
    opd_token_advantage,
)


def test_group_normalized_advantage_matches_formula():
    r = torch.tensor([1.0, 0.0, 0.5, 0.5, 0.2, 0.9])
    gid = ["a", "a", "a", "b", "b", "b"]
    adv = group_normalized_advantage(r, gid, eps=0.0)
    for idx in ([0, 1, 2], [3, 4, 5]):
        g = r[idx]
        expected = (g - g.mean()) / g.std(unbiased=False)
        assert torch.allclose(adv[idx], expected, atol=1e-6)
    # zero-variance group -> zero advantage (the "all 0 / all 1" trap the debug protocol monitors)
    assert torch.all(group_normalized_advantage([1.0, 1.0, 1.0], [0, 0, 0]) == 0)


def test_singleton_group_has_no_baseline():
    assert group_normalized_advantage([0.7], ["x"]).item() == 0.0


def _worked_example_keys():
    """Three trajectories of the GiGPO worked example. Every box is one step; A and B are the two tool
    calls whose anchor states (held posts) coincide across trajectories; R is the final answer.

    Distances to R: A: 6 / 3 / 2, B: 2 / 2 / 1.
    """
    traj1 = ["s0", "A", "m1", "m2", "m3", "B", "m4", "R1"]  # A at t=2 (T-t=6), B at t=6 (T-t=2)
    traj2 = ["s0", "A", "B", "n1", "R2"]  # A at t=2 (T-t=3), B at t=3 (T-t=2)
    traj3 = ["A", "B", "R3"]  # A at t=1 (T-t=2), B at t=2 (T-t=1)
    # unique keys for the non-shared steps so they stay singletons (s0 is shared by traj1/traj2 though)
    traj1 = [k if k in {"A", "B", "s0"} else f"t1-{k}" for k in traj1]
    traj2 = [k if k in {"A", "B", "s0"} else f"t2-{k}" for k in traj2]
    traj3 = [k if k in {"A", "B"} else f"t3-{k}" for k in traj3]
    return [traj1, traj2, traj3]


def test_gigpo_reproduces_worked_example():
    keys = _worked_example_keys()
    adv = gigpo_step_advantage(
        rewards=[1.0, 1.0, 1.0], group_ids=[0, 0, 0], anchor_keys=keys, gamma=0.9, eps=0.0
    )
    a_pos = [1, 1, 0]  # 0-based index of A in each trajectory
    b_pos = [5, 2, 1]
    got_a = [adv[i][a_pos[i]] for i in range(3)]
    got_b = [adv[i][b_pos[i]] for i in range(3)]
    np.testing.assert_allclose(got_a, [-1.357, 0.332, 1.024], atol=1e-3)
    np.testing.assert_allclose(got_b, [-0.707, -0.707, 1.414], atol=1e-3)


def test_gigpo_discounted_returns_and_singletons():
    keys = _worked_example_keys()
    adv = gigpo_step_advantage([1.0, 1.0, 1.0], [0, 0, 0], keys, gamma=0.9)
    # steps whose state appears only once in the group carry no step-level signal
    assert adv[0][2] == 0.0 and adv[2][2] == 0.0
    # gamma = 1 with equal rewards -> every return identical -> no signal anywhere
    flat = gigpo_step_advantage([1.0, 1.0, 1.0], [0, 0, 0], keys, gamma=1.0)
    assert all(np.allclose(a, 0.0) for a in flat)


def test_gigpo_never_groups_across_prompts():
    adv = gigpo_step_advantage([1.0, 0.0], ["p1", "p2"], [["s"], ["s"]], gamma=0.9)
    assert adv[0][0] == 0.0 and adv[1][0] == 0.0


def test_gigpo_input_validation():
    with pytest.raises(ValueError, match="gamma"):
        gigpo_step_advantage([1.0], [0], [["s"]], gamma=0.0)
    with pytest.raises(ValueError, match="same length"):
        gigpo_step_advantage([1.0, 0.0], [0], [["s"]])


def test_broadcast_step_advantage_respects_mask():
    step_adv = [np.array([0.5, -1.0], dtype=np.float32)]
    ids = torch.tensor([[-1, -1, 0, 0, -1, 1, 1, -1]])
    out = broadcast_step_advantage(step_adv, ids)
    assert out.tolist() == [[0.0, 0.0, 0.5, 0.5, 0.0, -1.0, -1.0, 0.0]]
    with pytest.raises(IndexError):
        broadcast_step_advantage(step_adv, torch.tensor([[2]]))


def test_opd_and_mixing_matches_worked_example():
    # GSPO advantage -1, a token strongly rejected by the teacher has OPD advantage -2,
    # weight 0.5 -> mixed advantage -2; a neutral token keeps ~-1.
    teacher = torch.tensor([[-3.0, -1.0]])
    student_old = torch.tensor([[-1.0, -1.0]])
    mask = torch.ones(1, 2)
    opd = opd_token_advantage(teacher, student_old, mask)
    assert opd.tolist() == [[-2.0, 0.0]]
    total = combine_advantages(torch.tensor([-1.0]), mask, opd_adv_tokens=opd, lambda_opd=0.5)
    assert total.tolist() == [[-2.0, -1.0]]


def test_combine_zeroes_non_generated_tokens():
    mask = torch.tensor([[0, 1, 1, 0]])
    step = torch.tensor([[9.0, 1.0, 1.0, 9.0]])
    total = combine_advantages(torch.tensor([2.0]), mask, step_adv_tokens=step, omega=0.5)
    assert total.tolist() == [[0.0, 2.5, 2.5, 0.0]]
    with pytest.raises(ValueError):
        combine_advantages(torch.tensor([1.0]), mask, omega=-1.0)


def test_luffy_rescues_all_zero_groups():
    on, off = luffy_mixed_group_advantage([0.0] * 8, [])
    assert torch.all(on == 0)  # plain GRPO: no gradient signal
    on, off = luffy_mixed_group_advantage([0.0] * 8, [1.0])
    assert torch.all(on < 0) and off.item() > 0  # teacher demo creates a learning direction


def test_group_reward_std_stats():
    stats = group_reward_std_stats([0, 0, 1, 0.5], ["a", "a", "b", "b"])
    assert stats["frac_zero_std_groups"] == 0.5
    assert stats["frac_all_zero_groups"] == 0.5
