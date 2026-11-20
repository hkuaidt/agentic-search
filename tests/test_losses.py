import torch

from agentic_search.algos import (
    gspo_sequence_loss,
    gspo_token_loss,
    luffy_loss,
    luffy_shaping,
    ppo_token_loss,
)


def _logits_setup(B=3, L=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    old = -torch.rand(B, L, generator=g) * 2 - 0.1
    delta = torch.randn(B, L, generator=g) * 1e-4  # tiny policy drift, like one optimizer step
    new = (old + delta).clone().requires_grad_(True)
    mask = torch.ones(B, L)
    mask[0, :2] = 0  # system/user tokens
    mask[1, 3:5] = 0  # a tool-result span inside the response
    return new, old, mask


def test_gspo_token_equals_gspo_when_advantage_is_sequence_constant():
    new, old, mask = _logits_setup()
    seq_adv = torch.tensor([1.0, -0.5, 0.3])
    tok_adv = seq_adv[:, None].expand_as(mask).clone()

    loss_tok = gspo_token_loss(new, old, tok_adv, mask, loss_agg_mode="seq-mean-token-mean").loss
    (g_tok,) = torch.autograd.grad(loss_tok, new)

    new2 = new.detach().clone().requires_grad_(True)
    loss_seq = gspo_sequence_loss(new2, old, seq_adv, mask)
    (g_seq,) = torch.autograd.grad(loss_seq, new2)

    assert torch.allclose(loss_tok, loss_seq, atol=1e-6)
    assert torch.allclose(g_tok, g_seq, atol=1e-6)


def test_gspo_ignores_masked_tokens():
    new, old, mask = _logits_setup()
    adv = torch.randn_like(mask)
    base = gspo_token_loss(new, old, adv, mask).loss
    perturbed_new = new.detach().clone()
    perturbed_new[mask == 0] += 5.0  # tool results / prompt tokens must not matter
    assert torch.allclose(base, gspo_token_loss(perturbed_new, old, adv, mask).loss)


def test_gspo_clips_on_sequence_ratio():
    old = torch.zeros(1, 4)
    new = torch.full((1, 4), 0.01, requires_grad=True)  # s = e^0.01 > 1 + 4e-4
    out = gspo_token_loss(new, old, torch.ones(1, 4), torch.ones(1, 4))
    (g,) = torch.autograd.grad(out.loss, new)
    assert out.clip_frac.item() == 1.0
    assert torch.all(g == 0)  # positive advantage + ratio above the upper bound -> no gradient


def test_ppo_token_loss_dual_clip_bounds_negative_advantages():
    old = torch.zeros(1, 1)
    new = torch.tensor([[3.0]], requires_grad=True)  # ratio e^3 ~ 20
    out = ppo_token_loss(new, old, torch.tensor([[-1.0]]), torch.ones(1, 1), dual_clip_c=3.0)
    assert torch.isclose(out.loss, torch.tensor(3.0))


def test_luffy_shaping_gradient_upweights_rare_teacher_tokens():
    p = torch.tensor([0.01, 0.9], requires_grad=True)
    (grad,) = torch.autograd.grad(luffy_shaping(p, 0.1).sum(), p)
    assert grad[0] > grad[1] * 10


def test_luffy_loss_pushes_teacher_tokens_up():
    on_new = torch.full((2, 3), -1.0, requires_grad=True)
    off_new = torch.full((1, 3), -4.0, requires_grad=True)
    out = luffy_loss(
        on_new,
        on_new.detach(),
        torch.full((2, 3), -0.5),
        torch.ones(2, 3),
        off_new,
        torch.full((1, 3), 1.0),
        torch.ones(1, 3),
    )
    g_on, g_off = torch.autograd.grad(out.loss, [on_new, off_new])
    assert torch.all(g_off < 0)  # gradient descent raises log-prob of teacher tokens
    assert torch.all(g_on > 0)  # and lowers the failing student tokens
