import numpy as np
import pytest

torch = pytest.importorskip("torch")

from apsp.policies.reinforce import ActorCritic, ReinforcePolicy, RLConfig, reinforce_loss  # noqa: E402


def test_masked_actions_have_zero_probability_and_forced_zero_grad():
    cfg = RLConfig(hidden=16)
    net = ActorCritic(38, 16)
    obs = torch.randn(3, 38)
    masks = torch.tensor([[True, False, True], [True, False, False], [True, True, True]])
    p = torch.softmax(ActorCritic.masked_logits(net.pi(obs), masks), -1)
    assert p[0, 1] == 0 and p[1, 1] == 0 and p[1, 2] == 0
    acts = torch.tensor([2, 0, 1])
    loss, _ = reinforce_loss(net, obs[1:2], masks[1:2], acts[1:2], torch.tensor([5.0]), RLConfig(c_v=0.0), 1)
    loss.backward()
    assert all(float(prm.grad.abs().sum()) == 0 for prm in net.pi.parameters())   # forced step: no policy grad
    net.zero_grad()
    loss, _ = reinforce_loss(net, obs, masks, acts, torch.tensor([1.0, 1.0, 1.0]), cfg, 3)
    loss.backward()
    assert all(torch.isfinite(prm.grad).all() for prm in net.parameters())


def test_learns_known_optimum_on_bandit():
    """Sanity env with a known optimum: action 2 always yields R=1, others 0."""
    torch.manual_seed(0)
    cfg = RLConfig(hidden=16, lr=1e-2, beta=0.0)
    net = ActorCritic(38, 16)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    obs = torch.zeros(64, 38)
    masks = torch.ones(64, 3, dtype=torch.bool)
    for _ in range(150):
        with torch.no_grad():
            a = torch.multinomial(torch.softmax(net.pi(obs), -1), 1)[:, 0]
        R = (a == 2).float()
        loss, _ = reinforce_loss(net, obs, masks, a, R, cfg, 64)
        opt.zero_grad(); loss.backward(); opt.step()
    pol = ReinforcePolicy(net, cfg, mode="greedy")
    assert pol.act(np.zeros(38, np.float32), (0, 1, 2)) == 2
    assert pol.act(np.zeros(38, np.float32), (0, 1)) in (0, 1)       # never outside the mask


@pytest.mark.slow
def test_synthetic_smoke_runs_within_cap():
    from apsp.policies.reinforce import synthetic_smoke
    s = synthetic_smoke(iterations=4, n_prompts=30)
    assert s["all_dev_spend_within_cap"]
    assert np.isfinite(s["last_loss"])
