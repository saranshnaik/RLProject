import numpy as np
import pytest
import torch

from rlagent.agents.scratch.buffers import ReplayBuffer, RolloutBuffer


def _reference_gae(rewards, values, dones, last_value, gamma, lam):
    """Plain-Python GAE for one env, written independently of the vectorised version."""
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    return np.array(advantages, dtype=np.float32)


def test_gae_matches_hand_computed_values():
    # One env, 3 steps, gamma=0.5, lambda=0.5, no episode ends.
    buf = RolloutBuffer(3, 1, 1, torch.device("cpu"))
    rewards, values = [1.0, 0.0, 2.0], [0.5, 1.0, 0.0]
    for r, v in zip(rewards, values, strict=True):
        buf.add(np.zeros((1, 1)), [0], [0.0], [r], [False], [v])
    buf.compute_gae(np.array([1.0], dtype=np.float32), gamma=0.5, gae_lambda=0.5)
    # deltas: 1+0.5*1-0.5=1.0 ; 0+0.5*0-1=-1.0 ; 2+0.5*1-0=2.5
    # A2=2.5 ; A1=-1+0.25*2.5=-0.375 ; A0=1+0.25*-0.375=0.90625
    assert buf.advantages[:, 0] == pytest.approx([0.90625, -0.375, 2.5])
    assert buf.returns[:, 0] == pytest.approx([0.90625 + 0.5, -0.375 + 1.0, 2.5])


def test_gae_cuts_bootstrapping_at_episode_end():
    rng = np.random.default_rng(0)
    n_steps, n_envs = 8, 3
    rewards = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    values = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    dones = rng.random((n_steps, n_envs)) < 0.3
    last = rng.normal(size=n_envs).astype(np.float32)

    buf = RolloutBuffer(n_steps, n_envs, 1, torch.device("cpu"))
    for t in range(n_steps):
        buf.add(
            np.zeros((n_envs, 1)),
            np.zeros(n_envs),
            np.zeros(n_envs),
            rewards[t],
            dones[t],
            values[t],
        )
    buf.compute_gae(last, gamma=0.99, gae_lambda=0.95)

    for e in range(n_envs):
        expected = _reference_gae(rewards[:, e], values[:, e], dones[:, e], last[e], 0.99, 0.95)
        assert buf.advantages[:, e] == pytest.approx(expected, abs=1e-5)


def test_minibatches_cover_every_sample_once():
    buf = RolloutBuffer(5, 2, 3, torch.device("cpu"))
    for t in range(5):
        buf.add(np.full((2, 3), t), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    seen = 0
    for obs, *_ in buf.minibatches(4, np.random.default_rng(0)):
        seen += len(obs)
    assert seen == 10


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(4, 2, torch.device("cpu"))
    for i in range(6):
        buf.add(np.full((1, 2), i), np.full((1, 2), i + 1), [i], [float(i)], [i % 2 == 0])
    assert len(buf) == 4
    # Oldest two (0, 1) were overwritten by 4, 5.
    assert sorted(buf.actions.tolist()) == [2, 3, 4, 5]
    obs, actions, rewards, next_obs, dones = buf.sample(16, np.random.default_rng(0))
    assert obs.shape == (16, 2)
    assert torch.equal(next_obs, obs + 1)
    assert set(actions.tolist()) <= {2, 3, 4, 5}
