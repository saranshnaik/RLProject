"""Replay buffer (DQN) and rollout buffer with GAE (PPO)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


class ReplayBuffer:
    """Circular buffer. ``dones`` must mean *terminated* (not truncated) so that time-limit
    endings still bootstrap from the next state."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, next_obs, actions, rewards, dones) -> None:
        """Add one transition per env (all arguments have a leading n_envs dimension)."""
        for i in range(len(actions)):
            self.obs[self.pos] = obs[i]
            self.next_obs[self.pos] = next_obs[i]
            self.actions[self.pos] = actions[i]
            self.rewards[self.pos] = rewards[i]
            self.dones[self.pos] = float(dones[i])
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        t = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return (
            t(self.obs),
            t(self.actions),
            t(self.rewards),
            t(self.next_obs),
            t(self.dones),
        )


class RolloutBuffer:
    """Fixed-size (n_steps, n_envs) on-policy storage."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, device: torch.device):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def add(self, obs, actions, log_probs, rewards, dones, values) -> None:
        """``dones[t]`` is True when the transition at step t ended an episode."""
        self.obs[self.pos] = obs
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Generalised advantage estimation. ``last_values`` is V(s_T) for the state after the
        final stored step; it is masked out wherever that step ended an episode."""
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_values = last_values if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def minibatches(self, batch_size: int, rng: np.random.Generator) -> Iterator[tuple]:
        total = self.n_steps * self.n_envs
        flat = lambda a: a.reshape(total, *a.shape[2:])  # noqa: E731
        data = [flat(a) for a in (self.obs, self.actions, self.log_probs, self.advantages)]
        data.append(flat(self.returns))
        perm = rng.permutation(total)
        for start in range(0, total, batch_size):
            idx = perm[start : start + batch_size]
            yield tuple(torch.as_tensor(a[idx], device=self.device) for a in data)
