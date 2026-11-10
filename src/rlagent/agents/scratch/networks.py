"""MLP building blocks for the from-scratch agents (flat vector observations only)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: type[nn.Module],
    orthogonal: bool = False,
    out_gain: float = 1.0,
) -> nn.Sequential:
    """Fully connected net. With ``orthogonal`` the init follows 'The 37 PPO details':
    gain sqrt(2) on hidden layers and ``out_gain`` on the output layer, zero biases."""
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        layers += [nn.Linear(last, width), activation()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    net = nn.Sequential(*layers)
    if orthogonal:
        linears = [m for m in net if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            gain = out_gain if i == len(linears) - 1 else math.sqrt(2)
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
    return net


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (tanh), as in SB3's default PPO policy."""

    def __init__(self, obs_dim: int, n_actions: int, net_arch: Sequence[int] = (64, 64)):
        super().__init__()
        self.actor = mlp(obs_dim, net_arch, n_actions, nn.Tanh, orthogonal=True, out_gain=0.01)
        self.critic = mlp(obs_dim, net_arch, 1, nn.Tanh, orthogonal=True, out_gain=1.0)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.dist(obs)
        return dist.log_prob(actions), dist.entropy(), self.value(obs)


class QNetwork(nn.Module):
    """MLP Q-function (ReLU, PyTorch default init like SB3's DQN)."""

    def __init__(self, obs_dim: int, n_actions: int, net_arch: Sequence[int] = (64, 64)):
        super().__init__()
        self.net = mlp(obs_dim, net_arch, n_actions, nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
