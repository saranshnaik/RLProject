"""Custom SB3 feature extractors."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SmallGridCNN(BaseFeaturesExtractor):
    """3x3 CNN for small channel-first float grids such as Warehouse-v0's ``obs_type="grid"``.

    SB3's NatureCNN uses 8x8 kernels with stride 4 and needs images of at least ~36 pixels; a
    12x12 warehouse would collapse to nothing. Padding keeps the spatial layout intact, and the
    last conv uses stride 2 to keep the linear layer small.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


FEATURE_EXTRACTORS = {"SmallGridCNN": SmallGridCNN}
