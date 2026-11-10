"""From-scratch PyTorch DQN and PPO (discrete actions, flat vector observations)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import VecEnv

from rlagent.agents.scratch.base import ScratchAgent
from rlagent.agents.scratch.dqn import DQNHParams, ScratchDQN
from rlagent.agents.scratch.ppo import PPOHParams, ScratchPPO
from rlagent.agents.scratch.schedules import parse_hyperparams
from rlagent.config import AlgoConfig

_AGENTS = {"ppo": (ScratchPPO, PPOHParams), "dqn": (ScratchDQN, DQNHParams)}
DEFAULT_NET_ARCH = [64, 64]


def build_scratch(algo: AlgoConfig, env: VecEnv, seed: int) -> ScratchAgent:
    obs_space, act_space = env.observation_space, env.action_space
    if not (isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1):
        raise ValueError(f"Scratch agents need flat Box observations, got {obs_space}")
    if not isinstance(act_space, gym.spaces.Discrete):
        raise ValueError(f"Scratch agents need Discrete actions, got {act_space}")
    unknown = set(algo.policy_kwargs) - {"net_arch"}
    if unknown:
        raise ValueError(f"Unsupported policy_kwargs for scratch agents: {sorted(unknown)}")

    agent_cls, hparams_cls = _AGENTS[algo.name]
    return agent_cls(
        env,
        obs_dim=obs_space.shape[0],
        n_actions=int(act_space.n),
        net_arch=algo.policy_kwargs.get("net_arch", DEFAULT_NET_ARCH),
        seed=seed,
        device=algo.device,
        hparams=hparams_cls(**parse_hyperparams(algo.hyperparams)),
    )


def load_scratch(path: str | Path, device: str = "cpu") -> ScratchAgent:
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    agent_cls, _ = _AGENTS[ckpt["algo"]]
    return agent_cls.from_checkpoint(ckpt, device)
