"""Build / load Stable Baselines3 models from an AlgoConfig."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from rlagent.config import AlgoConfig


@dataclass
class LinearSchedule:
    """progress_remaining (1 -> 0) -> initial * progress_remaining. Picklable, unlike a lambda."""

    initial: float

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial


def _algo_class(name: str) -> type[BaseAlgorithm]:
    if name == "ppo":
        return PPO
    if name == "dqn":
        return DQN
    if name == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO
    raise ValueError(f"Unknown algorithm '{name}'")


def parse_hyperparams(hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Turn 'lin_<x>' strings into linear schedules; leave everything else untouched."""
    parsed: dict[str, Any] = {}
    for key, value in hyperparams.items():
        if isinstance(value, str) and value.startswith("lin_"):
            value = LinearSchedule(float(value[4:]))
        parsed[key] = value
    return parsed


def build_model(algo: AlgoConfig, env: VecEnv, seed: int) -> BaseAlgorithm:
    if algo.impl != "sb3":
        raise NotImplementedError("From-scratch agents arrive in Phase 2")
    cls = _algo_class(algo.name)
    return cls(
        algo.policy,
        env,
        seed=seed,
        device=algo.device,
        policy_kwargs=dict(algo.policy_kwargs) or None,
        verbose=0,
        **parse_hyperparams(algo.hyperparams),
    )


def load_model(algo: AlgoConfig, path: str | Path, device: str | None = None) -> BaseAlgorithm:
    if algo.impl != "sb3":
        raise NotImplementedError("From-scratch agents arrive in Phase 2")
    return _algo_class(algo.name).load(str(path), device=device or algo.device)
