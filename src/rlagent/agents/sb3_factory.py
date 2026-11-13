"""Build / load Stable Baselines3 models from an AlgoConfig."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from rlagent.agents.feature_extractors import FEATURE_EXTRACTORS
from rlagent.agents.scratch.schedules import LinearSchedule, parse_hyperparams
from rlagent.config import AlgoConfig

__all__ = ["LinearSchedule", "build_model", "load_model", "parse_hyperparams"]


def _algo_class(name: str) -> type[BaseAlgorithm]:
    if name == "ppo":
        return PPO
    if name == "dqn":
        return DQN
    if name == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO
    raise ValueError(f"Unknown algorithm '{name}'")


def _policy_kwargs(algo: AlgoConfig) -> dict | None:
    """YAML-friendly policy kwargs: a feature extractor may be given by class name."""
    kwargs = dict(algo.policy_kwargs)
    extractor = kwargs.get("features_extractor_class")
    if isinstance(extractor, str):
        if extractor not in FEATURE_EXTRACTORS:
            known = sorted(FEATURE_EXTRACTORS)
            raise ValueError(f"Unknown features_extractor_class '{extractor}'; known: {known}")
        kwargs["features_extractor_class"] = FEATURE_EXTRACTORS[extractor]
    return kwargs or None


def build_model(algo: AlgoConfig, env: VecEnv, seed: int) -> BaseAlgorithm:
    cls = _algo_class(algo.name)
    return cls(
        algo.policy,
        env,
        seed=seed,
        device=algo.device,
        policy_kwargs=_policy_kwargs(algo),
        verbose=0,
        **parse_hyperparams(algo.hyperparams),
    )


def load_model(algo: AlgoConfig, path: str | Path, device: str | None = None) -> BaseAlgorithm:
    return _algo_class(algo.name).load(str(path), device=device or algo.device)
