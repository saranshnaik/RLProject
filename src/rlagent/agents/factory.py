"""Single entry point that builds / loads an agent regardless of implementation."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.vec_env import VecEnv

from rlagent.agents.sb3_factory import build_model, load_model
from rlagent.agents.scratch import build_scratch, load_scratch
from rlagent.config import AlgoConfig


def build_agent(algo: AlgoConfig, env: VecEnv, seed: int):
    if algo.impl == "scratch":
        return build_scratch(algo, env, seed)
    return build_model(algo, env, seed)


def load_agent(algo: AlgoConfig, path: str | Path, device: str | None = None):
    if algo.impl == "scratch":
        return load_scratch(path, device or algo.device)
    return load_model(algo, path, device)
