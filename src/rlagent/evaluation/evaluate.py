"""Evaluate a trained agent over N episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from rlagent.agents.factory import load_agent
from rlagent.config import ExperimentConfig
from rlagent.training.env_factory import build_eval_env
from rlagent.utils.run_io import load_saved_config, resolve_artifacts

EXTRA_KEYS = ("efficiency", "invalid_actions", "n_delivered", "optimal_steps")


def load_run(
    run_dir: str | Path,
    which: Literal["best", "final"] = "best",
    device: str | None = None,
    render_mode: str | None = None,
    seed: int = 12_345,
) -> tuple[Any, VecEnv, ExperimentConfig]:
    """Load model + a matching evaluation env (with the saved normalisation stats)."""
    cfg = load_saved_config(run_dir)
    model_path, vecnorm_path = resolve_artifacts(run_dir, which)
    model = load_agent(cfg.algo, model_path, device)
    env = build_eval_env(cfg, seed, render_mode=render_mode, vecnorm_path=vecnorm_path)
    return model, env, cfg


def evaluate_agent(model: Any, env: VecEnv, n_episodes: int, deterministic: bool = True) -> dict:
    """Roll out ``n_episodes`` on a single-env VecEnv whose envs are wrapped in Monitor."""
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    extras: dict[str, list[float]] = {key: [] for key in EXTRA_KEYS}
    obs = env.reset()
    while len(returns) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = env.step(action)
        if dones[0]:
            info = infos[0]
            episode = info["episode"]
            returns.append(float(episode["r"]))
            lengths.append(int(episode["l"]))
            if "is_success" in info:
                successes.append(float(info["is_success"]))
            for key in EXTRA_KEYS:
                if key in info:
                    extras[key].append(float(info[key]))
    result = {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)) if successes else None,
    }
    for key, values in extras.items():
        if values:
            result[f"mean_{key}"] = float(np.mean(values))
    return result


def evaluate_baseline(
    cfg: ExperimentConfig, name: str, n_episodes: int = 100, seed: int = 12_345
) -> dict:
    """Run a non-learning baseline in the config's environment."""
    from rlagent.agents.baselines import make_baseline

    env = build_eval_env(cfg, seed)
    try:
        result = evaluate_agent(make_baseline(name, env, seed), env, n_episodes)
    finally:
        env.close()
    result["baseline"] = name
    return result


def evaluate_run(
    run_dir: str | Path,
    n_episodes: int = 100,
    which: Literal["best", "final"] = "best",
    deterministic: bool = True,
    device: str | None = None,
    save: bool = True,
) -> dict:
    model, env, _ = load_run(run_dir, which, device)
    try:
        result = evaluate_agent(model, env, n_episodes, deterministic)
    finally:
        env.close()
    result["model"] = which
    result["deterministic"] = deterministic
    if save:
        (Path(run_dir) / f"eval_{which}.json").write_text(json.dumps(result, indent=2))
    return result
