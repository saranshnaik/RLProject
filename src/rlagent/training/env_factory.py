"""Vectorised environment construction from an ExperimentConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from rlagent.config import ExperimentConfig


def _gamma(cfg: ExperimentConfig) -> float:
    return float(cfg.algo.hyperparams.get("gamma", 0.99))


def env_kwargs(
    cfg: ExperimentConfig, training: bool = False, render_mode: str | None = None
) -> dict[str, Any]:
    """Env kwargs for training or evaluation.

    With a curriculum, ``max_size`` / ``max_items`` are padded to the largest stage (so the
    observation shape never changes), training starts from ``stages[0]``, and evaluation always
    uses the full target difficulty.
    """
    kwargs = dict(cfg.env.kwargs)
    if cfg.curriculum is not None:
        defaults = {"height": 8, "width": 8, "n_items": 1}
        stages = [{**defaults, **kwargs, **stage} for stage in cfg.curriculum.stages]
        sizes = [max(s["height"], s["width"]) for s in [*stages, {**defaults, **kwargs}]]
        items = [s["n_items"] for s in [*stages, {**defaults, **kwargs}]]
        kwargs.setdefault("max_size", max(sizes))
        kwargs.setdefault("max_items", max(items))
        if training:
            kwargs.update(cfg.curriculum.stages[0])
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return kwargs


def build_train_env(cfg: ExperimentConfig, seed: int) -> VecEnv:
    env_cls = SubprocVecEnv if cfg.env.subproc else DummyVecEnv
    venv = make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        seed=seed,
        env_kwargs=env_kwargs(cfg, training=True),
        vec_env_cls=env_cls,
    )
    if cfg.env.vec_normalize:
        venv = VecNormalize(
            venv,
            norm_obs=cfg.env.norm_obs,
            norm_reward=cfg.env.norm_reward,
            gamma=_gamma(cfg),
        )
    return venv


def build_eval_env(
    cfg: ExperimentConfig,
    seed: int,
    render_mode: str | None = None,
    vecnorm_path: str | Path | None = None,
) -> VecEnv:
    """Single-env evaluation VecEnv. Normalisation stats are frozen and rewards stay raw."""
    venv = make_vec_env(
        cfg.env.id, n_envs=1, seed=seed, env_kwargs=env_kwargs(cfg, render_mode=render_mode)
    )
    if cfg.env.vec_normalize:
        if vecnorm_path is not None and Path(vecnorm_path).exists():
            venv = VecNormalize.load(str(vecnorm_path), venv)
        else:
            venv = VecNormalize(venv, norm_obs=cfg.env.norm_obs, norm_reward=False)
        venv.training = False
        venv.norm_reward = False
    return venv
