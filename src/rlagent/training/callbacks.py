"""Callback wiring for training runs."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecEnv

from rlagent.config import ExperimentConfig
from rlagent.utils.run_io import VECNORM_BEST


class SaveVecNormalizeOnBest(BaseCallback):
    """Triggered by EvalCallback on a new best model, so normalisation stats match the weights."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(self.path))
        return True


def make_callbacks(
    cfg: ExperimentConfig, run_dir: Path, eval_env: VecEnv, verbose: bool = True
) -> CallbackList:
    n_envs = cfg.env.n_envs
    callbacks: list[BaseCallback] = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=max(cfg.train.eval_freq // n_envs, 1),
            n_eval_episodes=cfg.train.n_eval_episodes,
            deterministic=True,
            verbose=int(verbose),
            callback_on_new_best=SaveVecNormalizeOnBest(run_dir / VECNORM_BEST),
        )
    ]
    if cfg.train.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(cfg.train.checkpoint_freq // n_envs, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ckpt",
                save_vecnormalize=cfg.env.vec_normalize,
            )
        )
    return CallbackList(callbacks)
