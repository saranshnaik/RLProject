"""Callback wiring for training runs."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecEnv

from rlagent.config import CurriculumConfig, ExperimentConfig
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


class WarehouseMetricsCallback(BaseCallback):
    """Logs training-episode metrics read from Warehouse-v0 ``info`` dicts.

    ``rollout/success_rate``, ``rollout/efficiency`` (0 for failed episodes),
    ``rollout/invalid_actions`` and ``reward/<component>``: the mean per-episode total of each
    reward component, which shows *what* the agent is earning (essential for spotting reward
    hacking, where shaping dominates the task rewards).
    """

    def __init__(self, window: int = 100):
        super().__init__()
        self.episodes: deque[dict[str, float]] = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode_reward_components" in info:  # present only on the final step
                row = {
                    "success": float(info["is_success"]),
                    "efficiency": float(info["efficiency"]),
                    "invalid_actions": float(info["invalid_actions"]),
                }
                row.update({f"reward/{k}": v for k, v in info["episode_reward_components"].items()})
                self.episodes.append(row)
        return True

    def _on_rollout_end(self) -> None:
        if not self.episodes:
            return
        keys = self.episodes[0].keys()
        means = {k: float(np.mean([e[k] for e in self.episodes])) for k in keys}
        self.logger.record("rollout/success_rate", means.pop("success"))
        self.logger.record("rollout/efficiency", means.pop("efficiency"))
        self.logger.record("rollout/invalid_actions", means.pop("invalid_actions"))
        for key, value in means.items():
            self.logger.record(key, value)


class CurriculumCallback(BaseCallback):
    """Promotes the training env through difficulty stages based on training success rate."""

    def __init__(self, curriculum: CurriculumConfig, verbose: bool = True):
        super().__init__()
        self.curriculum, self.verbose = curriculum, verbose
        self.stage = 0
        self.recent: deque[float] = deque(maxlen=curriculum.window)
        self.episodes_in_stage = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "is_success" in info:  # episode end
                self.recent.append(float(info["is_success"]))
                self.episodes_in_stage += 1
        last_stage = len(self.curriculum.stages) - 1
        if (
            self.stage < last_stage
            and self.episodes_in_stage >= self.curriculum.window
            and np.mean(self.recent) >= self.curriculum.threshold
        ):
            self.stage += 1
            self.training_env.env_method("set_difficulty", **self.curriculum.stages[self.stage])
            if self.verbose:
                print(
                    f"Curriculum: stage {self.stage}/{last_stage} at "
                    f"{self.num_timesteps} steps -> {self.curriculum.stages[self.stage]}"
                )
            self.recent.clear()
            self.episodes_in_stage = 0
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("curriculum/stage", self.stage)
        if self.recent:
            self.logger.record("curriculum/stage_success", float(np.mean(self.recent)))


class ScratchEvalHook:
    """Plays the role of EvalCallback + CheckpointCallback for from-scratch agents.

    Called by the agent's ``learn`` after every step / iteration. Evaluates every
    ``eval_freq`` env steps, logs SB3-named ``eval/*`` keys, and keeps the best model.
    """

    def __init__(self, cfg: ExperimentConfig, run_dir: Path, eval_env: VecEnv, verbose: bool):
        self.cfg, self.run_dir, self.eval_env, self.verbose = cfg, run_dir, eval_env, verbose
        self.next_eval = cfg.train.eval_freq
        self.next_ckpt = cfg.train.checkpoint_freq or None
        self.best_mean = -float("inf")

    def __call__(self, agent) -> None:
        steps = agent.num_timesteps
        if steps >= self.next_eval:
            while self.next_eval <= steps:
                self.next_eval += self.cfg.train.eval_freq
            self._evaluate(agent)
        if self.next_ckpt is not None and steps >= self.next_ckpt:
            while self.next_ckpt <= steps:
                self.next_ckpt += self.cfg.train.checkpoint_freq
            self._checkpoint(agent)

    def _evaluate(self, agent) -> None:
        from stable_baselines3.common.vec_env import sync_envs_normalization

        from rlagent.evaluation.evaluate import evaluate_agent

        venv = agent.get_vec_normalize_env()
        if venv is not None:
            sync_envs_normalization(venv, self.eval_env)
        result = evaluate_agent(agent, self.eval_env, self.cfg.train.n_eval_episodes)
        agent.logger.record("eval/mean_reward", result["mean_return"])
        agent.logger.record("eval/mean_ep_length", result["mean_length"])
        if result["success_rate"] is not None:
            agent.logger.record("eval/success_rate", result["success_rate"])
        if self.verbose:
            print(
                f"Eval num_timesteps={agent.num_timesteps}, "
                f"episode_reward={result['mean_return']:.2f} +/- {result['std_return']:.2f}"
            )
        if result["mean_return"] > self.best_mean:
            self.best_mean = result["mean_return"]
            agent.save(self.run_dir / "best_model.pt")
            if venv is not None:
                venv.save(str(self.run_dir / VECNORM_BEST))

    def _checkpoint(self, agent) -> None:
        folder = self.run_dir / "checkpoints"
        folder.mkdir(exist_ok=True)
        agent.save(folder / f"ckpt_{agent.num_timesteps}_steps.pt")
        venv = agent.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(folder / f"ckpt_vecnormalize_{agent.num_timesteps}_steps.pkl"))


def make_callbacks(
    cfg: ExperimentConfig,
    run_dir: Path,
    eval_env: VecEnv,
    verbose: bool = True,
    after_eval: BaseCallback | None = None,
) -> CallbackList:
    n_envs = cfg.env.n_envs
    eval_cls: type[EvalCallback] = EvalCallback
    eval_kwargs: dict = {}
    if cfg.algo.name == "maskable_ppo":  # evaluation must respect the action masks too
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

        eval_cls, eval_kwargs = MaskableEvalCallback, {"use_masking": True}
    callbacks: list[BaseCallback] = [
        eval_cls(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=max(cfg.train.eval_freq // n_envs, 1),
            n_eval_episodes=cfg.train.n_eval_episodes,
            deterministic=True,
            verbose=int(verbose),
            callback_on_new_best=SaveVecNormalizeOnBest(run_dir / VECNORM_BEST),
            callback_after_eval=after_eval,
            **eval_kwargs,
        )
    ]
    if cfg.env.id == "Warehouse-v0":
        callbacks.append(WarehouseMetricsCallback())
    if cfg.curriculum is not None:
        callbacks.append(CurriculumCallback(cfg.curriculum, verbose))
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
