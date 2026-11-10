"""Shared plumbing for the from-scratch agents.

They duck-type the parts of the SB3 interface the rest of the project relies on:
``predict``, ``learn``, ``save``, ``set_logger``, ``get_vec_normalize_env``, ``num_timesteps``.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


class ScratchAgent:
    algo_name = ""

    def __init__(
        self,
        env: VecEnv | None,
        obs_dim: int,
        n_actions: int,
        net_arch: Sequence[int],
        seed: int,
        device: str = "cpu",
    ):
        self.env = env
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net_arch = list(net_arch)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.num_timesteps = 0
        self.logger = Logger(None, [])
        self.ep_info: deque[tuple[float, int]] = deque(maxlen=100)
        self._start_time = time.time()

    # -- SB3-compatible surface -------------------------------------------------------------
    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    def get_vec_normalize_env(self) -> VecNormalize | None:
        env = self.env
        while isinstance(env, VecEnvWrapper):
            if isinstance(env, VecNormalize):
                return env
            env = env.venv
        return None

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        raise NotImplementedError

    def learn(self, total_timesteps: int, callback=None) -> ScratchAgent:
        raise NotImplementedError

    # -- persistence ------------------------------------------------------------------------
    def _state(self) -> dict:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        payload = {
            "algo": self.algo_name,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "net_arch": self.net_arch,
            "state": self._state(),
        }
        torch.save(payload, str(path))

    # -- helpers ----------------------------------------------------------------------------
    def _tensor(self, obs) -> torch.Tensor:
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)

    def _track_episodes(self, infos: list[dict]) -> None:
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.ep_info.append((float(episode["r"]), int(episode["l"])))

    def _record_common(self, steps_this_run: int) -> None:
        """Log SB3-named rollout/time keys."""
        if self.ep_info:
            rewards, lengths = zip(*self.ep_info, strict=True)
            self.logger.record("rollout/ep_rew_mean", float(np.mean(rewards)))
            self.logger.record("rollout/ep_len_mean", float(np.mean(lengths)))
        elapsed = max(time.time() - self._start_time, 1e-8)
        self.logger.record("time/fps", int(steps_this_run / elapsed))
        self.logger.record("time/time_elapsed", int(elapsed))
        self.logger.record("time/total_timesteps", self.num_timesteps)
