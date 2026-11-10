"""DQN written from scratch: target network, Huber loss, gradient clipping, optional Double DQN.

Time-limit truncations are stored as non-terminal so the target still bootstraps through them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rlagent.agents.scratch.base import ScratchAgent
from rlagent.agents.scratch.buffers import ReplayBuffer
from rlagent.agents.scratch.networks import QNetwork
from rlagent.agents.scratch.schedules import Schedule, value_at

LOG_EVERY_STEPS = 2000


@dataclass
class DQNHParams:
    learning_rate: Schedule = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 100
    batch_size: int = 32
    tau: float = 1.0  # 1.0 = hard target update
    gamma: float = 0.99
    train_freq: int = 4  # env steps (per env) between gradient phases
    gradient_steps: int = 1  # -1 = as many as env steps collected since the last phase
    target_update_interval: int = 1000  # in total env steps
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.1
    max_grad_norm: float = 10.0
    double_dqn: bool = True


class ScratchDQN(ScratchAgent):
    algo_name = "dqn"

    def __init__(self, env, obs_dim, n_actions, net_arch, seed, device="cpu", hparams=None):
        super().__init__(env, obs_dim, n_actions, net_arch, seed, device)
        self.hp = hparams or DQNHParams()
        self.q_net = QNetwork(obs_dim, n_actions, net_arch).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, net_arch).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.exploration_rate = self.hp.exploration_initial_eps
        self.n_updates = 0

    # -- inference --------------------------------------------------------------------------
    @torch.no_grad()
    def q_values(self, obs) -> np.ndarray:
        return self.q_net(self._tensor(obs)).cpu().numpy()

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        actions = self.q_values(obs).argmax(-1)
        if not deterministic:
            explore = self.rng.random(len(actions)) < self.exploration_rate
            random_actions = self.rng.integers(0, self.n_actions, size=len(actions))
            actions = np.where(explore, random_actions, actions)
        return actions, None

    def _state(self) -> dict:
        return {"q_net": self.q_net.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str = "cpu") -> ScratchDQN:
        agent = cls(None, ckpt["obs_dim"], ckpt["n_actions"], ckpt["net_arch"], 0, device)
        agent.q_net.load_state_dict(ckpt["state"]["q_net"])
        agent.target_net.load_state_dict(ckpt["state"]["q_net"])
        return agent

    # -- training ---------------------------------------------------------------------------
    def _epsilon(self, progress: float) -> float:
        hp = self.hp
        if hp.exploration_fraction <= 0 or progress >= hp.exploration_fraction:
            return hp.exploration_final_eps
        slope = (hp.exploration_final_eps - hp.exploration_initial_eps) / hp.exploration_fraction
        return hp.exploration_initial_eps + progress * slope

    def learn(self, total_timesteps: int, callback=None) -> ScratchDQN:
        env, hp = self.env, self.hp
        n_envs = env.num_envs
        buffer = ReplayBuffer(hp.buffer_size, self.obs_dim, self.device)
        obs = env.reset()
        self._start_time = time.time()
        start_steps = self.num_timesteps
        last_target, last_log = self.num_timesteps, self.num_timesteps
        steps_since_train = 0
        train_stats: dict[str, list[float]] = {"loss": [], "q_mean": [], "td_error": []}

        while self.num_timesteps < total_timesteps:
            progress = self.num_timesteps / total_timesteps
            self.exploration_rate = self._epsilon(progress)
            lr = value_at(hp.learning_rate, 1.0 - progress)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            actions, _ = self.predict(obs, deterministic=False)
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += n_envs
            steps_since_train += 1
            self._track_episodes(infos)

            next_obs, terminated = new_obs.copy(), np.array(dones, dtype=bool)
            for i in range(n_envs):
                if dones[i]:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if infos[i].get("TimeLimit.truncated", False):
                        terminated[i] = False  # truncated, not terminal: keep bootstrapping
            buffer.add(obs, next_obs, actions, rewards, terminated)
            obs = new_obs

            if self.num_timesteps // hp.target_update_interval > (
                last_target // hp.target_update_interval
            ):
                self._update_target()
            last_target = self.num_timesteps

            if steps_since_train >= hp.train_freq and self.num_timesteps > hp.learning_starts:
                n_grad = steps_since_train * n_envs if hp.gradient_steps < 0 else hp.gradient_steps
                for _ in range(n_grad):
                    for key, val in self._train_step(buffer).items():
                        train_stats[key].append(val)
                steps_since_train = 0

            if callback is not None:
                callback(self)

            if self.num_timesteps // LOG_EVERY_STEPS > last_log // LOG_EVERY_STEPS:
                self._log(train_stats, self.num_timesteps - start_steps)
                self.logger.dump(self.num_timesteps)
                last_log = self.num_timesteps
        return self

    def _update_target(self) -> None:
        tau = self.hp.tau
        with torch.no_grad():
            for target, online in zip(
                self.target_net.parameters(), self.q_net.parameters(), strict=True
            ):
                target.mul_(1 - tau).add_(online, alpha=tau)

    def _train_step(self, buffer: ReplayBuffer) -> dict[str, float]:
        hp = self.hp
        obs, actions, rewards, next_obs, dones = buffer.sample(hp.batch_size, self.rng)
        with torch.no_grad():
            if hp.double_dqn:
                best = self.q_net(next_obs).argmax(1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, best).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(1).values
            target_q = rewards + (1.0 - dones) * hp.gamma * next_q
        q_all = self.q_net(obs)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), hp.max_grad_norm)
        self.optimizer.step()
        self.n_updates += 1
        return {
            "loss": loss.item(),
            "q_mean": q_all.mean().item(),
            "td_error": (current_q - target_q).abs().mean().item(),
        }

    def _log(self, stats: dict[str, list[float]], steps_this_run: int) -> None:
        self._record_common(steps_this_run)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)
        self.logger.record("train/n_updates", self.n_updates)
        for key, values in stats.items():
            if values:
                self.logger.record(f"train/{key}", float(np.mean(values)))
            values.clear()
