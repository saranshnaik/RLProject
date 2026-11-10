"""PPO (clipped surrogate, GAE) for discrete actions, written from scratch.

Follows 'The 37 Implementation Details of PPO': orthogonal init, Adam eps=1e-5, per-minibatch
advantage normalisation, gradient clipping, LR annealing, and time-limit bootstrapping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rlagent.agents.scratch.base import ScratchAgent
from rlagent.agents.scratch.buffers import RolloutBuffer
from rlagent.agents.scratch.networks import ActorCritic
from rlagent.agents.scratch.schedules import Schedule, value_at


@dataclass
class PPOHParams:
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Schedule = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: Schedule = 3e-4
    normalize_advantage: bool = True
    target_kl: float | None = None


class ScratchPPO(ScratchAgent):
    algo_name = "ppo"

    def __init__(self, env, obs_dim, n_actions, net_arch, seed, device="cpu", hparams=None):
        super().__init__(env, obs_dim, n_actions, net_arch, seed, device)
        self.hp = hparams or PPOHParams()
        self.net = ActorCritic(obs_dim, n_actions, net_arch).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, eps=1e-5)
        self.n_updates = 0

    # -- inference --------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        dist = self.net.dist(self._tensor(obs))
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions.cpu().numpy(), None

    @torch.no_grad()
    def action_probs(self, obs) -> np.ndarray:
        return self.net.dist(self._tensor(obs)).probs.cpu().numpy()

    def _state(self) -> dict:
        return {"net": self.net.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str = "cpu") -> ScratchPPO:
        agent = cls(None, ckpt["obs_dim"], ckpt["n_actions"], ckpt["net_arch"], 0, device)
        agent.net.load_state_dict(ckpt["state"]["net"])
        return agent

    # -- training ---------------------------------------------------------------------------
    def learn(self, total_timesteps: int, callback=None) -> ScratchPPO:
        env, hp = self.env, self.hp
        buffer = RolloutBuffer(hp.n_steps, env.num_envs, self.obs_dim, self.device)
        obs = env.reset()
        self._start_time = time.time()
        start_steps, iteration = self.num_timesteps, 0
        while self.num_timesteps < total_timesteps:
            remaining = 1.0 - self.num_timesteps / total_timesteps
            lr = value_at(hp.learning_rate, remaining)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            buffer.reset()
            obs = self._collect_rollout(obs, buffer)
            with torch.no_grad():
                last_values = self.net.value(self._tensor(obs)).cpu().numpy()
            buffer.compute_gae(last_values, hp.gamma, hp.gae_lambda)
            stats = self._update(buffer, value_at(hp.clip_range, remaining))

            iteration += 1
            self.logger.record("time/iterations", iteration)
            self._record_common(self.num_timesteps - start_steps)
            self.logger.record("train/learning_rate", lr)
            for key, val in stats.items():
                self.logger.record(f"train/{key}", val)
            if callback is not None:
                callback(self)
            self.logger.dump(self.num_timesteps)
        return self

    def _collect_rollout(self, obs: np.ndarray, buffer: RolloutBuffer) -> np.ndarray:
        env, hp = self.env, self.hp
        for _ in range(hp.n_steps):
            with torch.no_grad():
                t_obs = self._tensor(obs)
                dist = self.net.dist(t_obs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.net.value(t_obs)
            new_obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            self.num_timesteps += env.num_envs
            self._track_episodes(infos)

            # Time-limit truncation is not a real terminal state: bootstrap from V(final obs).
            rewards = np.array(rewards, dtype=np.float32)
            truncated = [
                i
                for i, info in enumerate(infos)
                if dones[i] and info.get("TimeLimit.truncated", False)
            ]
            if truncated:
                final_obs = np.stack([infos[i]["terminal_observation"] for i in truncated])
                with torch.no_grad():
                    final_values = self.net.value(self._tensor(final_obs)).cpu().numpy()
                rewards[truncated] += hp.gamma * final_values

            buffer.add(
                obs,
                actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                rewards,
                dones,
                values.cpu().numpy(),
            )
            obs = new_obs
        return obs

    def _update(self, buffer: RolloutBuffer, clip_range: float) -> dict[str, float]:
        hp = self.hp
        pg_losses, value_losses, entropy_losses, losses = [], [], [], []
        clip_fractions, approx_kls = [], []
        for _ in range(hp.n_epochs):
            epoch_kls = []
            for obs, actions, old_log_probs, advantages, returns in buffer.minibatches(
                hp.batch_size, self.rng
            ):
                log_probs, entropy, values = self.net.evaluate_actions(obs, actions)
                if hp.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_ratio = log_probs - old_log_probs
                ratio = log_ratio.exp()
                pg_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()
                loss = pg_loss + hp.ent_coef * entropy_loss + hp.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), hp.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fractions.append(((ratio - 1).abs() > clip_range).float().mean().item())
                epoch_kls.append(kl)
                approx_kls.append(kl)
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                losses.append(loss.item())
                self.n_updates += 1
            if hp.target_kl is not None and np.mean(epoch_kls) > 1.5 * hp.target_kl:
                break

        returns, values = buffer.returns.flatten(), buffer.values.flatten()
        var_returns = np.var(returns)
        explained = float("nan") if var_returns == 0 else 1 - np.var(returns - values) / var_returns
        return {
            "loss": float(np.mean(losses)),
            "policy_gradient_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": float(explained),
            "n_updates": self.n_updates,
        }
