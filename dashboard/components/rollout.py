"""Load agents and play Warehouse-v0 episodes for the dashboard (no Streamlit here).

Inference runs on CPU. Episodes are played on the raw ``WarehouseEnv`` (not a VecEnv), so there is
no auto-reset: the final frame of an episode is the real terminal state. Each ``Frame`` holds the
rendered image as PNG bytes (small enough to cache dozens of episodes), the agent's action
distribution for that state, and the reward that led into it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from rlagent.agents.baselines import GreedyAgent, OptimalPlannerAgent, RandomAgent
from rlagent.agents.factory import load_agent
from rlagent.config import ExperimentConfig
from rlagent.envs.warehouse.env import WarehouseEnv
from rlagent.envs.warehouse.rewards import COMPONENTS
from rlagent.training.env_factory import env_kwargs
from rlagent.utils.run_io import load_saved_config, resolve_artifacts

ACTION_NAMES = ("Up", "Down", "Left", "Right", "Pick", "Drop")
BASELINES = ("random", "greedy", "optimal")
SPLIT_BASE = {"train": WarehouseEnv.TRAIN_SEED_BASE, "test": WarehouseEnv.TEST_SEED_BASE}
SPLIT_SPAN = WarehouseEnv.SEED_SPAN


def layout_seed_for(split: str, index: int) -> int:
    """Layout seed of the ``index``-th layout of a split (test layouts are never trained on)."""
    return SPLIT_BASE[split] + int(index)


# -- data classes ------------------------------------------------------------------------------
@dataclass
class Frame:
    png: bytes
    action: int | None  # action taken from this state (None on the terminal frame)
    probs: np.ndarray | None  # action probabilities or Q-values for this state
    reward: float  # reward received on arriving at this state (0 on the first frame)
    total_return: float
    pos: tuple[int, int]
    n_carried: int
    n_delivered: int
    valid: bool  # whether the action that led here was valid
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class Episode:
    frames: list[Frame]
    actor: str
    layout_seed: int
    prob_label: str
    success: bool
    steps: int
    total_return: float
    optimal_steps: int | None
    efficiency: float | None
    invalid_actions: int
    component_totals: dict[str, float]

    @property
    def positions(self) -> list[tuple[int, int]]:
        return [f.pos for f in self.frames]


def encode_png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def decode_png(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def frames_to_gif(frames: list[Frame], fps: int = 6) -> bytes:
    """Encode an episode as a looping GIF (pauses on the last frame)."""
    images = [
        Image.open(io.BytesIO(f.png)).convert("P", palette=Image.Palette.ADAPTIVE) for f in frames
    ]
    durations = [int(1000 / fps)] * len(images)
    durations[-1] = 1500
    buf = io.BytesIO()
    images[0].save(
        buf, format="GIF", save_all=True, append_images=images[1:], duration=durations, loop=0
    )
    return buf.getvalue()


# -- loading -----------------------------------------------------------------------------------
@dataclass
class LoadedPolicy:
    name: str
    cfg: ExperimentConfig
    model: Any
    kind: str  # "maskable" | "ppo" | "dqn" | "scratch"
    normalizer: Any = None  # a VecNormalize holding frozen stats, or None

    @property
    def prob_label(self) -> str:
        return "Q-value" if self.kind == "dqn" else "Probability"


def load_policy(run_dir: str | Path, which: str = "final", name: str | None = None) -> LoadedPolicy:
    """Load a saved model (curated or from a run) on CPU, with its normalisation stats if any."""
    run_dir = Path(run_dir)
    cfg = load_saved_config(run_dir)
    if cfg.env.id != "Warehouse-v0":
        raise ValueError(f"The dashboard replays Warehouse-v0 agents only (this is {cfg.env.id})")
    model_path, vecnorm_path = resolve_artifacts(run_dir, which)  # type: ignore[arg-type]
    import torch

    torch.set_num_threads(1)
    model = load_agent(cfg.algo, model_path, "cpu")
    if cfg.algo.impl == "scratch":
        kind = "scratch"
    else:
        kind = {"maskable_ppo": "maskable", "ppo": "ppo", "dqn": "dqn"}[cfg.algo.name]
    normalizer = None
    if cfg.env.vec_normalize and vecnorm_path is not None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        dummy = DummyVecEnv([lambda: build_env(cfg)])
        normalizer = VecNormalize.load(str(vecnorm_path), dummy)
        normalizer.training = False
        normalizer.norm_reward = False
    return LoadedPolicy(name or run_dir.name, cfg, model, kind, normalizer)


def build_env(
    cfg: ExperimentConfig,
    split: str = "test",
    overrides: dict[str, Any] | None = None,
    drop_padding: bool = False,
) -> WarehouseEnv:
    """A rendering env with the config's evaluation difficulty on the chosen layout split.

    ``drop_padding`` removes max_size / max_items so a baseline can run on any grid size; models
    keep them because their observation shape is fixed.
    """
    kwargs = env_kwargs(cfg, render_mode="rgb_array")
    kwargs["split"] = split
    kwargs["n_layouts"] = None
    if drop_padding:
        kwargs.pop("max_size", None)
        kwargs.pop("max_items", None)
    kwargs.update(overrides or {})
    return WarehouseEnv(**kwargs)


def model_limits(cfg: ExperimentConfig) -> tuple[int, int]:
    """(max grid size, max items) the policy's observation can represent."""
    env = build_env(cfg)
    try:
        return env.max_size, env.max_items
    finally:
        env.close()


# -- actors ------------------------------------------------------------------------------------
class Actor(Protocol):
    name: str
    prob_label: str

    def act(
        self, env: WarehouseEnv, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[int, np.ndarray | None]: ...


class PolicyActor:
    def __init__(self, policy: LoadedPolicy):
        self.policy = policy
        self.name = policy.name
        self.prob_label = policy.prob_label

    def act(self, env, obs, deterministic: bool = True):
        policy = self.policy
        batch = np.asarray(obs, dtype=np.float32)[None]
        if policy.normalizer is not None:
            batch = policy.normalizer.normalize_obs(batch)
        model = policy.model
        if policy.kind == "maskable":
            masks = env.action_masks()[None]
            action = model.predict(batch, deterministic=deterministic, action_masks=masks)[0]
        else:
            action = model.predict(batch, deterministic=deterministic)[0]
        return int(np.asarray(action).reshape(-1)[0]), action_distribution(policy, batch, env)


def action_distribution(policy: LoadedPolicy, batch: np.ndarray, env: WarehouseEnv):
    """Per-action probabilities (PPO family) or Q-values (DQN) for one observation, or None."""
    import torch

    model = policy.model
    try:
        with torch.no_grad():
            if policy.kind == "scratch":
                if hasattr(model, "action_probs"):
                    return np.asarray(model.action_probs(batch))[0]
                if hasattr(model, "q_values"):
                    return np.asarray(model.q_values(batch))[0]
                return None
            obs_t, _ = model.policy.obs_to_tensor(batch)
            if policy.kind == "maskable":
                masks = env.action_masks()[None]
                dist = model.policy.get_distribution(obs_t, action_masks=masks)
                return dist.distribution.probs.cpu().numpy()[0]
            if policy.kind == "ppo":
                return model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy()[0]
            return model.q_net(obs_t).cpu().numpy()[0]
    except Exception:  # noqa: BLE001 - the chart is optional; never fail the replay for it
        return None


class BaselineActor:
    prob_label = "Probability"

    def __init__(self, name: str, env: WarehouseEnv, seed: int = 0):
        self.name = name
        if name == "random":
            self.agent = RandomAgent(env.action_space.n, seed)
        elif name == "greedy":
            self.agent = GreedyAgent(lambda: env)
        elif name == "optimal":
            self.agent = OptimalPlannerAgent(lambda: env)
        else:
            raise ValueError(f"Unknown baseline '{name}'; choose from {BASELINES}")

    def act(self, env, obs, deterministic: bool = True):
        action = self.agent.predict(np.asarray(obs)[None])[0]
        return int(np.asarray(action).reshape(-1)[0]), None


# -- episodes ----------------------------------------------------------------------------------
def _frame(env: WarehouseEnv, action, probs, reward, total, valid, components) -> Frame:
    return Frame(
        png=encode_png(env.render()),
        action=action,
        probs=None if probs is None else np.asarray(probs, dtype=float),
        reward=float(reward),
        total_return=float(total),
        pos=(int(env.pos[0]), int(env.pos[1])),
        n_carried=int(env.n_carried),
        n_delivered=int(env.n_delivered),
        valid=bool(valid),
        components=dict(components),
    )


def play_episode(
    actor: Actor,
    env: WarehouseEnv,
    layout_seed: int,
    episode_seed: int = 0,
    deterministic: bool = True,
) -> Episode:
    """Play one full episode and return every frame with the agent's action distribution."""
    obs, _ = env.reset(seed=episode_seed, options={"layout_seed": layout_seed})
    frames: list[Frame] = []
    totals = dict.fromkeys(COMPONENTS, 0.0)
    total, reward, valid, comps = 0.0, 0.0, True, dict.fromkeys(COMPONENTS, 0.0)
    info: dict[str, Any] = {}
    terminated = truncated = False
    while not (terminated or truncated):
        action, probs = actor.act(env, obs, deterministic)
        frames.append(_frame(env, action, probs, reward, total, valid, comps))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        valid = not info.get("invalid_action", False)
        comps = info.get("reward_components", {})
        for key, value in comps.items():
            totals[key] = totals.get(key, 0.0) + value
    frames.append(_frame(env, None, None, reward, total, valid, comps))
    return Episode(
        frames=frames,
        actor=actor.name,
        layout_seed=layout_seed,
        prob_label=getattr(actor, "prob_label", "Probability"),
        success=bool(terminated),
        steps=len(frames) - 1,
        total_return=float(total),
        optimal_steps=info.get("optimal_steps"),
        efficiency=info.get("efficiency"),
        invalid_actions=int(info.get("invalid_actions", 0)),
        component_totals=totals,
    )


def oscillation_score(positions: list[tuple[int, int]]) -> float:
    """Fraction of moves that return to the cell occupied two steps earlier (back-and-forth).

    A policy that farms a non-potential shaping bonus by stepping toward and away from its goal
    scores close to 1; an efficient delivery run scores near 0.
    """
    if len(positions) < 3:
        return 0.0
    back = sum(
        1
        for i in range(2, len(positions))
        if positions[i] == positions[i - 2] and positions[i] != positions[i - 1]
    )
    return back / (len(positions) - 2)
