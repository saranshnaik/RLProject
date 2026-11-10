"""Non-learning reference agents for Warehouse-v0.

random   uniform random actions.
greedy   always walks to the nearest available item; returns to the station when full or done.
optimal  executes the exact optimal plan from ``envs.warehouse.planner`` along shortest paths.

greedy and optimal need the warehouse layout, which is not in the observation, so they read the
underlying env directly (single-env evaluation only).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from rlagent.envs.warehouse.env import AVAILABLE, DELIVERED, WarehouseEnv
from rlagent.envs.warehouse.planner import optimal_plan

BASELINES = ("random", "greedy", "optimal")


class RandomAgent:
    def __init__(self, n_actions: int = 6, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        return self.rng.integers(0, self.n_actions, size=len(obs)), None


class _EnvAgent:
    def __init__(self, get_env: Callable[[], WarehouseEnv]):
        self._get_env = get_env

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        if len(obs) != 1:
            raise ValueError("Planner baselines support a single environment")
        return np.array([self._act(self._get_env())], dtype=np.int64), None

    def _act(self, env: WarehouseEnv) -> int:
        raise NotImplementedError

    @staticmethod
    def _walk(env: WarehouseEnv, target) -> int:
        action = env.layout.step_towards(env.pos, target)
        if action is None:
            raise RuntimeError(f"No move from {env.pos} towards {target}")
        return action


class GreedyAgent(_EnvAgent):
    def _act(self, env: WarehouseEnv) -> int:
        station = env.layout.station
        available = [i for i in range(env.n_items) if env.status[i] == AVAILABLE]
        full = env.n_carried >= env.capacity
        heading_home = env.n_carried > 0 and (full or not available)
        if heading_home and env.pos == station:
            return env.DROP
        if not full and env.item_index_here() is not None:
            return env.PICK
        if heading_home:
            return self._walk(env, station)
        dist_here = lambda i: env.layout.dist_map(tuple(map(int, env.items[i])))[env.pos]  # noqa: E731
        nearest = min(available, key=dist_here)
        return self._walk(env, tuple(map(int, env.items[nearest])))


class OptimalPlannerAgent(_EnvAgent):
    def __init__(self, get_env: Callable[[], WarehouseEnv]):
        super().__init__(get_env)
        self._trips: list[tuple[int, ...]] | None = None

    def _act(self, env: WarehouseEnv) -> int:
        if self._trips is None or env.t == 0:
            cells = [(int(y), int(x)) for y, x in env.items]
            self._trips = optimal_plan(env.layout, cells, env.capacity)[1]
        trip = next(t for t in self._trips if any(env.status[i] != DELIVERED for i in t))
        remaining = [i for i in trip if env.status[i] == AVAILABLE]
        if remaining:
            target = (int(env.items[remaining[0], 0]), int(env.items[remaining[0], 1]))
            return env.PICK if env.pos == target else self._walk(env, target)
        if env.pos == env.layout.station:
            return env.DROP
        return self._walk(env, env.layout.station)


def _warehouse_env(venv: VecEnv) -> WarehouseEnv:
    while not hasattr(venv, "envs"):
        venv = venv.venv
    return venv.envs[0].unwrapped


def make_baseline(name: str, venv: VecEnv, seed: int = 0):
    if name == "random":
        return RandomAgent(venv.action_space.n, seed)
    if name == "greedy":
        return GreedyAgent(lambda: _warehouse_env(venv))
    if name == "optimal":
        return OptimalPlannerAgent(lambda: _warehouse_env(venv))
    raise ValueError(f"Unknown baseline '{name}'; choose from {BASELINES}")
