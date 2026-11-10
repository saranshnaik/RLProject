"""Warehouse-v0: a robot picks items from shelf-side cells and delivers them to a station."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np

from rlagent.envs.warehouse import observations as obs_lib
from rlagent.envs.warehouse.layout import MOVES, Cell, Layout, generate_layout
from rlagent.envs.warehouse.planner import optimal_plan
from rlagent.envs.warehouse.renderer import HumanRenderer, render_ansi, render_rgb
from rlagent.envs.warehouse.rewards import (
    COMPONENTS,
    RewardConfig,
    potential,
    subgoal_distance,
)

AVAILABLE, CARRIED, DELIVERED = 0, 1, 2


class WarehouseEnv(gym.Env):
    """See ``docs/env_spec.md`` for the full specification.

    Actions: 0 up, 1 down, 2 left, 3 right, 4 pick, 5 drop (delivers everything carried;
    only valid at the station). The robot carries at most ``capacity`` items.
    """

    metadata = {"render_modes": ["rgb_array", "human", "ansi"], "render_fps": 8}
    UP, DOWN, LEFT, RIGHT, PICK, DROP = range(6)

    TRAIN_SEED_BASE = 0
    TEST_SEED_BASE = 10_000_000  # disjoint from every train layout seed
    SEED_SPAN = 1_000_000
    LAYOUT_CACHE_SIZE = 256

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        n_items: int = 1,
        capacity: int = 1,
        shelf_density: float = 0.7,
        n_layouts: int | None = None,
        split: str = "train",
        obs_type: str = "vector",
        max_steps: int | None = None,
        max_size: int | None = None,
        max_items: int | None = None,
        reward: RewardConfig | dict | None = None,
        shaping_gamma: float = 0.99,
        render_mode: str | None = None,
    ):
        if n_items < 1 or capacity < 1:
            raise ValueError("n_items and capacity must be >= 1")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if obs_type not in ("vector", "grid"):
            raise ValueError("obs_type must be 'vector' or 'grid'")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'")
        self.height, self.width = height, width
        self.max_size = max_size or max(height, width)
        self.max_items = max_items or n_items
        if self.max_size < max(height, width) or self.max_items < n_items:
            raise ValueError("max_size / max_items must be >= the actual size / n_items")
        if n_layouts is not None and n_layouts < 1:
            raise ValueError("n_layouts must be >= 1 or None (unlimited)")
        self.n_items, self.capacity = n_items, capacity
        self.shelf_density = shelf_density
        self.n_layouts, self.split = n_layouts, split
        self.obs_type = obs_type
        self.max_steps = max_steps or 4 * n_items * (height + width) + 20
        self.reward_cfg = (
            reward if isinstance(reward, RewardConfig) else RewardConfig(**(reward or {}))
        )
        self.shaping_gamma = shaping_gamma
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(6)
        if obs_type == "vector":
            self.observation_space = gym.spaces.Box(
                -1.0, 1.0, (obs_lib.vector_dim(self.max_items),), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                0.0, 1.0, (obs_lib.GRID_CHANNELS, self.max_size, self.max_size), dtype=np.float32
            )

        self._layouts: OrderedDict[int, Layout] = OrderedDict()
        self._human: HumanRenderer | None = None
        # Episode state (populated by reset)
        self.layout: Layout
        self.layout_seed = -1
        self.pos: Cell = (0, 0)
        self.items = np.zeros((n_items, 2), dtype=np.int64)
        self.status = np.zeros(n_items, dtype=np.int8)
        self.t = 0
        self.n_available = self.n_carried = self.n_delivered = 0

    # -- layouts ----------------------------------------------------------------------------
    def _sample_layout_seed(self) -> int:
        base = self.TRAIN_SEED_BASE if self.split == "train" else self.TEST_SEED_BASE
        pool = self.SEED_SPAN if self.n_layouts is None else self.n_layouts
        return base + int(self.np_random.integers(0, pool))

    def _get_layout(self, seed: int) -> Layout:
        layout = self._layouts.get(seed)
        if layout is None:
            layout = generate_layout(self.height, self.width, self.shelf_density, seed)
            if len(layout.free_cells) < self.n_items:
                raise ValueError(
                    f"Layout {seed} has {len(layout.free_cells)} free cells for {self.n_items} "
                    "items; lower shelf_density or raise the grid size"
                )
            self._layouts[seed] = layout
            if len(self._layouts) > self.LAYOUT_CACHE_SIZE:
                self._layouts.popitem(last=False)
        return layout

    # -- gym API ----------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.layout_seed = int(options.get("layout_seed", self._sample_layout_seed()))
        self.layout = self._get_layout(self.layout_seed)
        if "items" in options:
            cells = [tuple(map(int, c)) for c in options["items"]]
            free = set(self.layout.free_cells)
            if len(set(cells)) != self.n_items or not set(cells) <= free:
                raise ValueError("options['items'] must be n_items distinct free cells")
        else:
            pool = self.layout.shelf_adjacent_cells
            if len(pool) < self.n_items:
                pool = self.layout.free_cells
            picks = self.np_random.choice(len(pool), size=self.n_items, replace=False)
            cells = [pool[int(i)] for i in picks]
        self.items = np.array(cells, dtype=np.int64)
        self.status = np.zeros(self.n_items, dtype=np.int8)
        self.pos = self.layout.station
        self.t = 0
        self.n_available, self.n_carried, self.n_delivered = self.n_items, 0, 0
        self._invalid_count = 0
        self._optimal: int | None = None
        self._episode_components = dict.fromkeys(COMPONENTS, 0.0)

        self.padded_shelves = np.pad(
            self.layout.shelves.astype(np.float32), obs_lib.PATCH_RADIUS, constant_values=1.0
        )
        self.shelf_plane = np.ones((self.max_size, self.max_size), dtype=np.float32)
        self.shelf_plane[: self.height, : self.width] = self.layout.shelves
        self._phi = potential(self)

        if self.render_mode == "human":
            self.render()
        return self._observation(), {"layout_seed": self.layout_seed}

    def step(self, action):
        action = int(action)
        if not 0 <= action < 6:
            raise ValueError(f"Invalid action {action}")
        cfg = self.reward_cfg
        self.t += 1
        comps = dict.fromkeys(COMPONENTS, 0.0)
        comps["step"] = cfg.step
        pre_distance = subgoal_distance(self) if cfg.shaping == "naive" and action < 4 else None
        valid = True

        if action < 4:
            dy, dx = MOVES[action]
            ny, nx = self.pos[0] + dy, self.pos[1] + dx
            if self.layout.is_free(ny, nx):
                self.pos = (ny, nx)
            else:
                valid = False
        elif action == self.PICK:
            index = self.item_index_here()
            if index is not None and self.n_carried < self.capacity:
                self.status[index] = CARRIED
                self.n_available -= 1
                self.n_carried += 1
                comps["pick"] = cfg.pick
            else:
                valid = False
        else:  # DROP
            if self.n_carried > 0 and self.pos == self.layout.station:
                delivered = self.n_carried
                self.status[self.status == CARRIED] = DELIVERED
                self.n_delivered += delivered
                self.n_carried = 0
                comps["delivery"] = cfg.delivery * delivered
            else:
                valid = False

        if not valid:
            comps["invalid"] = cfg.invalid
            self._invalid_count += 1
        terminated = self.n_delivered == self.n_items
        if terminated:
            comps["complete"] = cfg.complete
        truncated = (not terminated) and self.t >= self.max_steps

        if cfg.shaping == "pbrs":
            phi_new = 0.0 if terminated else potential(self)
            comps["shaping"] = cfg.shaping_scale * (self.shaping_gamma * phi_new - self._phi)
            self._phi = phi_new
        elif cfg.shaping == "naive" and pre_distance is not None and valid:
            if subgoal_distance(self) < pre_distance:
                comps["shaping"] = cfg.naive_bonus

        for key, value in comps.items():
            self._episode_components[key] += value
        reward = float(sum(comps.values()))
        info: dict[str, Any] = {"reward_components": comps, "invalid_action": not valid}
        if terminated or truncated:
            optimal = self.optimal_steps()
            info.update(
                is_success=terminated,
                optimal_steps=optimal,
                efficiency=(optimal / self.t) if terminated else 0.0,
                invalid_actions=self._invalid_count,
                n_delivered=self.n_delivered,
                episode_reward_components=dict(self._episode_components),
            )
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, info

    # -- helpers ----------------------------------------------------------------------------
    def item_index_here(self) -> int | None:
        for i in range(self.n_items):
            if (
                self.status[i] == AVAILABLE
                and (int(self.items[i, 0]), int(self.items[i, 1])) == self.pos
            ):
                return i
        return None

    def _observation(self) -> np.ndarray:
        if self.obs_type == "vector":
            return obs_lib.vector_observation(self)
        return obs_lib.grid_observation(self)

    def optimal_steps(self) -> int:
        """Minimum steps to complete this episode's orders from the start state (cached)."""
        if self._optimal is None:
            cells = [(int(y), int(x)) for y, x in self.items]
            self._optimal = optimal_plan(self.layout, cells, self.capacity)[0]
        return self._optimal

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid actions (for MaskablePPO)."""
        mask = np.zeros(6, dtype=bool)
        for action, (dy, dx) in enumerate(MOVES):
            mask[action] = self.layout.is_free(self.pos[0] + dy, self.pos[1] + dx)
        mask[self.PICK] = self.item_index_here() is not None and self.n_carried < self.capacity
        mask[self.DROP] = self.n_carried > 0 and self.pos == self.layout.station
        return mask

    # -- rendering --------------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return render_ansi(self)
        frame = render_rgb(self)
        if self.render_mode == "human":
            if self._human is None:
                self._human = HumanRenderer(self.metadata["render_fps"])
            self._human.draw(frame)
            return None
        return frame

    def close(self) -> None:
        if self._human is not None:
            self._human.close()
            self._human = None
