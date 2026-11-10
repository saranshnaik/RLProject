"""Observation builders for Warehouse-v0.

vector: [pos(2), carried/capacity(1), station offset(2), per item (dy, dx, available)*max_items,
         7x7 local occupancy patch(49), fraction of items left(1), fraction of time left(1)]
grid:   channel-first float planes padded to (max_size, max_size):
        shelves (padding counts as shelf), robot, available items, station,
        carried/capacity (constant), time left (constant)
Offsets are normalised by ``max_size`` so observation scale does not depend on the grid size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rlagent.envs.warehouse.env import WarehouseEnv

PATCH_RADIUS = 3
PATCH = 2 * PATCH_RADIUS + 1
GRID_CHANNELS = 6


def vector_dim(max_items: int) -> int:
    return 2 + 1 + 2 + 3 * max_items + PATCH * PATCH + 2


def vector_observation(env: WarehouseEnv) -> np.ndarray:
    size = env.max_size
    y, x = env.pos
    obs = np.zeros(vector_dim(env.max_items), dtype=np.float32)
    obs[0] = y / (size - 1)
    obs[1] = x / (size - 1)
    obs[2] = env.n_carried / env.capacity
    sy, sx = env.layout.station
    obs[3] = (sy - y) / size
    obs[4] = (sx - x) / size
    for i in range(env.n_items):
        if env.status[i] == 0:
            iy, ix = env.items[i]
            base = 5 + 3 * i
            obs[base] = (iy - y) / size
            obs[base + 1] = (ix - x) / size
            obs[base + 2] = 1.0
    patch_start = 5 + 3 * env.max_items
    obs[patch_start : patch_start + PATCH * PATCH] = env.padded_shelves[
        y : y + PATCH, x : x + PATCH
    ].ravel()
    obs[-2] = (env.n_available + env.n_carried) / env.n_items
    obs[-1] = 1.0 - env.t / env.max_steps
    return obs


def grid_observation(env: WarehouseEnv) -> np.ndarray:
    size = env.max_size
    planes = np.zeros((GRID_CHANNELS, size, size), dtype=np.float32)
    planes[0] = env.shelf_plane
    planes[1][env.pos] = 1.0
    for i in range(env.n_items):
        if env.status[i] == 0:
            planes[2][tuple(env.items[i])] = 1.0
    planes[3][env.layout.station] = 1.0
    planes[4] = env.n_carried / env.capacity
    planes[5] = 1.0 - env.t / env.max_steps
    return planes
