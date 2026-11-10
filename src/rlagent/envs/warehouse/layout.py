"""Procedural warehouse layouts: shelf blocks, aisles, and cached BFS distance maps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]
MOVES: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right
MIN_SIZE = 4


def bfs_distances(blocked: np.ndarray, source: Cell) -> np.ndarray:
    """Shortest-path step counts from ``source``; -1 where unreachable or blocked."""
    h, w = blocked.shape
    grid = blocked.tolist()
    dist = [[-1] * w for _ in range(h)]
    if grid[source[0]][source[1]]:
        return np.array(dist, dtype=np.int32)
    dist[source[0]][source[1]] = 0
    queue = deque([source])
    while queue:
        y, x = queue.popleft()
        nxt = dist[y][x] + 1
        for dy, dx in MOVES:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not grid[ny][nx] and dist[ny][nx] < 0:
                dist[ny][nx] = nxt
                queue.append((ny, nx))
    return np.array(dist, dtype=np.int32)


@dataclass
class Layout:
    """A static warehouse floor. ``shelves`` is True where the robot cannot walk."""

    shelves: np.ndarray
    station: Cell
    _maps: dict[Cell, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        walkable = ~self.shelves
        station_map = self.dist_map(self.station)
        free = [
            (y, x)
            for y, x in zip(*np.nonzero(walkable), strict=True)
            if (y, x) != self.station and station_map[y, x] > 0
        ]
        self.free_cells: list[Cell] = [(int(y), int(x)) for y, x in free]
        padded = np.pad(self.shelves, 1, constant_values=False)
        near_shelf = padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
        self.shelf_adjacent_cells: list[Cell] = [c for c in self.free_cells if near_shelf[c]]
        # Upper bound on any shortest path (triangle inequality via the station).
        self.diameter_bound: int = 2 * int(station_map.max()) + 1

    @property
    def height(self) -> int:
        return self.shelves.shape[0]

    @property
    def width(self) -> int:
        return self.shelves.shape[1]

    def dist_map(self, cell: Cell) -> np.ndarray:
        cached = self._maps.get(cell)
        if cached is None:
            cached = bfs_distances(self.shelves, cell)
            self._maps[cell] = cached
        return cached

    def distance(self, a: Cell, b: Cell) -> int:
        return int(self.dist_map(b)[a])

    def is_free(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width and not self.shelves[y, x]

    def step_towards(self, cell: Cell, target: Cell) -> int | None:
        """First move (0..3) of a shortest path from ``cell`` to ``target``; None if there."""
        dist = self.dist_map(target)
        here = dist[cell]
        if here <= 0:
            return None
        for action, (dy, dx) in enumerate(MOVES):
            ny, nx = cell[0] + dy, cell[1] + dx
            if self.is_free(ny, nx) and dist[ny, nx] == here - 1:
                return action
        return None


def generate_layout(height: int, width: int, shelf_density: float, seed: int) -> Layout:
    """Shelf rows (two thick, every third row is an aisle) with random gaps.

    Row 0, the bottom row and column 0 are always aisles. Any floor cell that ends up cut off
    from the station is turned into shelf, so every walkable cell is reachable.
    """
    if min(height, width) < MIN_SIZE:
        raise ValueError(f"Grid must be at least {MIN_SIZE}x{MIN_SIZE}, got {height}x{width}")
    if not 0.0 <= shelf_density <= 1.0:
        raise ValueError("shelf_density must be in [0, 1]")
    rng = np.random.default_rng([seed, height, width, round(shelf_density * 1000)])
    shelves = np.zeros((height, width), dtype=bool)
    draws = rng.random((height, width)) < shelf_density
    for row in range(1, height - 1):
        if row % 3 in (1, 2):
            shelves[row, 1:] = draws[row, 1:]
    station = (height - 1, 0)
    unreachable = (~shelves) & (bfs_distances(shelves, station) < 0)
    shelves |= unreachable
    return Layout(shelves=shelves, station=station)
