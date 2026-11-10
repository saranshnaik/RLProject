"""Rendering: pure-NumPy ``rgb_array`` (headless-safe), ``ansi`` text, and a pygame window."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rlagent.envs.warehouse.env import WarehouseEnv

CELL = 32
FLOOR = (245, 245, 240)
SHELF = (70, 76, 92)
GRID_LINE = (215, 215, 208)
STATION = (76, 175, 112)
ITEM = (244, 162, 54)
ROBOT = (52, 120, 220)


def _disc_mask(size: int, radius: float, center: tuple[float, float] | None = None) -> np.ndarray:
    cy, cx = center if center is not None else ((size - 1) / 2, (size - 1) / 2)
    yy, xx = np.mgrid[:size, :size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def render_rgb(env: WarehouseEnv, cell: int = CELL) -> np.ndarray:
    h, w = env.height, env.width
    img = np.empty((h * cell, w * cell, 3), dtype=np.uint8)
    img[:] = FLOOR
    img[np.kron(env.layout.shelves, np.ones((cell, cell), dtype=bool))] = SHELF
    img[::cell, :] = GRID_LINE
    img[:, ::cell] = GRID_LINE

    def block(y: int, x: int) -> tuple[slice, slice]:
        return slice(y * cell + 1, (y + 1) * cell), slice(x * cell + 1, (x + 1) * cell)

    sy, sx = env.layout.station
    img[block(sy, sx)] = STATION
    inset = max(cell // 4, 2)
    for (iy, ix), status in zip(env.items, env.status, strict=True):
        if status == 0:
            ys, xs = block(int(iy), int(ix))
            img[ys.start + inset : ys.stop - inset, xs.start + inset : xs.stop - inset] = ITEM

    ry, rx = env.pos
    ys, xs = block(ry, rx)
    sub = img[ys, xs]
    sub[_disc_mask(sub.shape[0], cell * 0.38)] = ROBOT
    for k in range(env.n_carried):  # one orange dot per carried item
        dot = _disc_mask(sub.shape[0], cell * 0.09, (cell * 0.5, cell * (0.28 + 0.2 * k)))
        sub[dot] = ITEM
    return img


def render_ansi(env: WarehouseEnv) -> str:
    rows = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            if (y, x) == env.pos:
                row.append("R")
            elif (y, x) == env.layout.station:
                row.append("S")
            elif env.layout.shelves[y, x]:
                row.append("#")
            else:
                row.append(".")
        rows.append(row)
    for (iy, ix), status in zip(env.items, env.status, strict=True):
        if status == 0 and rows[iy][ix] == ".":
            rows[iy][ix] = "i"
    return "\n".join("".join(r) for r in rows)


class HumanRenderer:
    """Pygame window. Imported lazily so headless use never needs a display."""

    def __init__(self, fps: int):
        self.fps = fps
        self._screen = None
        self._clock = None

    def draw(self, frame: np.ndarray) -> None:
        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            pygame.display.set_caption("Warehouse-v0")
            self._clock = pygame.time.Clock()
        pygame.surfarray.blit_array(self._screen, frame.swapaxes(0, 1))
        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(self.fps)

    def close(self) -> None:
        if self._screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._screen = None
