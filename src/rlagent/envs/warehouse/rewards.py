"""Reward configuration and shaping for Warehouse-v0.

Shaping modes
-------------
none   sparse task rewards only.
pbrs   potential-based shaping F = gamma * phi(s') - phi(s) (Ng et al., 1999), with
       phi(terminal) = 0. Provably leaves the optimal policy unchanged.
naive  +naive_bonus whenever a move reduces the distance to the current subgoal and nothing
       for moving away. Not potential-based: oscillating farms it (the reward-hacking demo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rlagent.envs.warehouse.env import WarehouseEnv

COMPONENTS = ("step", "pick", "delivery", "complete", "invalid", "shaping")


@dataclass(frozen=True)
class RewardConfig:
    step: float = -0.01
    pick: float = 1.0
    delivery: float = 5.0  # per item delivered
    complete: float = 10.0  # all items delivered
    invalid: float = -0.1
    shaping: Literal["none", "naive", "pbrs"] = "none"
    shaping_scale: float = 1.0  # multiplies the PBRS term
    naive_bonus: float = 0.1

    def __post_init__(self) -> None:
        if self.shaping not in ("none", "naive", "pbrs"):
            raise ValueError(f"Unknown shaping '{self.shaping}'")


def subgoal_distance(env: WarehouseEnv) -> int:
    """Steps from the robot to its current subgoal.

    Subgoal = the station when the robot is full or has nothing left to pick (and carries
    something), otherwise the nearest available item. 0 when there is nothing left to do.
    """
    layout, pos = env.layout, env.pos
    if env.n_carried > 0 and (env.n_carried >= env.capacity or env.n_available == 0):
        return int(layout.dist_map(layout.station)[pos])
    if env.n_available > 0:
        return min(
            int(layout.dist_map((int(y), int(x)))[pos])
            for (y, x), s in zip(env.items, env.status, strict=True)
            if s == 0
        )
    return 0


def potential(env: WarehouseEnv) -> float:
    """phi(s) = -(remaining work) - normalised distance to the subgoal.

    Remaining work counts 2 per item still to pick and 1 per item still to deliver, so phi is
    exactly 0 once everything is delivered (the terminal convention needs no special-casing).
    The distance term lies in (-1, 0], so picking up or delivering never lowers the potential.
    """
    remaining = 2 * env.n_available + env.n_carried
    return -remaining - subgoal_distance(env) / env.layout.diameter_bound
