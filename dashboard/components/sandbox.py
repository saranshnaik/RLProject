"""Custom-layout helpers for the Sandbox page (no Streamlit here)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlagent.envs.warehouse.env import WarehouseEnv

from .rollout import (
    BASELINES,
    BaselineActor,
    Episode,
    LoadedPolicy,
    PolicyActor,
    build_env,
    layout_seed_for,
    model_limits,
    play_episode,
)

MAX_BASELINE_ITEMS = 6  # the exact planner is a brute-force search over item orderings


@dataclass(frozen=True)
class SandboxParams:
    split: str = "test"
    layout_index: int = 0
    item_seed: int = 0
    height: int = 12
    width: int = 12
    n_items: int = 3
    capacity: int = 2
    shelf_density: float = 0.7

    @property
    def layout_seed(self) -> int:
        return layout_seed_for(self.split, self.layout_index)

    def difficulty(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "n_items": self.n_items,
            "capacity": self.capacity,
            "shelf_density": self.shelf_density,
        }


def baseline_env(params: SandboxParams, reward: dict | None = None) -> WarehouseEnv:
    """A rendering env of any size (no observation padding) for baselines and manual play."""
    kwargs = params.difficulty() | {"split": params.split, "render_mode": "rgb_array"}
    if reward is not None:
        kwargs["reward"] = reward
    return WarehouseEnv(**kwargs)


def model_env(params: SandboxParams, policy: LoadedPolicy) -> WarehouseEnv:
    return build_env(policy.cfg, params.split, params.difficulty())


def model_fits(params: SandboxParams, policy: LoadedPolicy) -> tuple[bool, str]:
    """Whether the policy's fixed-size observation can represent this layout."""
    max_size, max_items = model_limits(policy.cfg)
    if max(params.height, params.width) > max_size or params.n_items > max_items:
        return False, (
            f"This model's observation supports at most {max_size}x{max_size} grids and "
            f"{max_items} items."
        )
    return True, ""


def check_params(params: SandboxParams) -> str | None:
    """A message describing why the layout cannot be built, or None if it is fine."""
    try:
        env = baseline_env(params)
        try:
            env.reset(seed=params.item_seed, options={"layout_seed": params.layout_seed})
        finally:
            env.close()
    except ValueError as exc:
        return str(exc)
    return None


def run_agents(
    params: SandboxParams, baselines: list[str], policy: LoadedPolicy | None = None
) -> dict[str, Episode]:
    """Play one episode per requested agent on the same layout and item placement."""
    results: dict[str, Episode] = {}
    for name in baselines:
        if name not in BASELINES:
            raise ValueError(f"Unknown baseline '{name}'")
        if name == "optimal" and params.n_items > MAX_BASELINE_ITEMS:
            continue
        env = baseline_env(params)
        try:
            results[name] = play_episode(
                BaselineActor(name, env, seed=params.item_seed),
                env,
                params.layout_seed,
                params.item_seed,
            )
        finally:
            env.close()
    if policy is not None:
        env = model_env(params, policy)
        try:
            results[policy.name] = play_episode(
                PolicyActor(policy), env, params.layout_seed, params.item_seed
            )
        finally:
            env.close()
    return results


def results_frame(episodes: dict[str, Episode]) -> pd.DataFrame:
    rows = []
    for name, ep in episodes.items():
        rows.append(
            {
                "Agent": name,
                "Success": ep.success,
                "Steps": ep.steps,
                "Optimal steps": ep.optimal_steps,
                "Efficiency": None if ep.efficiency is None else round(ep.efficiency, 3),
                "Return": round(ep.total_return, 2),
                "Invalid actions": ep.invalid_actions,
            }
        )
    return pd.DataFrame(rows)
