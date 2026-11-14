"""Scripted reward-hacking demonstration: farm a shaping bonus by oscillating.

No trained model is needed. A fixed script steps toward the nearest item and straight back again.
Under potential-based shaping (PBRS) that loop is worth about nothing, so the agent has no reason
to do it; under the naive "bonus for getting closer" shaping every other step is rewarded, so the
loop pays ``naive_bonus / 2`` per step forever.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlagent.envs.warehouse.env import AVAILABLE, WarehouseEnv

from .rollout import BaselineActor, layout_seed_for, play_episode

REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}  # up<->down, left<->right

SHAPING_ARMS: tuple[tuple[str, dict], ...] = (
    ("none (sparse)", {"shaping": "none"}),
    ("naive, bonus 0.1 (default)", {"shaping": "naive", "naive_bonus": 0.1}),
    ("naive, bonus 1.0 (e1_naive_big)", {"shaping": "naive", "naive_bonus": 1.0}),
    ("PBRS", {"shaping": "pbrs"}),
)


@dataclass(frozen=True)
class DemoSetup:
    size: int = 12
    n_items: int = 3
    capacity: int = 2
    layout_index: int = 0
    steps: int | None = None  # None = until the time limit


def _env(reward: dict, setup: DemoSetup) -> WarehouseEnv:
    return WarehouseEnv(
        height=setup.size,
        width=setup.size,
        n_items=setup.n_items,
        capacity=setup.capacity,
        reward=reward,
        shaping_gamma=0.995,
        split="test",
        render_mode="rgb_array",
    )


def oscillation_return(reward: dict, setup: DemoSetup | None = None) -> dict[str, float]:
    """Total reward of the toward/away loop up to the time limit (no item is picked)."""
    setup = setup or DemoSetup()
    env = _env(reward, setup)
    env.reset(seed=0, options={"layout_seed": layout_seed_for("test", setup.layout_index)})
    total = shaping = 0.0
    steps = 0
    last = None
    for t in range(setup.steps or env.max_steps):
        if t % 2 == 0:
            targets = [
                (int(y), int(x))
                for (y, x), s in zip(env.items, env.status, strict=True)
                if s == AVAILABLE
            ]
            target = min(targets, key=lambda c: env.layout.distance(env.pos, c))
            action = env.layout.step_towards(env.pos, target)
            if action is None:  # already standing on the item: just step away
                action = next(a for a, ok in enumerate(env.action_masks()[:4]) if ok)
        else:
            action = REVERSE[last]
        last = action
        _, reward_t, terminated, truncated, info = env.step(action)
        total += reward_t
        shaping += info["reward_components"]["shaping"]
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return {"return": total, "shaping": shaping, "steps": steps}


def completion_return(reward: dict, setup: DemoSetup | None = None) -> dict[str, float]:
    """Total reward when the optimal planner simply does the job on the same layout."""
    setup = setup or DemoSetup()
    env = _env(reward, setup)
    actor = BaselineActor("optimal", env)
    episode = play_episode(actor, env, layout_seed_for("test", setup.layout_index), 0)
    env.close()
    return {
        "return": episode.total_return,
        "shaping": episode.component_totals.get("shaping", 0.0),
        "steps": episode.steps,
    }


def hacking_table(setup: DemoSetup | None = None) -> pd.DataFrame:
    """One row per shaping arm: reward for oscillating vs. for finishing the task."""
    setup = setup or DemoSetup()
    rows = []
    for label, reward in SHAPING_ARMS:
        loop = oscillation_return(reward, setup)
        done = completion_return(reward, setup)
        rows.append(
            {
                "reward shaping": label,
                f"oscillate {loop['steps']} steps": round(loop["return"], 2),
                "...of which shaping": round(loop["shaping"], 2),
                f"finish the task ({done['steps']} steps)": round(done["return"], 2),
                "hacking pays": loop["return"] > done["return"],
            }
        )
    return pd.DataFrame(rows)
