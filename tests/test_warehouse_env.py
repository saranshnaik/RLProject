import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import rlagent  # noqa: F401  (registers Warehouse-v0)
from rlagent.agents.baselines import OptimalPlannerAgent
from rlagent.envs.warehouse import WarehouseEnv

KWARGS = dict(height=8, width=8, n_items=3, capacity=2)


def make(**overrides) -> WarehouseEnv:
    return WarehouseEnv(**{**KWARGS, **overrides})


def run_optimal(env: WarehouseEnv, seed=0):
    agent = OptimalPlannerAgent(lambda: env)
    obs, info = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        action = agent.predict(obs[None])[0][0]
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            return terminated, steps, info, total


@pytest.mark.parametrize("obs_type", ["vector", "grid"])
def test_gymnasium_and_sb3_checkers(obs_type):
    gym_check_env(make(obs_type=obs_type), skip_render_check=True)
    sb3_check_env(make(obs_type=obs_type), warn=False)


def test_registered_with_gymnasium():
    env = gym.make("Warehouse-v0", n_items=2)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("obs_type", ["vector", "grid"])
def test_observations_stay_in_space_during_random_play(obs_type):
    env = make(obs_type=obs_type, max_size=10, max_items=5)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset()


def test_seeded_reset_is_deterministic():
    def trace(seed):
        env = make()
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for a in [0, 3, 3, 1, 4, 2, 5, 0]:
            out.append(env.step(a)[0])
        return np.stack(out)

    assert np.array_equal(trace(3), trace(3))
    assert not np.array_equal(trace(3), trace(4))


def test_truncation_at_max_steps_without_success():
    env = make(max_steps=15)
    env.reset(seed=0)
    for step in range(1, 16):
        _, _, terminated, truncated, info = env.step(env.DROP)  # always invalid at start
        assert not terminated
        assert truncated == (step == 15)
    assert info["is_success"] is False
    assert info["efficiency"] == 0.0
    assert info["invalid_actions"] == 15


@pytest.mark.parametrize("n_items,capacity", [(1, 1), (3, 1), (4, 2), (3, 3)])
def test_optimal_play_terminates_in_planned_steps(n_items, capacity):
    env = make(n_items=n_items, capacity=capacity)
    terminated, steps, info, _ = run_optimal(env, seed=n_items)
    assert terminated and info["is_success"] is True
    assert steps == info["optimal_steps"]
    assert info["efficiency"] == pytest.approx(1.0)
    assert info["invalid_actions"] == 0
    assert info["n_delivered"] == n_items


def test_reward_components_sum_to_reward():
    env = make(reward={"shaping": "pbrs"})
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        _, reward, terminated, truncated, info = env.step(int(rng.integers(6)))
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        if terminated or truncated:
            totals = info["episode_reward_components"]
            assert set(totals) == set(info["reward_components"])
            env.reset()


def test_invalid_actions_are_penalised_and_do_not_change_state():
    env = make()
    env.reset(seed=0)
    pos = env.pos
    for action in (env.LEFT, env.DOWN, env.PICK, env.DROP):  # wall, wall, nothing, nothing
        _, reward, _, _, info = env.step(action)
        assert info["invalid_action"]
        assert info["reward_components"]["invalid"] == env.reward_cfg.invalid
        assert env.pos == pos


def test_action_mask_matches_actual_validity():
    env = make(n_items=4, capacity=2)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    for _ in range(600):
        mask = env.action_masks()
        action = int(rng.integers(6))
        _, _, terminated, truncated, info = env.step(action)
        assert mask[action] == (not info["invalid_action"])
        if terminated or truncated:
            env.reset()


def test_carrying_never_exceeds_capacity_and_pick_needs_an_item():
    env = make(n_items=3, capacity=1)
    env.reset(seed=0)
    cells = [tuple(map(int, c)) for c in env.items]
    env.pos = cells[0]
    assert env.step(env.PICK)[4]["invalid_action"] is False
    env.pos = cells[1]
    assert env.step(env.PICK)[4]["invalid_action"] is True  # full
    assert env.n_carried == 1


def test_layout_pool_and_train_test_split():
    fixed = make(n_layouts=1)
    seeds = set()
    for i in range(5):
        _, info = fixed.reset(seed=i)
        seeds.add(info["layout_seed"])
    assert len(seeds) == 1

    train, test = make(n_layouts=None), make(split="test")
    train_seeds = {train.reset(seed=i)[1]["layout_seed"] for i in range(20)}
    test_seeds = {test.reset(seed=i)[1]["layout_seed"] for i in range(20)}
    assert max(train_seeds) < WarehouseEnv.TEST_SEED_BASE <= min(test_seeds)
    assert len(train_seeds) > 10


def test_reset_options_force_layout_and_items():
    env = make(n_items=2)
    _, info = env.reset(seed=0, options={"layout_seed": 77})
    cells = env.layout.free_cells[:2]
    env.reset(options={"layout_seed": 77, "items": cells})
    assert info["layout_seed"] == 77
    assert [tuple(map(int, c)) for c in env.items] == cells
    with pytest.raises(ValueError):
        env.reset(options={"layout_seed": 77, "items": [(0, 0), (0, 0)]})


def test_items_are_distinct_reachable_and_not_on_station():
    env = make(n_items=5)
    for seed in range(20):
        env.reset(seed=seed)
        cells = [tuple(map(int, c)) for c in env.items]
        assert len(set(cells)) == 5
        assert env.layout.station not in cells
        assert all(env.layout.distance(env.layout.station, c) > 0 for c in cells)


def test_padded_observations_have_fixed_shape_across_sizes():
    shapes = set()
    for size in (6, 8, 10):
        env = make(height=size, width=size, obs_type="grid", max_size=10, max_items=4)
        obs, _ = env.reset(seed=0)
        shapes.add(obs.shape)
        assert (obs[0, size:, :] == 1).all() and (obs[0, :, size:] == 1).all()  # padding = wall
    assert shapes == {(6, 10, 10)}
    lengths = {make(n_items=n, max_items=5).reset(seed=0)[0].shape for n in (1, 3, 5)}
    assert len(lengths) == 1


def test_too_dense_layout_raises_clear_error():
    env = WarehouseEnv(height=4, width=4, n_items=30)
    with pytest.raises(ValueError, match="free cells"):
        env.reset(seed=0)


def test_rgb_and_ansi_rendering():
    env = make(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (8 * 32, 8 * 32, 3) and frame.dtype == np.uint8
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) >= 4

    text = make(render_mode="ansi")
    text.reset(seed=0)
    rows = text.render().split("\n")
    assert len(rows) == 8 and all(len(r) == 8 for r in rows)
    assert sum(r.count("i") for r in rows) == 3 and "".join(rows).count("R") == 1


def test_human_rendering_with_dummy_display(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make(render_mode="human")
    env.reset(seed=0)
    env.step(env.UP)
    env.close()
