import pytest

from rlagent.agents.baselines import OptimalPlannerAgent
from rlagent.envs.warehouse import RewardConfig, WarehouseEnv
from rlagent.envs.warehouse.rewards import potential

BASE = dict(height=8, width=8, n_items=3, capacity=2)


def env_with(shaping: str, **kw) -> WarehouseEnv:
    return WarehouseEnv(**BASE, reward=RewardConfig(shaping=shaping, **kw), shaping_gamma=1.0)


def test_no_shaping_gives_zero_shaping_component():
    env = env_with("none")
    env.reset(seed=0)
    for action in (env.UP, env.DOWN, env.UP):
        assert env.step(action)[4]["reward_components"]["shaping"] == 0.0


def test_pbrs_returning_to_the_same_state_nets_zero():
    env = env_with("pbrs")
    env.reset(seed=0)
    total = 0.0
    for action in (env.UP, env.UP, env.DOWN, env.DOWN):  # out and back along column 0
        total += env.step(action)[4]["reward_components"]["shaping"]
    assert env.pos == env.layout.station
    assert total == pytest.approx(0.0, abs=1e-9)


def test_pbrs_telescopes_to_minus_initial_potential_over_an_episode():
    """With gamma = 1 and phi(terminal) = 0, total shaping is exactly -phi(s0): a constant, so
    shaping cannot change which policy is optimal."""
    for seed in range(5):
        env = env_with("pbrs")
        agent = OptimalPlannerAgent(lambda env=env: env)
        obs, _ = env.reset(seed=seed)
        phi0 = potential(env)
        total = 0.0
        while True:
            obs, _, terminated, truncated, info = env.step(agent.predict(obs[None])[0][0])
            total += info["reward_components"]["shaping"]
            if terminated or truncated:
                break
        assert terminated
        assert total == pytest.approx(-phi0, abs=1e-6)


def test_pbrs_never_penalises_progress_events():
    """Picking up and delivering must not lower the potential (otherwise the agent would be
    discouraged from the very events the task rewards)."""
    env = env_with("pbrs")
    agent = OptimalPlannerAgent(lambda: env)
    obs, _ = env.reset(seed=1)
    events = 0
    while True:
        action = int(agent.predict(obs[None])[0][0])
        obs, _, terminated, truncated, info = env.step(action)
        if action in (env.PICK, env.DROP):
            events += 1
            assert info["reward_components"]["shaping"] > 0
        if terminated or truncated:
            break
    assert events >= 4


def test_pbrs_rewards_moving_towards_the_subgoal():
    env = env_with("pbrs")
    env.reset(seed=0)
    target = tuple(map(int, env.items[0]))
    nearest = min(
        (tuple(map(int, c)) for c in env.items), key=lambda c: env.layout.distance(env.pos, c)
    )
    action = env.layout.step_towards(env.pos, nearest)
    assert env.step(action)[4]["reward_components"]["shaping"] > 0
    assert target is not None


def test_naive_shaping_can_be_farmed_by_oscillating():
    env = env_with("naive", naive_bonus=0.1)
    env.reset(seed=0)
    nearest = min(
        (tuple(map(int, c)) for c in env.items), key=lambda c: env.layout.distance(env.pos, c)
    )
    toward = env.layout.step_towards(env.pos, nearest)
    away = {0: 1, 1: 0, 2: 3, 3: 2}[toward]
    total = 0.0
    for _ in range(10):  # step towards (bonus), step back (no bonus), repeat
        total += env.step(toward)[1]
        total += env.step(away)[1]
    # Each pair earns +0.1 bonus against 2 * -0.01 step cost, without ever making progress.
    assert env.n_delivered == 0 and env.n_carried == 0
    assert total == pytest.approx(10 * (0.1 - 0.02), abs=1e-6)


def test_unknown_shaping_rejected():
    with pytest.raises(ValueError):
        RewardConfig(shaping="bogus")


def test_reward_config_accepts_dict_from_yaml():
    env = WarehouseEnv(**BASE, reward={"shaping": "pbrs", "delivery": 3.0})
    assert env.reward_cfg.shaping == "pbrs" and env.reward_cfg.delivery == 3.0
