from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from rlagent.agents.baselines import GreedyAgent, OptimalPlannerAgent, RandomAgent, make_baseline
from rlagent.cli import app
from rlagent.config import load_config
from rlagent.envs.warehouse import WarehouseEnv
from rlagent.evaluation.evaluate import evaluate_baseline

CONFIG = Path(__file__).parent.parent / "configs" / "warehouse" / "ppo_vector.yaml"


def episode_steps(env: WarehouseEnv, agent, seed: int) -> tuple[int, bool, dict]:
    obs, _ = env.reset(seed=seed)
    steps = 0
    while True:
        obs, _, terminated, truncated, info = env.step(int(agent.predict(obs[None])[0][0]))
        steps += 1
        if terminated or truncated:
            return steps, terminated, info


@pytest.mark.parametrize("n_items,capacity,size", [(2, 2, 8), (5, 3, 12), (6, 2, 12)])
def test_optimal_is_never_beaten_and_greedy_always_succeeds(n_items, capacity, size):
    env = WarehouseEnv(height=size, width=size, n_items=n_items, capacity=capacity)
    greedy, optimal = GreedyAgent(lambda: env), OptimalPlannerAgent(lambda: env)
    for seed in range(15):
        g_steps, g_ok, _ = episode_steps(env, greedy, seed)
        o_steps, o_ok, info = episode_steps(env, optimal, seed)
        assert g_ok and o_ok
        assert o_steps == info["optimal_steps"] <= g_steps


def test_greedy_is_strictly_worse_somewhere():
    env = WarehouseEnv(height=12, width=12, n_items=5, capacity=3)
    greedy = GreedyAgent(lambda: env)
    gaps = [
        episode_steps(env, greedy, s)[0]
        - episode_steps(env, OptimalPlannerAgent(lambda: env), s)[0]
        for s in range(30)
    ]
    assert min(gaps) >= 0 and max(gaps) > 0


def test_random_agent_rarely_succeeds_on_a_real_task():
    env = WarehouseEnv(height=8, width=8, n_items=3, capacity=2)
    agent = RandomAgent(seed=0)
    successes = sum(episode_steps(env, agent, s)[1] for s in range(20))
    assert successes <= 2


def test_evaluate_baseline_through_the_vecenv_path():
    cfg = load_config(CONFIG, ["env.kwargs.n_items=3", "env.kwargs.capacity=2"])
    optimal = evaluate_baseline(cfg, "optimal", n_episodes=10)
    greedy = evaluate_baseline(cfg, "greedy", n_episodes=10)
    random = evaluate_baseline(cfg, "random", n_episodes=10)
    assert optimal["success_rate"] == greedy["success_rate"] == 1.0
    assert optimal["mean_efficiency"] == pytest.approx(1.0)
    assert greedy["mean_efficiency"] <= 1.0
    assert optimal["mean_return"] >= greedy["mean_return"] - 1e-9 > random["mean_return"]
    assert optimal["mean_invalid_actions"] == 0
    assert random["mean_invalid_actions"] > 0


def test_unknown_baseline_rejected():
    with pytest.raises(ValueError):
        make_baseline("bogus", None)


def test_baseline_cli(tmp_path):
    result = CliRunner().invoke(
        app, ["baseline", str(CONFIG), "--agent", "greedy", "--episodes", "5"]
    )
    assert result.exit_code == 0, result.output
    assert '"success_rate": 1.0' in result.output


def test_heldout_split_is_evaluable():
    cfg = load_config(CONFIG, ["env.kwargs.split=test"])
    assert evaluate_baseline(cfg, "optimal", n_episodes=5)["success_rate"] == 1.0
    assert np.isclose(evaluate_baseline(cfg, "optimal", n_episodes=5)["mean_efficiency"], 1.0)
