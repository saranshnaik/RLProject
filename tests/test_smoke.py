import gymnasium as gym
import stable_baselines3  # noqa: F401
from typer.testing import CliRunner

import rlagent
from rlagent.cli import app


def test_version():
    assert rlagent.__version__


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cartpole_steps():
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.shape == (4,)
    env.close()
