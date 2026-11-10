"""From-scratch agents: tiny end-to-end runs, save/load round trip, and slow learning checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from rlagent.agents.scratch import load_scratch
from rlagent.cli import app
from rlagent.config import AlgoConfig
from rlagent.evaluation.evaluate import load_run

CONFIGS = Path(__file__).parent.parent / "configs" / "benchmarks"
TINY = ["train.total_timesteps=2048", "train.eval_freq=1024", "train.n_eval_episodes=2"]
PPO_TINY = ["algo.hyperparams.n_steps=64", "algo.hyperparams.batch_size=64"]
DQN_TINY = [
    "algo.hyperparams.learning_starts=100",
    "algo.hyperparams.train_freq=4",
    "algo.hyperparams.gradient_steps=1",
]


def _train(config: str, tmp_path: Path, extra: list[str]) -> Path:
    args = ["train", str(CONFIGS / config), "--runs-dir", str(tmp_path), "--quiet"]
    for item in extra:
        args += ["--set", item]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    return next(tmp_path.glob("*/*/*/seed_*"))


@pytest.mark.parametrize(
    "config,extra",
    [("cartpole_ppo_scratch.yaml", PPO_TINY), ("cartpole_dqn_scratch.yaml", DQN_TINY)],
)
def test_scratch_run_contract_and_roundtrip(config, extra, tmp_path):
    run_dir = _train(config, tmp_path, TINY + extra)
    for name in ["config.yaml", "progress.csv", "final_model.pt", "best_model.pt", "metadata.json"]:
        assert (run_dir / name).exists(), name

    # Same metric keys as SB3 so curves can be overlaid.
    columns = set(pd.read_csv(run_dir / "progress.csv").columns)
    assert {"eval/mean_reward", "rollout/ep_rew_mean", "time/total_timesteps"} <= columns

    model, env, cfg = load_run(run_dir, "final")
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    reloaded = load_scratch(run_dir / "final_model.pt")
    action2, _ = reloaded.predict(obs, deterministic=True)
    assert np.array_equal(action, action2)
    env.close()

    result = CliRunner().invoke(app, ["evaluate", str(run_dir), "--episodes", "2", "--final"])
    assert result.exit_code == 0, result.output


def test_scratch_rejects_unsupported_config():
    with pytest.raises(ValueError):
        AlgoConfig(name="maskable_ppo", impl="scratch")
    with pytest.raises(ValueError):
        AlgoConfig(name="ppo", impl="scratch", policy="CnnPolicy")


@pytest.mark.slow
@pytest.mark.parametrize(
    "config,steps",
    [("cartpole_ppo_scratch.yaml", 60_000), ("cartpole_dqn_scratch.yaml", 50_000)],
)
def test_scratch_learns_cartpole(config, steps, tmp_path):
    run_dir = _train(config, tmp_path, [f"train.total_timesteps={steps}", "train.eval_freq=5000"])
    best = pd.read_csv(run_dir / "progress.csv")["eval/mean_reward"].max()
    assert best >= 150, f"best eval return only {best}"
