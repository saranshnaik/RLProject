"""Warehouse-specific training features: curriculum, CNN policy, metrics logging."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from rlagent.cli import app
from rlagent.config import load_config
from rlagent.envs.warehouse import WarehouseEnv
from rlagent.training.env_factory import build_eval_env, build_train_env, env_kwargs

CONFIG = Path(__file__).parent.parent / "configs" / "warehouse" / "ppo_vector.yaml"
TINY = [
    "train.total_timesteps=1024",
    "train.eval_freq=512",
    "train.n_eval_episodes=2",
    "env.n_envs=2",
    "algo.hyperparams.n_steps=64",
    "algo.hyperparams.batch_size=64",
]
STAGES = (
    "curriculum={stages: [{height: 6, width: 6, n_items: 1}, {height: 8, width: 8, n_items: 2}],"
    " threshold: 0.5, window: 10}"
)


def _train(tmp_path, overrides):
    args = ["train", str(CONFIG), "--runs-dir", str(tmp_path), "--quiet"]
    for item in TINY + overrides:
        args += ["--set", item]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    return next(tmp_path.glob("*/*/*/seed_*"))


def test_difficulty_change_is_deferred_to_reset_and_keeps_shapes():
    env = WarehouseEnv(height=6, width=6, n_items=1, max_size=10, max_items=3, obs_type="grid")
    first, _ = env.reset(seed=0)
    env.set_difficulty(height=10, width=10, n_items=3)
    assert (env.height, env.n_items) == (6, 1)  # running episode untouched
    env.step(0)
    second, _ = env.reset()
    assert (env.height, env.n_items) == (10, 3)
    assert first.shape == second.shape == (6, 10, 10)
    assert env.max_steps == 4 * 3 * 20 + 20  # auto max_steps follows the new difficulty


def test_difficulty_cannot_exceed_padding_and_rejects_unknown_keys():
    env = WarehouseEnv(height=6, width=6, n_items=1)
    with pytest.raises(ValueError):
        env.set_difficulty(bogus=1)
    env.set_difficulty(height=9)
    with pytest.raises(ValueError, match="max_size"):
        env.reset(seed=0)


def test_env_kwargs_pad_to_largest_stage_and_start_at_stage_zero():
    cfg = load_config(CONFIG, [STAGES])
    train, evaluate = env_kwargs(cfg, training=True), env_kwargs(cfg)
    assert (train["height"], train["n_items"]) == (6, 1)  # stage 0
    assert (evaluate["height"], evaluate["n_items"]) == (12, 3)  # full target
    assert train["max_size"] == evaluate["max_size"] == 12
    assert train["max_items"] == evaluate["max_items"] == 3
    venv, eval_env = build_train_env(cfg, 0), build_eval_env(cfg, 0)
    assert venv.observation_space == eval_env.observation_space


def test_curriculum_config_validation():
    with pytest.raises(ValidationError):
        load_config(CONFIG, ["curriculum={stages: [{height: 6}]}"])  # needs >= 2 stages
    with pytest.raises(ValidationError):
        load_config(CONFIG, [STAGES, "algo.impl=scratch"])  # curriculum is an SB3-only feature


def test_curriculum_promotes_and_metrics_are_logged(tmp_path):
    run = _train(
        tmp_path,
        [
            "train.total_timesteps=6000",
            "algo.hyperparams.n_steps=128",
            "curriculum={stages: [{height: 6, width: 6, n_items: 1, capacity: 1},"
            " {height: 8, width: 8, n_items: 2, capacity: 1}], threshold: 0.0, window: 10}",
        ],
    )
    log = pd.read_csv(run / "progress.csv")
    assert log["curriculum/stage"].max() == 1  # threshold 0 -> promoted after `window` episodes
    for column in ("rollout/success_rate", "rollout/efficiency", "reward/delivery", "reward/step"):
        assert column in log.columns, column


def test_grid_cnn_policy_trains_and_evaluates(tmp_path):
    run = _train(
        tmp_path,
        [
            "env.kwargs.obs_type=grid",
            "env.kwargs.height=8",
            "env.kwargs.width=8",
            "env.kwargs.n_items=1",
            "algo.policy=CnnPolicy",
            "algo.policy_kwargs={features_extractor_class: SmallGridCNN,"
            " features_extractor_kwargs: {features_dim: 64}, normalize_images: false}",
        ],
    )
    result = CliRunner().invoke(app, ["evaluate", str(run), "--episodes", "2", "--final"])
    assert result.exit_code == 0, result.output


def test_evaluate_override_runs_on_the_heldout_split_without_clobbering(tmp_path):
    run = _train(tmp_path, [])
    runner = CliRunner()
    assert runner.invoke(app, ["evaluate", str(run), "--episodes", "2", "--final"]).exit_code == 0
    standard = (run / "eval_final.json").read_text()
    result = runner.invoke(
        app, ["evaluate", str(run), "--episodes", "2", "--final", "--set", "env.kwargs.split=test"]
    )
    assert result.exit_code == 0, result.output
    assert (run / "eval_final.json").read_text() == standard  # untagged override is not saved
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(run),
            "--episodes",
            "2",
            "--final",
            "-s",
            "env.kwargs.split=test",
            "--tag",
            "test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (run / "eval_final_test.json").exists()
    assert np.isfinite(
        float(__import__("json").loads((run / "eval_final_test.json").read_text())["mean_return"])
    )
