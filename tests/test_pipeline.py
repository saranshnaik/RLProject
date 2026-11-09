"""End-to-end: train a tiny run, evaluate it, record it."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rlagent.cli import app

CONFIGS = Path(__file__).parent.parent / "configs" / "benchmarks"
TINY = ["train.total_timesteps=512", "train.eval_freq=256", "train.n_eval_episodes=2"]


def _train(config: str, tmp_path: Path, extra: list[str]) -> Path:
    args = ["train", str(CONFIGS / config), "--runs-dir", str(tmp_path), "--quiet"]
    for item in TINY + extra:
        args += ["--set", item]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    run_dirs = list(tmp_path.glob("*/*/*/seed_*"))
    assert len(run_dirs) == 1
    return run_dirs[0]


@pytest.mark.parametrize(
    "config,extra",
    [("cartpole_ppo.yaml", []), ("cartpole_dqn.yaml", ["algo.hyperparams.learning_starts=100"])],
)
def test_train_evaluate_record(config, extra, tmp_path):
    run_dir = _train(config, tmp_path, extra)
    for name in ["config.yaml", "progress.csv", "final_model.zip", "metadata.json"]:
        assert (run_dir / name).exists(), name
    assert (run_dir / "tb").is_dir()

    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", str(run_dir), "--episodes", "3", "--final"])
    assert result.exit_code == 0, result.output
    metrics = json.loads(result.output)
    assert metrics["n_episodes"] == 3
    assert metrics["mean_length"] > 0

    gif = tmp_path / "out.gif"
    result = runner.invoke(
        app, ["record", str(run_dir), "--out", str(gif), "--final", "--max-frames", "20"]
    )
    assert result.exit_code == 0, result.output
    assert gif.stat().st_size > 0


def test_existing_run_dir_is_not_clobbered(tmp_path):
    _train("cartpole_ppo.yaml", tmp_path, [])
    result = CliRunner().invoke(
        app, ["train", str(CONFIGS / "cartpole_ppo.yaml"), "--runs-dir", str(tmp_path), "--quiet"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, FileExistsError)
