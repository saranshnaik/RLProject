"""MaskablePPO support: masks are respected in training, evaluation and recording."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from rlagent.cli import app
from rlagent.evaluation.evaluate import evaluate_agent, load_run

CONFIG = Path(__file__).parent.parent / "configs" / "warehouse" / "ppo_vector.yaml"
TINY = [
    "algo.name=maskable_ppo",
    "train.total_timesteps=2048",
    "train.eval_freq=1024",
    "train.n_eval_episodes=2",
    "env.n_envs=2",
    "algo.hyperparams.n_steps=64",
    "algo.hyperparams.batch_size=64",
    "env.kwargs.height=8",
    "env.kwargs.width=8",
    "env.kwargs.n_items=2",
]


def test_maskable_ppo_trains_evaluates_and_never_acts_invalidly(tmp_path):
    args = ["train", str(CONFIG), "--runs-dir", str(tmp_path), "--quiet"]
    for item in TINY:
        args += ["--set", item]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    run = next(tmp_path.glob("*/*/*/seed_*"))
    assert (run / "final_model.zip").exists()

    model, env, _ = load_run(run, "final")
    metrics = evaluate_agent(model, env, 10, deterministic=False)
    env.close()
    assert metrics["mean_invalid_actions"] == 0  # masks make invalid actions impossible
    assert np.isfinite(metrics["mean_return"])

    gif = tmp_path / "masked.gif"
    result = runner.invoke(
        app, ["record", str(run), "--out", str(gif), "--final", "--max-frames", "15"]
    )
    assert result.exit_code == 0, result.output
    assert gif.stat().st_size > 0
