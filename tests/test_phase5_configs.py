"""Every group of the Phase 5 matrix must at least train and evaluate for a few steps on CPU."""

from pathlib import Path

import pytest

from rlagent.config import load_config
from rlagent.evaluation.evaluate import evaluate_run
from rlagent.training.matrix import expand_jobs, load_spec
from rlagent.training.train import train

PHASE5 = Path(__file__).parent.parent / "configs" / "experiments" / "phase5.yaml"
COMMON = [
    "algo.device=cpu",
    "train.total_timesteps=600",
    "train.eval_freq=300",
    "train.n_eval_episodes=1",
    "env.n_envs=2",
    "algo.hyperparams.batch_size=64",
]
BY_ALGO = {
    "ppo": ["algo.hyperparams.n_steps=64"],
    "maskable_ppo": ["algo.hyperparams.n_steps=64"],
    "dqn": [
        "algo.hyperparams.learning_starts=100",
        "algo.hyperparams.buffer_size=2000",
        "algo.hyperparams.target_update_interval=200",
    ],
}
SPEC = load_spec(PHASE5)
FIRST_JOB = {j.group: j for j in reversed(expand_jobs(SPEC, "runs"))}  # seed 0 of each group


@pytest.mark.slow
@pytest.mark.parametrize("group", sorted(FIRST_JOB))
def test_group_trains_and_evaluates(group, tmp_path):
    job = FIRST_JOB[group]
    probe = load_config(job.base, list(job.overrides))
    overrides = [*job.overrides, *COMMON, *BY_ALGO[probe.algo.name]]
    if probe.curriculum is not None:
        overrides.append("curriculum.window=10")
    cfg = load_config(job.base, overrides)
    run_dir = train(cfg, tmp_path, verbose=False)
    for tag, extra in job.evals:
        result = evaluate_run(run_dir, 2, "final", overrides=list(extra), tag=tag)
        assert result["n_episodes"] == 2
    assert (run_dir / "eval_final_test.json").exists()
