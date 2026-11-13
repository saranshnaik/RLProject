"""Optuna tuning: search-space handling, tied values, pruning hook and the report."""

from pathlib import Path

import optuna
import yaml

from rlagent.training.tune import (
    ParamSpec,
    load_tune_spec,
    run_tuning,
    suggest,
    trial_overrides,
    write_report,
)

ROOT = Path(__file__).parent.parent
BASE = ROOT / "configs" / "warehouse" / "ppo_vector.yaml"


def _spec_file(tmp_path, n_trials=2):
    data = {
        "study": "tiny",
        "base": str(BASE),
        "n_trials": n_trials,
        "eval_episodes": 2,
        "prune": {"n_startup_trials": 5},
        "overrides": {
            "train.total_timesteps": 2048,
            "train.eval_freq": 1024,
            "train.n_eval_episodes": 2,
            "env.n_envs": 2,
            "algo.hyperparams.n_steps": 64,
            "algo.hyperparams.batch_size": 64,
            "env.kwargs.height": 6,
            "env.kwargs.width": 6,
            "env.kwargs.n_items": 1,
        },
        "space": {
            "algo.hyperparams.learning_rate": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-3,
                "log": True,
                "template": "lin_{}",
            },
            "algo.hyperparams.gamma": {"type": "categorical", "choices": [0.99, 0.995]},
        },
        "tie": {"env.kwargs.shaping_gamma": "algo.hyperparams.gamma"},
    }
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_sampling_template_and_tie(tmp_path):
    spec = load_tune_spec(_spec_file(tmp_path))
    trial = optuna.create_study().ask()
    overrides = trial_overrides(spec, trial, device="cpu")
    assert overrides["algo.hyperparams.learning_rate"].startswith("lin_")
    assert overrides["env.kwargs.shaping_gamma"] == overrides["algo.hyperparams.gamma"]
    assert overrides["algo.device"] == "cpu" and overrides["env.kwargs.height"] == 6
    int_value = suggest(optuna.create_study().ask(), "n", ParamSpec(type="int", low=3, high=10))
    assert 3 <= int_value <= 10


def test_tuning_runs_trials_reports_and_writes_best_config(tmp_path):
    spec = load_tune_spec(_spec_file(tmp_path, n_trials=2))
    runs = tmp_path / "_tune"
    study = run_tuning(spec, runs)
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert all(t.intermediate_values for t in study.trials)  # the after-eval hook reported
    assert all(0.0 <= t.value <= 1.0 for t in study.trials)

    best = write_report(spec, runs, tmp_path / "out")
    payload = yaml.safe_load(best.read_text())
    assert "algo.hyperparams.learning_rate" in payload["overrides"]
    assert (tmp_path / "out" / "tiny_trials.csv").exists()

    # a second invocation finds the finished study and adds nothing
    assert len(run_tuning(spec, runs).trials) == 2
