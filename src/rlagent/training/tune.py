"""Hyperparameter search with Optuna (TPE sampler + median pruner).

The objective is the mean *efficiency* of the final policy (failed episodes count as 0, so it
rewards solving the task quickly), measured on fresh layouts of the **train** distribution so the
held-out test layouts are never used for model selection. Trials report ``eval/success_rate`` after
every periodic evaluation so poor ones are pruned early.

Studies live in a SQLite file, so several worker processes can share one study:
``rlagent tune spec.yaml --workers 4`` starts four of them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import yaml
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3.common.callbacks import BaseCallback

from rlagent.config import load_config
from rlagent.training.matrix import as_overrides


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParamSpec(_Strict):
    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None
    template: str | None = None  # e.g. "lin_{}" turns a learning rate into a linear schedule


class PruneSpec(_Strict):
    n_startup_trials: int = 5
    n_warmup_steps: int = 0


class TuneSpec(_Strict):
    study: str
    base: str
    n_trials: int = Field(20, ge=1)
    eval_episodes: int = Field(50, ge=1)
    seed: int = 0  # training seed used by every trial
    prune: PruneSpec = Field(default_factory=PruneSpec)
    overrides: dict[str, Any] = Field(default_factory=dict)  # the (cheaper) tuning problem
    space: dict[str, ParamSpec]
    tie: dict[str, str] = Field(default_factory=dict)  # {target: source}: copy a sampled value


def load_tune_spec(path: str | Path) -> TuneSpec:
    return TuneSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def suggest(trial: optuna.Trial, key: str, p: ParamSpec) -> Any:
    if p.type == "categorical":
        if not p.choices:
            raise ValueError(f"{key}: categorical parameters need `choices`")
        value = trial.suggest_categorical(key, p.choices)
    elif p.low is None or p.high is None:
        raise ValueError(f"{key}: float/int parameters need `low` and `high`")
    elif p.type == "int":
        value = trial.suggest_int(key, int(p.low), int(p.high))
    else:
        value = trial.suggest_float(key, p.low, p.high, log=p.log)
    return p.template.format(f"{value:.6g}") if p.template else value


def trial_overrides(spec: TuneSpec, trial: optuna.Trial, device: str | None = None) -> dict:
    """All config overrides for one trial: fixed problem + sampled values + tied values."""
    sampled = {key: suggest(trial, key, p) for key, p in spec.space.items()}
    merged = {**spec.overrides, **sampled}
    for target, source in spec.tie.items():
        merged[target] = sampled[source]
    if device:
        merged["algo.device"] = device
    return merged


class PruneCallback(BaseCallback):
    """Passed as the evaluation callback's ``callback_after_eval``: reports and may prune."""

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial

    def _on_step(self) -> bool:
        buffer = getattr(self.parent, "_is_success_buffer", [])
        score = float(np.mean(buffer)) if len(buffer) else float(self.parent.last_mean_reward)
        self.trial.report(score, self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def make_objective(spec: TuneSpec, runs_root: Path, device: str | None = None):
    from rlagent.evaluation.evaluate import evaluate_run
    from rlagent.training.train import train

    def objective(trial: optuna.Trial) -> float:
        overrides = trial_overrides(spec, trial, device)
        trial.set_user_attr("overrides", overrides)
        cfg = load_config(
            spec.base,
            [*as_overrides(overrides), f"name=trial_{trial.number}", f"train.seed={spec.seed}"],
        )
        run_dir = train(
            cfg, runs_root, overwrite=True, verbose=False, after_eval=PruneCallback(trial)
        )
        result = evaluate_run(run_dir, spec.eval_episodes, "final", save=False)
        trial.set_user_attr("success_rate", result["success_rate"])
        return float(result.get("mean_efficiency", result["success_rate"] or 0.0))

    return objective


def storage_url(spec: TuneSpec, runs_root: Path) -> str:
    runs_root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(runs_root / (spec.study + '.db')).as_posix()}"


def get_study(spec: TuneSpec, runs_root: Path) -> optuna.Study:
    return optuna.create_study(
        study_name=spec.study,
        storage=storage_url(spec, runs_root),
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=spec.prune.n_startup_trials,
            n_warmup_steps=spec.prune.n_warmup_steps,
        ),
    )


def run_tuning(
    spec: TuneSpec, runs_root: str | Path = "runs/_tune", device: str | None = None
) -> optuna.Study:
    """Run trials in this process until the study holds ``spec.n_trials`` finished trials."""
    runs_root = Path(runs_root)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = get_study(spec, runs_root)
    objective = make_objective(spec, runs_root / spec.study, device)
    finished = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    ]
    # Count before every trial (MaxTrialsCallback only checks afterwards, so it would always run
    # one trial too many). Parallel workers can overshoot by at most workers - 1 trials.
    while len(study.get_trials(deepcopy=False, states=finished)) < spec.n_trials:
        study.optimize(objective, n_trials=1, catch=(Exception,))  # a crash must not kill the study
    return study


def launch_workers(
    spec_path: str | Path, workers: int, runs_root: str | Path, device: str | None
) -> int:
    """Start ``workers`` independent processes on the same study; returns the worst exit code."""
    command = [sys.executable, "-m", "rlagent.cli", "tune", str(spec_path), "--workers", "1"]
    command += ["--runs-dir", str(runs_root)] + (["--device", device] if device else [])
    processes = [subprocess.Popen(command) for _ in range(workers)]
    return max(p.wait() for p in processes)


def write_report(spec: TuneSpec, runs_root: str | Path, out_dir: str | Path) -> Path:
    """Dump all trials to CSV and the best trial's overrides to a ready-to-use YAML."""
    study = get_study(spec, Path(runs_root))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe()
    trials.to_csv(out_dir / f"{spec.study}_trials.csv", index=False)
    best_path = out_dir / f"{spec.study}_best.yaml"
    completed = [t for t in study.trials if t.value is not None]
    if completed:
        best = max(completed, key=lambda t: t.value)
        payload = {
            "trial": best.number,
            "objective_mean_efficiency": round(best.value, 4),
            "success_rate": best.user_attrs.get("success_rate"),
            "overrides": best.user_attrs.get("overrides", best.params),
        }
        best_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return best_path
