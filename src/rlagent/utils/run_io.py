"""The run-directory contract.

runs/<env_id>/<algo>-<impl>/<exp_name>/seed_<k>/
    config.yaml  progress.csv  tb/  checkpoints/
    best_model.(zip|pt)  vecnormalize_best.pkl  final_model.(zip|pt)  vecnormalize.pkl
    metadata.json  eval_<best|final>.json

SB3 agents save ``.zip``; from-scratch agents save ``.pt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rlagent.config import ExperimentConfig, dump_config, load_config

CONFIG_FILE = "config.yaml"
BEST_MODEL = "best_model.zip"
FINAL_MODEL = "final_model.zip"
VECNORM_BEST = "vecnormalize_best.pkl"
VECNORM_FINAL = "vecnormalize.pkl"
MODEL_SUFFIXES = (".zip", ".pt")


def model_suffix(impl: str) -> str:
    return ".pt" if impl == "scratch" else ".zip"


def run_dir_for(root: str | Path, cfg: ExperimentConfig, seed: int | None = None) -> Path:
    seed = cfg.train.seed if seed is None else seed
    env_id = cfg.env.id.replace("/", "_")
    return Path(root) / env_id / f"{cfg.algo.name}-{cfg.algo.impl}" / cfg.name / f"seed_{seed}"


def save_config(cfg: ExperimentConfig, run_dir: Path) -> None:
    dump_config(cfg, run_dir / CONFIG_FILE)


def load_saved_config(run_dir: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    return load_config(Path(run_dir) / CONFIG_FILE, overrides)


def _find_model(run_dir: Path, stem: str) -> Path | None:
    for suffix in MODEL_SUFFIXES:
        if (run_dir / f"{stem}{suffix}").exists():
            return run_dir / f"{stem}{suffix}"
    return None


def resolve_artifacts(run_dir: str | Path, which: Literal["best", "final"] = "best"):
    """Return (model_path, vecnormalize_path_or_None), falling back to the final model."""
    run_dir = Path(run_dir)
    best = _find_model(run_dir, "best_model") if which == "best" else None
    if best is not None:
        model, vecnorm = best, run_dir / VECNORM_BEST
    else:
        model, vecnorm = _find_model(run_dir, "final_model"), run_dir / VECNORM_FINAL
    if model is None:
        raise FileNotFoundError(f"No model found in {run_dir}")
    return model, (vecnorm if vecnorm.exists() else None)
