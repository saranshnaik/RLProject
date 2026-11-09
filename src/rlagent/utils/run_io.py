"""The run-directory contract.

runs/<env_id>/<algo>-<impl>/<exp_name>/seed_<k>/
    config.yaml  progress.csv  tb/  checkpoints/
    best_model.zip  vecnormalize_best.pkl  final_model.zip  vecnormalize.pkl  metadata.json
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


def run_dir_for(root: str | Path, cfg: ExperimentConfig, seed: int | None = None) -> Path:
    seed = cfg.train.seed if seed is None else seed
    env_id = cfg.env.id.replace("/", "_")
    return Path(root) / env_id / f"{cfg.algo.name}-{cfg.algo.impl}" / cfg.name / f"seed_{seed}"


def save_config(cfg: ExperimentConfig, run_dir: Path) -> None:
    dump_config(cfg, run_dir / CONFIG_FILE)


def load_saved_config(run_dir: str | Path) -> ExperimentConfig:
    return load_config(Path(run_dir) / CONFIG_FILE)


def resolve_artifacts(run_dir: str | Path, which: Literal["best", "final"] = "best"):
    """Return (model_path, vecnormalize_path_or_None), falling back to the final model."""
    run_dir = Path(run_dir)
    if which == "best" and (run_dir / BEST_MODEL).exists():
        model, vecnorm = run_dir / BEST_MODEL, run_dir / VECNORM_BEST
    else:
        model, vecnorm = run_dir / FINAL_MODEL, run_dir / VECNORM_FINAL
    if not model.exists():
        raise FileNotFoundError(f"No model found in {run_dir} (looked for {model.name})")
    return model, (vecnorm if vecnorm.exists() else None)
