"""Orchestrates one training run and writes the run directory."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from stable_baselines3.common.logger import (
    CSVOutputFormat,
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
)

from rlagent.agents.factory import build_agent
from rlagent.config import ExperimentConfig
from rlagent.training.callbacks import ScratchEvalHook, make_callbacks
from rlagent.training.env_factory import build_eval_env, build_train_env
from rlagent.utils.run_io import VECNORM_FINAL, model_suffix, run_dir_for, save_config
from rlagent.utils.seeding import EVAL_SEED_OFFSET, seed_everything


def _make_logger(run_dir: Path, verbose: bool) -> Logger:
    formats = [
        CSVOutputFormat(str(run_dir / "progress.csv")),
        TensorBoardOutputFormat(str(run_dir / "tb")),
    ]
    if verbose:
        formats.insert(0, HumanOutputFormat(sys.stdout))
    return Logger(str(run_dir), formats)


def train(
    cfg: ExperimentConfig,
    runs_root: str | Path = "runs",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Train one agent for ``cfg.train.seed`` and return its run directory."""
    seed = cfg.train.seed
    run_dir = run_dir_for(runs_root, cfg, seed)
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists (use --overwrite or another name)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir)

    seed_everything(seed)
    train_env = build_train_env(cfg, seed)
    eval_env = build_eval_env(cfg, seed + EVAL_SEED_OFFSET)
    try:
        model = build_agent(cfg.algo, train_env, seed)
        model.set_logger(_make_logger(run_dir, verbose))
        if cfg.algo.impl == "scratch":
            callbacks = ScratchEvalHook(cfg, run_dir, eval_env, verbose)
        else:
            callbacks = make_callbacks(cfg, run_dir, eval_env, verbose)

        start = time.time()
        model.learn(total_timesteps=cfg.train.total_timesteps, callback=callbacks)
        elapsed = time.time() - start

        model.save(str(run_dir / f"final_model{model_suffix(cfg.algo.impl)}"))
        venv = model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(run_dir / VECNORM_FINAL))
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {"seed": seed, "train_seconds": round(elapsed, 1), "device": str(model.device)},
                indent=2,
            )
        )
    finally:
        train_env.close()
        eval_env.close()
    return run_dir
