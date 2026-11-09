"""Command-line entry point."""

import json
from pathlib import Path
from typing import Annotated

import typer

from rlagent.config import load_config

app = typer.Typer(
    help="rlagent: train, evaluate, record and compare RL agents.", no_args_is_help=True
)

RunDir = Annotated[Path, typer.Argument(help="A run directory (runs/<env>/<algo>/<name>/seed_<k>)")]


@app.callback()
def _main() -> None:
    """Root command group."""


@app.command()
def train(
    config: Annotated[Path, typer.Argument(help="Experiment YAML")],
    seed: Annotated[int | None, typer.Option(help="Override train.seed")] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", help="Override, e.g. train.total_timesteps=5000"),
    ] = None,
    runs_dir: Annotated[Path, typer.Option(help="Root folder for run output")] = Path("runs"),
    overwrite: Annotated[bool, typer.Option(help="Replace an existing run dir")] = False,
    quiet: Annotated[bool, typer.Option(help="Suppress stdout training logs")] = False,
) -> None:
    """Train one agent from a YAML config."""
    from rlagent.training.train import train as run_train

    overrides = list(set_ or [])
    if seed is not None:
        overrides.append(f"train.seed={seed}")
    cfg = load_config(config, overrides)
    run_dir = run_train(cfg, runs_dir, overwrite=overwrite, verbose=not quiet)
    typer.echo(f"Run saved to {run_dir}")


@app.command()
def evaluate(
    run_dir: RunDir,
    episodes: Annotated[int, typer.Option(help="Number of evaluation episodes")] = 100,
    final: Annotated[bool, typer.Option(help="Use final model instead of best")] = False,
    stochastic: Annotated[bool, typer.Option(help="Sample actions instead of argmax")] = False,
) -> None:
    """Evaluate a trained run and print metrics (also saved as eval_<best|final>.json)."""
    from rlagent.evaluation.evaluate import evaluate_run

    result = evaluate_run(
        run_dir,
        n_episodes=episodes,
        which="final" if final else "best",
        deterministic=not stochastic,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def record(
    run_dir: RunDir,
    out: Annotated[Path, typer.Option(help="Output .gif or .mp4")] = Path("rollout.gif"),
    final: Annotated[bool, typer.Option(help="Use final model instead of best")] = False,
    max_frames: Annotated[int, typer.Option(help="Frame cap")] = 600,
    fps: Annotated[int, typer.Option(help="Frames per second")] = 30,
) -> None:
    """Record one episode of a trained agent."""
    from rlagent.evaluation.record import record_run

    path = record_run(
        run_dir, out, which="final" if final else "best", max_frames=max_frames, fps=fps
    )
    typer.echo(f"Saved {path}")


if __name__ == "__main__":
    app()
