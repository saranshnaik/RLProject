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
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", help="Override the saved config, e.g. env.kwargs.split=test"),
    ] = None,
    tag: Annotated[str, typer.Option(help="Save as eval_<best|final>_<tag>.json")] = "",
) -> None:
    """Evaluate a trained run and print metrics (also saved as eval_<best|final>.json)."""
    from rlagent.evaluation.evaluate import evaluate_run

    result = evaluate_run(
        run_dir,
        n_episodes=episodes,
        which="final" if final else "best",
        deterministic=not stochastic,
        overrides=list(set_ or []),
        tag=tag,
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


@app.command()
def baseline(
    config: Annotated[Path, typer.Argument(help="Experiment YAML whose env section is used")],
    agent: Annotated[str, typer.Option(help="random | greedy | optimal")] = "optimal",
    episodes: Annotated[int, typer.Option(help="Number of episodes")] = 100,
    seed: Annotated[int, typer.Option(help="Evaluation seed")] = 12_345,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", help="Override, e.g. env.kwargs.split=test"),
    ] = None,
) -> None:
    """Evaluate a non-learning baseline (Warehouse-v0) in the config's environment."""
    from rlagent.evaluation.evaluate import evaluate_baseline

    cfg = load_config(config, list(set_ or []))
    typer.echo(json.dumps(evaluate_baseline(cfg, agent, episodes, seed), indent=2))


@app.command()
def compare(
    groups: Annotated[
        list[Path],
        typer.Argument(help="Group dirs holding seed_* runs, e.g. runs/CartPole-v1/ppo-sb3/zoo"),
    ],
    out: Annotated[Path, typer.Option(help="Output prefix, e.g. results/cartpole_ppo")] = Path(
        "results/compare"
    ),
    key: Annotated[str, typer.Option(help="progress.csv metric to plot")] = "eval/mean_reward",
    title: Annotated[str, typer.Option(help="Plot title")] = "",
) -> None:
    """Overlay learning curves across seeds and write a summary table."""
    from rlagent.evaluation.compare import compare as run_compare

    summary = run_compare(groups, out, key=key, title=title)
    typer.echo(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    typer.echo(f"Wrote {out}_curves.png and {out}_summary.csv")


@app.command()
def matrix(
    spec: Annotated[Path, typer.Argument(help="Experiment-matrix YAML")],
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Group name or glob (repeatable)")
    ] = None,
    tier: Annotated[str | None, typer.Option(help="core | optional")] = None,
    seeds: Annotated[
        str | None, typer.Option(help="Comma-separated seed override, e.g. 0,1")
    ] = None,
    workers: Annotated[int | None, typer.Option(help="Parallel processes (spec default)")] = None,
    device: Annotated[str | None, typer.Option(help="Override algo.device, e.g. cpu")] = None,
    runs_dir: Annotated[Path, typer.Option(help="Root folder for run output")] = Path("runs"),
    dry_run: Annotated[bool, typer.Option(help="Print the plan and time estimate only")] = False,
) -> None:
    """Train and evaluate every (group, seed) job of an experiment matrix. Resumable."""
    from rlagent.training.matrix import expand_jobs, load_spec, plan_table, run_matrix

    matrix_spec = load_spec(spec)
    seed_list = [int(s) for s in seeds.split(",")] if seeds else None
    jobs = expand_jobs(matrix_spec, runs_dir, only, tier, device, seed_list)
    if not jobs:
        typer.echo("No jobs match the given filters.")
        raise typer.Exit(1)
    n_workers = workers or matrix_spec.workers
    typer.echo(plan_table(jobs, matrix_spec, n_workers))
    if dry_run:
        return
    results = run_matrix(matrix_spec, jobs, runs_dir, n_workers, echo=typer.echo)
    failed = [r for r in results if "error" in r]
    if failed:
        typer.echo(f"{len(failed)} job(s) failed; re-run the same command to retry them.")
        raise typer.Exit(1)


@app.command()
def bench(
    base: Annotated[Path, typer.Option(help="Config to benchmark")] = Path(
        "configs/warehouse/maskable_grid.yaml"
    ),
    variants: Annotated[
        str, typer.Option(help="Comma-separated: gpu,cpu,gpu_16env,cpu_16env")
    ] = "gpu,gpu_16env,cpu",
    workers: Annotated[str, typer.Option(help="Comma-separated worker counts")] = "1,2,4,6",
    steps: Annotated[int, typer.Option(help="Env steps per measurement run")] = 100_000,
    out: Annotated[Path, typer.Option(help="CSV with the measurements")] = Path(
        "results/throughput.csv"
    ),
) -> None:
    """Measure training throughput (steps/s) per device, rollout shape and worker count."""
    import pandas as pd

    from rlagent.training.matrix import benchmark

    rows = benchmark(
        base,
        variants.split(","),
        [int(w) for w in workers.split(",")],
        steps,
        echo=typer.echo,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    best = max(rows, key=lambda r: r["sps_total"])
    typer.echo(
        f"Best aggregate: {best['variant']} with {best['workers']} workers = "
        f"{best['sps_total']:.0f} steps/s. Wrote {out}"
    )


@app.command()
def aggregate(
    spec: Annotated[Path, typer.Argument(help="Experiment-matrix YAML")],
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("results"),
    runs_dir: Annotated[Path, typer.Option(help="Root folder holding the runs")] = Path("runs"),
    reps: Annotated[int, typer.Option(help="Bootstrap resamples")] = 2000,
) -> None:
    """IQM + 95% bootstrap CI tables and figures for a finished (or partial) matrix."""
    from rlagent.evaluation.aggregate import aggregate as run_aggregate

    summary = run_aggregate(spec, out, runs_dir, reps)
    typer.echo(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    typer.echo(f"Wrote tables and figures to {out}/")


@app.command()
def tune(
    spec: Annotated[Path, typer.Argument(help="Tuning spec YAML (configs/sweeps/*.yaml)")],
    workers: Annotated[int, typer.Option(help="Parallel worker processes")] = 1,
    device: Annotated[str | None, typer.Option(help="Override algo.device, e.g. cpu")] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where trial runs and the study DB live")] = Path(
        "runs/_tune"
    ),
    out: Annotated[Path, typer.Option(help="Folder for the trials CSV and best config")] = Path(
        "results/tuning"
    ),
    report_only: Annotated[bool, typer.Option(help="Skip tuning; just write the report")] = False,
) -> None:
    """Optuna search; a worker runs trials until the study holds the spec's n_trials."""
    from rlagent.training.tune import (
        launch_workers,
        load_tune_spec,
        run_tuning,
        write_report,
    )

    tune_spec = load_tune_spec(spec)
    if not report_only:
        if workers > 1:
            if launch_workers(spec, workers, runs_dir, device) != 0:
                typer.echo("A tuning worker exited with an error.")
        else:
            run_tuning(tune_spec, runs_dir, device)
    best = write_report(tune_spec, runs_dir, out)
    typer.echo(f"Report written to {out}/ (best config: {best})")


@app.command()
def watch(
    root: Annotated[Path, typer.Argument(help="Folder to scan for runs")] = Path("runs"),
    match: Annotated[str, typer.Option("--match", "-m", help="Path must contain this")] = "",
    interval: Annotated[float, typer.Option(help="Refresh period in seconds")] = 5.0,
    once: Annotated[bool, typer.Option(help="Print one snapshot and exit")] = False,
) -> None:
    """Live progress bars for training runs (reads each run's progress.csv)."""
    from rlagent.utils.watch import watch as run_watch

    run_watch(root, match, interval, once)


if __name__ == "__main__":
    app()
