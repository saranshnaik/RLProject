"""Experiment-matrix runner.

A spec (``configs/experiments/*.yaml``) lists *groups*: a base config plus overrides, run for
several seeds. This module expands the spec into (group, seed) jobs, estimates how long they take,
runs them over a pool of worker processes, and evaluates every finished run on the standard
protocol (train layouts + held-out test layouts). Finished jobs are skipped, so re-running the
same command resumes an interrupted matrix.

Run directories follow the usual contract with the group name as the experiment name:
``runs/<env>/<algo>-<impl>/<group>/seed_<k>``, which is exactly what ``rlagent compare`` and
``rlagent aggregate`` read.
"""

from __future__ import annotations

import heapq
import json
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rlagent.config import load_config
from rlagent.utils.run_io import MODEL_SUFFIXES, run_dir_for


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroupSpec(_Strict):
    base: str | None = None  # defaults to the spec's base config
    seeds: list[int] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    est_sps: float | None = None  # expected env steps/s of one worker (for the time estimate)
    tier: Literal["core", "optional"] | None = None
    note: str = ""


class EvalSpec(_Strict):
    tag: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)


class ExperimentDef(_Strict):
    title: str
    groups: list[str]


class MatrixSpec(_Strict):
    name: str
    base: str
    workers: int = Field(4, ge=1)
    threads_per_worker: int = Field(2, ge=1)
    eval_episodes: int = Field(100, ge=1)
    eval_model: Literal["best", "final"] = "final"
    default_sps: float = Field(500, gt=0)
    eval_overhead_s: float = Field(90, ge=0)
    eval: list[EvalSpec] = Field(default_factory=lambda: [EvalSpec()])
    defaults: GroupSpec = Field(default_factory=GroupSpec)
    groups: dict[str, GroupSpec]
    experiments: dict[str, ExperimentDef] = Field(default_factory=dict)


def load_spec(path: str | Path) -> MatrixSpec:
    spec = MatrixSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    for key, experiment in spec.experiments.items():
        unknown = [g for g in experiment.groups if g not in spec.groups]
        if unknown:
            raise ValueError(f"Experiment {key} refers to unknown groups {unknown}")
    return spec


def as_overrides(mapping: dict[str, Any]) -> list[str]:
    """``{"a.b": [1, 2]}`` -> ``["a.b=[1, 2]"]`` (JSON is valid YAML, so ``null`` round-trips)."""
    return [f"{key}={json.dumps(value)}" for key, value in mapping.items()]


@dataclass(frozen=True)
class Job:
    group: str
    seed: int
    base: str
    overrides: tuple[str, ...]
    steps: int
    device: str
    est_seconds: float
    run_dir: Path
    tier: str = "core"
    evals: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())  # (tag, overrides)


def eval_file(which: str, tag: str) -> str:
    return f"eval_{which}{'_' + tag if tag else ''}.json"


def training_done(run_dir: Path) -> bool:
    """metadata.json is written last by ``train``, so it marks a fully finished training run."""
    done = (run_dir / "metadata.json").exists()
    return done and any((run_dir / f"final_model{s}").exists() for s in MODEL_SUFFIXES)


def job_done(job: Job, spec: MatrixSpec) -> bool:
    return training_done(job.run_dir) and all(
        (job.run_dir / eval_file(spec.eval_model, tag)).exists() for tag, _ in job.evals
    )


def expand_jobs(
    spec: MatrixSpec,
    runs_root: str | Path = "runs",
    only: list[str] | None = None,
    tier: str | None = None,
    device: str | None = None,
    seeds: list[int] | None = None,
) -> list[Job]:
    """Resolve and validate every (group, seed) job (bad configs fail here, before any training)."""
    evals = tuple((e.tag, tuple(as_overrides(e.overrides))) for e in spec.eval)
    jobs = []
    for group, g in spec.groups.items():
        if only and not any(fnmatch(group, pattern) for pattern in only):
            continue
        group_tier = g.tier or spec.defaults.tier or "core"
        if tier and group_tier != tier:
            continue
        base = g.base or spec.base
        merged = {**spec.defaults.overrides, **g.overrides}
        if device:
            merged["algo.device"] = device
        group_seeds = seeds or g.seeds or spec.defaults.seeds or [0]
        for seed in group_seeds:
            overrides = [*as_overrides(merged), f"name={json.dumps(group)}", f"train.seed={seed}"]
            cfg = load_config(base, overrides)
            sps = g.est_sps or spec.defaults.est_sps or spec.default_sps
            steps = cfg.train.total_timesteps
            jobs.append(
                Job(
                    group=group,
                    seed=seed,
                    base=base,
                    overrides=tuple(overrides),
                    steps=steps,
                    device=cfg.algo.device,
                    est_seconds=steps / sps + spec.eval_overhead_s,
                    run_dir=run_dir_for(runs_root, cfg, seed),
                    tier=group_tier,
                    evals=evals,
                )
            )
    return jobs


def estimate_makespan(durations: list[float], workers: int) -> float:
    """Wall time of a longest-job-first schedule over ``workers`` identical workers."""
    free = [0.0] * workers
    heapq.heapify(free)
    for duration in sorted(durations, reverse=True):
        heapq.heappush(free, heapq.heappop(free) + duration)
    return max(free)


def fmt_duration(seconds: float) -> str:
    hours, rest = divmod(int(round(seconds)), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m"


def measured_sps(run_dir: Path) -> float | None:
    """Training throughput (env steps/s) recorded in a finished run's metadata, if any."""
    try:
        meta = json.loads((run_dir / "metadata.json").read_text())
        steps = load_config(run_dir / "config.yaml").train.total_timesteps
        return steps / float(meta["train_seconds"])
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


def plan_table(jobs: list[Job], spec: MatrixSpec, workers: int) -> str:
    """Human-readable plan: per-group cost, what is already done, and the wall-clock estimate."""
    header = f"{'group':<18}{'seeds':>6}{'steps/run':>11}{'est/run':>9}{'todo':>6}"
    lines = [header + f"{'meas sps':>10}  tier"]
    todo: list[Job] = []
    for group in dict.fromkeys(j.group for j in jobs):
        group_jobs = [j for j in jobs if j.group == group]
        pending = [j for j in group_jobs if not job_done(j, spec)]
        todo += pending
        first = group_jobs[0]
        rates = [r for j in group_jobs if (r := measured_sps(j.run_dir))]
        measured = f"{sum(rates) / len(rates):.0f}" if rates else "-"
        lines.append(
            f"{group:<18}{len(group_jobs):>6}{first.steps / 1e6:>10.1f}M"
            f"{fmt_duration(first.est_seconds):>9}{len(pending):>6}{measured:>10}  {first.tier}"
        )
    total = sum(j.est_seconds for j in todo)
    wall = estimate_makespan([j.est_seconds for j in todo], workers)
    lines += [
        "",
        f"{len(todo)} of {len(jobs)} jobs to run on {workers} workers",
        f"compute {fmt_duration(total)} -> estimated wall time {fmt_duration(wall)}",
    ]
    return "\n".join(lines)


def run_job(job: Job, runs_root: str, eval_episodes: int, eval_model: str) -> dict[str, Any]:
    """Train (unless already trained) and evaluate one job. Runs inside a worker process."""
    from rlagent.evaluation.evaluate import evaluate_run
    from rlagent.training.train import train

    start = time.time()
    if not training_done(job.run_dir):
        train(load_config(job.base, list(job.overrides)), runs_root, overwrite=True, verbose=False)
    for tag, overrides in job.evals:
        if not (job.run_dir / eval_file(eval_model, tag)).exists():
            evaluate_run(
                job.run_dir,
                n_episodes=eval_episodes,
                which=eval_model,  # type: ignore[arg-type]
                overrides=list(overrides),
                tag=tag,
            )
    return {"group": job.group, "seed": job.seed, "seconds": round(time.time() - start, 1)}


@contextmanager
def _thread_env(threads: int) -> Iterator[None]:
    """Child processes inherit these, which keeps parallel workers from oversubscribing the CPU."""
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: str(threads) for k in keys})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_devices(jobs: list[Job]) -> None:
    if any(j.device.startswith("cuda") for j in jobs):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "These jobs request device=cuda but CUDA is not available. Turn the GPU on, "
                "or run with --device cpu (much slower for the CNN policies)."
            )


def run_matrix(
    spec: MatrixSpec,
    jobs: list[Job],
    runs_root: str | Path = "runs",
    workers: int | None = None,
    echo=print,
) -> list[dict[str, Any]]:
    """Run all unfinished jobs; returns one result dict per job (with ``error`` on failure)."""
    pending = [j for j in jobs if not job_done(j, spec)]
    workers = min(workers or spec.workers, max(len(pending), 1))
    results: list[dict[str, Any]] = []
    if not pending:
        echo("Nothing to do: every job is already complete.")
        return results
    check_devices(pending)
    pending.sort(key=lambda j: j.est_seconds, reverse=True)  # longest first
    with (
        _thread_env(spec.threads_per_worker),
        ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool,
    ):
        futures = {
            pool.submit(run_job, j, str(runs_root), spec.eval_episodes, spec.eval_model): j
            for j in pending
        }
        for n, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                result = future.result()
                echo(
                    f"[{n}/{len(pending)}] done  {job.group}/seed_{job.seed} "
                    f"({fmt_duration(result['seconds'])})"
                )
            except Exception:  # keep the rest of the matrix going
                result = {"group": job.group, "seed": job.seed, "error": traceback.format_exc()}
                echo(f"[{n}/{len(pending)}] FAILED {job.group}/seed_{job.seed}\n{result['error']}")
            results.append(result)
    return results


BENCH_VARIANTS: dict[str, dict[str, Any]] = {
    "gpu": {},
    "cpu": {"algo.device": "cpu"},
    "gpu_16env": {"env.n_envs": 16, "algo.hyperparams.n_steps": 64},  # half the forward passes
    "cpu_16env": {"algo.device": "cpu", "env.n_envs": 16, "algo.hyperparams.n_steps": 64},
}


def benchmark(
    base: str | Path,
    variants: list[str],
    worker_counts: list[int],
    steps: int = 100_000,
    threads: int = 2,
    root: str | Path | None = None,
    echo=print,
) -> list[dict[str, Any]]:
    """Measure aggregate env steps/s for each (variant, number of parallel workers).

    Every cell trains ``n`` short runs concurrently and averages their training times. The result
    shows which device / rollout shape / worker count gives the best total throughput, which
    calibrates the ``est_sps`` values in an experiment spec."""
    import tempfile

    root = Path(root or tempfile.mkdtemp(prefix="rlbench_"))
    rows = []
    for variant in variants:
        overrides = {
            "train.total_timesteps": steps,
            "train.eval_freq": steps * 10,  # no evaluation during the measurement
            **BENCH_VARIANTS[variant],
        }
        for n in worker_counts:
            group = f"{variant}_w{n}"
            spec = MatrixSpec(
                name="bench",
                base=str(base),
                workers=n,
                threads_per_worker=threads,
                groups={},
                eval=[],
            )
            jobs = []
            for seed in range(n):
                cfg_overrides = [*as_overrides(overrides), f"name={json.dumps(group)}"]
                cfg = load_config(base, [*cfg_overrides, f"train.seed={seed}"])
                jobs.append(
                    Job(
                        group,
                        seed,
                        str(base),
                        (*cfg_overrides, f"train.seed={seed}"),
                        steps,
                        cfg.algo.device,
                        0.0,
                        run_dir_for(root, cfg, seed),
                    )
                )
            results = run_matrix(spec, jobs, root, n, echo=lambda *_: None)
            errors = [r["error"] for r in results if "error" in r]
            if errors:
                raise RuntimeError(f"benchmark job failed:\n{errors[0]}")
            rates = [r for j in jobs if (r := measured_sps(j.run_dir))]
            per_worker = sum(rates) / len(rates)
            row = {
                "variant": variant,
                "workers": n,
                "sps_per_worker": per_worker,
                "sps_total": per_worker * n,
            }
            rows.append(row)
            echo(
                f"{variant:<10} workers={n}  {per_worker:7.0f} steps/s each  "
                f"{per_worker * n:7.0f} total"
            )
    return rows
