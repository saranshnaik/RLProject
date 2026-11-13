"""Experiment-matrix runner: expansion, time estimate, resume and an end-to-end tiny run."""

from pathlib import Path

import pytest
import yaml

from rlagent.training.matrix import (
    estimate_makespan,
    expand_jobs,
    job_done,
    load_spec,
    plan_table,
    run_matrix,
)

ROOT = Path(__file__).parent.parent
PHASE5 = ROOT / "configs" / "experiments" / "phase5.yaml"
BASE = ROOT / "configs" / "warehouse" / "ppo_vector.yaml"
TINY = {
    "train.total_timesteps": 1024,
    "train.eval_freq": 512,
    "train.n_eval_episodes": 2,
    "env.n_envs": 2,
    "algo.hyperparams.n_steps": 64,
    "algo.hyperparams.batch_size": 64,
    "env.kwargs.height": 6,
    "env.kwargs.width": 6,
    "env.kwargs.n_items": 1,
}


def _tiny_spec(tmp_path, **extra):
    data = {
        "name": "tiny",
        "base": str(BASE),
        "workers": 2,
        "threads_per_worker": 1,
        "eval_episodes": 2,
        "eval": [
            {"tag": "", "overrides": {}},
            {
                "tag": "test",
                "overrides": {"env.kwargs.split": "test", "env.kwargs.n_layouts": None},
            },
        ],
        "defaults": {"seeds": [0, 1], "overrides": TINY},
        "groups": {"alpha": {}, "beta": {"overrides": {"env.kwargs.reward.shaping": "none"}}},
        **extra,
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_phase5_spec_expands_to_valid_jobs():
    spec = load_spec(PHASE5)
    jobs = expand_jobs(spec, "runs")
    assert len(jobs) == 32 and len({j.run_dir for j in jobs}) == 32
    by_group = {}
    for job in jobs:
        by_group.setdefault(job.group, []).append(job)
    assert len(by_group["headline"]) == 5 and by_group["headline"][0].steps == 5_000_000
    assert by_group["ref3m"][0].steps == 3_000_000
    assert by_group["ref3m"][0].run_dir == Path("runs/Warehouse-v0/maskable_ppo-sb3/ref3m/seed_0")
    assert by_group["e2_dqn"][0].run_dir.parent.parent.name == "dqn-sb3"
    assert any("n_layouts=10" in o for o in by_group["e4_layouts10"][0].overrides)
    assert {j.tier for j in by_group["e5_no_curriculum"]} == {"optional"}


def test_filters_and_device_override():
    spec = load_spec(PHASE5)
    assert {j.group for j in expand_jobs(spec, only=["e4_*"])} == {"e4_layouts10", "e4_layouts100"}
    assert all(j.tier == "core" for j in expand_jobs(spec, tier="core"))
    assert len(expand_jobs(spec, only=["ref3m"], seeds=[7])) == 1
    cpu_jobs = expand_jobs(spec, only=["headline"], device="cpu")
    assert {j.device for j in cpu_jobs} == {"cpu"}


def test_makespan_is_longest_first_schedule():
    assert estimate_makespan([4, 3, 3, 2], workers=2) == 6  # {4,2} and {3,3}
    assert estimate_makespan([5], workers=4) == 5
    assert estimate_makespan([1, 1, 1, 1], workers=2) == 2


def test_unknown_group_in_experiment_is_rejected(tmp_path):
    path = _tiny_spec(tmp_path, experiments={"E": {"title": "t", "groups": ["nope"]}})
    with pytest.raises(ValueError, match="unknown groups"):
        load_spec(path)


def test_run_matrix_trains_evaluates_and_resumes(tmp_path):
    spec = load_spec(_tiny_spec(tmp_path))
    runs = tmp_path / "runs"
    jobs = expand_jobs(spec, runs)
    assert len(jobs) == 4 and not any(job_done(j, spec) for j in jobs)
    assert "4 of 4 jobs to run" in plan_table(jobs, spec, 2)

    messages = []
    results = run_matrix(spec, jobs, runs, echo=messages.append)
    assert len(results) == 4 and not any("error" in r for r in results), results
    for job in jobs:
        assert job_done(job, spec)
        assert (job.run_dir / "eval_final.json").exists()
        assert (job.run_dir / "eval_final_test.json").exists()
    assert "0 of 4 jobs to run" in plan_table(jobs, spec, 2)

    messages.clear()
    assert run_matrix(spec, jobs, runs, echo=messages.append) == []  # resumable no-op
    assert "Nothing to do" in messages[0]


def test_failed_job_is_reported_not_fatal(tmp_path):
    spec = load_spec(
        _tiny_spec(tmp_path, groups={"bad": {"overrides": {"algo.hyperparams.n_steps": 64}}})
    )
    jobs = expand_jobs(spec, tmp_path / "runs", seeds=[0])
    broken = type(jobs[0])(**{**jobs[0].__dict__, "base": str(tmp_path / "missing.yaml")})
    results = run_matrix(spec, [broken], tmp_path / "runs", echo=lambda *_: None)
    assert len(results) == 1 and "error" in results[0]


def test_benchmark_measures_throughput_for_each_cell(tmp_path):
    from rlagent.training.matrix import benchmark

    rows = benchmark(
        BASE,
        ["cpu"],
        [1, 2],
        steps=1024,
        threads=1,
        root=tmp_path,
        echo=lambda *_: None,
    )
    assert [(r["variant"], r["workers"]) for r in rows] == [("cpu", 1), ("cpu", 2)]
    assert all(r["sps_per_worker"] > 0 for r in rows)
    assert rows[1]["sps_total"] == pytest.approx(2 * rows[1]["sps_per_worker"])
