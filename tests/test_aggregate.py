"""Aggregation of finished matrix runs: bootstrap CIs, tables and figures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rlagent.evaluation.aggregate import aggregate, bootstrap_iqm_ci, collect, interval_table
from rlagent.training.matrix import load_spec

SPEC = {
    "name": "fake",
    "base": "unused.yaml",
    "groups": {"good": {}, "bad": {}, "missing": {}},
    "experiments": {
        "E1": {"title": "Good vs bad", "groups": ["bad", "good", "missing"]},
    },
}


def _write_run(root: Path, group: str, seed: int, success: float, efficiency: float, split="both"):
    run = root / "Env" / "algo-sb3" / group / f"seed_{seed}"
    run.mkdir(parents=True)
    steps = np.arange(1, 6) * 1000
    pd.DataFrame(
        {"time/total_timesteps": steps, "eval/success_rate": np.linspace(0, success, 5)}
    ).to_csv(run / "progress.csv", index=False)
    result = {"success_rate": success, "mean_efficiency": efficiency, "mean_length": 50.0}
    result.update({"mean_return": 30.0, "mean_invalid_actions": 0.0})
    (run / "eval_final.json").write_text(json.dumps(result))
    if split == "both":
        (run / "eval_final_test.json").write_text(json.dumps(result))


def _fixture(tmp_path):
    runs = tmp_path / "runs"
    for seed, (s, e) in enumerate([(0.95, 0.9), (1.0, 0.95), (0.9, 0.85), (0.98, 0.92)]):
        _write_run(runs, "good", seed, s, e)
    for seed, (s, e) in enumerate([(0.2, 0.1), (0.3, 0.15), (0.25, 0.1)]):
        _write_run(runs, "bad", seed, s, e)
    _write_run(runs, "missing", 0, 0.5, 0.5, split="train-only")  # no held-out eval -> skipped
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(SPEC))
    return runs, spec_path


def test_bootstrap_ci_properties():
    assert bootstrap_iqm_ci(np.array([0.7, 0.7, 0.7])) == pytest.approx((0.7, 0.7))
    assert bootstrap_iqm_ci(np.array([0.4])) == (0.4, 0.4)
    values = np.array([0.1, 0.5, 0.52, 0.55, 0.9])
    low, high = bootstrap_iqm_ci(values, reps=500)
    assert values.min() <= low <= high <= values.max()
    assert bootstrap_iqm_ci(values, reps=500) == (low, high)  # deterministic


def test_collect_skips_runs_without_heldout_eval(tmp_path):
    runs, spec_path = _fixture(tmp_path)
    df = collect(load_spec(spec_path), runs)
    assert set(df.group) == {"good", "bad"} and len(df) == 7
    table = interval_table(df, "test_success_rate", reps=300).set_index("group")
    assert table.loc["good", "iqm"] > table.loc["bad", "iqm"]
    assert table.loc["good", "n_seeds"] == 4


def test_aggregate_writes_tables_and_figures(tmp_path):
    runs, spec_path = _fixture(tmp_path)
    out = tmp_path / "results"
    summary = aggregate(spec_path, out, runs, reps=300)
    assert {"group", "split", "metric", "iqm", "ci_low", "ci_high"} <= set(summary.columns)
    for name in ("fake_summary.csv", "fake_summary.md", "fake_runs.csv", "fake_E1.png"):
        assert (out / name).stat().st_size > 0, name
    assert (out / "fake_E1_curves.png").stat().st_size > 0
    markdown = (out / "fake_summary.md").read_text()
    assert "Good vs bad" in markdown and "| good | 4 |" in markdown and "missing" not in markdown
