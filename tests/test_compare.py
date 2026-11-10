import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from rlagent.cli import app
from rlagent.evaluation.compare import iqm


def _fake_run(root, impl, seed, scale):
    run = root / "Env-v0" / f"ppo-{impl}" / "zoo" / f"seed_{seed}"
    run.mkdir(parents=True)
    steps = np.arange(1000, 6000, 1000)
    pd.DataFrame(
        {"time/total_timesteps": steps, "eval/mean_reward": steps * scale / 1000 + seed}
    ).to_csv(run / "progress.csv", index=False)
    (run / "eval_best.json").write_text(json.dumps({"mean_return": 10.0 * scale + seed}))


def test_iqm_ignores_outliers():
    assert iqm([0, 1, 1, 1, 100]) == pytest.approx(1.0)


def test_compare_cli_writes_summary_and_plot(tmp_path):
    for seed in range(3):
        _fake_run(tmp_path, "sb3", seed, 1.0)
        _fake_run(tmp_path, "scratch", seed, 2.0)
    prefix = tmp_path / "out" / "cmp"
    result = CliRunner().invoke(
        app,
        [
            "compare",
            str(tmp_path / "Env-v0" / "ppo-sb3" / "zoo"),
            str(tmp_path / "Env-v0" / "ppo-scratch" / "zoo"),
            "--out",
            str(prefix),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(tmp_path / "out" / "cmp_summary.csv").set_index("group")
    assert summary.loc["ppo-sb3/zoo", "n_seeds"] == 3
    assert summary.loc["ppo-sb3/zoo", "mean"] == pytest.approx(11.0)
    assert summary.loc["ppo-scratch/zoo", "mean"] == pytest.approx(21.0)
    assert (tmp_path / "out" / "cmp_curves.png").stat().st_size > 0
