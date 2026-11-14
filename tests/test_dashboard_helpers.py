"""Non-UI logic behind the Streamlit dashboard (dashboard/components)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dashboard"))

from components import data, paths, rollout, sandbox, shaping_demo, stats, tables  # noqa: E402

CURATED = REPO / "models" / "warehouse-maskable-ppo"


# -- stats -------------------------------------------------------------------------------------
def test_iqm_ignores_outliers():
    assert stats.iqm([0, 1, 1, 1, 1, 100]) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_iqm_and_handles_tiny_input():
    values = np.array([0.90, 0.92, 0.94, 0.96, 0.98])
    low, high = stats.bootstrap_iqm_ci(values, reps=500)
    assert low <= stats.iqm(values) <= high
    assert stats.bootstrap_iqm_ci([0.5]) == (0.5, 0.5)
    assert all(np.isnan(stats.bootstrap_iqm_ci([])))


def test_iqm_table_missing_column_is_empty():
    df = pd.DataFrame({"group": ["a", "a"], "x": [1, 2]})
    assert stats.iqm_table(df, "nope").empty
    table = stats.iqm_table(df, "x", reps=200)
    assert table.loc[0, "group"] == "a" and table.loc[0, "n_seeds"] == 2


def _curve(steps, values):
    return pd.DataFrame({"step": steps, "value": values})


def test_mean_ci_curve_two_seeds():
    a = _curve([0, 100], [0.0, 1.0])
    b = _curve([0, 100], [0.0, 0.5])
    out = stats.mean_ci_curve([a, b], n_points=11)
    assert out["mean"].iloc[-1] == pytest.approx(0.75)
    assert (out["high"] >= out["mean"]).all() and (out["low"] <= out["mean"]).all()
    assert out["n_seeds"].eq(2).all()
    assert out["high"].iloc[-1] > out["low"].iloc[-1]


def test_mean_ci_curve_single_seed_has_no_band_and_unfinished_seed_is_truncated():
    one = stats.mean_ci_curve([_curve([0, 10], [0, 1])], n_points=5)
    assert (one["high"] == one["low"]).all()
    long = _curve([0, 100], [0, 1])
    short = _curve([0, 50], [0, 1])
    out = stats.mean_ci_curve([long, short], n_points=21)
    assert out["n_seeds"].iloc[0] == 2 and out["n_seeds"].iloc[-1] == 1
    assert out["step"].iloc[-1] == 100


def test_mean_ci_curve_empty_and_smooth():
    assert stats.mean_ci_curve([]).empty
    assert list(stats.smooth([1, 3, 5], 2)) == [1.0, 2.0, 4.0]
    assert list(stats.smooth([1, 3], 1)) == [1.0, 3.0]


# -- data discovery ----------------------------------------------------------------------------
def _make_run(root: Path, group: str, seed: int, model: str | None = None, progress=True):
    run = root / "Warehouse-v0" / "maskable_ppo-sb3" / group / f"seed_{seed}"
    run.mkdir(parents=True)
    if progress:
        pd.DataFrame({"time/total_timesteps": [100, 200], "eval/success_rate": [0.1, 0.5]}).to_csv(
            run / "progress.csv", index=False
        )
    if model:
        (run / "config.yaml").write_text("name: x\n")
        (run / model).write_bytes(b"x")
    return run


def test_paths_respect_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RLAGENT_RUNS_ROOT", str(tmp_path / "nowhere"))
    assert paths.runs_root() == tmp_path / "nowhere"
    monkeypatch.delenv("RLAGENT_RUNS_ROOT")
    assert paths.runs_root() == paths.REPO_ROOT / "runs"


def test_discovery_handles_missing_and_lists_groups(tmp_path):
    assert data.discover_run_groups(tmp_path / "missing") == []
    assert data.discover_models(tmp_path / "missing") == []
    _make_run(tmp_path, "g1", 0, "final_model.zip")
    _make_run(tmp_path, "g1", 1)  # in progress: no model
    _make_run(tmp_path, "g2", 0)
    (tmp_path / "Warehouse-v0" / "stray_file.txt").write_text("x")
    groups = data.discover_run_groups(tmp_path)
    assert [g.name for g in groups] == ["g1", "g2"]
    g1 = data.find_group(groups, "g1")
    assert g1.seeds == (0, 1) and data.runnable_seeds(g1) == [0]
    assert data.find_group(groups, "zzz") is None


def test_curated_model_discovery(tmp_path):
    model = tmp_path / "m1"
    model.mkdir()
    (model / "config.yaml").write_text("x: 1")
    (model / "best_model.zip").write_bytes(b"x")
    (model / "README.md").write_text("card")
    (tmp_path / "incomplete").mkdir()
    entries = data.discover_models(tmp_path)
    assert [e.name for e in entries] == ["m1"]
    assert entries[0].checkpoints == ("best",) and entries[0].readme == "card"


def test_curve_loading_tolerates_partial_and_missing_files(tmp_path):
    run = _make_run(tmp_path, "g", 0)
    assert data.progress_columns(run) == ["time/total_timesteps", "eval/success_rate"]
    curve = data.load_curve(run, "eval/success_rate")
    assert list(curve["value"]) == [0.1, 0.5] and data.last_step(run) == 200
    assert data.load_curve(run, "no/such") is None
    # half-written last line, as left by a run in progress
    with open(run / "progress.csv", "a") as handle:
        handle.write("300")
    assert data.load_curve(run, "eval/success_rate") is not None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert data.load_curve(empty, "x") is None and data.last_step(empty) is None
    (empty / "progress.csv").write_text("")
    assert data.load_curve(empty, "x") is None
    assert data.read_csv_safe(tmp_path / "nope.csv") is None


def test_preferred_metric():
    assert data.preferred_metric(["a", "eval/success_rate", "rollout/ep_rew_mean"]) == (
        "eval/success_rate"
    )
    assert data.preferred_metric(["a", "b"]) == "a" and data.preferred_metric([]) is None


# -- tables ------------------------------------------------------------------------------------
SUMMARY = pd.DataFrame(
    [
        ("headline", "test", m, 5, 0.9, 0.0, i, lo, hi)
        for m, i, lo, hi in [
            ("success_rate", 1.0, 0.99, 1.0),
            ("mean_efficiency", 0.93, 0.92, 0.96),
            ("mean_length", 62.0, 58.0, 63.0),
        ]
    ],
    columns=["group", "split", "metric", "n_seeds", "mean", "std", "iqm", "ci_low", "ci_high"],
)
BASELINES = pd.DataFrame(
    [
        ("medium (12x12)", "random", 0.0, 308.0, 0.0),
        ("medium (12x12)", "optimal", 1.0, 55.0, 1.0),
        ("medium (12x12)", "greedy", 1.0, 62.0, 0.89),
        ("small (8x8)", "greedy", 1.0, 26.0, 1.0),
    ],
    columns=["setting", "agent", "success_rate", "mean_steps", "mean_efficiency"],
)


def test_fmt_interval():
    assert tables.fmt_interval(0.934, 0.92, 0.96) == "0.93 [0.92, 0.96]"
    assert tables.fmt_interval(0.5, float("nan"), float("nan")) == "0.50"
    assert tables.fmt_interval(float("nan"), 0, 1) == "-"


def test_wide_table_and_missing_group():
    wide = tables.wide_table(SUMMARY, ["headline", "ghost"], "test")
    assert wide.loc[0, "Efficiency"] == "0.93 [0.92, 0.96]" and wide.loc[0, "Seeds"] == 5
    assert wide.loc[1, "Efficiency"] == "-"


def test_rl_vs_baselines_orders_and_restricts_rl_to_medium():
    medium = tables.rl_vs_baselines(SUMMARY, BASELINES, "medium (12x12)")
    assert list(medium["agent"])[0] == "Random" and list(medium["kind"]).count("rl") == 1
    assert list(medium["agent"])[-1] == "Optimal"
    small = tables.rl_vs_baselines(SUMMARY, BASELINES, "small (8x8)")
    assert "rl" not in set(small["kind"])
    assert tables.rl_vs_baselines(None, None, "medium (12x12)").empty
    assert not tables.valid_summary(pd.DataFrame({"a": [1]}))


def test_committed_result_tables_have_the_expected_columns():
    summary = data.read_csv_safe(REPO / "results" / "phase5_summary.csv")
    baselines = data.read_csv_safe(REPO / "results" / "warehouse_baselines.csv")
    assert tables.valid_summary(summary) and tables.valid_baselines(baselines)
    frame = tables.rl_vs_baselines(summary, baselines, tables.baseline_settings(baselines)[1])
    assert {"Random", "Greedy", "Optimal"} <= set(frame["agent"])


# -- rollout / sandbox (no model) --------------------------------------------------------------
def test_baseline_episodes_and_png_roundtrip():
    params = sandbox.SandboxParams(height=8, width=8, n_items=2)
    episodes = sandbox.run_agents(params, ["greedy", "optimal"])
    assert set(episodes) == {"greedy", "optimal"}
    opt, greedy = episodes["optimal"], episodes["greedy"]
    assert opt.success and greedy.success and opt.steps <= greedy.steps
    assert opt.efficiency == pytest.approx(1.0)
    assert len(opt.frames) == opt.steps + 1 and opt.frames[-1].action is None
    assert rollout.decode_png(opt.frames[0].png).shape == (8 * 32, 8 * 32, 3)
    assert rollout.frames_to_gif(opt.frames[:5]).startswith(b"GIF8")
    table = sandbox.results_frame(episodes)
    assert list(table["Agent"]) == ["greedy", "optimal"]
    # same layout and seed -> same episode
    again = sandbox.run_agents(params, ["greedy"])["greedy"]
    assert again.steps == greedy.steps


def test_sandbox_skips_optimal_for_many_items_and_reports_bad_layouts():
    big = sandbox.SandboxParams(height=10, width=10, n_items=7)
    assert "optimal" not in sandbox.run_agents(big, ["optimal"])
    assert sandbox.check_params(sandbox.SandboxParams()) is None
    assert sandbox.check_params(sandbox.SandboxParams(height=4, width=4, n_items=20)) is not None


def test_oscillation_score():
    assert rollout.oscillation_score([(0, 0), (0, 1)]) == 0.0
    assert rollout.oscillation_score([(0, 0), (0, 1), (0, 0), (0, 1), (0, 0)]) == 1.0
    assert rollout.oscillation_score([(0, 0), (0, 1), (0, 2), (0, 3)]) == 0.0


def test_shaping_demo_naive_pays_for_oscillation_and_pbrs_does_not():
    setup = shaping_demo.DemoSetup(size=8, n_items=2)
    naive = shaping_demo.oscillation_return({"shaping": "naive", "naive_bonus": 1.0}, setup)
    pbrs = shaping_demo.oscillation_return({"shaping": "pbrs"}, setup)
    sparse = shaping_demo.oscillation_return({"shaping": "none"}, setup)
    assert naive["shaping"] > 10 and naive["shaping"] > abs(pbrs["shaping"]) * 5
    assert sparse["shaping"] == 0.0
    table = shaping_demo.hacking_table(setup)
    assert len(table) == 4 and table["hacking pays"].dtype == bool


# -- the committed model ------------------------------------------------------------------------
@pytest.mark.skipif(not (CURATED / "final_model.zip").exists(), reason="no curated model")
def test_curated_model_plays_a_held_out_episode_on_cpu():
    policy = rollout.load_policy(CURATED, "final")
    assert str(policy.model.device) == "cpu" and policy.kind == "maskable"
    env = rollout.build_env(policy.cfg, "test")
    episode = rollout.play_episode(
        rollout.PolicyActor(policy), env, rollout.layout_seed_for("test", 0)
    )
    env.close()
    assert episode.success and episode.invalid_actions == 0
    first = episode.frames[0]
    assert first.probs is not None and first.probs.shape == (6,)
    assert first.probs.sum() == pytest.approx(1.0, abs=1e-4)
    ok, _ = sandbox.model_fits(sandbox.SandboxParams(), policy)
    big_ok, why = sandbox.model_fits(sandbox.SandboxParams(height=16, width=16), policy)
    assert ok and not big_ok and "12x12" in why
