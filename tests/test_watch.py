"""Live progress display: status parsing and line formatting."""

from rlagent.utils.watch import find_runs, format_line, read_status, watch


def _fake_run(root, name="exp", done=False):
    run = root / "Env" / "algo" / name / "seed_3"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("train:\n  seed: 3\n  total_timesteps: 1000\n")
    (run / "progress.csv").write_text(
        "time/total_timesteps,time/time_elapsed,rollout/success_rate,eval/success_rate\n"
        "250,10,0.5,\n500,20,0.75,0.6\n"
    )
    if done:
        (run / "final_model.zip").write_bytes(b"x")
    return run


def test_status_fraction_eta_and_line(tmp_path):
    run = _fake_run(tmp_path)
    status = read_status(run)
    assert status.steps == 500 and status.total == 1000
    assert status.fraction == 0.5
    assert status.eta == 20  # 500 steps took 20 s, so 500 more take 20 s
    assert status.eval_success == 0.6  # last non-empty value
    line = format_line(status, 10)
    assert "50.0%" in line and "train  75%" in line and "eval  60%" in line


def test_find_runs_filters_and_incomplete_runs_are_skipped(tmp_path):
    run = _fake_run(tmp_path, "alpha")
    _fake_run(tmp_path, "beta")
    assert find_runs(tmp_path, "alpha") == [run]
    (run / "progress.csv").write_text("")  # half-written file
    assert read_status(run) is None


def test_watch_once_prints_one_line_per_run(tmp_path, capsys):
    _fake_run(tmp_path, done=True)
    watch(tmp_path, once=True)
    out = capsys.readouterr().out
    assert out.count("seed_3") == 1 and "done" in out
