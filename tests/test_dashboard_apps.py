"""Smoke tests: every dashboard page renders without an exception, with and without runs/."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = [
    "app.py",
    "pages/1_Watch_Agent.py",
    "pages/2_Learning_Curves.py",
    "pages/3_Compare_Agents.py",
    "pages/4_Reward_Lab.py",
    "pages/5_Sandbox.py",
]


@pytest.fixture(autouse=True)
def fresh_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _run(page: str) -> AppTest:
    return AppTest.from_file(str(DASHBOARD / page), default_timeout=120).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_runs_folder(page, monkeypatch, tmp_path):
    """Fresh clone: committed models/ and results/ only, no runs/ at all."""
    monkeypatch.setenv("RLAGENT_RUNS_ROOT", str(tmp_path / "does-not-exist"))
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_page_degrades_when_everything_is_missing(page, monkeypatch, tmp_path):
    """Missing results, models, figures and runs: messages instead of tracebacks."""
    for var in ("RUNS", "MODELS", "RESULTS", "FIGURES"):
        monkeypatch.setenv(f"RLAGENT_{var}_ROOT", str(tmp_path / f"no-{var.lower()}"))
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_page_with_a_run_in_progress(monkeypatch, tmp_path):
    """A group with only progress.csv (training, no model yet) must not break any page."""
    import pandas as pd

    run = tmp_path / "Warehouse-v0" / "maskable_ppo-sb3" / "e1_naive_big" / "seed_0"
    run.mkdir(parents=True)
    pd.DataFrame({"time/total_timesteps": [1000, 2000], "eval/success_rate": [0.1, 0.2]}).to_csv(
        run / "progress.csv", index=False
    )
    monkeypatch.setenv("RLAGENT_RUNS_ROOT", str(tmp_path))
    for page in ("pages/2_Learning_Curves.py", "pages/4_Reward_Lab.py", "pages/1_Watch_Agent.py"):
        at = _run(page)
        assert not at.exception, (page, [e.value for e in at.exception])
    reward_lab = _run("pages/4_Reward_Lab.py")
    assert any("still training" in i.value or "Pending" in i.value for i in reward_lab.info)
