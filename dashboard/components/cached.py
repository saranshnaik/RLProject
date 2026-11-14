"""Streamlit-cached wrappers around the plain helpers.

* Discovery lists directory names only and is cached for a few seconds, so a training job that adds
  runs shows up soon, without re-listing on every widget click.
* CSV tables are cached on (path, mtime), curves per group and metric on demand.
* Models are ``cache_resource`` objects (one per checkpoint); episodes are cached by their inputs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data, rollout
from .paths import figures_root, models_root, results_root, runs_root


@st.cache_data(ttl=15, show_spinner=False)
def models() -> list[data.ModelEntry]:
    return data.discover_models(models_root())


@st.cache_data(ttl=15, show_spinner=False)
def run_groups() -> list[data.RunGroup]:
    return data.discover_run_groups(runs_root())


@st.cache_data(show_spinner=False)
def _csv(path: str, stamp: float) -> pd.DataFrame | None:
    return data.read_csv_safe(path)


def results_csv(name: str) -> pd.DataFrame | None:
    path = results_root() / name
    return _csv(str(path), data.mtime(path))


def result_files(pattern: str) -> list[str]:
    return [str(p) for p in data.list_files(results_root(), pattern)]


def figure_files(pattern: str) -> list[str]:
    return [str(p) for p in data.list_files(figures_root(), pattern)]


@st.cache_data(ttl=30, show_spinner=False)
def group_curves(group: data.RunGroup, column: str) -> dict[int, pd.DataFrame]:
    """Curves of one metric for every seed of a group (reads only that column)."""
    return data.load_group_curves(group, column)


@st.cache_data(ttl=60, show_spinner=False)
def group_columns(group: data.RunGroup) -> list[str]:
    for seed in group.seeds:
        columns = data.progress_columns(group.seed_dir(seed))
        if columns:
            return columns
    return []


@st.cache_resource(max_entries=4, show_spinner="Loading model...")
def policy(run_dir: str, which: str, stamp: float) -> rollout.LoadedPolicy:
    return rollout.load_policy(run_dir, which)


def load_policy(run_dir: str, which: str) -> rollout.LoadedPolicy:
    model_file = data.find_model_file(run_dir, which)
    return policy(run_dir, which, data.mtime(model_file) if model_file else 0.0)


@st.cache_data(max_entries=24, show_spinner="Playing episode...")
def episode(
    run_dir: str,
    which: str,
    stamp: float,
    split: str,
    layout_index: int,
    episode_seed: int,
    deterministic: bool,
) -> rollout.Episode:
    loaded = policy(run_dir, which, stamp)
    env = rollout.build_env(loaded.cfg, split)
    try:
        return rollout.play_episode(
            rollout.PolicyActor(loaded),
            env,
            rollout.layout_seed_for(split, layout_index),
            episode_seed,
            deterministic,
        )
    finally:
        env.close()


def play_model_episode(
    run_dir: str, which: str, split: str, layout_index: int, episode_seed: int, deterministic: bool
) -> rollout.Episode:
    model_file = data.find_model_file(run_dir, which)
    stamp = data.mtime(model_file) if model_file else 0.0
    return episode(run_dir, which, stamp, split, layout_index, episode_seed, deterministic)


@st.cache_data(show_spinner=False)
def hacking_table(size: int, n_items: int, capacity: int, layout_index: int) -> pd.DataFrame:
    from .shaping_demo import DemoSetup, hacking_table

    return hacking_table(DemoSetup(size, n_items, capacity, layout_index))
