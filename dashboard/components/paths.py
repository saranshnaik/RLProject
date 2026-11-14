"""Where the dashboard looks for data.

The defaults are the repository folders. Environment variables override them, which lets tests
(and curious users) point the dashboard at a different ``runs/`` tree, or at a missing one, without
touching the real folders.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _root(env_var: str, default: str) -> Path:
    value = os.environ.get(env_var)
    return Path(value) if value else REPO_ROOT / default


def runs_root() -> Path:
    return _root("RLAGENT_RUNS_ROOT", "runs")


def models_root() -> Path:
    return _root("RLAGENT_MODELS_ROOT", "models")


def results_root() -> Path:
    return _root("RLAGENT_RESULTS_ROOT", "results")


def figures_root() -> Path:
    return _root("RLAGENT_FIGURES_ROOT", "docs/figures")
