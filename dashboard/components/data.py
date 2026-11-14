"""Discovery and loading of results, curated models and training runs (no Streamlit here).

Nothing in this module walks the whole ``runs/`` tree or reads a ``progress.csv`` unless asked to;
discovery only lists directory names, and curve loading reads just the requested columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MODEL_SUFFIXES = (".zip", ".pt")
SEED_RE = re.compile(r"^seed_(\d+)$")
STEP_COLUMN = "time/total_timesteps"


# -- small tables in results/ ------------------------------------------------------------------
def read_csv_safe(path: str | Path, **kwargs) -> pd.DataFrame | None:
    """Read a CSV that may be missing, empty, or being written; None on any failure."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, **kwargs)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return df if len(df.columns) else None


def mtime(path: str | Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def list_files(directory: str | Path, pattern: str) -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob(pattern)) if directory.is_dir() else []


# -- curated models ----------------------------------------------------------------------------
def find_model_file(run_dir: str | Path, which: str) -> Path | None:
    """The saved model for ``which`` in ('final', 'best') inside a model or run directory."""
    for suffix in MODEL_SUFFIXES:
        candidate = Path(run_dir) / f"{which}_model{suffix}"
        if candidate.is_file():
            return candidate
    return None


def available_checkpoints(run_dir: str | Path) -> list[str]:
    return [w for w in ("final", "best") if find_model_file(run_dir, w) is not None]


@dataclass(frozen=True)
class ModelEntry:
    name: str
    path: str  # directory containing config.yaml and the model file
    checkpoints: tuple[str, ...]
    readme: str = ""


def discover_models(models_dir: str | Path) -> list[ModelEntry]:
    """Curated models: sub-folders of ``models/`` with a config.yaml and a saved model."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    entries = []
    for child in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        checkpoints = available_checkpoints(child)
        if not (child / "config.yaml").is_file() or not checkpoints:
            continue
        readme = child / "README.md"
        text = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
        entries.append(ModelEntry(child.name, str(child), tuple(checkpoints), text))
    return entries


# -- training runs -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunGroup:
    """A directory of ``seed_*`` runs: runs/<env>/<algo>-<impl>/<name>/."""

    env: str
    algo: str
    name: str
    path: str
    seeds: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"{self.env} / {self.algo} / {self.name}"

    def seed_dir(self, seed: int) -> Path:
        return Path(self.path) / f"seed_{seed}"


def discover_run_groups(runs_dir: str | Path) -> list[RunGroup]:
    """List run groups by directory name only (a few ``iterdir`` calls, no file reads)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    groups: list[RunGroup] = []
    try:
        for env_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            for algo_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
                for group_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
                    seeds = sorted(
                        int(m.group(1))
                        for p in group_dir.iterdir()
                        if p.is_dir() and (m := SEED_RE.match(p.name))
                    )
                    if seeds:
                        groups.append(
                            RunGroup(
                                env_dir.name,
                                algo_dir.name,
                                group_dir.name,
                                str(group_dir),
                                tuple(seeds),
                            )
                        )
    except OSError:
        return groups
    return groups


def find_group(groups: list[RunGroup], name: str) -> RunGroup | None:
    for group in groups:
        if group.name == name:
            return group
    return None


def runnable_seeds(group: RunGroup) -> list[int]:
    """Seeds of a group that already have a saved model (so they can be replayed)."""
    return [
        s
        for s in group.seeds
        if (group.seed_dir(s) / "config.yaml").is_file()
        and available_checkpoints(group.seed_dir(s))
    ]


def progress_columns(run_dir: str | Path) -> list[str]:
    df = read_csv_safe(Path(run_dir) / "progress.csv", nrows=0)
    return [] if df is None else list(df.columns)


def load_curve(run_dir: str | Path, column: str) -> pd.DataFrame | None:
    """One metric from ``progress.csv`` as columns ``step``, ``value`` (None if unavailable)."""
    df = read_csv_safe(
        Path(run_dir) / "progress.csv",
        usecols=lambda c: c in (STEP_COLUMN, column),
        on_bad_lines="skip",  # a run in progress can leave a half-written last line
    )
    if df is None or column not in df or STEP_COLUMN not in df:
        return None
    df = df.dropna(subset=[STEP_COLUMN, column])
    if df.empty:
        return None
    return df.rename(columns={STEP_COLUMN: "step", column: "value"})[["step", "value"]].reset_index(
        drop=True
    )


def load_group_curves(group: RunGroup, column: str) -> dict[int, pd.DataFrame]:
    out = {}
    for seed in group.seeds:
        curve = load_curve(group.seed_dir(seed), column)
        if curve is not None:
            out[seed] = curve
    return out


def last_step(run_dir: str | Path) -> int | None:
    """Latest logged training step of a run (None if it has no readable progress)."""
    df = read_csv_safe(Path(run_dir) / "progress.csv", usecols=[STEP_COLUMN], on_bad_lines="skip")
    if df is None or df.empty or df[STEP_COLUMN].dropna().empty:
        return None
    return int(df[STEP_COLUMN].dropna().max())


def preferred_metric(columns: list[str]) -> str | None:
    for candidate in (
        "eval/success_rate",
        "rollout/success_rate",
        "eval/mean_reward",
        "rollout/ep_rew_mean",
    ):
        if candidate in columns:
            return candidate
    return columns[0] if columns else None
