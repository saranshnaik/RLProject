"""Aggregate runs across seeds: overlaid learning curves + a summary table.

A *group* is a directory holding ``seed_*`` run dirs, e.g. ``runs/CartPole-v1/ppo-sb3/zoo``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import trim_mean  # noqa: E402

STEP_KEY = "time/total_timesteps"


def seed_dirs(group: str | Path) -> list[Path]:
    dirs = sorted(Path(group).glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    if not dirs:
        raise FileNotFoundError(f"No seed_* run directories under {group}")
    return dirs


def group_label(group: str | Path) -> str:
    group = Path(group)
    return f"{group.parent.name}/{group.name}"  # e.g. ppo-scratch/zoo


def load_curve(run_dir: Path, key: str = "eval/mean_reward") -> pd.DataFrame:
    df = pd.read_csv(run_dir / "progress.csv", usecols=lambda c: c in (STEP_KEY, key))
    return df.dropna(subset=[key, STEP_KEY]).rename(columns={STEP_KEY: "step", key: "value"})


def final_score(run_dir: Path, key: str = "eval/mean_reward") -> float:
    """Best-checkpoint 100-episode return if ``rlagent evaluate`` was run, else last eval."""
    eval_file = run_dir / "eval_best.json"
    if eval_file.exists():
        return float(json.loads(eval_file.read_text())["mean_return"])
    return float(load_curve(run_dir, key)["value"].iloc[-1])


def iqm(values: list[float] | np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of the scores."""
    return float(trim_mean(np.asarray(values, dtype=float), 0.25))


def summarize(groups: dict[str, list[Path]], key: str = "eval/mean_reward") -> pd.DataFrame:
    rows = []
    for label, runs in groups.items():
        scores = [final_score(r, key) for r in runs]
        rows.append(
            {
                "group": label,
                "n_seeds": len(scores),
                "mean": np.mean(scores),
                "std": np.std(scores),
                "iqm": iqm(scores),
                "min": np.min(scores),
                "max": np.max(scores),
            }
        )
    return pd.DataFrame(rows)


def plot_curves(
    groups: dict[str, list[Path]], out: Path, key: str = "eval/mean_reward", title: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, runs in groups.items():
        curves = [load_curve(r, key) for r in runs]
        last = min(c["step"].max() for c in curves)
        grid = np.linspace(0, last, 200)
        stacked = np.stack([np.interp(grid, c["step"], c["value"]) for c in curves])
        mean, std = stacked.mean(0), stacked.std(0)
        ax.plot(grid, mean, label=f"{label} (n={len(runs)})")
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def compare(
    group_dirs: list[str | Path],
    out_prefix: str | Path,
    key: str = "eval/mean_reward",
    title: str = "",
) -> pd.DataFrame:
    groups = {group_label(g): seed_dirs(g) for g in group_dirs}
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(groups, key)
    summary.to_csv(out_prefix.parent / f"{out_prefix.name}_summary.csv", index=False)
    plot_curves(groups, out_prefix.parent / f"{out_prefix.name}_curves.png", key, title)
    return summary
