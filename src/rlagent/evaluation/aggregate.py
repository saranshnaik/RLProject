"""Aggregate a finished experiment matrix into tables and figures.

For every group, the per-seed results (``eval_<model>[_<tag>].json`` written by ``rlagent matrix``)
are reduced to the mean, standard deviation, interquartile mean (IQM) and a 95% percentile-bootstrap
confidence interval of the IQM over seeds, following Agarwal et al. (2021, "Deep RL at the Edge of
the Statistical Precipice"). The IQM / bootstrap are implemented here instead of calling
``rliable.library`` because that module does not import with pandas 3 (its ``arch`` dependency
breaks); the IQM is the same ``scipy.stats.trim_mean(x, 0.25)`` that ``rliable.metrics`` uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlagent.evaluation.compare import iqm, plot_curves, seed_dirs  # noqa: E402
from rlagent.training.matrix import MatrixSpec, eval_file, load_spec  # noqa: E402

METRICS = ("success_rate", "mean_efficiency", "mean_length", "mean_return", "mean_invalid_actions")
SPLITS = {"train": "", "test": "test"}  # split name -> evaluation tag


def find_group_dir(runs_root: str | Path, group: str) -> Path | None:
    matches = sorted(Path(runs_root).glob(f"*/*/{group}"))
    return matches[0] if matches else None


def collect(spec: MatrixSpec, runs_root: str | Path = "runs") -> pd.DataFrame:
    """One row per finished (group, seed): ``<split>_<metric>`` columns for each evaluated split."""
    rows = []
    for group in spec.groups:
        group_dir = find_group_dir(runs_root, group)
        if group_dir is None:
            continue
        for run in seed_dirs(group_dir):
            row: dict = {"group": group, "seed": int(run.name.split("_")[1])}
            for split, tag in SPLITS.items():
                path = run / eval_file(spec.eval_model, tag)
                if path.exists():
                    data = json.loads(path.read_text())
                    row.update({f"{split}_{m}": data.get(m) for m in METRICS})
            if "test_success_rate" in row:  # skip runs that were not fully evaluated
                rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_iqm_ci(
    values: np.ndarray, reps: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the IQM, resampling seeds with replacement."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True)
    stats = np.apply_along_axis(iqm, 1, draws)
    alpha = (1 - ci) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def interval_table(df: pd.DataFrame, column: str, reps: int = 2000) -> pd.DataFrame:
    rows = []
    for group, frame in df.groupby("group", sort=False):
        values = frame[column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        low, high = bootstrap_iqm_ci(values, reps)
        rows.append(
            {
                "group": group,
                "n_seeds": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "iqm": iqm(values),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def long_summary(df: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        for metric in METRICS:
            column = f"{split}_{metric}"
            if column in df and df[column].notna().any():
                table = interval_table(df, column, reps)
                table.insert(1, "split", split)
                table.insert(2, "metric", metric)
                frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def markdown_table(summary: pd.DataFrame, groups: list[str]) -> str:
    """Held-out success / efficiency / episode length per group, "IQM [95% CI]"."""

    def cell(group: str, metric: str, digits: int) -> str:
        row = summary[
            (summary.group == group) & (summary.split == "test") & (summary.metric == metric)
        ]
        if row.empty:
            return "-"
        r = row.iloc[0]
        return f"{r.iqm:.{digits}f} [{r.ci_low:.{digits}f}, {r.ci_high:.{digits}f}]"

    lines = [
        "| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |",
        "|---|---|---|---|---|",
    ]
    for group in groups:
        seeds = summary[(summary.group == group) & (summary.split == "test")].n_seeds
        if seeds.empty:
            continue
        lines.append(
            f"| {group} | {int(seeds.max())} | {cell(group, 'success_rate', 2)} | "
            f"{cell(group, 'mean_efficiency', 2)} | {cell(group, 'mean_length', 0)} |"
        )
    return "\n".join(lines)


def plot_intervals(summary: pd.DataFrame, groups: list[str], title: str, out: Path) -> None:
    """Held-out success and efficiency per group: IQM point with its 95% bootstrap interval."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.9 * len(groups) + 1.8), sharey=True)
    for ax, metric, label in zip(
        axes, ("success_rate", "mean_efficiency"), ("success rate", "efficiency"), strict=True
    ):
        for y, group in enumerate(groups):
            row = summary[
                (summary.group == group) & (summary.split == "test") & (summary.metric == metric)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.errorbar(
                r.iqm,
                y,
                xerr=[[r.iqm - r.ci_low], [r.ci_high - r.iqm]],
                fmt="o",
                color=f"C{y}",
                capsize=4,
            )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(f"held-out {label} (IQM, 95% CI)")
        ax.grid(alpha=0.3)
    axes[0].set_yticks(range(len(groups)), groups)
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def aggregate(
    spec_path: str | Path,
    out_dir: str | Path = "results",
    runs_root: str | Path = "runs",
    reps: int = 2000,
) -> pd.DataFrame:
    """Write ``<out>/<name>_summary.csv``/``.md`` plus one interval and one curve figure per
    experiment. Groups without finished runs are skipped, so partial results still aggregate."""
    spec = load_spec(spec_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(spec, runs_root)
    if df.empty:
        raise FileNotFoundError(f"No evaluated runs found under {runs_root} for {spec.name}")
    df.to_csv(out_dir / f"{spec.name}_runs.csv", index=False)
    summary = long_summary(df, reps)
    summary.to_csv(out_dir / f"{spec.name}_summary.csv", index=False)

    sections = []
    for key, experiment in spec.experiments.items():
        groups = [g for g in experiment.groups if g in set(df.group)]
        if not groups:
            continue
        plot_intervals(summary, groups, experiment.title, out_dir / f"{spec.name}_{key}.png")
        curves = {g: seed_dirs(find_group_dir(runs_root, g)) for g in groups}  # type: ignore[arg-type]
        plot_curves(
            curves,
            out_dir / f"{spec.name}_{key}_curves.png",
            key="eval/success_rate",
            title=experiment.title,
        )
        sections.append(f"### {experiment.title}\n\n{markdown_table(summary, groups)}\n")
    (out_dir / f"{spec.name}_summary.md").write_text("\n".join(sections), encoding="utf-8")
    return summary
