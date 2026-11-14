"""Statistics used by the dashboard: IQM, bootstrap intervals and mean +/- CI learning curves."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import trim_mean


def iqm(values) -> float:
    """Interquartile mean: the mean of the middle 50% of the scores."""
    arr = np.asarray(values, dtype=float)
    return float(trim_mean(arr, 0.25)) if len(arr) else float("nan")


def bootstrap_iqm_ci(
    values, reps: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval of the IQM, resampling seeds with replacement."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) < 2:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(reps, len(arr)), replace=True)
    stat = trim_mean(draws, 0.25, axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(stat, alpha)), float(np.quantile(stat, 1 - alpha))


def iqm_table(df: pd.DataFrame, column: str, reps: int = 2000) -> pd.DataFrame:
    """Per-group seed count, mean, IQM and bootstrap interval of ``column``."""
    columns = ["group", "n_seeds", "mean", "iqm", "ci_low", "ci_high"]
    if column not in df or "group" not in df:
        return pd.DataFrame(columns=columns)
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
                "mean": float(values.mean()),
                "iqm": iqm(values),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def smooth(values, window: int) -> np.ndarray:
    """Trailing moving average (window in logged points); window <= 1 returns the input."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) == 0:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def mean_ci_curve(
    curves: list[pd.DataFrame], n_points: int = 200, confidence: float = 0.95, window: int = 1
) -> pd.DataFrame:
    """Combine per-seed curves (columns ``step``, ``value``) into a mean and a t-interval.

    Every seed is smoothed, then linearly interpolated onto a shared step grid. A seed contributes
    only up to its own last step, so an unfinished run does not distort the average after it ends
    (``n_seeds`` records how many seeds each point averages). With one seed the band collapses to
    the mean. Returns columns step, mean, low, high, n_seeds (empty if there is nothing to plot).
    """
    usable = [c.sort_values("step") for c in curves if c is not None and len(c) > 0]
    if not usable:
        return pd.DataFrame(columns=["step", "mean", "low", "high", "n_seeds"])
    start = max(float(c["step"].iloc[0]) for c in usable)
    stop = max(float(c["step"].iloc[-1]) for c in usable)
    if stop <= start:
        start = min(float(c["step"].iloc[0]) for c in usable)
    grid = np.linspace(start, stop, n_points) if stop > start else np.array([stop])
    matrix = np.full((len(usable), len(grid)), np.nan)
    for i, curve in enumerate(usable):
        steps = curve["step"].to_numpy(dtype=float)
        values = smooth(curve["value"].to_numpy(dtype=float), window)
        inside = (grid >= steps[0]) & (grid <= steps[-1])
        matrix[i, inside] = np.interp(grid[inside], steps, values)
    count = np.sum(~np.isnan(matrix), axis=0)
    keep = count > 0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty / single-seed columns
        mean = np.nanmean(matrix, axis=0)
        sd = np.nanstd(matrix, axis=0, ddof=1)
        tcrit = np.array([sps.t.ppf(0.5 + confidence / 2, max(n - 1, 1)) for n in count])
        half = np.where(count > 1, tcrit * sd / np.sqrt(np.maximum(count, 1)), 0.0)
    out = pd.DataFrame(
        {"step": grid, "mean": mean, "low": mean - half, "high": mean + half, "n_seeds": count}
    )
    return out[keep].reset_index(drop=True)
