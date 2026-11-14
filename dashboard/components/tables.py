"""Reshape the committed result CSVs into the tables the pages display (pure pandas)."""

from __future__ import annotations

import math

import pandas as pd

METRIC_LABELS = {
    "success_rate": "Success",
    "mean_efficiency": "Efficiency",
    "mean_length": "Episode length",
    "mean_return": "Return",
    "mean_invalid_actions": "Invalid actions",
}
SUMMARY_COLUMNS = {"group", "split", "metric", "n_seeds", "mean", "iqm", "ci_low", "ci_high"}
BASELINE_COLUMNS = {"setting", "agent", "success_rate", "mean_steps", "mean_efficiency"}


def valid_summary(df: pd.DataFrame | None) -> bool:
    return df is not None and SUMMARY_COLUMNS <= set(df.columns) and len(df) > 0


def valid_baselines(df: pd.DataFrame | None) -> bool:
    return df is not None and BASELINE_COLUMNS <= set(df.columns) and len(df) > 0


def fmt_interval(point: float, low: float, high: float, digits: int = 2) -> str:
    if point is None or (isinstance(point, float) and math.isnan(point)):
        return "-"
    text = f"{point:.{digits}f}"
    if low is None or high is None or math.isnan(low) or math.isnan(high):
        return text
    return f"{text} [{low:.{digits}f}, {high:.{digits}f}]"


def interval_row(summary: pd.DataFrame, group: str, split: str, metric: str) -> pd.Series | None:
    match = summary[
        (summary["group"] == group) & (summary["split"] == split) & (summary["metric"] == metric)
    ]
    return None if match.empty else match.iloc[0]


def wide_table(
    summary: pd.DataFrame,
    groups: list[str],
    split: str = "test",
    metrics: tuple[str, ...] = ("success_rate", "mean_efficiency", "mean_length"),
) -> pd.DataFrame:
    """One row per group: seeds, then ``IQM [95% CI]`` text for each metric."""
    rows = []
    for group in groups:
        row: dict = {"Group": group}
        n_seeds = None
        for metric in metrics:
            r = interval_row(summary, group, split, metric)
            digits = 1 if metric in ("mean_length", "mean_return") else 2
            if r is None:
                row[METRIC_LABELS[metric]] = "-"
            else:
                n_seeds = int(r["n_seeds"])
                row[METRIC_LABELS[metric]] = fmt_interval(
                    r["iqm"], r["ci_low"], r["ci_high"], digits
                )
        row["Seeds"] = "-" if n_seeds is None else n_seeds
        rows.append(row)
    cols = ["Group", "Seeds", *[METRIC_LABELS[m] for m in metrics]]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def baseline_settings(baselines: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(baselines["setting"]))


def rl_vs_baselines(
    summary: pd.DataFrame | None,
    baselines: pd.DataFrame | None,
    setting: str,
    rl_group: str = "headline",
    rl_label: str = "RL agent (MaskablePPO)",
) -> pd.DataFrame:
    """Random / greedy / optimal rows for one setting plus the RL agent's IQM and interval.

    RL numbers exist only for the 12x12 / 3-item setting, so the RL row is added only for the
    setting whose name starts with ``medium``.
    """
    rows = []
    if valid_baselines(baselines):
        for _, b in baselines[baselines["setting"] == setting].iterrows():
            rows.append(
                {
                    "agent": str(b["agent"]).capitalize(),
                    "kind": "baseline",
                    "success": float(b["success_rate"]),
                    "success_lo": math.nan,
                    "success_hi": math.nan,
                    "efficiency": float(b["mean_efficiency"]),
                    "efficiency_lo": math.nan,
                    "efficiency_hi": math.nan,
                    "steps": float(b["mean_steps"]),
                }
            )
    if valid_summary(summary) and setting.startswith("medium"):
        s = interval_row(summary, rl_group, "test", "success_rate")
        e = interval_row(summary, rl_group, "test", "mean_efficiency")
        n = interval_row(summary, rl_group, "test", "mean_length")
        if s is not None and e is not None:
            rows.append(
                {
                    "agent": rl_label,
                    "kind": "rl",
                    "success": float(s["iqm"]),
                    "success_lo": float(s["ci_low"]),
                    "success_hi": float(s["ci_high"]),
                    "efficiency": float(e["iqm"]),
                    "efficiency_lo": float(e["ci_low"]),
                    "efficiency_hi": float(e["ci_high"]),
                    "steps": math.nan if n is None else float(n["iqm"]),
                }
            )
    order = {"Random": 0, "Greedy": 1, rl_label: 2, "Optimal": 3}
    rows.sort(key=lambda r: order.get(r["agent"], 9))
    return pd.DataFrame(rows)
