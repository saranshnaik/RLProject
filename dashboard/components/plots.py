"""Plotly figure builders (they take plain data and return a Figure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Okabe-Ito: distinguishable for the common kinds of colour-blindness, readable on light and dark.
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#999999",
)
ACTION_NAMES = ("Up", "Down", "Left", "Right", "Pick", "Drop")


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=title,
        margin={"l": 10, "r": 10, "t": 40 if title else 10, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    return fig


def curves_figure(
    curves: dict[str, pd.DataFrame], ylabel: str, title: str | None = None, height: int = 420
) -> go.Figure:
    """Mean line with a shaded CI band per label (frames have step, mean, low, high, n_seeds)."""
    fig = go.Figure()
    for i, (label, df) in enumerate(curves.items()):
        if df is None or df.empty:
            continue
        c = color(i)
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["high"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["low"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_rgba(c, 0.18),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["mean"],
                mode="lines",
                name=label,
                legendgroup=label,
                line={"color": c, "width": 2},
                customdata=df["n_seeds"],
                hovertemplate="%{y:.3f} (%{customdata} seeds)",
            )
        )
    fig.update_xaxes(title="Training steps")
    fig.update_yaxes(title=ylabel)
    return _layout(fig, height, title)


def interval_figure(
    table: pd.DataFrame,
    label_col: str,
    value_col: str,
    low_col: str | None = None,
    high_col: str | None = None,
    xlabel: str = "",
    points: pd.DataFrame | None = None,
    points_label_col: str | None = None,
    points_value_col: str | None = None,
    height: int | None = None,
    xrange: tuple[float, float] | None = None,
    highlight: dict[str, str] | None = None,
) -> go.Figure:
    """Horizontal dot-and-whisker chart (point = IQM, whiskers = interval) with optional seeds."""
    labels = list(table[label_col])
    colors = [(highlight or {}).get(lab, color(0)) for lab in labels]
    fig = go.Figure()
    error = None
    if low_col and high_col:
        error = {
            "type": "data",
            "symmetric": False,
            "array": (table[high_col] - table[value_col]).clip(lower=0),
            "arrayminus": (table[value_col] - table[low_col]).clip(lower=0),
            "thickness": 2,
            "width": 6,
        }
    fig.add_trace(
        go.Scatter(
            x=table[value_col],
            y=labels,
            mode="markers",
            name="IQM (95% CI)",
            marker={"size": 11, "color": colors},
            error_x=error,
        )
    )
    if points is not None and len(points):
        fig.add_trace(
            go.Scatter(
                x=points[points_value_col],
                y=points[points_label_col],
                mode="markers",
                name="Single seeds",
                marker={"size": 6, "color": "rgba(128,128,128,0.55)"},
            )
        )
    fig.update_xaxes(title=xlabel, range=xrange)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return _layout(fig, height or 90 + 38 * len(labels)).update_layout(hovermode="closest")


def action_figure(
    probs: np.ndarray | None,
    chosen: int | None,
    label: str = "Probability",
    masked: np.ndarray | None = None,
    height: int = 260,
) -> go.Figure:
    """Bar per action; the chosen action is highlighted, masked (invalid) actions are grey."""
    fig = go.Figure()
    if probs is not None:
        colors = []
        for i in range(len(probs)):
            if masked is not None and not masked[i]:
                colors.append("rgba(128,128,128,0.45)")
            elif i == chosen:
                colors.append(PALETTE[1])
            else:
                colors.append(PALETTE[0])
        fig.add_trace(
            go.Bar(
                x=list(ACTION_NAMES[: len(probs)]),
                y=probs,
                marker={"color": colors},
                text=[f"{p:.2f}" for p in probs],
                textposition="outside",
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(title=label, rangemode="tozero")
    fig.update_layout(showlegend=False, hovermode="closest")
    return _layout(fig, height).update_layout(hovermode="closest")


def components_figure(totals: dict[str, float], height: int = 260) -> go.Figure:
    """Where the episode return came from, by reward component."""
    keys = [k for k, v in totals.items() if abs(v) > 1e-9]
    values = [totals[k] for k in keys]
    fig = go.Figure(
        go.Bar(
            x=keys,
            y=values,
            marker={"color": [PALETTE[2] if v >= 0 else PALETTE[1] for v in values]},
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Summed reward")
    fig.update_layout(showlegend=False, hovermode="closest")
    return _layout(fig, height).update_layout(hovermode="closest")


def grouped_bars(
    table: pd.DataFrame,
    category_col: str,
    value_col: str,
    low_col: str | None = None,
    high_col: str | None = None,
    ylabel: str = "",
    yrange: tuple[float, float] | None = None,
    height: int = 340,
) -> go.Figure:
    error = None
    if low_col and high_col and low_col in table and high_col in table:
        error = {
            "type": "data",
            "symmetric": False,
            "array": (table[high_col] - table[value_col]).clip(lower=0).fillna(0),
            "arrayminus": (table[value_col] - table[low_col]).clip(lower=0).fillna(0),
        }
    fig = go.Figure(
        go.Bar(
            x=table[category_col],
            y=table[value_col],
            error_y=error,
            marker={"color": [color(i) for i in range(len(table))]},
            text=[f"{v:.2f}" for v in table[value_col]],
            textposition="outside",
        )
    )
    fig.update_yaxes(title=ylabel, range=yrange)
    fig.update_layout(showlegend=False, hovermode="closest")
    return _layout(fig, height).update_layout(hovermode="closest")
