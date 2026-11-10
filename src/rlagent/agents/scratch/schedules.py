"""Hyperparameter schedules shared by SB3 and from-scratch agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Schedule = float | Callable[[float], float]


@dataclass
class LinearSchedule:
    """progress_remaining (1 -> 0) -> initial * progress_remaining. Picklable, unlike a lambda."""

    initial: float

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial


def parse_hyperparams(hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Turn 'lin_<x>' strings into linear schedules; leave everything else untouched."""
    parsed: dict[str, Any] = {}
    for key, value in hyperparams.items():
        if isinstance(value, str) and value.startswith("lin_"):
            value = LinearSchedule(float(value[4:]))
        parsed[key] = value
    return parsed


def value_at(schedule: Schedule, progress_remaining: float) -> float:
    return float(schedule(progress_remaining)) if callable(schedule) else float(schedule)
