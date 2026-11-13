"""Live terminal progress bars for training runs, read from each run's ``progress.csv``."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

BAR_WIDTH = 28


@dataclass
class RunStatus:
    label: str
    steps: int
    total: int
    elapsed: float
    train_success: float | None
    eval_success: float | None
    stage: int | None
    done: bool

    @property
    def fraction(self) -> float:
        return min(self.steps / self.total, 1.0) if self.total else 0.0

    @property
    def eta(self) -> float | None:
        if self.done or self.steps <= 0 or self.elapsed <= 0:
            return None
        return self.elapsed * (self.total - self.steps) / self.steps


def _last(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = frame[column].dropna()
    return float(values.iloc[-1]) if len(values) else None


def read_status(run_dir: Path) -> RunStatus | None:
    """Status of one run, or None while its files are missing / half-written."""
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        frame = pd.read_csv(run_dir / "progress.csv")
    except (OSError, ValueError, yaml.YAMLError, pd.errors.EmptyDataError):
        return None
    steps = _last(frame, "time/total_timesteps")
    if steps is None:
        return None
    stage = _last(frame, "curriculum/stage")
    return RunStatus(
        label=f"{run_dir.parent.name}/seed_{cfg['train']['seed']}",
        steps=int(steps),
        total=int(cfg["train"]["total_timesteps"]),
        elapsed=_last(frame, "time/time_elapsed") or 0.0,
        train_success=_last(frame, "rollout/success_rate"),
        eval_success=_last(frame, "eval/success_rate"),
        stage=None if stage is None else int(stage),
        done=any((run_dir / f"final_model{s}").exists() for s in (".zip", ".pt")),
    )


def find_runs(root: Path, match: str = "") -> list[Path]:
    return sorted(p.parent for p in root.rglob("progress.csv") if match in str(p.parent))


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m{rest % 60:02d}s"


def format_line(status: RunStatus, label_width: int) -> str:
    filled = int(round(status.fraction * BAR_WIDTH))
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    parts = [
        f"{status.label:<{label_width}} {bar} {status.fraction:6.1%}",
        f"{status.steps / 1e6:5.2f}M/{status.total / 1e6:.1f}M",
    ]
    if status.train_success is not None:
        parts.append(f"train {status.train_success:4.0%}")
    if status.eval_success is not None:
        parts.append(f"eval {status.eval_success:4.0%}")
    if status.stage is not None:
        parts.append(f"stage {status.stage}")
    parts.append("done" if status.done else f"eta {_duration(status.eta)}")
    return "  ".join(parts)


def watch(root: Path, match: str = "", interval: float = 5.0, once: bool = False) -> None:
    """Redraw one progress bar per run in place until every run has finished (or Ctrl-C)."""
    os.system("")  # enables ANSI escape handling on legacy Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    drawn = 0
    try:
        while True:
            statuses = [s for p in find_runs(root, match) if (s := read_status(p))]
            width = max((len(s.label) for s in statuses), default=0)
            lines = [format_line(s, width) for s in statuses] or ["(no runs found yet)"]
            if drawn:
                sys.stdout.write(f"\x1b[{drawn}A\r\x1b[0J")  # move up and clear the old block
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            drawn = len(lines)
            if once or (statuses and all(s.done for s in statuses)):
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
