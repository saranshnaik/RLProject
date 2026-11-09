"""Record a rollout of a trained agent to a GIF / MP4."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import imageio.v2 as imageio

from rlagent.evaluation.evaluate import load_run


def record_run(
    run_dir: str | Path,
    out: str | Path,
    which: Literal["best", "final"] = "best",
    max_frames: int = 600,
    fps: int = 30,
    deterministic: bool = True,
    seed: int = 0,
) -> Path:
    model, env, _ = load_run(run_dir, which, render_mode="rgb_array", seed=seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        obs = env.reset()
        frames.append(env.render())
        while len(frames) < max_frames:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                break  # the env has already auto-reset; skip that frame
            frames.append(env.render())
    finally:
        env.close()
    if out.suffix.lower() == ".gif":
        imageio.mimsave(out, frames, duration=1000 / fps, loop=0)
    else:
        imageio.mimsave(out, frames, fps=fps)
    return out
