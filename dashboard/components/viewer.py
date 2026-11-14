"""Streamlit widgets that display a played episode: summary, stepper, auto-play, GIF, table."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from . import plots
from .rollout import ACTION_NAMES, Episode, frames_to_gif, oscillation_score

IMAGE_WIDTH = 440


def episode_summary(ep: Episode) -> None:
    cols = st.columns(5)
    cols[0].metric("Outcome", "Success" if ep.success else "Failed (time limit)")
    cols[1].metric("Steps", ep.steps, help="Optimal planner: " + _fmt(ep.optimal_steps))
    cols[2].metric(
        "Efficiency",
        "-" if ep.efficiency is None else f"{ep.efficiency:.2f}",
        help="Optimal steps / agent steps (0 if the episode failed).",
    )
    cols[3].metric("Return", f"{ep.total_return:.2f}")
    cols[4].metric("Invalid actions", ep.invalid_actions)


def _fmt(value) -> str:
    return "-" if value is None else str(value)


def _draw(ep: Episode, index: int, show_probs: bool) -> None:
    frame = ep.frames[index]
    n = len(ep.frames) - 1
    left, right = st.columns([5, 6])
    with left:
        st.image(frame.png, width=IMAGE_WIDTH)
    with right:
        a, b, c, d = st.columns(4)
        a.metric("Step", f"{index} / {n}")
        b.metric("Return", f"{frame.total_return:.2f}")
        c.metric("Last reward", f"{frame.reward:+.2f}")
        d.metric("Delivered", f"{frame.n_delivered}  (carrying {frame.n_carried})")
        if frame.action is None:
            st.caption("Final state of the episode." if index else "Empty episode.")
        else:
            name = ACTION_NAMES[frame.action]
            st.markdown(f"**Next action:** {name}")
        if index > 0 and not frame.valid:
            st.warning("The previous action was invalid (the robot did not move or act).")
        if show_probs and frame.probs is not None:
            st.plotly_chart(
                plots.action_figure(frame.probs, frame.action, ep.prob_label),
                width="stretch",
            )
        elif show_probs and frame.action is not None:
            st.caption("This agent does not expose action probabilities.")
        comps = {k: v for k, v in frame.components.items() if abs(v) > 1e-9}
        if comps:
            st.caption("Reward on arrival: " + ", ".join(f"{k} {v:+.3f}" for k, v in comps.items()))


def episode_viewer(ep: Episode, key: str, show_probs: bool = True) -> None:
    """Step slider with prev / next buttons, auto-play, whole-episode GIF and a step table."""
    n = len(ep.frames) - 1
    step_key = f"{key}_step"
    if step_key not in st.session_state or st.session_state[step_key] > n:
        st.session_state[step_key] = 0

    def move(delta: int) -> None:
        st.session_state[step_key] = max(0, min(n, st.session_state[step_key] + delta))

    c1, c2, c3, c4, c5 = st.columns([1, 1, 6, 2, 2])
    c1.button("Prev", key=f"{key}_prev", on_click=move, args=(-1,), width="stretch")
    c2.button("Next", key=f"{key}_next", on_click=move, args=(1,), width="stretch")
    c3.slider("Step", 0, n, key=step_key, label_visibility="collapsed")
    fps = c4.slider("Speed (steps/s)", 1, 15, 6, key=f"{key}_fps", label_visibility="collapsed")
    play = c5.button("Auto-play", key=f"{key}_play", width="stretch")

    holder = st.empty()
    if play:
        for i in range(st.session_state[step_key], n + 1):
            with holder.container():
                _draw(ep, i, show_probs)
            time.sleep(1.0 / fps)
    else:
        with holder.container():
            _draw(ep, st.session_state[step_key], show_probs)

    with st.expander("Whole episode as a GIF"):
        if st.toggle("Build GIF", key=f"{key}_gif"):
            gif = frames_to_gif(ep.frames)
            st.image(gif, width=IMAGE_WIDTH)
            st.download_button("Download GIF", gif, f"{key}.gif", "image/gif", key=f"{key}_dl")
    with st.expander("Step-by-step table"):
        st.dataframe(step_table(ep), width="stretch", hide_index=True)


def step_table(ep: Episode) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(ep.frames):
        row = {
            "step": i,
            "row": f.pos[0],
            "col": f.pos[1],
            "reward": round(f.reward, 3),
            "return": round(f.total_return, 2),
            "valid": f.valid,
            "next action": None if f.action is None else ACTION_NAMES[f.action],
        }
        if f.probs is not None and f.action is not None:
            row[f"{ep.prob_label} of chosen"] = round(float(f.probs[f.action]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def hacking_stats(ep: Episode) -> None:
    """Numbers that make a shaping exploit visible: oscillation and where the return came from."""
    score = oscillation_score(ep.positions)
    cols = st.columns(3)
    cols[0].metric(
        "Back-and-forth moves",
        f"{score:.0%}",
        help="Share of moves that return to the cell occupied two steps earlier.",
    )
    shaping = ep.component_totals.get("shaping", 0.0)
    cols[1].metric("Shaping reward", f"{shaping:+.2f}")
    cols[2].metric("Task reward", f"{ep.total_return - shaping:+.2f}")
    st.plotly_chart(plots.components_figure(ep.component_totals), width="stretch")
