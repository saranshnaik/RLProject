"""Watch a trained agent play: pick a model, a layout and step through the episode."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import cached, data, ui, viewer  # noqa: E402
from components.rollout import SPLIT_SPAN  # noqa: E402

ui.setup("Watch Agent")
st.write(
    "Pick a trained model and a layout, then step through the episode. The bars show what the "
    "policy believed at each state (for MaskablePPO, probabilities after invalid actions are "
    "masked out)."
)

models = cached.models()
groups = [g for g in cached.run_groups() if g.env == "Warehouse-v0" and data.runnable_seeds(g)]

with st.sidebar:
    st.header("Agent")
    sources = ["Curated model (models/)"] + (["Training run (runs/)"] if groups else [])
    source = st.radio("Source", sources, disabled=len(sources) == 1)

run_dir, which, label = None, None, ""
with st.sidebar:
    if source.startswith("Curated"):
        if not models:
            st.info("No curated models found in `models/`.")
        else:
            entry = st.selectbox("Model", models, format_func=lambda m: m.name)
            which = st.selectbox("Checkpoint", list(entry.checkpoints))
            run_dir, label = entry.path, entry.name
            with st.expander("Model card"):
                st.markdown(entry.readme or "_No README._")
    else:
        group = st.selectbox("Run group", groups, format_func=lambda g: g.label)
        seed = st.selectbox("Seed", data.runnable_seeds(group))
        run_dir = str(group.seed_dir(seed))
        which = st.selectbox("Checkpoint", data.available_checkpoints(run_dir))
        label = f"{group.name} / seed {seed}"
        st.caption("Runs still in training only have a `best` checkpoint so far.")

    st.header("Episode")
    split = st.radio(
        "Layouts",
        ["test", "train"],
        format_func=lambda s: "Held-out (test)" if s == "test" else "Training split",
        help="Held-out layouts come from a seed range the agent never trained on.",
    )
    if "watch_layout" not in st.session_state:
        st.session_state["watch_layout"] = 0
    st.number_input("Layout index", 0, SPLIT_SPAN - 1, key="watch_layout", step=1)
    st.button(
        "Random layout",
        on_click=lambda: st.session_state.update(watch_layout=random.randrange(10_000)),
    )
    episode_seed = st.number_input(
        "Item placement seed", 0, 1_000_000, 0, help="Which shelf cells hold the items."
    )
    deterministic = st.toggle("Deterministic actions", True)

if run_dir is None or which is None:
    st.info("Choose a model in the sidebar.")
    st.stop()

try:
    episode = cached.play_model_episode(
        run_dir,
        which,
        split,
        int(st.session_state["watch_layout"]),
        int(episode_seed),
        deterministic,
    )
except Exception as exc:  # noqa: BLE001 - show a message, never a traceback
    ui.error_box(f"play {label}", exc)
    st.stop()

st.subheader(f"{label} on {split} layout {st.session_state['watch_layout']}")
viewer.episode_summary(episode)
viewer.episode_viewer(episode, key="watch")
