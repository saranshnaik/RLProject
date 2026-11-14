"""Sandbox: generate a layout, run baselines and a trained model on it, or drive the robot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from components import cached, plots, sandbox, ui, viewer  # noqa: E402
from components.rollout import ACTION_NAMES, PolicyActor, encode_png  # noqa: E402
from components.sandbox import SandboxParams  # noqa: E402

ui.setup("Sandbox")
st.write(
    "Build a layout from a seed, then run the baselines and a trained model on it, or take "
    "control of the robot yourself. Layouts bigger than the trained model's 12x12 are still "
    "fine for the baselines."
)

models = cached.models()

with st.sidebar:
    st.header("Layout")
    split = st.radio(
        "Seed range",
        ["test", "train"],
        format_func=lambda s: "Held-out (test)" if s == "test" else "Training split",
    )
    layout_index = st.number_input("Layout seed", 0, 999_999, 0, step=1)
    item_seed = st.number_input("Item placement seed", 0, 1_000_000, 0, step=1)
    height = st.slider("Height", 4, 20, 12)
    width = st.slider("Width", 4, 20, 12)
    n_items = st.slider("Items", 1, 8, 3)
    capacity = st.slider("Carrying capacity", 1, 4, 2)
    density = st.slider("Shelf density", 0.2, 0.9, 0.7, 0.05)

params = SandboxParams(
    split, int(layout_index), int(item_seed), height, width, n_items, capacity, float(density)
)
problem = sandbox.check_params(params)
if problem:
    st.error(f"This layout cannot be generated: {problem}")
    st.stop()

signature = str(params)
tab_run, tab_drive = st.tabs(["Run agents", "Drive it yourself"])

# -- Run agents ---------------------------------------------------------------------------------
with tab_run:
    policy, fit_note = None, ""
    left, right = st.columns([2, 3])
    with left:
        picked = st.multiselect(
            "Baselines", ["random", "greedy", "optimal"], default=["greedy", "optimal"]
        )
        if params.n_items > sandbox.MAX_BASELINE_ITEMS and "optimal" in picked:
            st.caption(
                f"The optimal planner is skipped above {sandbox.MAX_BASELINE_ITEMS} items "
                "(exact search is exponential)."
            )
        use_model = False
        entry = None
        if models:
            use_model = st.checkbox("Trained model", True)
            if use_model:
                entry = st.selectbox("Model", models, format_func=lambda m: m.name)
        else:
            st.caption("No curated model in `models/`; baselines only.")
        run = st.button("Run on this layout", type="primary")
    with right:
        preview = sandbox.baseline_env(params)
        preview.reset(seed=params.item_seed, options={"layout_seed": params.layout_seed})
        st.image(encode_png(preview.render()), caption="Generated layout", width=380)
        preview.close()

    if run:
        try:
            if use_model and entry is not None:
                policy = cached.load_policy(entry.path, entry.checkpoints[0])
                ok, fit_note = sandbox.model_fits(params, policy)
                if not ok:
                    policy = None
            st.session_state["sandbox_results"] = (
                signature,
                sandbox.run_agents(params, picked, policy),
                fit_note,
            )
        except Exception as exc:  # noqa: BLE001
            ui.error_box("run the agents", exc)

    stored = st.session_state.get("sandbox_results")
    if stored and stored[0] == signature:
        _, episodes, note = stored
        if note:
            st.warning(f"Model skipped. {note}")
        if episodes:
            st.dataframe(sandbox.results_frame(episodes), width="stretch", hide_index=True)
            name = st.selectbox("Watch", list(episodes))
            viewer.episode_viewer(episodes[name], key="sandbox")
        else:
            st.info("Nothing to show: pick at least one agent.")
    elif stored:
        st.info("The layout settings changed. Press **Run on this layout** to update the results.")

# -- Drive it yourself --------------------------------------------------------------------------
with tab_drive:
    # An optional model to ask for advice. When it fits this layout the manual env is built with
    # the model's observation format, so the same state can be fed to it.
    drive_policy, drive_note = None, ""
    if models:
        model_choice = st.selectbox("Ask a model", models, format_func=lambda m: m.name, key="ask")
        try:
            candidate = cached.load_policy(model_choice.path, model_choice.checkpoints[0])
            fits, drive_note = sandbox.model_fits(params, candidate)
            drive_policy = candidate if fits else None
        except Exception as exc:  # noqa: BLE001
            drive_note = f"{type(exc).__name__}: {exc}"
    drive_sig = signature + (drive_policy.name if drive_policy else "")

    if st.session_state.get("manual_sig") != drive_sig:
        env = (
            sandbox.model_env(params, drive_policy)
            if drive_policy
            else sandbox.baseline_env(params)
        )
        obs, _ = env.reset(seed=params.item_seed, options={"layout_seed": params.layout_seed})
        st.session_state.update(
            manual_sig=drive_sig,
            manual_env=env,
            manual_obs=obs,
            manual_total=0.0,
            manual_done=False,
            manual_info={},
            manual_log=[],
            manual_hint=None,
        )
    env = st.session_state["manual_env"]

    def act(action: int) -> None:
        obs, reward, terminated, truncated, info = env.step(action)
        st.session_state["manual_obs"] = obs
        st.session_state["manual_total"] += reward
        st.session_state["manual_log"].append(ACTION_NAMES[action])
        st.session_state["manual_hint"] = None
        if terminated or truncated:
            st.session_state["manual_done"] = True
            st.session_state["manual_info"] = info

    def restart() -> None:
        st.session_state["manual_sig"] = None

    def ask_model(step: bool) -> None:
        actor = PolicyActor(drive_policy)
        action, probs = actor.act(env, st.session_state["manual_obs"])
        if step:
            act(action)
        st.session_state["manual_hint"] = (action, probs)

    done = st.session_state["manual_done"]
    left, right = st.columns([5, 6])
    with left:
        st.image(encode_png(env.render()), width=viewer.IMAGE_WIDTH)
    with right:
        a, b, c = st.columns(3)
        a.metric("Steps", f"{env.t} / {env.max_steps}")
        b.metric("Return", f"{st.session_state['manual_total']:.2f}")
        c.metric("Delivered", f"{env.n_delivered} / {env.n_items}  (carrying {env.n_carried})")

        masks = env.action_masks() if not done else np.zeros(6, dtype=bool)
        pad = st.columns(3)
        layout_buttons = {(0, 1): 0, (1, 0): 2, (1, 2): 3, (2, 1): 1}  # (row, col) -> action
        for row in range(3):
            cols = st.columns(3) if row else pad
            for col in range(3):
                action = layout_buttons.get((row, col))
                if action is not None:
                    cols[col].button(
                        ACTION_NAMES[action],
                        key=f"move_{action}",
                        on_click=act,
                        args=(action,),
                        disabled=not masks[action],
                        width="stretch",
                    )
        d1, d2, d3 = st.columns(3)
        d1.button(
            "Pick", key="pick", on_click=act, args=(4,), disabled=not masks[4], width="stretch"
        )
        d2.button(
            "Drop", key="drop", on_click=act, args=(5,), disabled=not masks[5], width="stretch"
        )
        d3.button("Restart", key="restart", on_click=restart, width="stretch")
        st.caption(
            "Buttons are disabled for invalid actions. Pick works on an item cell, Drop at the "
            "green packing station."
        )

        if drive_policy is not None:
            q1, q2 = st.columns(2)
            q1.button(
                "What would it do?",
                on_click=ask_model,
                args=(False,),
                disabled=done,
                width="stretch",
            )
            q2.button(
                "Let it act", on_click=ask_model, args=(True,), disabled=done, width="stretch"
            )
            hint = st.session_state["manual_hint"]
            if hint is not None and hint[1] is not None:
                st.plotly_chart(
                    plots.action_figure(hint[1], hint[0], "Probability"), width="stretch"
                )
        elif models:
            st.caption(f"The model is unavailable on this layout. {drive_note}")
        if done:
            info = st.session_state["manual_info"]
            if info.get("is_success"):
                st.success(
                    f"Order complete in {env.t} steps. The optimal planner needs "
                    f"{info.get('optimal_steps')} (efficiency {info.get('efficiency', 0):.2f})."
                )
            else:
                st.error("Time limit reached before all items were delivered.")
        if st.session_state["manual_log"]:
            st.caption("Moves: " + " ".join(st.session_state["manual_log"][-40:]))
