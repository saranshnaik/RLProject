"""Reward lab: the E1 shaping ablation and the reward-hacking demonstration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import cached, data, plots, stats, tables, ui, viewer  # noqa: E402
from components.paths import results_root  # noqa: E402

ui.setup("Reward Lab")

E1_GROUPS = ["e1_sparse", "e1_naive", "e1_naive_big", "ref3m"]
E1_LABELS = {
    "e1_sparse": "sparse (no shaping)",
    "e1_naive": "naive, bonus 0.1",
    "e1_naive_big": "naive, bonus 1.0 (hacking demo)",
    "ref3m": "PBRS (reference arm)",
}

st.markdown(
    "Reward shaping adds a hint on top of the task reward. **Potential-based shaping (PBRS)** "
    "adds `gamma * phi(s') - phi(s)`, which provably leaves the optimal policy unchanged "
    "(Ng et al., 1999). **Naive shaping** pays a bonus whenever a move gets closer to the goal "
    "and nothing for moving away. It is not potential-based, so a policy can farm it by "
    "stepping toward and away from the goal. That is reward hacking."
)

# -- E1 results ---------------------------------------------------------------------------------
st.subheader("E1: does shaping help, and can it hurt?")
summary = cached.results_csv("phase5_summary.csv")
if not tables.valid_summary(summary):
    ui.missing_file(results_root() / "phase5_summary.csv", "results table")
else:
    present = [g for g in E1_GROUPS if g in set(summary["group"])]
    table = tables.wide_table(summary, present, "test")
    table["Group"] = table["Group"].map(E1_LABELS)
    st.dataframe(table, width="stretch", hide_index=True)
    absent = [E1_LABELS[g] for g in E1_GROUPS if g not in present]
    if absent:
        st.caption(
            "Not in the results table yet (training or aggregation pending): "
            + ", ".join(absent)
            + "."
        )
    st.caption(
        "Held-out layouts, IQM [95% CI over seeds], 3M steps. Compare success and efficiency only: "
        "the naive arms add their bonus to the return, so returns are not comparable across arms. "
        "Finding so far: with the curriculum and masking in place, shaping barely matters, and the "
        "default naive bonus (0.1) was not exploited."
    )

groups = {g.name: g for g in cached.run_groups() if g.env == "Warehouse-v0"}
live = [name for name in E1_GROUPS if name in groups]

st.subheader("Learning curves")
if live:
    metrics = ["eval/success_rate", "rollout/efficiency", "reward/shaping", "rollout/ep_rew_mean"]
    metric = st.selectbox("Metric", metrics, help="Read from each run's progress.csv.")
    curves = {}
    for name in live:
        seeds = cached.group_curves(groups[name], metric)
        if seeds:
            curves[E1_LABELS[name]] = stats.mean_ci_curve(list(seeds.values()))
    if curves:
        st.plotly_chart(plots.curves_figure(curves, metric), width="stretch")
        st.caption(
            "Mean and 95% interval across seeds; runs still training stop at their latest step."
        )
    else:
        st.info(f"None of the E1 runs have logged `{metric}` yet.")
else:
    st.info("No E1 runs in `runs/` here, so these are the committed figures.")
    c1, c2 = st.columns(2)
    with c1:
        ui.show_image(results_root() / "phase5_E1_shaping_curves.png")
    with c2:
        ui.show_image(results_root() / "phase5_E1_shaping.png")

# -- scripted hacking demo ----------------------------------------------------------------------
st.subheader("Reward hacking, demonstrated with a script")
st.write(
    "No training needed: a fixed script steps toward the nearest item and straight back, never "
    "picking anything. The table compares what that loop earns (up to the time limit) with what "
    "the optimal planner earns by actually finishing the job, under each reward design."
)
c1, c2, c3 = st.columns(3)
size = c1.slider("Grid size", 8, 16, 12)
n_items = c2.slider("Items", 1, 5, 3)
layout_index = c3.number_input("Held-out layout", 0, 9999, 0)
try:
    table = cached.hacking_table(size, n_items, 2, int(layout_index))
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Under PBRS the loop earns about nothing, so there is no incentive to hack. Under naive "
        "shaping the loop earns `bonus / 2` per step for the whole time limit. Whether that beats "
        "finishing depends on the bonus: at 0.1 it does not, at 1.0 it does. This is why the "
        "default-bonus agents did not hack, and why `e1_naive_big` was added."
    )
except Exception as exc:  # noqa: BLE001
    ui.error_box("run the shaping demo", exc)

# -- replay of the trained hacking agent --------------------------------------------------------
st.subheader("Replay: the agent trained with a large naive bonus (e1_naive_big)")
big = groups.get("e1_naive_big")
if big is None:
    st.info(
        "Pending: `runs/Warehouse-v0/maskable_ppo-sb3/e1_naive_big` does not exist in this "
        "checkout. It is produced by `uv run rlagent matrix configs/experiments/phase5.yaml` "
        "(two seeds with `naive_bonus: 1.0`). The scripted demo above shows the exploit "
        "that this agent is expected to find; whether it actually does is not known yet."
    )
else:
    seeds = data.runnable_seeds(big)
    if not seeds:
        steps = [data.last_step(big.seed_dir(s)) for s in big.seeds]
        done = max((s for s in steps if s), default=None)
        st.info(
            "Pending: `e1_naive_big` is still training and has not saved a model yet"
            + (f" (latest logged step: {done:,})." if done else ".")
            + " Reload this page once it has a checkpoint."
        )
    else:
        seed = st.selectbox("Seed", seeds, key="hack_seed")
        run_dir = str(big.seed_dir(seed))
        which = st.selectbox("Checkpoint", data.available_checkpoints(run_dir), key="hack_ckpt")
        split = st.radio("Layouts", ["test", "train"], horizontal=True, key="hack_split")
        layout_index = st.number_input("Layout index", 0, 9999, 0, key="hack_layout")
        try:
            episode = cached.play_model_episode(run_dir, which, split, int(layout_index), 0, True)
        except Exception as exc:  # noqa: BLE001
            ui.error_box("replay e1_naive_big", exc)
        else:
            viewer.episode_summary(episode)
            viewer.hacking_stats(episode)
            viewer.episode_viewer(episode, key="hack")
            st.caption(
                "A hacking policy shows a high share of back-and-forth moves, a large shaping "
                "reward and a failed episode. A policy that learned the task shows a high "
                "task reward and almost no back-and-forth moves."
            )
