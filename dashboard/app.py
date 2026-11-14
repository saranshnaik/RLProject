"""Dashboard home: headline result, summary table and the best GIFs.

Run with:  uv run streamlit run dashboard/app.py
Works from a fresh clone: it reads only the committed results/, models/ and docs/figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components import cached, tables, ui  # noqa: E402
from components.paths import figures_root, results_root  # noqa: E402

ui.setup("RL Warehouse Robot")
st.markdown(
    "A reinforcement-learning agent that picks items in a procedurally generated warehouse and "
    "delivers them to a packing station. It is trained with **MaskablePPO + a grid CNN** and "
    "compared with random, greedy and optimal-planner baselines on layouts it never saw in "
    "training. Use the pages in the sidebar to watch it, inspect learning curves, compare "
    "agents, explore reward design, or try your own layout."
)

summary = cached.results_csv("phase5_summary.csv")
baselines = cached.results_csv("warehouse_baselines.csv")

st.subheader("Headline result")
if tables.valid_summary(summary):
    success = tables.interval_row(summary, "headline", "test", "success_rate")
    eff = tables.interval_row(summary, "headline", "test", "mean_efficiency")
    greedy_eff = None
    if tables.valid_baselines(baselines):
        pick = baselines[
            baselines["setting"].str.startswith("medium") & (baselines["agent"] == "greedy")
        ]
        greedy_eff = None if pick.empty else float(pick["mean_efficiency"].iloc[0])
    if success is not None and eff is not None:
        a, b, c = st.columns(3)
        a.metric(
            "Held-out success (IQM)",
            f"{success['iqm']:.0%}",
            help=f"95% bootstrap CI [{success['ci_low']:.2f}, {success['ci_high']:.2f}], "
            f"{int(success['n_seeds'])} seeds, 100 episodes each, 12x12 layouts never trained on.",
        )
        b.metric(
            "Efficiency (IQM)",
            f"{eff['iqm']:.2f}",
            delta=None if greedy_eff is None else f"{eff['iqm'] - greedy_eff:+.2f} vs greedy",
            help="Optimal planner steps / agent steps. 1.00 is optimal.",
        )
        c.metric(
            "Seeds", int(success["n_seeds"]), help="Independent training runs (5M steps each)."
        )
        st.caption(
            f"Efficiency 95% CI [{eff['ci_low']:.2f}, {eff['ci_high']:.2f}]. "
            "Greedy planner efficiency on the same layouts: "
            + ("-" if greedy_eff is None else f"{greedy_eff:.2f}")
            + "."
        )
    else:
        st.info("The headline group is not in `results/phase5_summary.csv` yet.")

    st.subheader("All experiment groups (held-out layouts)")
    groups = list(dict.fromkeys(summary["group"]))
    st.dataframe(tables.wide_table(summary, groups, "test"), width="stretch", hide_index=True)
    st.caption(
        "IQM [95% bootstrap CI over seeds]. Ablation arms use 3 seeds and 3M steps, so their "
        "intervals understate the real uncertainty. Details: `docs/experiments.md`."
    )
else:
    ui.missing_file(results_root() / "phase5_summary.csv", "results table")

st.subheader("Agents in action")
gifs = [Path(p) for p in cached.figure_files("*.gif")] + [
    Path(p) for p in cached.result_files("*.gif")
]
captions = {
    "warehouse_maskable_ppo": "Trained agent (MaskablePPO) on a held-out layout",
    "warehouse_optimal": "Optimal planner on a held-out layout",
    "cartpole_ppo": "CartPole, PPO",
    "lunarlander_ppo": "LunarLander, PPO",
}
# Warehouse GIFs first: that is the headline.
gifs.sort(key=lambda p: (not p.stem.startswith("warehouse"), p.stem))
if gifs:
    cols = st.columns(min(len(gifs), 3))
    for i, gif in enumerate(gifs):
        with cols[i % len(cols)]:
            st.image(str(gif), caption=captions.get(gif.stem, gif.stem.replace("_", " ")))
else:
    ui.missing_file(figures_root(), "GIF folder")

headline_png = results_root() / "phase5_H_headline.png"
if headline_png.is_file():
    with st.expander("Headline seeds (figure)"):
        st.image(str(headline_png))

st.subheader("Pages")
st.markdown(
    "- **Watch Agent**: step through an episode and see the policy's action probabilities.\n"
    "- **Learning Curves**: mean and 95% CI across seeds (from `runs/` when present).\n"
    "- **Compare Agents**: RL vs greedy vs optimal vs random with IQM and bootstrap intervals.\n"
    "- **Reward Lab**: the reward-shaping ablation and the reward-hacking demo.\n"
    "- **Sandbox**: generate a layout and run any agent on it, or drive the robot yourself."
)
