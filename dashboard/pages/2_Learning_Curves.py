"""Learning curves: mean and 95% CI across seeds from runs/, or the committed figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import cached, data, plots, stats, ui  # noqa: E402

ui.setup("Learning Curves")

FIGURE_TITLES = {
    "phase5_H_headline_curves": "Headline agent (5 seeds)",
    "phase5_E1_shaping_curves": "E1: reward shaping",
    "phase5_E2_ppo_vs_dqn_curves": "E2: PPO vs DQN",
    "phase5_E3_observation_curves": "E3: observation design",
    "phase5_E4_generalization_curves": "E4: training-layout diversity",
    "phase5_E5_curriculum_curves": "E5: curriculum on vs off",
    "phase5_E6_masking_curves": "E6: masking vs penalty",
    "cartpole_ppo_curves": "CartPole, PPO (SB3 vs scratch)",
    "cartpole_dqn_curves": "CartPole, DQN (SB3 vs scratch)",
    "lunarlander_ppo_curves": "LunarLander, PPO (SB3 vs scratch)",
}


def committed_figures() -> None:
    files = [Path(p) for p in cached.result_files("*_curves.png")]
    if not files:
        st.info("No learning-curve figures are available in `results/`.")
        return
    choice = st.selectbox(
        "Figure", files, format_func=lambda p: FIGURE_TITLES.get(p.stem, p.stem.replace("_", " "))
    )
    ui.show_image(choice)
    st.caption("Mean across seeds with a shaded band, drawn when the experiments were aggregated.")


groups = cached.run_groups()
if not groups:
    st.info(
        "No `runs/` folder was found, so this page shows the committed figures from `results/`. "
        "After training (for example `uv run rlagent train ...`) interactive curves appear here."
    )
    committed_figures()
    st.stop()

with st.sidebar:
    st.header("Curves")
    default = [g for g in groups if g.name == "headline"][:1] or groups[:1]
    chosen = st.multiselect(
        "Run groups",
        groups,
        default=default,
        format_func=lambda g: g.label,
        max_selections=6,
        help="Reads only the selected metric from each seed's progress.csv.",
    )
    window = st.slider("Smoothing (logged points)", 1, 30, 1)
    show_seeds = st.checkbox("Show individual seeds", False)

if not chosen:
    st.info("Select at least one run group in the sidebar.")
else:
    columns = sorted({c for g in chosen for c in cached.group_columns(g)} - {data.STEP_COLUMN})
    if not columns:
        st.warning("The selected groups have no readable `progress.csv` yet.")
    else:
        metric = st.selectbox(
            "Metric", columns, index=columns.index(data.preferred_metric(columns) or columns[0])
        )
        curves, per_seed = {}, {}
        for group in chosen:
            seeds = cached.group_curves(group, metric)
            if not seeds:
                st.caption(f"`{group.name}`: no values for `{metric}` yet.")
                continue
            label = group.name if len({g.name for g in chosen}) == len(chosen) else group.label
            curves[label] = stats.mean_ci_curve(list(seeds.values()), window=window)
            per_seed[label] = seeds
        if curves:
            fig = plots.curves_figure(curves, metric)
            if show_seeds:
                import plotly.graph_objects as go

                for i, (label, seeds) in enumerate(per_seed.items()):
                    for seed, df in seeds.items():
                        fig.add_trace(
                            go.Scatter(
                                x=df["step"],
                                y=stats.smooth(df["value"], window),
                                mode="lines",
                                name=f"{label} seed {seed}",
                                legendgroup=label,
                                line={"color": plots.color(i), "width": 1, "dash": "dot"},
                                opacity=0.6,
                                showlegend=False,
                            )
                        )
            st.plotly_chart(fig, width="stretch")
            n = {label: int(df["n_seeds"].max()) for label, df in curves.items()}
            st.caption(
                "Line: mean across seeds. Band: 95% t-interval across seeds (needs 2+ seeds). "
                "Seeds per group: " + ", ".join(f"{k} = {v}" for k, v in n.items()) + ". "
                "Runs still training are drawn up to their latest logged step."
            )

with st.expander("Committed figures (results/)", expanded=False):
    committed_figures()
