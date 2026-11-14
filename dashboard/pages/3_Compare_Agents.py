"""Compare the RL agent with random, greedy and optimal baselines (IQM + bootstrap CIs)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import cached, plots, stats, tables, ui  # noqa: E402

ui.setup("Compare Agents")
st.write(
    "Efficiency is the optimal planner's step count divided by the agent's (0 for a failed "
    "episode), so 1.00 is optimal. All agents are scored on held-out layouts."
)

summary = cached.results_csv("phase5_summary.csv")
runs = cached.results_csv("phase5_runs.csv")
baselines = cached.results_csv("warehouse_baselines.csv")
have_summary = tables.valid_summary(summary)
have_baselines = tables.valid_baselines(baselines)

# -- 1. RL vs baselines -------------------------------------------------------------------------
st.subheader("RL agent vs baselines")
if not have_baselines:
    ui.missing_file("results/warehouse_baselines.csv", "baseline table")
else:
    settings = tables.baseline_settings(baselines)
    default = next((i for i, s in enumerate(settings) if s.startswith("medium")), 0)
    setting = st.selectbox("Setting", settings, index=default)
    frame = tables.rl_vs_baselines(summary if have_summary else None, baselines, setting)
    if frame.empty:
        st.info("No rows for this setting.")
    else:
        if not setting.startswith("medium"):
            st.caption(
                "The RL agent was trained and evaluated on the 12x12 / 3-item setting only, so it "
                "has no row here."
            )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                plots.grouped_bars(
                    frame,
                    "agent",
                    "success",
                    "success_lo",
                    "success_hi",
                    "Held-out success rate",
                    (0, 1.12),
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                plots.grouped_bars(
                    frame,
                    "agent",
                    "efficiency",
                    "efficiency_lo",
                    "efficiency_hi",
                    "Efficiency (optimal steps / agent steps)",
                    (0, 1.12),
                ),
                width="stretch",
            )
        show = frame.assign(
            Success=[
                tables.fmt_interval(r.success, r.success_lo, r.success_hi)
                for r in frame.itertuples()
            ],
            Efficiency=[
                tables.fmt_interval(r.efficiency, r.efficiency_lo, r.efficiency_hi)
                for r in frame.itertuples()
            ],
            Steps=frame["steps"].round(1),
        )[["agent", "Success", "Efficiency", "Steps"]].rename(columns={"agent": "Agent"})
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption(
            "Baselines: point estimates over 200 episodes (greedy and optimal are deterministic "
            "given a layout). RL agent: IQM over 5 training seeds with a 95% bootstrap interval. "
            "Steps are episode lengths (failed episodes count at the time limit)."
        )

# -- 2. any experiment groups, bootstrap from per-seed results ---------------------------------
st.subheader("Experiment groups: IQM with bootstrap intervals")
needed = {"group", "seed"}
if runs is None or not needed <= set(runs.columns):
    ui.missing_file("results/phase5_runs.csv", "per-seed results table")
    if have_summary:
        st.caption("Falling back to the pre-computed intervals in `phase5_summary.csv`.")
else:
    metric_cols = sorted(c for c in runs.columns if c not in needed)
    with st.sidebar:
        st.header("Experiment groups")
        all_groups = list(dict.fromkeys(runs["group"]))
        chosen = st.multiselect(
            "Groups",
            all_groups,
            default=[g for g in ("headline", "ref3m", "e1_sparse") if g in all_groups],
        )
        split = st.radio(
            "Evaluated on",
            ["test", "train"],
            format_func=lambda s: "Held-out layouts" if s == "test" else "Training layouts",
        )
        metrics_for_split = [
            c.removeprefix(f"{split}_") for c in metric_cols if c.startswith(f"{split}_")
        ]
        metric = (
            st.selectbox(
                "Metric",
                metrics_for_split,
                index=metrics_for_split.index("mean_efficiency")
                if "mean_efficiency" in metrics_for_split
                else 0,
                format_func=lambda m: tables.METRIC_LABELS.get(m, m),
            )
            if metrics_for_split
            else None
        )
        reps = st.select_slider("Bootstrap resamples", [500, 1000, 2000, 5000, 10000], 2000)
    if not chosen or metric is None:
        st.info("Pick at least one group in the sidebar.")
    else:
        column = f"{split}_{metric}"
        sub = runs[runs["group"].isin(chosen)]
        table = stats.iqm_table(sub, column, reps)
        if table.empty:
            st.info("No data for that combination.")
        else:
            points = sub.dropna(subset=[column])
            fig = plots.interval_figure(
                table,
                "group",
                "iqm",
                "ci_low",
                "ci_high",
                xlabel=tables.METRIC_LABELS.get(metric, metric),
                points=points,
                points_label_col="group",
                points_value_col=column,
            )
            if have_baselines and metric in ("success_rate", "mean_efficiency"):
                ref = baselines[baselines["setting"].str.startswith("medium")]
                col = "success_rate" if metric == "success_rate" else "mean_efficiency"
                for agent, dash in (("greedy", "dash"), ("optimal", "dot")):
                    row = ref[ref["agent"] == agent]
                    if not row.empty:
                        fig.add_vline(
                            x=float(row[col].iloc[0]),
                            line_dash=dash,
                            line_color="gray",
                            annotation_text=agent,
                            annotation_position="top",
                        )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                table.rename(
                    columns={
                        "group": "Group",
                        "n_seeds": "Seeds",
                        "mean": "Mean",
                        "iqm": "IQM",
                        "ci_low": "CI low",
                        "ci_high": "CI high",
                    }
                ).round(3),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "IQM = mean of the middle 50% of the per-seed scores (Agarwal et al., 2021). "
                "The interval resamples seeds with replacement. With 3 seeds it is close to the "
                "min-max range and understates the uncertainty, so overlapping or separated "
                "intervals are not a significance test. Dashed/dotted lines: greedy and optimal "
                "planners on the same 12x12 setting."
            )
            with st.expander("Per-seed numbers"):
                st.dataframe(
                    sub[["group", "seed", column]].sort_values(["group", "seed"]),
                    width="stretch",
                    hide_index=True,
                )
