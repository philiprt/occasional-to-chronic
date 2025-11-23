import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from .aggregate_and_visualize import (
    group_diff,
    prob_group_diff_occur_chance,
    prob_median_occur_chance,
)


def draw_diamond_and_center(points, ax, color, alpha, show_latpred, label=None):

    shape_points = np.array(
        [
            points["right"],
            points["top"],
            points["left"],
            points["bottom"],
            points["right"],  # Close the polygon by returning to start
        ]
    )

    ax.fill(shape_points[:, 0], shape_points[:, 1], alpha=alpha, color=color, lw=0)

    # ax.plot(
    #     points["center"][0],
    #     points["center"][1],
    #     "o",
    #     color="k",
    #     markersize=17,
    #     label="_",
    #     zorder=10,
    # )
    ax.plot(
        points["center"][0],
        points["center"][1],
        "o",
        color=color,
        markersize=13,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=label,
        zorder=10,
    )
    if show_latpred:
        ax.plot(
            points["center_latpred"][0],
            points["center_latpred"][1],
            "s",
            color=color,
            markersize=4,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="_",
            zorder=15,
        )


def make_diamond_plot(ax, n, sp, groups, quantiles, colors, alpha, show_latpred):

    x, y = sp["x"]["values"], sp["y"]["values"]

    for m, g in enumerate(groups):

        xg = x.loc[groups[g]]
        yg = y.loc[groups[g]]
        xglp = sp["x"]["latpred"].loc[groups[g]]
        yglp = sp["y"]["latpred"].loc[groups[g]]

        xgm = xg.median()
        ygm = yg.median()
        xglpm = xglp.median()
        yglpm = yglp.median()

        xgq = [xg.quantile(q / 100) for q in quantiles]
        ygq = [yg.quantile(q / 100) for q in quantiles]

        points = dict(
            center=(xgm, ygm),
            left=(xgq[0], ygm),
            right=(xgq[1], ygm),
            top=(xgm, ygq[1]),
            bottom=(xgm, ygq[0]),
            center_latpred=(xglpm, yglpm),
        )
        draw_diamond_and_center(
            points, ax, colors[m], alpha=alpha, label=g, show_latpred=show_latpred
        )

    ax.annotate(
        f"{chr(97 + n)}",
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlabel(sp["x"]["name"])
    ax.set_ylabel(sp["y"]["name"])
    if "ticks" in sp["x"]:
        ax.set_xticks(sp["x"]["ticks"])
    if "ticks" in sp["y"]:
        ax.set_yticks(sp["y"]["ticks"])


def get_group_statistics(
    subplots,
    groups,
    analysis,
    analysis_latpred,
    binary_group_names,
    binary_diff_ensemble,
    exclusion_diff_ensemble,
    binary_median_ensemble,
    exclusion_median_ensemble,
):

    # get unique variables and the residuals from latitude predictions
    uvar_names = {sp["x"]["var"]: sp["x"]["name"] for sp in subplots} | {
        sp["y"]["var"]: sp["y"]["name"] for sp in subplots
    }
    uvar_names = {k: v for k, v in uvar_names.items() if "latitude" not in k.lower()}
    uvars = list(uvar_names.keys())

    # get all permutations of pairs of keys in groups
    group_pairs = list(itertools.combinations(groups, 2))

    group_vals_and_diff = pd.DataFrame(
        [
            group_diff(
                groups[gp[0]],
                groups[gp[1]],
                analysis.loc[:, uvars],
            )
            for gp in group_pairs
        ],
        index=[f"{gp[0]} vs. {gp[1]}" for gp in group_pairs],
    )
    group_vals_and_diff.columns = [
        uvar_names[v].split("(")[0] for v in group_vals_and_diff.columns
    ]

    group_diff_with_prob_chance = pd.DataFrame(
        [
            prob_group_diff_occur_chance(
                groups[gp[0]],
                groups[gp[1]],
                analysis.loc[:, uvars] - analysis_latpred.loc[:, uvars],
                (
                    binary_diff_ensemble.loc[:, uvars]
                    if gp[0] in binary_group_names and gp[1] in binary_group_names
                    else exclusion_diff_ensemble.loc[:, uvars]
                ),
            )
            for gp in group_pairs
        ],
        index=[f"{gp[0]} vs. {gp[1]}" for gp in group_pairs],
    )
    group_diff_with_prob_chance.columns = [
        uvar_names[v].split("(")[0] for v in group_diff_with_prob_chance.columns
    ]

    group_median_with_prob_chance = pd.DataFrame(
        [
            prob_median_occur_chance(
                groups[g],
                analysis.loc[:, uvars] - analysis_latpred.loc[:, uvars],
                (
                    binary_median_ensemble.loc[:, uvars]
                    if g in binary_group_names
                    else exclusion_median_ensemble.loc[:, uvars]
                ),
            )
            for g in groups
        ],
        index=[g for g in groups],
    )
    group_median_with_prob_chance.columns = [
        uvar_names[v].split("(")[0] for v in group_median_with_prob_chance.columns
    ]

    group_stats = dict(
        group_vals_and_diff=group_vals_and_diff,
        group_diff_with_prob_chance=group_diff_with_prob_chance,
        group_median_with_prob_chance=group_median_with_prob_chance,
    )

    return group_stats


def make_diamond_figure(
    subplots,
    diamond_groups,
    quantiles,
    colors,
    analysis,
    analysis_latpred,
    groups,
    binary_group_names,
    binary_diff_ensemble,
    exclusion_diff_ensemble,
    binary_median_ensemble,
    exclusion_median_ensemble,
    alpha=0.2,
    fig_size=(9, 8),
    legend_ncol=None,
    subplot_spacing=None,
    show_latpred=True,
):

    diamond_groups = {g: groups[diamond_groups[g]] for g in diamond_groups}

    for sp in subplots:
        sp["x"]["values"] = analysis[sp["x"]["var"]]
        sp["x"]["latpred"] = analysis_latpred[sp["x"]["var"]]
        sp["y"]["values"] = analysis[sp["y"]["var"]]
        sp["y"]["latpred"] = analysis_latpred[sp["y"]["var"]]

    fig, ax = plt.subplots(len(subplots) // 2, 2, figsize=fig_size)
    ax = ax.flatten()

    for n, sp in enumerate(subplots):
        make_diamond_plot(
            ax[n], n, sp, diamond_groups, quantiles, colors, alpha, show_latpred
        )

    # Create single legend above all subplots
    handles, labels = ax[0].get_legend_handles_labels()  # Get from first subplot
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=legend_ncol if legend_ncol is not None else (len(diamond_groups) + 1) // 2,
        frameon=False,
        markerscale=0.7,
    )

    plt.tight_layout()

    if subplot_spacing is None:
        subplot_spacing = dict(top=0.85, wspace=0.2, hspace=0.25)
    plt.subplots_adjust(**subplot_spacing)

    group_stats = get_group_statistics(
        subplots,
        diamond_groups,
        analysis,
        analysis_latpred,
        binary_group_names,
        binary_diff_ensemble,
        exclusion_diff_ensemble,
        binary_median_ensemble,
        exclusion_median_ensemble,
    )

    return fig, ax, group_stats
