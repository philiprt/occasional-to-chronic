"""
Violin histogram figure creation utilities.

This module contains functions for creating violin plot histograms and group comparison figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


def _extract_significance_value(diff_sig, sp_var, significance_level, sig_index=0):
    """
    Extract significance value from diff_sig, which can be either a DataFrame or a list of DataFrames.
    For lists, uses items sequentially starting from sig_index.

    Parameters:
    -----------
    diff_sig : pandas.DataFrame or list
        Either a single DataFrame or a list of DataFrames
    sp_var : str
        Variable name for extracting values from DataFrames
    significance_level : float
        Significance level for extracting values from DataFrames
    sig_index : int, optional
        Starting index for sequential extraction from lists (default: 0)

    Returns:
    --------
    tuple : (float or None, int)
        The extracted significance value (or None if extraction fails) and the next index to use
    """
    if isinstance(diff_sig, list):
        # Handle list of DataFrames - use items sequentially
        for i in range(len(diff_sig)):
            check_index = (sig_index + i) % len(diff_sig)
            item = diff_sig[check_index]

            if item is not None:
                if hasattr(item, "loc"):
                    # It's a DataFrame, extract the value
                    try:
                        value = item.loc[sp_var, significance_level]
                        if value is not None:
                            next_index = (check_index + 1) % len(diff_sig)
                            return value, next_index
                    except (KeyError, TypeError):
                        continue
                else:
                    # It's a scalar value
                    next_index = (check_index + 1) % len(diff_sig)
                    return item, next_index

        # If no value found, try to find the last non-None value
        for item in reversed(diff_sig):
            if item is not None:
                if hasattr(item, "loc"):
                    try:
                        value = item.loc[sp_var, significance_level]
                        if value is not None:
                            return value, sig_index
                    except (KeyError, TypeError):
                        continue
                else:
                    return item, sig_index

        return None, sig_index
    else:
        # Handle single DataFrame or scalar
        if hasattr(diff_sig, "loc"):
            # It's a DataFrame, extract the value
            try:
                value = diff_sig.loc[sp_var, significance_level]
                return value, sig_index
            except (KeyError, TypeError):
                return None, sig_index
        else:
            # It's a scalar value
            return diff_sig, sig_index


def hist_violin(
    ax,
    samples,
    group_latpred_medians,
    global_median,
    ylab,
    group_names,
    violin_locs,
    sp_letter,
    sp_letter_loc=(0.04, 0.88),
    percentiles=[17, 83],
    mnmx_pctls=[1, 99],
    n_bins=15,
    width_factor="norm",
    sample_names=None,
    diff_sig=None,
    leglab=None,
    colors=None,
    ylim=None,
    yticks=None,
    sp_var=None,
    significance_level=None,
):
    """
    Create a violin histogram plot.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    samples : dict
        Dictionary of sample data for each group
    group_latpred_medians : dict
        Dictionary of group median values of variable predicted by latitude regression to plot as horizontal lines
    global_median : float
        Global median value to plot as horizontal line
    ylab : str
        Y-axis label
    group_names : list
        Names of groups for x-axis labels
    sp_letter : str
        Subplot letter (a, b, c, d)
    sp_letter_loc : tuple, optional
        Location for subplot letter annotation
    percentiles : list, optional
        Percentiles to plot as vertical lines
    mnmx_pctls : list, optional
        Min/max percentiles for binning
    n_bins : int, optional
        Number of histogram bins
    width_factor : float, optional
        Factor to scale violin width
    sample_names : list, optional
        Names of samples (keys from samples dict)
    violin_locs : list, optional
        Horizontal positions for violins
    diff_sig : float or list, optional
        Significance level for difference testing. Can be a single value or a list of values.
        If a list is provided, the first non-None value will be used for plotting.
        If list elements are DataFrames, values will be extracted using sp_var and significance_level.
    leglab : bool, optional
        Whether to add legend labels
    colors : list, optional
        Colors for plotting
    ylim : list, optional
        Y-axis limits
    yticks : list, optional
        Y-axis tick positions
    sp_var : str, optional
        Variable name for extracting values from DataFrames in diff_sig list
    significance_level : float, optional
        Significance level for extracting values from DataFrames in diff_sig list
    """
    if sample_names is None:
        sample_names = [nm for nm in samples]
    samples = {g: [s for s in samples[g] if ~np.isnan(s)] for g in samples}
    svec = np.concatenate([samples[g] for g in samples])
    mn = np.percentile(svec, mnmx_pctls[0])
    mx = np.percentile(svec[np.isfinite(svec)], mnmx_pctls[1])
    rng = mx - mn
    buff = 0.05
    bins = np.linspace(mn - buff * rng, mx + buff * mx, n_bins + 1)

    vl = [x for x in range(6)] if violin_locs is None else violin_locs

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    s_medians = []
    s_medians_latpred = []
    # Track which significance value to use for each group pair
    sig_index = 0
    for n, g in enumerate(sample_names):
        s = samples[g]
        v, _, patches = ax.hist(
            s,
            bins=bins,
            bottom=violin_locs[n],
            density=True,
            orientation="horizontal",
            color=colors[0],
            alpha=0.8,
            zorder=1,
        )
        [p.remove() for p in patches]

        if width_factor == "norm":
            width_factor = 0.45 / max(v)

        w = np.diff(bins)[0]
        ax.barh(
            bins[:-1] + 0.5 * w,
            v * width_factor,
            w,
            left=vl[n],
            color=colors[0 if n % 2 == 0 else 3],
            lw=1.5,
            zorder=1,
            label=None,
        )
        ax.barh(
            bins[:-1] + 0.5 * w,
            -v * width_factor,
            w,
            left=vl[n],
            color=colors[0 if n % 2 == 0 else 3],
            lw=1.5,
            zorder=1,
            label=None,
        )
        # add a dummy element for legend representing violin histograms
        ax.fill(
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            lw=0,
            color=colors[0],
            label="Group distribution" if leglab and n == 1 else None,
        )
        ax.plot(
            [violin_locs[n], violin_locs[n]],
            np.percentile(s, percentiles),
            "k",
            zorder=2,
            label=None,
        )
        ax.plot(
            vl[n],
            np.median(s),
            "o",
            color=colors[1],
            markeredgecolor="k",
            markersize=7,
            markeredgewidth=1,
            zorder=2,
            label="Group median" if leglab and n == 1 else None,
        )
        s_medians.append(np.median(s))
        s_medians_latpred.append(
            group_latpred_medians[g] if group_latpred_medians is not None else 0
        )
        if n % 2 == 1 and diff_sig is not None:
            # Use helper function to extract significance value
            sig_value, sig_index = _extract_significance_value(
                diff_sig, sp_var, significance_level, sig_index
            )
            diff_latpred = (
                np.abs(s_medians_latpred[n] - s_medians_latpred[n - 1])
                if group_latpred_medians is not None
                else 0
            )

            if sig_value is not None:
                x1, x2 = vl[n - 1] - 0.5, vl[n] + 0.5
                ymid = s_medians[n] + (s_medians[n - 1] - s_medians[n]) / 2
                y1, y2 = (
                    ymid - (diff_latpred + sig_value) / 2,
                    ymid + (diff_latpred + sig_value) / 2,
                )
                ax.fill(
                    [x1, x2, x2, x1],
                    [y1, y1, y2, y2],
                    lw=0,
                    color=colors[2],
                    alpha=1,
                    zorder=-10,
                    label=(
                        "Median difference 95% significance"
                        if leglab and n == 1
                        else None
                    ),
                )

    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.axhline(
        global_median,
        c="k",
        linestyle="--",
        lw=1,
        label="Global median" if "leglab" else None,
    )
    ax.set_xlim([violin_locs[0] - 1, violin_locs[-1] + 1])
    ax.set_xticks(violin_locs)
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_ylabel(ylab)
    ax.annotate(
        text=sp_letter,
        xy=sp_letter_loc,
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(color="gray", linestyle=":", linewidth=0.5, zorder=-10)


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.colors) * i - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.colors),
                    height,
                    facecolor=c,
                    edgecolor="none",
                )
            )

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def create_violin_histogram_figure(
    violin_groups,
    subplots,
    analysis,
    groups,
    analysis_latpred=None,
    median_diff_percentiles=None,
    significance_level=95,
    colors=None,
    figsize=(8.5, 6),
    violin_locs=None,
    save_path=None,
    save_filename=None,
):
    """
    Create a group comparison figure with violin plots.

    Parameters:
    -----------
    violin_groups : dict
        Dictionary mapping display names to group keys (e.g., {"Low Latitude": "lowLat"})
    subplots : list of dict
        List of subplot configurations with keys like 'var', 'mnmx_pctls', 'nbins', etc.
    analysis : pandas.DataFrame
        Analysis data containing the variables to plot
    analysis_latpred : pandas.DataFrame
        Analysis data containing the variables predicted by latitude regression
    groups : dict
        Dictionary mapping group keys to boolean masks for data selection
    median_diff_percentiles : pandas.DataFrame, optional
        Median difference percentiles for significance testing
    significance_level : float, optional
        Significance level for difference testing (default: 0.05)
    colors : list, optional
        List of colors for the plots
    figsize : tuple, optional
        Figure size (default: (8.5, 6))
    violin_locs : list, optional
        Horizontal positions for violin plots (default: [0, 1, 3, 4, 6, 7])
    save_path : str, optional
        Path to the directory where the figure will be saved (default: None)
    save_filename : str, optional
        Base filename for the saved figure (default: None)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Set default violin locations if not provided
    if violin_locs is None:
        violin_locs = [0, 1, 3, 4, 6, 7]

    # Set default colors if not provided
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Extract group keys and names from the dictionary
    violin_group_keys = list(violin_groups.values())
    violin_group_names = list(violin_groups.keys())

    # Update subplots with quantities
    for sp in subplots:
        quantities = dict(
            samples={
                g: analysis[sp["var"]].loc[groups[g]].values for g in violin_group_keys
            },
            group_latpred_medians=(
                {
                    g: analysis_latpred[sp["var"]].loc[groups[g]].median()
                    for g in violin_group_keys
                }
                if analysis_latpred is not None
                else None
            ),
            global_median=analysis[sp["var"]].median(),
        )

        # Add significance data if available
        if median_diff_percentiles is not None:
            if isinstance(median_diff_percentiles, list):
                # Handle list of DataFrames
                diff_sig_list = []
                for k, df in enumerate(median_diff_percentiles):
                    diff_sig_list.append(df.loc[sp["var"], significance_level])
                # Use update method to avoid type annotation issues
                quantities.update({"diff_sig": diff_sig_list})
            else:
                # Handle single DataFrame
                quantities["diff_sig"] = median_diff_percentiles.loc[
                    sp["var"], significance_level
                ]

        sp.update(quantities)

    # Create figure and axes
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(
        int(np.ceil(len(subplots) / 2)), 2, figsize=figsize, sharex=True
    )
    axes = axes.flatten()

    # Assign axes to subplots and remove unused axes
    [sp.update(dict(axis=axes[n])) for n, sp in enumerate(subplots)]
    [ax.remove() for ax in axes[len(subplots) :]]

    # Create violin plots for each subplot
    for k, sp in enumerate(subplots):
        ax = sp["axis"]
        hist_violin(
            ax=ax,
            samples=sp["samples"],
            sample_names=violin_group_keys,
            global_median=sp["global_median"],
            group_latpred_medians=sp["group_latpred_medians"],
            ylab=sp["ylab"],
            group_names=violin_group_names,
            violin_locs=violin_locs,
            sp_letter=chr(97 + k),
            width_factor=sp["wfact"],
            mnmx_pctls=sp["mnmx_pctls"],
            n_bins=sp["nbins"],
            ylim=sp["ylim"] if "ylim" in sp else None,
            yticks=sp["yticks"] if "yticks" in sp else None,
            diff_sig=sp.get("diff_sig"),
            leglab=sp.get("leglab"),
            colors=colors,
            sp_var=sp["var"],
            significance_level=significance_level,
        )

    # Add spacing above top row
    axes[0].annotate(
        text=r"$~$",
        xy=(0.5, 1.25),
        xycoords="axes fraction",
    )

    # Create legend
    leg_init = fig.legend()
    ordered_leg_handles = [leg_init.legend_handles[k] for k in [3, 0, 1, 2]]
    ordered_leg_labels = [h.get_label() for h in ordered_leg_handles if h is not None]
    # Create a new list with the MulticolorPatch
    final_handles = [
        ordered_leg_handles[0],
        MulticolorPatch([colors[0], colors[3]]),
        ordered_leg_handles[2],
        ordered_leg_handles[3],
    ]
    leg_init.remove()
    leg = fig.legend(
        handles=final_handles,
        labels=ordered_leg_labels,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        frameon=False,
        handler_map={MulticolorPatch: MulticolorPatchHandler()},
    )

    # Apply tight layout
    fig.tight_layout()

    # Save figure if path and filename are provided
    if save_path and save_filename:
        fig.savefig(f"{save_path}/{save_filename}.png", dpi=300)
        fig.savefig(f"{save_path}/{save_filename}.pdf", dpi=300)

    return fig
