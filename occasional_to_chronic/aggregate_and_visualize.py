import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

from cartopy import crs as ccrs, feature as cfeature


def dt_table(table_dt, table_groups, table_scenarios):

    fig, ax = plt.subplots(figsize=(6, 4.25))
    vmax = 50
    m = ax.imshow(table_dt, cmap="Blues", vmin=0, vmax=vmax)
    for x in range(len(table_dt.index)):
        for y in range(len(table_dt.columns)):
            ax.text(
                x=y,
                y=x,
                s=f"{table_dt.iloc[x, y]}",
                ha="center",
                va="center",
                color="k" if table_dt.iloc[x, y] < vmax / 2 else "w",
                fontsize=8,
            )

    grps = [table_groups[g] for g in table_dt.index]
    ax.set_yticks(range(len(grps)))
    ax.set_yticklabels(grps)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(range(len(table_dt.columns)))
    ax.set_xticklabels([])

    # Remove the ticks but keep the labels; add padding between labels and axis
    ax.tick_params(axis="both", which="both", length=0, pad=8)

    # Remove the axes border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.text(
        x=4,
        y=-3.0,
        s="Median transition durations (years)",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.text(x=-0.75, y=-2, s="SLR Scenario", ha="right", va="center", fontweight="bold")
    ax.text(
        x=-0.75, y=-1, s="Starting Year", ha="right", va="center", fontweight="bold"
    )
    for c in range(len(table_dt.columns)):
        rect = patches.Rectangle(
            xy=(-0.5 + c, -1.5),
            width=1,
            height=1,
            linewidth=0,
            facecolor=[0.9, 0.9, 0.9, 1.0] if c % 2 == 0 else [0.95, 0.95, 0.95, 1.0],
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            x=c,
            y=-1,
            s=f"{table_dt.columns[c][1]}",
            ha="center",
            va="center",
            rotation=45,
            fontsize=9,
            fontweight="bold",
        )
        if c in [1, 4, 7]:
            rect = patches.Rectangle(
                xy=(-1.5 + c, -2.5),
                width=3,
                height=1,
                linewidth=0,
                facecolor=(
                    [0.8, 0.8, 0.8, 1.0] if c % 2 != 0 else [0.85, 0.85, 0.85, 1.0]
                ),
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(
                x=c,
                y=-2,
                s=f"{table_scenarios[table_dt.columns[c][0]]}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()

    return fig


def o2c_scatter_map(
    fig, ax, x, y, c, squares, vmin, vmax, alpha, splab, title, cbar_label
):
    ax.add_feature(cfeature.LAND.with_scale("110m"), color="gray")
    ax.gridlines(linewidth=0.5, color="k", linestyle=":")
    co = ax.scatter(
        x=x[~squares],
        y=y[~squares],
        c=c[~squares],
        edgecolors="k",
        linewidths=0.5,
        s=70,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        cmap="plasma_r",
        label=None,
    )
    co = ax.scatter(
        x=x[squares],
        y=y[squares],
        c=c[squares],
        marker="s",
        edgecolors="k",
        linewidths=0.5,
        s=60,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        cmap="plasma_r",
        label=None,
    )
    ax.set_global()
    title_y = 1.03
    ax.annotate(
        text=splab,
        xy=(0.0, title_y),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
    )
    ax.annotate(
        text=title,
        xy=(0.05, title_y),
        xycoords="axes fraction",
        fontsize=13,
    )
    fig.colorbar(co, ax=ax, label=cbar_label, shrink=0.8, pad=0.02, extend="both")


def o2c_scatter_plot(
    ax, x, y, ghl, col, ms, title, splab, r_loc, xlim, xlab, ylim, ylab
):

    notna = np.isfinite(x) & np.isfinite(y)
    x, y = x.loc[notna], y.loc[notna]
    r = np.corrcoef(x, y)[0, 1]

    ghl = ghl[notna]

    ax.scatter(
        x[ghl],
        y[ghl],
        c=col[1],
        s=ms,
        edgecolor="k",
        alpha=1,
        linewidth=0.5,
        zorder=100,
    )
    ax.scatter(x, y, c=col[0], s=ms, edgecolor="k", alpha=1, linewidth=0.5, zorder=50)

    ax.grid(color="gray", linestyle=":", linewidth=0.5, zorder=-10)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlab)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylab)
    if ylab is None:
        ax.set_yticklabels([])

    ax.annotate(
        text=title,
        xy=(0.5, -0.33),
        xycoords="axes fraction",
        fontsize=11,
        fontweight="normal",
        horizontalalignment="center",
    )
    ax.annotate(
        text=splab,
        xy=(0.04, 0.88),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
    )
    ax.annotate(
        text=f"r = {r:0.2f}",
        xy=(0.96, 0.04) if r_loc is None else r_loc,
        xycoords="axes fraction",
        fontsize=11,
        fontweight="normal",
        horizontalalignment="right",
    )


def hist_violin(
    ax,
    samples,
    median,
    ylab,
    group_names,
    sp_letter,
    sp_letter_loc=(0.04, 0.88),
    percentiles=[17, 83],
    mnmx_pctls=[1, 99],
    n_bins=15,
    width_factor=5,
    sample_names=None,
    violin_locs=None,
    diff_sig=None,
    leglab=None,
    colors=None,
    ylim=None,
    yticks=None,
):
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
        if n % 2 == 1 and diff_sig is not None:
            x1, x2 = vl[n - 1] - 0.5, vl[n] + 0.5
            ymid = s_medians[n] + (s_medians[n - 1] - s_medians[n]) / 2
            y1, y2 = ymid - diff_sig / 2, ymid + diff_sig / 2
            ax.fill(
                [x1, x2, x2, x1],
                [y1, y1, y2, y2],
                lw=0,
                color=colors[2],
                alpha=1,
                zorder=-10,
                label=(
                    "Median difference 95% significance" if leglab and n == 1 else None
                ),
            )

    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.axhline(
        median,
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


def plot_dymx_and_xdays(ax, dymx, thrsh, slr, offsets, colors, labels=False):
    xo, yo = offsets[0], offsets[1]
    # daily max sea levels
    ax.plot(
        dymx.index[[0, -1]] + xo,
        [slr + yo for n in range(2)],
        linewidth=1,
        linestyle="--",
        color="k",
        zorder=10,
        label="Mean Higher High Water (MHHW)" if labels else None,
    )
    ax.plot(
        dymx.index + xo,
        dymx.values + slr,
        linewidth=1,
        color=colors[0],
        zorder=0,
        label="Daily maximum sea level" if labels else None,
        rasterized=True,
    )

    # daily max above threshold
    abv = dymx.loc[dymx + slr > thrsh].index
    ax.scatter(
        dymx.loc[abv].index + xo,
        dymx.loc[abv].values + slr,
        zorder=20,
        color=colors[1],
        s=12,
        label="Threshold exceedance" if labels else None,
        rasterized=True,
        edgecolors="k",
        linewidths=0.5,
    )


def plot_change_with_slr(
    ax,
    name,
    dymx,
    dst,
    dst_fact,
    colors,
    offsets=[0, 0],
    thrsh=0,
    slr=0,
    name_offset=0,
    scale_bottom=0,
    labels=False,
):
    xo, yo = offsets[0], offsets[1]
    dymx += yo
    thrsh += yo

    # flodding threshold
    ax.plot(
        [0, dymx.index[-1] + xo],
        [thrsh, thrsh],
        label="Arbitrary threshold" if labels else None,
        linewidth=1.5,
        linestyle="-",
        color="k",
        zorder=10,
    )

    plot_dymx_and_xdays(
        ax=ax,
        dymx=dymx,
        thrsh=thrsh,
        slr=0,
        offsets=[0, yo],
        colors=colors,
        labels=True if labels else False,
    )
    plot_dymx_and_xdays(
        ax=ax,
        dymx=dymx,
        thrsh=thrsh,
        slr=slr,
        offsets=[xo, yo],
        colors=colors,
    )

    do = 750
    ax.fill_betweenx(
        dst.index + yo,
        -dst.values * dst_fact - do,
        -do,
        lw=0,
        color=colors[2],
        alpha=1,
        zorder=5,
        label="Distribution of daily max sea levels" if labels else None,
    )
    tail = (dst.index + yo) >= thrsh
    ax.fill_betweenx(
        dst.index[tail] + yo,
        -dst.values[tail] * dst_fact - do,
        -do,
        lw=0.5,
        color=colors[1],
        edgecolor="k",
        alpha=1,
        zorder=5,
        label="Tail of distribution above threshold" if labels else None,
    )
    ax.fill_betweenx(
        slr + dst.index + yo,
        dymx.index[-1] + do + xo,
        dymx.index[-1] + dst.values * dst_fact + do + xo,
        lw=0,
        color=colors[2],
        alpha=1,
        zorder=5,
    )
    tail = (slr + dst.index + yo) >= thrsh
    ax.fill_betweenx(
        slr + dst.index[tail] + yo,
        dymx.index[-1] + do + xo,
        dymx.index[-1] + dst.values[tail] * dst_fact + do + xo,
        lw=0.5,
        color=colors[1],
        edgecolor="k",
        alpha=1,
        zorder=5,
    )

    xdh = dymx.index[-1] + (xo - dymx.index[-1]) / 2
    xdh_offset = 0.25 * (xo - dymx.index[-1]) / 2
    ax.arrow(
        x=xdh - xdh_offset,
        y=yo,
        dx=0,
        dy=slr,
        width=50,
        head_width=250,
        head_length=3.5,
        length_includes_head=True,
        color="black",
    )
    ax.text(
        x=xdh + xdh_offset,
        y=0.8 * slr / 2 + yo,
        s=r"$\Delta h$",
        fontsize=13,
        va="center",
        ha="center",
    )
    ax.text(
        x=0,
        y=yo + name_offset,
        s=name,
        fontsize=11,
        va="center",
        ha="left",
        bbox=dict(
            boxstyle="square",
            pad=0.5,
            facecolor="w",
            edgecolor="k",
            linewidth=0.5,
        ),
        zorder=100,
    )
    if labels:
        xmn = (dymx.index[-1] - dymx.index[0]) / 2
        ycat = 3 * (thrsh - yo)
        ysubcat = -9
        ax.text(
            x=xmn,
            y=ycat,
            s="Occasional Exceedance",
            fontsize=13,
            fontweight="bold",
            verticalalignment="bottom",
            horizontalalignment="center",
        )
        ax.text(
            x=xmn,
            y=ycat + ysubcat,
            s="1 day per year (median)",
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
        )
        ax.text(
            x=xmn + xo,
            y=ycat,
            s="Chronic Exceedance",
            fontsize=13,
            fontweight="bold",
            verticalalignment="bottom",
            horizontalalignment="center",
        )
        ax.text(
            x=xmn + xo,
            y=ycat + ysubcat,
            s="26 days per year (median)",
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
        )
        y_scale = scale_bottom
        x_scale = 1200  # xdh + 500  # 1500
        x0 = -1000
        sl_scale = 20
        t_scale = 5  # years
        t_scale_offset = 0
        ax.plot(
            [x0 + x_scale for _ in range(2)],
            [y_scale + y for y in [0, sl_scale]],
            lw=1.0,
            color="k",
        )
        ax.text(
            x=x0 + x_scale - 200,
            y=y_scale + 0.5 * sl_scale,
            s=f"{sl_scale} cm",
            fontsize=9,
            # color="gray",
            verticalalignment="center",
            horizontalalignment="right",
        )
        ax.plot(
            [x0 + x_scale + x + t_scale_offset for x in [0, t_scale * 365.25]],
            [y_scale + 0.0 * sl_scale for _ in range(2)],
            lw=1.0,
            color="k",
        )
        ax.text(
            x=x0 + x_scale + 0.5 * t_scale * 365.25 + t_scale_offset,
            y=y_scale - (0.0 * sl_scale + 7),
            s=f"{t_scale} years",
            fontsize=9,
            # color="gray",
            verticalalignment="center",
            horizontalalignment="center",
        )


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
