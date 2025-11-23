import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from scipy.stats import t

from cartopy import crs as ccrs, feature as cfeature


def group_diff(group_1, group_2, data):
    g1_val = data.loc[group_1].median(axis=0)
    g2_val = data.loc[group_2].median(axis=0)
    diff = data.loc[group_2].median(axis=0) - data.loc[group_1].median(axis=0)
    return pd.Series(
        [
            (round(v1, 1), round(v2, 1), round(d, 1))
            for v1, v2, d in zip(g1_val.values, g2_val.values, diff.values)
        ],
        index=diff.index,
    )


def prob_group_diff_occur_chance(group_1, group_2, data, diff_ensemble):
    diff = data.loc[group_2].median(axis=0) - data.loc[group_1].median(axis=0)
    # Align the indices before comparison
    diff_ensemble, diff = diff_ensemble.align(diff, axis=1, copy=False)
    prob = (diff_ensemble > diff.abs()).sum(axis=0) / diff_ensemble.index.size
    return pd.Series(
        [(round(d, 1), round(p, 3)) for d, p in zip(diff.values, prob.values)],
        index=diff.index,
    )


def prob_median_occur_chance(group, data, median_ensemble):
    median = data.loc[group].median(axis=0)
    prob = (median_ensemble > median.abs()).sum(axis=0) / median_ensemble.index.size
    return pd.Series(
        [(round(d, 1), round(p, 3)) for d, p in zip(median.values, prob.values)],
        index=median.index,
    )


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
                fontsize=9,
            )

    grps = [table_groups[g] for g in table_dt.index]
    ax.set_yticks(range(len(grps)))
    ax.set_yticklabels(grps, fontsize=9)
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
        x=2.5,
        y=-3.0,
        s="Median transition durations (years)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=9,
    )
    ax.text(
        x=-0.75,
        y=-2,
        s="SLR Scenario",
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=9,
    )
    ax.text(
        x=-0.75,
        y=-1,
        s="Starting Year",
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=9,
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
            fontsize=8,
            fontweight="bold",
        )
        if c in [1, 3, 5]:
            rect = patches.Rectangle(
                xy=(-1.5 + c, -2.5),
                width=2,
                height=1,
                linewidth=0,
                facecolor=(
                    [0.8, 0.8, 0.8, 1.0] if c % 3 != 0 else [0.85, 0.85, 0.85, 1.0]
                ),
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(
                x=c - 0.5,
                y=-2,
                s=f"{table_scenarios[table_dt.columns[c][0]]}",
                ha="center",
                va="center",
                fontsize=9,
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
    ax, x, y, ghl, col, ms, title, splab, r_loc, xlim, xticks, xlab, ylim, yticks, ylab
):

    notna = np.isfinite(x) & np.isfinite(y)
    x, y = x.loc[notna], y.loc[notna]
    r = np.corrcoef(x, y)[0, 1]

    ghl = [g[notna] for g in ghl]

    ax.scatter(x, y, c=col[0], s=ms, edgecolor="k", alpha=1, linewidth=0.5, zorder=50)
    if ghl is not None:
        for n, g in enumerate(ghl):
            ax.scatter(
                x[g],
                y[g],
                c=col[n + 1],
                s=ms,
                edgecolor="k",
                alpha=1,
                linewidth=0.5,
                zorder=100,
            )
    # ax.plot(np.median(x), np.median(y), "k+", ms=10, zorder=100)
    ax.axhline(y=np.median(y), color="k", linestyle="-", linewidth=0.5, zorder=100)
    ax.axvline(x=np.median(x), color="k", linestyle="-", linewidth=0.5, zorder=100)

    ax.grid(color="gray", linestyle=":", linewidth=0.5, zorder=-10)
    ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlab is None:
        ax.set_xticklabels([])
    ax.set_xlabel(xlab)
    ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
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
        width=40,
        head_width=200,
        head_length=6,
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
        ycat = 2.4 * (thrsh - yo)
        ysubcat = -18
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
        y_scale = scale_bottom + 5
        x_scale = 1200  # xdh + 500  # 1500
        x0 = -1000
        sl_scale = 50
        t_scale = 5  # years
        t_scale_offset = 0
        ax.plot(
            [x0 + x_scale for _ in range(2)],
            [y_scale + y for y in [0, sl_scale]],
            lw=1.0,
            color="k",
        )
        ax.text(
            x=x0 + x_scale - 120,
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
            y=y_scale - (0.0 * sl_scale + 11),
            s=f"{t_scale} years",
            fontsize=9,
            # color="gray",
            verticalalignment="center",
            horizontalalignment="center",
        )


def corr_calc(q1, q2, criteria=None):
    z = np.isfinite(q1) & np.isfinite(q2)
    z = z if criteria is None else z & criteria
    return np.corrcoef(q1[z], q2[z])[0, 1]


def factor_correlation_matrix(analysis_df, scenario, central_est_type, chronic_freq):

    quantities_dict = {
        "∆h": analysis_df[f"dh_{central_est_type}_{chronic_freq}_days"],
        "∆t": analysis_df[f"dt_{central_est_type}_{chronic_freq}_days_{scenario}_2025"],
        "∆h 100": analysis_df[f"dh_{central_est_type}_100_days"],
        "∆t 100": analysis_df[f"dt_{central_est_type}_100_days_{scenario}_2025"],
        "Storm": analysis_df.res_hf_dymx_std,
        "MSL var": analysis_df.res_momn_q75_std,
        "HT mod": analysis_df.tide_dymx_std,
        "Td range": analysis_df.tide_range,
        "SLR": analysis_df[f"slr_total_{scenario}_2025_2055"],
        "Mass/Def": analysis_df[f"slr_massdef_{scenario}_2025_2055"],
        "Stero": analysis_df[f"slr_ocean_dyn_{scenario}_2025_2055"],
        "VLM": analysis_df[f"slr_vlm_{scenario}_2025_2055"],
        "Lat": np.abs(analysis_df.lat),
        "HDI": analysis_df.hdi,
    }

    quantities = list(quantities_dict.values())
    names = list(quantities_dict.keys())

    rmat = pd.DataFrame(index=names, columns=names).astype(float)
    for i, q1 in enumerate(quantities):
        for j, q2 in enumerate(quantities):
            rmat.iloc[i, j] = corr_calc(q1, q2)

    # determine significance of correlations based on t-test
    dof = 14  # degrees of freedom
    tmat = rmat / np.sqrt((1 - rmat**2) / (dof))  # t-statistic
    pvals = 2 * t.sf(np.abs(tmat), dof)
    pvalmat = np.zeros_like(pvals).astype(str)
    rmat_sig = np.zeros_like(rmat).astype(str)
    for m in range(len(names)):
        for n in range(len(names)):
            if m == n:
                pvalmat[m, n] = "-"
                rmat_sig[m, n] = "-"
            elif pvals[m, n] <= 0.01:  # highly significant
                pvalmat[m, n] = f"**{pvals[m, n]:.3f}"
                rmat_sig[m, n] = f"**{rmat.iloc[m, n]:.2f}"
            elif pvals[m, n] <= 0.05:  # significant
                pvalmat[m, n] = f"*{pvals[m, n]:.3f}"
                rmat_sig[m, n] = f"*{rmat.iloc[m, n]:.2f}"
            else:
                pvalmat[m, n] = f"{pvals[m, n]:.3f}"
                rmat_sig[m, n] = f"{rmat.iloc[m, n]:.2f}"
    pvalmat = pd.DataFrame(pvalmat, index=names, columns=names)
    rmat_sig = pd.DataFrame(rmat_sig, index=names, columns=names)

    return rmat_sig, pvalmat


def map_groups(groups_to_map, group_colors, groups, analysis):

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7.5, 4),
        subplot_kw=dict(projection=ccrs.Mollweide(central_longitude=210)),
    )
    ax.add_feature(cfeature.LAND.with_scale("110m"), color="lightgray")
    ax.gridlines(linewidth=0.5, color="k", linestyle=":")

    for n, g in enumerate(groups_to_map):
        co = ax.scatter(
            x=analysis.lon.loc[groups[groups_to_map[g]]],
            y=analysis.lat.loc[groups[groups_to_map[g]]],
            c=group_colors[n],
            edgecolors="k",
            linewidths=0.5,
            s=50,
            label=g,
            transform=ccrs.PlateCarree(),
        )

    ax.legend(
        ncol=(n + 2) // 2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False
    )
    ax.set_global()
    plt.tight_layout()

    return fig, ax
