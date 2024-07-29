import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import utide

from quality_control import quality_control


def load_and_process_station_data(
    tg, min_hpd, min_dpy, min_yfi, tide_prd_dir, qc_fig_dir=None, save_fig=False
):

    # --------------------------------------------------------------------

    uhid = tg.name
    lat = tg["latitude"]
    hrs_in_epoch = int(24 * 365.25 * 19)  # 19 years

    # --------------------------------------------------------------------
    # LOAD DATA FOR THIS STATION

    hsl = xr.load_dataset(f"./data/tide_gauge_data/h{uhid:03d}.nc")
    hsl = hsl.isel(record_id=0).sea_level.to_pandas()
    hsl.index = hsl.index.round("h")
    hsl = hsl.loc[~hsl.index.duplicated(keep="first")]
    hsl = hsl / 10  # convert to cm

    hsl_orig = hsl.copy()
    hsl, qc = quality_control(uhid, hsl)

    hsl -= hsl.iloc[-hrs_in_epoch:].mean()

    # --------------------------------------------------------------------
    # PREP

    # trim missing values at the beginning and end of the time series
    first_last_val = hsl.dropna().index[[0, -1]]
    hsl = hsl.loc[first_last_val[0] : first_last_val[1]]

    # get years of first and last data
    first_year = first_last_val[0].year
    last_year = first_last_val[1].year

    # calculate daily averages for days with sufficient hours
    dsl = (
        hsl.groupby(pd.Grouper(freq="D"))
        .apply(lambda x: x.mean() if (x.dropna().count() > min_hpd) else None)
        .dropna()
    )
    dsl.index += pd.Timedelta("12 hours")

    # check to see if this record has sufficient quality years and eject if not
    day_count = dsl.groupby(dsl.index.year).apply(lambda x: x.dropna().count())
    quality_years = day_count.loc[day_count > min_dpy].index.values
    if quality_years.size < min_yfi:
        return None

    # --------------------------------------------------------------------
    # TIDE PREDICTION

    tide_prd_file = f"{tide_prd_dir}t{uhid:03d}.csv"
    if os.path.exists(tide_prd_file):
        tide = pd.read_csv(tide_prd_file, index_col="time", parse_dates=True)[
            "tide_prediction"
        ]
    else:
        hsl_epoch = hsl.iloc[-hrs_in_epoch:].dropna()

        # fit function and reconstruct; remove met year shift for solving/reconstructing
        coef = utide.solve(
            hsl_epoch.index.values,
            hsl_epoch.values,
            lat=lat,
            verbose=False,
        )
        coef["slope"] = 0
        utb = utide.reconstruct(
            hsl_orig.index.values,
            coef,
            verbose=False,
        )
        tide = pd.Series(utb.h, index=hsl_orig.index, name="tide_prediction")
        tide.to_csv(tide_prd_file, index=True)

    tide_orig = tide.copy()
    tide = tide.loc[hsl.index]
    tide -= tide.loc[hsl.iloc[-hrs_in_epoch:].dropna().index].mean()

    # --------------------------------------------------------------------
    # CALCULATE AND REMOVE TREND

    # calculate trend in valid daily averages
    dsl_t_index = np.array([k for k, t in enumerate(hsl.index) if t in dsl.index])
    m = np.polyfit(dsl_t_index, dsl.values, 1)

    # reconstruct trend for hourly values
    hsl_trnd = pd.Series(index=hsl.index)
    hsl_trnd = pd.Series(np.polyval(m, range(hsl.index.size)), index=hsl.index)

    # --------------------------------------------------------------------
    # MAKE AND SAVE BASIC PLOT TO CHECK FOR DATA IRREGULARITIES

    if save_fig:

        fig = plt.figure(dpi=500, figsize=[8, 8])

        ax1 = plt.subplot(211)
        plt.plot(hsl_orig, label="ORIGINAL hourly sea level")
        # plt.plot(tide, label="predicted tide")
        plt.plot(hsl_orig - tide_orig, label="nontidal residuals")
        plt.legend()

        ax2 = plt.subplot(212, sharex=ax1)
        if qc:
            plt.plot(hsl, label="QC'd hourly sea level")
            # plt.plot(tide, label="predicted tide")
            plt.plot(hsl - tide, label="nontidal residuals")
            plt.legend()
        else:
            plt.text(
                hsl_orig.index[0] + (hsl_orig.index[-1] - hsl_orig.index[0]) / 2,
                0.5,
                "No additional QC applied",
                ha="center",
            )

        fig.suptitle(f"{uhid:03d}: {tg.station_name}")

        plt.tight_layout()

        station_name = "".join(
            ["_" if c in [" "] else c for c in tg.station_name.lower()]
        )
        station_name = "".join([c for c in station_name if c not in [",", "'", "."]])
        fig_file = f"{qc_fig_dir}{uhid:03d}_{station_name}.png"
        plt.savefig(fig_file)
        plt.close()

    # --------------------------------------------------------------------

    return hsl, hsl_trnd, tide, quality_years, first_year, last_year


def station_analysis(tg, min_hpd, min_dpy, min_yfi, tide_prd_dir, qc_fig_dir):

    # --------------------------------------------------------------------
    # LOAD AND PROCESS DATA FOR THIS STATION

    uhid = tg.name

    result = load_and_process_station_data(
        tg,
        min_hpd,
        min_dpy,
        min_yfi,
        tide_prd_dir,
        qc_fig_dir,
        save_fig=True,
    )

    if result is None:
        return None
    else:
        hsl, hsl_trnd, tide, quality_years = result[0], result[1], result[2], result[3]
        first_year, last_year = result[4], result[5]

    # detrended hourly values
    hsldt = hsl - hsl_trnd

    # calculate tidal residuals
    res = hsldt - tide

    # --------------------------------------------------------------------
    # USE MET YEARS

    # we want to work with meteorological years (May–April) so shift all times back
    # 120 days (i.e., total days in January-April) to artificially make May 1 serve as
    # January 1
    hsldt.index -= pd.Timedelta("120 days")
    tide.index -= pd.Timedelta("120 days")

    # --------------------------------------------------------------------
    # CALCULATE DELTA H

    # get daily max of detrended hourly values
    dmxdt = hsldt.groupby(pd.Grouper(freq="D")).apply(
        lambda x: x.mean() if (x.dropna().count() > min_hpd) else None
    )

    # calculate high and low
    high = np.floor(np.max(dmxdt))
    low = np.floor(0.5 * np.std(dmxdt))  # halfstd

    # define range of thresholds
    thresholds = np.arange(low, high + 0.1, 0.1)  # 1 mm increments

    # get number of daily max exceedances in each met year over each threshold
    dmxdt_gy = dmxdt.groupby(dmxdt.index.year)
    Nxpy = pd.concat(
        [dmxdt_gy.apply(lambda x: (x > thrsh).sum()) for thrsh in thresholds], axis=1
    ).loc[quality_years, :]
    Nxpy.columns = thresholds

    # calculate the median number of daily max exceedances per met year for each threshold
    Nxpy_median = Nxpy.median(axis=0)
    Nxpy_mean = Nxpy.mean(axis=0)

    def sl_trans(Nx, xhi):
        min_1 = Nx.loc[Nx >= 1].idxmin()
        min_xhi = Nx.loc[Nx >= xhi].idxmin()
        sl_dif = np.round(min_1 - min_xhi, 1)
        return sl_dif

    del_days = [20, 26, 50]
    dh_median = {f"dh_median_{dd}_days": sl_trans(Nxpy_median, dd) for dd in del_days}
    dh_mean = {f"dh_mean_{dd}_days": sl_trans(Nxpy_mean, dd) for dd in del_days}

    # --------------------------------------------------------------------
    # TIDAL AND RESIDUAL STATISTICS

    # get daily max residuals
    res_dymx = (
        res.groupby(pd.Grouper(freq="D"))
        .apply(lambda x: x.mean() if (x.dropna().count() > min_hpd) else None)
        .dropna()
    )

    # get monthly mean statistics
    res_mo_grps = res.groupby(pd.Grouper(freq="MS"))
    res_momn = res_mo_grps.mean()
    res_momn_q75_std = res_momn.loc[res_momn >= res_momn.quantile(0.75)].std()
    res_momn_amx_std = res_momn.groupby(pd.Grouper(freq="A")).max().std()

    # get high frequency residuals and statistics
    res_hf = res_mo_grps.apply(lambda x: x - x.mean()).droplevel(0)
    res_hf_dymx = (
        res_hf.groupby(pd.Grouper(freq="D"))
        .apply(lambda x: x.mean() if (x.dropna().count() > min_hpd) else None)
        .dropna()
    )

    # get tidal daily max
    tide_dymx = tide.groupby(pd.Grouper(freq="D")).apply(
        lambda x: x.mean() if (x.dropna().count() > min_hpd) else None
    )

    # calculate final statistics and aggregate
    stats = dict(
        res_dymx_std=res_dymx.std(),
        res_dymx_top50_std=res_hf.nlargest(int(50 * res_dymx.size / 365.25)).std(),
        res_momn_std=res_mo_grps.mean().std(),
        res_momn_q75_std=res_momn_q75_std,
        res_momn_amx_std=res_momn_amx_std,
        res_hf_dymx_std=res_hf_dymx.std(),
        res_hf_dymx_top50_std=res_hf.nlargest(
            int(50 * res_hf_dymx.size / 365.25)
        ).std(),
        tide_dymx_std=tide_dymx.std(),
        tide_dymx_top50_std=tide.nlargest(int(50 * tide_dymx.size / 365.25)).std(),
    )

    # --------------------------------------------------------------------

    analysis = pd.Series(
        {
            **dict(
                name=tg.station_name,
                country=tg.station_country,
                lat=tg.latitude,
                lon=tg.longitude,
                hdi=tg.hdi,
                first_year=first_year,
                last_year=last_year,
                n_good_years=quality_years.size,
            ),
            **dh_mean,
            **dh_median,
            **stats,
        }
    )
    analysis.name = uhid

    # --------------------------------------------------------------------

    return analysis
