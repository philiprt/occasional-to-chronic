import numpy as np
import pandas as pd


def estimate_step(hsl, t_step):
    Nt = hsl.size
    if type(t_step) is str:
        t_step = [t_step]
    step = pd.concat([pd.Series(0, index=hsl.index) for _ in t_step], axis=1)
    for k, t in enumerate(t_step):
        step.loc[t:, k] = 1
    A = np.vstack([np.ones(Nt), np.arange(Nt), step.values.T]).T
    x = hsl.values
    z = ~np.isnan(x)
    c = np.linalg.lstsq(A[z, :], x[z], rcond=None)[0]
    # y = pd.Series(A @ c, index=hsl.index)
    return A[:, 2:] @ c[2:]


def quality_control(uhid, hsl):

    qc = True

    if uhid == 3:  # Baltra
        # remove some bad data (positive outliers) around 2020–2021
        hsl.loc[hsl > 350] = None

    elif uhid == 8:  # Yap
        # remove one bad negative outlier in late 2019
        hsl.loc[hsl < -50] = None

    elif uhid == 15:  # Papeete, Tahiti
        # remove some bad data (positive outliers) in late 2021
        hsl0 = hsl.loc["2021"]
        hsl0.loc[hsl0 > 160] = None

    elif uhid == 17:  # Hiva Oa
        # remove two stretches of sketchy data inconsistent with rest of the record
        hsl.loc["2021-08":"2021-10"] = None
        hsl.loc["2023-05":] = None

    elif uhid == 30:  # Santa Cruz, Ecuador
        # remove spikes
        hsl.loc["2021-03-05 10":"2021-03-05 16"] = None
        drop = [
            "2019-08-08 05:00:00",
            "2019-08-08 06:00:00",
            "2019-11-08 01:00:00",
            "2019-11-11 05:00:00",
            "2019-11-11 06:00:00",
            "2022-01-15 22:00:00",
            "2022-01-16 16:00:00",
            "2022-01-17 11:00:00",
            "2022-01-18 01:00:00",
            "2020-04-17 09:00:00",
            "2020-06-17 03:00:00",
            "2021-01-13 00:00:00",
            "2021-02-08 13:00:00",
            "2021-03-05 10:00:00",
            "2021-03-05 11:00:00",
            "2021-04-27 02:00:00",
            "2021-04-27 03:00:00",
            "2021-04-27 16:00:00",
            "2021-04-28 00:00:00",
            "2021-04-28 07:00:00",
            "2021-04-28 09:00:00",
            "2021-05-09 12:00:00",
            "2021-10-14 04:00:00",
            "2021-11-05 04:00:00",
            "2021-11-10 02:00:00",
            "2021-11-14 06:00:00",
            "2022-01-31 02:00:00",
            "2022-01-31 16:00:00",
            "2022-02-21 03:00:00",
            "2022-03-04 04:00:00",
            "2022-03-10 03:00:00",
            "2022-03-24 03:00:00",
            "2022-03-25 03:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 31:  # Nuku Hiva
        # remove three hours of bad data in 2010
        hsl.loc["2010-02-27 18":"2010-02-27 20"] = None
        # remove end of record due to sketchy data inconsistent with rest of the record
        hsl.loc["2019-10":] = None

    elif uhid == 33:  # Bitung, Indonesia
        # remove negative spike
        hsl.loc["2016-01-01 00"] = None

    elif uhid == 49:  # Minamitorishima
        # remove end of record due to sketchy data inconsistent with rest of the record
        hsl.loc["2021":] = None

    elif uhid == 56:  # Pago Pago
        # remove period following 2009 earthquake
        hsl.loc["2009-09":] = None

    elif uhid == 80:  # Antofagasta, Chile
        # remove some bad positive outliers
        hsl0 = hsl.loc["2022-07":"2022-12"]
        hsl0.loc[hsl0 > 170] = None
        # remove this stretch of data with standard deviation larger than rest of record
        hsl.loc["2010-11":"2011-11"] = None
        # remove one clear neagtive outlier
        hsl.loc["2019-06-05 14"] = None

    elif uhid == 81:  # Valpariso, Chile
        # remove this stretch of data that is offset low compared to the rest of record
        hsl.loc["1971-06":"1978-12"] = None

    elif uhid == 87:  # Quepos, Costa Rica
        # timing issues
        hsl.loc["2015-05":"2015-08"] = None
        # remove spike
        hsl.loc["2019-05-29 02"] = None
        # apparent step across data gap
        hsl = hsl - estimate_step(hsl, t_step="2015-01-01")

    elif uhid == 94:  # Matarani, Peru
        # remove spikes
        hsl.loc["2020-05-26 00"] = None
        hsl.loc["2020-05-29 18"] = None

    elif uhid == 101:  # Mombasa, Kenya
        # nontidal residuals early in the record have a different character; potential timing issue
        hsl.loc[:"1997"] = None

    elif uhid == 108:  # Male, Maldives
        # timing issues
        hsl.loc["2020-06"] = None

    elif uhid == 109:  # Gan, Maldives
        # remove negative spikes
        drop = [
            "2022-09-12 18:00:00",
            "2022-09-15 14:00:00",
            "2022-09-15 16:00:00",
            "2022-09-26 15:00:00",
            "2022-09-27 10:00:00",
            "2022-10-01 16:00:00",
            "2022-10-07 15:00:00",
            "2022-10-08 15:00:00",
            "2022-10-11 18:00:00",
            "2022-10-15 14:00:00",
            "2022-10-17 19:00:00",
            "2022-10-18 07:00:00",
            "2022-10-18 14:00:00",
            "2022-10-29 12:00:00",
            "2022-12-18 02:00:00",
            "2022-12-20 09:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 114:  # Salalah, Oman
        # correct offset at the end of 2016
        hsl = hsl - estimate_step(hsl, t_step="2019-01-01 00")

    elif uhid == 115:  # Colombo, Sri Lanka
        # correct offset at the end of 2016
        hsl = hsl - estimate_step(hsl, t_step="2016-12-01")

    elif uhid == 117:  # Hanimaadhoo, Maldives
        # remove three hours of bad data in 2010
        hsl.loc["2021-01-06 22":"2021-01-07 01"] = None

    elif uhid == 121:  # Pt. La Rue, Seychelles
        # remove negative spikes
        drop = [
            "2021-09-26 17:00:00",
            "2021-09-26 19:00:00",
            "2021-10-30 18:00:00",
            "2021-12-31 01:00:00",
            "2022-02-01 22:00:00",
            "2022-02-08 04:00:00",
            "2022-02-10 20:00:00",
            "2022-02-10 21:00:00",
            "2022-02-16 17:00:00",
            "2022-02-28 20:00:00",
            "2022-03-08 20:00:00",
            "2022-03-09 17:00:00",
            "2022-03-12 18:00:00",
            "2022-03-12 19:00:00",
            "2022-03-12 21:00:00",
            "2022-03-16 17:00:00",
            "2022-03-18 18:00:00",
            "2022-03-18 19:00:00",
            "2022-03-22 18:00:00",
            "2022-03-22 20:00:00",
            "2022-03-23 10:00:00",
            "2022-03-26 16:00:00",
            "2022-03-26 17:00:00",
            "2022-03-26 18:00:00",
            "2022-03-26 20:00:00",
            "2022-03-29 17:00:00",
            "2022-04-05 19:00:00",
            "2022-04-06 20:00:00",
            "2022-04-08 18:00:00",
            "2022-04-20 17:00:00",
            "2022-04-22 20:00:00",
            "2022-04-24 17:00:00",
            "2022-04-24 18:00:00",
            "2022-04-24 20:00:00",
            "2022-05-01 20:00:00",
            "2022-05-04 06:00:00",
            "2022-05-05 20:00:00",
            "2022-05-09 20:00:00",
            "2022-05-30 18:00:00",
            "2022-06-16 22:00:00",
            "2022-06-17 11:00:00",
            "2022-06-23 00:00:00",
            "2022-07-09 22:00:00",
            "2022-09-07 21:00:00",
            "2022-09-11 18:00:00",
            "2022-09-11 19:00:00",
            "2022-09-13 21:00:00",
            "2022-09-17 17:00:00",
            "2022-09-17 20:00:00",
            "2022-09-17 21:00:00",
            "2022-09-21 18:00:00",
            "2022-09-21 19:00:00",
            "2022-10-11 19:00:00",
            "2022-10-13 17:00:00",
            "2022-10-13 18:00:00",
            "2022-10-17 17:00:00",
            "2022-10-19 18:00:00",
            "2022-10-19 19:00:00",
            "2022-10-22 16:00:00",
            "2022-10-22 17:00:00",
            "2022-10-22 23:00:00",
            "2022-10-25 17:00:00",
            "2022-10-28 16:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 123:  # Sabang, Indonesia
        # remove bad data (positive outliers) around 2020–2021
        hsl.loc[hsl > 250] = None

    elif uhid == 124:  # Chittagong, Bangladesh
        # remove short spikes
        hsl.loc["2017-12-15 18":"2017-12-16 00"] = None
        hsl.loc["2017-12-25 10":"2017-12-25 11"] = None
        hsl.loc["2018-01-04 03"] = None
        hsl.loc["2018-01-20 03"] = None
        hsl.loc["2018-01-21 11"] = None
        hsl.loc["2018-10-28 07"] = None
        hsl.loc["2020-02-01 21"] = None
        hsl.loc["2020-02-02 18"] = None
        hsl.loc["2020-02-10 06"] = None
        hsl.loc["2020-02-14 14"] = None
        hsl.loc["2020-03-04 14":"2020-03-04 15"] = None
        hsl.loc["2021-09-27 05"] = None
        hsl.loc["2021-11-01 18"] = None
        hsl.loc["2021-11-26 03"] = None

    elif uhid == 147:  # Karachi, Pakistan
        # remove short spike
        hsl.loc["2018-10-01 02"] = None

    elif uhid == 151:  # Zanzibar, Tanzania
        # remove short spikes
        hsl.loc["2022-02-04 14"] = None
        hsl.loc["2022-02-22 14"] = None
        hsl.loc["2022-02-24 07":"2022-02-24 11"] = None
        hsl.loc["2022-03-20 15"] = None
        hsl.loc["2022-03-30 23"] = None
        hsl.loc["2022-04-05 22"] = None
        hsl.loc["2022-04-22 18"] = None
        hsl.loc["2022-04-23 10"] = None
        hsl.loc["2022-04-23 21"] = None
        hsl.loc["2022-04-24 20"] = None
        hsl.loc["2022-05-01 20"] = None
        hsl.loc["2022-08-04 21"] = None
        hsl.loc["2022-08-17 16"] = None
        hsl.loc["2022-08-21 05"] = None
        hsl.loc["2022-08-26 11"] = None
        hsl.loc["2022-09-14 19"] = None
        hsl.loc["2022-09-20 09"] = None
        hsl.loc["2022-09-22 15"] = None

    elif uhid == 168:  # Darwin, Australia
        # nontidal residuals early in the record have greater std; some evidence of timing issues
        hsl.loc[:"1991"] = None

    elif uhid == 184:  # Port Elizabeth, South Africa
        # remove earliest short section of data due to timing issues
        hsl.loc[:"1975"] = None
        # remove data spikes
        hsl0 = hsl.loc["2018-08-03":"2018-08-04"]
        hsl0.loc[hsl0 > 205] = None
        hsl.loc["2018-09-10 15"] = None
        hsl1 = hsl.loc["2018-09-29":"2018-10-14"]
        hsl1.loc[hsl1 > 230] = None
        hsl.loc["2018-10-07 18"] = None
        hsl.loc["2018-10-14 05"] = None
        hsl.loc["2018-10-14 11"] = None
        hsl2 = hsl.loc["2019-09":"2019-12"]
        hsl2.loc[hsl2 > 250] = None
        hsl3 = hsl.loc["2020-01":"2020-02"]
        hsl3.loc[hsl3 > 220] = None
        hsl.loc["2021-02-17 06"] = None
        hsl.loc["2021-06-03 09"] = None
        hsl.loc["2021-06-17 11"] = None
        hsl.loc["2022-03-25 11"] = None
        hsl.loc["2022-04-16 07":"2022-04-16 08"] = None
        hsl.loc["2022-04-17 11":"2022-04-17 12"] = None
        hsl.loc["2022-04-24 04"] = None
        hsl.loc["2022-04-25 06":"2022-04-25 07"] = None
        hsl.loc["2022-04-29 06"] = None
        hsl.loc["2022-05-19 07"] = None
        hsl.loc["2022-05-21 07"] = None
        hsl.loc["2022-06-13 12"] = None
        hsl.loc["2022-06-16 10"] = None
        hsl.loc["2022-07-05 12":"2022-07-05 13"] = None
        hsl.loc["2022-07-18 09"] = None
        hsl.loc["2022-07-28 11"] = None
        hsl.loc["2022-08-01 06"] = None
        hsl.loc["2022-08-10 14"] = None
        hsl.loc["2022-08-14 06"] = None
        hsl.loc["2022-08-16 09"] = None
        hsl.loc["2022-08-17 07"] = None
        hsl.loc["2022-08-18 07"] = None
        hsl.loc["2022-09-07 06"] = None
        hsl.loc["2022-09-08 07"] = None
        hsl.loc["2022-10-05 11"] = None
        hsl.loc["2022-10-08 06"] = None
        hsl.loc["2022-11-23 09"] = None
        hsl.loc["2023-08-31 11"] = None

    elif uhid == 211:  # Ponta Delgada, Portugal
        # remove negative spikes
        drop = [
            "2020-09-30 16:00:00",
            "2020-12-19 04:00:00",
            "2021-02-07 14:00:00",
            "2021-02-24 15:00:00",
            "2021-02-24 16:00:00",
            "2021-02-24 20:00:00",
            "2021-02-25 00:00:00",
            "2021-02-25 01:00:00",
            "2021-02-25 02:00:00",
            "2021-02-25 03:00:00",
            "2021-02-25 04:00:00",
            "2021-02-25 05:00:00",
            "2021-02-25 06:00:00",
            "2021-02-26 05:00:00",
            "2021-05-24 06:00:00",
            "2021-08-09 17:00:00",
            "2021-11-21 01:00:00",
            "2021-12-04 07:00:00",
            "2022-02-10 17:00:00",
            "2022-03-04 11:00:00",
            "2022-04-18 06:00:00",
            "2022-06-10 04:00:00",
            "2022-06-30 16:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 221:  # Simon's Town, South Africa
        # uncertain stability after 2017
        hsl.loc["2017":] = None
        # timing issues
        hsl.loc["1967-06-26":"1967-07-02"] = None
        hsl.loc["1991-07-01":"1991-07-09"] = None
        hsl.loc["1992-01-01":"1992-01-08"] = None
        hsl.loc["2003-02-01":"2003-02-07"] = None
        hsl.loc["2004-07-19":"2004-07-31"] = None

    elif uhid == 257:  # Settlement Point, Bahamas
        # remove negative spikes
        drop = [
            "2019-10-31 02:00:00",
            "2019-10-31 18:00:00",
            "2019-11-03 01:00:00",
            "2019-11-05 17:00:00",
            "2019-11-05 23:00:00",
            "2019-11-06 23:00:00",
            "2019-11-09 08:00:00",
            "2019-11-10 17:00:00",
            "2019-11-13 00:00:00",
            "2019-11-13 02:00:00",
            "2019-11-15 06:00:00",
            "2019-11-19 14:00:00",
            "2019-11-27 14:00:00",
            "2019-12-06 21:00:00",
            "2019-12-12 03:00:00",
            "2019-12-29 20:00:00",
            "2019-12-30 13:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 259:  # Bermuda
        # isolated, offset data
        hsl.loc["2020-07"] = None

    elif uhid == 271:  # Fort de France, France
        # data early in record fragmented and unclear level consistency
        hsl.loc[:"2000"] = None

    elif uhid == 283:  # Fortaleza, Brazil
        # suspect data
        hsl.loc["2019-02-06":"2019-02-28"] = None
        # remove negative spikes
        drop = [
            "2022-06-14 22:00:00",
            "2022-06-26 15:00:00",
            "2022-06-28 20:00:00",
            "2022-08-02 07:00:00",
            "2022-08-02 12:00:00",
            "2022-08-03 07:00:00",
            "2022-08-03 15:00:00",
            "2022-08-04 06:00:00",
            "2022-08-11 07:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 286:  # Puerto Deseado, Argentina
        drop = [
            "2019-05-15 15:00:00",
            "2019-06-07 22:00:00",
            "2019-06-14 15:00:00",
            "2019-06-27 19:00:00",
            "2019-09-21 10:00:00",
            "2020-01-14 20:00:00",
            "2020-01-16 19:00:00",
            "2020-02-02 02:00:00",
            "2020-02-25 17:00:00",
            "2020-05-01 14:00:00",
            "2020-06-11 21:00:00",
            "2020-07-09 20:00:00",
            "2020-07-10 07:00:00",
            "2020-07-11 02:00:00",
            "2020-07-12 08:00:00",
            "2020-07-12 09:00:00",
            "2020-07-12 13:00:00",
            "2020-07-15 06:00:00",
            "2020-07-15 07:00:00",
            "2020-07-16 02:00:00",
            "2020-07-18 05:00:00",
            "2020-07-18 09:00:00",
            "2020-07-20 01:00:00",
            "2020-07-20 21:00:00",
            "2020-07-21 02:00:00",
            "2020-07-21 07:00:00",
            "2020-07-22 03:00:00",
            "2020-07-22 08:00:00",
            "2020-07-22 13:00:00",
            "2020-07-24 00:00:00",
            "2020-07-24 12:00:00",
            "2020-07-25 06:00:00",
            "2020-07-26 16:00:00",
            "2020-07-27 03:00:00",
            "2020-07-27 19:00:00",
            "2020-07-27 23:00:00",
            "2020-07-28 04:00:00",
            "2020-07-28 08:00:00",
            "2020-07-28 09:00:00",
            "2020-07-29 05:00:00",
            "2020-07-29 09:00:00",
            "2020-07-29 10:00:00",
            "2020-07-30 03:00:00",
            "2020-07-31 00:00:00",
            "2020-07-31 02:00:00",
            "2020-07-31 06:00:00",
            "2020-07-31 07:00:00",
            "2020-07-31 12:00:00",
            "2020-07-31 13:00:00",
            "2020-08-01 04:00:00",
            "2020-08-01 08:00:00",
            "2020-08-01 13:00:00",
            "2020-08-01 18:00:00",
            "2020-08-01 23:00:00",
            "2020-08-02 04:00:00",
            "2020-08-02 05:00:00",
            "2020-08-02 09:00:00",
            "2020-08-02 10:00:00",
            "2020-08-02 11:00:00",
            "2020-08-02 14:00:00",
            "2020-08-03 00:00:00",
            "2020-08-03 02:00:00",
            "2020-08-03 05:00:00",
            "2020-08-03 06:00:00",
            "2020-08-03 10:00:00",
            "2020-08-03 15:00:00",
            "2020-08-03 20:00:00",
            "2020-08-04 01:00:00",
            "2020-08-04 02:00:00",
            "2020-08-04 03:00:00",
            "2020-08-04 06:00:00",
            "2020-08-04 16:00:00",
            "2020-08-05 03:00:00",
            "2020-08-05 04:00:00",
            "2020-08-05 14:00:00",
            "2020-08-05 17:00:00",
            "2020-08-05 23:00:00",
            "2020-08-06 04:00:00",
            "2020-08-06 08:00:00",
            "2020-08-07 00:00:00",
            "2020-08-07 05:00:00",
            "2020-08-07 09:00:00",
            "2020-08-07 15:00:00",
            "2020-08-08 00:00:00",
            "2020-08-08 01:00:00",
            "2020-08-08 05:00:00",
            "2020-08-09 01:00:00",
            "2020-08-09 06:00:00",
            "2020-08-09 16:00:00",
            "2020-08-09 22:00:00",
            "2020-08-10 03:00:00",
            "2020-08-10 07:00:00",
            "2020-08-10 08:00:00",
            "2020-08-10 12:00:00",
            "2020-08-11 03:00:00",
            "2020-08-11 08:00:00",
            "2020-08-12 04:00:00",
            "2020-08-12 12:00:00",
            "2020-08-17 04:00:00",
            "2020-08-23 05:00:00",
            "2020-08-24 23:00:00",
            "2020-08-26 03:00:00",
            "2020-08-27 09:00:00",
            "2020-08-28 05:00:00",
            "2020-08-31 08:00:00",
            "2020-08-31 21:00:00",
            "2020-09-01 09:00:00",
            "2020-09-02 00:00:00",
            "2020-09-02 05:00:00",
            "2020-09-03 01:00:00",
            "2020-09-03 02:00:00",
            "2020-09-03 06:00:00",
            "2020-09-03 21:00:00",
            "2020-09-04 07:00:00",
            "2020-09-04 17:00:00",
            "2020-09-05 03:00:00",
            "2020-09-06 04:00:00",
            "2020-09-06 09:00:00",
            "2020-09-07 00:00:00",
            "2020-09-08 06:00:00",
            "2020-09-09 02:00:00",
            "2020-09-09 12:00:00",
            "2020-09-09 20:00:00",
            "2020-09-10 08:00:00",
            "2020-09-14 05:00:00",
            "2020-09-14 20:00:00",
            "2020-09-15 18:00:00",
            "2020-09-16 05:00:00",
            "2020-09-16 10:00:00",
            "2020-09-16 14:00:00",
            "2020-09-16 18:00:00",
            "2020-09-16 19:00:00",
            "2020-09-16 21:00:00",
            "2020-09-17 02:00:00",
            "2020-09-18 08:00:00",
            "2020-09-18 09:00:00",
            "2020-09-18 10:00:00",
            "2020-09-18 22:00:00",
            "2020-09-19 12:00:00",
            "2020-09-19 21:00:00",
            "2020-09-19 22:00:00",
            "2020-09-20 13:00:00",
            "2020-09-20 16:00:00",
            "2020-09-20 23:00:00",
            "2020-09-21 02:00:00",
            "2020-09-21 06:00:00",
            "2020-09-21 22:00:00",
            "2020-09-21 23:00:00",
            "2020-09-22 02:00:00",
            "2020-09-22 03:00:00",
            "2020-09-22 06:00:00",
            "2020-09-22 07:00:00",
            "2020-09-22 13:00:00",
            "2020-09-23 02:00:00",
            "2020-09-23 03:00:00",
            "2020-09-23 04:00:00",
            "2020-09-23 07:00:00",
            "2020-09-23 09:00:00",
            "2020-09-26 08:00:00",
            "2020-09-26 10:00:00",
            "2020-09-26 13:00:00",
            "2020-09-26 23:00:00",
            "2020-09-27 11:00:00",
            "2020-09-28 01:00:00",
            "2020-09-28 06:00:00",
            "2020-10-06 04:00:00",
            "2020-10-06 09:00:00",
            "2020-10-06 14:00:00",
            "2020-10-07 00:00:00",
            "2020-10-07 22:00:00",
            "2020-10-08 01:00:00",
            "2020-10-08 02:00:00",
            "2020-10-08 04:00:00",
            "2020-10-08 06:00:00",
            "2020-10-08 11:00:00",
            "2020-10-08 15:00:00",
            "2020-10-09 02:00:00",
            "2020-10-09 06:00:00",
            "2020-10-09 07:00:00",
            "2020-10-09 22:00:00",
            "2020-10-11 09:00:00",
            "2020-10-11 12:00:00",
            "2020-10-11 19:00:00",
            "2020-10-12 13:00:00",
            "2020-10-13 01:00:00",
            "2020-10-13 03:00:00",
            "2020-10-14 12:00:00",
            "2020-10-14 16:00:00",
            "2020-10-14 17:00:00",
            "2020-10-16 02:00:00",
            "2020-10-17 00:00:00",
            "2020-10-17 06:00:00",
            "2020-10-20 07:00:00",
            "2020-10-20 08:00:00",
            "2020-10-20 16:00:00",
            "2020-10-22 15:00:00",
            "2020-10-22 19:00:00",
            "2020-10-24 08:00:00",
            "2020-10-24 10:00:00",
            "2020-10-24 11:00:00",
            "2020-10-24 12:00:00",
            "2020-10-24 13:00:00",
            "2020-10-24 14:00:00",
            "2020-10-24 17:00:00",
            "2020-10-25 00:00:00",
            "2020-10-25 22:00:00",
            "2020-10-26 03:00:00",
            "2020-10-27 01:00:00",
            "2020-10-27 18:00:00",
            "2020-10-27 22:00:00",
            "2020-10-30 05:00:00",
            "2020-10-30 19:00:00",
            "2020-10-30 21:00:00",
            "2020-10-30 23:00:00",
            "2020-11-01 11:00:00",
            "2020-11-02 08:00:00",
            "2020-11-02 12:00:00",
            "2020-11-02 13:00:00",
            "2020-11-02 18:00:00",
            "2020-11-02 21:00:00",
            "2020-11-03 01:00:00",
            "2020-11-03 04:00:00",
            "2020-11-03 07:00:00",
            "2020-11-03 08:00:00",
            "2020-11-03 12:00:00",
            "2020-11-04 23:00:00",
            "2020-11-05 17:00:00",
            "2020-11-06 00:00:00",
            "2020-11-06 02:00:00",
            "2020-11-06 04:00:00",
            "2020-11-06 07:00:00",
            "2020-11-06 10:00:00",
            "2020-11-06 14:00:00",
            "2020-11-06 19:00:00",
            "2020-11-07 08:00:00",
            "2020-11-08 00:00:00",
            "2020-11-08 05:00:00",
            "2020-11-08 23:00:00",
            "2020-11-09 03:00:00",
            "2020-11-09 10:00:00",
            "2020-11-09 21:00:00",
            "2020-11-10 06:00:00",
            "2020-11-10 10:00:00",
            "2020-11-10 19:00:00",
            "2020-11-11 05:00:00",
            "2020-11-11 07:00:00",
            "2020-11-11 14:00:00",
            "2020-11-11 18:00:00",
            "2020-11-13 22:00:00",
            "2020-11-15 05:00:00",
            "2020-11-16 23:00:00",
            "2020-11-17 03:00:00",
            "2020-11-17 19:00:00",
            "2020-11-17 21:00:00",
            "2020-11-18 04:00:00",
            "2020-11-18 11:00:00",
            "2020-11-18 13:00:00",
            "2020-11-18 14:00:00",
            "2020-11-18 20:00:00",
            "2020-11-18 21:00:00",
            "2020-11-18 22:00:00",
            "2020-11-19 04:00:00",
            "2020-11-19 10:00:00",
            "2020-11-19 14:00:00",
            "2020-11-19 17:00:00",
            "2020-11-20 06:00:00",
            "2020-11-21 10:00:00",
            "2020-11-21 12:00:00",
            "2020-11-21 14:00:00",
            "2020-11-21 15:00:00",
            "2020-11-22 14:00:00",
            "2020-11-22 19:00:00",
            "2020-11-24 00:00:00",
            "2020-11-25 22:00:00",
            "2020-11-26 11:00:00",
            "2020-11-26 19:00:00",
            "2020-11-26 20:00:00",
            "2020-11-26 21:00:00",
            "2020-11-27 00:00:00",
            "2020-11-27 03:00:00",
            "2020-11-27 15:00:00",
            "2020-11-28 00:00:00",
            "2020-11-28 03:00:00",
            "2020-11-28 21:00:00",
            "2020-11-28 23:00:00",
            "2020-11-29 03:00:00",
            "2020-11-29 17:00:00",
            "2020-11-30 02:00:00",
            "2020-12-03 21:00:00",
            "2020-12-04 22:00:00",
            "2020-12-05 02:00:00",
            "2020-12-05 04:00:00",
            "2020-12-06 00:00:00",
            "2020-12-06 01:00:00",
            "2020-12-06 06:00:00",
            "2020-12-06 07:00:00",
            "2020-12-06 08:00:00",
            "2020-12-06 09:00:00",
            "2020-12-06 16:00:00",
            "2020-12-07 06:00:00",
            "2020-12-07 07:00:00",
            "2020-12-07 11:00:00",
            "2020-12-07 15:00:00",
            "2020-12-07 19:00:00",
            "2020-12-08 00:00:00",
            "2020-12-08 04:00:00",
            "2020-12-08 06:00:00",
            "2020-12-08 09:00:00",
            "2020-12-08 11:00:00",
            "2020-12-09 05:00:00",
            "2020-12-09 12:00:00",
            "2020-12-09 18:00:00",
            "2020-12-10 01:00:00",
            "2020-12-10 03:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 290:  # Port Stantley, Falkland Islands
        # suspect data
        hsl.loc["2016-11-16 20":"2016-11-16 22"] = None
        # focus on record before 2018 due to apparent nonliner subsidence and other data issues
        hsl = hsl.loc[:"2016-09"]

    elif uhid == 299:  # Qaqortoq, Greenland
        # suspect data
        hsl.loc["2022-09-24":"2022-10-23"] = None
        # correct offset after gap in 2022
        hsl = hsl - estimate_step(hsl, t_step="2022-07-01")

    elif uhid == 332:  # Bundaberg, Australia
        # timing issues
        hsl.loc["1987-03-30":"1987-06-03"] = None
        # correct offset after gap in 2022
        hsl = hsl - estimate_step(hsl, t_step="2018-12-31 18")

    elif uhid == 334:  # Townsville, Australia
        # correct offset after gap in 2022
        hsl = hsl - estimate_step(hsl, t_step="2018-12-31 18")

    elif uhid == 345:  # Nakano Shima, Japan
        # remove 2012– due to apparent change in trend; likely onset of subsidence
        hsl.loc["2022":] = None

    elif uhid == 351:  # Ofunato, Japan
        # remove 2012– due to drastic change in subsidence and trend
        hsl.loc["2012-01-01":] = None

    elif uhid == 359:  # Naze, Japan
        # isolate central, stable portion of the record
        hsl = hsl.loc["1976-01":"2020-03"]

    elif uhid == 370:  # Manila, Phillipines
        # correct offset after gap in 2022
        hsl = hsl - estimate_step(hsl, t_step="2017-09-01 00")

    elif uhid == 371:  # Lagaspe, Phillipines
        # timing issues
        hsl.loc["1985-01-31":"1985-02-11"] = None
        # correct multiple offsets between spans of data
        hsl = hsl - estimate_step(
            hsl,
            t_step=[
                "2014-02-01",
                "2015-10-20",
                "2016-07-01",
                "2017-09-01",
                "2023-07-26 17",
            ],
        )

    elif uhid == 372:  # Davao, Philliipines
        # remove suspect early portions of the record
        hsl.loc[:"2003"] = None

    elif uhid == 547:  # Barbers Point, Hawaii
        # remove suspect, spotty data
        hsl.loc["2019-01":"2019-02"] = None
        # remove spikes
        drop = [
            "2019-03-18 12:00:00",
            "2019-03-18 13:00:00",
            "2019-03-18 14:00:00",
            "2020-07-03 23:00:00",
            "2020-12-22 08:00:00",
            "2021-02-05 06:00:00",
            "2021-03-13 05:00:00",
            "2021-05-26 02:00:00",
            "2022-01-22 15:00:00",
            "2022-01-22 16:00:00",
            "2022-01-22 17:00:00",
            "2022-01-22 18:00:00",
            "2022-01-22 19:00:00",
        ]
        hsl.loc[drop] = None

    elif uhid == 552:  # Kawaihae, Hawaii
        # exclude most recent data due to possible seismic subsidence (unsure)
        hsl.loc["2021-03":] = None
        # correct offset after known 2006 earthquake
        hsl = hsl - estimate_step(hsl, t_step="2006-10-15 18")

    elif uhid == 570:  # Yakutat, Alaska
        # data early in the record has a different character and timing issues
        hsl.loc[:"1977"] = None

    elif uhid == 684:  # Puerto Montt, Chile
        # residuals early in the record have a different character; likely timing issues
        hsl.loc[:"1999-01"] = None

    elif uhid == 708:  # Salvador, Brazil
        # remove large negative spike
        hsl.loc["2019-06-17 15"] = None
        # remove positive spikes
        drop = [
            "2022-01-04 23:00:00",
            "2022-01-12 17:00:00",
            "2022-01-25 00:00:00",
            "2022-02-10 02:00:00",
            "2022-02-15 00:00:00",
            "2022-02-20 00:00:00",
            "2022-02-22 23:00:00",
            "2022-02-28 01:00:00",
            "2022-03-07 23:00:00",
            "2022-03-09 01:00:00",
            "2022-03-10 01:00:00",
            "2022-03-19 23:00:00",
            "2022-07-06 04:00:00",
            "2022-07-28 12:00:00",
            "2022-10-07 23:00:00",
            "2022-10-13 23:00:00",
            "2022-10-17 23:00:00",
            "2022-10-24 00:00:00",
        ]
        hsl.loc[drop] = None
        # correct offset near end of the record
        hsl = hsl - estimate_step(hsl, t_step="2023-02-15")

    elif uhid == 825:  # Cuxhaven, Germany
        # remove large negative spike
        hsl.loc["2015-10-28 09":"2015-10-28 11"] = None

    elif uhid == 826:  # Stockholm, Sweden
        # correct offset near end of the record
        hsl = hsl - estimate_step(hsl, t_step="2020-04")

    elif uhid == 833:  # Nain, Canada
        # residuals early in the record have a different character; likely timing issues
        hsl.loc[:"1999"] = None

    elif uhid == 830:  # La Coruna, Spain
        # egregious timing issues (there appear to be many additional smaller issues)
        hsl.loc["1973-11-18":"1973-12-18"] = None
        hsl.loc["2004-06":"2004-09"] = None
        hsl.loc["2017-09-30 23":"2017-11-30 21"] = None

    elif uhid == 835:  # Castletownbere, Ireland
        # timing issues
        hsl.loc["2021-11":"2021-12"] = None

    else:
        qc = False

    return hsl, qc