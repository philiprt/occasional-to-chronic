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

    if uhid == 8:  # Yap
        # remove one bad negative outlier in late 2019
        hsl.loc[hsl < -50] = None

    elif uhid == 16:  # Rikitea
        # remove section of data that appears to be height-limited or capped
        hsl.loc["2003-07":"2007-09"] = None
        # remove negative spikes
        hsl.loc[
            [
                "2022-04-18 17",
                "2022-07-08 23",
                "2022-07-09 20",
                "2022-07-14 05",
                "2022-08-15 09",
                "2022-09-17 19",
                "2022-09-24 01",
                "2022-10-27 20",
            ]
        ] = None

    elif uhid == 17:  # Hiva Oa
        # remove two stretches of data inconsistent with rest of the record
        hsl.loc["2021-08":"2021-10"] = None
        hsl.loc["2023-05":] = None

    elif uhid == 31:  # Nuku Hiva
        # remove three hours of bad data in 2010
        hsl.loc["2010-02-27 18":"2010-02-27 20"] = None
        # remove end of record due to data inconsistent with rest of the record
        hsl.loc["2019-10":] = None

    elif uhid == 33:  # Bitung, Indonesia
        # remove negative spike
        hsl.loc["2016-01-01 00"] = None

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

    elif uhid == 114:  # Salalah, Oman
        # correct offset at the end of 2016
        hsl = hsl - estimate_step(hsl, t_step="2019-01-01 00")

    elif uhid == 115:  # Colombo, Sri Lanka
        # correct offset at the end of 2016
        hsl = hsl - estimate_step(hsl, t_step="2016-12-01")

    elif uhid == 123:  # Sabang, Indonesia
        # remove bad data (positive outliers) around 2020–2021
        hsl.loc[hsl > 250] = None

    elif uhid == 124:  # Chittagong, Bangladesh
        # remove short spikes
        hsl.loc["2021-09-27 05"] = None

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

    elif uhid == 221:  # Simon's Town, South Africa
        # uncertain stability after 2017
        hsl.loc["2017":] = None
        # timing issues
        hsl.loc["1967-06-26":"1967-07-02"] = None
        hsl.loc["1991-07-01":"1991-07-09"] = None
        hsl.loc["1992-01-01":"1992-01-08"] = None
        hsl.loc["2003-02-01":"2003-02-07"] = None
        hsl.loc["2004-07-19":"2004-07-31"] = None

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
        hsl.loc[:"2011-04-01":] = None

    elif uhid == 356:  # Maisaka, Japan
        # remove early record due to suspect vertical control
        hsl.loc[:"1963-07-01"] = None

    elif uhid == 359:  # Naze, Japan
        # isolate central, stable portion of the record
        hsl = hsl.loc["1976-01":"2020-03"]

    elif uhid == 370:  # Manila, Phillipines
        # correct offset after gap in 2022
        hsl = hsl - estimate_step(hsl, t_step="2015-01-01 00")
        # remove offset section with timing issues
        hsl.loc["2024-05"] = None

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
        # spotty, suspect data
        hsl.loc["2024-09":] = None

    elif uhid == 4:  # Nauru
        # short spike in residuals
        hsl.loc["2023-07-01 00":"2023-07-01 01"] = None
        # timing off after gap through end of series
        hsl.loc["2023-10-01 00":] = None

    elif uhid == 12:  # Fanning, Kiribati
        # timing and vertical control issues early in the record
        hsl.loc[:"1975-01"] = None

    elif uhid == 19:  # Noumea, New Caledonia
        # suspect spike
        hsl.loc["2003-03-13 22":"2003-03-14 02"] = None

    # elif uhid == 34:  # Cabo San Lucas, Mexico
    #     # suspect timing and stability prior to 1990
    #     hsl.loc[:"1990"] = None

    elif uhid == 64:  # Port Morseby, Papua New Guinea
        # severe timing issues
        hsl.loc["1994-12-01":] = None

    elif uhid == 72:  # Bluff, New Zealand
        # suspect timing prior to 1990
        hsl.loc[:"1990"] = None

    elif uhid == 84:  # Lobos de Afuera, Peru
        # suspect timing and stability 2007-2010
        hsl.loc["2007":"2010"] = None

    elif uhid == 85:  # Buena Ventura, Colombia
        hsl.loc[["1953-06-21 18", "1953-06-21 19"]] = None  # short spike

    elif uhid == 86:  # La Union, El Salvador
        # # bad timing
        # hsl.loc["1979-12-01 15":"1979-12-05"] = None
        # hsl.loc["1980-07-11":"1980-07-19"] = None
        # suspect vertical stability
        hsl.loc[:"1960-09"] = None
        hsl = hsl - estimate_step(hsl, t_step="1970-01-01")

    elif uhid == 89:  # Manta, Ecuador
        # poor quality record
        hsl.loc[:] = None

    elif uhid == 93:  # Callao, Peru
        # bad timing
        # hsl.loc["2022-01-15 18":"2022-01-19 00"] = None
        # bad timing and odd shifts throughout the record
        hsl.loc[:] = None

    elif uhid == 96:  # San Juan, Peru
        # strange data during first half of record; appears blocky in places like restricted to discrete values.
        hsl.loc[:"1992-01-05"] = None

    elif uhid == 98:  # Esmeraldas, Ecuador
        hsl.loc["2005-06"] = None  # severe timing issues
        # loss of vertical control
        hsl.loc["2010-02-03 04":"2010-02-04 14"] = None

    elif uhid == 107:  # Padang, Indonesia
        # loss of vertical control
        hsl.loc["2023-10-27":] = None
        # spikes
        hsl.loc["2019-08-25 16":"2019-08-25 20"] = None
        hsl.loc[
            [
                "2019-09-09 14",
                "2019-09-10 01",
                "2019-09-10 04",
                "2019-09-10 08",
                "2019-09-10 10",
                "2019-09-10 14",
                "2019-09-11 14",
                "2019-09-12 08",
                "2019-09-12 13",
                "2019-09-13 11",
                "2019-10-16 22",
                "2019-10-22 09",
                "2019-12-04 21",
                "2020-03-02 02",
                "2020-06-10 08",
                "2020-06-12 07",
                "2020-06-12 21",
                "2020-06-13 13",
                "2020-06-13 18",
                "2020-06-13 20",
                "2020-06-14 05",
                "2020-06-14 08",
                "2020-06-14 11",
                "2020-06-14 12",
                "2020-06-14 14",
                "2020-06-16 17",
                "2020-06-18 17",
                "2020-06-19 17",
                "2020-06-20 01",
                "2020-06-20 08",
                "2020-06-20 13",
                "2020-06-20 14",
                "2020-06-20 16",
                "2020-06-21 04",
                "2020-08-31 12",
            ]
        ] = None

    elif uhid == 113:  # Masirah, Oman
        hsl.loc["2018-04-28":"2018-09-18"] = None  # suspect data
        hsl.loc["2019-07":] = None  # loss of vertical control

    elif uhid == 122:  # Sibolga, Indonesia
        # loss of vertical control and suspect data
        hsl.loc["2020-07-26 10":] = None

    elif uhid == 125:  # Prigi, Indonesia
        hsl.loc["2021-11-15":] = None  # negative spikes and spotty data
        # spikes
        hsl.loc[
            [
                "2018-08-30 02",
                "2019-09-01 05",
                "2019-09-07 01",
                "2019-11-19 07",
            ]
        ] = None
        hsl.loc["2019-11-25 00":"2019-12-01 00"] = None  # suspect data

    elif uhid == 127:  # Syowa, Antarctica, Antarctica
        # potential vertical stability issues during first part of the record
        hsl.loc[:"2007"] = None

    elif uhid == 134:  # Hiron Point, Bangladesh
        # loss of vertical control and timing
        hsl.loc["2000-11-16 17":"2000-11-21 18"] = None

    elif uhid == 139:  # Khepupara, Bangladesh
        # unclear vertical stability
        hsl[:] = None

    elif uhid == 142:  # Langkawi, Malaysia
        # loss of vertical control
        hsl.loc["2018-02-28 20":"2018-03-31 20"] = None
        # severe timing issues
        hsl.loc["2022-09":"2023-12"] = None

    elif uhid == 148:  # Ko Taphao Noi, Thailand
        # spotty, suspect data
        hsl.loc["2020-05-07 21":"2020-06-08 13"] = None

    elif uhid == 150:  # Nosy Be, Madagascar
        hsl.loc["1966-08-15 04":"1966-08-24 10"] = None  # suspect
        hsl.loc["1972-03-24 19":"1972-03-30 05"] = None  # loss of vertical control

    elif uhid == 157:  # Vishakhapatnam, India
        hsl.loc[:] = None  # poor vertical control throughout
        # hsl.loc["2014-02-03 04":"2014-06-08 22"] = None
        # hsl.loc["2014-10-12 02":"2014-11-03"] = None
        # hsl.loc["2017-04-13 17"] = None
        # hsl.loc[["2017-06-09 22", "2017-06-09 23"]] = None
        # hsl.loc["2019-07-09 00":"2019-08-08 11"] = None
        # hsl = hsl - estimate_step(
        #     hsl,
        #     t_step=[
        #         "2014-09-01 00",
        #         "2016-02-02 11",
        #         "2019-07-08 00",
        #     ],
        # )

    elif uhid == 160:  # Surabaya, Indonesia
        # severe timing issues
        hsl.loc["1993-07-09 15":"1993-07-27 02"] = None
        hsl.loc["1997-01-24 11":"1997-01-31 03"] = None
        hsl.loc["1999-01-14 03":"1999-01-25 05"] = None

    elif uhid == 177:  # Mawson, Australia
        hsl.loc["1992-01-05 17":"1992-03-10 09"] = None

    elif uhid == 179:  # Saint Paul, France
        hsl.loc["2004-12-26 22":"2004-12-27 08"] = None  # negative spike
        hsl.loc["2008-11-23 05":"2016-04-27 13"] = None  # bad stability
        hsl.loc["2021-09-22 05":"2021-09-22 08"] = None  # spike
        hsl = hsl - estimate_step(hsl, t_step="2019-12-26 03")

    elif uhid == 180:  # Kerguelen, France
        hsl.loc["2022-05-16 06":"2022-05-16 10"] = None  # spike
        hsl = hsl - estimate_step(hsl, t_step="2016-01-01 00")

    elif uhid == 185:  # Mossel Bay, South Africa
        hsl.loc["1984-01-16 02":"1984-04-15 21"] = None  # severe timing issues

    elif uhid == 187:  # East London, South Africa
        hsl.loc["2000-12-31 22":"2002-12-31 21"] = None  # poor stability

    elif uhid == 188:  # Richard's Bay, South Africa
        hsl.loc["2003-01-01 00":"2003-02-14 11"] = None  # severe timing issues
        hsl.loc["2005-05-20 12":"2005-05-23 12"] = None  # severe timing issues
        hsl.loc["1997-05-22 08":"1997-12-31 21"] = None  # poor stability

    elif uhid == 210:  # Flores, Santa Cruz, Portugal
        # vertical stability is suspect in early part of the record with an interannaual rise similar in magnitude to the largest storm surge
        hsl.loc[:"1982-01-21"] = None

    elif uhid == 217:  # Las Palmas, Spain
        hsl.loc[["2021-05-14 07", "2021-05-14 08", "2021-05-14 09"]] = None  # spike

    elif uhid == 218:  # Funchal, Portugal
        # record contains mulitple stretches of data during the 1990s and early 2000s
        # that appear suspect, though the signature is not clearly unphysical. the peaks
        # are uncharacteristic of other surge events in the record, however. Given that
        # there are other GN records in the vicinity, we ignore this one.
        hsl.loc[:] = None

    elif uhid == 220:  # Walvis Bay, Namibia
        hsl.loc[["1966-12-31 22", "1966-12-31 23"]] = None  # isolated and suspect spike
        hsl.loc["1978-12-15 08":"1978-12-21 06"] = None  # severe timing issues
        hsl.loc["1987-09-20 09":"1987-09-23 15"] = None  # severe timing issues
        hsl.loc["1994-07-27 22":"1994-08-24 08"] = None  # severe timing issues
        hsl.loc["1994-09-05 13":"1994-09-05 22"] = None  # suspect spike after gap
        hsl.loc["1995-08-17 05":"1995-09-03 05"] = None  # severe timing issues
        hsl.loc["1995-12-04 09":"1995-12-16 13"] = None  # severe timing issues

    elif uhid == 223:  # Dakar, Senegal
        # end of record has vertical control and timing issues
        hsl.loc["2022-07-12 14":"2023-11-30 23"] = None
        hsl.loc["2022-01-19 21"] = None  # spike
        hsl.loc[["2022-03-17 03", "2022-03-17 23", "2022-03-22 21"]] = None  # spikes
        hsl.loc["2021-07-12 14"] = None  # spike
        hsl.loc[
            [
                "2020-09-11 01",
                "2020-09-19 04",
                "2020-09-26 13",
                "2020-09-30 21",
                "2020-10-08 06",
                "2019-08-18 23",
                "2019-09-16 18",
                "2019-09-16 21",
                "2019-09-16 22",
                "2019-09-24 09",
                "2019-10-03 01",
                "2019-10-03 02",
                "2019-10-04 01",
                "2019-10-04 02",
                "2019-10-05 13",
                "2019-08-12 02",
                "2019-09-07 12",
                "2019-09-22 10",
                "2019-09-29 16",
                "2019-10-02 13",
                "2021-10-03 00",
                "2021-10-05 10",
                "2021-10-08 07",
                "2021-09-23 05",
            ]
        ] = None  # spikes

    elif uhid == 240:  # Fernandina Beach, FL
        hsl.loc[:"1960"] = None  # remove old disontinguous data

    elif uhid == 247:  # La Guaira, Venezuela (Bolivarian Republic of)
        hsl.loc[:] = None  # strange blocky data throughout

    elif uhid == 250:  # Veracruz, Ver., Mexico
        # fix potential level shifts
        hsl = hsl - estimate_step(hsl, t_step=["2005-01-01 00", "2007-01-01 00"])

    elif uhid == 265:  # Cartagena, Colombia
        hsl.loc[:] = None  # poor stability and overall quality

    elif uhid == 266:  # Cristobal, Panama
        # vertical stability is an issue later in the record, but there is more than
        # enough data to use the earlier portion of the record
        hsl.loc["1980":] = None

    elif uhid == 267:  # Mona Island, PR
        # loss of vertical control in early record
        hsl = hsl - estimate_step(hsl, t_step="2009-01-01 00")

    elif uhid == 273:  # Port-aux-basques, Canada
        # susepect vertical stability and severe timing issues
        hsl.loc["1999-09-15":"2000-03-01"] = None

    elif uhid == 280:  # Ilha Fiscal, RJ, Brazil
        # suspect timing and stability
        hsl.loc["1984-07-04":"1984-09-23"] = None
        # suspect stability
        hsl.loc["1994-01-24 11":"1994-01-31 04"] = None

    elif uhid == 281:  # Cananeia, Brazil
        # extremely large event (3 m) that looks suspect but is not clearly unphysical;
        # no known event can be identified that would have produced such a surge
        hsl.loc["1999-10-29 12":"1999-10-30 11"] = None

    elif uhid == 289:  # Gibraltar, UK
        # spotty suspect data
        hsl.loc["2017-08-14 11":"2018-06-24 19"] = None

    elif uhid == 291:  # Ascension, United Kingdom
        # short, poor-quality record
        hsl.loc[:] = None

    elif uhid == 303:  # Tumaco, Colombia
        # tidal amplitude too large and timing off
        hsl.loc["1994-12-01 18":"1994-12-02 09"] = None
        # loss of vertical control
        hsl = hsl - estimate_step(hsl, t_step=["1998-10-01 06", "1998-11-01 05"])

    elif uhid == 316:  # Acapulco, Gro., Mexico
        # severe timing issues
        hsl.loc["1988-11-01 06":"1988-11-28 21"] = None
        hsl.loc["1989-12-01 06":"1990-01-01 05"] = None
        # severe timing issues and loss of vertical control
        hsl.loc["1995-12-29 02":"1995-12-31 23"] = None

    elif uhid == 328:  # Ko Lak, Thailand
        # loss of vertical control and bad timing
        hsl.loc["2011-11-20 16":"2011-11-22 16"] = None
        # spike
        hsl.loc["2021-05-31 19"] = None

    elif uhid == 340:  # Kaohsiung, Taiwan (Province of China)
        # loss of vertical control and/or suspect data
        hsl.loc[:"1981"] = None
        hsl.loc["2001-05":"2003-08"] = None

    elif uhid == 341:  # Keelung, Taiwan (Province of China)
        # time series has suspect vertical control throughout
        hsl.loc[:] = None

    elif uhid == 357:  # Keelung, Taiwan (Province of China)
        # time series has suspect vertical control throughout
        hsl.loc[:] = None

    elif uhid == 363:  # Miyakejima, Japan
        # nonlinear VLM
        hsl.loc[:] = None

    elif uhid == 379:  # Cebu, Philippines (the)
        # short record and suspect vertical stability throughout
        hsl.loc[:] = None

    elif uhid == 383:  # Vung Tau, Viet Nam
        # spikes and suspect residuals
        hsl.loc[
            [
                "2019-01-19 23",
                "2020-12-05 02",
                "2020-12-05 03",
                "2020-12-05 04",
                "2020-12-05 05",
                "2020-12-05 06",
                "2020-12-17 02",
                "2020-12-17 03",
                "2020-12-17 04",
                "2020-12-17 05",
                "2020-12-17 06",
                "2020-12-17 07",
            ]
        ] = None

    elif uhid == 394:  # Salina Cruz, Mexico
        # brief loss of vertical control
        hsl.loc["1983-03-01 09":"1983-07-01 19"] = None
        # severe timing issues
        hsl.loc["1985-10-12 12":"1985-10-17 20"] = None
        hsl.loc["1986-05-14 19":"1986-05-26 02"] = None
        hsl.loc["1986-07-01 18":"1986-07-07 13"] = None
        # severe timing issues and potential loss of vertical control
        hsl.loc["1956-12-24 19":"1958-01-15 17"] = None

    elif uhid == 401:  # Apia, Samoa
        hsl = hsl - estimate_step(hsl, t_step="2009-01-01 00")

    elif uhid == 418:  # Waikelo, Indonesia
        # short and poor quality record
        hsl.loc[:] = None

    elif uhid == 541:  # Bamfield, Canada
        # spike
        hsl.loc["2021-05-08 15"] = None

    elif uhid == 641:  # Shanwei, China
        # vertical potentially lost at the end of the time series
        hsl.loc["1997-01-01 00":] = None

    elif uhid == 654:  # Currimao, Philippines (the)
        # data spikes
        hsl.loc[
            [
                "2018-06-14 07",
                "2021-04-28 05",
                "2021-04-29 04",
                "2021-04-29 05",
                "2021-04-30 19",
            ]
        ] = None

    elif uhid == 671:  # La Paz, Mexico
        # loss of vertical control
        hsl.loc["1974-05-20 06":"1974-06-03 09"] = None

    elif uhid == 672:  # Puerto Angel, Mexico
        # severe timing issues or potentially just bad data
        hsl.loc["1962-10-23 22":"1963-01-01 05"] = None
        # unclear vertical stability at the end of the record
        hsl.loc["1983":] = None

    elif uhid == 673:  # Mazatlan, Mexico
        hsl.loc["1954-09-29 07":"1954-09-30 06"] = None

    elif uhid == 675:  # San Jose, Guatemala
        # vertical stability is doubtful in multiple places;
        # electing to remove the record from analysis
        hsl.loc[:] = None

    elif uhid == 676:  # Topolobampo, Mexico
        # severe timing issues
        hsl.loc["1960-06-24 08":"1960-06-25 06"] = None

    elif uhid == 678:  # Paita, Peru
        # many short gaps and unclear vertical stability throughout
        hsl.loc[:] = None

    elif uhid == 680:  # Macquarie Is., Australia
        # many short gaps and unclear vertical stability throughout
        hsl.loc[:] = None

    elif uhid == 683:  # Pisco, Peru
        # suspect vertical stability throughout
        hsl.loc[:] = None

    elif uhid == 699:  # Tanjong Pagar, Singapore
        # severe timing issues and spikes
        hsl.loc["2017-01-01 02":"2017-02-28 22"] = None
        hsl.loc["2018-02-02 00":"2018-02-28 22"] = None
        hsl.loc["2019-02-19 10":"2019-02-20 15"] = None
        hsl.loc["2019-06-06 10":"2019-06-07 08"] = None
        hsl.loc["2019-06-25 21":"2019-06-26 14"] = None

    elif uhid == 701:  # Port Nolloth, South Africa
        # severe timing issues
        hsl.loc["2004-07-30 07":"2004-11-04 03"] = None
        hsl.loc["2007-01-05 11":"2007-01-10 03"] = None

    elif uhid == 702:  # Luderitz, South Africa
        # suspicious extremely large values
        hsl.loc["2015-09-13 09":"2015-09-13 15"] = None
        # severe timing issues
        hsl.loc["1961-04-23 16":"1961-04-27 06"] = None
        hsl.loc["1978-05-16 15":"1978-05-25 07"] = None
        hsl.loc["1978-09-04 08":"1978-09-08 04"] = None
        hsl.loc["1978-10-13 23":"1978-10-17 07"] = None
        hsl.loc["1978-12-24 09":"1978-12-30 07"] = None
        # spikes
        hsl.loc["1972-02-19 23":"1972-02-24 02"] = None
        hsl.loc[
            [
                "1982-03-21 00",
                "1982-03-21 01",
                "1982-03-21 02",
                "1982-03-21 03",
                "1982-03-27 05",
                "1982-03-27 06",
                "1982-03-27 07",
                "1982-03-27 08",
                "1982-03-28 23",
                "1982-03-29 00",
                "1982-03-29 01",
                "1982-03-29 02",
                "1982-04-08 04",
                "1982-04-08 05",
                "1982-04-08 06",
                "1982-04-10 10",
                "1982-04-10 11",
                "1982-04-10 12",
                "1982-04-10 13",
                "1982-04-20 00",
                "1982-04-20 01",
                "1982-04-20 02",
                "1982-04-20 03",
                "1982-03-22 23",
                "1982-03-23 00",
                "1982-03-23 01",
                "1982-03-23 02",
                "1982-03-26 23",
                "1982-03-27 00",
                "1982-03-27 01",
                "1982-03-27 02",
                "1982-03-29 05",
                "1982-03-29 06",
                "1982-03-29 07",
                "1982-03-29 08",
                "1982-04-08 10",
                "1982-04-08 11",
                "1982-04-08 12",
                "1982-04-08 13",
                "1982-04-10 04",
                "1982-04-10 05",
                "1982-04-10 06",
                "1982-04-22 00",
                "1982-04-22 01",
                "1982-04-22 02",
                "1982-04-22 03",
            ]
        ] = None
        # loss of vertical control
        hsl.loc["1990":"1998"] = None

    elif uhid == 703:  # Saldahna Bay, South Africa
        # severe timing issues
        hsl.loc["1982-08-20 08":"1982-08-24 05"] = None
        hsl.loc["1987-12-07 12":"1987-12-21 07"] = None
        hsl.loc["1990-08-07 09":"1990-08-10 14"] = None
        hsl.loc["1990-08-14 08":"1990-08-17 06"] = None
        hsl.loc["1990-11-26 09":"1990-11-27 02"] = None
        hsl.loc["1990-11-28 09":"1990-11-29 02"] = None
        hsl.loc["1993-07-08 12":"1993-07-12 12"] = None
        hsl.loc["1995-05-31 21":"1995-06-19 07"] = None
        hsl.loc["1996-05-23 13":"1996-05-27 06"] = None
        # gaps and probable loss of vertical control
        hsl.loc["1982-10-11 13":"1984-06-06 06"] = None
        # repair loss of vertical control
        hsl = hsl - estimate_step(hsl, t_step="2017-01-01 00")

    elif uhid == 704:  # Cape Town, South Africa
        # severe timing issues
        hsl.loc["1981-02-10 14":"1981-02-15 01"] = None
        hsl.loc["1981-03-20 10":"1981-03-23 21"] = None
        hsl.loc["1982-03-15 08":"1982-03-19 07"] = None
        # spikes
        hsl.loc["1982-11-12 23":"1982-11-13 07"] = None
        hsl.loc["1982-11-14 23":"1982-11-15 07"] = None
        # severe timing issues and loss of vertical control
        hsl.loc["2014-04-30 23":"2014-06-01 04"] = None
        # repair loss of vertical control
        hsl = hsl - estimate_step(hsl, t_step=["2015-05-25 00", "2018-01-01 00"])

    elif uhid == 712:  # Recife, USCGS, Brazil
        # severe timing issues
        hsl.loc["1968-04-01 10":"1968-04-30 00"] = None

    elif uhid == 779:  # Ciudad del Carmen, Mexico
        # loss of vertical control
        hsl.loc["1991-10-11 08":] = None
        hsl = hsl - estimate_step(
            hsl,
            t_step=[
                "1977-11-01 00",
                "1980-05-01 00",
                "1981-11-11 22",
                "1984-01-01 00",
                "1990-10-01 00",
            ],
        )

    elif uhid == 780:  # Puerto Cortes, Honduras
        # data is blocky and strange; does not look physical
        hsl.loc[:] = None

    elif uhid == 802:  # Maloy, Norway
        # short period with suspect vertical control at beginning of record
        hsl.loc[:"1986-08-12 19"] = None
        # flatline
        hsl.loc["2020-03-23 15":"2020-03-24 06"] = None

    elif uhid == 804:  # Tregde, Norway
        # strange, block data; appears to be in increments of 10 cm
        hsl.loc[:] = None

    elif uhid == 805:  # Vardo, Norway
        # flatline
        hsl.loc["2019-02-11 16":"2019-02-12 06"] = None
        # level across long gap
        hsl = hsl - estimate_step(hsl, t_step="1980-01-01 00")

    elif uhid == 808:  # Thule (Pituffik), Denmark
        # severe timing issues
        hsl.loc["2022-07-05 09":"2022-07-05 11"] = None
        hsl.loc["2023-10-05 12":"2023-10-06 08"] = None
        hsl.loc["2023-11-17 11":"2023-11-21 13"] = None
        # spike
        hsl.loc["2023-11-28 12"] = None
        # loss of vertical control
        hsl.loc["2023-08-19 10":"2023-08-19 15"] = None
        hsl = hsl - estimate_step(hsl, t_step="2023-08-19 12")

    elif uhid == 823:  # Nylesund, Norway
        # spikes
        hsl.loc[
            [
                "1994-01-01 00",
                "1995-01-01 00",
                "1997-01-01 00",
                "1998-01-01 00",
                "1999-01-01 00",
                "2000-01-01 00",
                "2001-01-01 00",
                "2002-01-01 00",
                "2003-01-01 00",
            ]
        ] = None
        # early record struggles with vertical control
        hsl.loc[:"1991-06-30"] = None

    else:
        qc = False

    return hsl, qc
