import numpy as np


def get_station_groups(analysis, islands, incorporated):

    groups = {
        "all": analysis.hdi > -1000,
        # global south and global north
        "globalN": analysis.hdi >= 0.8,
        "globalS": analysis.hdi < 0.8,
        # low latitude (< 30) and high latitude (> 30)
        "highLat": np.abs(analysis.lat) > 30,
        "lowLat": np.abs(analysis.lat) <= 30,
        # East and West Hemispheres
        "westHemi": (analysis.lon < 30) | (analysis.lon >= 180),
        "eastHemi": (analysis.lon >= 30) & (analysis.lon < 180),
        # islands and continents
        "island": analysis.index.isin(islands.uhid),
        "continent": ~analysis.index.isin(islands.uhid),
        # colonized or otherwise incorporated
        "incorporated": analysis.index.isin(incorporated.uhid),
        # other
        "30_good_years": analysis.n_good_years >= 30,
    }

    binary_group_names = list(groups.keys())

    groups = {
        **groups,
        **{
            "lowLat_island": groups["island"] & groups["lowLat"],
            "highLat_island": groups["island"] & groups["highLat"],
            "lowLat_continent": groups["continent"] & groups["lowLat"],
            "highLat_continent": groups["continent"] & groups["highLat"],
            # lat bands east and west
            "highLat_westHemi": groups["highLat"] & groups["westHemi"],
            "highLat_eastHemi": groups["highLat"] & ~groups["westHemi"],
            "lowLat_westHemi": groups["lowLat"] & groups["westHemi"],
            "lowLat_eastHemi": groups["lowLat"] & ~groups["westHemi"],
            # lat bands and domestic
            "highLat_domestic": groups["highLat"] & ~groups["incorporated"],
            "lowLat_domestic": groups["lowLat"] & ~groups["incorporated"],
            # islands in the global south
            "globalS_island": groups["island"] & groups["globalS"],
            "globalS_continent": groups["continent"] & groups["globalS"],
            # global north high- and low-latitude
            "globalN_highLat": groups["globalN"] & groups["highLat"],
            "globalN_lowLat": groups["globalN"] & groups["lowLat"],
            # global north incorporated
            "globalN_incorporated": groups["globalN"] & groups["incorporated"],
            # global north excluding incorporated (i.e., domestic)
            "globalN_domestic": groups["globalN"] & ~groups["incorporated"],
            # continental in the global north
            "globalN_island": groups["island"] & groups["globalN"],
            "globalN_continent": groups["globalN"] & groups["continent"],
            # global north west hemisphere
            "globalN_westHemi": groups["globalN"] & groups["westHemi"],
            "globalN_eastHemi": groups["globalN"] & ~groups["westHemi"],
        },
    }

    groups = {
        **groups,
        **{
            "globalN_westHemi_continent": groups["globalN_westHemi"]
            & groups["continent"],
            "globalN_westHemi_highLat": groups["globalN_westHemi"] & groups["highLat"],
            "globalN_northwestHemi_highLat": groups["globalN_westHemi"]
            & (analysis.lat >= 30),
            "globalN_eastHemi_continent": groups["globalN_eastHemi"]
            & groups["continent"],
            "globalN_eastHemi_highLat": groups["globalN_eastHemi"] & groups["highLat"],
            "globalN_northeastHemi_highLat": groups["globalN_eastHemi"]
            & (analysis.lat >= 30),
        },
    }

    groups = {
        **groups,
        **{
            "globalN_domestic_westHemi": groups["globalN_domestic"]
            & groups["westHemi"],
            "globalN_domestic_eastHemi": groups["globalN_domestic"]
            & ~groups["westHemi"],
            "globalN_domestic_highLat": groups["globalN_domestic"] & groups["highLat"],
            "globalN_domestic_lowLat": groups["globalN_domestic"] & groups["lowLat"],
        },
    }

    groups = {
        **groups,
        **{
            "globalN_domestic_NAEU": (
                groups["globalN_domestic_westHemi"] & (analysis.lat > 22)
            ),
            "globalN_domestic_NA": (
                groups["globalN_domestic"]
                & (analysis.lat > 22)
                & (analysis.lon < 310)
                & (analysis.lon >= 170)
            ),
            "globalN_domestic_WNA": (
                groups["globalN_domestic"]
                & (analysis.lat > 22)
                & (analysis.lon < 255)
                & (analysis.lon >= 170)
            ),
            "globalN_domestic_ENA": (
                groups["globalN_domestic"]
                & (analysis.lat > 22)
                & (analysis.lon < 310)
                & (analysis.lon >= 255)
            ),
            "globalN_domestic_EU": (
                groups["globalN_domestic"]
                & (analysis.lat > 22)
                & ((analysis.lon >= 310) | (analysis.lon < 35))
            ),
            "globalN_domestic_SHSA": (
                groups["globalN_domestic"]
                & (analysis.lat <= -10)
                & (analysis.lon < 330)
                & (analysis.lon >= 260)
            ),
            "globalN_domestic_AUNZ": (
                groups["globalN_domestic"] & (analysis.lat < -9) & (analysis.lon < 200)
            ),
            "globalN_domestic_ASIA": (
                groups["globalN_domestic"]
                & (analysis.lat > 20)
                & (analysis.lon < 160)
                & (analysis.lon > 100)
            ),
            "globalN_domestic_westHemiOther": (
                groups["globalN_domestic_westHemi"] & (analysis.lat <= 22)
            ),
            "globalS_continent_CASA": (
                groups["globalS"]
                & groups["continent"]
                & (analysis.lon >= 180)
                & (analysis.lon < 330)
            ),
            "globalS_continent_ASIA": (
                groups["globalS"]
                & groups["continent"]
                & (analysis.lon >= 60)
                & (analysis.lon < 180)
            ),
            "globalS_continent_AF": (
                groups["globalS"]
                & groups["continent"]
                & ((analysis.lon >= 330) | (analysis.lon < 60))
            ),
        },
    }

    groups = {
        **groups,
        **{
            "globalN_domestic_other": groups["globalN_domestic"]
            & ~groups["globalN_domestic_NAEU"]
            & ~groups["globalN_domestic_NA"]
            & ~groups["globalN_domestic_EU"]
            & ~groups["globalN_domestic_SHSA"]
            & ~groups["globalN_domestic_AUNZ"]
            & ~groups["globalN_domestic_ASIA"],
            "globalN_domestic_highLat_EU": groups["globalN_domestic_EU"]
            & groups["highLat"],
            "globalN_domestic_highLat_NA": groups["globalN_domestic_NA"]
            & groups["highLat"],
            "globalN_domestic_exEU": groups["globalN_domestic"]
            & ~groups["globalN_domestic_EU"],
        },
    }

    groups = {
        **groups,
        **{
            "island_indian": groups["island"]
            & (analysis.lon > 30)
            & (analysis.lon <= 120),
            "island_pacific": groups["island"]
            & (analysis.lon > 120)
            & (
                ((analysis.lon < 290) & (analysis.lat < 10))
                | ((analysis.lon < 260) & (analysis.lat >= 10))
            ),
            "island_atlantic": groups["island"]
            & (
                (analysis.lon < 30)
                | (
                    ((analysis.lon >= 290) & (analysis.lat < 10))
                    | ((analysis.lon >= 260) & (analysis.lat >= 10))
                )
            ),
        },
    }

    groups = {
        **groups,
        **{
            "globalS_island": groups["globalS"] & groups["island"],
            "globalS_continent": groups["globalS"] & groups["continent"],
            "globalS_island_indian": groups["globalS"] & groups["island_indian"],
            "globalS_island_pacific": groups["globalS"] & groups["island_pacific"],
            "globalS_island_atlantic": groups["globalS"] & groups["island_atlantic"],
            "globalS_indopacific": groups["globalS"]
            & (analysis.lon > 75)
            & (analysis.lon < 240),
            # & (np.abs(analysis.lat) < 30),
        },
    }

    groups = {
        **groups,
        **{
            "globalS_not_indopacific": groups["globalS"]
            & ~groups["globalS_indopacific"],
        },
    }

    # convert to numpy boolean to avoid issues with pandas indices
    groups = {
        g: groups[g].values if type(groups[g]) is not np.ndarray else groups[g]
        for g in groups
    }

    return groups, binary_group_names
