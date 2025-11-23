import numpy as np

import matplotlib.pyplot as plt
from cartopy import crs as ccrs


def generate_random_spherical_points(n):
    """Generate n random points in spherical coordinates (latitude and longitude) including polar regions."""
    # Uniform distribution of longitude
    longitude = np.random.uniform(-180, 180, n)

    # Sine distribution of latitude for uniform surface area coverage
    latitude = np.arcsin(np.random.uniform(-1, 1, n)) * 180 / np.pi

    return latitude, longitude


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points on the earth."""
    R = 6371  # Radius of Earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(
        np.radians(lat2)
    ) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if (
        (q.x <= max(p.x, r.x))
        and (q.x >= min(p.x, r.x))
        and (q.y <= max(p.y, r.y))
        and (q.y >= min(p.y, r.y))
    ):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:
        # Clockwise orientation
        return 1
    elif val < 0:
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def does_intersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False


def find_nearest_neighbors(lat, lon):
    """Find a non-intersecting path connecting nearest neighbors."""
    n = len(lat)
    path = []
    visited = np.zeros(n, dtype=bool)
    current = np.random.randint(n)
    path.append(current)
    visited[current] = True

    for _ in range(1, n):
        neighbors = sorted(
            (
                (i, haversine_distance(lat[current], lon[current], lat[i], lon[i]))
                for i in range(n)
                if not visited[i] and (i != current)
            ),
            key=lambda x: x[1],
        )

        for neighbor, _ in neighbors:
            new_segment = [
                Point(lon[current], lat[current]),
                Point(lon[neighbor], lat[neighbor]),
            ]
            intersections = []
            for i in range(len(path) - 2):
                intersections.append(
                    does_intersect(
                        *new_segment,
                        *[
                            Point(lon[path[i]], lat[path[i]]),
                            Point(lon[path[i + 1]], lat[path[i + 1]]),
                        ],
                    )
                )
            if not any(
                does_intersect(
                    *new_segment,
                    *[
                        Point(lon[path[i]], lat[path[i]]),
                        Point(lon[path[i + 1]], lat[path[i + 1]]),
                    ],
                )
                for i in range(len(path) - 2)
            ):
                path.append(neighbor)
                current = neighbor
                visited[current] = True
                break
            else:
                # No valid neighbor found, return None
                return None

    # check if the closing segment from last to first point intersects other segments
    for i in range(1, n):
        closing_segment = [
            Point(lon[path[-1]], lat[path[-1]]),
            Point(lon[path[0]], lat[path[0]]),
        ]
        if any(
            does_intersect(
                *closing_segment,
                *[
                    Point(lon[path[i]], lat[path[i]]),
                    Point(lon[path[i + 1]], lat[path[i + 1]]),
                ],
            )
            for i in range(1, len(path) - 2)
        ):
            return None

    # return path

    plon = [lon[m] for m in path]
    plat = [lat[n] for n in path]

    return plat, plon


def plot_path_on_map(plat, plon):
    """Plot the path on a 2D map with correct wrapping around longitude."""
    plt.figure()

    for i in range(1, len(plat)):
        plt.plot([plon[i - 1], plon[i]], [plat[i - 1], plat[i]], "-o", color="r")

    for i in range(0, len(plat)):
        plt.text(plon[i], plat[i], str(i))

    plt.xlim(-360, 360)  # Adjusted to show the wrapping
    plt.ylim(-90, 90)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Random Path on 2D Map")
    plt.grid(True)
    plt.show()


def tg_in_polygon(p, a):
    return np.array(
        [
            p.contains_point(p.get_transform().transform((tg.lon, tg.lat)))
            for _, tg in a.iterrows()
        ]
    )


def make_polygon_find_tg_inside(
    ax, px, py, a, col=None, show_polygon=True, zshift=0, labels=None
):
    col = ["b", "m", "k"] if col is None else col

    # make geodetic polygon and get tg locations inside it
    (p,) = ax.fill(
        px,
        py,
        color=col[0],
        alpha=1.0,
        lw=1,
        zorder=-100 + zshift,
        transform=ccrs.Geodetic(),
        label=labels[0] if labels is not None else None,
    )
    ax.plot(
        px + [px[0]],
        py + [py[0]],
        col[0],
        ls="--",
        lw=1,
        zorder=-50 + zshift,
        transform=ccrs.Geodetic(),
    )
    tg_in = tg_in_polygon(p, a)

    # if all or none of the tg locations are in the patch, try reversing the polygon
    if sum(tg_in) in [0, len(tg_in)]:
        p.remove()
        px = px[::-1]
        py = py[::-1]
        (p,) = ax.fill(
            px,
            py,
            color=col[0],
            alpha=1.0,
            lw=0,
            zorder=-100 + zshift,
            transform=ccrs.Geodetic(),
        )
        tg_in = tg_in_polygon(p, a)

    # if tg locations are divided between in and out, then show in and out
    if sum(tg_in) not in [0, len(tg_in)]:
        if show_polygon:
            a.loc[tg_in].plot(
                ax=ax,
                color=col[2],
                markersize=20,
                edgecolor="white",
                lw=0.5,
                transform=ccrs.Geodetic(),
                zorder=zshift,
                label=labels[2] if labels is not None else None,
            )
            a.loc[~tg_in].plot(
                ax=ax,
                color=col[1],
                markersize=20,
                edgecolor="white",
                lw=0.5,
                transform=ccrs.Geodetic(),
                zorder=-10 + zshift,
                label=labels[1] if labels is not None else None,
            )
        if not show_polygon:
            p.remove()
        return tg_in
    # otherwise, remove the patch
    else:
        p.remove()
        return None
