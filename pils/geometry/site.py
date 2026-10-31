"""Positions on the site: the telescopes, the drone, and how one sees the other.

Positions are (E, N, U) metres in a flat frame around a reference site (the LAT,
:data:`ENU_REFERENCE_SITE`) -- the frame a planner (GOSE) places its drone fields
in, good to centimetres over the site. The sites are so3g's (``so3g.proj.coords.SITES``),
imported only when needed. Azimuths are clockwise from north, in degrees.
"""

from __future__ import annotations

import math

import numpy as np

DEG = np.pi / 180.0
EARTH_RADIUS_M = 6_371_000.0
ENU_REFERENCE_SITE = "so_lat"


# so3g's own sites: a location given for one of them is refused, not silently moved
SO_SITES = frozenset(("act", "so_lat", "so_sat1", "so_sat2", "so_sat3", "so", "_default"))


def register_site(key: str, lon: float, lat: float, elev: float) -> None:
    """Add a site that is not in so3g's registry (e.g. CLASS's) under ``key``, at ``lon``, ``lat``
    (degrees east, north) and ``elev`` (metres): everything that places a telescope then finds it like
    an SO site. Moving one of so3g's own sites is refused."""
    import so3g.proj.coords as coords

    new = coords.EarthlySite(float(lon), float(lat), float(elev))
    old = coords.SITES.get(key)
    if old is not None and key in SO_SITES and (old.lon, old.lat, old.elev) != (new.lon, new.lat, new.elev):
        raise ValueError(f"site {key!r} is an so3g site; give a new key for another location")
    coords.SITES[key] = new


def site_enu(site_key, ref_key: str = ENU_REFERENCE_SITE) -> np.ndarray:
    """Return (E, N, U) of a site in metres relative to the reference site.

    ``site_key`` is a so3g SITES key, or already an (E, N, U) triple which is
    passed through (for telescopes not in the registry).
    """
    if not isinstance(site_key, str):
        return np.asarray(site_key, float)
    from so3g import proj

    s, r = proj.coords.SITES[site_key], proj.coords.SITES[ref_key]
    dlat, dlon = math.radians(s.lat - r.lat), math.radians(s.lon - r.lon)
    return np.array(
        [
            dlon * EARTH_RADIUS_M * math.cos(math.radians(r.lat)),
            dlat * EARTH_RADIUS_M,
            s.elev - r.elev,
        ]
    )


def enu_to_geodetic(
    enu: np.ndarray, ref_key: str = ENU_REFERENCE_SITE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return latitude, longitude (deg) and height (m) of (n, 3) ENU points: the inverse of :func:`site_enu`."""
    from so3g import proj

    r = proj.coords.SITES[ref_key]
    enu = np.atleast_2d(enu)
    lat = r.lat + np.degrees(enu[:, 1] / EARTH_RADIUS_M)
    lon = r.lon + np.degrees(
        enu[:, 0] / (EARTH_RADIUS_M * math.cos(math.radians(r.lat)))
    )
    return lat, lon, r.elev + enu[:, 2]


def azel_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    """Return the ENU unit vector of the direction (az, el) in degrees."""
    az, el = az_deg * DEG, el_deg * DEG
    return np.array(
        [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)]
    )


def enu_to_azel(vec: np.ndarray) -> tuple[float, float, float]:
    """Return (az, el) in degrees and range in metres of an ENU vector."""
    e, n, u = vec
    az = math.degrees(math.atan2(e, n)) % 360.0
    el = math.degrees(math.atan2(u, math.hypot(e, n)))
    return az, el, float(np.linalg.norm(vec))


def direction_from(site_key, point_enu: np.ndarray) -> tuple[float, float, float]:
    """Return (az, el, range) of an ENU point as seen from a site (key or ENU triple)."""
    return enu_to_azel(np.asarray(point_enu, float) - site_enu(site_key))


def seen_from(enu: np.ndarray, site: np.ndarray) -> dict[str, np.ndarray]:
    """See (n, 3) ENU points from a site's ENU position: pils' azel columns.

    Returns ``az_deg``, ``el_deg``, ``srange_m`` and the offset ``e_m``, ``n_m``, ``u_m``.
    """
    d = np.atleast_2d(enu) - np.asarray(site, float)
    horizontal = np.hypot(d[:, 0], d[:, 1])
    return {
        "az_deg": np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0,
        "el_deg": np.degrees(np.arctan2(d[:, 2], horizontal)),
        "srange_m": np.linalg.norm(d, axis=1),
        "e_m": d[:, 0],
        "n_m": d[:, 1],
        "u_m": d[:, 2],
    }


def enu_from_reference(
    site_key, az_deg: float, el_deg: float, height_m: float
) -> np.ndarray:
    """Place a point on a telescope's line of sight at (az, el), ``height_m`` above that telescope."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    out = height_m / max(math.tan(e), 1e-6)  # how far out that height puts it
    return site_enu(site_key) + np.array(
        [out * math.sin(a), out * math.cos(a), height_m]
    )


def enu_from_distance(
    site_key, az_deg: float, el_deg: float, distance_m: float
) -> np.ndarray:
    """Place a point on a telescope's line of sight at (az, el), ``distance_m`` away."""
    return site_enu(site_key) + distance_m * azel_to_unit(az_deg, el_deg)


def circular_mean_deg(az_deg) -> float:
    """Return the mean of azimuths [deg] on the circle (an arithmetic mean breaks across 0/360)."""
    a = np.radians(np.asarray(az_deg, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360.0)
