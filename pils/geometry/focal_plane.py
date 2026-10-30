"""A telescope's focal plane: its wafers, where they lie under a boresight rotation.

Focal-plane points are (xi, eta) in radians at boresight rotation 0 (the
nominal frame); turned by the boresight rotation, they are in the sky-aligned
frame a drone's path is expressed in (:mod:`pils.geometry.crossings`). An SO
wafer is the flat-top hexagon of :data:`DEFAULT_WAFER_RADIUS_RAD` around its
centre; a wafer read from a detector table carries its own outline and pixels.
The quaternions are so3g's, imported only when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEG = np.pi / 180.0


@dataclass(frozen=True)
class Wafer:
    """A wafer: its nominal, un-rotated centre in focal-plane (xi, eta) radians.

    An SO wafer is the flat-top hexagon of ``wafer_radius_rad`` around that
    centre. A wafer read from a detector table carries its own ``outline``
    instead -- the convex hull of its pixels, as (dxi, deta) offsets from the
    centre, counter-clockwise -- and the ``pixels`` themselves, absolute
    (xi, eta) at boresight 0.
    """

    name: str
    xi: float
    eta: float
    outline: tuple[tuple[float, float], ...] = ()
    pixels: np.ndarray | None = field(
        default=None, compare=False, hash=False, repr=False
    )


# The SO SATs' seven wafers (ws0 at the centre, the others around it)
DEFAULT_WAFERS: tuple[Wafer, ...] = (
    Wafer("ws0", 0.000, 0.000),
    Wafer("ws1", 0.000, -0.221),
    Wafer("ws2", -0.191, -0.111),
    Wafer("ws3", -0.191, 0.111),
    Wafer("ws4", 0.000, 0.221),
    Wafer("ws5", 0.191, 0.111),
    Wafer("ws6", 0.191, -0.111),
)

# Circumradius of the illuminated hexagon (half-width at boresight 0)
DEFAULT_WAFER_RADIUS_RAD = 0.105
# Centre-to-centre distance of neighbouring wafers
WAFER_PITCH_RAD = 0.221

_HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))  # flat-top hexagon


def _wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def wafer_quat(wafer: Wafer, boresight_deg: float, dxi: float = 0.0):
    """Return the focal-plane quaternion of a wafer centre rotated by the boresight angle.

    Optionally shifted by ``dxi`` along xi (a scan's edges). Rotate, decompose
    to (xi, eta, gamma), force gamma = 0, rebuild: the gamma reset is
    deliberate, as in SO's pointing notebook.
    """
    import so3g.proj.quat as quat

    q = quat.euler(2, -boresight_deg * DEG) * quat.rotation_xieta(wafer.xi, wafer.eta)
    xi, eta, _ = quat.decompose_xieta(q)
    return quat.rotation_xieta(xi + dxi, eta, 0.0)


def rotated_xieta(xi: float, eta: float, boresight_deg: float) -> tuple[float, float]:
    """Return a focal-plane point after the boresight rotation (sky-aligned frame)."""
    import so3g.proj.quat as quat

    q = quat.euler(2, -boresight_deg * DEG) * quat.rotation_xieta(xi, eta)
    x, e, _ = quat.decompose_xieta(q)
    return float(x), float(e)


def wafer_polygon(
    wafer: Wafer, boresight_deg: float, radius_rad: float = DEFAULT_WAFER_RADIUS_RAD
) -> np.ndarray:
    """Return the (n, 2) vertices of the wafer's area in the rotated frame.

    Its own outline if it has one, else the hexagon of ``radius_rad``.
    """
    offsets = wafer.outline or [
        (radius_rad * np.cos(a), radius_rad * np.sin(a)) for a in _HEX_ANGLES
    ]
    return np.array(
        [
            rotated_xieta(wafer.xi + dx, wafer.eta + dy, boresight_deg)
            for dx, dy in offsets
        ]
    )


def sky_of(az_deg: float, el_deg: float, xi: float, eta: float) -> tuple[float, float]:
    """Return the sky direction (az, el) [deg] of focal-plane point (xi, eta) [rad] with the boresight at (az, el)."""
    import so3g.proj.quat as quat

    q = quat.rotation_lonlat(-az_deg * DEG, el_deg * DEG, 0.0) * quat.rotation_xieta(
        xi, eta, 0.0
    )
    lon, lat, _ = quat.decompose_lonlat(q)
    az = float(-np.asarray(lon) / DEG)
    return az_deg + _wrap_deg(az - az_deg), float(np.asarray(lat) / DEG)
