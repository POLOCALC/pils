"""The drone in a telescope's focal plane, and the detectors it crosses.

The drone's path is expressed in the telescope's focal plane at every instant
(exact so3g quaternions); a detector is crossed each time the path enters its
beam, a circle of a given radius around the detector's centre. The track may be
a planned flight's (GOSE's simulation) or a flown one's: only the sampling is a
choice, fine enough that the drone cannot jump across a beam between two samples.

Angles are in degrees; (xi, eta) are focal-plane coordinates in the
sky-aligned frame (boresight rotation 0) unless said otherwise.
"""

from __future__ import annotations

import numpy as np

DEG = np.pi / 180


def relative_path(
    tel_az: np.ndarray, tel_el: np.ndarray, dr_az: np.ndarray, dr_el: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Place the drone in the boresight's focal plane: (xi, eta) in degrees, sky-aligned.

    Parameters
    ----------
    tel_az, tel_el : array
        The telescope's boresight (true azimuth, elevation) over time, degrees.
    dr_az, dr_el : array
        The drone as the telescope sees it, at the same times, degrees.
    """
    import so3g.proj.quat as quat

    q_bore = quat.rotation_lonlat(-np.asarray(tel_az) * DEG, np.asarray(tel_el) * DEG)
    q_drone = quat.rotation_lonlat(-np.asarray(dr_az) * DEG, np.asarray(dr_el) * DEG)
    xi, eta, _ = quat.decompose_xieta(~q_bore * q_drone)
    return np.rad2deg(np.asarray(xi)), np.rad2deg(np.asarray(eta))


def rotate(xi_deg: np.ndarray, eta_deg: np.ndarray, boresight_deg: float) -> np.ndarray:
    """Turn focal-plane points (degrees) into the sky-aligned frame of a boresight rotation: (n, 2) degrees."""
    import so3g.proj.quat as quat

    q = quat.euler(2, -boresight_deg * DEG) * quat.rotation_xieta(
        np.asarray(xi_deg) * DEG, np.asarray(eta_deg) * DEG
    )
    x, e, _ = quat.decompose_xieta(q)
    return np.column_stack([np.rad2deg(np.asarray(x)), np.rad2deg(np.asarray(e))])


def crossings(
    path_xi: np.ndarray,
    path_eta: np.ndarray,
    pixels: np.ndarray,
    radius_deg: float,
    chunk: int = 256,
) -> np.ndarray:
    """Count how many times a path enters each pixel's circle of ``radius_deg`` (starting inside counts once)."""
    out = np.zeros(len(pixels), int)
    near = (
        (pixels[:, 0] >= path_xi.min() - radius_deg)
        & (pixels[:, 0] <= path_xi.max() + radius_deg)
        & (pixels[:, 1] >= path_eta.min() - radius_deg)
        & (pixels[:, 1] <= path_eta.max() + radius_deg)
    )
    idx = np.flatnonzero(near)
    for k in range(0, len(idx), chunk):
        sel = idx[k : k + chunk]
        inside = (
            np.hypot(
                pixels[sel, 0, None] - path_xi[None],
                pixels[sel, 1, None] - path_eta[None],
            )
            < radius_deg
        )
        out[sel] = inside[:, 0] + np.count_nonzero(
            inside[:, 1:] & ~inside[:, :-1], axis=1
        )
    return out


def crossing_times(
    t: np.ndarray,
    path_xi: np.ndarray,
    path_eta: np.ndarray,
    pixels: np.ndarray,
    radius_deg: float,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """List every pass of the path through a pixel's circle of ``radius_deg``.

    Returns (pixel index, seconds inside) per crossing, the same crossings
    :func:`crossings` counts. A pass straight through the middle lasts
    2 x radius / speed; a grazing one is shorter.
    """
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    idx_out, dur_out = [], []
    near = (
        (pixels[:, 0] >= path_xi.min() - radius_deg)
        & (pixels[:, 0] <= path_xi.max() + radius_deg)
        & (pixels[:, 1] >= path_eta.min() - radius_deg)
        & (pixels[:, 1] <= path_eta.max() + radius_deg)
    )
    idx = np.flatnonzero(near)
    for k in range(0, len(idx), chunk):
        sel = idx[k : k + chunk]
        inside = (
            np.hypot(
                pixels[sel, 0, None] - path_xi[None],
                pixels[sel, 1, None] - path_eta[None],
            )
            < radius_deg
        )
        edges = np.diff(np.pad(inside.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        rows, starts = np.nonzero(edges == 1)
        _, ends = np.nonzero(edges == -1)  # same row order: every entry has its exit
        idx_out.append(sel[rows])
        dur_out.append((ends - starts) * dt)
    if not idx_out:
        return np.zeros(0, int), np.zeros(0)
    return np.concatenate(idx_out), np.concatenate(dur_out)


def demod_limit_s(
    hwp_hz: float | None, beam_fraction: float = 0.5, kept: float = 0.95
) -> float | None:
    """Return the shortest crossing straight through a beam that a HWP demodulation still follows.

    Below it, the low-pass at 0.95 x the rotation (sotodlib's demod_tod) keeps
    less than ``kept`` of the peak. A central pass through the circle of
    ``beam_fraction`` x FWHM lasts 2 x beam_fraction x FWHM / v and its Gaussian
    has sigma_t = FWHM / 2.355 / v; the peak kept is
    erf(sqrt(2) pi sigma_t f_c). None without a HWP.
    """
    from scipy.special import erfinv

    if not hwp_hz:
        return None
    sigma_t = erfinv(kept) / (np.sqrt(2) * np.pi * 0.95 * float(hwp_hz))
    return float(2 * beam_fraction * 2 * np.sqrt(2 * np.log(2)) * sigma_t)
