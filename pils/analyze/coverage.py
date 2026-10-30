"""The focal-plane coverage of a flown flight: the detectors the drone crossed.

The drone as a telescope saw it (the flight's ``azel/<TELESCOPE>``) and the
telescope's own pointing (``telescope/<TELESCOPE>``, its encoders) give the
drone's path in the focal plane; a detector is crossed each time the path comes
within a fraction of its beam (``pils.geometry.crossings``), the same rule a
planner applies to a simulated flight.

The result is plot-ready data, not a figure: the path (thinned) and every
pixel with its crossings.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pils import store
from pils.geometry.crossings import crossings, relative_path
from pils.geometry.detectors import SOURCE_BAND, FocalPlaneModel, rotated_pixels
from pils.geometry.focal_plane import Wafer

# The most samples of the path returned (it is drawn, not analysed)
MAX_POINTS = 4000


def thin(*arrays: np.ndarray, max_points: int = MAX_POINTS) -> list[list[float]]:
    """Every array at most ``max_points`` samples (evenly), rounded, as lists."""
    n = len(arrays[0])
    keep = (
        np.unique(np.linspace(0, n - 1, min(n, max_points)).astype(int))
        if n
        else np.arange(0)
    )
    return [np.round(np.asarray(a, float)[keep], 5).tolist() for a in arrays]


def latest_dataset(flight_dir: str | Path, dataset: str) -> Path | None:
    """The latest revision of one of a flight's datasets (``proc/<dataset>``), or None."""
    root = Path(flight_dir) / "proc" / dataset
    rev = store.latest_version(root) if root.is_dir() else None
    return root / rev if rev else None


def flight_coverage(
    flight_dir: str | Path,
    telescope: str,
    source_ghz: float,
    *,
    wafers: Sequence[Wafer],
    focal_plane: FocalPlaneModel,
    beam_fraction: float = 0.5,
    az_offset_deg: float = 0.0,
    start: float | None = None,
    end: float | None = None,
    max_points: int = MAX_POINTS,
) -> dict[str, Any]:
    """A flown flight in a telescope's focal plane: the detectors the drone crossed.

    Parameters
    ----------
    flight_dir : str or Path
        The flight's folder (its ``proc/azel/<TELESCOPE>`` and ``proc/telescope/<TELESCOPE>``).
    telescope : str
        The telescope's name, as the datasets have it (upper case).
    source_ghz : float
        The drone's source; its band gives the beam.
    wafers : sequence of Wafer
        The telescope's wafers.
    focal_plane : FocalPlaneModel
        Its detectors' model (``pils.geometry.detectors.focal_plane_model``): bands and beams.
    beam_fraction : float
        A detector is crossed within ``beam_fraction`` x its beam's FWHM.
    az_offset_deg : float
        The encoder's azimuth less the true one.
    start, end : float, optional
        Only [start, end] (UNIX s).
    max_points : int
        The most samples of the path returned.

    Returns
    -------
    dict
        The window, the boresight rotation (the median of the encoder's), the band and beam,
        the path (``track``: xi, eta in degrees) and every pixel (``pixels``: wafer, xi, eta,
        crossings), the number crossed and the revisions read.
    """
    name = telescope.upper()
    azel_dir = latest_dataset(flight_dir, f"azel/{name}")
    pointing_dir = latest_dataset(flight_dir, f"telescope/{name}")
    if azel_dir is None:
        raise ValueError(f"no azel/{name} in {flight_dir} (pils package)")
    if pointing_dir is None:
        raise ValueError(f"no telescope/{name} in {flight_dir} (pils telescope)")
    drone = store.read_columns(
        azel_dir, "azel_data", ["timestamp", "az_deg", "el_deg"]
    ).drop_nulls()
    pointing = store.read_columns(
        pointing_dir,
        "pointing",
        ["timestamp", "az_enc_deg", "el_enc_deg", "boresight_enc_deg", "flagged"],
    )
    pointing = pointing.filter(~pointing["flagged"]).drop_nulls()
    t = pointing["timestamp"].to_numpy()
    lo = max(start if start is not None else -np.inf, float(drone["timestamp"].min()))
    hi = min(end if end is not None else np.inf, float(drone["timestamp"].max()))
    keep = (t >= lo) & (t <= hi)
    if keep.sum() < 2:
        raise ValueError(f"{name} has no pointing while the drone was seen (or in the window)")
    t = t[keep]
    tel_az = pointing["az_enc_deg"].to_numpy()[keep] - az_offset_deg
    tel_el = pointing["el_enc_deg"].to_numpy()[keep]
    boresight = float(np.median(pointing["boresight_enc_deg"].to_numpy()[keep]))
    # The drone where the telescope's samples are (its azimuth unwrapped, for 0/360)
    dt = drone["timestamp"].to_numpy()
    dr_az = np.interp(t, dt, np.unwrap(np.radians(drone["az_deg"].to_numpy()))) * 180 / np.pi
    dr_el = np.interp(t, dt, drone["el_deg"].to_numpy())
    xi, eta = relative_path(tel_az, tel_el, dr_az, dr_el)

    band = SOURCE_BAND.get(int(source_ghz))
    fwhm = focal_plane.fwhm_arcmin.get(band) if band else None
    if fwhm is None:
        raise ValueError(
            f"no band of {name} for {source_ghz:g} GHz (its bands: {', '.join(focal_plane.bands)})"
        )
    radius = float(beam_fraction) * fwhm / 60
    pixels = rotated_pixels(wafers, focal_plane.wafer_type, boresight)
    names, xs, ys, counts = [], [], [], []
    for wafer, xy in pixels.items():
        n = crossings(np.asarray(xi), np.asarray(eta), xy, radius) if len(xy) else np.zeros(0, int)
        names += [wafer] * len(xy)
        xs += np.round(xy[:, 0], 4).tolist()
        ys += np.round(xy[:, 1], 4).tolist()
        counts += [int(c) for c in n]
    track_xi, track_eta = thin(xi, eta, max_points=max_points)
    return {
        "telescope": name,
        "window": [float(t[0]), float(t[-1])],
        "samples": int(len(t)),
        "boresight_deg": round(boresight, 3),
        "az_offset_deg": az_offset_deg,
        "source_ghz": float(source_ghz),
        "band": band,
        "beam_fwhm_arcmin": fwhm,
        "radius_deg": radius,
        "track": {"xi": track_xi, "eta": track_eta},
        "pixels": {"wafer": names, "xi": xs, "eta": ys, "crossings": counts},
        "crossed": int(sum(1 for c in counts if c)),
        "datasets": {"azel": azel_dir.name, "pointing": pointing_dir.name},
    }
