"""A telescope's detectors: their positions on its wafers, their bands and beams.

The SO SATs' detector layout is SO's hardware model (sotodlib's
``sim_nominal`` and ``sim_detectors_toast``): a SAT wafer carries 430 dichroic
pixels, the same positions for both bands of its wafer type -- what differs
between bands is the beam (30' at 90 GHz, 17' at 150, 11' at 230, 9' at 290).
The layouts are shipped (``data/focal_planes/<wafer type>.npz``, built with
sotodlib 0.6.24): sotodlib and TOAST are needed only to build them again. Each
wafer's pattern is placed on the wafer centres given (the pointing team's,
:data:`pils.geometry.focal_plane.DEFAULT_WAFERS`); a wafer read from a detector
table (the LAT, CLASS) carries its own pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from pils.geometry.crossings import rotate
from pils.geometry.focal_plane import Wafer

# The layouts shipped, one per wafer type
SHIPPED = Path(__file__).parent / "data" / "focal_planes"

# wafer type -> the telescope of the nominal hardware model that carries it
TYPE_TELESCOPE = {"SAT_LF": "SAT4", "SAT_MF": "SAT1", "SAT_UHF": "SAT3"}

# what the drone team calls a source -> the band name of the hardware model
SOURCE_BAND = {
    30: "SAT_f030",
    40: "SAT_f040",
    90: "SAT_f090",
    150: "SAT_f150",
    220: "SAT_f230",
    230: "SAT_f230",
    280: "SAT_f290",
    290: "SAT_f290",
}

# which bands a wafer of each type carries -- the same information the hardware
# model gives, without needing sotodlib to answer "can this SAT see that source?"
TYPE_BANDS = {
    "SAT_LF": ("SAT_f030", "SAT_f040"),
    "SAT_MF": ("SAT_f090", "SAT_f150"),
    "SAT_UHF": ("SAT_f230", "SAT_f290"),
    # not SO: CLASS's W-band cameras. The pixels come from the telescope's detector table,
    # the beam from its beam_fwhm_arcmin
    "CLASS_W": ("SAT_f090",),
    # the SO LAT's mid-frequency wafers, from a detector table of its own (generated from
    # sotodlib's hardware model); the band names are the drone team's, as for CLASS
    "LAT_MF": ("SAT_f090", "SAT_f150"),
}
WAFER_TYPES = tuple(TYPE_BANDS)


def sees(wafer_type: str, source: float | str) -> bool:
    """Tell whether a telescope carrying ``wafer_type`` detectors has a band for this drone source."""
    return band_of_source(source) in TYPE_BANDS[wafer_type]


def band_of_source(source: float | str) -> str:
    """Name the hardware-model band of a source: ``90`` / ``"90"`` / ``"SAT_f090"`` -> ``SAT_f090``."""
    s = str(source).strip()
    if s.startswith("SAT_f"):
        return s
    key = int(round(float(s)))
    if key not in SOURCE_BAND:
        raise ValueError(
            f"unknown drone source {source!r}; known: {sorted(SOURCE_BAND)}"
        )
    return SOURCE_BAND[key]


def source_label(band: str) -> str:
    """Name a band in GHz: ``SAT_f090`` -> ``90 GHz``."""
    return f"{int(band.split('_f')[-1])} GHz"


@dataclass(frozen=True)
class FocalPlaneModel:
    """Nominal pixel positions of one wafer type, in the wafer's own frame."""

    wafer_type: str
    xi: np.ndarray  # (n_pixels,) rad, relative to the wafer centre
    eta: np.ndarray
    bands: tuple[str, ...]
    fwhm_arcmin: Mapping[str, float]

    @property
    def n_pixels(self) -> int:
        """Return the number of pixels of a wafer."""
        return len(self.xi)


@lru_cache(maxsize=4)
def nominal_focal_plane(wafer_type: str = "SAT_MF") -> FocalPlaneModel:
    """Return the nominal pixel pattern of one wafer of ``wafer_type``.

    The shipped layout if there is one; else built from sotodlib's hardware
    model (which imports TOAST: a few seconds, and only from the main thread).
    """
    if wafer_type not in TYPE_TELESCOPE:
        raise ValueError(
            f"unknown wafer type {wafer_type!r}; known: {sorted(TYPE_TELESCOPE)}"
        )
    shipped = SHIPPED / f"{wafer_type}.npz"
    if shipped.is_file():
        with np.load(shipped) as f:
            bands = tuple(str(b) for b in f["bands"])
            return FocalPlaneModel(
                wafer_type,
                f["xi"],
                f["eta"],
                bands,
                {b: float(w) for b, w in zip(bands, f["fwhm"], strict=True)},
            )
    return build_nominal_focal_plane(wafer_type)


def build_nominal_focal_plane(wafer_type: str) -> FocalPlaneModel:
    """Build the pixel pattern of one wafer of ``wafer_type`` from sotodlib's hardware model."""
    import so3g
    from so3g.proj import quat
    from sotodlib.sim_hardware import sim_detectors_toast, sim_nominal

    hw = sim_nominal()
    sim_detectors_toast(hw, TYPE_TELESCOPE[wafer_type])
    dets = hw.data["detectors"]
    slot = next(iter(dets.values()))[
        "wafer_slot"
    ]  # one wafer is enough: they are identical
    one = {k: d for k, d in dets.items() if d["wafer_slot"] == slot}
    # the hardware model stores TOAST quaternions (x, y, z, w); so3g wants (w, x, y, z)
    q = np.array([[d["quat"][3], *d["quat"][:3]] for d in one.values()])
    xi, eta, _ = quat.decompose_xieta(so3g.proj.quat.G3VectorQuat(q))
    xi, eta = np.asarray(xi), np.asarray(eta)
    pix = np.array([d["pixel"] for d in one.values()])
    bands = tuple(sorted({str(d["band"]) for d in one.values()}))
    fwhm = {str(d["band"]): float(d["fwhm"]) for d in one.values()}
    # one position per pixel (the two polarisations and the two bands share it)
    _, first = np.unique(pix, return_index=True)
    xi, eta = xi[np.sort(first)], eta[np.sort(first)]
    return FocalPlaneModel(wafer_type, xi - xi.mean(), eta - eta.mean(), bands, fwhm)


def focal_plane_model(tel) -> FocalPlaneModel:
    """Return the detector model of one telescope (its ``wafers``, ``wafer_type``, ``beam_fwhm_arcmin``).

    The SO hardware model for an SO wafer type; for a telescope with its own
    detector table (CLASS, the LAT) the bands of its type and its own beam --
    its pixels are on its wafers (``Wafer.pixels``), so the model's per-wafer
    pattern is empty.
    """
    if not tel.wafers:
        return nominal_focal_plane(tel.wafer_type)
    if tel.wafer_type not in TYPE_BANDS:
        raise ValueError(
            f"{tel.name}: unknown wafer type {tel.wafer_type!r}; known: {sorted(TYPE_BANDS)}"
        )
    bands = TYPE_BANDS[tel.wafer_type]
    return FocalPlaneModel(
        tel.wafer_type,
        np.zeros(0),
        np.zeros(0),
        bands,
        {b: float(tel.beam_fwhm_arcmin) for b in bands},
    )


def pixel_positions(
    wafers: Sequence[Wafer], wafer_type: str = "SAT_MF"
) -> dict[str, np.ndarray]:
    """Return the nominal pixel centres of a whole focal plane, per wafer.

    (n_pixels, 2) arrays of (xi, eta) in radians at boresight 0: a wafer's own
    pixels when it was read from a detector table, else the wafer pattern of the
    SO hardware model placed on the wafer's centre.
    """
    if all(w.pixels is not None for w in wafers):
        return {w.name: np.asarray(w.pixels, float) for w in wafers}
    fp = nominal_focal_plane(wafer_type)
    return {
        w.name: (
            np.asarray(w.pixels, float)
            if w.pixels is not None
            else np.column_stack([fp.xi + w.xi, fp.eta + w.eta])
        )
        for w in wafers
    }


def rotated_pixels(
    wafers: Sequence[Wafer], wafer_type: str, boresight_deg: float
) -> dict[str, np.ndarray]:
    """Return every pixel's (xi, eta) in degrees, in the sky-aligned frame of a boresight rotation."""
    return {
        name: rotate(np.rad2deg(p[:, 0]), np.rad2deg(p[:, 1]), boresight_deg)
        for name, p in pixel_positions(wafers, wafer_type).items()
    }


def rotated_eta(xi: np.ndarray, eta: np.ndarray, boresight_deg: float) -> np.ndarray:
    """Rotate whole arrays of focal-plane points by the boresight; return their eta only."""
    import so3g.proj.quat as quat

    q = quat.euler(2, -np.deg2rad(boresight_deg)) * quat.rotation_xieta(
        np.asarray(xi), np.asarray(eta)
    )
    return np.asarray(quat.decompose_xieta(q)[1])
