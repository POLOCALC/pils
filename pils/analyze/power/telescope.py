"""Telescope detector beams: solid angle and gain, from the telescopes config."""

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.integrate import trapezoid

from pils.config.telescopes import TELESCOPES_CONFIG

_ANGLE_TO_RAD = {"rad": 1.0, "deg": np.pi / 180, "arcmin": np.pi / (180 * 60)}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Telescopes config: a YAML file (argument or env) or the packaged one."""
    path = path or os.getenv("PILS_TELESCOPES_CONFIG")
    if path is None:
        return copy.deepcopy(TELESCOPES_CONFIG)
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def gaussian_solid_angle(fwhm_rad: float) -> float:
    """Ω = 2π σ² of a Gaussian beam (≈ 1.133 FWHM²), in sr."""
    return float(2 * np.pi * (fwhm_rad / np.sqrt(8 * np.log(2))) ** 2)


def profile_solid_angle(angle_rad: np.ndarray, power: np.ndarray) -> float:
    """Ω = 2π ∫ P(θ)/P_max sin θ dθ of a radial beam profile, in sr."""
    order = np.argsort(angle_rad)
    theta = np.asarray(angle_rad, float)[order]
    norm = np.asarray(power, float)[order] / np.max(power)
    return float(2 * np.pi * trapezoid(norm * np.sin(theta), theta))


def detector_gain_dbi(solid_angle_sr: float) -> float:
    """Forward gain 10 log10(4π / Ω)."""
    return float(10 * np.log10(4 * np.pi / solid_angle_sr))


@dataclass
class TelescopeBeam:
    """Detector beam of a telescope at the source frequency."""

    telescope: str
    wafer_slot: str
    band: str
    frequency_ghz: float
    solid_angle_sr: float
    method: str
    fwhm_arcmin: float | None = None
    source_file: str | None = None

    @property
    def gain_dbi(self) -> float:
        return detector_gain_dbi(self.solid_angle_sr)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "telescope": self.telescope,
            "wafer_slot": self.wafer_slot,
            "band": self.band,
            "frequency_ghz": self.frequency_ghz,
            "solid_angle_sr": self.solid_angle_sr,
            "detector_gain_dbi": self.gain_dbi,
            "method": self.method,
            "fwhm_arcmin": self.fwhm_arcmin,
            "source_file": self.source_file,
        }


def telescope_beam(
    telescope: str,
    wafer_slot: str,
    band: str,
    frequency_ghz: float,
    config: dict[str, Any] | None = None,
) -> TelescopeBeam:
    """Detector beam of ``telescope``/``wafer_slot`` for a source band and frequency.

    Raises
    ------
    ValueError
        If the telescope or band is not in the config.
    FileNotFoundError
        If a beam file is missing.
    """
    config = config or load_config()
    telescopes = config.get("telescopes", {})
    key = telescope.lower()
    if key not in telescopes:
        raise ValueError(f"Unknown telescope '{telescope}' (have {sorted(telescopes)})")
    if band not in telescopes[key]:
        raise ValueError(
            f"No {band} beam for {telescope} (have {sorted(telescopes[key])})"
        )
    entry = dict(telescopes[key][band])
    entry |= entry.pop("wafer_slots", {}).get(wafer_slot, {})

    beam_dir = Path(
        os.getenv("PILS_TELESCOPE_BEAMS_PATH") or config.get("beam_dir") or "."
    )
    names = {"telescope": key, "wafer_slot": wafer_slot, "freq": int(frequency_ghz)}

    def beam_file(template: str) -> Path:
        path = beam_dir / template.format(**names)
        if not path.is_file():
            raise FileNotFoundError(
                f"Beam file not found: {path} (set PILS_TELESCOPE_BEAMS_PATH)"
            )
        return path

    common = {
        "telescope": key,
        "wafer_slot": wafer_slot,
        "band": band,
        "frequency_ghz": float(frequency_ghz),
    }
    if "fwhm_arcmin" in entry:
        fwhm = float(entry["fwhm_arcmin"])
        omega = gaussian_solid_angle(fwhm * _ANGLE_TO_RAD["arcmin"])
        return TelescopeBeam(
            **common, solid_angle_sr=omega, method="gaussian", fwhm_arcmin=fwhm
        )

    averaged = [float(f) for f in entry.get("band_average_frequencies_ghz", [])]
    if frequency_ghz in averaged:
        path = beam_file(entry["fwhm_table"])
        fwhm = float(pl.read_csv(path)["fwhm"].mean())  # type: ignore[arg-type]
        omega = gaussian_solid_angle(fwhm * _ANGLE_TO_RAD["arcmin"])
        return TelescopeBeam(
            **common,
            solid_angle_sr=omega,
            method="gaussian, band-average FWHM",
            fwhm_arcmin=fwhm,
            source_file=str(path),
        )

    path = beam_file(entry["profile"])
    angle, power = np.loadtxt(path, unpack=True)
    unit = entry.get("profile_angle_unit", "rad")
    omega = profile_solid_angle(angle * _ANGLE_TO_RAD[unit], power)
    fwhm = None
    if "fwhm_table" in entry:
        try:
            table = pl.read_csv(beam_file(entry["fwhm_table"]))
            match = table.filter(pl.col("freq") == frequency_ghz)
            fwhm = float(match["fwhm"][0]) if match.height else None
        except FileNotFoundError:
            pass
    return TelescopeBeam(
        **common,
        solid_angle_sr=omega,
        method="measured profile",
        fwhm_arcmin=fwhm,
        source_file=str(path),
    )
