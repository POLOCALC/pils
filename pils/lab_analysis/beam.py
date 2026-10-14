"""Antenna beam patterns: FWHM per cut and frequency, and the gain."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from pils.lab_analysis.base import LabAnalysis, LabVersion
from pils.lab_analysis.store import LabStore
from pils.utils.hdf5 import load_h5, new_version_name
from pils.utils.logging_config import get_logger
from pils.utils.units import db_to_linear

logger = get_logger(__name__)

FWHM_PER_SIGMA = 2 * np.sqrt(np.log(2))  # for exp(-x²/σ²)
BEAM_EFFICIENCY = 1.13  # Ω ≈ 1.13 θ_E θ_H for a Gaussian beam
DB_PER_RELATIVE = 10 / np.log(10)  # d(10 log10 x) = 4.343 dx / x


def gaussian(
    x: np.ndarray, amplitude: float, centre: float, sigma: float
) -> np.ndarray:
    """``amplitude * exp(-(x - centre)² / sigma²)``."""
    return amplitude * np.exp(-((x - centre) ** 2) / sigma**2)


def expected_fwhm_deg(gain_dbi: float) -> float:
    """FWHM (deg) of a symmetric Gaussian beam with this gain."""
    return float(np.degrees(np.sqrt(4 * np.pi / (1.133 * db_to_linear(gain_dbi)))))


def normalised_cut(cut: dict[str, np.ndarray], frequency_ghz: float) -> pl.DataFrame:
    """One frequency of a cut, S12 normalised to the on-axis co-polar value."""
    copol = cut["copol"]
    at_freq = copol["Freq(GHz)"] == frequency_ghz
    angle = np.asarray(copol["Ang(DEG)"][at_freq], dtype=float)
    s12 = np.asarray(copol["S12(DB)"][at_freq], dtype=float)
    on_axis = s12[np.argmin(np.abs(angle))]
    return pl.DataFrame({"angle_deg": angle, "s12_db": s12 - on_axis}).sort("angle_deg")


def fit_fwhm(
    cut: pl.DataFrame, half_width_deg: float
) -> tuple[float, float, dict[str, float]]:
    """Gaussian fit of the main lobe (|angle| < half_width) in linear power.

    Returns
    -------
    tuple
        FWHM (deg), its 1σ error, and the fit parameters.
    """
    angle = cut["angle_deg"].to_numpy()
    mask = (angle > -half_width_deg) & (angle < half_width_deg)
    if mask.sum() < 4:
        raise ValueError(f"Only {mask.sum()} points within ±{half_width_deg:.1f}°")
    popt, pcov = curve_fit(
        gaussian,
        angle[mask],
        db_to_linear(cut["s12_db"].to_numpy()[mask]),
        p0=[1.0, 0.0, 2.0],
    )
    sigma, sigma_err = abs(popt[2]), float(np.sqrt(pcov[2, 2]))
    params = {
        "amplitude": float(popt[0]),
        "centre_deg": float(popt[1]),
        "sigma_deg": float(sigma),
        "n_points": int(mask.sum()),
    }
    return FWHM_PER_SIGMA * sigma, FWHM_PER_SIGMA * sigma_err, params


def gain_from_fwhm(fwhm: pl.DataFrame) -> pl.DataFrame:
    """Directivity gain from the beam widths: G = 4π / Ω.

    Uses the H and E planes (Ω = 1.13 θ_H θ_E) when both are present,
    otherwise the D plane (Ω = 1.13 θ_D²).
    """
    planes = set(fwhm["plane"])

    def plane(name: str) -> pl.DataFrame:
        return fwhm.filter(pl.col("plane") == name).sort("frequency_ghz")

    if {"Hplane", "Eplane"} <= planes:
        h, e = plane("Hplane"), plane("Eplane")
        if h["frequency_ghz"].to_list() != e["frequency_ghz"].to_list():
            raise ValueError("H and E planes have different frequencies")
        th, dth = np.radians(h["fwhm_deg"]), np.radians(h["fwhm_err_deg"])
        te, dte = np.radians(e["fwhm_deg"]), np.radians(e["fwhm_err_deg"])
        omega = BEAM_EFFICIENCY * th * te
        omega_err = BEAM_EFFICIENCY * np.sqrt((te * dth) ** 2 + (th * dte) ** 2)
        freqs, method = h["frequency_ghz"], "H+E planes"
    elif "Dplane" in planes:
        d = plane("Dplane")
        td, dtd = np.radians(d["fwhm_deg"]), np.radians(d["fwhm_err_deg"])
        omega = BEAM_EFFICIENCY * td**2
        omega_err = 2 * BEAM_EFFICIENCY * td * dtd
        freqs, method = d["frequency_ghz"], "D plane"
    else:
        raise ValueError(f"No plane to compute the gain from (have {sorted(planes)})")
    omega, omega_err = np.asarray(omega), np.asarray(omega_err)
    return pl.DataFrame(
        {
            "frequency_ghz": freqs,
            "gain_dbi": 10 * np.log10(4 * np.pi / omega),
            "gain_err_db": DB_PER_RELATIVE * omega_err / omega,
            "solid_angle_sr": omega,
            "method": [method] * len(omega),
        }
    )


class BeamAnalysis(LabAnalysis):
    """FWHM and gain of an antenna from beam cuts measured in the anechoic chamber.

    The raw file holds ``<plane>/<copol|crosspol>/{Freq(GHz), Ang(DEG),
    S12(DB), ...}`` for the H, E and D planes.

    Examples
    --------
    >>> beam = BeamAnalysis.from_folder(".../ANT_001/radiation_pattern")
    ... # doctest: +SKIP
    >>> version = beam.run_analysis(nominal_gain_dbi=15.0, save_data=True)
    ... # doctest: +SKIP
    """

    analysis_name = "beam"

    def nominal_gain(self, store: LabStore | None) -> float:
        """Nominal gain of the receiving antenna (STOUT specifications)."""
        components = self.measurement.config.get("components", {})
        item_id = components.get("receiving_antenna", {}).get("item_id")
        if item_id is None:
            raise ValueError("The notes do not name the receiving antenna")
        item = (store or LabStore()).item(item_id)
        return float(item["specifications"]["gain"])

    def run_analysis(
        self,
        nominal_gain_dbi: float | None = None,
        n_fwhm: float = 1.0,
        store: LabStore | None = None,
        save_data: bool = False,
        plot: bool = False,
    ) -> LabVersion:
        """Fit the main lobe of every cut and frequency, then compute the gain.

        Parameters
        ----------
        nominal_gain_dbi : float, optional
            Sets the fit range (±``n_fwhm`` expected FWHMs); read from the
            receiving antenna in STOUT when None.
        n_fwhm : float
            Fit half-width in expected FWHMs.
        store : LabStore, optional
            For the STOUT lookup of the antenna.
        """
        if self.measurement.raw_path is None:
            raise FileNotFoundError("The measurement has no raw file")
        if nominal_gain_dbi is None:
            nominal_gain_dbi = self.nominal_gain(store)
        half_width = expected_fwhm_deg(nominal_gain_dbi) * n_fwhm
        raw = load_h5(self.measurement.raw_path)

        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        for plane, cut in raw.items():
            for freq in np.unique(cut["copol"]["Freq(GHz)"]):
                try:
                    fwhm, err, params = fit_fwhm(normalised_cut(cut, freq), half_width)
                except (RuntimeError, ValueError) as e:
                    failures.append(f"{plane} {freq:g} GHz: {e}")
                    continue
                rows.append(
                    {
                        "plane": plane,
                        "frequency_ghz": float(freq),
                        "fwhm_deg": fwhm,
                        "fwhm_err_deg": err,
                    }
                    | params
                )
        if failures:
            logger.warning(f"{len(failures)} fit(s) failed: {failures}")
        fwhm_table = pl.DataFrame(rows).sort(["plane", "frequency_ghz"])
        gain = gain_from_fwhm(fwhm_table)

        version = LabVersion(
            version_name=new_version_name(),
            tables={"fwhm": fwhm_table, "gain": gain},
            metadata=self._base_metadata()
            | {
                "nominal_gain_dbi": nominal_gain_dbi,
                "n_fwhm": n_fwhm,
                "fit_half_width_deg": half_width,
                "failed_fits": failures,
                "grid_status": self.measurement.config.get(
                    "auxiliary_components", {}
                ).get("grid_status"),
            },
        )
        if save_data:
            self._save(version)
        if plot:
            self.save_plots(version, raw)
        return version

    def save_plots(
        self, version: LabVersion, raw: dict[str, Any] | None = None
    ) -> list[Path]:
        """Cuts with their fit at a few frequencies, FWHM and gain vs frequency."""
        from matplotlib.figure import Figure

        raw = raw if raw is not None else load_h5(self.measurement.raw_path)
        fwhm, gain = version["fwhm"], version["gain"]
        folder = self.figures_dir(version.version_name)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []

        planes = sorted(raw)
        fig = Figure(figsize=(5 * len(planes), 4), layout="constrained")
        axes = np.atleast_1d(fig.subplots(1, len(planes)))
        for ax, plane in zip(axes, planes, strict=True):
            freqs = np.unique(raw[plane]["copol"]["Freq(GHz)"])
            for freq in freqs[:: max(1, len(freqs) // 4)]:
                cut = normalised_cut(raw[plane], freq)
                (line,) = ax.plot(
                    cut["angle_deg"], cut["s12_db"], label=f"{freq:g} GHz"
                )
                fit = fwhm.filter(
                    (pl.col("plane") == plane) & (pl.col("frequency_ghz") == freq)
                )
                if fit.height:
                    x = cut["angle_deg"].to_numpy()
                    model = gaussian(
                        x,
                        fit["amplitude"][0],
                        fit["centre_deg"][0],
                        fit["sigma_deg"][0],
                    )
                    with np.errstate(divide="ignore"):
                        ax.plot(x, 10 * np.log10(model), "--", color=line.get_color())
            ax.set(title=plane, xlabel="Angle [deg]", ylabel="Normalised S12 [dB]")
            ax.set_ylim(-40, 3)
            ax.legend(fontsize=8)
        paths.append(folder / "beam_cuts.png")
        fig.savefig(paths[-1], dpi=150)

        fig = Figure(figsize=(11, 4), layout="constrained")
        ax_fwhm, ax_gain = fig.subplots(1, 2)
        for plane in sorted(set(fwhm["plane"])):
            part = fwhm.filter(pl.col("plane") == plane)
            ax_fwhm.errorbar(
                part["frequency_ghz"],
                part["fwhm_deg"],
                part["fwhm_err_deg"],
                marker="o",
                ms=3,
                label=plane,
            )
        ax_fwhm.set(xlabel="Frequency [GHz]", ylabel="FWHM [deg]", title="Beam width")
        ax_fwhm.legend()
        ax_gain.errorbar(
            gain["frequency_ghz"],
            gain["gain_dbi"],
            gain["gain_err_db"],
            marker="o",
            ms=3,
            color="black",
        )
        ax_gain.set(
            xlabel="Frequency [GHz]",
            ylabel="Gain [dBi]",
            title=f"Gain ({gain['method'][0]})",
        )
        paths.append(folder / "fwhm_gain.png")
        fig.savefig(paths[-1], dpi=150)
        logger.info(f"Saved beam plots to {folder}")
        return paths
