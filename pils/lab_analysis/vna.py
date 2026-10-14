"""VNA two-port measurements (Touchstone .s2p): S parameters per frequency."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils.lab_analysis.base import LabAnalysis, LabVersion
from pils.utils.hdf5 import new_version_name
from pils.utils.logging_config import get_logger
from pils.utils.units import db_to_linear, linear_to_db

logger = get_logger(__name__)

DEFAULT_FREQUENCIES_GHZ = (84.0, 87.0, 90.0, 93.0, 96.0, 99.0, 102.0)
S_PARAMS = ("s11", "s21", "s12", "s22")  # Touchstone two-port column order
_FREQ_UNITS = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}


def read_touchstone(path: str | Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Read a two-port Touchstone file.

    Returns
    -------
    tuple[pl.DataFrame, dict]
        ``frequency_ghz`` and, per S parameter, ``<s>_db`` (20 log10 |S|)
        and ``<s>_deg``; plus the header (comments and option line).
    """
    unit, fmt = "ghz", "ma"
    comments: list[str] = []
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("!"):
            comments.append(line[1:].strip())
            continue
        if line.startswith("#"):
            options = line[1:].lower().split()
            unit = next((o for o in options if o in _FREQ_UNITS), unit)
            fmt = next((o for o in options if o in ("db", "ma", "ri")), fmt)
            continue
        rows.append([float(v) for v in line.split("!")[0].split()])

    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != 9:
        raise ValueError(f"{path}: expected 9 columns (two-port), got {data.shape}")
    columns: dict[str, np.ndarray] = {"frequency_ghz": data[:, 0] * _FREQ_UNITS[unit]}
    for i, name in enumerate(S_PARAMS):
        a, b = data[:, 1 + 2 * i], data[:, 2 + 2 * i]
        if fmt == "db":
            mag_db, deg = a, b
        elif fmt == "ma":
            mag_db, deg = 20 * np.log10(a), b
        else:  # ri
            mag_db = 20 * np.log10(np.hypot(a, b))
            deg = np.degrees(np.arctan2(b, a))
        columns[f"{name}_db"] = mag_db
        columns[f"{name}_deg"] = deg
    header = {"comments": comments, "frequency_unit": unit, "format": fmt}
    return pl.DataFrame(columns), header


def mismatch_corrected(spectrum: pl.DataFrame) -> pl.DataFrame:
    """Add transmissions corrected for the reflection at the input port.

    ``S21 / (1 - |S11|²)`` and ``S12 / (1 - |S22|²)`` in power terms: the
    transmission relative to the power accepted by the device.
    """

    def corrected(transmission: str, reflection: str) -> np.ndarray:
        t = db_to_linear(spectrum[transmission].to_numpy())
        r = db_to_linear(spectrum[reflection].to_numpy())
        return linear_to_db(t / (1 - r))

    return spectrum.with_columns(
        pl.Series("s21_corrected_db", corrected("s21_db", "s11_db")),
        pl.Series("s12_corrected_db", corrected("s12_db", "s22_db")),
    )


def frequency_summary(
    spectrum: pl.DataFrame,
    frequencies_ghz: tuple[float, ...] | list[float],
    window_ghz: float = 0.1,
    corrected: bool = False,
) -> pl.DataFrame:
    """Mean and standard deviation of each S parameter around each frequency.

    Points with ``f0 - window_ghz < f < f0 + window_ghz`` are averaged. With ``corrected``
    the transmissions are the mismatch-corrected ones.
    """
    source = {name: f"{name}_db" for name in S_PARAMS}
    if corrected:
        source |= {"s21": "s21_corrected_db", "s12": "s12_corrected_db"}
    rows = []
    freq = spectrum["frequency_ghz"].to_numpy()
    for f0 in frequencies_ghz:
        mask = (freq > f0 - window_ghz) & (freq < f0 + window_ghz)
        if not mask.any():
            raise ValueError(f"No VNA points within {window_ghz} GHz of {f0} GHz")
        row: dict[str, float | int] = {
            "frequency_ghz": float(f0),
            "n_points": int(mask.sum()),
        }
        for name, column in source.items():
            values = spectrum[column].to_numpy()[mask]
            row[f"{name}_avg_db"] = float(np.mean(values))
            row[f"{name}_std_db"] = float(np.std(values))
        rows.append(row)
    return pl.DataFrame(rows)


class VNAAnalysis(LabAnalysis):
    """S parameters of a component from a two-port VNA measurement.

    Used for transmission, attenuation, coupling, directivity and
    insertion-loss test types: the raw file is ``<stem>_raw.s2p``.

    Examples
    --------
    >>> vna = VNAAnalysis.from_folder(".../WR10/attenuator/ATT_002/attenuation")
    ... # doctest: +SKIP
    >>> version = vna.run_analysis(save_data=True, plot=True)  # doctest: +SKIP
    >>> version["summary"]  # doctest: +SKIP
    """

    analysis_name = "vna"

    def run_analysis(
        self,
        frequencies_ghz: tuple[float, ...] | list[float] = DEFAULT_FREQUENCIES_GHZ,
        window_ghz: float = 0.1,
        corrected: bool = False,
        save_data: bool = False,
        plot: bool = False,
    ) -> LabVersion:
        """Summarise the S parameters at the given frequencies.

        Parameters
        ----------
        frequencies_ghz : sequence of float
            Frequencies of the summary.
        window_ghz : float
            Half-width of the averaging window.
        corrected : bool
            Summarise the mismatch-corrected transmissions.
        save_data : bool
            Save the run to the results file.
        plot : bool
            Save the plots of the run (implies a results folder).
        """
        if self.measurement.raw_path is None:
            raise FileNotFoundError("The measurement has no raw file")
        spectrum, header = read_touchstone(self.measurement.raw_path)
        spectrum = mismatch_corrected(spectrum)
        summary = frequency_summary(spectrum, frequencies_ghz, window_ghz, corrected)

        version = LabVersion(
            version_name=new_version_name(),
            tables={"spectrum": spectrum, "summary": summary},
            metadata=self._base_metadata()
            | {
                "frequencies_ghz": list(frequencies_ghz),
                "window_ghz": window_ghz,
                "corrected": corrected,
                "touchstone": header,
            },
        )
        if save_data:
            self._save(version)
        if plot:
            self.save_plots(version)
        return version

    def save_plots(self, version: LabVersion) -> list[Path]:
        """S parameters and per-frequency transmission zooms (PNG)."""
        from matplotlib.figure import Figure

        spectrum, summary = version["spectrum"], version["summary"]
        window = version.metadata.get("window_ghz", 0.1)
        folder = self.figures_dir(version.version_name)
        folder.mkdir(parents=True, exist_ok=True)
        freq = spectrum["frequency_ghz"].to_numpy()

        fig = Figure(figsize=(12, 9), layout="constrained")
        axes = fig.subplots(2, 2)
        for ax, name in zip(axes.flat, ["s11", "s12", "s21", "s22"], strict=True):
            ax.plot(freq, spectrum[f"{name}_db"], label="Measured")
            if f"{name}_corrected_db" in spectrum.columns:
                ax.plot(
                    freq, spectrum[f"{name}_corrected_db"], label="Mismatch corrected"
                )
                ax.legend()
            ax.set(title=name.upper(), xlabel="Frequency [GHz]", ylabel="|S| [dB]")
        fig.suptitle(
            f"{self.measurement.folder.parent.name} {self.measurement.test_type}"
        )
        paths = [folder / "s_parameters.png"]
        fig.savefig(paths[-1], dpi=150)

        n = summary.height
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig = Figure(figsize=(4.5 * ncols, 3.5 * nrows), layout="constrained")
        axes = list(np.atleast_1d(fig.subplots(nrows, ncols)).flat)
        column = "s21_corrected_db" if version.metadata.get("corrected") else "s21_db"
        for ax, row in zip(axes, summary.iter_rows(named=True), strict=False):
            f0 = row["frequency_ghz"]
            mask = (freq > f0 - window) & (freq < f0 + window)
            ax.plot(freq[mask], spectrum[column].to_numpy()[mask], "o-", color="black")
            avg, std = row["s21_avg_db"], row["s21_std_db"]
            ax.axhline(avg, color="green", ls="--")
            ax.axhspan(avg - std, avg + std, color="green", alpha=0.3)
            ax.set(title=f"S21 @ {f0:g} GHz", xlabel="Frequency [GHz]", ylabel="[dB]")
        for ax in axes[n:]:
            ax.set_visible(False)
        paths.append(folder / "s21_frequency_zoom.png")
        fig.savefig(paths[-1], dpi=150)
        logger.info(f"Saved VNA plots to {folder}")
        return paths
