"""Time-domain steps of the power analysis: scan range, ADC envelope, binning."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks, medfilt

from pils.lab_analysis.store import Item, LabStore
from pils.utils.uncertainty import linear_interpolate_rows


def scanning_range(
    timestamp: ArrayLike,
    altitude: ArrayLike,
    min_altitude_m: float,
    peak_distance: int = 75,
) -> tuple[float, float]:
    """Start and end of the scan from the altitude peaks above ``min_altitude_m``.

    The scan starts half a peak spacing before the first peak and ends at
    the last one.

    Raises
    ------
    ValueError
        If there are fewer than two altitude peaks.
    """
    t = np.asarray(timestamp, dtype=float)
    alt = np.asarray(altitude, dtype=float)
    high = alt > min_altitude_m
    peaks, _ = find_peaks(alt[high], distance=peak_distance)
    if len(peaks) < 2:
        raise ValueError(
            f"Found {len(peaks)} altitude peak(s) above {min_altitude_m} m: no scan"
        )
    t_peaks = t[high][peaks]
    spacing = float(np.mean(np.gradient(t_peaks)))
    return float(t_peaks[0] - spacing / 2), float(t_peaks[-1])


def adc_envelope(
    amplitude: ArrayLike, window: int, kernel_size: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-filtered signal and its block max (envelope) and min (zero level).

    The source is square-wave modulated: ``window`` samples cover one high
    or low half period.
    """
    filtered = medfilt(np.asarray(amplitude, dtype=float), kernel_size=kernel_size)
    envelope = np.empty_like(filtered)
    zero = np.empty_like(filtered)
    for start in range(0, len(filtered), window):
        block = filtered[start : start + window]
        envelope[start : start + window] = block.max()
        zero[start : start + window] = block.min()
    return filtered, envelope, zero


def bin_mean(
    time: ArrayLike, values: ArrayLike, factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Means of consecutive blocks of ``factor`` samples and the error on the mean."""
    n = len(time) // factor * factor
    t = np.asarray(time, dtype=float)[:n].reshape(-1, factor)
    v = np.asarray(values, dtype=float)[:n].reshape(-1, factor)
    return t.mean(axis=1), v.mean(axis=1), v.std(axis=1, ddof=1) / np.sqrt(factor)


class DiodeCalibration:
    """Diode responsivity at one frequency on a (temperature, control voltage) grid.

    Only the control voltages measured at every temperature are kept, so
    the curves can be interpolated in temperature.
    """

    COLUMNS = (
        "diode_output_mv",
        "diode_output_err_mv",
        "diode_input_mw",
        "diode_input_err_mw",
    )

    def __init__(self, responsivity, frequency_ghz: float) -> None:
        import polars as pl

        part = responsivity.filter(
            (pl.col("frequency_ghz") - frequency_ghz).abs() < 1e-6
        )
        if part.is_empty():
            raise ValueError(f"No diode responsivity at {frequency_ghz} GHz")
        temps = sorted(part["temperature_c"].unique().to_list())
        common = None
        for temp in temps:
            cv = set(part.filter(pl.col("temperature_c") == temp)["control_voltage_v"])
            common = cv if common is None else common & cv
        self.control_voltage_v = np.array(sorted(common or []))
        if len(self.control_voltage_v) < 2:
            raise ValueError(
                "Fewer than two control voltages shared by all temperatures"
            )
        self.temperature_c = np.array(temps, dtype=float)
        part = part.filter(pl.col("control_voltage_v").is_in(self.control_voltage_v))
        self.grids = {}
        for column in self.COLUMNS:
            grid = np.empty((len(temps), len(self.control_voltage_v)))
            for k, temp in enumerate(temps):
                rows = part.filter(pl.col("temperature_c") == temp).sort(
                    "control_voltage_v"
                )
                grid[k] = rows[column].to_numpy()
            self.grids[column] = grid

    @classmethod
    def from_store(
        cls, store: LabStore, detector: Item, frequency_ghz: float
    ) -> "DiodeCalibration":
        """Build the calibration of a detector from the lab store."""
        return cls(store.responsivity(detector), frequency_ghz)

    def curves(self, temperature_c: ArrayLike) -> dict[str, np.ndarray]:
        """Calibration curves at each temperature (clamped to the measured range).

        Returns (n, n_cv) arrays per column, linear in temperature.
        """
        temp = np.clip(
            np.asarray(temperature_c, dtype=float),
            self.temperature_c[0],
            self.temperature_c[-1],
        )
        k = np.clip(
            np.searchsorted(self.temperature_c, temp) - 1,
            0,
            len(self.temperature_c) - 2,
        )
        t0, t1 = self.temperature_c[k], self.temperature_c[k + 1]
        w = ((temp - t0) / (t1 - t0))[:, None] if len(self.temperature_c) > 1 else 0.0
        out = {}
        for column, grid in self.grids.items():
            if len(self.temperature_c) == 1:
                out[column] = np.repeat(grid, len(temp), axis=0)
            else:
                out[column] = grid[k] * (1 - w) + grid[k + 1] * w
        return out

    def input_power(
        self,
        diode_output_mv: ArrayLike,
        diode_output_err_mv: ArrayLike,
        temperature_c: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Diode input power (mW, 1σ) from its output voltage at a temperature."""
        c = self.curves(temperature_c)
        return linear_interpolate_rows(
            c["diode_output_mv"],
            c["diode_input_mw"],
            c["diode_output_err_mv"],
            c["diode_input_err_mw"],
            diode_output_mv,
            diode_output_err_mv,
        )
