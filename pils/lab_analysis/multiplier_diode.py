"""Multiplier output power and diode responsivity (climate chamber).

The power meter reads the multiplier output through a directional coupler
and waveguides; the diode detector sits on the coupled port. The raw file
holds ``"<f> GHz"/"<t> C"/"<cv> V"/{Time [s], Power [mW], Diode output [mV]}``
with the attenuator control voltage as attribute.
"""

from pathlib import Path

import h5py
import numpy as np
import polars as pl

from pils.lab_analysis.base import LabAnalysis, LabVersion
from pils.lab_analysis.notes import component_ids
from pils.lab_analysis.store import Item, LabStore
from pils.utils.hdf5 import new_version_name
from pils.utils.logging_config import get_logger
from pils.utils.units import dbm_to_mw, mw_to_dbm

logger = get_logger(__name__)

POWER_METER_OFFSET_DB = 0.2  # VDI PM5B specification
BLOCK_SAMPLES = 30  # low-power runs alternate 30 samples off / 30 on
LOW_POWER_MIN_SAMPLES = 100


def _finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def summarise_step(power: np.ndarray, diode: np.ndarray) -> dict[str, float]:
    """Mean power and diode output of one control-voltage step.

    Long (low-power) steps alternate source off/on blocks: the signal is the
    on-off difference, averaged over the positive differences, and the off
    level is the power-meter baseline. Short steps are plain averages.
    """
    if len(power) <= LOW_POWER_MIN_SAMPLES:
        finite = np.isfinite(power)
        return {
            "power_mw": float(np.mean(power[finite])),
            "power_err_mw": float(np.std(power[finite])),
            "diode_output_mv": float(np.mean(diode)),
            "diode_output_err_mv": float(np.std(diode)),
            "power_baseline_mw": 0.0,
        }
    d_power, d_diode, baseline = [], [], []
    # Pairs start at even blocks; a trailing partial pair gives NaN and is
    # dropped by the positive-difference selection
    n_blocks = -(-len(power) // BLOCK_SAMPLES)
    for n in range(0, n_blocks, 2):
        off = slice(n * BLOCK_SAMPLES, (n + 1) * BLOCK_SAMPLES)
        on = slice((n + 1) * BLOCK_SAMPLES, (n + 2) * BLOCK_SAMPLES)
        d_power.append(_finite_mean(power[on]) - _finite_mean(power[off]))
        d_diode.append(_finite_mean(diode[on]) - _finite_mean(diode[off]))
        baseline.append(_finite_mean(power[off]))
    d_power_a, d_diode_a = np.asarray(d_power), np.asarray(d_diode)
    pos_p, pos_d = d_power_a[d_power_a > 0], d_diode_a[d_diode_a > 0]
    return {
        "power_mw": float(np.mean(pos_p)) if len(pos_p) else float("nan"),
        "power_err_mw": float(np.std(pos_p)) if len(pos_p) else float("nan"),
        "diode_output_mv": float(np.mean(pos_d)) if len(pos_d) else float("nan"),
        "diode_output_err_mv": float(np.std(pos_d)) if len(pos_d) else float("nan"),
        "power_baseline_mw": float(np.mean(baseline)),
    }


def read_raw(path: str | Path) -> pl.DataFrame:
    """One row per frequency, temperature and control-voltage step."""
    rows = []
    with h5py.File(path, "r") as f:
        for freq_group in f.values():
            for temp_group in freq_group.values():
                for step in temp_group.values():
                    rows.append(
                        {
                            "frequency_ghz": float(freq_group.attrs["Frequency"]),
                            "temperature_c": float(temp_group.attrs["Temperature"]),
                            "control_voltage_v": float(step.attrs["control_voltage"]),
                            "pm_range_mw": float(step.attrs["PM range"]),
                            "n_samples": len(step["Time [s]"]),
                        }
                        | summarise_step(
                            np.asarray(step["Power [mW]"][()], float),
                            np.asarray(step["Diode output [mV]"][()], float),
                        )
                    )
    return pl.DataFrame(rows).sort(
        ["frequency_ghz", "temperature_c", "control_voltage_v"]
    )


def subtract_baseline(steps: pl.DataFrame) -> pl.DataFrame:
    """Subtract the mean power-meter baseline from the short (high-power) steps.

    Per frequency and temperature: the baseline measured in the long steps
    applies to the short ones, which have none.
    """
    keys = ["frequency_ghz", "temperature_c"]
    measured = pl.col("power_baseline_mw") != 0
    mean_baseline = (
        pl.col("power_baseline_mw").filter(measured).mean().over(keys).fill_null(0.0)
    )
    return steps.with_columns(
        pl.when(measured)
        .then(pl.col("power_mw"))
        .otherwise(pl.col("power_mw") - mean_baseline)
        .alias("power_mw")
    )


class MultiplierDiodeAnalysis(LabAnalysis):
    """Multiplier output power and diode responsivity from one chamber run.

    The same raw file sits in the multiplier ``output_power`` folder and the
    detector ``responsivity`` folder; a saved run is written to both.

    Examples
    --------
    >>> run = MultiplierDiodeAnalysis.from_folder(".../DET_001/responsivity")
    ... # doctest: +SKIP
    >>> version = run.run_analysis(store=LabStore(), save_data=True)
    ... # doctest: +SKIP
    """

    analysis_name = "multiplier_diode"
    TEST_TYPES = {"multiplier": "output_power", "detector": "responsivity"}

    def components(self, store: LabStore) -> dict[str, Item]:
        """Items named in the notes: multiplier, detector, coupler, waveguides."""
        ids = component_ids(self.measurement.config)
        needed = {"multiplier", "detector", "aux.directional_coupler"}
        missing = needed - set(ids)
        if missing:
            raise ValueError(f"The notes do not name: {sorted(missing)}")
        return {role: store.item(item_id) for role, item_id in ids.items()}

    def run_analysis(
        self,
        store: LabStore | None = None,
        save_data: bool = False,
        plot: bool = False,
    ) -> LabVersion:
        """Process the run with the coupler and waveguide lab results.

        Power meter reading -> coupler output: add back the waveguide
        losses and the power-meter offset. Diode input: coupler output
        scaled by coupling - transmission of the coupler.
        """
        if self.measurement.raw_path is None:
            raise FileNotFoundError("The measurement has no raw file")
        store = store or LabStore()
        items = self.components(store)
        coupler = items["aux.directional_coupler"]
        waveguides = {r: i for r, i in items.items() if r.startswith("aux.waveguides")}

        steps = subtract_baseline(read_raw(self.measurement.raw_path))
        corrections = []
        for freq in steps["frequency_ghz"].unique().sort().to_list():
            transmission, transmission_err = store.transmission(
                coupler, "transmission", freq
            )
            coupling, coupling_err = store.transmission(
                coupler, "coupling_performance", freq
            )
            wg = [
                store.transmission(i, "transmission", freq) for i in waveguides.values()
            ]
            corrections.append(
                {
                    "frequency_ghz": freq,
                    "coupler_transmission_db": transmission,
                    "coupler_transmission_err_db": transmission_err,
                    "coupler_coupling_db": coupling,
                    "coupler_coupling_err_db": coupling_err,
                    "waveguides_transmission_db": float(sum(v for v, _ in wg)),
                    "waveguides_transmission_err_db": float(
                        np.sqrt(sum(e**2 for _, e in wg))
                    ),
                }
            )
        corrections_table = pl.DataFrame(corrections)
        table = self._apply_corrections(
            steps.join(corrections_table, on="frequency_ghz")
        )

        version = LabVersion(
            version_name=new_version_name(),
            tables={"responsivity": table, "corrections": corrections_table},
            metadata=self._base_metadata()
            | {
                "power_meter_offset_db": POWER_METER_OFFSET_DB,
                "items": {
                    role: {"item_id": i.get("item_id"), "name": i.get("name")}
                    for role, i in items.items()
                },
            },
        )
        if save_data:
            for folder in self.output_folders(store, items):
                self._save(version, folder / "proc" / self.analysis_name)
        if plot:
            self.save_plots(version)
        return version

    @staticmethod
    def _apply_corrections(rows: pl.DataFrame) -> pl.DataFrame:
        loss = pl.col("waveguides_transmission_db") - POWER_METER_OFFSET_DB
        coupled = pl.col("coupler_coupling_db") - pl.col("coupler_transmission_db")
        power_dbm = (
            mw_to_dbm(rows["power_mw"].to_numpy()) - rows.select(loss).to_series()
        )
        rows = rows.with_columns(
            pl.Series("power_dbm", power_dbm),
            pl.Series("power_mw", dbm_to_mw(power_dbm)),
            # Waveguide uncertainties are negligible: scale only
            (pl.col("power_err_mw") * 10 ** (loss.abs() / 10)).alias("power_err_mw"),
        )
        rows = rows.with_columns(
            (pl.col("power_mw") * 10 ** (coupled / 10)).alias("diode_input_mw")
        )
        relative = (
            (pl.col("power_err_mw") / pl.col("power_mw")) ** 2
            + (np.log(10) / 10) ** 2
            * (
                pl.col("coupler_coupling_err_db") ** 2
                + pl.col("coupler_transmission_err_db") ** 2
            )
        ).sqrt()
        rows = rows.with_columns(
            (pl.col("diode_input_mw") * relative).alias("diode_input_err_mw")
        )
        return rows.select(
            "frequency_ghz",
            "temperature_c",
            "control_voltage_v",
            "power_mw",
            "power_err_mw",
            "power_dbm",
            "diode_output_mv",
            "diode_output_err_mv",
            "diode_input_mw",
            "diode_input_err_mw",
            "power_baseline_mw",
            "pm_range_mw",
            "n_samples",
        )

    def output_folders(self, store: LabStore, items: dict[str, Item]) -> list[Path]:
        """Test folders the run is saved in (own folder first)."""
        if self.results_dir.parent.parent != self.measurement.folder:
            # Redirected output: one sub-folder per test type
            base = self.results_dir.parent.parent
            return [base / t for t in self.TEST_TYPES.values()]
        folders = [self.measurement.folder]
        for role, test_type in self.TEST_TYPES.items():
            folder = store.test_folder(items[role], test_type)
            if folder != self.measurement.folder and folder.is_dir():
                folders.append(folder)
        return folders

    def save_plots(self, version: LabVersion) -> list[Path]:
        """Output power and diode response vs control voltage, per frequency."""
        from matplotlib.figure import Figure

        table = version["responsivity"]
        folder = self.figures_dir(version.version_name)
        folder.mkdir(parents=True, exist_ok=True)
        freqs = table["frequency_ghz"].unique().sort().to_list()
        paths = []
        for name, y, label in [
            ("output_power", "power_mw", "Coupler output power [mW]"),
            ("responsivity", "diode_output_mv", "Diode output [mV]"),
        ]:
            ncols = min(4, len(freqs))
            nrows = int(np.ceil(len(freqs) / ncols))
            fig = Figure(figsize=(4 * ncols, 3.2 * nrows), layout="constrained")
            axes = list(np.atleast_1d(fig.subplots(nrows, ncols)).flat)
            x = "control_voltage_v" if name == "output_power" else "diode_input_mw"
            for ax, freq in zip(axes, freqs, strict=False):
                part = table.filter(pl.col("frequency_ghz") == freq)
                for temp in part["temperature_c"].unique().sort().to_list():
                    rows = part.filter(pl.col("temperature_c") == temp).sort(x)
                    ax.plot(rows[x], rows[y], ".-", label=f"{temp:g} °C")
                ax.set(title=f"{freq:g} GHz", ylabel=label)
                ax.set_xlabel(
                    "Control voltage [V]"
                    if x == "control_voltage_v"
                    else "Diode input [mW]"
                )
                if x == "diode_input_mw":
                    ax.set_xscale("log")
                    ax.set_yscale("log")
            for ax in axes[len(freqs) :]:
                ax.set_visible(False)
            axes[0].legend(fontsize=7)
            paths.append(folder / f"{name}.png")
            fig.savefig(paths[-1], dpi=150)
        logger.info(f"Saved multiplier/diode plots to {folder}")
        return paths
