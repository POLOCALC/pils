"""Finding and reading lab results of inventory items.

The lab tree is ``<root>/<band>/<category>/<item name>/<test_type>/``. For
each quantity the store returns a standard table, read from the latest PILS
analysis run when there is one (``proc/<analysis>/<stem>/rev_*/``, or a
legacy ``<analysis>_solution.h5``), otherwise from the processed files written
before PILS had lab analyses (``*.csv``, ``*_processed.*``).
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils.utils import hdf5
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_LAB_ROOT = Path("/mnt/data/POLOCALC/lab")

Item = dict[str, Any]


class LabDataNotFound(LookupError):
    """No lab result for an item / test type / frequency."""


def _read_csv(path: Path, names: dict[str, str]) -> pl.DataFrame:
    """CSV with columns renamed (missing ones ignored), frequency as float."""
    frame = pl.read_csv(path)
    frame = frame.rename({k: v for k, v in names.items() if k in frame.columns})
    if "frequency_ghz" in frame.columns:
        frame = frame.with_columns(pl.col("frequency_ghz").cast(pl.Float64))
    return frame


def default_lab_root() -> Path:
    """``PILS_LAB_DATA_PATH``, else STOUT's data path + ``lab``, else the share."""
    env = os.getenv("PILS_LAB_DATA_PATH")
    if env:
        return Path(env)
    try:
        from stout.config import Config  # type: ignore

        return Path(Config.MAIN_DATA_PATH) / "lab"
    except ImportError:
        return DEFAULT_LAB_ROOT


def _stout_item(item_id: str) -> Item:
    try:
        from stout.exceptions import NotFoundError  # type: ignore
        from stout.services.inventory.service import InventoryService  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Looking up items by id needs STOUT (pip install pils[stout]); "
            "or pass the item dicts directly."
        ) from e
    try:
        return InventoryService().get_item_by_id(item_id)
    except NotFoundError as e:
        raise LabDataNotFound(f"No inventory item {item_id}") from e


def value_at(
    table: pl.DataFrame, frequency_ghz: float, column: str, err_column: str
) -> tuple[float, float]:
    """(value, error) of a per-frequency table at one frequency."""
    match = table.filter((pl.col("frequency_ghz") - frequency_ghz).abs() < 1e-6)
    if match.is_empty():
        available = sorted(table["frequency_ghz"].unique().to_list())
        raise LabDataNotFound(
            f"No {column} at {frequency_ghz} GHz (available: {available})"
        )
    return float(match[column][0]), float(match[err_column][0])


class LabStore:
    """Lab results of inventory items.

    Parameters
    ----------
    root : str | Path | None
        Lab data root; see :func:`default_lab_root`.
    get_item : callable, optional
        ``item_id -> item dict``; defaults to STOUT's inventory.

    Examples
    --------
    >>> store = LabStore("/mnt/data/POLOCALC/lab")  # doctest: +SKIP
    >>> store.transmission(coupler, "transmission")  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str | Path | None = None,
        get_item: Callable[[str], Item] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_lab_root()
        self._get_item = get_item or _stout_item

    # -- items and folders -------------------------------------------------

    def item(self, item_id: str) -> Item:
        """Inventory item by id."""
        return self._get_item(item_id)

    def item_folder(self, item: Item) -> Path:
        """``<root>/<band>/<category>/<name>`` of an item.

        The band comes from the item's ``*band`` specifications (e.g.
        ``input_band``/``output_band``); the first one with a folder wins.
        """
        specs = item.get("specifications") or {}
        bands = [str(v) for k, v in specs.items() if "band" in k and v]
        candidates = [self.root / b / item["category"] / item["name"] for b in bands]
        for folder in candidates:
            if folder.is_dir():
                return folder
        if not candidates:
            raise LabDataNotFound(f"{item.get('name')}: no band in its specifications")
        return candidates[0]

    def test_folder(self, item: Item, test_type: str) -> Path:
        """``<item folder>/<test_type>``."""
        return self.item_folder(item) / test_type

    def _latest_run(
        self, folder: Path, analysis: str
    ) -> tuple[dict[str, pl.DataFrame], dict[str, Any]] | None:
        from pils.lab_analysis.base import latest_run

        return latest_run(folder / "proc" / analysis, analysis)

    @staticmethod
    def _legacy_file(folder: Path, *patterns: str) -> Path:
        for pattern in patterns:
            files = sorted(
                p for p in folder.glob(pattern) if p.is_file() and "_raw" not in p.name
            )
            if files:
                return files[-1]  # latest measurement
        raise LabDataNotFound(f"No result ({', '.join(patterns)}) in {folder}")

    # -- quantities ----------------------------------------------------------

    def vna_summary(self, item: Item, test_type: str) -> pl.DataFrame:
        """S parameters per frequency: ``frequency_ghz``, ``s21_avg_db``...

        Older CSVs labelled the Touchstone S21 column "S12" (and S12 "S21");
        they are relabelled here.
        """
        folder = self.test_folder(item, test_type)
        run = self._latest_run(folder, "vna")
        if run is not None:
            return run[0]["summary"]
        names = {"Frequency": "frequency_ghz", "Frequency [GHz]": "frequency_ghz"}
        swap = {"S11": "s11", "S12": "s21", "S21": "s12", "S22": "s22"}
        for old, new in swap.items():
            names[f"{old} avg [dB]"] = f"{new}_avg_db"
            names[f"{old} std [dB]"] = f"{new}_std_db"
        return _read_csv(self._legacy_file(folder, "*.csv"), names)

    def transmission(
        self, item: Item, test_type: str, frequency_ghz: float
    ) -> tuple[float, float]:
        """Forward transmission S21 (dB, 1σ) of an item at one frequency."""
        return value_at(
            self.vna_summary(item, test_type), frequency_ghz, "s21_avg_db", "s21_std_db"
        )

    def insertion_loss(self, item: Item, frequency_ghz: float) -> tuple[float, float]:
        """Antenna insertion loss (dB, 1σ) at one frequency.

        Read from the processed CSV: it is not the raw S21 of the .s2p file,
        so a VNA run of this test type is not used.
        """
        folder = self.test_folder(item, "insertion_loss")
        table = _read_csv(
            self._legacy_file(folder, "*.csv"),
            {
                "Frequency [GHz]": "frequency_ghz",
                "Insertion Loss [dB]": "insertion_loss_db",
                "Err Insertion Loss [dB]": "insertion_loss_err_db",
            },
        )
        return value_at(
            table, frequency_ghz, "insertion_loss_db", "insertion_loss_err_db"
        )

    def reflection(
        self,
        item: Item,
        frequency_ghz: float,
        dataset: str = "fsp_impedance_kapton",
        test_type: str = "impedance_matching",
    ) -> tuple[float, float]:
        """Reflection S11 (dB, 1σ) at one frequency.

        Impedance-matching files hold several set-ups (``fsp_impedance``,
        ``fsp_impedance_kapton``, ``reflections``...); ``dataset`` picks one.
        """
        folder = self.test_folder(item, test_type)
        path = self._legacy_file(folder, "*[0-9].h5py", "*_processed.h5py")
        data = hdf5.load_h5(path)
        if dataset not in data:
            raise LabDataNotFound(
                f"No '{dataset}' in {path} (available: {sorted(data)})"
            )
        table = pl.DataFrame(
            {
                "frequency_ghz": np.asarray(data[dataset]["frequency [GHz]"], float),
                "s11_avg_db": data[dataset]["S11 avg [dB]"],
                "s11_std_db": data[dataset]["S11 std [dB]"],
            }
        )
        return value_at(table, frequency_ghz, "s11_avg_db", "s11_std_db")

    def antenna_gain(self, item: Item) -> pl.DataFrame:
        """``frequency_ghz``, ``gain_dbi``, ``gain_err_db``."""
        folder = self.test_folder(item, "radiation_pattern")
        run = self._latest_run(folder, "beam")
        if run is not None:
            return run[0]["gain"]
        return _read_csv(
            self._legacy_file(folder, "*gain_processed.csv"),
            {
                "Frequency [GHz]": "frequency_ghz",
                "Gain [dBi]": "gain_dbi",
                "err gain [dB]": "gain_err_db",
            },
        )

    def antenna_fwhm(self, item: Item) -> pl.DataFrame:
        """``plane``, ``frequency_ghz``, ``fwhm_deg``, ``fwhm_err_deg``."""
        folder = self.test_folder(item, "radiation_pattern")
        run = self._latest_run(folder, "beam")
        if run is not None:
            return run[0]["fwhm"]
        data = hdf5.load_h5(self._legacy_file(folder, "*fwhm_processed.hdf5"))
        return pl.concat(
            pl.DataFrame(
                {
                    "plane": [plane] * len(cut["Frequency [GHz]"]),
                    "frequency_ghz": cut["Frequency [GHz]"],
                    "fwhm_deg": cut["FWHM [deg]"],
                    "fwhm_err_deg": cut["err FWHM [deg]"],
                }
            )
            for plane, cut in data.items()
        )

    def responsivity(self, item: Item) -> pl.DataFrame:
        """Diode responsivity: one row per frequency, temperature, control voltage.

        Columns: ``frequency_ghz``, ``temperature_c``, ``control_voltage_v``,
        ``diode_output_mv``, ``diode_output_err_mv``, ``diode_input_mw``,
        ``diode_input_err_mw`` (and the measured power columns).
        """
        folder = self.test_folder(item, "responsivity")
        run = self._latest_run(folder, "multiplier_diode")
        if run is not None:
            return run[0]["responsivity"]
        data = hdf5.load_h5(self._legacy_file(folder, "*_processed.h5py"))
        return legacy_responsivity_table(data)


_LEGACY_RESPONSIVITY = {
    "Control Voltage [V]": "control_voltage_v",
    "Power [mW]": "power_mw",
    "err power [mW]": "power_err_mw",
    "Power [dBm]": "power_dbm",
    "Diode output [mV]": "diode_output_mv",
    "err diode output [mV]": "diode_output_err_mv",
    "Diode input pwr [mW]": "diode_input_mw",
    "err diode input [mW]": "diode_input_err_mw",
}


def legacy_responsivity_table(data: dict[str, Any]) -> pl.DataFrame:
    """Nested ``"<f> GHz"/"<t> C"/<columns>`` dicts -> long table."""
    frames = []
    for freq_key, temperatures in data.items():
        for temp_key, columns in temperatures.items():
            n = len(columns["Control Voltage [V]"])
            frame = {
                "frequency_ghz": [float(freq_key.split(" GHz")[0])] * n,
                "temperature_c": [float(temp_key.split(" C")[0])] * n,
            }
            for old, new in _LEGACY_RESPONSIVITY.items():
                if old in columns:
                    frame[new] = np.asarray(columns[old], dtype=float)
            frames.append(pl.DataFrame(frame))
    return pl.concat(frames, how="diagonal").sort(
        ["frequency_ghz", "temperature_c", "control_voltage_v"]
    )
