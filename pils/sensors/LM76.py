import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import yaml as yaml_module

from ..utils.tools import get_logpath_from_datapath


class LM76:
    def __init__(self, path: Path, logpath: Path | None = None) -> None:
        """
        Initialize LM76 temperature sensor.

        Parameters
        ----------
        path : Path
            Directory containing the LM76 CSV data file (``*TMP*.csv``).
        logpath : Path, optional
            Path to the log file. If None, the logpath is inferred from the
            data file path. Defaults to None.
        """

        self.path = path
        self.data_path = None
        self.data = None

        files = list(path.glob("*"))
        self.data_path = next((f for f in files if f.match("*TMP*.csv")), None)

        if logpath is not None:
            self.logpath = logpath
        elif self.data_path is not None:
            try:
                self.logpath = get_logpath_from_datapath(self.data_path)
            except FileNotFoundError:
                self.logpath = None
        else:
            self.logpath = None

    def _read_settings_from_config(self):
        """
        Read LM76 settings from the ``*_config.yml`` file in the parent directory.

        Looks for a YAML config file alongside the sensor directory and extracts
        the configuration block (interval, tcrit, thyst, tlow, thigh) for the
        first sensor key starting with ``LM76``.

        Returns
        -------
        dict or None
            Dictionary of LM76 configuration settings, or None if no config
            file is found or if the config does not contain an LM76 sensor
            section.
        """

        # Look for config file in the same directory as the ADC file
        lm76_dir = os.path.dirname(self.path)
        config_files = glob.glob(os.path.join(lm76_dir, "*_config.yml"))

        if config_files:
            try:
                with open(config_files[0]) as f:
                    config = yaml_module.safe_load(f)

                # Navigate to LM76 sensor:
                sensors = config.get("sensors", {})
                for sensor_name, sensor_config in sensors.items():
                    if sensor_name.startswith("LM76"):
                        lm76_settings = sensor_config.get("configuration", {})
                        if lm76_settings is not None:
                            return lm76_settings

            except Exception:
                pass

    def load_data(self) -> None:
        """
        Load LM76 sensor data from CSV file.

        Reads the CSV file at ``self.data_path``, converts the ``timestamp_ns``
        column from nanoseconds to seconds (stored as ``timestamp``), and reads
        sensor configuration from the config file. Populates ``self.data`` as a
        dictionary with keys ``'data'`` (a :class:`polars.DataFrame`) and
        ``'metadata'`` (configuration dict or None).

        Raises
        ------
        FileNotFoundError
            If the CSV file at ``self.data_path`` does not exist.
        AttributeError
            If ``self.data_path`` is None (no matching CSV file was found
            during initialization).
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Sensor file not found: {self.data_path}")

        lm76_data = pl.read_csv(self.data_path)
        lm76_data = lm76_data.with_columns(
            [(pl.col("timestamp_ns") * 1e-9).alias("timestamp")]
        )  # correct units of time

        lm76_settings = self._read_settings_from_config()

        self.data = {}
        self.data["metadata"] = lm76_settings
        self.data["data"] = lm76_data

    def plot(self) -> None:
        """
        Plot LM76 temperature and thermostat status data.

        Displays a two-panel figure: temperature over time (top) and scatter
        plots of the critical, high, and low status flags (bottom).

        Raises
        ------
        ValueError
            If no data has been loaded. Call :meth:`load_data` first.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        fig, axs = plt.subplots(2, 1)
        fig.suptitle("LM76 Termostat")
        axs[0].plot(self.data["timestamp"], self.data["temperature_c"].values)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel(r"Temperature [$^\circ$C]")

        axs[1].scatter(
            self.data["timestamp"],
            self.data["status_crit"].values,
            label="status TCrit",
        )
        axs[1].scatter(
            self.data["timestamp"],
            self.data["status_high"].values,
            label="status THigh",
        )
        axs[1].scatter(
            self.data["timestamp"], self.data["status_low"].values, label="status TLow"
        )
        axs[1].legend()

        plt.tight_layout()
        plt.show()
