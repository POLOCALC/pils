import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import os
import glob
import yaml as yaml_module

from pathlib import Path

from ..utils.tools import get_logpath_from_datapath, is_ascii_file

class LM76:
    def __init__(self,path: Path, logpath: Path | None = None) -> None:
        """
        Initialize LM76 temperature sensor.

        Parameters
        ----------
        path : Path
            Directory containing LM76 csv file.
        logpath : Optional[Path], default=None
            Optional path to log file. If None, will be inferred.
        """

        self.path = path
        self.data_path = None
        self.data = None
        
        files = list(path.glob("*"))
        self.data_path = next((f for f in files if f.match("*TMP*.csv")), None)

        if logpath is not None:
            self.logpath = logpath

        else:
            try:
                self.logpath = get_logpath_from_datapath(self.data_path)
            except FileNotFoundError:
                self.logpath = None

    def _read_settings_from_config(self):
        """
        Read lm76 settings (interval,tcrit,thyst,tlow,thigh) from onfig.yml file in the same directory.

        Returns
        -------
        np.array()
                    lm76 settings: interval,tcrit,thyst,tlow,thigh
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
        """Load LM76 data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Sensor file not found: {self.data_path}")

        lm76_data = pl.read_csv(self.data_path)
        lm76_data = lm76_data.with_columns([(pl.col("timestamp_ns")*1e-9).alias("timestamp")]) # correct units of time

        lm76_settings = self._read_settings_from_config()

        self.data = {}
        self.data["metadata"] = lm76_settings
        self.data["data"] = lm76_data      
        
    def plot(self):
        """Plot LM76 data.

        Raises
        ------
        ValueError
            If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("LM76 Termostat")
        axs[0].plot(self.data["timestamp"], self.data["temperature_c"].values)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel(r"Temperature [$^\circ$C]")
  
        axs[1].scatter(self.data["timestamp"], self.data["status_crit"].values, label="status TCrit")
        axs[1].scatter(self.data["timestamp"], self.data["status_high"].values, label="status THigh")
        axs[1].scatter(self.data["timestamp"], self.data["status_low"].values, label="status TLow")
        axs[1].legend()

        plt.tight_layout()
        plt.show()