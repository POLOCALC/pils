import glob
import os
import struct
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Check if yaml is available for config file reading
try:
    import yaml  # noqa: F401

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..utils.tools import get_logpath_from_datapath, is_ascii_file

ADS1015_VALUE_GAIN = {
    1: 4.096,
    2: 2.048,
    4: 1.024,
    8: 0.512,
    16: 0.256,
}


def decode_adc_file_struct(adc_path):
    """
    Decodes the old ADC file written in a structured binary format and returns its content as a polars DataFrame.

    Parameters
    ----------
    adc_path : str
        Path to the ADC file to be decoded.

    Returns
    -------
    adc_data : polars.DataFrame
        DataFrame with columns ["time", "reading_time", "amplitude", "datetime"].
        Time is the timestamp in seconds since the epoch, reading_time is the time
        it took to take the measurement, amplitude is the measurement itself, and
        datetime is the converted timestamp in datetime format.
    """

    pattern = "dqf"

    with open(adc_path, "rb") as f:
        data = f.read()

    reps = int(len(data) / 20)
    vals = struct.unpack("<" + pattern * reps, data)
    vals = np.reshape(np.array(vals), (reps, 3))
    adc_data = pl.DataFrame(
        {"timestamp": vals[:, 0], "reading_time": vals[:, 1], "amplitude": vals[:, 2]}
    )
    adc_data = adc_data.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime")
    )
    return adc_data


def decode_adc_file_ascii(adc_path, gain_config=16):
    """
    Decodes the last version of ADC file written in ASCII format and returns its content as a polars DataFrame.

    Parameters
    ----------
    adc_path : str
        Path to the ADC file to be decoded.

    Returns
    -------
    adc_data : polars.DataFrame
        DataFrame containing tuples of (timestamp, value) where timestamp is the
        time in seconds since the epoch and value is the corresponding measurement.
    """

    gain = ADS1015_VALUE_GAIN[gain_config]  # Need to be tracked from the config file

    timestamps = []
    amplitudes = []
    with open(adc_path, "rb") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        try:
            # Decode the line from bytes to ASCII and split
            text_line = line.decode("ascii").strip()

            # Skip empty lines
            if not text_line:
                continue

            parts = text_line.split()
            # Expect at least 2 values: timestamp and amplitude
            if len(parts) < 2:
                # Silently skip incomplete lines (e.g., EOF without newline)
                continue

            timestamp = int(parts[0])
            value = int(parts[1])
            timestamps.append(timestamp)
            amplitudes.append(value)
        except ValueError as e:
            print(f"Warning: Line {line_num} could not be parsed as integers: {e}")
            continue
        except Exception as e:
            print(f"Error decoding line {line_num}: {e}")
            continue

    adc_data = pl.DataFrame({"timestamp": timestamps, "amplitude": amplitudes})
    adc_data = adc_data.with_columns(
        [
            (pl.col("amplitude") * gain / 2048 * 1e3).alias("amplitude"),
            (pl.col("timestamp") / 1e6).alias("timestamp"),  # convert from us to seconds
        ]
    )

    return adc_data


class ADC:
    def __init__(self, path, logpath=None, gain_config=None):
        """
        Initialize ADC sensor.

        Args:
            path: Path to ADC data file
            logpath: Path to log file (optional)
            gain_config: ADC gain configuration (1, 2, 4, 8, or 16).
                        If None, attempts to read from config.yml in the same folder.
                        Defaults to 16 if not found.
        """
        files = list(path.glob("*"))

        for f in files:
            if f.name.lower().endswith("adc.bin"):
                self.data_path = f

        self.data = None

        if logpath is not None:
            self.logpath = logpath
        else:
            self.logpath = get_logpath_from_datapath(self.data_path)

        self.tstart = None

        with open(self.data_path, "rb") as f:
            data = f.read()
            self.is_ascii = is_ascii_file(data)
            f.close()

        # Auto-detect gain from config file if not provided
        if gain_config is None:
            gain_config = self._read_gain_from_config()

        self.gain_config = gain_config
        self.gain = ADS1015_VALUE_GAIN[gain_config]

    def _read_gain_from_config(self) -> int:
        """
        Read ADC gain from config.yml file in the same directory.

        Returns:
            Gain configuration value (defaults to 16 if not found)
        """
        if not YAML_AVAILABLE:
            return 16

        # Import yaml locally to avoid Pylance unbound error
        import yaml as yaml_module

        # Look for config file in the same directory as the ADC file
        adc_dir = os.path.dirname(self.data_path)
        config_files = glob.glob(os.path.join(adc_dir, "*_config.yml"))

        if not config_files:
            # Also try just config.yml
            config_path = os.path.join(adc_dir, "config.yml")
            if os.path.exists(config_path):
                config_files = [config_path]

        if config_files:
            try:
                with open(config_files[0], "r") as f:
                    config = yaml_module.safe_load(f)

                # Navigate to sensors.ADC_1.configuration.gain
                sensors = config.get("sensors", {})
                for sensor_name, sensor_config in sensors.items():
                    if sensor_name.startswith("ADC"):
                        gain = sensor_config.get("configuration", {}).get("gain")
                        if gain is not None:
                            return int(gain)
            except Exception:
                pass

        return 16  # Default gain

    def load_data(self):
        if self.is_ascii:
            self.data = decode_adc_file_ascii(self.data_path, self.gain_config)
        else:
            self.data = decode_adc_file_struct(self.data_path)

    def plot(self):
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        plt.figure(figsize=(10, 5))
        plt.plot(self.data["timestamp"], self.data["amplitude"], color="crimson")
        plt.ylabel("ADC amplitude [mV]")
        plt.xlabel("Time [s]")
        plt.show()
