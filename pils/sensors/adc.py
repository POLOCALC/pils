"""ADC readings of the payload detector (ADS1015/ADS1115).

Notes
-----
The format is detected from the data; amplitudes are in mV after loading
(``amplitude_mv`` in the schema) whatever the file stores.

| Payload | File | Content |
|---------|------|---------|
| 2025 | ``*_ADC.bin``, ASCII | time (us) and ADS1015 counts per line; mV with the gain of ``config.yml`` |
| December 2024 | ``*_ADC.bin``, 20-byte records | time (s), reading time (us), amplitude (mV) |
| April 2024 | ``*_ADC.bin``, 12-byte records | time (s), amplitude (V) |
| 2022, 2023 | ``*_ADC*.pck`` | see ``pils.sensors.legacy`` |

A record cut at the end of a binary file (power loss) and preallocated
zeros are dropped. Times are those of the payload computer's clock.
"""

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml as yaml_module

from ..utils.logging_config import get_logger
from ..utils.tools import get_logpath_from_datapath, is_ascii_file

logger = get_logger(__name__)

ADS1015_VALUE_GAIN = {
    1: 4.096,
    2: 2.048,
    4: 1.024,
    8: 0.512,
    16: 0.256,
}


# Binary ADC records: time (s), [reading time (us),] amplitude
STRUCT_LAYOUTS = {
    "dqf": [("timestamp", "<f8"), ("reading_time", "<i8"), ("amplitude", "<f4")],
    "df": [("timestamp", "<f8"), ("amplitude", "<f4")],  # April 2024, volts
}


def detect_struct_layout(data: bytes) -> str:
    """Layout of a binary ADC file: the one whose times advance steadily."""
    best, best_score = "dqf", -1.0
    for name, fields in STRUCT_LAYOUTS.items():
        record = np.dtype(fields)
        n = min(len(data) // record.itemsize, 2000)
        if n < 3:
            continue
        times = np.frombuffer(data[: n * record.itemsize], record)["timestamp"]
        steps = np.diff(times)
        with np.errstate(invalid="ignore"):
            plausible = (times > 1e9) & (times < 4e9)
            score = float(np.mean(plausible[1:] & (steps >= 0) & (steps < 1.0)))
        if score > best_score:
            best, best_score = name, score
    return best


def decode_adc_file_struct(adc_path: str | Path) -> pl.DataFrame:
    """
    Decode a binary ADC file (fixed-size records) into a DataFrame.

    Parameters
    ----------
    adc_path : Union[str, Path]
        Path to the ADC file to be decoded.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ["time", "reading_time", "amplitude", "datetime"].
        Time is the timestamp in seconds since the epoch, reading_time is the time
        it took to take the measurement, amplitude is the measurement itself, and
        datetime is the converted timestamp in datetime format.
    """
    with open(adc_path, "rb") as f:
        data = f.read()

    layout = detect_struct_layout(data)
    record = np.dtype(STRUCT_LAYOUTS[layout])
    # A record cut at the end (power loss) and preallocated zeros are dropped
    values = np.frombuffer(
        data[: len(data) // record.itemsize * record.itemsize], record
    )
    values = values[np.isfinite(values["timestamp"]) & (values["timestamp"] > 1.0)]
    columns = {"timestamp": values["timestamp"].astype(float)}
    if "reading_time" in (record.names or ()):
        columns["reading_time"] = values["reading_time"]
    amplitude = values["amplitude"].astype(float)
    # April 2024 (12-byte records) stored volts; the later format millivolts
    columns["amplitude"] = amplitude * 1e3 if layout == "df" else amplitude
    adc_data = pl.DataFrame(columns)
    adc_data = adc_data.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime")
    )
    return adc_data


def decode_adc_file_ascii(adc_path: str | Path, gain_config: int = 16) -> pl.DataFrame:
    """
    Decode an ASCII ADC file (time and counts per line) into a DataFrame.

    Parameters
    ----------
    adc_path : Union[str, Path]
        Path to the ADC file to be decoded.
    gain_config : int, optional
        ADC gain configuration (1, 2, 4, 8, or 16). Defaults to 16.

    Returns
    -------
    pl.DataFrame
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
            logger.warning(f"Line {line_num} could not be parsed as integers: {e}")
            continue
        except Exception as e:
            logger.error(f"Error decoding line {line_num}: {e}")
            continue

    adc_data = pl.DataFrame({"timestamp": timestamps, "amplitude": amplitudes})
    adc_data = adc_data.with_columns(
        [
            (pl.col("amplitude") * gain / 2048 * 1e3).alias("amplitude"),
            (pl.col("timestamp") / 1e6).alias(
                "timestamp"
            ),  # convert from us to seconds
        ]
    )

    return adc_data


class ADC:
    """ADC sensor of the payload; the file format is detected (see the module)."""

    def __init__(
        self, path: Path, logpath: str | None = None, gain_config: int | None = None
    ) -> None:
        """
        Initialize ADC sensor.

        Parameters
        ----------
        path : Path
            Path to ADC data file directory.
        logpath : Optional[str], optional
            Path to log file (optional).
        gain_config : Optional[int], optional
            ADC gain configuration (1, 2, 4, 8, or 16).
            If None, attempts to read from config.yml in the same folder.
            Defaults to 16 if not found.
        """
        path = Path(path)
        files = sorted(f for f in path.glob("*") if f.name.lower().endswith("adc.bin"))
        legacy = sorted(path.glob("*_ADC.pck")) + sorted(path.glob("*_ADC_*.pck"))
        if not files and not legacy:
            raise FileNotFoundError(f"No ADC file (*_ADC.bin, *_ADC*.pck) in {path}")
        # 2022 and 2023 payloads: pickle files (see pils.sensors.legacy)
        self.legacy_folder = path if not files else None
        self.data_path = files[-1] if files else legacy[0]

        self.data: pl.DataFrame | None = None
        self.tstart = None

        # Handle logpath
        if logpath is not None:
            self.logpath = logpath
        else:
            try:
                self.logpath = get_logpath_from_datapath(self.data_path)
            except FileNotFoundError:
                self.logpath = None

        with open(self.data_path, "rb") as f:
            self.is_ascii = self.legacy_folder is None and is_ascii_file(f.read())

        # Auto-detect gain from config file if not provided
        if gain_config is None:
            gain_config = self._read_gain_from_config()

        self.gain_config = gain_config
        self.gain = ADS1015_VALUE_GAIN[gain_config]

    def _read_gain_from_config(self) -> int:
        """
        Read ADC gain from config.yml file in the same directory.

        Returns
        -------
        int
            Gain configuration value (defaults to 16 if not found).
        """
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
                with open(config_files[0]) as f:
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

    def load_data(self) -> None:
        """Load ADC data from file (auto-detects ASCII or binary format)."""
        if self.legacy_folder is not None:
            from .legacy import read_legacy_adc

            self.data = read_legacy_adc(self.legacy_folder)
        elif self.is_ascii:
            self.data = decode_adc_file_ascii(self.data_path, self.gain_config)
        else:
            self.data = decode_adc_file_struct(self.data_path)

    def plot(self) -> None:
        """Plot ADC amplitude vs time.

        Raises
        ------
        ValueError
            If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        plt.figure(figsize=(10, 5))
        plt.plot(self.data["timestamp"], self.data["amplitude"], color="crimson")
        plt.ylabel("ADC amplitude [mV]")
        plt.xlabel("Time [s]")
        plt.show()
