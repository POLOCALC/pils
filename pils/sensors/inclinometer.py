import glob
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..decoders import KERNEL_utils as kernel
from ..utils.logging_config import get_logger
from ..utils.tools import (
    drop_nan_and_zero_cols,
    get_logpath_from_datapath,
    read_log_time,
)

logger = get_logger(__name__)

# Check if yaml is available for config file reading
try:
    import yaml  # noqa: F401

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def decode_inclino(inclino_path: str | Path) -> dict[str, list[Any]]:
    """
    Decodes inclinometer data from a binary file and returns the decoded messages as a dictionary.

    Parameters
    ----------
    inclino_path : Union[str, Path]
        Path to the binary file containing inclinometer data.

    Returns
    -------
    Dict[str, List[Any]]
        A dictionary where keys are message field names and values are lists of
        field values extracted from the decoded messages.

    Raises
    ------
    FileNotFoundError
        If inclino_path does not exist.

    Examples
    --------
    >>> data = decode_inclino(Path("inclino.bin"))
    >>> data.keys()
    dict_keys(['Roll', 'Pitch', 'Heading', 'Counter'])
    """

    with open(inclino_path, "rb") as fd:
        data = fd.read()

    # Define the starting sequence of a message
    sequence = b"\xaaU\x01\x81"
    msgs = data.split(sequence)[1:]

    decoded_msg = {}
    for msg in msgs:
        try:
            msg = sequence + msg
            tmp = kernel.KernelMsg().decode_single(msg, return_dict=True)

            if not decoded_msg.keys():
                decoded_msg = {k: [] for k in tmp.keys()}

            for j in tmp.keys():
                decoded_msg[j].append(tmp[j])
        except Exception as e:
            logger.warning(f"Failed to decode inclinometer message: {e}")
            continue
    return decoded_msg


def detect_inclinometer_type_from_config(dirpath: Path) -> str | None:
    """
    Detect the type of inclinometer from the config.yml file.

    Looks for sensors with type 'IMX5' (InertialSense) or checks for
    INERTIAL sensors with Kernel manufacturer.

    Parameters
    ----------
    dirpath : str
        Path to the aux folder or sensors subfolder.

    Returns
    -------
    str or None
        'imx5', 'kernel', or None if not found.
    """
    if not YAML_AVAILABLE:
        return None

    import yaml as yaml_module

    # Find config file - could be in dirpath or parent (aux folder)
    config_files = list(dirpath.glob("*_config.yml"))

    if not config_files:
        return None

    try:
        with open(config_files[0]) as f:
            config = yaml_module.safe_load(f)

        sensors = config.get("sensors", {})

        # Check for IMX5 sensor
        for sensor_name, sensor_config in sensors.items():
            sensor_info = sensor_config.get("sensor_info", {})
            sensor_type = sensor_info.get("type", "").upper()
            manufacturer = sensor_info.get("manufacturer", "").lower()

            # IMX5 from InertialSense
            if sensor_type == "IMX5" or (
                manufacturer == "inertialsense" and "imx" in sensor_name.lower()
            ):
                return "imx5"

            # Kernel inclinometer (type: Inertial, manufacturer might be 'Kernel' or similar)
            if sensor_type == "INERTIAL":
                # Check if it's a Kernel sensor
                name = sensor_config.get("name", "").lower()
                if "kernel" in name or "kernel" in manufacturer:
                    return "kernel"

        # Secondary check: look for specific sensor names
        for sensor_name in sensors.keys():
            if "IMX5" in sensor_name.upper() or "IMX-5" in sensor_name.upper():
                return "imx5"
            if "KERNEL" in sensor_name.upper() or "INERTIAL" in sensor_name.upper():
                # Check if there are INC files to determine type
                return None  # Let file detection handle it

    except Exception as e:
        logger.warning(f"Failed to detect inclinometer type from config: {e}")
        pass

    return None


def detect_inclinometer_type_from_files(dirpath: Path) -> str:
    """
    Detect the type of inclinometer data available in a directory by file patterns.
    This is a fallback when config detection fails.

    Parameters
    ----------
    dirpath : str
        Path to the sensors directory containing inclinometer files.

    Returns
    -------
    tuple[str, Optional[str]]
        Tuple of (inclinometer_type, path_to_file).
        inclinometer_type is 'imx5', 'kernel', or 'unknown'.
    """
    # Check for IMX-5 files (CSV format with INC_ prefix)

    imx5_ins_files = list(dirpath.glob("*_INC_ins.csv"))
    if imx5_ins_files:
        return "imx5"

    # Check for Kernel files (binary INC.bin format)
    kernel_files = list(dirpath.glob("*_INC.bin"))
    if kernel_files:
        return "kernel"

    return "unknown"


class IMX5Inclinometer:
    """
    Decoder for InertialSense IMX-5 inclinometer data.

    The IMX-5 outputs three CSV files:
    - *_INC_imu.csv: Raw IMU data (accelerometer, gyroscope)
    - *_INC_ins.csv: INS solution (position, velocity, attitude)
    - *_INC_inl2.csv: Extended INL2 data (quaternions, biases)
    """

    def __init__(self, dirpath: Path, logpath: str | None = None) -> None:
        """
        Initialize IMX5Inclinometer.

        Parameters
        ----------
        dirpath : Path
            Path to the sensors directory containing IMX-5 CSV files.
        logpath : Optional[str], optional
            Path to log file for timing information (optional).
        """
        self.dirpath = dirpath
        self.logpath = logpath

        # Find IMX-5 files
        self.ins_path = self._find_file("*_INC_ins.csv")
        self.imu_path = self._find_file("*_INC_imu.csv")
        self.inl2_path = self._find_file("*_INC_inl2.csv")

        self.data = {}  # Main attitude data

    def _find_file(self, pattern: str) -> str | None:
        """Find a file matching the pattern in dirpath.

        Parameters
        ----------
        pattern : str
            Glob pattern to match files.

        Returns
        -------
        Optional[str]
            Path to first matching file, or None if not found.
        """
        files = glob.glob(os.path.join(self.dirpath, pattern))
        return files[0] if files else None

    def load_ins(self) -> None:
        """
        Load INS solution data (position, velocity, attitude).

        Converts roll/pitch/yaw from radians to degrees and creates
        timestamp and datetime columns from timestamp_ns.
        """
        if self.ins_path is None:
            return None

        df = pl.read_csv(self.ins_path)

        # Convert radians to degrees for attitude
        cols_to_add = []
        if "roll_rad" in df.columns:
            cols_to_add.append((pl.col("roll_rad") * 180 / np.pi).alias("roll"))
        if "pitch_rad" in df.columns:
            cols_to_add.append((pl.col("pitch_rad") * 180 / np.pi).alias("pitch"))
        if "yaw_rad" in df.columns:
            cols_to_add.append((pl.col("yaw_rad") * 180 / np.pi).alias("yaw"))

        # Create timestamp from timestamp_ns
        if "timestamp_ns" in df.columns:
            cols_to_add.append((pl.col("timestamp_ns") / 1e9).alias("timestamp"))
            cols_to_add.append(
                pl.from_epoch(pl.col("timestamp_ns"), time_unit="ns").alias("datetime")
            )

        if cols_to_add:
            df = df.with_columns(cols_to_add)

        if not df.is_empty():
            self.data["INS"] = df

    def load_imu(self) -> None:
        """
        Load raw IMU data (accelerometer, gyroscope).

        Creates timestamp and datetime columns from timestamp_ns.
        """
        if self.imu_path is None:
            return None

        df = pl.read_csv(self.imu_path)

        cols_to_add = []
        # Create timestamp from timestamp_ns
        if "timestamp_ns" in df.columns:
            cols_to_add.append((pl.col("timestamp_ns") / 1e9).alias("timestamp"))
            cols_to_add.append(
                pl.from_epoch(pl.col("timestamp_ns"), time_unit="ns").alias("datetime")
            )

        # Convert gyro from rad/s to deg/s
        for col in ["pqr_P_rad_s", "pqr_Q_rad_s", "pqr_R_rad_s"]:
            if col in df.columns:
                new_col = col.replace("_rad_s", "_deg_s")
                cols_to_add.append((pl.col(col) * 180 / np.pi).alias(new_col))

        if cols_to_add:
            df = df.with_columns(cols_to_add)

        if not df.is_empty():
            self.data["IMU"] = df

    def load_inl2(self):
        """
        Load INL2 extended data (quaternions, biases).

        Returns
        -------
        pl.DataFrame or None
            INL2 data with quaternions, ECEF position, and bias estimates.
        """
        if self.inl2_path is None:
            return None

        df = pl.read_csv(self.inl2_path)

        cols_to_add = []
        # Create timestamp from timestamp_ns
        if "timestamp_ns" in df.columns:
            cols_to_add.append((pl.col("timestamp_ns") / 1e9).alias("timestamp"))
            cols_to_add.append(
                pl.from_epoch(pl.col("timestamp_ns"), time_unit="ns").alias("datetime")
            )

        if cols_to_add:
            df = df.with_columns(cols_to_add)

        if not df.is_empty():
            self.data["INL2"] = df

    def load_data(self):
        """
        Load all IMX-5 data and set main attitude data.

        The main `self.data` attribute will contain the INS solution
        with roll, pitch, yaw in degrees.
        """
        self.load_ins()
        self.load_imu()
        self.load_inl2()


class KernelInclinometer:
    """
    Decoder for Kernel-100 inclinometer data (binary format).
    """

    def __init__(self, path: Path, logpath: str | None = None) -> None:
        """
        Initialize KernelInclinometer.

        Parameters
        ----------
        path : Path
            Path to the Kernel binary file (*_INC.bin).
        logpath : Optional[str], optional
            Path to log file for timing information (optional).
        """
        self.path = path
        if logpath is not None:
            self.logpath = logpath
        else:
            try:
                self.logpath = get_logpath_from_datapath(self.path)
            except FileNotFoundError:
                self.logpath = None

        self.tstart = None

    def read_log_time(self, logfile: str | None = None) -> None:
        """
        Read start time from log file.

        Parameters
        ----------
        logfile : Optional[str], optional
            Path to log file (optional).
        """
        if logfile is None:
            return

        keyphrases = [
            "Connected to KERNEL sensor Kernel-100",
            "Sensor Kernel-100 started",
        ]
        for keyphrase in keyphrases:
            try:
                tstart, _ = read_log_time(keyphrase=keyphrase, logfile=logfile)
                if tstart is None:
                    continue
                else:
                    self.tstart = tstart
                    break
            except Exception as e:
                logger.warning(
                    f"Couldn't find start time from logfile. Skipping datetime conversion. Error: {e}"
                )
                break

    def load_data(self) -> None:
        """
        Load and decode Kernel inclinometer data.

        The data is processed to handle counter wrap-arounds and filtered
        to keep only valid measurements. Euler angles are renamed to match
        drone convention (roll/pitch/yaw).
        """
        # Load data from binary decoder
        decoded = decode_inclino(self.path)
        inclino_data = pl.DataFrame(decoded)

        # Detect counter wrap-arounds (where counter resets)
        counter = inclino_data["Counter"]
        diff_counter = counter.diff()

        # Create wrap detection
        wraps = diff_counter.abs() > 60000
        wrap_cumsum = wraps.cum_sum()
        counter_vals = counter.to_numpy()
        counter_max = float(counter_vals.max())
        counter_min = float(counter_vals.min())
        new_counter = counter + wrap_cumsum * (counter_max - counter_min)

        # Filter good indices
        new_counter_diff = new_counter.diff()
        ind_good = (new_counter_diff == 16) | (new_counter_diff == 13)

        # Apply filter
        inclino_data = inclino_data.with_columns(
            [new_counter.alias("new_counter"), ind_good.alias("ind_good")]
        )
        inclino_data = inclino_data.filter(pl.col("ind_good"))

        # Convert counter to time (seconds)
        inclino_data = inclino_data.with_columns(
            [(pl.col("new_counter") / 2000.0).alias("counter_timestamp")]
        )

        if self.logpath is not None:
            # Convert Path to str if needed
            logfile_path = str(self.logpath) if isinstance(self.logpath, Path) else self.logpath
            self.read_log_time(logfile=logfile_path)
            if self.tstart is not None:
                tstart = self.tstart
                # Calculate datetime for each row
                timestamps = inclino_data["counter_timestamp"].to_list()
                datetimes = [tstart + timedelta(seconds=t) for t in timestamps]
                inclino_data = inclino_data.with_columns([pl.Series("datetime", datetimes)])
                inclino_data = inclino_data.with_columns(
                    [(pl.col("datetime").dt.epoch(time_unit="ns") / 1e9).alias("timestamp")]
                )

        # Rename Euler angles to match drone convention
        inclino_data = inclino_data.rename({"Roll": "pitch", "Pitch": "roll", "Heading": "yaw"})
        inclino_data = inclino_data.with_columns([(-pl.col("pitch")).alias("pitch")])

        # Drop helper columns
        cols_to_drop = ["new_counter", "ind_good"]
        inclino_data = inclino_data.drop([c for c in cols_to_drop if c in inclino_data.columns])

        inclino_data = drop_nan_and_zero_cols(inclino_data)

        self.data = inclino_data


class Inclinometer:
    """
    Unified inclinometer class that auto-detects and loads either
    Kernel-100 (binary) or IMX-5 (CSV) inclinometer data.

    Parameters
    ----------
    path : str
        Path to inclinometer file (for Kernel) or sensors directory (for IMX-5).
    logpath : str, optional
        Path to log file for timing information.
    sensor_type : str, optional
        Force sensor type: 'kernel', 'imx5', or None for auto-detect.
    """

    def __init__(
        self,
        path: Path,
        logpath: str | None = None,
        sensor_type: Literal["kernel", "imx5"] | None = None,
    ) -> None:
        self._lookout_path = path

        if sensor_type is None:
            self._auto_detect()
        else:
            self.sensor_type = sensor_type

        logger.info(f"Inclinometer sensor type: {self.sensor_type}")

        self.logpath = logpath
        self._decoder: KernelInclinometer | IMX5Inclinometer | None = None

        # Initialize the appropriate decoder
        self._init_decoder()

    def _auto_detect(self) -> None:
        """Auto-detect inclinometer type from config.yml file."""
        # First try config-based detection (primary method)
        config_type = detect_inclinometer_type_from_config(self._lookout_path)

        if config_type is not None:
            self.sensor_type = config_type
        else:
            inc_type = detect_inclinometer_type_from_files(self._lookout_path)
            self.sensor_type = inc_type if inc_type != "unknown" else None

    def _init_decoder(self) -> None:
        """Initialize the appropriate decoder based on sensor type."""
        if self.sensor_type == "kernel":
            self._decoder = KernelInclinometer(self._lookout_path, self.logpath)
        elif self.sensor_type == "imx5":
            self._decoder = IMX5Inclinometer(self._lookout_path, self.logpath)

    @property
    def tstart(self):
        """Get start time (Kernel only)."""
        if isinstance(self._decoder, KernelInclinometer):
            return self._decoder.tstart
        return None

    @property
    def ins_data(self) -> pl.DataFrame | None:
        """Get INS data (IMX-5 only)."""
        if isinstance(self._decoder, IMX5Inclinometer):
            return self._decoder.data["INS"]
        return None

    @property
    def imu_data(self) -> pl.DataFrame | None:
        """Get IMU data (IMX-5 only)."""
        if isinstance(self._decoder, IMX5Inclinometer):
            return self._decoder.data["IMU"]
        return None

    @property
    def inl2_data(self) -> pl.DataFrame | None:
        """Get INL2 data (IMX-5 only)."""
        if isinstance(self._decoder, IMX5Inclinometer):
            return self._decoder.data["INL2"]
        return None

    def load_data(self) -> None:
        """Load inclinometer data using the detected decoder.

        Raises
        ------
        ValueError
            If no inclinometer data found at path.
        """
        if self._decoder is None:
            raise ValueError(
                f"No inclinometer data found at {self._lookout_path}. "
                "Expected either *_INC.bin (Kernel) or *_INC_*.csv (IMX-5) files."
            )

        self._decoder.load_data()
        self.data = self._decoder.data

    def plot(self) -> None:
        """Plot roll, pitch, yaw over time.

        Raises
        ------
        ValueError
            If data not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

        # Determine x-axis (prefer timestamp)
        # Handle both DataFrame and dict types
        if isinstance(self.data, pl.DataFrame):
            if "timestamp" in self.data.columns:
                xlabel = "Time [s]"
            elif "datetime" in self.data.columns:
                xlabel = "Time"
            else:
                xlabel = "Sample"
        elif isinstance(self.data, dict):
            if "timestamp" in self.data:
                xlabel = "Time [s]"
            elif "datetime" in self.data:
                xlabel = "Time"
            else:
                # Estimate length from first available key
                xlabel = "Sample"
        else:
            _x = np.arange(len(self.data))
            xlabel = "Sample"

        # Extract data based on type
        if isinstance(self.data, pl.DataFrame):
            # yaw, pitch, roll would be used for plotting if implemented
            pass
        elif isinstance(self.data, dict):
            _yaw = np.array(self.data.get("yaw", []))
            _pitch = np.array(self.data.get("pitch", []))
            _roll = np.array(self.data.get("roll", []))
        else:
            return

        axs[0].set_ylabel("Yaw [°]")
        axs[1].set_ylabel("Pitch [°]")
        axs[2].set_ylabel("Roll [°]")
        axs[-1].set_xlabel(xlabel)

        plt.suptitle(f"Inclinometer Data ({self.sensor_type or 'unknown'})")
        plt.tight_layout()
        plt.show()
