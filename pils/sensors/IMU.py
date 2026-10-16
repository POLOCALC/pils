import io
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Columns after the timestamp (microseconds since the epoch) in each file
SENSOR_COLUMNS = {
    "baro": ["pressure", "temperature"],
    "magneto": ["mag_x", "mag_y", "mag_z"],
    "accelero": ["acc_x", "acc_y", "acc_z"],
    "gyro": ["gyro_x", "gyro_y", "gyro_z"],
}

# File name patterns in the sensors folder, newest first
SENSOR_FILES = {
    "baro": ("*_BAR.bin", "barometer.bin"),
    "magneto": ("*_MAG.bin", "magnetometer.bin"),
    "accelero": ("*_ACC.bin", "accelerometer.bin"),
    "gyro": ("*_GYR.bin", "gyroscope.bin"),
}

# Shorter than this, a sensor is reported as stopped early
SHORT_RECORDING_S = 60.0


def _find_file(dirpath: Path, sensor_type: str) -> Path | None:
    for pattern in SENSOR_FILES[sensor_type]:
        matches = sorted(Path(dirpath).glob(pattern))
        if matches:
            return matches[0]
    return None


class IMUSensor:
    """
    Individual IMU sensor (barometer, accelerometer, gyroscope, or magnetometer).

    Each file is text, one sample per line: the timestamp in microseconds
    since the epoch, then the values (see ``SENSOR_COLUMNS``). Accelerations
    are in g, pressure in hPa and temperature in deg C; gyroscope and
    magnetometer values are as logged.

    Attributes
    ----------
    path : Path or None
        Path to sensor data file (None if the payload has no such file).
    type : str
        Sensor type (baro, accelero, gyro, magneto).
    data : Optional[pl.DataFrame]
        Polars DataFrame with sensor data (None until load_data is called,
        or if the file is missing or empty).
    """

    def __init__(self, path: str | Path | None, sensor_type: str) -> None:
        """Initialize IMU sensor.

        Parameters
        ----------
        path : Union[str, Path] or None
            Path to sensor file.
        sensor_type : str
            Sensor type (baro, accelero, gyro, magneto).
        """
        self.path = Path(path) if path is not None else None
        self.type = sensor_type
        self.data: pl.DataFrame | None = None

    def load_data(self) -> None:
        """
        Load sensor data from the file.

        Sets the `data` attribute with a polars DataFrame containing:

        - ``timestamp``: seconds since the epoch
        - ``timestamp_us``: microseconds since the epoch, as logged
        - the sensor columns (defined in SENSOR_COLUMNS)
        - ``datetime``: UTC

        A recording that stopped uncleanly ends with zero bytes or a cut
        line; these, and any other malformed line, are dropped. Barometer
        samples with zero pressure (the first one) are dropped too.
        """
        self.data = None
        if self.path is None or not self.path.exists():
            return
        raw = self.path.read_bytes()
        end = raw.find(b"\x00")
        if end >= 0:
            raw = raw[:end]
        # The last line is complete only if it ends with a newline
        raw = raw[: raw.rfind(b"\n") + 1]
        if not raw:
            return

        columns = ["timestamp_us", *SENSOR_COLUMNS[self.type]]
        data = pl.read_csv(
            io.BytesIO(raw),
            has_header=False,
            new_columns=columns,
            separator=" ",
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )
        data = data.with_columns(
            pl.col("timestamp_us").cast(pl.Int64, strict=False),
            *(pl.col(c).cast(pl.Float64, strict=False) for c in columns[1:]),
        ).drop_nulls()
        if self.type == "baro":
            data = data.filter(pl.col("pressure") != 0)
        if self.type == "magneto":
            data = data.with_columns(pl.col(columns[1:]).cast(pl.Int64))
        dropped = raw.count(b"\n") - data.height
        if dropped:
            logger.info(f"{self.path.name}: dropped {dropped} malformed lines")

        data = data.with_columns(
            (pl.col("timestamp_us") / 1e6).alias("timestamp"),
            pl.from_epoch(pl.col("timestamp_us"), time_unit="us")
            .dt.replace_time_zone("UTC")
            .alias("datetime"),
        ).select(["timestamp", "timestamp_us", *columns[1:], "datetime"])
        if data.is_empty():
            return
        span = data["timestamp"][-1] - data["timestamp"][0]
        if span < SHORT_RECORDING_S:
            logger.warning(
                f"{self.path.name}: only {span:.1f} s of data, the recording "
                "stopped early"
            )
        self.data = data


class IMU:
    """
    IMU sensor container with barometer, accelerometer, gyroscope, and magnetometer.

    Attributes
    ----------
    dirpath : Path
        Directory containing IMU sensor files.
    barometer : IMUSensor
        Barometer sensor instance.
    accelerometer : IMUSensor
        Accelerometer sensor instance.
    gyroscope : IMUSensor
        Gyroscope sensor instance.
    magnetometer : IMUSensor
        Magnetometer sensor instance.
    """

    def __init__(self, dirpath: Path) -> None:
        """
        Initialize IMU with all four sensors.

        Parameters
        ----------
        dirpath : Path
            Directory containing sensor files: ``*_BAR.bin``, ``*_ACC.bin``,
            ``*_GYR.bin`` and ``*_MAG.bin``.
        """
        self.dirpath = Path(dirpath)
        self.barometer = IMUSensor(_find_file(self.dirpath, "baro"), "baro")
        self.accelerometer = IMUSensor(_find_file(self.dirpath, "accelero"), "accelero")
        self.gyroscope = IMUSensor(_find_file(self.dirpath, "gyro"), "gyro")
        self.magnetometer = IMUSensor(_find_file(self.dirpath, "magneto"), "magneto")

    def load_all(self) -> None:
        """
        Load data for all IMU sensors.

        Calls load_data() on each sensor (barometer, accelerometer,
        gyroscope, magnetometer).

        Raises
        ------
        FileNotFoundError
            If the folder has none of the IMU files.
        """
        sensors = [s for s in vars(self).values() if isinstance(s, IMUSensor)]
        if all(sensor.path is None for sensor in sensors):
            raise FileNotFoundError(f"No IMU files in {self.dirpath}")
        for sensor in sensors:
            sensor.load_data()

    @property
    def data(self) -> dict[str, pl.DataFrame] | None:
        """The loaded sensors with data, by name, or None if there are none."""
        tables = {
            name: sensor.data
            for name, sensor in vars(self).items()
            if isinstance(sensor, IMUSensor) and sensor.data is not None
        }
        return tables or None
