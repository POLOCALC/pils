from pathlib import Path

import polars as pl

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

SENSOR_COLUMNS = {
    "baro": ["timestamp", "pressure", "temperature"],
    "magneto": ["timestamp", "mag_x", "mag_y", "mag_z"],
    "accelero": ["timestamp", "acc_x", "acc_y", "acc_z"],
    "gyro": ["timestamp", "x", "y", "z"],
}

# IMX5 raw IMU CSV (*_INC_imu.csv / IMX5-Sensors_*_imu.csv) -> SENSOR_COLUMNS names.
# Values keep the IMX5 units (m/s², rad/s) and body frame.
IMX5_COLUMNS = {
    "accelero": {"acc_X_m_s2": "acc_x", "acc_Y_m_s2": "acc_y", "acc_Z_m_s2": "acc_z"},
    "gyro": {"pqr_P_rad_s": "x", "pqr_Q_rad_s": "y", "pqr_R_rad_s": "z"},
}
IMX5_IMU_PATTERNS = ("*_INC_imu.csv", "IMX5*_imu.csv")


class IMUSensor:
    """
    Individual IMU sensor (barometer, accelerometer, gyroscope, or magnetometer).

    Attributes
    ----------
    path : str | Path
        Path to sensor data file.
    type : str
        Sensor type (baro, accelero, gyro, magneto).
    data : Optional[pl.DataFrame]
        Polars DataFrame with sensor data (None until load_data is called).
    """

    def __init__(self, path: str | Path, sensor_type: str) -> None:
        """Initialize IMU sensor.

        Parameters
        ----------
        path : Union[str, Path]
            Path to sensor binary file.
        sensor_type : str
            Sensor type (baro, accelero, gyro, magneto).
        """
        self.path = path
        self.type = sensor_type
        self.data: pl.DataFrame | None = None

    def load_data(self) -> None:
        """
        Load sensor data from binary file.

        Reads space-separated data and converts timestamp to datetime.
        Sets the `data` attribute with a polars DataFrame containing:
        - timestamp column (microseconds)
        - sensor-specific columns (defined in SENSOR_COLUMNS)
        - datetime column (converted from timestamp)

        An empty file (sensor not recorded) leaves `data` as None.
        An IMX5 ``.csv`` is read with `_load_imx5` instead.
        """
        if Path(self.path).stat().st_size == 0:
            logger.warning(f"IMU {self.type} file is empty: {self.path}")
            self.data = None
            return
        if Path(self.path).suffix.lower() == ".csv":
            self._load_imx5()
            return
        self.data = pl.read_csv(
            self.path,
            has_header=False,
            new_columns=SENSOR_COLUMNS[self.type],
            separator=" ",
        )
        self.data = self.data.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="us").alias("datetime")
        )

    def _load_imx5(self) -> None:
        """Load accelerometer/gyroscope from the IMX5 raw IMU CSV.

        ``timestamp_ns`` becomes ``timestamp`` in microseconds, like the
        ``.bin`` files, and the axes are renamed to SENSOR_COLUMNS.
        """
        columns = IMX5_COLUMNS[self.type]
        self.data = pl.read_csv(self.path).select(
            (pl.col("timestamp_ns") // 1000).alias("timestamp"),
            *(pl.col(src).alias(dst) for src, dst in columns.items()),
            pl.from_epoch(pl.col("timestamp_ns"), time_unit="ns").alias("datetime"),
        )


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
            Directory containing sensor binary files:
            - barometer.bin
            - accelerometer.bin
            - gyroscope.bin
            - magnetometer.bin
            Newer acquisitions write them into an ``Inertial_<timestamp>.bin/``
            subfolder instead, which is used when present. Others name them
            ``<timestamp>_BAR.bin``, ``_ACC.bin``, ``_GYR.bin``, ``_MAG.bin``
            directly in ``dirpath``.
            When an IMX5 raw IMU file (``*_INC_imu.csv`` or
            ``IMX5-Sensors_*_imu.csv``) is in ``dirpath``, the accelerometer
            and gyroscope are read from it instead (in those acquisitions
            the ``.bin`` ones are empty).
        """
        dirpath = Path(dirpath)
        imx5_path = self._find_imx5(dirpath)
        if not (dirpath / "barometer.bin").exists():
            subdirs = sorted(p for p in dirpath.glob("Inertial_*.bin") if p.is_dir())
            if subdirs:
                dirpath = subdirs[0]
        self.dirpath = dirpath
        if imx5_path is not None:
            logger.info(f"IMU accelerometer/gyroscope from IMX5: {imx5_path}")
        self.barometer = IMUSensor(self._find_file("barometer", "BAR"), "baro")
        self.accelerometer = IMUSensor(
            imx5_path or self._find_file("accelerometer", "ACC"), "accelero"
        )
        self.gyroscope = IMUSensor(imx5_path or self._find_file("gyroscope", "GYR"), "gyro")
        self.magnetometer = IMUSensor(self._find_file("magnetometer", "MAG"), "magneto")

    @staticmethod
    def _find_imx5(dirpath: Path) -> Path | None:
        """First non-empty IMX5 raw IMU CSV in ``dirpath``, or None."""
        for pattern in IMX5_IMU_PATTERNS:
            for path in sorted(dirpath.glob(pattern)):
                if path.stat().st_size > 0:
                    return path
        return None

    def _find_file(self, name: str, suffix: str) -> Path:
        """``<name>.bin`` if present, else the first ``*_<suffix>.bin``."""
        path = self.dirpath / f"{name}.bin"
        if not path.exists():
            matches = sorted(self.dirpath.glob(f"*_{suffix}.bin"))
            if matches:
                path = matches[0]
        return path

    def load_all(self) -> None:
        """
        Load data for all IMU sensors.

        Calls load_data() on each sensor (barometer, accelerometer,
        gyroscope, magnetometer).
        """
        for sensor in vars(self).values():
            if isinstance(sensor, IMUSensor):
                sensor.load_data()
