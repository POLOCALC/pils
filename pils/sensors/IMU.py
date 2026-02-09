from pathlib import Path

import polars as pl

SENSOR_COLUMNS = {
    "baro": ["timestamp", "pressure", "temperature"],
    "magneto": ["timestamp", "mag_x", "mag_y", "mag_z"],
    "accelero": ["timestamp", "acc_x", "acc_y", "acc_z"],
    "gyro": ["timestamp", "x", "y", "z"],
}


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
        """
        self.data = pl.read_csv(
            self.path,
            has_header=False,
            new_columns=SENSOR_COLUMNS[self.type],
            separator=" ",
        )
        self.data = self.data.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="us").alias("datetime")
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
        """
        self.dirpath = dirpath
        self.barometer = IMUSensor(dirpath / "barometer.bin", "baro")
        self.accelerometer = IMUSensor(dirpath / "accelerometer.bin", "accelero")
        self.gyroscope = IMUSensor(dirpath / "gyroscope.bin", "gyro")
        self.magnetometer = IMUSensor(dirpath / "magnetometer.bin", "magneto")

    def load_all(self) -> None:
        """
        Load data for all IMU sensors.

        Calls load_data() on each sensor (barometer, accelerometer,
        gyroscope, magnetometer).
        """
        for sensor in vars(self).values():
            if isinstance(sensor, IMUSensor):
                sensor.load_data()
