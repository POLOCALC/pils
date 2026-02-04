import os
from pathlib import Path

import polars as pl

from ..utils.tools import get_path_from_keyword

SENSOR_COLUMNS = {
    "baro": ["timestamp", "pressure", "temperature"],
    "magneto": ["timestamp", "mag_x", "mag_y", "mag_z"],
    "accelero": ["timestamp", "acc_x", "acc_y", "acc_z"],
    "gyro": ["timestamp", "x", "y", "z"],
}


class IMUSensor:
    def __init__(self, path, type):
        self.path = path
        self.type = type
        self.data = None

    def load_data(self):
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
    def __init__(self, dirpath: Path):
        self.dirpath = dirpath
        self.barometer = IMUSensor(dirpath / "barometer.bin", "baro")
        self.accelerometer = IMUSensor(dirpath / "accelerometer.bin", "accelero")
        self.gyroscope = IMUSensor(dirpath / "gyroscope.bin", "gyro")
        self.magnetometer = IMUSensor(dirpath / "magnetometer.bin", "magneto")

    def load_all(self):
        for sensor in vars(self).values():
            if isinstance(sensor, IMUSensor):
                sensor.load_data()
