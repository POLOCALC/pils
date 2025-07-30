import pandas as pd
from tools import get_path_from_keyword
import os

SENSOR_COLUMNS = {"baro":["timestamp", "pressure", "temperature"],
                  "magneto":["timestamp", "mag_x", "mag_y", "mag_z"],
                  "accelero":["timestamp", "acc_x", "acc_y", "acc_z"],
                  "gyro":["timestamp", "x", "y", "z"],
}

class IMUSensor:
    def __init__(self, path, type):
        self.path = path
        self.type = type
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.path, names=SENSOR_COLUMNS[self.type], sep=" ")
        self.data["datetime"] = pd.to_datetime(self.data["timestamp"], unit="us")
        

class IMU:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.barometer = IMUSensor(get_path_from_keyword(dirpath, "barometer.bin"), "baro")
        self.accelerometer = IMUSensor(get_path_from_keyword(dirpath, "accelerometer.bin"), "accelero")
        self.gyroscope = IMUSensor(get_path_from_keyword(dirpath, "gyroscope.bin"), "gryo")
        self.magnetometer = IMUSensor(get_path_from_keyword(dirpath, "magnetometer.bin"), "magneto")

    def load_all(self):
        for sensor in vars(self).values():
            sensor.load_data()
