"""
datahandler.py

Central coordination layer for loading:
- drone data (DJI / BlackSquare)
- payload sensors
- litchi logs
- camera data

Compatible with new filename convention:
YYYYMMDD_HHMMSS_{drone|litchi}.csv
"""

import os
import datetime
import pandas as pd

from pathhandler import PathHandler

from sensors.inclinometer import Inclinometer
from sensors.gps import GPS
from sensors.adc import ADC
from sensors.IMU import IMUSensor
from sensors.camera import Camera

from drones.litchi import Litchi
from drones.DJIDrone import DJIDrone
from drones.BlackSquareDrone import BlackSquareDrone


# =============================================================================
# Drone factory
# =============================================================================

def drone_init(drone_model, drone_path):
    drone_model = drone_model.lower()
    if drone_model == "dji":
        return DJIDrone(drone_path)
    elif drone_model == "blacksquare":
        return BlackSquareDrone(drone_path)
    else:
        raise ValueError(
            f"[drone_init] Unknown drone model '{drone_model}'"
        )


# =============================================================================
# Timestamp utilities
# =============================================================================

def extract_timestamp_from_drone_file(drone_file):
    """
    Extract UTC timestamp from:
    YYYYMMDD_HHMMSS_drone.csv
    """
    name = os.path.basename(drone_file)
    ts = datetime.datetime.strptime(
        name, "%Y%m%d_%H%M%S_drone.csv"
    )
    return pd.Timestamp(ts, tz="UTC")


def find_first_drone_file(dirpath):
    """
    Find the first *_drone.csv file in directory tree.
    Assumes one flight per directory.
    """
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.endswith("_drone.csv"):
                return os.path.join(root, f)
    return None


# =============================================================================
# Payload container
# =============================================================================

class Payload:
    """
    Groups all payload sensors and loads them uniformly.
    """

    def __init__(self, pathhandler):
        self.gps = GPS(pathhandler.gps)
        self.adc = ADC(pathhandler.adc)
        #self.inclino = Inclinometer(pathhandler.inclino)
        self.baro = IMUSensor(pathhandler.baro, "baro")
        self.accelero = IMUSensor(pathhandler.accelero, "accelero")
        self.magneto = IMUSensor(pathhandler.magneto, "magneto")
        self.gyro = IMUSensor(pathhandler.gyro, "gyro")

    def load_data(self):
        for attr in vars(self).values():
            if hasattr(attr, "load_data") and getattr(attr, "path", None):
                attr.load_data()


# =============================================================================
# DataHandler
# =============================================================================

class DataHandler:
    """
    High-level interface to all flight data.

    Parameters
    ----------
    flight_dir : str
        Path to flight_YYYYMMDD_HHMM directory
    drone_model : str
        'DJI' or 'BlackSquare'
    """

    def __init__(self, flight_dir, drone_model="DJI"):

        # ---------------------------------------------------------------------
        # 1. Initialize paths
        # ---------------------------------------------------------------------
        self.paths = PathHandler(flight_dir)
        self.paths.get_filenames()

        if self.paths.drone is None:
            raise FileNotFoundError(
                f"[DataHandler] No *_drone.csv found in {flight_dir}"
            )

        # ---------------------------------------------------------------------
        # 2. Initialize drone
        # ---------------------------------------------------------------------
        self.drone_model = drone_model.lower()
        self.drone = drone_init(self.drone_model, self.paths.drone)
        self.drone_timestamp = extract_timestamp_from_drone_file(self.paths.drone)

        # ---------------------------------------------------------------------
        # 3. Initialize payload sensors
        # ---------------------------------------------------------------------
        self.payload = Payload(self.paths)

        # ---------------------------------------------------------------------
        # 4. Initialize Litchi and camera (optional)
        # ---------------------------------------------------------------------
        self.litchi = Litchi(self.paths.litchi) if self.paths.litchi else None
        self.camera = Camera(self.paths.camera) if self.paths.camera else None

    # -------------------------------------------------------------------------
    # Load all data
    # -------------------------------------------------------------------------
    def load_data(self):
        """
        Load all available data streams.
        """
        for attr_name, attr in vars(self).items():
            if hasattr(attr, "load_data"):
                # Only load if path exists or if object has no path attribute
                if hasattr(attr, "path"):
                    if getattr(attr, "path") is not None:
                        attr.load_data()
                else:
                    attr.load_data()
