"""Drones module - Drone-specific data parsers and utilities."""

import logging
import os

from pils.drones.blacksquare import BlackSquare  # noqa: F401
from pils.drones.dji.dat import DJIDat  # noqa: F401
from pils.drones.dji.litchi import Litchi
from pils.drones.dji.m400.flightrecord import M400FlightRecord
from pils.drones.dji.m400.telemetry import M400Telemetry
from pils.drones.dji.m600.telemetry import M600Telemetry
from pils.drones.telemetry import Telemetry, load_telemetry

logger = logging.getLogger(__name__)


def drone_init(drone_model: str, drone_path: str):
    """
    Initialize appropriate drone class based on model.

    Parameters
    ----------
    drone_model : str
        'dji_dat' (the .DAT file or a DatCon CSV), 'litchi', 'blacksquare',
        'm600_telemetry' or 'm400_telemetry' (LAGER's uav_telemetry.bin) or
        'm400_flightrecord' (the app's .txt)
    drone_path : str
        Path to drone data file or directory

    Returns
    -------
    object
        Initialized drone object
    """
    drone_model = drone_model.lower()

    if drone_model == "dji_dat":
        return DJIDat(drone_path)
    elif drone_model == "blacksquare":
        return BlackSquare(drone_path)
    elif drone_model == "litchi":
        return Litchi(drone_path)
    elif drone_model == "m600_telemetry":
        return M600Telemetry(drone_path)
    elif drone_model == "m400_telemetry":
        return M400Telemetry(drone_path)
    elif drone_model == "m400_flightrecord":
        return M400FlightRecord(drone_path)
    else:
        raise ValueError(f"Unknown drone model '{drone_model}'")


def find_first_drone_file(dirpath: str) -> str | None:
    """
    Find the first *_drone.csv file in directory tree.

    Assumes one flight per directory.

    Parameters
    ----------
    dirpath : str
        Directory to search

    Returns
    -------
    Optional[str]
        Path to drone file or None if not found
    """
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.endswith("_drone.csv"):
                return os.path.join(root, f)
    return None


__all__ = [
    "drone_init",
    "find_first_drone_file",
    "DJIDat",
    "BlackSquare",
    "Litchi",
    "M400FlightRecord",
    "M400Telemetry",
    "M600Telemetry",
    "Telemetry",
    "load_telemetry",
]
