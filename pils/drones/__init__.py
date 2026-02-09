"""
Drones module - Drone-specific data parsers and utilities.
"""

import logging
import os

from pils.drones.BlackSquareDrone import BlackSquareDrone  # noqa: F401
from pils.drones.DJIDrone import DJIDrone  # noqa: F401
from pils.drones.litchi import Litchi

logger = logging.getLogger(__name__)


def drone_init(drone_model: str, drone_path: str):
    """
    Initialize appropriate drone class based on model.

    Parameters
    ----------
    drone_model : str
        'dji', 'blacksquare', or 'litchi'
    drone_path : str
        Path to drone data file or directory

    Returns
    -------
    object
        Initialized drone object
    """
    drone_model = drone_model.lower()

    if drone_model == "dji":
        return DJIDrone(drone_path)
    elif drone_model == "blacksquare":
        return BlackSquareDrone(drone_path)
    elif drone_model == "litchi":
        return Litchi(drone_path)
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
    "DJIDrone",
    "BlackSquareDrone",
    "Litchi",
]
