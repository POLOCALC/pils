"""
PILS - POLOCALC Inertial & Drone Loading System.

Loading, decoding, synchronizing and analyzing the flight data of POLOCALC's
drone missions, with flights found through STOUT.

Modules:
    - loader: PathLoader and StoutLoader, to find flights and their raw data paths
    - flight: Flight, holding a flight's drone and payload data
      (``add_drone_data``, ``add_sensor_data``, ``sync``, ``save``/``load``)
    - synchronizer: Synchronizer, aligning the sources by GPS and pitch correlation
    - sensors, drones, decoders: the sensor and drone log readers
    - store: Parquet revisions of the processed data
    - analyze: AZEL, PPK, EKF, power and the flight packaging
    - utils: helpers (paths, logs, GNSS time, units)

Example Usage:
    from pils.flight import Flight
    from pils.loader import PathLoader

    info = PathLoader("/data").load_single_flight(flight_name="flight_...")
    flight = Flight(info)
    flight.add_drone_data()
    flight.add_sensor_data(["gps", "imu", "adc"])
    synced = flight.sync()
    print(synced["drone"].head())
"""

__version__ = "2026.11.2"
__author__ = "POLOCALC Team"
__license__ = "MIT"

import importlib
from typing import Any

# Imported when first used, so that reading saved data (pils.store,
# pils.schema) does not load the decoders and their dependencies (it needs
# only polars and STOUT's data layout)
_LAZY = {
    "Flight": ("pils.flight", "Flight"),
    "PILS": ("pils.pils", "PILS"),
    "PathLoader": ("pils.loader", "PathLoader"),
    "StoutLoader": ("pils.loader", "StoutLoader"),
    "Synchronizer": ("pils.synchronizer", "Synchronizer"),
    "decoders": ("pils.decoders", None),
    "drones": ("pils.drones", None),
    "sensors": ("pils.sensors", None),
    "utils": ("pils.utils", None),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'pils' has no attribute {name!r}")
    module, attribute = _LAZY[name]
    value = importlib.import_module(module)
    if attribute is not None:
        value = getattr(value, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "Flight",
    "PILS",
    "StoutLoader",
    "PathLoader",
    "Synchronizer",
    "sensors",
    "drones",
    "decoders",
    "utils",
    "__version__",
    "__author__",
    "__license__",
]
