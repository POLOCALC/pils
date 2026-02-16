"""
PILS - POLOCALC Inertial & Drone Loading System

A comprehensive Python package for loading, decoding, and analyzing flight data
from drone missions with integrated STOUT campaign management system support.

Modules:
    - loader: StoutDataLoader for querying flights and accessing raw data paths
    - decoder: DataDecoder for parsing and decoding sensor and drone data
    - handler: FlightDataHandler for unified data access with automatic decoding
    - sensors: Sensor-specific data decoders (GPS, IMU, Camera, etc.)
    - drones: Drone-specific data parsers (DJI, BlackSquare, Litchi)
    - utils: Utility functions for path handling and log parsing

Example Usage:
    from pils import FlightDataHandler

    # Initialize with STOUT database
    handler = FlightDataHandler(use_stout=True)

    # Load and decode all flights from a campaign
    flights = handler.load_campaign_flights(campaign_id="camp-123")

    # Access decoded sensor data
    for flight in flights:
        print(f"Flight: {flight.flight_name}")
        print(f"GPS data: {flight.payload.gps.data.head()}")
        print(f"IMU data: {flight.payload.imu.data}")
"""

__version__ = "2026.2.6"
__author__ = "POLOCALC Team"
__license__ = "MIT"

from . import decoders, drones, sensors, utils
from .flight import Flight
from .loader import PathLoader, StoutLoader
from .pils import PILS
from .synchronizer import Synchronizer

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
