"""
DataDecoder - Unified data decoding for flight sensor and drone data.

Handles automatic decoding of:
- Drone data (DJI, BlackSquare, Litchi)
- Sensor data (GPS, IMU, Camera, ADC, Inclinometer)
- Log files

The decoder automatically detects data types and applies appropriate decoders.
"""

import os
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

from pils.datahandler import Payload
from pils.utils.tools import get_path_from_keyword
from pils.sensors.gps import GPS
from pils.sensors.IMU import IMU
from pils.sensors.adc import ADC
from pils.sensors.camera import Camera
from pils.sensors.inclinometer import Inclinometer
from pils.drones import (
    DJIDrone,
    BlackSquareDrone,
    Litchi,
    drone_init,
    find_first_drone_file,
)

logger = logging.getLogger(__name__)


class DataDecoder:
    """
    Unified decoder for POLOCALC flight data.

    Automatically detects and decodes:
    - Drone flight logs (DJI, BlackSquare)
    - Payload sensor data (GPS, IMU, Camera, etc.)
    - Litchi flight plans and telemetry
    - Raw data files

    Attributes:
        flight_path: Path to flight directory
        drone: Decoded drone data
        payload: Decoded sensor payload
        litchi: Litchi flight data
    """

    def __init__(
        self,
        flight_path: str,
        drone_model: Optional[str] = None,
        adc_gain_config: Optional[int] = None,
    ):
        """
        Initialize DataDecoder for a flight.

        Args:
            flight_path: Path to flight directory (containing drone/, aux/, proc/)
            drone_model: Drone model ('dji', 'blacksquare', or None for auto-detect)
            adc_gain_config: ADC gain configuration (1, 2, 4, 8, or 16).
                             If None, auto-detects from config.yml in the aux folder.
        """
        self.flight_path = flight_path
        self.drone_model = drone_model
        self.adc_gain_config = adc_gain_config

        # Data attributes
        self.drone = None
        self.payload = None
        self.litchi = None
        self.raw_files = {}

        logger.info(f"Initialized DataDecoder for {flight_path}")

    def load_all(self) -> Dict[str, Any]:
        """
        Load and decode all available data from the flight.

        Returns:
            Dictionary with decoded data:
            {
                'drone': drone_data,
                'payload': payload_data,
                'litchi': litchi_data,
                'raw_files': file_paths,
            }
        """
        logger.info(f"Loading all data from {self.flight_path}")

        result = {}

        # Load drone data
        try:
            self.load_drone_data()
            result["drone"] = self.drone
        except Exception as e:
            logger.warning(f"Could not load drone data: {e}")

        # Load payload sensor data
        try:
            self.load_payload_data()
            result["payload"] = self.payload
        except Exception as e:
            logger.warning(f"Could not load payload data: {e}")

        # Load litchi data if available
        try:
            self.load_litchi_data()
            result["litchi"] = self.litchi
        except Exception as e:
            logger.debug(f"Could not load litchi data: {e}")

        # Catalog raw files
        self.raw_files = self._catalog_raw_files()
        result["raw_files"] = self.raw_files

        return result

    def load_drone_data(self) -> None:
        """
        Load drone flight log data.

        Attempts to:
        1. Auto-detect drone type from data structure
        2. Parse drone CSV files
        3. Create drone data object with telemetry
        """
        drone_folder = os.path.join(self.flight_path, "drone")

        if not os.path.exists(drone_folder):
            logger.warning(f"Drone folder not found: {drone_folder}")
            return

        try:
            # Find drone data file
            drone_file = find_first_drone_file(drone_folder)
            if not drone_file:
                logger.warning(f"No drone CSV file found in {drone_folder}")
                return

            # Auto-detect drone model if not specified
            if not self.drone_model:
                self.drone_model = self._detect_drone_model(drone_folder)

            # Initialize and load drone
            if self.drone_model:
                self.drone = drone_init(self.drone_model, drone_file)
                self.drone.load_data()

            logger.info(f"Loaded {self.drone_model} drone data")
        except Exception as e:
            logger.error(f"Error loading drone data: {e}")
            raise

    def load_payload_data(self) -> None:
        """
        Load payload sensor data.

        Attempts to load:
        - GPS (UBX binary format)
        - IMU sensors (barometer, accelerometer, gyroscope, magnetometer)
        - ADC data
        - Inclinometer
        - Camera/video files
        """
        aux_folder = os.path.join(self.flight_path, "aux")

        if not os.path.exists(aux_folder):
            logger.warning(f"Aux folder not found: {aux_folder}")
            return

        try:
            self.payload = Payload(
                dirpath=aux_folder, adc_gain_config=self.adc_gain_config
            )
            self.payload.load_all()

            logger.info(f"Loaded payload data from {aux_folder}")
        except Exception as e:
            logger.error(f"Error loading payload data: {e}")
            raise

    def load_litchi_data(self) -> None:
        """
        Load Litchi flight plan and telemetry data if available.
        """
        aux_folder = os.path.join(self.flight_path, "aux")

        if not os.path.exists(aux_folder):
            return

        try:
            # Look for Litchi CSV file
            litchi_file = get_path_from_keyword(aux_folder, "litchi.csv")

            self.litchi = Litchi(litchi_file)
            self.litchi.load_data()

            logger.info(f"Loaded Litchi data from {litchi_file}")
        except Exception as e:
            logger.debug(f"Could not load Litchi data: {e}")

    def load_specific_sensors(self, sensor_types: List[str]) -> Dict[str, Any]:
        """
        Load specific sensor data only.

        Args:
            sensor_types: List of sensor types to load
                         ('gps', 'imu', 'adc', 'camera', 'inclinometer')

        Returns:
            Dictionary mapping sensor_type to loaded data
        """
        result = {}
        aux_folder = os.path.join(self.flight_path, "aux")

        if not os.path.exists(aux_folder):
            logger.warning(f"Aux folder not found: {aux_folder}")
            return result

        try:
            for sensor_type in sensor_types:
                try:
                    if sensor_type.lower() == "gps":
                        gps_file = get_path_from_keyword(aux_folder, ".bin")
                        if gps_file:
                            gps = GPS(gps_file)
                            gps.load_data()
                            result["gps"] = gps

                    elif sensor_type.lower() == "imu":
                        imu = IMU(aux_folder)
                        imu.load_all()
                        result["imu"] = imu

                    elif sensor_type.lower() == "adc":
                        adc_file = get_path_from_keyword(aux_folder, "adc")
                        if adc_file:
                            adc = ADC(adc_file, gain_config=self.adc_gain_config)
                            adc.load_data()
                            result["adc"] = adc

                    elif sensor_type.lower() == "camera":
                        camera_file = get_path_from_keyword(aux_folder, ".mp4")
                        if not camera_file:
                            camera_file = get_path_from_keyword(aux_folder, ".jpg")
                        if camera_file:
                            camera = Camera(camera_file)
                            camera.load_data()
                            result["camera"] = camera

                    elif sensor_type.lower() == "inclinometer":
                        inclino_file = get_path_from_keyword(aux_folder, "inclino")
                        if inclino_file:
                            # Handle both single file and multiple files
                            if isinstance(inclino_file, list):
                                inclino = Inclinometer(aux_folder)
                            else:
                                inclino = Inclinometer(aux_folder)
                            inclino.load_data()
                            result["inclinometer"] = inclino

                except Exception as e:
                    logger.warning(f"Could not load {sensor_type}: {e}")

            logger.info(f"Loaded {len(result)} sensor types")
            return result

        except ImportError as e:
            logger.error(f"Missing sensor modules: {e}")
            raise

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available data without loading everything.

        Returns:
            Dictionary with information about available data
        """
        summary = {
            "flight_path": self.flight_path,
            "drone_folder": os.path.exists(os.path.join(self.flight_path, "drone")),
            "aux_folder": os.path.exists(os.path.join(self.flight_path, "aux")),
            "proc_folder": os.path.exists(os.path.join(self.flight_path, "proc")),
            "available_sensors": self._detect_available_sensors(),
            "drone_model": self.drone_model,
        }
        return summary

    # ==================== Private Methods ====================

    def _detect_drone_model(self, drone_folder: str) -> Optional[str]:
        """Auto-detect drone model from data structure."""
        try:
            # Look for drone-specific patterns in filenames
            dji_pattern = get_path_from_keyword(drone_folder, "DJI")
            if dji_pattern:
                return "dji"

            blacksquare_pattern = get_path_from_keyword(drone_folder, "blacksquare")
            if blacksquare_pattern:
                return "blacksquare"

            # Default to DJI
            return "dji"
        except Exception:
            return "dji"

    def _detect_available_sensors(self) -> List[str]:
        """Detect which sensors have data available."""
        sensors = []
        aux_folder = os.path.join(self.flight_path, "aux")

        if not os.path.exists(aux_folder):
            return sensors

        try:
            if get_path_from_keyword(aux_folder, ".bin"):
                sensors.append("gps")
            if (
                get_path_from_keyword(aux_folder, "baro")
                or get_path_from_keyword(aux_folder, "accel")
                or get_path_from_keyword(aux_folder, "gyro")
            ):
                sensors.append("imu")
            if get_path_from_keyword(aux_folder, "adc"):
                sensors.append("adc")
            if get_path_from_keyword(aux_folder, ".mp4") or get_path_from_keyword(
                aux_folder, ".jpg"
            ):
                sensors.append("camera")
            if get_path_from_keyword(aux_folder, "inclino"):
                sensors.append("inclinometer")

        except Exception as e:
            logger.debug(f"Error detecting sensors: {e}")

        return sensors

    def _catalog_raw_files(self) -> Dict[str, List[str]]:
        """
        Catalog all raw files in the flight directory.

        Returns:
            Dictionary mapping folder names to lists of files
        """
        catalog = {}

        for folder_name in ["drone", "aux", "proc"]:
            folder_path = os.path.join(self.flight_path, folder_name)
            if os.path.exists(folder_path):
                files = []
                for root, dirs, filenames in os.walk(folder_path):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
                catalog[folder_name] = files

        return catalog
