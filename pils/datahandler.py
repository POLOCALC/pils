"""
DataHandler - Unified payload data loading.

This module provides the Payload class for loading all sensor data
from a flight's auxiliary data folder. For synchronization of payload
and drone data, use the Synchronizer class from pils.synchronizer.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from pathlib import Path
import numpy as np
import yaml
import polars as pl

from pils.utils.tools import get_path_from_keyword
from pils.sensors.gps import GPS
from pils.sensors.IMU import IMU
from pils.sensors.adc import ADC
from pils.sensors.camera import Camera
from pils.sensors.inclinometer import Inclinometer

logger = logging.getLogger(__name__)


class Payload:
    """
    Groups all payload sensors and loads them uniformly.

    Attributes:
        gps: GPS sensor data
        imu: IMU sensor data (barometer, accelerometer, gyroscope, magnetometer)
        adc: Analog-to-digital converter data
        inclinometer: Inclinometer data
        camera: Camera/video data
        config: Configuration from config.yml file
        available_sensors: List of sensors detected from config.yml
    """

    def __init__(self, dirpath: str, adc_gain_config: Optional[int] = None):
        """
        Initialize Payload from directory path.

        Args:
            dirpath: Path to aux folder containing sensor data
            adc_gain_config: ADC gain configuration (1, 2, 4, 8, or 16).
                             If None, auto-detects from config.yml in the aux folder.
        """
        self.dirpath = dirpath
        self.adc_gain_config = adc_gain_config
        self.gps = None
        self.imu = None
        self.adc = None
        self.inclinometer = None
        self.camera = None
        self._sensor_info: Dict[str, Dict[str, Any]] = {}
        self.config: Optional[Dict[str, Any]] = None
        self.available_sensors: List[str] = []

        # Load config and detect available sensors
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from config.yml file in the aux folder."""

        # Find config.yml file
        config_path = None
        aux_path = Path(self.dirpath)
        for f in aux_path.glob("*_config.yml"):
            config_path = f
            break

        if config_path is None:
            # Try direct config.yml
            if (aux_path / "config.yml").exists():
                config_path = aux_path / "config.yml"

        if config_path is None:
            logger.debug("No config.yml found in aux folder")
            return

        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            self._detect_available_sensors()
        except Exception as e:
            logger.warning(f"Failed to load config.yml: {e}")

    def _detect_available_sensors(self) -> None:
        """Detect available sensors from the config.yml file."""
        if self.config is None:
            return

        sensors_config = self.config.get("sensors", {})
        camera_config = self.config.get("camera", None)

        self.available_sensors = []

        for sensor_name, sensor_info in sensors_config.items():
            sensor_type = sensor_info.get("sensor_info", {}).get("type", "").upper()

            if sensor_type == "GPS":
                self.available_sensors.append("gps")
            elif sensor_type == "ADC":
                self.available_sensors.append("adc")
                # Also extract ADC gain if not already set
                if self.adc_gain_config is None:
                    gain = sensor_info.get("configuration", {}).get("gain")
                    if gain:
                        self.adc_gain_config = gain
                        logger.info(f"Auto-detected ADC gain: {gain}")
            elif sensor_type == "INERTIAL":
                self.available_sensors.append("imu")
            elif sensor_type == "IMX5":
                self.available_sensors.append("inclinometer")
            elif sensor_type == "LM76":
                self.available_sensors.append("temperature")
            elif sensor_type == "DAC":
                self.available_sensors.append("dac")

        if camera_config:
            self.available_sensors.append("camera")

        # Remove duplicates while preserving order
        self.available_sensors = list(dict.fromkeys(self.available_sensors))
        logger.info(f"Detected sensors from config: {self.available_sensors}")

    def get_available_sensors(self) -> List[str]:
        """
        Get list of sensors available according to config.yml.

        Returns:
            List of sensor names: 'gps', 'adc', 'imu', 'inclinometer', 'camera', etc.
        """
        return self.available_sensors.copy()

    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the full configuration from config.yml.

        Returns:
            Configuration dictionary or None if not loaded.
        """
        return self.config

    def get_sensor_availability(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about which sensors are configured and which files exist.

        Returns a dictionary with sensor details including:
        - configured: Whether sensor is in config.yml
        - files_exist: Whether the data files are present
        - file_paths: List of paths to the data files (for multi-file sensors)
        - sub_sensors: Sub-components for composite sensors like IMU

        Returns:
            Dictionary mapping sensor names to their status info.
        """
        availability = {}

        # Define sensor file patterns: (config_type, file_code, friendly_name, is_imu_component, is_multi_file)
        sensor_patterns = [
            ("GPS", "GPS", "GPS", False, False),
            ("ADC", "ADC", "ADC", False, False),
            ("INERTIAL", "ACC", "Accelerometer", True, False),
            ("INERTIAL", "GYR", "Gyroscope", True, False),
            ("INERTIAL", "MAG", "Magnetometer", True, False),
            ("INERTIAL", "BAR", "Barometer", True, False),
            (
                "IMX5",
                "INC",
                "Inclinometer",
                False,
                True,
            ),  # Inclinometer has multiple files
            (
                "LM76",
                "TMP",
                "Temperature",
                False,
                True,
            ),  # Temperature may have multiple files
        ]

        aux_path = Path(self.dirpath)
        sensors_path = aux_path / "sensors"
        camera_path = aux_path / "camera"

        # Get timestamp prefix from config files
        timestamp_prefix = None
        for config_file in aux_path.glob("*_config.yml"):
            # Extract timestamp from filename like "20251201_151517_config.yml"
            parts = config_file.stem.split("_")
            if len(parts) >= 2:
                timestamp_prefix = f"{parts[0]}_{parts[1]}"
                break

        if not timestamp_prefix:
            logger.warning("Could not determine timestamp prefix from config.yml")
            return availability

        # Check each sensor pattern
        imu_components = {}
        for (
            config_type,
            file_code,
            friendly_name,
            is_imu,
            is_multi_file,
        ) in sensor_patterns:
            # Check if this sensor type is in config
            config_has_sensor = False
            if self.config:
                for sensor_name, sensor_info in self.config.get("sensors", {}).items():
                    if sensor_info.get("sensor_info", {}).get("type") == config_type:
                        config_has_sensor = True
                        break

            # Look for the data files
            files_found = []

            # Search in sensors folder first, then in aux folder
            search_paths = (
                [sensors_path, aux_path] if sensors_path.exists() else [aux_path]
            )
            for search_path in search_paths:
                # Try exact pattern first: TIMESTAMP_CODE.*
                matching_files = list(
                    search_path.glob(f"{timestamp_prefix}_{file_code}.*")
                )
                if matching_files:
                    files_found.extend(matching_files)

                # Try with underscore: TIMESTAMP_CODE_*
                matching_files = list(
                    search_path.glob(f"{timestamp_prefix}_{file_code}_*")
                )
                if matching_files:
                    files_found.extend(matching_files)

            # Remove duplicates while preserving order and check file sizes
            files_found = list(dict.fromkeys(files_found))
            valid_files = []
            for f in files_found:
                try:
                    file_size = f.stat().st_size
                    # Only include files with size > 0
                    if file_size > 0:
                        valid_files.append(f)
                except (OSError, FileNotFoundError):
                    pass

            files_found = sorted(valid_files)

            sensor_key = file_code.lower()
            if config_type == "INERTIAL":
                sensor_key = f"imu_{sensor_key.lower()}"

            availability[sensor_key] = {
                "configured": config_has_sensor,
                "files_exist": len(files_found) > 0,
                "file_paths": [str(f) for f in files_found],
                "file_count": len(files_found),
                "friendly_name": friendly_name,
                "config_type": config_type,
                "is_multi_file": is_multi_file,
            }

            if is_imu:
                imu_components[sensor_key] = availability[sensor_key]

        # Check for camera
        camera_files = []

        # Check in camera subfolder
        if camera_path.exists():
            video_files = list(camera_path.glob(f"{timestamp_prefix}_video.*"))
            photo_files = list(camera_path.glob(f"{timestamp_prefix}_photo.*"))
            camera_files.extend(video_files)
            camera_files.extend(photo_files)

        # Check in aux folder as fallback
        video_files = list(aux_path.glob(f"{timestamp_prefix}_video.*"))
        photo_files = list(aux_path.glob(f"{timestamp_prefix}_photo.*"))
        camera_files.extend(video_files)
        camera_files.extend(photo_files)

        # Remove duplicates while preserving order and check file sizes
        camera_files = list(dict.fromkeys(camera_files))
        valid_camera_files = []
        for f in camera_files:
            try:
                file_size = f.stat().st_size
                # Only include files with size > 0
                if file_size > 0:
                    valid_camera_files.append(f)
            except (OSError, FileNotFoundError):
                pass

        camera_files = sorted(valid_camera_files)

        camera_configured = self.config and self.config.get("camera") is not None
        availability["camera"] = {
            "configured": camera_configured,
            "files_exist": len(camera_files) > 0,
            "file_paths": [str(f) for f in camera_files],
            "file_count": len(camera_files),
            "friendly_name": "Camera",
            "config_type": "CAMERA",
            "is_multi_file": True,
        }

        return availability

    def load_all(self):
        """
        Attempt to load all available sensor data.

        Uses sensor_availability to determine which sensors have files,
        then loads only those sensors that exist.
        Handles the folder structure: aux/sensors/ and aux/camera/
        """
        # Get sensor availability
        availability = self.get_sensor_availability()

        aux_path = Path(self.dirpath)
        sensors_path = aux_path / "sensors"
        camera_path = aux_path / "camera"

        # Load GPS if files exist
        if availability.get("gps", {}).get("files_exist"):
            try:
                gps_file = availability["gps"]["file_paths"][0]
                self.gps = GPS(gps_file)
                self.gps.load_data()
                logger.info(f"Loaded GPS data from {Path(gps_file).name}")
            except Exception as e:
                logger.debug(f"Could not load GPS: {e}")

        # Load IMU components if files exist
        imu_files_available = any(
            availability.get(f"imu_{comp}", {}).get("files_exist")
            for comp in ["acc", "gyr", "mag", "bar"]
        )
        if imu_files_available:
            try:
                imu_path = str(sensors_path) if sensors_path.exists() else self.dirpath
                self.imu = IMU(imu_path)
                self.imu.load_all()
                logger.info(
                    "Loaded IMU data (Accelerometer, Gyroscope, Magnetometer, Barometer)"
                )
            except Exception as e:
                logger.debug(f"Could not load IMU: {e}")

        # Load ADC if files exist
        if availability.get("adc", {}).get("files_exist"):
            try:
                adc_file = availability["adc"]["file_paths"][0]
                self.adc = ADC(adc_file, gain_config=self.adc_gain_config)
                self.adc.load_data()
                logger.info(f"Loaded ADC data from {Path(adc_file).name}")
            except Exception as e:
                logger.debug(f"Could not load ADC: {e}")

        # Load Camera if files exist
        if availability.get("camera", {}).get("files_exist"):
            try:
                camera_file = availability["camera"]["file_paths"][0]
                self.camera = Camera(camera_file)
                self.camera.load_data()
                logger.info(f"Loaded Camera data from {Path(camera_file).name}")
            except Exception as e:
                logger.debug(f"Could not load Camera: {e}")

        # Load Inclinometer if files exist
        if availability.get("inc", {}).get("files_exist"):
            try:
                inclinometer_path = (
                    str(sensors_path) if sensors_path.exists() else self.dirpath
                )
                self.inclinometer = Inclinometer(inclinometer_path)
                self.inclinometer.load_data()
                logger.info(
                    f"Loaded Inclinometer data ({availability['inc']['file_count']} files)"
                )
            except Exception as e:
                logger.debug(f"Could not load Inclinometer: {e}")

    def get_sensor_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about loaded sensors including sample rates and time ranges.

        Returns:
            Dictionary with sensor names as keys and info dicts as values.
            Each info dict contains: 'samples', 'duration_s', 'sample_rate_hz',
            't_start', 't_end'
        """
        info = {}

        # GPS
        if self.gps is not None and self.gps.data is not None:
            data = self.gps.data
            if "timestamp" in data.columns and len(data) > 1:
                timestamps = data["timestamp"].to_numpy()
                duration = timestamps[-1] - timestamps[0]
                info["gps"] = {
                    "samples": len(data),
                    "duration_s": float(duration),
                    "sample_rate_hz": len(data) / duration if duration > 0 else 0,
                    "t_start": float(timestamps[0]),
                    "t_end": float(timestamps[-1]),
                }

        # ADC
        if self.adc is not None and self.adc.data is not None:
            data = self.adc.data
            if "timestamp" in data.columns and len(data) > 1:
                timestamps = data["timestamp"].to_numpy()
                duration = timestamps[-1] - timestamps[0]
                info["adc"] = {
                    "samples": len(data),
                    "duration_s": float(duration),
                    "sample_rate_hz": len(data) / duration if duration > 0 else 0,
                    "t_start": float(timestamps[0]),
                    "t_end": float(timestamps[-1]),
                }

        # Inclinometer
        if self.inclinometer is not None and self.inclinometer.data is not None:
            data = self.inclinometer.data
            if "timestamp" in data.columns and len(data) > 1:
                timestamps = data["timestamp"].to_numpy()
                duration = timestamps[-1] - timestamps[0]
                info["inclinometer"] = {
                    "samples": len(data),
                    "duration_s": float(duration),
                    "sample_rate_hz": len(data) / duration if duration > 0 else 0,
                    "t_start": float(timestamps[0]),
                    "t_end": float(timestamps[-1]),
                }

        # IMU sensors
        if self.imu is not None:
            for sensor_name in [
                "barometer",
                "accelerometer",
                "gyroscope",
                "magnetometer",
            ]:
                sensor = getattr(self.imu, sensor_name, None)
                if sensor is not None and sensor.data is not None:
                    data = sensor.data
                    if "timestamp" in data.columns and len(data) > 1:
                        timestamps = data["timestamp"].to_numpy()
                        duration = timestamps[-1] - timestamps[0]
                        info[f"imu_{sensor_name}"] = {
                            "samples": len(data),
                            "duration_s": float(duration),
                            "sample_rate_hz": (
                                len(data) / duration if duration > 0 else 0
                            ),
                            "t_start": float(timestamps[0]),
                            "t_end": float(timestamps[-1]),
                        }

        self._sensor_info = info
        return info

    def get_max_sample_rate(self) -> float:
        """
        Get the maximum sample rate among all loaded sensors.

        Returns:
            Maximum sample rate in Hz.
        """
        if not self._sensor_info:
            self.get_sensor_info()

        if not self._sensor_info:
            return 0.0

        return max(s["sample_rate_hz"] for s in self._sensor_info.values())

    def get_common_time_range(self) -> Tuple[float, float]:
        """
        Get the common time range where all sensors have data.

        Returns:
            Tuple of (t_start, t_end) in seconds.
        """
        if not self._sensor_info:
            self.get_sensor_info()

        if not self._sensor_info:
            return (0.0, 0.0)

        t_start = max(s["t_start"] for s in self._sensor_info.values())
        t_end = min(s["t_end"] for s in self._sensor_info.values())

        return (t_start, t_end)

    def summary(self) -> str:
        """
        Get a summary of loaded sensors and their properties.

        Returns:
            Formatted string with sensor information.
        """
        info = self.get_sensor_info()
        if not info:
            return "No sensors loaded."

        lines = ["Payload Sensor Summary", "=" * 50]

        for name, data in info.items():
            lines.append(f"\n{name.upper()}")
            lines.append(f"  Samples: {data['samples']}")
            lines.append(f"  Duration: {data['duration_s']:.2f} s")
            lines.append(f"  Sample Rate: {data['sample_rate_hz']:.2f} Hz")
            lines.append(f"  Time Range: {data['t_start']:.3f} - {data['t_end']:.3f} s")

        lines.append("\n" + "=" * 50)
        lines.append(
            "To synchronize payload data with drone data, use the Synchronizer class:\n"
        )
        lines.append("  from pils.synchronizer import Synchronizer")
        lines.append("  sync = Synchronizer()")
        lines.append("  sync.add_source('payload', payload_data)")
        lines.append("  sync.add_source('drone', drone_data)")
        lines.append("  synchronized = sync.synchronize()")

        return "\n".join(lines)
