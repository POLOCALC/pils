from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pils.utils.tools import get_path_from_keyword
from pils.drones.DJIDrone import DJIDrone
from pils.drones.BlackSquareDrone import BlackSquareDrone
from pils.drones.litchi import Litchi

from pils.sensors.sensors import sensor_config


class Flight:
    """
    Flight data container stored in RAM for maximum speed.

    This class provides a hierarchical structure to store and access drone flight data
    and sensor payloads. Data is stored in RAM for fast access using both attribute
    and dictionary-style notation.

    Attributes:
        flight_info (Dict): Dictionary containing flight configuration paths
        flight_path (Path): Path to the flight directory
        metadata (Dict): Flight metadata (duration, date, conditions, etc.)
        raw_data (RawData): Container for drone and payload sensor data
        adc_gain_config (Optional): Configuration for ADC gain settings

    Examples:
        >>> # Create a flight instance
        >>> flight_info = {
        ...     "drone_data_folder_path": "/data/flight_001/drone",
        ...     "aux_data_folder_path": "/data/flight_001/aux"
        ... }
        >>> flight = Flight(flight_info)
        >>>
        >>> # Add metadata
        >>> flight.set_metadata({
        ...     'flight_time': '2025-01-28 14:30:00',
        ...     'duration': 1800,
        ...     'weather': 'clear'
        ... })
        >>>
        >>> # Load drone data (auto-detects DJI or BlackSquare)
        >>> flight.add_drone_data(dji_drone_loader='dat')
        >>>
        >>> # Load sensor data
        >>> flight.add_sensor_data(['gps', 'imu', 'adc'])
        >>>
        >>> # Access data using attributes
        >>> drone_df = flight.raw_data.drone_data.drone
        >>> gps_df = flight.raw_data.payload_data.gps
        >>>
        >>> # Or use dictionary-style access (same speed!)
        >>> drone_df = flight['raw_data']['drone_data']['drone']
        >>> gps_df = flight['raw_data']['payload']['gps']
        >>>
        >>> # Perform operations on the data
        >>> high_altitude = drone_df.filter(pl.col('altitude') > 100)
        >>> print(f"Points above 100m: {len(high_altitude)}")
    """

    def __init__(self, flight_info: Dict):
        """
        Initialize a Flight data container.

        Args:
            flight_info (Dict): Dictionary containing at minimum:
                - 'drone_data_folder_path': Path to drone data folder
                - 'aux_data_folder_path': Path to auxiliary sensor data folder

        Examples:
            >>> flight_info = {
            ...     "drone_data_folder_path": "/mnt/data/flight_001/drone",
            ...     "aux_data_folder_path": "/mnt/data/flight_001/aux"
            ... }
            >>> flight = Flight(flight_info)
        """
        self.flight_info = flight_info
        self.flight_path = Path(flight_info["drone_data_folder_path"]).parent
        self.metadata = {}
        self.set_metadata()

        self.raw_data = RawData()
        self.adc_gain_config = None

    def set_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set flight metadata.

        Args:
            metadata (Dict[str, Any]): Dictionary containing metadata fields
                such as flight_time, duration, weather conditions, pilot info, etc.

        Examples:
            >>> flight.set_metadata({
            ...     'flight_time': '2025-01-28 14:30:00',
            ...     'duration': 1800,
            ...     'pilot': 'John Doe',
            ...     'weather': 'clear',
            ...     'temperature': 22.5
            ... })
            >>> print(flight.metadata['flight_time'])
            '2025-01-28 14:30:00'
        """
        # Allow caller-provided metadata or derive from self.flight_info
        info_source: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            info_source = metadata
        elif isinstance(self.flight_info, dict):
            info_source = self.flight_info

        # Support both `takeoff_time`/`landing_time` and `takeoff_datetime`/`landing_datetime`
        takeoff = info_source.get("takeoff_time") or info_source.get("takeoff_datetime")
        landing = info_source.get("landing_time") or info_source.get("landing_datetime")

        if takeoff is not None and landing is not None:
            try:
                self.metadata["takeoff_time"] = takeoff
                # If subtraction works, store duration; otherwise keep raw landing
                self.metadata["flight_time"] = landing - takeoff
            except Exception:
                self.metadata["takeoff_time"] = takeoff
                self.metadata["landing_time"] = landing

        # Optional flight name
        if "flight_name" in info_source:
            self.metadata["flight_name"] = info_source.get("flight_name")

    def _detect_drone_model(self, drone_folder: str) -> str:
        """
        Auto-detect drone model from data folder structure.

        Args:
            drone_folder (str): Path to the drone data folder

        Returns:
            str: Detected drone model ('dji' or 'blacksquare')

        Note:
            This is an internal method. Defaults to 'dji' if detection fails.
        """
        # Prefer resolving the drone model from the stout inventory when a
        # `drone_id` is available in the flight info. This avoids relying on
        # filename heuristics and does not modify the database.
        try:
            drone_id = (
                self.flight_info.get("drone_id")
                if isinstance(self.flight_info, dict)
                else None
            )
            if drone_id:
                try:
                    from stout.services.inventory.service import InventoryService

                    inventory = InventoryService()
                    item = inventory.get_item_by_id(drone_id)
                    specs = (
                        item.get("specifications")
                        if isinstance(item.get("specifications"), dict)
                        else {}
                    )
                    model_val = (
                        specs.get("model") or item.get("name") or item.get("category")
                    )
                    if isinstance(model_val, str):
                        m = model_val.lower()
                        if "matrice" in m:
                            return "dji"
                        if "black" in m or "blacksquare" in m:
                            return "blacksquare"
                        # If we can't map to a known driver, return the raw model string
                        return model_val
                except Exception:
                    # If inventory lookup fails, fall back to folder heuristics below
                    pass

            # Fallback: look for drone-specific patterns in filenames
            dji_pattern = get_path_from_keyword(str(drone_folder), "DJI")
            if dji_pattern:
                return "dji"

            blacksquare_pattern = get_path_from_keyword(
                str(drone_folder), "blacksquare"
            )
            if blacksquare_pattern:
                return "blacksquare"

            # Default to DJI if nothing matches
            return "dji"
        except Exception:
            return "dji"

    def add_drone_data(
        self,
        dji_dat_loader: bool = True,
        drone_model: Optional[str] = None,
    ) -> "DroneData":
        """
        Load drone telemetry data based on auto-detected drone model.

        Automatically detects whether the drone is DJI or BlackSquare and loads
        the appropriate data format. For DJI drones, also loads Litchi flight logs
        if available.

        Args:
            dji_drone_loader (str, optional): Data loader type for DJI drones.
                Options: 'dat', 'csv'. Defaults to 'dat'.

        Returns:
            DroneData: Reference to the loaded drone data

        Raises:
            ValueError: If an unknown drone model is detected

        Examples:
            >>> # Load DJI drone data using .DAT files
            >>> flight.add_drone_data(dji_drone_loader='dat')
            >>>
            >>> # Access drone telemetry
            >>> print(flight.raw_data.drone_data.drone.head())
            >>>
            >>> # Access Litchi waypoint data (if DJI)
            >>> if flight.raw_data.drone_data.litchi is not None:
            ...     print(flight.raw_data.drone_data.litchi.head())
            >>>
            >>> # Alternative: use dictionary access
            >>> drone_data = flight['raw_data']['drone_data']['drone']
        """

        # Resolve drone folder
        if not isinstance(self.flight_info, dict):
            raise ValueError(
                "flight_info must be a dict containing 'drone_data_folder_path'"
            )

        drone_folder = self.flight_info.get("drone_data_folder_path")

        if not drone_model:
            drone_model = self._detect_drone_model(drone_folder)

        # Find candidate files
        available_files = glob.glob(str(drone_folder) + "/*")
        drone_data_path = None
        litchi_path = None

        for file in available_files:
            fname = file.lower()
            if (
                fname.endswith("drone.dat")
                and dji_dat_loader
                and "dji" in drone_model.lower()
            ):
                drone_data_path = file
            elif fname.endswith("drone.csv"):
                drone_data_path = file
            if fname.endswith("litchi.csv") and "dji" in drone_model.lower():
                litchi_data_path = file

        # Load according to detected model
        if isinstance(drone_model, str) and "dji" in drone_model.lower():
            if drone_data_path is None:
                # try passing folder to DJIDrone which may discover files
                drone = DJIDrone(drone_folder)
            else:
                drone = DJIDrone(drone_data_path)
            drone.load_data(use_dat=dji_dat_loader)
            drone_data = drone.data

            # load litchi if available (prefer explicit litchi file path)
            litchi_loader = Litchi(litchi_data_path)
            litchi_loader.load_data()
            litchi_data = litchi_loader.data

        elif isinstance(drone_model, str) and (
            "black" in drone_model.lower() or "blacksquare" in drone_model.lower()
        ):
            drone = BlackSquareDrone(drone_folder)
            drone.load_data()
            drone_data = drone.data
            litchi_data = None

        else:
            try:
                drone = DJIDrone(drone_data_path or drone_folder)
                drone.load_data(use_dat=dji_dat_loader)
                drone_data = drone.data
                litchi_loader = Litchi(litchi_path or drone_folder)
                litchi_loader.load_data()
                litchi_data = litchi_loader.data
            except Exception:
                drone = BlackSquareDrone(drone_folder)
                drone.load_data()
                drone_data = drone.data
                litchi_data = None

        self.raw_data.drone_data = DroneData(drone_data, litchi_data)
        return self.raw_data.drone_data

    def _read_sensor_data(self, sensor_name: str, sensor_folder: Path) -> Optional[Any]:
        """
        Read sensor data based on sensor type.

        Args:
            sensor_name (str): Name of the sensor ('gps', 'imu', 'adc', 'inclinometer')
            sensor_folder (Path): Path to the sensors folder

        Returns:
            Sensor object or None if sensor not found

        Note:
            This is an internal method used by add_sensor_data().
        """
        result = None

        config = sensor_config.get(sensor_name.lower())

        if config:
            sensor = config["class"](sensor_folder)

            getattr(sensor, config["load_method"])()
            result = sensor.data

        return result

    def add_sensor_data(self, sensor_name: Union[str, List[str]]) -> None:
        """
        Load sensor data from the payload.

        Loads one or more sensors from the auxiliary data folder. Sensors are
        automatically detected and loaded based on their type.

        Args:
            sensor_name (Union[str, List[str]]): Single sensor name or list of sensor names.
                Supported sensors: 'gps', 'imu', 'adc', 'inclinometer'

        Examples:
            >>> # Load a single sensor
            >>> flight.add_sensor_data('gps')
            >>> print(flight.raw_data.payload_data.gps)
            >>>
            >>> # Load multiple sensors at once
            >>> flight.add_sensor_data(['gps', 'imu', 'adc'])
            >>>
            >>> # Access sensor data
            >>> gps_data = flight.raw_data.payload_data.gps
            >>> imu_data = flight.raw_data.payload_data.imu
            >>>
            >>> # Or use dictionary-style
            >>> gps_data = flight['raw_data']['payload']['gps']
            >>>
            >>> # Filter GPS data
            >>> high_accuracy = gps_data.filter(pl.col('accuracy') < 5.0)
            >>>
            >>> # List all loaded sensors
            >>> print(flight.raw_data.payload_data.list_loaded_sensors())
        """
        sensor_path = Path(self.flight_info["aux_data_folder_path"]) / "sensors"

        if self.raw_data.payload_data is None:
            self.raw_data.payload_data = PayloadData()

        if isinstance(sensor_name, str):
            sensor_name = [sensor_name]

        for sensor in sensor_name:
            sensor_data = self._read_sensor_data(sensor, sensor_path)
            setattr(self.raw_data.payload_data, sensor, sensor_data)

    def __getitem__(self, key):
        """
        Dictionary-style access to flight data.

        Args:
            key (str): Key to access ('raw_data' or 'metadata')

        Returns:
            Corresponding data object

        Raises:
            KeyError: If key is not found

        Examples:
            >>> # Access raw data
            >>> raw_data = flight['raw_data']
            >>>
            >>> # Access metadata
            >>> metadata = flight['metadata']
            >>>
            >>> # Chain dictionary access
            >>> drone_data = flight['raw_data']['drone_data']['drone']
        """
        if key == "raw_data":
            return self.raw_data
        elif key == "metadata":
            return self.metadata
        else:
            raise KeyError(f"Key '{key}' not found")


class RawData:
    """
    Container for raw flight data including drone telemetry and payload sensors.

    Attributes:
        drone_data (Optional[DroneData]): Drone telemetry data
        payload_data (Optional[PayloadData]): Payload sensor data

    Examples:
        >>> # Access drone data
        >>> drone_telemetry = raw_data.drone_data.drone
        >>>
        >>> # Access payload sensors
        >>> gps_data = raw_data.payload_data.gps
        >>>
        >>> # Dictionary-style access
        >>> drone_telemetry = raw_data['drone_data']['drone']
        >>> gps_data = raw_data['payload']['gps']
        >>>
        >>> # Print summary
        >>> print(raw_data)
    """

    def __init__(self):
        """Initialize empty RawData container."""
        self.drone_data = None
        self.payload_data = None

    def __getitem__(self, key):
        """
        Dictionary-style access to raw data components.

        Args:
            key (str): Key to access ('drone_data', 'payload_data', or 'payload')

        Returns:
            Corresponding data object

        Raises:
            KeyError: If key is not found
        """
        if key == "drone_data":
            return self.drone_data
        elif key == "payload_data" or key == "payload":
            return self.payload_data
        else:
            raise KeyError(f"Key '{key}' not found")

    def __repr__(self):
        """Return string representation of loaded data."""
        output = []
        if self.drone_data:
            output.append("=== DRONE DATA ===")
            output.append(str(self.drone_data))
        if self.payload_data:
            output.append("\n=== PAYLOAD DATA ===")
            output.append(str(self.payload_data))
        return "\n".join(output) if output else "No data loaded"


class DroneData:
    """
    Container for drone telemetry data.

    Attributes:
        drone (Optional[pl.DataFrame]): Drone telemetry data
        litchi (Optional[pl.DataFrame]): Litchi flight log data (DJI only)

    Examples:
        >>> # Access drone telemetry
        >>> telemetry = drone_data.drone
        >>> print(telemetry.columns)
        >>>
        >>> # Filter by altitude
        >>> high_flight = telemetry.filter(pl.col('altitude') > 50)
        >>>
        >>> # Access Litchi waypoints
        >>> if drone_data.litchi is not None:
        ...     waypoints = drone_data.litchi
        ...     print(waypoints.head())
        >>>
        >>> # Dictionary-style access
        >>> telemetry = drone_data['drone']
        >>> waypoints = drone_data['litchi']
    """

    def __init__(
        self,
        drone_df: Union[Dict[str, "pl.DataFrame"], "pl.DataFrame", None] = None,
        litchi_df: Optional["pl.DataFrame"] = None,
    ) -> None:
        """
        Initialize DroneData container.

        Args:
            drone_df (Union[Dict[str, pl.DataFrame], pl.DataFrame, None]): Drone telemetry data.
                Can be a single DataFrame or a dict of DataFrames keyed by sensor name.
            litchi_df (Optional[pl.DataFrame]): Litchi flight log DataFrame
        """
        self.drone = drone_df
        self.litchi = litchi_df

    def __getitem__(self, key):
        """
        Dictionary-style access to drone data.

        Args:
            key (str): Key to access ('drone' or 'litchi')

        Returns:
            Corresponding DataFrame

        Raises:
            KeyError: If key is not found
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found")

    def __repr__(self):
        """Return string representation of drone data."""
        output = []
        if self.drone is not None:
            output.append(f"Drone:\n{self.drone}")
        if self.litchi is not None:
            output.append(f"\nLitchi:\n{self.litchi}")
        return "\n".join(output)


class PayloadData:
    """
    Container for payload sensor data with dynamic attributes.

    Sensors are added as attributes dynamically, allowing for flexible
    sensor configurations. Each sensor is accessible both as an attribute
    and through dictionary-style access.

    Examples:
        >>> # Access sensors as attributes
        >>> gps_data = payload_data.gps
        >>> imu_data = payload_data.imu
        >>> adc_data = payload_data.adc
        >>>
        >>> # Access sensors using dictionary-style
        >>> gps_data = payload_data['gps']
        >>> imu_data = payload_data['imu']
        >>>
        >>> # List all loaded sensors
        >>> sensors = payload_data.list_loaded_sensors()
        >>> print(f"Loaded sensors: {sensors}")
        >>>
        >>> # Iterate over all sensors
        >>> for sensor_name in payload_data.list_loaded_sensors():
        ...     sensor_data = getattr(payload_data, sensor_name)
        ...     print(f"{sensor_name}: {sensor_data.shape}")
        >>>
        >>> # Print summary
        >>> print(payload_data)
    """

    def __init__(self):
        """Initialize empty PayloadData container."""
        pass

    def __getitem__(self, key):
        """
        Dictionary-style access to sensor data.

        Args:
            key (str): Sensor name to access

        Returns:
            Sensor data object

        Raises:
            KeyError: If sensor is not found

        Examples:
            >>> gps = payload_data['gps']
            >>> imu = payload_data['imu']
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Sensor '{key}' not found")

    def list_loaded_sensors(self) -> List[str]:
        """
        List all currently loaded sensors.

        Returns:
            List[str]: List of sensor names

        Examples:
            >>> sensors = payload_data.list_loaded_sensors()
            >>> print(sensors)
            ['gps', 'imu', 'adc', 'inclinometer']
            >>>
            >>> # Check if specific sensor is loaded
            >>> if 'gps' in payload_data.list_loaded_sensors():
            ...     print("GPS data available")
        """
        return [
            attr
            for attr in dir(self)
            if not attr.startswith("_")
            and attr != "list_loaded_sensors"
            and not callable(getattr(self, attr))
        ]

    def __repr__(self):
        """Return string representation of all loaded sensors."""
        output = []
        for sensor_name in self.list_loaded_sensors():
            output.append(f"{sensor_name}:\n{getattr(self, sensor_name)}\n")
        return "\n".join(output) if output else "No sensors loaded"
