"""Containers of the loaded data of a flight: raw, drone and payload tables."""

import logging
from typing import Any, Optional, Union, overload

import polars as pl

logger = logging.getLogger(__name__)


class RawData:
    """
    Container for raw flight data including drone telemetry and payload sensors.

    Attributes
    ----------
    drone_data : DroneData
        Drone telemetry data (initialized empty, populated by add_drone_data())
    payload_data : PayloadData
        Payload sensor data (initialized empty, populated by add_sensor_data())

    Examples
    --------
    >>> # Access drone data
    >>> drone_telemetry = raw_data.drone_data.drone
    >>> # Access payload sensors
    >>> gps_data = raw_data.payload_data.gps
    >>> # Dictionary-style access
    >>> drone_telemetry = raw_data['drone_data']['drone']
    >>> gps_data = raw_data['payload']['gps']
    >>> # Print summary
    >>> print(raw_data)
    """

    def __init__(self):
        """Initialize empty RawData container with empty data objects."""
        self.drone_data: DroneData = DroneData(None, None)
        self.payload_data: PayloadData = PayloadData()

    def __getitem__(self, key):
        """
        Dictionary-style access to raw data components.

        Parameters
        ----------
        key : str
            Key to access ('drone_data', 'payload_data', or 'payload')

        Returns
        -------
        object
            Corresponding data object

        Raises
        ------
        KeyError
            If key is not found
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
        # Check if drone data has been loaded
        if len(self.drone_data.drone) > 0 or len(self.drone_data.litchi) > 0:
            output.append("=== DRONE DATA ===")
            output.append(str(self.drone_data))
        # Check if payload data has sensors loaded
        if len(self.payload_data.list_loaded_sensors()) > 0:
            output.append("\n=== PAYLOAD DATA ===")
            output.append(str(self.payload_data))
        return "\n".join(output) if output else "No data loaded"


class DroneData:
    """
    Container for drone telemetry data.

    Attributes
    ----------
    drone : Union[Dict[str, pl.DataFrame], pl.DataFrame, None]
        Drone telemetry data
    litchi : Optional[pl.DataFrame]
        Litchi flight log data (DJI only)

    Examples
    --------
    >>> # Access drone telemetry
    >>> telemetry = drone_data.drone
    >>> print(telemetry.columns)
    >>> # Filter by altitude
    >>> high_flight = telemetry.filter(pl.col('altitude') > 50)
    >>> # Access Litchi waypoints
    >>> if drone_data.litchi is not None:
    ...     waypoints = drone_data.litchi
    ...     print(waypoints.head())
    >>> # Dictionary-style access
    >>> telemetry = drone_data['drone']
    >>> waypoints = drone_data['litchi']
    """

    def __init__(
        self,
        drone_df: Union[dict[str, "pl.DataFrame"], "pl.DataFrame", None] = None,
        litchi_df: Optional["pl.DataFrame"] = None,
        records: dict[str, "pl.DataFrame"] | None = None,
    ) -> None:
        """
        Initialize DroneData container.

        Parameters
        ----------
        drone_df : Union[Dict[str, pl.DataFrame], pl.DataFrame, None], default=None
            Drone telemetry data.
            Can be a single DataFrame or a dict of DataFrames keyed by sensor name.
        litchi_df : Optional[pl.DataFrame], default=None
            Litchi flight log DataFrame
        records : dict[str, pl.DataFrame], optional
            Other DJI .dat records (IMU attitude, motors, battery, ...), one
            table per record, with ``tick`` and drone-clock UTC ``timestamp``
        """
        # Initialize with empty DataFrames to avoid Optional types
        self.drone: dict[str, pl.DataFrame] | pl.DataFrame = (
            drone_df if drone_df is not None else pl.DataFrame()
        )
        self.litchi: pl.DataFrame = (
            litchi_df if litchi_df is not None else pl.DataFrame()
        )
        self.records: dict[str, pl.DataFrame] = records or {}

    def __getitem__(self, key: str) -> Union["pl.DataFrame", dict[str, "pl.DataFrame"]]:
        """
        Dictionary-style access to drone data.

        Parameters
        ----------
        key : str
            Key to access ('drone' or 'litchi')

        Returns
        -------
        Union[pl.DataFrame, Dict[str, pl.DataFrame]]
            Corresponding DataFrame

        Raises
        ------
        KeyError
            If key is not found
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

    Examples
    --------
    >>> # Access sensors as attributes
    >>> gps_data = payload_data.gps
    >>> imu_data = payload_data.imu
    >>> adc_data = payload_data.adc
    >>> # Access sensors using dictionary-style
    >>> gps_data = payload_data['gps']
    >>> imu_data = payload_data['imu']
    >>> # List all loaded sensors
    >>> sensors = payload_data.list_loaded_sensors()
    >>> print(f"Loaded sensors: {sensors}")
    >>> # Iterate over all sensors
    >>> for sensor_name in payload_data.list_loaded_sensors():
    ...     sensor_data = getattr(payload_data, sensor_name)
    ...     print(f"{sensor_name}: {sensor_data.shape}")
    >>> # Print summary
    >>> print(payload_data)
    """

    def __init__(self):
        """Initialize empty PayloadData container.

        Sensor attributes are set dynamically via add_sensor_data().
        """
        pass

    def __getattr__(self, name: str) -> Any:
        """
        Access sensor data attributes.

        This method is called when an attribute is not found through normal lookup.
        It provides better error messages for missing sensors.

        Parameters
        ----------
        name : str
            Sensor name (gps, imu, adc, inclinometer, camera, etc.)

        Returns
        -------
        Any
            Sensor data (typically pl.DataFrame or sensor object)

        Raises
        ------
        AttributeError
            If sensor not loaded. Error message includes list of available sensors.

        Examples
        --------
        >>> gps_data = payload_data.gps  # Returns GPS DataFrame if loaded
        >>> # If sensor not loaded, raises AttributeError with helpful message
        """
        # This will be called only if attribute doesn't exist via normal lookup
        available = self.list_loaded_sensors()
        if available:
            raise AttributeError(
                f"Sensor '{name}' not loaded. Available sensors: {available}"
            )
        else:
            raise AttributeError(
                f"Sensor '{name}' not loaded. No sensors currently loaded."
            )

    @overload
    def __setattr__(self, name: str, value: pl.DataFrame) -> None: ...

    @overload
    def __setattr__(self, name: str, value: Any) -> None: ...

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set sensor data attribute dynamically.

        Parameters
        ----------
        name : str
            Sensor name (gps, imu, adc, inclinometer, etc.)
        value : Any
            Sensor data (typically pl.DataFrame or sensor object)
        """
        object.__setattr__(self, name, value)

    def __getitem__(self, key: str) -> Any:
        """
        Dictionary-style access to sensor data.

        Parameters
        ----------
        key : str
            Sensor name to access

        Returns
        -------
        Any
            Sensor data object (DataFrame or other sensor object)

        Raises
        ------
        KeyError
            If sensor is not found

        Examples
        --------
        >>> gps = payload_data['gps']
        >>> imu = payload_data['imu']
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Sensor '{key}' not found")

    def __contains__(self, key: str) -> bool:
        """
        Check if a sensor exists in payload data.

        Parameters
        ----------
        key : str
            Sensor name to check

        Returns
        -------
        bool
            True if sensor exists, False otherwise

        Examples
        --------
        >>> if 'gps' in payload_data:
        ...     print("GPS sensor available")
        """
        return hasattr(self, key)

    def list_loaded_sensors(self) -> list[str]:
        """
        List all currently loaded sensors.

        Returns
        -------
        List[str]
            List of sensor names

        Examples
        --------
        >>> sensors = payload_data.list_loaded_sensors()
        >>> print(sensors)
        ['gps', 'imu', 'adc', 'inclinometer']
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
