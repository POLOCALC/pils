import glob
import json
import logging
import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import h5py
import polars as pl

from pils import schema
from pils.drones.BlackSquareDrone import BlackSquareDrone
from pils.drones.DJIDrone import DJIDrone
from pils.drones.litchi import Litchi
from pils.sensors.camera import Camera
from pils.sensors.sensors import sensor_config
from pils.synchronizer import Synchronizer
from pils.utils.hdf5 import read_table, write_table
from pils.utils.tools import get_path_from_keyword, standardize_datetimes

logger = logging.getLogger(__name__)


def _get_current_timestamp() -> str:
    """
    Get current timestamp in rev_YYYYMMDD_hhmmss format.

    Returns
    -------
    str
        Timestamp string with seconds precision
    """
    return datetime.now().strftime("rev_%Y%m%d_%H%M%S")


def _get_package_version() -> str:
    """
    Get PILS package version.

    Returns
    -------
    str
        Version string or 'unknown'
    """
    try:
        import pils

        return getattr(pils, "__version__", "unknown")
    except Exception:
        return "unknown"


def _standardize(data: Any, kind: str | None = None) -> Any:
    """A loaded table, or dict of tables, in the schema (:mod:`pils.schema`).

    Columns get the schema's names and units, datetimes are UTC microseconds,
    and the tables of a dict get ``snake_case`` names.
    """
    if isinstance(data, pl.DataFrame):
        return standardize_datetimes(schema.canonical(data, kind))
    if isinstance(data, dict):
        return {
            schema.snake_case(key): _standardize(value, kind)
            for key, value in data.items()
        }
    return data


_LEGACY_SYNC_COORDINATES = {
    "latitude": "lat_deg",
    "longitude": "lon_deg",
    "altitude": "h_ell_m",
}
_SYNC_KINDS = {"drone": "dji", "reference_gps": "gps"}


def _legacy_sync_table(key: str, df: pl.DataFrame) -> tuple[str, pl.DataFrame]:
    """A synchronized table from a file written before the schema, renamed."""
    df = df.rename(
        {k: v for k, v in _LEGACY_SYNC_COORDINATES.items() if k in df.columns}
    )
    if key.startswith("drone_"):
        record = key.removeprefix("drone_")
        return f"drone_{schema.record_name(record)}", schema.canonical(
            df, record=record
        )
    kind = _SYNC_KINDS.get(key, key.split("_")[0])  # imu_barometer -> imu
    return key, schema.canonical(df, kind)


def _standardize_records(records: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """DJI .dat record tables in the schema, e.g. ``imu_atti_0``."""
    return {
        schema.record_name(name): standardize_datetimes(
            schema.canonical(table, record=name)
        )
        for name, table in records.items()
    }


def _parse_temporal_dtype(dtype: str) -> pl.DataType:
    """Polars temporal dtype from its string form, e.g. ``Datetime(time_unit='us', time_zone='UTC')``."""
    unit_match = re.search(r"time_unit='(\w+)'", dtype)
    unit = unit_match.group(1) if unit_match else "us"
    if dtype.startswith("Datetime"):
        zone_match = re.search(r"time_zone='([^']+)'", dtype)
        return pl.Datetime(unit, zone_match.group(1) if zone_match else None)  # type: ignore[arg-type]
    if dtype.startswith("Duration"):
        return pl.Duration(unit)  # type: ignore[arg-type]
    if dtype.startswith("Date"):
        return pl.Date()
    return pl.Time()


def _serialize_for_hdf5(obj: Any) -> Any:
    """
    Convert Python objects to HDF5-compatible types for attrs.

    Parameters
    ----------
    obj : Any
        Object to serialize

    Returns
    -------
    Any
        HDF5-compatible object
    """
    if obj is None:
        return "None"
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return json.dumps(obj)
    elif isinstance(obj, (list, tuple)):
        return json.dumps(list(obj))
    else:
        return str(obj)


def _deserialize_from_hdf5(value: Any, hint: str | None = None) -> Any:
    """
    Deserialize values from HDF5 attrs.

    Parameters
    ----------
    value : Any
        Value to deserialize
    hint : Optional[str], default=None
        Type hint (e.g., 'dict', 'list')

    Returns
    -------
    Any
        Deserialized object
    """
    if value == "None":
        return None
    if isinstance(value, (str, int, float, bool)):
        if hint == "dict" and isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        elif hint == "list" and isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
    return value


class Flight:
    """
    This class provides a hierarchical structure to store and access drone flight data
    and sensor payloads. Data is stored in RAM for fast access using both attribute
    and dictionary-style notation.

    Attributes
    ----------
    flight_info : Dict
        Dictionary containing flight configuration paths
    flight_path : Path
        Path to the flight directory
    metadata : Dict
        Flight metadata (duration, date, conditions, etc.)
    raw_data : RawData
        Container for drone and payload sensor data
    sync_data : Optional[dict[str, pl.DataFrame]]
        Synchronized flight data (populated after calling sync())
    adc_gain_config : Optional
        Configuration for ADC gain settings

    Examples
    --------
    >>> # Create a flight instance
    >>> flight_info = {
    ...     "drone_data_folder_path": "/data/flight_001/drone",
    ...     "aux_data_folder_path": "/data/flight_001/aux"
    ... }
    >>> flight = Flight(flight_info)
    >>> # Add metadata
    >>> flight.set_metadata({
    ...     'flight_time': '2025-01-28 14:30:00',
    ...     'duration': 1800,
    ...     'weather': 'clear'
    ... })
    >>> # Load drone data (auto-detects DJI or BlackSquare)
    >>> flight.add_drone_data(dji_drone_loader='dat')
    >>> # Load sensor data
    >>> flight.add_sensor_data(['gps', 'imu', 'adc'])
    >>> # Load camera data (Sony or Alvium)
    >>> flight.add_camera_data(use_photogrammetry=False, get_sony_angles=True)
    >>> # Access data using attributes
    >>> drone_df = flight.raw_data.drone_data.drone
    >>> gps_df = flight.raw_data.payload_data.gps
    >>> camera_df = flight.raw_data.payload_data.camera
    >>> # Or use dictionary-style access (same speed!)
    >>> drone_df = flight['raw_data']['drone_data']['drone']
    >>> gps_df = flight['raw_data']['payload']['gps']
    >>> camera_df = flight['raw_data']['payload']['camera']
    >>> # Synchronize all data sources
    >>> sync_df = flight.sync(target_rate={'drone': 10.0, 'payload': 100.0})
    >>> # Perform operations on the data
    >>> high_altitude = drone_df.filter(pl.col('altitude') > 100)
    >>> print(f"Points above 100m: {len(high_altitude)}")
    """

    def __init__(self, flight_info: dict[str, Any]):
        """
        Initialize a Flight data container.

        Parameters
        ----------
        flight_info : Dict
            Dictionary containing at minimum:
            - 'drone_data_folder_path': Path to drone data folder
            - 'aux_data_folder_path': Path to auxiliary sensor data folder

        Examples
        --------
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
        self.sync_data: dict[str, pl.DataFrame] | None = None
        self.sync_offsets: dict[str, float] = {}
        self.sync_rates: dict[str, float] = {}
        self.adc_gain_config = None
        # Set by the loaders; saved with the raw data
        self.__drone_model: str | None = None
        self.__inclinometer: str | None = None
        self.__camera_model: str | None = None
        self.__use_photogrammetry = False

    @classmethod
    def from_hdf5(
        cls,
        filepath: str | Path,
        sync_version: str | None | bool = None,
        load_raw: bool = True,
    ) -> "Flight":
        """
        Load flight data from HDF5 file.

        Loads metadata and raw_data hierarchy. Optionally loads a specific
        synchronized data version or the latest available version.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to HDF5 file
        sync_version : Union[str, None, bool], default=None
            Specific sync version to load (e.g., 'rev_20260202_1430').
            If None and synchronized data exists, loads latest version.
            Set to False to skip loading synchronized data.
        load_raw : bool, default=True
            If True, loads raw_data. If False, only loads metadata and sync data.

        Returns
        -------
        Flight
            Returns new Flight instance

        Raises
        ------
        ImportError
            If h5py is not installed
        FileNotFoundError
            If HDF5 file doesn't exist
        ValueError
            If requested sync version not found

        Examples
        --------
        >>> # Load from file
        >>> flight = Flight.from_hdf5('flight_001.h5')
        >>> # Load specific sync version
        >>> flight = Flight.from_hdf5('flight_001.h5', sync_version='rev_20260202_1430')
        >>> # Load only metadata and raw data
        >>> flight = Flight.from_hdf5('flight_001.h5', sync_version=False)
        """

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")

        with h5py.File(str(filepath), "r") as f:
            # Load metadata
            metadata_dict = {}
            flight_info_dict = {}
            if "metadata" in f:
                metadata_group = f["metadata"]
                assert isinstance(metadata_group, h5py.Group)
                for key in metadata_group.attrs:
                    if key.startswith("flight_info_"):
                        # Strip prefix and add to flight_info
                        clean_key = key.replace("flight_info_", "", 1)
                        flight_info_dict[clean_key] = metadata_group.attrs[key]
                    else:
                        metadata_dict[key] = metadata_group.attrs[key]

            # Create new instance
            flight = cls(
                flight_info=flight_info_dict if flight_info_dict else metadata_dict
            )

            # Load metadata
            if "metadata" in f:
                metadata_group = f["metadata"]
                assert isinstance(metadata_group, h5py.Group)
                flight._load_metadata_from_hdf5(metadata_group, flight)

            # Load raw data
            if load_raw and "raw_data" in f:
                raw_data_group = f["raw_data"]
                assert isinstance(raw_data_group, h5py.Group)
                flight._load_raw_data_from_hdf5(raw_data_group, flight)

            # Load sync_data if available
            if sync_version is not False and "sync_data" in f:
                sync_data_group = f["sync_data"]
                assert isinstance(sync_data_group, h5py.Group)
                available_versions = sorted(
                    [k for k in sync_data_group.keys() if k.startswith("rev_")]
                )
                if available_versions:
                    # Determine which version to load
                    if sync_version is None:
                        # Load latest version
                        sync_version = available_versions[-1]
                    elif sync_version not in available_versions:
                        raise ValueError(
                            f"Sync version '{sync_version}' not found. "
                            f"Available versions: {available_versions}"
                        )

                    revision_group = sync_data_group[sync_version]
                    assert isinstance(revision_group, h5py.Group)
                    # Load all datasets in the revision
                    sync_dict = {}
                    for key in revision_group.keys():
                        dataset_group = revision_group[key]
                        assert isinstance(dataset_group, h5py.Group)
                        df = flight._load_dataframe_from_hdf5(dataset_group)
                        if df is not None:
                            sync_dict[key] = df
                    if sync_dict:
                        flight.sync_data = dict(
                            _legacy_sync_table(key, df) for key, df in sync_dict.items()
                        )

        return flight

    @staticmethod
    def _load_metadata_from_hdf5(
        metadata_group: "h5py.Group", flight: "Flight"
    ) -> None:
        """
        Load metadata from HDF5 group.

        Parameters
        ----------
        metadata_group : h5py.Group
            Metadata group
        flight : Flight
            Flight object to populate
        """
        # Reconstruct flight_info from attrs
        flight_info = {}
        flight_metadata = {}

        for key, value in metadata_group.attrs.items():
            if key.startswith("flight_info_"):
                info_key = key.replace("flight_info_", "")
                flight_info[info_key] = _deserialize_from_hdf5(value)
            elif key.startswith("flight_metadata_"):
                meta_key = key.replace("flight_metadata_", "")
                flight_metadata[meta_key] = _deserialize_from_hdf5(value)

        if flight_info:
            flight.flight_info.update(flight_info)
        if flight_metadata:
            flight.metadata.update(flight_metadata)

    @staticmethod
    def _load_raw_data_from_hdf5(raw_group: "h5py.Group", flight: "Flight") -> None:
        """
        Load raw data hierarchy from HDF5 group.

        Parameters
        ----------
        raw_group : h5py.Group
            raw_data group
        flight : Flight
            Flight object to populate
        """

        # Load drone data
        if "drone_data" in raw_group:
            drone_group = raw_group["drone_data"]
            assert isinstance(drone_group, h5py.Group)
            drone_df = None
            litchi_df = None

            if "drone" in drone_group:
                drone_data = drone_group["drone"]
                assert isinstance(drone_data, h5py.Group)
                drone_df = Flight._load_dataframe_from_hdf5(drone_data)

            if "litchi" in drone_group:
                litchi_data = drone_group["litchi"]
                assert isinstance(litchi_data, h5py.Group)
                litchi_df = Flight._load_dataframe_from_hdf5(litchi_data)

            records = {}
            if "dat_records" in drone_group:
                for record_name, item in drone_group["dat_records"].items():
                    table = Flight._load_dataframe_from_hdf5(item)
                    if table is not None:
                        records[record_name] = table

            # Files written before the schema: rename to it (no-op otherwise)
            if drone_df is not None:
                drone_df = schema.canonical(drone_df, "dji")
            if litchi_df is not None:
                litchi_df = schema.canonical(litchi_df, "litchi")
            records = {
                schema.record_name(name): schema.canonical(table, record=name)
                for name, table in records.items()
            }
            if drone_df is not None or litchi_df is not None:
                flight.raw_data.drone_data = DroneData(drone_df, litchi_df, records)

        # Load payload data
        if "payload_data" in raw_group:
            payload_group = raw_group["payload_data"]
            assert isinstance(payload_group, h5py.Group)
            flight.raw_data.payload_data = PayloadData()

            for sensor_name in payload_group.keys():
                sensor_data = payload_group[sensor_name]
                assert isinstance(sensor_data, h5py.Group)
                sensor_df = Flight._load_dataframe_from_hdf5(sensor_data)
                if sensor_df is not None:
                    setattr(
                        flight.raw_data.payload_data,
                        sensor_name,
                        schema.canonical(sensor_df, sensor_name),
                    )
                    continue
                # A group of tables (e.g. the inclinometer message types)
                tables = {
                    name: Flight._load_dataframe_from_hdf5(item)
                    for name, item in sensor_data.items()
                    if isinstance(item, h5py.Group)
                }
                tables = {
                    schema.snake_case(k): schema.canonical(v, sensor_name)
                    for k, v in tables.items()
                    if v is not None
                }
                if tables:
                    setattr(flight.raw_data.payload_data, sensor_name, tables)

    @staticmethod
    def _load_dataframe_from_hdf5(
        dataset_group: "h5py.Group",
    ) -> Optional["pl.DataFrame"]:
        """
        Load a Polars DataFrame from HDF5 dataset group.

        Parameters
        ----------
        dataset_group : h5py.Group
            Group containing column datasets (see
            :func:`pils.utils.hdf5.write_table`)

        Returns
        -------
        pl.DataFrame or None
            Polars DataFrame if data found, None otherwise
        """
        if "columns" not in dataset_group.attrs:
            return None
        df = read_table(dataset_group)
        return df if df.width else None

    def set_metadata(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Set flight metadata.

        Parameters
        ----------
        metadata : Dict[str, Any], optional
            Dictionary containing metadata fields
            such as flight_time, duration, weather conditions, pilot info, etc.

        Examples
        --------
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
        # First, store all provided metadata fields
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        # Then extract time-related fields for special processing
        info_source: dict[str, Any] = {}
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

        # Optional flight name (only if not already set)
        if "flight_name" in info_source and "flight_name" not in self.metadata:
            self.metadata["flight_name"] = info_source.get("flight_name")

    def _detect_drone_model(self, drone_folder: str) -> str:
        """
        Auto-detect drone model from data folder structure.

        Parameters
        ----------
        drone_folder : str
            Path to the drone data folder

        Returns
        -------
        str
            Detected drone model ('dji' or 'blacksquare')

        Notes
        -----
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
                    if item:
                        specs_obj = (
                            item.get("specifications")
                            if isinstance(item, dict)
                            else None
                        )
                        specs = specs_obj if isinstance(specs_obj, dict) else {}
                        model_val = (
                            specs.get("model")
                            or item.get("name")
                            or item.get("category")
                        )
                        if isinstance(model_val, str):
                            m = model_val.lower()
                            if "matrice" in m:
                                return "dji"
                            if "black" in m or "blacksquare" in m:
                                return "blacksquare"
                            # If we can't map to a known driver, return the raw model string
                            return m
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
        drone_model: str | None = None,
    ):
        """
        Load drone telemetry data based on auto-detected drone model.

        Automatically detects whether the drone is DJI or BlackSquare and loads
        the appropriate data format. For DJI drones, also loads Litchi flight logs
        if available.

        Parameters
        ----------
        dji_dat_loader : bool, default=True
            If True, uses .DAT format for DJI drones.
            If False, uses .CSV format.
        drone_model : Optional[str], default=None
            Drone model to load. If None, will auto-detect.

        Returns
        -------
        DroneData
            Reference to the loaded drone data

        Raises
        ------
        ValueError
            If an unknown drone model is detected

        Examples
        --------
        >>> # Load DJI drone data using .DAT files (default)
        >>> flight.add_drone_data(dji_dat_loader=True)
        >>> # Load DJI drone data using .CSV files
        >>> flight.add_drone_data(dji_dat_loader=False)
        >>> # Access drone telemetry
        >>> print(flight.raw_data.drone_data.drone.head())
        >>> # Access Litchi waypoint data (if DJI)
        >>> if flight.raw_data.drone_data.litchi is not None:
        ...     print(flight.raw_data.drone_data.litchi.head())
        >>> # Alternative: use dictionary access
        >>> drone_data = flight['raw_data']['drone_data']['drone']
        """

        # Resolve drone folder
        if not isinstance(self.flight_info, dict):
            raise ValueError(
                "flight_info must be a dict containing 'drone_data_folder_path'"
            )

        drone_folder = self.flight_info.get("drone_data_folder_path")
        if not drone_folder:
            raise ValueError("drone_data_folder_path not found in flight_info")

        if not drone_model:
            drone_model = self._detect_drone_model(str(drone_folder))

        self.__drone_model = drone_model

        # Find candidate files
        available_files = glob.glob(str(drone_folder) + "/*")
        records: dict[str, pl.DataFrame] = {}  # other .dat records (DJI)

        if isinstance(self.__drone_model, str) and "dji" in self.__drone_model.lower():
            drone_data_path = None
            litchi_data_path = None

            for file in available_files:
                fname = file.lower()

                if dji_dat_loader and fname.endswith("drone.dat"):
                    drone_data_path = file

                else:
                    if fname.endswith("drone.csv") and drone_data_path is None:
                        drone_data_path = file

                if fname.endswith("litchi.csv") and "dji" in self.__drone_model.lower():
                    litchi_data_path = file

            logger.info(f"Drone : {drone_data_path}")
            drone = DJIDrone(drone_data_path)
            drone.load_data(use_dat=dji_dat_loader)
            drone_data = drone.data
            records = drone.records

            litchi_data = None

            if litchi_data_path is not None:
                litchi_loader = Litchi(litchi_data_path)
                litchi_loader.load_data()
                litchi_data = litchi_loader.data

        elif isinstance(self.__drone_model, str) and (
            "black" in self.__drone_model.lower()
            or "blacksquare" in self.__drone_model.lower()
        ):
            drone = BlackSquareDrone(drone_folder)
            drone.load_data()
            drone_data = drone.data
            litchi_data = None

        else:
            try:
                drone = DJIDrone(drone_data_path or str(drone_folder))
                drone.load_data(use_dat=dji_dat_loader)
                drone_data = drone.data
                litchi_loader = Litchi(litchi_data_path or str(drone_folder))
                litchi_loader.load_data()
                litchi_data = litchi_loader.data
            except Exception:
                drone = BlackSquareDrone(str(drone_folder))
                drone.load_data()
                drone_data = drone.data
                litchi_data = None

        drone_kind = (
            "blacksquare"
            if isinstance(self.__drone_model, str)
            and "black" in self.__drone_model.lower()
            else "dji"
        )
        self.raw_data.drone_data = DroneData(
            _standardize(drone_data, drone_kind),
            _standardize(litchi_data, "litchi"),
            _standardize_records(records),
        )

    def _read_sensor_data(self, sensor_name: str, sensor_folder: Path) -> Any | None:
        """
        Read sensor data based on sensor type.

        Parameters
        ----------
        sensor_name : str
            Name of the sensor ('gps', 'imu', 'adc', 'inclinometer')
        sensor_folder : Path
            Path to the sensors folder

        Returns
        -------
        Optional[Any]
            Sensor object or None if sensor not found

        Notes
        -----
        This is an internal method used by add_sensor_data().
        """
        result = None
        config = sensor_config.get(sensor_name.lower())

        if config:
            sensor = config["class"](sensor_folder)

            if sensor_name == "inclinometer":
                self.__inclinometer = sensor.sensor_type

            getattr(sensor, config["load_method"])()
            result = sensor.data

        return result

    def add_sensor_data(self, sensor_name: str | list[str]) -> None:
        """
        Load sensor data from the payload.

        Loads one or more sensors from the auxiliary data folder. Sensors are
        automatically detected and loaded based on their type.

        Parameters
        ----------
        sensor_name : Union[str, List[str]]
            Single sensor name or list of sensor names.
            Supported sensors: 'gps', 'imu', 'adc', 'inclinometer'

        Examples
        --------
        >>> # Load a single sensor
        >>> flight.add_sensor_data('gps')
        >>> print(flight.raw_data.payload_data.gps)
        >>> # Load multiple sensors at once
        >>> flight.add_sensor_data(['gps', 'imu', 'adc'])
        >>> # Access sensor data
        >>> gps_data = flight.raw_data.payload_data.gps
        >>> imu_data = flight.raw_data.payload_data.imu
        >>> # Or use dictionary-style
        >>> gps_data = flight['raw_data']['payload']['gps']
        >>> # Filter GPS data
        >>> high_accuracy = gps_data.filter(pl.col('accuracy') < 5.0)
        >>> # List all loaded sensors
        >>> print(flight.raw_data.payload_data.list_loaded_sensors())
        """
        sensor_path = Path(self.flight_info["aux_data_folder_path"]) / "sensors"

        if sensor_path.exists():
            if isinstance(sensor_name, str):
                sensor_name = [sensor_name]

            for sensor in sensor_name:
                sensor_data = self._read_sensor_data(sensor, sensor_path)
                # {"data", "metadata"} (e.g. GPS); the inclinometer and the IMU
                # are dicts of tables
                if isinstance(sensor_data, dict) and {"data", "metadata"} <= set(
                    sensor_data
                ):
                    setattr(
                        self.raw_data.payload_data,
                        sensor,
                        _standardize(sensor_data["data"], sensor),
                    )
                    self.flight_info["flight_info"].update(
                        {f"{sensor}_metadata": sensor_data["metadata"]}
                    )

                else:
                    setattr(
                        self.raw_data.payload_data,
                        sensor,
                        _standardize(sensor_data, sensor),
                    )
        else:
            logger.info("Sensor datasets are not available")

    def add_camera_data(
        self, use_photogrammetry: bool = False, get_sony_angles: bool = True
    ) -> None:
        """
        Load camera data from the payload.

        Supports both video cameras (Sony RX0 MarkII with telemetry, Alvium industrial)
        and photogrammetry-processed data. For video cameras, can compute Euler angles
        (roll, pitch, yaw) and quaternions from inertial measurement data.

        Parameters
        ----------
        use_photogrammetry : bool, default=False
            If True, loads pre-processed photogrammetry results from proc_data folder.
            If False, loads camera data from aux_data/camera folder (video or logs).
        get_sony_angles : bool, default=True
            For Sony cameras, whether to compute Euler angles and quaternions from
            telemetry gyro/accel data using AHRS (Madgwick) filter.

        Raises
        ------
        FileNotFoundError
            If camera data folder or photogrammetry folder not found

        Examples
        --------
        >>> # Load Sony RX0 MarkII video data with angles computed
        >>> flight.add_camera_data(use_photogrammetry=False, get_sony_angles=True)
        >>> # Load pre-processed photogrammetry results
        >>> flight.add_camera_data(use_photogrammetry=True)
        >>> # Load camera video data without computing angles
        >>> flight.add_camera_data(use_photogrammetry=False, get_sony_angles=False)
        """

        self.__use_photogrammetry = use_photogrammetry
        if use_photogrammetry:
            self.__camera_data_type = "photogrammetry"
            path = Path(self.flight_info["proc_data_folder_path"]) / "photogrammetry"
        else:
            self.__camera_data_type = "camera"
            path = Path(self.flight_info["aux_data_folder_path"]) / "camera"

        if path.exists():
            camera = Camera(path, use_photogrammetry=use_photogrammetry)

            camera.load_data()

            self.raw_data.payload_data.camera = _standardize(camera.data[0], "camera")

            self.__camera_model = camera.data[1]

        else:
            logger.info("Camera Path does not exist")

    def sync(
        self,
        target_rate: dict[str, float] | None = None,
        use_rtk_data: bool = True,
        common_time: bool = True,
        drone_records: Sequence[str] | Literal["all"] = ("imu_atti_0",),
        **kwargs,
    ) -> dict[str, pl.DataFrame]:
        """
        Synchronize flight data using GPS-based correlation.

        Creates a Synchronizer instance, adds available data sources,
        performs synchronization, and stores the result in sync_data attribute.

        Parameters
        ----------
        target_rate : dict, default=None
            Target sample rate in Hz of the different sensors; if None the following
            rates are applied:
            - 10 Hz for drone and litchi
            - 100 Hz for payload sensors (including inclinometer and ADC)
        use_rtk_data : bool, default=True
            For DJI drones: if True, use RTK data; if False, use standard GPS
        common_time: bool
            Interpolate all the data at a common time, with a sampliing frequency
            determined by the target_rate. If False, the time is just shifted and the
            other columns are not touched
        drone_records : sequence of str or "all", default=("imu_atti_0",)
            Other drone records (DJI .dat, see ``raw_data.drone_data.records``)
            to synchronize, each as ``drone_<name>``; they take the drone GPS
            offset. The default is the flight controller attitude (200 Hz).
            DJI names such as ``"IMU_ATTI(0)"`` are accepted too.
        **kwargs : dict
            Additional arguments passed to Synchronizer.synchronize()

        Returns
        -------
        pl.DataFrame
            Synchronized data

        Raises
        ------
        ValueError
            If no GPS payload data available (required as reference)

        Examples
        --------
        >>> # Basic synchronization with RTK data
        >>> flight.add_sensor_data(['gps', 'imu', 'adc'])
        >>> flight.add_drone_data()
        >>> sync_df = flight.sync(target_rate={'drone': 10.0, 'payload': 100.0}, use_rtk_data=True)
        >>> # Use standard GPS instead of RTK
        >>> sync_df = flight.sync(target_rate={'drone': 10.0}, use_rtk_data=False)
        >>> # Synchronization is stored in flight.sync_data as a dict of DataFrames
        >>> print(list(flight.sync_data.keys()))
        """

        # Check if GPS payload data is available
        if not self.raw_data.payload_data or "gps" not in self.raw_data.payload_data:
            raise ValueError(
                "GPS payload data is required as reference timebase. "
                "Call flight.add_sensor_data(['gps']) first."
            )

        # Create synchronizer
        sync = Synchronizer()

        # Add GPS payload as reference (mandatory)
        gps_sensor = self.raw_data.payload_data["gps"]
        gps_data = gps_sensor.data if hasattr(gps_sensor, "data") else gps_sensor
        sync.add_gps_reference(
            gps_data,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )

        # Add drone GPS if available
        drone_data = self.raw_data.drone_data.drone
        drone_has_data = (isinstance(drone_data, dict) and len(drone_data) > 0) or (
            isinstance(drone_data, pl.DataFrame) and len(drone_data) > 0
        )

        # A copy: the defaults below must not change the caller's dict
        target_rate = dict(target_rate or {})

        if drone_has_data:
            # Ensure target_rate dict has drone key with default 10 Hz
            if "drone" not in target_rate:
                target_rate["drone"] = 10.0
            drone_df = drone_data
            if isinstance(drone_df, dict):  # ArduPilot messages
                drone_df = drone_df.get("gps", next(iter(drone_df.values())))

            model = (self.__drone_model or "").lower()
            records_gps = (self.raw_data.drone_data.records or {}).get("gps")
            if "dji" in model and use_rtk_data and "rtk_lat_deg" in drone_df.columns:
                lat_col, lon_col, alt_col = "rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m"
            else:
                # The main receiver: in the drone table (DatCon CSV,
                # BlackSquare) or, for DJI .dat files, its own record
                if "gps_lat_deg" in drone_df.columns:
                    prefix = "gps_"
                elif "dji" in model and records_gps is not None:
                    drone_df, prefix = records_gps, ""
                else:
                    prefix = ""
                lat_col, lon_col = f"{prefix}lat_deg", f"{prefix}lon_deg"
                alt_col = f"{prefix}h_ell_m"
                # Heights above mean sea level: to the ellipsoid with the geoid
                # separation of the payload receiver
                separation = (gps_data["h_ell_m"] - gps_data["h_msl_m"]).median()
                drone_df = drone_df.with_columns(
                    (pl.col(f"{prefix}h_msl_m") + separation).alias(alt_col)
                )

            sync.add_drone_gps(
                drone_df,
                timestamp_col="timestamp",
                lat_col=lat_col,
                lon_col=lon_col,
                alt_col=alt_col,
            )

        # Add litchi GPS if available
        litchi_df = self.raw_data.drone_data.litchi
        if (
            isinstance(litchi_df, pl.DataFrame)
            and litchi_df.height > 0
            and {"lat_deg", "lon_deg"} <= set(litchi_df.columns)
        ):
            if "drone" not in target_rate:
                target_rate["drone"] = 10.0
            sync.add_litchi_gps(litchi_df)

        # Add other drone records (DJI .dat attitude, motors, ...)
        records = getattr(self.raw_data.drone_data, "records", None) or {}
        if drone_has_data and records:
            if drone_records != "all":
                wanted = {schema.record_name(name) for name in drone_records}
                missing = sorted(wanted - set(records))
                if missing:
                    logger.warning(f"Drone records not found: {missing}")
                records = {k: v for k, v in records.items() if k in wanted}
            sync.add_drone_records(records)

        # Add inclinometer if available
        if "inclinometer" in self.raw_data.payload_data:
            if "inclinometer" not in target_rate:
                target_rate["inclinometer"] = 100.0

            incl_sensor = self.raw_data.payload_data["inclinometer"]
            incl_data = (
                incl_sensor.data if hasattr(incl_sensor, "data") else incl_sensor
            )
            if isinstance(incl_data, dict):  # IMX-5: the INS table
                incl_data = incl_data["ins"]
            sync.add_inclinometer(incl_data, self.__inclinometer)

        # Add Camera data if available

        if "camera" in self.raw_data.payload_data:
            sync.add_camera(
                self.raw_data.payload_data["camera"],
                use_photogrammetry=self.__use_photogrammetry,
                camera_model=self.__camera_model,
            )

        # Add other payload sensors
        payload = self.raw_data.payload_data

        # Add ADC if available
        if "adc" in payload:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            adc_sensor = payload["adc"]
            adc_data = adc_sensor.data if hasattr(adc_sensor, "data") else adc_sensor
            sync.add_payload_sensor("adc", adc_data)

        # Add LM76 if available
        if "lm76" in payload:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            lm76_sensor = payload["lm76"]
            lm76_data = (
                lm76_sensor.data if hasattr(lm76_sensor, "data") else lm76_sensor
            )
            sync.add_payload_sensor("lm76", lm76_data)

        # Add IMU sensors if available ({"barometer": df, ...})
        if "imu" in payload and payload["imu"] is not None:
            if "payload" not in target_rate:
                target_rate["payload"] = 100.0
            imu_data = payload["imu"]
            if not isinstance(imu_data, dict):  # an IMU object
                imu_data = imu_data.data or {}
            for name, table in imu_data.items():
                if isinstance(table, pl.DataFrame) and not table.is_empty():
                    sync.add_payload_sensor(f"imu_{name}", table)

        # Perform synchronization
        self.sync_data = sync.synchronize(
            target_rate=target_rate,
            common_time=common_time,
            **kwargs,
        )
        # Detected time offsets in seconds, as in Synchronizer.offsets: the
        # inclinometer and camera offsets are relative to Litchi
        self.sync_offsets = {
            name: float(offset["time_offset"])
            for name, offset in sync.offsets.items()
            if "time_offset" in offset
        }
        # Rates (Hz) of the synchronized tables, as used
        self.sync_rates = dict(sync.rates)

        return self.sync_data

    # -- storage (pils.store) ------------------------------------------------

    def _raw_tables(self) -> dict[str, Any]:
        drone = self.raw_data.drone_data
        tables: dict[str, Any] = {"drone": {}, "payload": {}}
        if drone is not None:
            if isinstance(drone.drone, pl.DataFrame | dict) and len(drone.drone):
                tables["drone"]["drone"] = drone.drone
            if isinstance(drone.litchi, pl.DataFrame) and drone.litchi.height:
                tables["drone"]["litchi"] = drone.litchi
            if drone.records:
                tables["drone"]["records"] = drone.records
        for name in self.raw_data.payload_data.list_loaded_sensors():
            data = self.raw_data.payload_data[name]
            data = data.data if hasattr(data, "data") else data
            if isinstance(data, pl.DataFrame | dict):
                tables["payload"][name] = data
        return {k: v for k, v in tables.items() if v}

    def save(
        self,
        directory: str | Path,
        sync_metadata: dict[str, Any] | None = None,
        revision: str | None = None,
    ) -> str | None:
        """Save the flight: raw data and a new synchronized revision.

        Layout (see :mod:`pils.store`)::

            <directory>/
            ├── raw/                      # replaced at each save
            │   ├── manifest.json         # flight info, loaders, code version
            │   ├── drone/drone.parquet, drone/litchi.parquet,
            │   │   drone/records/<record>.parquet
            │   └── payload/<sensor>.parquet, payload/<sensor>/<table>.parquet
            └── sync/
                └── rev_YYYYMMDD_hhmmss/  # one per save with sync_data
                    ├── manifest.json     # offsets, rates, raw data used
                    └── <table>.parquet

        Parameters
        ----------
        directory : str or Path
            Folder of the flight's products, e.g. ``<flight>/proc/flight``.
        sync_metadata : dict, optional
            Stored in the synchronized revision's manifest (e.g. the run).
        revision : str, optional
            Name of the synchronized revision (default: the current time).

        Returns
        -------
        str or None
            Name of the synchronized revision, or None without sync_data.
        """
        from pils import store

        directory = Path(directory)
        raw = store.write_tables(
            directory / "raw",
            self._raw_tables(),
            metadata={
                "flight_info": self.flight_info,
                "flight_metadata": self.metadata,
                "drone_model": self.__drone_model,
                "inclinometer": self.__inclinometer,
                "camera_model": self.__camera_model,
                "use_photogrammetry": self.__use_photogrammetry,
            },
            kind="flight_raw",
            replace=True,
        )
        if not self.sync_data:
            return None
        if revision is None:
            revision = store.new_version_name()
            while (directory / "sync" / revision).exists():
                import time

                time.sleep(1)
                revision = store.new_version_name()
        store.save_version(
            directory / "sync",
            revision,
            dict(self.sync_data),
            metadata={
                "raw_saved_at": store.read_manifest(raw)["saved_at"],
                "time_offsets": self.sync_offsets,
                "rates": self.sync_rates,
                **(sync_metadata or {}),
            },
            kind="flight_sync",
        )
        return revision

    @classmethod
    def load(
        cls, directory: str | Path, sync_revision: str | None | bool = None
    ) -> "Flight":
        """Load a flight saved with :meth:`save`.

        Parameters
        ----------
        directory : str or Path
            Folder given to :meth:`save`.
        sync_revision : str, None or False, default=None
            Synchronized revision to load: None for the latest, False for
            none.

        Returns
        -------
        Flight
            The flight, ready to be synchronized again.
        """
        from pils import store

        directory = Path(directory)
        tables, manifest = store.read_tables(directory / "raw")
        meta = manifest["metadata"]
        flight = cls(meta["flight_info"])
        flight.metadata.update(meta.get("flight_metadata") or {})
        flight.__drone_model = meta.get("drone_model")
        flight.__inclinometer = meta.get("inclinometer")
        flight.__camera_model = meta.get("camera_model")
        flight.__use_photogrammetry = bool(meta.get("use_photogrammetry"))

        drone: dict[str, Any] = tables.get("drone", {})  # type: ignore[assignment]
        if drone:
            flight.raw_data.drone_data = DroneData(
                drone.get("drone", pl.DataFrame()),
                drone.get("litchi"),
                drone.get("records"),
            )
        payload: dict[str, Any] = tables.get("payload", {})  # type: ignore[assignment]
        for name, data in payload.items():
            setattr(flight.raw_data.payload_data, name, data)

        if sync_revision is not False:
            revisions = store.list_versions(directory / "sync")
            if sync_revision is None and revisions:
                sync_revision = revisions[-1]
            if sync_revision:
                if sync_revision not in revisions:
                    raise ValueError(
                        f"Sync revision '{sync_revision}' not found: {revisions}"
                    )
                sync_tables, sync_meta = store.load_version(
                    directory / "sync", sync_revision
                )
                flight.sync_data = sync_tables  # type: ignore[assignment]
                flight.sync_offsets = sync_meta.get("time_offsets", {})
                flight.sync_rates = sync_meta.get("rates", {})
        return flight

    def to_hdf5(
        self,
        filepath: str | Path | None = None,
        sync_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save flight data to HDF5 file.

        .. deprecated::
            Use :meth:`save` (Parquet tables, see :mod:`pils.store`);
            :meth:`from_hdf5` still reads HDF5 files.

        Saves metadata and raw_data hierarchy.

        Parameters
        ----------
        filepath : Union[str, Path], optional
            Path to output HDF5 file
        sync_metadata : Dict[str, Any], optional
            Additional metadata to store with synchronized data revision.
            Will be saved as attributes on the revision group.
            Example: {'comment': 'Initial sync', 'target_rate': 10.0}

        Returns
        -------
        str
            Timestamp string for the save operation

        Raises
        ------
        ImportError
            If h5py is not installed
        ValueError
            If no data to save

        Examples
        --------
        >>> # Save raw data
        >>> flight.to_hdf5('flight_001.h5')
        >>> # Save with sync metadata
        >>> flight.to_hdf5('flight_001.h5', sync_metadata={'comment': 'High rate sync', 'rate': 100.0})
        >>> # For synchronization, use Synchronizer separately:
        >>> from pils.synchronizer import Synchronizer
        >>> sync = Synchronizer()
        >>> sync.add_gps_reference(flight.raw_data.payload_data.gps)
        >>> # ... add other sources ...
        >>> result = sync.synchronize(target_rate={'drone': 10.0, 'payload': 100.0})
        """
        warnings.warn(
            "Flight.to_hdf5 is deprecated: use Flight.save (Parquet, pils.store)",
            DeprecationWarning,
            stacklevel=2,
        )

        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        else:
            filepath = Path(self.flight_info["proc_data_folder_path"])

        with h5py.File(str(filepath), "a") as f:
            # Save metadata
            self._save_metadata_to_hdf5(f)

            # Save raw data
            self._save_raw_data_to_hdf5(f)

            # Save sync_data if available
            if self.sync_data is not None and len(self.sync_data) > 0:
                self._save_sync_data_to_hdf5(f, sync_metadata)

        return _get_current_timestamp()

    def _save_metadata_to_hdf5(self, h5file: "h5py.File") -> None:
        """
        Save metadata to HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle
        """
        if "metadata" not in h5file:
            metadata_group = h5file.create_group("metadata")
        else:
            metadata_group = h5file["metadata"]

        # Save flight_info as attrs
        if self.flight_info:
            for key, value in self.flight_info.items():
                try:
                    metadata_group.attrs[f"flight_info_{key}"] = _serialize_for_hdf5(
                        value
                    )
                except Exception as e:
                    logger.info(f"Warning: Could not save flight_info[{key}]: {e}")

        # Save flight_metadata as attrs
        if self.metadata:
            for key, value in self.metadata.items():
                try:
                    metadata_group.attrs[f"flight_metadata_{key}"] = (
                        _serialize_for_hdf5(value)
                    )
                except Exception as e:
                    logger.info(f"Warning: Could not save metadata[{key}]: {e}")

    def _save_raw_data_to_hdf5(self, h5file: "h5py.File") -> None:
        """
        Save raw data hierarchy to HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle
        """

        if "raw_data" not in h5file:
            raw_data_group = h5file.create_group("raw_data")
        else:
            raw_data_group = h5file["raw_data"]
            assert isinstance(raw_data_group, h5py.Group)
        # Raw data is replaced at each save: the sync revisions record which
        # raw data (saved_at) they were made from
        raw_data_group.attrs["saved_at"] = _get_current_timestamp()
        raw_data_group.attrs["pils_version"] = _get_package_version()

        # Save drone data
        drone_has_data = (
            (
                isinstance(self.raw_data.drone_data.drone, dict)
                and len(self.raw_data.drone_data.drone) > 0
            )
            or (
                isinstance(self.raw_data.drone_data.drone, pl.DataFrame)
                and len(self.raw_data.drone_data.drone) > 0
            )
            or len(self.raw_data.drone_data.litchi) > 0
        )
        if drone_has_data:
            if "drone_data" not in raw_data_group:
                drone_group = raw_data_group.create_group("drone_data")
            else:
                drone_group = raw_data_group["drone_data"]
                assert isinstance(drone_group, h5py.Group)

            drone_data = self.raw_data.drone_data.drone
            if isinstance(drone_data, dict) and len(drone_data) > 0:
                pass  # Handle dict case if needed
            elif isinstance(drone_data, pl.DataFrame) and len(drone_data) > 0:
                self._save_dataframe_to_hdf5(drone_group, "drone", drone_data)

            if len(self.raw_data.drone_data.litchi) > 0:
                self._save_dataframe_to_hdf5(
                    drone_group, "litchi", self.raw_data.drone_data.litchi
                )

            records = getattr(self.raw_data.drone_data, "records", None) or {}
            if records:
                if "dat_records" in drone_group:
                    del drone_group["dat_records"]
                records_group = drone_group.create_group("dat_records")
                for record_name, table in records.items():
                    self._save_dataframe_to_hdf5(records_group, record_name, table)

        # Save payload data
        if len(self.raw_data.payload_data.list_loaded_sensors()) > 0:
            if "payload_data" not in raw_data_group:
                payload_group = raw_data_group.create_group("payload_data")
            else:
                payload_group = raw_data_group["payload_data"]
                assert isinstance(payload_group, h5py.Group)

            for sensor_name in self.raw_data.payload_data.list_loaded_sensors():
                sensor_data = getattr(self.raw_data.payload_data, sensor_name)
                if isinstance(sensor_data, dict):
                    # e.g. the inclinometer: one table per message type
                    if sensor_name in payload_group:
                        del payload_group[sensor_name]
                    sensor_group = payload_group.create_group(sensor_name)
                    for table_name, table in sensor_data.items():
                        if isinstance(table, pl.DataFrame):
                            self._save_dataframe_to_hdf5(
                                sensor_group, table_name, table
                            )
                elif sensor_data is not None:
                    self._save_dataframe_to_hdf5(
                        payload_group, sensor_name, sensor_data
                    )

    def _save_sync_data_to_hdf5(
        self, h5file: "h5py.File", sync_metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Save synchronized data to HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle
        sync_metadata : Dict[str, Any], optional
            Additional metadata to store as attributes on revision group
        """
        if self.sync_data is None or len(self.sync_data) == 0:
            return

        if "sync_data" not in h5file:
            sync_group = h5file.create_group("sync_data")
        else:
            sync_group = h5file["sync_data"]
            assert isinstance(sync_group, h5py.Group)

        # Create revision group
        revision_name = _get_current_timestamp()
        if revision_name in sync_group:
            del sync_group[revision_name]
        revision_group = sync_group.create_group(revision_name)

        # Save each key's DataFrame as a dataset
        if self.sync_data is not None:
            for key, df in self.sync_data.items():
                if isinstance(df, pl.DataFrame) and len(df) > 0:
                    self._save_dataframe_to_hdf5(revision_group, key, df)

            # Save metadata on revision group
            revision_group.attrs["created_at"] = revision_name
            revision_group.attrs["n_keys"] = len(self.sync_data)
        revision_group.attrs["pils_version"] = _get_package_version()
        if "raw_data" in h5file:
            revision_group.attrs["raw_data_saved_at"] = h5file["raw_data"].attrs.get(
                "saved_at", ""
            )

        # Save user-provided metadata
        if sync_metadata:
            for key, value in sync_metadata.items():
                try:
                    revision_group.attrs[f"user_{key}"] = _serialize_for_hdf5(value)
                except Exception as e:
                    logger.info(f"Warning: Could not save sync_metadata[{key}]: {e}")

    def _save_dataframe_to_hdf5(
        self, parent_group: "h5py.Group", name: str, df: "pl.DataFrame"
    ) -> None:
        """
        Save a Polars DataFrame to HDF5 (see :func:`pils.utils.hdf5.write_table`).

        Parameters
        ----------
        parent_group : h5py.Group
            Parent HDF5 group
        name : str
            Dataset name
        df : pl.DataFrame
            DataFrame to save
        """
        write_table(parent_group, name, df)

    def __getitem__(self, key):
        """
        Dictionary-style access to flight data.

        Parameters
        ----------
        key : str
            Key to access ('raw_data' or 'metadata')

        Returns
        -------
        object
            Corresponding data object

        Raises
        ------
        KeyError
            If key is not found

        Examples
        --------
        >>> # Access raw data
        >>> raw_data = flight['raw_data']
        >>> # Access metadata
        >>> metadata = flight['metadata']
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
