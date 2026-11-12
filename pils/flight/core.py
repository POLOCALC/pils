"""Flight: load, synchronize, save and load the data of one flight."""

import glob
import logging
from pathlib import Path
from typing import Any

import polars as pl

from pils import schema
from pils.drones.BlackSquareDrone import BlackSquareDrone
from pils.drones.DJIDrone import DJIDrone
from pils.drones.litchi import Litchi
from pils.flight.containers import DroneData, RawData
from pils.flight.hdf5 import FlightHDF5Mixin
from pils.flight.sync import FlightSyncMixin
from pils.sensors.camera import Camera
from pils.sensors.sensors import sensor_config
from pils.utils.tools import get_path_from_keyword, standardize_datetimes

logger = logging.getLogger(__name__)


def _standardize(data: Any, kind: str | None = None) -> Any:
    """Rename a loaded table, or dict of tables, to the schema (:mod:`pils.schema`).

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


def _standardize_records(records: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """DJI .dat record tables in the schema, e.g. ``imu_atti_0``."""
    return {
        schema.record_name(name): standardize_datetimes(
            schema.canonical(table, record=name)
        )
        for name, table in records.items()
    }


def _attribute_alias(name: str) -> property:
    """Return a property reading and writing the attribute ``name``."""
    return property(
        lambda self: getattr(self, name),
        lambda self, value: setattr(self, name, value),
    )


class Flight(FlightSyncMixin, FlightHDF5Mixin):
    """
    Data of one flight: drone, payload sensors and synchronized tables.

    A hierarchical structure to store and access drone flight data and sensor
    payloads. Data is stored in RAM for fast access using both attribute and
    dictionary-style notation.

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
    >>> flight.add_drone_data(dji_dat_loader=True)
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
    >>> # Perform operations on the data (DJI RTK heights, metres)
    >>> high_altitude = drone_df.filter(pl.col('rtk_h_ell_m') > 100)
    >>> print(f"Points above 100m: {len(high_altitude)}")
    """

    # The loader state was name-mangled (``self.__drone_model``); the old
    # names stay readable and writable for the code and tests that use them
    _Flight__drone_model = _attribute_alias("_drone_model")
    _Flight__inclinometer = _attribute_alias("_inclinometer")
    _Flight__camera_model = _attribute_alias("_camera_model")
    _Flight__use_photogrammetry = _attribute_alias("_use_photogrammetry")

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
        # Each offset with how it was found (method, correlation, ...)
        self.sync_offset_details: dict[str, dict[str, Any]] = {}
        # The reference timebase: "payload_gps" or "drone_gps"
        self.sync_reference: str | None = None
        self.sync_rates: dict[str, float] = {}
        # The drone table's columns the synchronized positions (lat_deg,
        # lon_deg, h_ell_m) come from, e.g. {"lat": "rtk_lat_deg", ...}:
        # RTK positions or the drone's own GNSS receiver
        self.sync_position_columns: dict[str, dict[str, str]] = {}
        self.adc_gain_config = None
        # Set by the loaders; saved with the raw data
        self._drone_model: str | None = None
        self._inclinometer: str | None = None
        self._camera_model: str | None = None
        self._use_photogrammetry = False

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
            except TypeError:  # not subtractable (e.g. strings): keep both
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
                    import pils.integrations.stout as stout_db

                    m = stout_db.drone_model(drone_id)
                    if m is not None:
                        if "matrice" in m:
                            return "dji"
                        if "black" in m or "blacksquare" in m:
                            return "blacksquare"
                        # If we can't map to a known driver, return the raw model string
                        return m
                except Exception as e:
                    # Broad: an optional lookup in the STOUT database (driver,
                    # connection, missing drone); the folder heuristics below
                    # are the fallback
                    logger.warning(
                        f"Drone model of {drone_id} not found in STOUT ({e}); "
                        "guessing it from the file names"
                    )

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
        except OSError as e:
            logger.warning(f"Could not read {drone_folder} ({e}); assuming a DJI drone")
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

        self._drone_model = drone_model

        # Find candidate files
        available_files = glob.glob(str(drone_folder) + "/*")
        if not available_files:
            raise FileNotFoundError(f"No drone log in {drone_folder} (empty folder)")
        records: dict[str, pl.DataFrame] = {}  # other .dat records (DJI)

        # DJI logs by name: the flight log (*drone.dat, or *drone.csv from
        # DatCon) and the Litchi mission log (*litchi.csv)
        drone_data_path = None
        litchi_data_path = None
        for file in available_files:
            fname = file.lower()
            if dji_dat_loader and fname.endswith("drone.dat"):
                drone_data_path = file
            elif fname.endswith("drone.csv") and drone_data_path is None:
                drone_data_path = file
            if fname.endswith("litchi.csv"):
                litchi_data_path = file

        model = self._drone_model.lower() if isinstance(self._drone_model, str) else ""
        is_dji = "dji" in model
        if not is_dji and "black" not in model and drone_data_path is not None:
            # Unknown model, but a DJI flight log in the folder
            logger.info(
                f"Drone model {self._drone_model!r}: DJI log found, read as DJI"
            )
            is_dji = True

        if is_dji:
            if drone_data_path is None:
                raise FileNotFoundError(
                    f"No DJI drone log (*drone.dat, *drone.csv) in {drone_folder}"
                )
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

        else:
            # BlackSquare, or an unknown model without a DJI log
            if "black" not in model:
                logger.info(
                    f"Drone model {self._drone_model!r}: no DJI log, read as BlackSquare"
                )
            drone = BlackSquareDrone(str(drone_folder))
            drone.load_data()
            drone_data = drone.data
            litchi_data = None

        drone_kind = "dji" if is_dji else "blacksquare"
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
                self._inclinometer = sensor.sensor_type

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
                    self.flight_info.setdefault("flight_info", {}).update(
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
        self._use_photogrammetry = use_photogrammetry
        if use_photogrammetry:
            self._camera_data_type = "photogrammetry"
            path = Path(self.flight_info["proc_data_folder_path"]) / "photogrammetry"
        else:
            self._camera_data_type = "camera"
            path = Path(self.flight_info["aux_data_folder_path"]) / "camera"

        if path.exists():
            camera = Camera(path, use_photogrammetry=use_photogrammetry)

            camera.load_data()

            self.raw_data.payload_data.camera = _standardize(camera.data[0], "camera")

            self._camera_model = camera.data[1]

        else:
            logger.info("Camera Path does not exist")

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

    @staticmethod
    def _carry_over_raw(raw_dir: Path, tables: dict[str, Any]) -> dict[str, Any]:
        """Add to ``tables`` the saved raw tables of sources not in them.

        A source is ``drone/<table>`` (``drone``, ``litchi``, ``records``) or
        ``payload/<sensor>``. A save with fewer sources (a re-package without
        some sensor) keeps the ones already saved rather than dropping them.

        Returns
        -------
        dict
            For the raw manifest: the sources carried over, and when they
            were saved; empty if none.
        """
        from pils import store

        if not (raw_dir / store.MANIFEST).is_file():
            return {}
        manifest = store.read_manifest(raw_dir)

        def source(key: str) -> str:
            return "/".join(key.split("/")[:2])

        new = {source(key) for key in store._flatten(tables)}
        carried = sorted(
            key for key in manifest.get("tables", {}) if source(key) not in new
        )
        if not carried:
            return {}
        for key in carried:
            *parents, name = key.split("/")
            node = tables
            for parent in parents:
                node = node.setdefault(parent, {})
            node[name] = store.read_parquet(raw_dir / f"{key}.parquet")
        sources = sorted({source(key) for key in carried})
        previous = manifest.get("metadata", {}).get("carried_over", {})
        logger.warning(f"Raw data not reloaded, kept from the previous save: {sources}")
        return {
            "carried_over": {
                s: previous.get(s, manifest.get("saved_at")) for s in sources
            }
        }

    def save(
        self,
        directory: str | Path,
        sync_metadata: dict[str, Any] | None = None,
        revision: str | None = None,
    ) -> str | None:
        """Save the flight: raw data and a new synchronized revision.

        Layout (see :mod:`pils.store`)::

            <directory>/
            ├── raw/                      # rewritten at each save (sources
            │   │                         # not reloaded are kept)
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
        tables = self._raw_tables()
        carried = self._carry_over_raw(directory / "raw", tables)
        raw = store.write_tables(
            directory / "raw",
            tables,
            metadata={
                "flight_info": self.flight_info,
                "flight_metadata": self.metadata,
                "drone_model": self._drone_model,
                "inclinometer": self._inclinometer,
                "camera_model": self._camera_model,
                "use_photogrammetry": self._use_photogrammetry,
                **carried,
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
                "time_offset_details": self.sync_offset_details,
                "reference": self.sync_reference,
                "rates": self.sync_rates,
                "position_columns": self.sync_position_columns,
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
        flight._drone_model = meta.get("drone_model")
        flight._inclinometer = meta.get("inclinometer")
        flight._camera_model = meta.get("camera_model")
        flight._use_photogrammetry = bool(meta.get("use_photogrammetry"))

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
                flight.sync_offset_details = sync_meta.get("time_offset_details", {})
                flight.sync_reference = sync_meta.get("reference")
                flight.sync_rates = sync_meta.get("rates", {})
                flight.sync_position_columns = sync_meta.get("position_columns", {})
        return flight

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
