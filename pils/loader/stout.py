"""
Stout Data Loader - Load flight data paths from the STOUT database.

This module provides a convenient interface to query and load flight data
from the STOUT campaign management system. It supports loading data at
multiple levels: all campaigns, single flights, and filtered flights by date.

Usage:
    from polocalc_data_loader import StoutDataLoader

    loader = StoutDataLoader()

    # Load all flights from all campaigns
    all_flights = loader.load_all_campaign_flights()

    # Load single flight by ID
    flight_data = loader.load_single_flight(flight_id='some-id')

    # Load flights by date range
    flights = loader.load_flights_by_date(start_date='2025-01-01', end_date='2025-01-15')
"""

import importlib
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pils.config import DRONE_MAP, SENSOR_MAP

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StoutLoader:
    """
    Data loader for STOUT campaign management system.

    Provides methods to load flight data paths and associated metadata
    from the STOUT database and file system.

    Attributes
    ----------
    campaign_service : Optional[CampaignService]
        Service for accessing campaign and flight data
    base_data_path : Optional[Path]
        Base path where all campaign data is stored
    """

    def __init__(self):
        """
        Initialize the StoutDataLoader.

        Initializes the loader and attempts to connect to stout campaign service.
        Falls back to filesystem queries if stout import fails.
        """

        self.campaign_service = None

        try:
            from stout.config import Config  # type: ignore
            from stout.services.campaigns import CampaignService  # type: ignore

            self.campaign_service = CampaignService()
            self.base_data_path = Config.MAIN_DATA_PATH
            logger.info(
                f"Initialized with stout database, base path: {self.base_data_path}"
            )
        except ImportError as e:
            logger.warning(
                f"Could not import stout: {e}. Falling back to filesystem queries."
            )
            self.use_stout = False

    def load_all_flights(self) -> list[dict[str, Any]]:
        """
        Load all flights from all campaigns.

        Returns
        -------
        list[dict[str, Any]]
            list of flight dictionaries containing flight metadata and paths.
            Each flight dict includes: flight_id, flight_name, campaign_id,
            takeoff_datetime, landing_datetime, and folder paths.
        """
        logger.info("Loading all flights from all campaigns...")

        if self.campaign_service is None:
            raise RuntimeError("Campaign service not initialized")
        try:
            flights = self.campaign_service.get_all_flights()
            logger.info(f"Loaded {len(flights)} flights from database")
            return flights
        except Exception as e:
            logger.error(f"Error loading flights from database: {e}")
            raise

    def load_all_campaign_flights(
        self, campaign_id: str | None = None, campaign_name: str | None = None
    ) -> list[dict[str, Any]] | None:
        """
        Load all flights from a specific campaign.

        Parameters
        ----------
        campaign_id : Optional[str]
            Campaign ID to load
        campaign_name : Optional[str]
            Campaign name to load (alternative to campaign_id)

        Returns
        -------
        Optional[dict[str, Any]]
            Campaign dictionary with metadata and paths, or None if not found.
        """
        if not campaign_id and not campaign_name:
            raise ValueError("Either flight_id or flight_name must be provided")

        logger.info(
            f"Loading single flight: flight_id={campaign_id}, flight_name={campaign_name}"
        )

        if self.campaign_service is None:
            raise RuntimeError("Campaign service not initialized")
        try:
            # Get the campaign_id first if only campaign_name provided
            cid = campaign_id or campaign_name

            if not cid:
                raise ValueError("Either campaign_id or campaign_name must be provided")

            flights = self.campaign_service.get_flights_by_campaign(campaign_id=cid)

            for flight in flights:
                if flight:
                    logger.info(f"Loaded flight: {flight.get('flight_name')}")
            return flights
        except Exception as e:
            logger.error(f"Error loading flight from database: {e}")
            raise

    def load_single_flight(
        self, flight_id: str | None = None, flight_name: str | None = None
    ) -> dict[str, Any] | None:
        """
        Load data for a single flight.

        Parameters
        ----------
        flight_id : Optional[str]
            Flight ID to load
        flight_name : Optional[str]
            Flight name to load (alternative to flight_id)

        Returns
        -------
        Optional[dict[str, Any]]
            Flight dictionary with metadata and paths, or None if not found.
        """
        if not flight_id and not flight_name:
            raise ValueError("Either flight_id or flight_name must be provided")

        logger.info(
            f"Loading single flight: flight_id={flight_id}, flight_name={flight_name}"
        )

        if self.campaign_service is None:
            raise RuntimeError("Campaign service not initialized")
        try:
            flight = self.campaign_service.get_flight(
                flight_name=flight_name, flight_id=flight_id
            )
            if flight:
                logger.info(f"Loaded flight: {flight.get('flight_name')}")
            return flight
        except Exception as e:
            logger.error(f"Error loading flight from database: {e}")
            raise

    def load_flights_by_date(
        self, start_date: str, end_date: str, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Load flights within a date range.

        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        campaign_id : Optional[str]
            Filter by campaign ID (optional)

        Returns
        -------
        list[dict[str, Any]]
            list of flight dictionaries matching the date range.
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
            end_dt = (
                datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            ).replace(tzinfo=UTC)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}") from e

        logger.info(f"Loading flights between {start_date} and {end_date}")

        if self.campaign_service is None:
            raise RuntimeError("Campaign service not initialized")
        try:
            # Get all flights and filter by date
            all_flights = self.campaign_service.get_all_flights()

            filtered_flights = []
            for flight in all_flights:
                takeoff = flight.get("takeoff_datetime")
                if takeoff is None:
                    continue
                if isinstance(takeoff, str):
                    takeoff = datetime.fromisoformat(takeoff.replace("Z", "+00:00"))

                if start_dt <= takeoff < end_dt:
                    if campaign_id is None or flight.get("campaign_id") == campaign_id:
                        filtered_flights.append(flight)

            logger.info(f"Loaded {len(filtered_flights)} flights in date range")
            return filtered_flights
        except Exception as e:
            logger.error(f"Error loading flights by date from database: {e}")
            raise

    def load_specific_data(
        self, flight_id: str, data_types: list[str] | None = None
    ) -> dict[str, list[str]]:
        """
        Load specific data types from a flight.

        Supported data types depend on the flight structure:
        - 'drone': Drone raw data (from drone folder)
        - 'aux': Auxiliary data (from aux folder)
        - 'proc': Processed data (from proc folder)
        - 'camera': Camera-specific data
        - 'gps': GPS-specific data
        - 'imu': IMU-specific data

        Parameters
        ----------
        flight_id : str
            Flight ID to load data from
        data_types : Optional[list[str]]
            list of data types to load. If None, loads all available.

        Returns
        -------
        dict[str, list[str]]
            dictionary mapping data_type to list of file paths.
        """
        if not flight_id:
            raise ValueError("flight_id is required")

        logger.info(f"Loading specific data for flight {flight_id}: {data_types}")

        # Get flight metadata first
        flight = self.load_single_flight(flight_id=flight_id)
        if not flight:
            raise ValueError(f"Flight {flight_id} not found")

        return self._collect_specific_data(flight, data_types)

    # ==================== Database Methods ====================

    def _load_flights_by_date_from_db(
        self, start_dt: datetime, end_dt: datetime, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Load flights by date range from stout database."""
        if self.campaign_service is None:
            raise RuntimeError("Campaign service not initialized")
        try:
            # Get all flights and filter by date
            all_flights = self.campaign_service.get_all_flights()

            filtered_flights = []
            for flight in all_flights:
                takeoff = flight.get("takeoff_datetime")
                if takeoff is None:
                    continue
                if isinstance(takeoff, str):
                    takeoff = datetime.fromisoformat(takeoff.replace("Z", "+00:00"))

                if start_dt <= takeoff < end_dt:
                    if campaign_id is None or flight.get("campaign_id") == campaign_id:
                        filtered_flights.append(flight)

            logger.info(f"Loaded {len(filtered_flights)} flights in date range")
            return filtered_flights
        except Exception as e:
            logger.error(f"Error loading flights by date from database: {e}")
            raise

    # ==================== Filesystem Methods ====================

    def _load_all_flights_from_filesystem(self) -> list[dict[str, Any]]:
        """Load all flights by scanning filesystem structure."""
        flights = []
        if self.base_data_path is None:
            logger.warning("Base data path not set")
            return flights
        campaigns_dir = Path(self.base_data_path) / "campaigns"

        if not campaigns_dir.exists():
            logger.warning(f"Campaigns directory not found: {campaigns_dir}")
            return flights

        # Traverse: campaigns -> date folders -> flight folders
        for campaign_path in campaigns_dir.iterdir():
            if not campaign_path.is_dir():
                continue
            campaign_name = campaign_path.name

            for date_path in campaign_path.iterdir():
                if not date_path.is_dir():
                    continue
                date_folder = date_path.name

                for flight_path in date_path.iterdir():
                    if not flight_path.is_dir():
                        continue
                    flight_name = flight_path.name

                    flight_dict = self._build_flight_dict_from_filesystem(
                        campaign_name, date_folder, flight_name, str(flight_path)
                    )
                    if flight_dict:
                        flights.append(flight_dict)

        logger.info(f"Loaded {len(flights)} flights from filesystem")
        return flights

    def _load_single_flight_from_filesystem(
        self, flight_id: str | None = None, flight_name: str | None = None
    ) -> dict[str, Any] | None:
        """Load single flight from filesystem."""
        all_flights = self._load_all_flights_from_filesystem()

        for flight in all_flights:
            if flight_id and flight.get("flight_id") == flight_id:
                return flight
            if flight_name and flight.get("flight_name") == flight_name:
                return flight

        return None

    def _load_flights_by_date_from_filesystem(
        self, start_dt: datetime, end_dt: datetime, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Load flights by date range from filesystem."""
        all_flights = self._load_all_flights_from_filesystem()

        filtered_flights = []
        for flight in all_flights:
            takeoff = flight.get("takeoff_datetime")
            if takeoff is None:
                continue
            if isinstance(takeoff, str):
                takeoff = datetime.fromisoformat(takeoff.replace("Z", "+00:00"))

            if start_dt <= takeoff < end_dt:
                if campaign_id is None or flight.get("campaign_id") == campaign_id:
                    filtered_flights.append(flight)

        return filtered_flights

    def _build_flight_dict_from_filesystem(
        self, campaign_name: str, date_folder: str, flight_name: str, flight_path: str
    ) -> dict[str, Any] | None:
        """Build flight dictionary from filesystem structure."""
        try:
            # Extract date from folder name (YYYYMMDD format)
            takeoff_date = datetime.strptime(date_folder, "%Y%m%d").replace(tzinfo=UTC)

            flight_path_obj = Path(flight_path)
            flight_dict = {
                "campaign_name": campaign_name,
                "flight_name": flight_name,
                "flight_date": date_folder,
                "takeoff_datetime": takeoff_date.isoformat(),
                "landing_datetime": takeoff_date.isoformat(),  # Not available from filesystem
                "drone_data_folder_path": str(flight_path_obj / "drone"),
                "aux_data_folder_path": str(flight_path_obj / "aux"),
                "processed_data_folder_path": str(flight_path_obj / "proc"),
            }
            return flight_dict
        except Exception as e:
            logger.warning(f"Could not build flight dict for {flight_name}: {e}")
            return None

    # ==================== Data Collection Methods ====================

    def _collect_specific_data(
        self, flight: dict[str, Any], data_types: list[str] | None = None
    ) -> dict[str, list[str]]:
        """
        Collect specific data types from a flight.

        Parameters
        ----------
        flight : dict[str, Any]
            Flight dictionary from database
        data_types : Optional[list[str]]
            list of data types to collect. If None, all available.

        Returns
        -------
        dict[str, list[str]]
            dictionary mapping data_type to list of file paths.
        """
        result = {}

        if data_types is None:
            data_types = ["drone", "aux", "proc", "camera", "gps", "imu"]

        for data_type in data_types:
            result[data_type] = self._get_data_files(flight, data_type)

        return result

    def _get_data_files(self, flight: dict[str, Any], data_type: str) -> list[str]:
        """
        Get list of files for a specific data type within a flight.

        Parameters
        ----------
        flight : dict[str, Any]
            Flight dictionary
        data_type : str
            Type of data to retrieve

        Returns
        -------
        list[str]
            list of file paths
        """
        files = []

        # Map data types to folder paths
        folder_map = {
            "drone": flight.get("drone_data_folder_path"),
            "aux": flight.get("aux_data_folder_path"),
            "proc": flight.get("processed_data_folder_path"),
        }

        # For specific sensor types, look in drone folder
        sensor_map = {
            "camera": ["*.jpg", "*.png", "*.tiff"],
            "gps": ["*.csv", "*.txt"],  # GPS data typically in CSV
            "imu": ["*.csv", "*.bin"],  # IMU data in CSV or binary
        }

        if data_type in folder_map:
            folder_path = folder_map[data_type]
            if folder_path and Path(folder_path).exists():
                files = self._list_files_recursive(folder_path)

        elif data_type in sensor_map:
            # Look in drone folder for sensor-specific data
            drone_folder = folder_map.get("drone")
            if drone_folder and Path(drone_folder).exists():
                sensor_folder = Path(drone_folder) / data_type
                if sensor_folder.exists():
                    files = self._list_files_recursive(str(sensor_folder))

        return files

    def _list_files_recursive(self, directory: str) -> list[str]:
        """
        Recursively list all files in a directory.

        Parameters
        ----------
        directory : str
            Directory path

        Returns
        -------
        list[str]
            list of absolute file paths
        """
        files = []
        try:
            dir_path = Path(directory)
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    files.append(str(file_path))
        except Exception as e:
            logger.warning(f"Error listing files in {directory}: {e}")

        return files

    # ==================== Utility Methods ====================

    def get_campaign_list(self) -> list[dict[str, Any]]:
        """
        Get list of all campaigns.

        Returns
        -------
        list[dict[str, Any]]
            list of campaign dictionaries with metadata.
        """
        if self.use_stout and self.campaign_service:
            try:
                return self.campaign_service.get_all_campaigns()
            except Exception as e:
                logger.error(f"Error loading campaigns from database: {e}")
                raise
        else:
            return self._get_campaigns_from_filesystem()

    def _get_campaigns_from_filesystem(self) -> list[dict[str, Any]]:
        """Get campaigns from filesystem."""
        campaigns = []
        if self.base_data_path is None:
            return campaigns
        campaigns_dir = Path(self.base_data_path) / "campaigns"

        if campaigns_dir.exists():
            for campaign_path in campaigns_dir.iterdir():
                if campaign_path.is_dir():
                    campaigns.append(
                        {
                            "name": campaign_path.name,
                            "path": str(campaign_path),
                        }
                    )

        return campaigns

    # ==================== DataFrame Loading Methods ====================

    def load_flight_data(
        self,
        flight_id: str | None = None,
        flight_name: str | None = None,
        sensors: list[str] | None = None,
        drones: list[str] | None = None,
        freq_interpolation: float | None = None,
        dji_drone_type: str | None = None,
        drone_correct_timestamp: bool | None = True,
        polars_interpolation: bool | None = True,
        align_drone: bool | None = True,
    ) -> dict[str, Any]:
        """
        Load flight data and return dataframes for requested sensors and drones.

        This is the main method to load flight data. It returns a dictionary
        containing flight metadata and dataframes for the requested data types.

        Supported sensors:
        - 'gps': GPS data from UBX/BIN file
        - 'imu': IMU data
        - 'adc': ADC sensor data
        - 'camera': Camera data
        - 'inclinometer': Inclinometer data

        Supported drones:
        - 'dji': DJI drone telemetry from DAT file
        - 'litchi': Litchi flight logs
        - 'blacksquare': BlackSquare drone logs

        Parameters
        ----------
        flight_id : Optional[str]
            Flight ID to load
        flight_name : Optional[str]
            Flight name to load (alternative to flight_id)
        sensors : Optional[list[str]]
            list of sensor types to load. If None, loads ['gps'].
        drones : Optional[list[str]]
            list of drone types to load. If None, loads ['dji'].

        Returns
        -------
        dict[str, Any]
            dictionary containing:
                - 'flight_info': Flight metadata dictionary
                - '<sensor_type>': DataFrame for each requested sensor
                - '<drone_type>': DataFrame for each requested drone

        Examples
        --------
        >>> loader = StoutDataLoader()
        >>> data = loader.load_flight_data(
        ...     flight_id='some-id',
        ...     sensors=['gps', 'imu'],
        ...     drones=['dji']
        ... )
        >>> gps_df = data['gps']
        >>> drone_df = data['dji']
        """
        if sensors is None:
            sensors = ["gps"]
        if drones is None:
            drones = ["dji"]

        # Load flight metadata
        flight_info = self.load_single_flight(
            flight_id=flight_id, flight_name=flight_name
        )
        if not flight_info:
            raise ValueError(
                f"Flight not found: flight_id={flight_id}, flight_name={flight_name}"
            )

        result: dict[str, Any] = {"flight_info": flight_info}

        # Load requested sensors
        for sensor_type in sensors:
            if sensor_type not in SENSOR_MAP:
                logger.warning(
                    f"Unknown sensor type: {sensor_type}. Available: {list(SENSOR_MAP.keys())}"
                )
                continue
            df = self._load_sensor_dataframe(
                flight_info, sensor_type, freq_interpolation
            )
            result[sensor_type] = df
            logger.info(
                f"Loaded {sensor_type} data: {df.shape if df is not None else 'None'}"
            )

        # Load requested drones
        for drone_type in drones:
            if drone_type not in DRONE_MAP:
                logger.warning(
                    f"Unknown drone type: {drone_type}. Available: {list(DRONE_MAP.keys())}"
                )
                continue
            if drone_type == "dji" and dji_drone_type is not None:
                dji_drone_type = dji_drone_type
            df = self._load_drone_dataframe(
                flight_info,
                drone_type,
                dji_drone_type,
                drone_correct_timestamp,
                polars_interpolation,
                align_drone,
            )
            result[drone_type] = df
            logger.info(
                f"Loaded {drone_type} data: {'OK' if df is not None else 'None'}"
            )

        return result

    def _load_sensor_dataframe(
        self,
        flight_info: dict[str, Any],
        sensor_type: str,
        freq_interpolation: float | None = None,
    ) -> Any | None:
        """
        Load sensor data and return as polars DataFrame.

        Parameters
        ----------
        flight_info : dict[str, Any]
            Flight metadata dictionary
        sensor_type : str
            Type of sensor to load (must be in SENSOR_MAP)

        Returns
        -------
        Optional[Any]
            polars.DataFrame with sensor data, or None if not found
        """
        config = SENSOR_MAP[sensor_type]

        # Sensors are loaded from aux_data_folder_path
        folder_path = flight_info.get("aux_data_folder_path")

        logger.info(f"Loading {sensor_type} data from: {folder_path}")
        if not folder_path or not Path(folder_path).exists():
            logger.warning(f"{sensor_type}: Aux folder not found: {folder_path}")
            return None

        # Find data file/directory
        data_path = None
        is_directory = config.get("is_directory", False)

        for pattern in config["patterns"]:
            if is_directory:
                # Look for directory
                dirs = list(Path(folder_path).glob(pattern.rstrip("/")))
                if dirs:
                    data_path = str(dirs[0])
                    break
            else:
                # Look for file
                files = list(Path(folder_path).glob(pattern))
                if files:
                    data_path = str(files[0])
                    break

        if not data_path:
            logger.warning(
                f"{sensor_type}: No data found in {folder_path} with patterns {config['patterns']}"
            )
            return None

        logger.info(f"Loading {sensor_type} data from: {data_path}")

        # Dynamic import and instantiation
        module = importlib.import_module(config["module"], package="pils")
        sensor_class = getattr(module, config["class"])

        # Instantiate and load data
        sensor = sensor_class(data_path)
        sensor.load_data(freq_interpolation=freq_interpolation)
        return sensor.data

    def _load_drone_dataframe(
        self,
        flight_info: dict[str, Any],
        drone_type: str = "dji",
        dji_drone_type: str | None = None,
        drone_correct_timestamp: bool | None = True,
        polars_interpolation: bool | None = True,
        align_drone: bool | None = True,
    ) -> Any | None:
        """
        Load drone telemetry and return as polars DataFrame.

        Parameters
        ----------
        flight_info : dict[str, Any]
            Flight metadata dictionary
        drone_type : str
            Type of drone to load (must be in DRONE_MAP)

        Returns
        -------
        Optional[Any]
            polars.DataFrame with drone telemetry, or None if not found
        """
        config = DRONE_MAP[drone_type]

        # Drones are loaded from drone_data_folder_path
        folder_path = flight_info.get("drone_data_folder_path")
        if not folder_path or not Path(folder_path).exists():
            logger.warning(f"{drone_type}: Drone folder not found: {folder_path}")
            return None

        # Find data file
        data_path = None
        for pattern in config["patterns"]:
            files = list(Path(folder_path).glob(pattern))
            if files:
                data_path = str(files[0])
                break

        if not data_path:
            logger.warning(
                f"{drone_type}: No data found in {folder_path} with patterns {config['patterns']}"
            )
            return None

        logger.info(f"Loading {drone_type} data from: {data_path}")

        # Dynamic import and instantiation
        module = importlib.import_module(config["module"], package="pils")
        drone_class = getattr(module, config["class"])

        # Instantiate and load data
        drone = drone_class(data_path, source_format=dji_drone_type)
        drone.load_data(
            correct_timestamp=drone_correct_timestamp,
            polars_interpolation=polars_interpolation,
            align=align_drone,
        )

        return drone.data

    def get_available_sensors(self) -> list[str]:
        """Return list of available sensor types."""
        return list(SENSOR_MAP.keys())

    def get_available_drones(self) -> list[str]:
        """Return list of available drone types."""
        return list(DRONE_MAP.keys())
