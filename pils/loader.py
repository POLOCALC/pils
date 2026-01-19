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

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StoutDataLoader:
    """
    Data loader for STOUT campaign management system.

    Provides methods to load flight data paths and associated metadata
    from the STOUT database and file system.

    Attributes:
        campaign_service: Service for accessing campaign and flight data
        base_data_path: Base path where all campaign data is stored
    """

    def __init__(self, use_stout: bool = True, base_data_path: Optional[str] = None):
        """
        Initialize the StoutDataLoader.

        Args:
            use_stout: If True, uses stout services to query database.
                      If False, queries filesystem directly.
            base_data_path: Base path for data storage. If None, uses stout config.
        """
        self.use_stout = use_stout
        self.campaign_service = None
        self.base_data_path = base_data_path

        if use_stout:
            try:
                from stout.services.campaigns import CampaignService  # type: ignore
                from stout.config import Config  # type: ignore

                self.campaign_service = CampaignService()
                self.base_data_path = self.base_data_path or Config.MAIN_DATA_PATH
                logger.info(
                    f"Initialized with stout database, base path: {self.base_data_path}"
                )
            except ImportError as e:
                logger.warning(
                    f"Could not import stout: {e}. Falling back to filesystem queries."
                )
                self.use_stout = False

        if not self.use_stout and not base_data_path:
            raise ValueError("base_data_path is required when use_stout=False")

    def load_all_campaign_flights(self) -> List[Dict[str, Any]]:
        """
        Load all flights from all campaigns.

        Returns:
            List of flight dictionaries containing flight metadata and paths.
            Each flight dict includes: flight_id, flight_name, campaign_id,
            takeoff_datetime, landing_datetime, and folder paths.
        """
        logger.info("Loading all flights from all campaigns...")

        if self.use_stout and self.campaign_service:
            return self._load_all_flights_from_db()
        else:
            return self._load_all_flights_from_filesystem()

    def load_single_flight(
        self, flight_id: Optional[str] = None, flight_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load data for a single flight.

        Args:
            flight_id: Flight ID to load
            flight_name: Flight name to load (alternative to flight_id)

        Returns:
            Flight dictionary with metadata and paths, or None if not found.
        """
        if not flight_id and not flight_name:
            raise ValueError("Either flight_id or flight_name must be provided")

        logger.info(
            f"Loading single flight: flight_id={flight_id}, flight_name={flight_name}"
        )

        if self.use_stout and self.campaign_service:
            return self._load_single_flight_from_db(flight_id, flight_name)
        else:
            return self._load_single_flight_from_filesystem(flight_id, flight_name)

    def load_flights_by_date(
        self, start_date: str, end_date: str, campaign_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load flights within a date range.

        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            campaign_id: Filter by campaign ID (optional)

        Returns:
            List of flight dictionaries matching the date range.
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            end_dt = (
                datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            ).replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")

        logger.info(f"Loading flights between {start_date} and {end_date}")

        if self.use_stout and self.campaign_service:
            return self._load_flights_by_date_from_db(start_dt, end_dt, campaign_id)
        else:
            return self._load_flights_by_date_from_filesystem(
                start_dt, end_dt, campaign_id
            )

    def load_specific_data(
        self, flight_id: str, data_types: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Load specific data types from a flight.

        Supported data types depend on the flight structure:
        - 'drone': Drone raw data (from drone folder)
        - 'aux': Auxiliary data (from aux folder)
        - 'proc': Processed data (from proc folder)
        - 'camera': Camera-specific data
        - 'gps': GPS-specific data
        - 'imu': IMU-specific data

        Args:
            flight_id: Flight ID to load data from
            data_types: List of data types to load. If None, loads all available.

        Returns:
            Dictionary mapping data_type to list of file paths.
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

    def _load_all_flights_from_db(self) -> List[Dict[str, Any]]:
        """Load all flights from stout database."""
        if self.campaign_service is None:
            raise RuntimeError("Campaign service not initialized")
        try:
            flights = self.campaign_service.get_all_flights()
            logger.info(f"Loaded {len(flights)} flights from database")
            return flights
        except Exception as e:
            logger.error(f"Error loading flights from database: {e}")
            raise

    def _load_single_flight_from_db(
        self, flight_id: Optional[str] = None, flight_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load single flight from stout database."""
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

    def _load_flights_by_date_from_db(
        self, start_dt: datetime, end_dt: datetime, campaign_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
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

    def _load_all_flights_from_filesystem(self) -> List[Dict[str, Any]]:
        """Load all flights by scanning filesystem structure."""
        flights = []
        if self.base_data_path is None:
            logger.warning("Base data path not set")
            return flights
        campaigns_dir = os.path.join(self.base_data_path, "campaigns")

        if not os.path.exists(campaigns_dir):
            logger.warning(f"Campaigns directory not found: {campaigns_dir}")
            return flights

        # Traverse: campaigns -> date folders -> flight folders
        for campaign_name in os.listdir(campaigns_dir):
            campaign_path = os.path.join(campaigns_dir, campaign_name)
            if not os.path.isdir(campaign_path):
                continue

            for date_folder in os.listdir(campaign_path):
                date_path = os.path.join(campaign_path, date_folder)
                if not os.path.isdir(date_path):
                    continue

                for flight_name in os.listdir(date_path):
                    flight_path = os.path.join(date_path, flight_name)
                    if not os.path.isdir(flight_path):
                        continue

                    flight_dict = self._build_flight_dict_from_filesystem(
                        campaign_name, date_folder, flight_name, flight_path
                    )
                    if flight_dict:
                        flights.append(flight_dict)

        logger.info(f"Loaded {len(flights)} flights from filesystem")
        return flights

    def _load_single_flight_from_filesystem(
        self, flight_id: Optional[str] = None, flight_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load single flight from filesystem."""
        all_flights = self._load_all_flights_from_filesystem()

        for flight in all_flights:
            if flight_id and flight.get("flight_id") == flight_id:
                return flight
            if flight_name and flight.get("flight_name") == flight_name:
                return flight

        return None

    def _load_flights_by_date_from_filesystem(
        self, start_dt: datetime, end_dt: datetime, campaign_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
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
    ) -> Optional[Dict[str, Any]]:
        """Build flight dictionary from filesystem structure."""
        try:
            # Extract date from folder name (YYYYMMDD format)
            takeoff_date = datetime.strptime(date_folder, "%Y%m%d").replace(
                tzinfo=timezone.utc
            )

            flight_dict = {
                "campaign_name": campaign_name,
                "flight_name": flight_name,
                "flight_date": date_folder,
                "takeoff_datetime": takeoff_date.isoformat(),
                "landing_datetime": takeoff_date.isoformat(),  # Not available from filesystem
                "drone_data_folder_path": os.path.join(flight_path, "drone"),
                "aux_data_folder_path": os.path.join(flight_path, "aux"),
                "processed_data_folder_path": os.path.join(flight_path, "proc"),
            }
            return flight_dict
        except Exception as e:
            logger.warning(f"Could not build flight dict for {flight_name}: {e}")
            return None

    # ==================== Data Collection Methods ====================

    def _collect_specific_data(
        self, flight: Dict[str, Any], data_types: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Collect specific data types from a flight.

        Args:
            flight: Flight dictionary from database
            data_types: List of data types to collect. If None, all available.

        Returns:
            Dictionary mapping data_type to list of file paths.
        """
        result = {}

        if data_types is None:
            data_types = ["drone", "aux", "proc", "camera", "gps", "imu"]

        for data_type in data_types:
            result[data_type] = self._get_data_files(flight, data_type)

        return result

    def _get_data_files(self, flight: Dict[str, Any], data_type: str) -> List[str]:
        """
        Get list of files for a specific data type within a flight.

        Args:
            flight: Flight dictionary
            data_type: Type of data to retrieve

        Returns:
            List of file paths
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
            if folder_path and os.path.exists(folder_path):
                files = self._list_files_recursive(folder_path)

        elif data_type in sensor_map:
            # Look in drone folder for sensor-specific data
            drone_folder = folder_map.get("drone")
            if drone_folder and os.path.exists(drone_folder):
                sensor_folder = os.path.join(drone_folder, data_type)
                if os.path.exists(sensor_folder):
                    files = self._list_files_recursive(sensor_folder)

        return files

    def _list_files_recursive(self, directory: str) -> List[str]:
        """
        Recursively list all files in a directory.

        Args:
            directory: Directory path

        Returns:
            List of absolute file paths
        """
        files = []
        try:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        except Exception as e:
            logger.warning(f"Error listing files in {directory}: {e}")

        return files

    # ==================== Utility Methods ====================

    def get_campaign_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all campaigns.

        Returns:
            List of campaign dictionaries with metadata.
        """
        if self.use_stout and self.campaign_service:
            try:
                return self.campaign_service.get_all_campaigns()
            except Exception as e:
                logger.error(f"Error loading campaigns from database: {e}")
                raise
        else:
            return self._get_campaigns_from_filesystem()

    def _get_campaigns_from_filesystem(self) -> List[Dict[str, Any]]:
        """Get campaigns from filesystem."""
        campaigns = []
        if self.base_data_path is None:
            return campaigns
        campaigns_dir = os.path.join(self.base_data_path, "campaigns")

        if os.path.exists(campaigns_dir):
            for campaign_name in os.listdir(campaigns_dir):
                campaign_path = os.path.join(campaigns_dir, campaign_name)
                if os.path.isdir(campaign_path):
                    campaigns.append(
                        {
                            "name": campaign_name,
                            "path": campaign_path,
                        }
                    )

        return campaigns
