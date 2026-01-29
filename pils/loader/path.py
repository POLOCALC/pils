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
import importlib

from pils.config import SENSOR_MAP, DRONE_MAP

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PathLoader:
    """
    Data loader for STOUT campaign management system.

    Provides methods to load flight data paths and associated metadata
    from the STOUT database and file system.

    Attributes:
        campaign_service: Service for accessing campaign and flight data
        base_data_path: Base path where all campaign data is stored
    """

    def __init__(self, base_data_path):
        """
        Initialize the StoutDataLoader.

        Args:
            use_stout: If True, uses stout services to query database.
                      If False, queries filesystem directly.
            base_data_path: Base path for data storage. If None, uses stout config.
        """

        self.campaign_service = None
        self.base_data_path = base_data_path

    def load_all_campaign_flights(self) -> List[Dict[str, Any]]:
        """
        Load all flights from all campaigns.

        Returns:
            List of flight dictionaries containing flight metadata and paths.
            Each flight dict includes: flight_id, flight_name, campaign_id,
            takeoff_datetime, landing_datetime, and folder paths.
        """
        logger.info("Loading all flights from all campaigns...")

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

        all_flights = self.load_all_campaign_flights()

        for flight in all_flights:
            if flight_id and flight.get("flight_id") == flight_id:
                return flight
            if flight_name and flight.get("flight_name") == flight_name:
                return flight

        return None

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
