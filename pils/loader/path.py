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

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


class PathLoader:
    """
    Data loader for STOUT campaign management system.

    Provides methods to load flight data paths and associated metadata
    from the STOUT database and file system.

    Attributes
    ----------
    campaign_service : Service
        Service for accessing campaign and flight data
    base_data_path : Optional[Path]
        Base path where all campaign data is stored
    """

    def __init__(self, base_data_path: str | Path | None):
        """
        Initialize the PathLoader.

        Parameters
        ----------
        base_data_path : Union[str, Path, None]
            Base path for data storage. Accepts string or Path object.
        """
        self.base_data_path: Path | None = (
            Path(base_data_path) if base_data_path is not None else None
        )

    def load_all_flights(self) -> list[dict[str, Any]]:
        """
        Load all flights from all campaigns.

        Returns
        -------
        List[Dict[str, Any]]
            List of flight dictionaries containing flight metadata and paths.
            Each flight dict includes: flight_id, flight_name, campaign_id,
            takeoff_datetime, landing_datetime, and folder paths.
        """
        logger.info("Loading all flights from all campaigns...")

        flights: list[dict[str, Any]] = []
        if self.base_data_path is None:
            logger.warning("Base data path not set")
            return flights
        campaigns_dir = self.base_data_path / "campaigns"

        if not campaigns_dir.exists():
            logger.warning(f"Campaigns directory not found: {campaigns_dir}")
            return flights

        logger.debug(f"Scanning campaigns directory: {campaigns_dir}")

        # Traverse: campaigns -> date folders -> flight folders
        for campaign_path in campaigns_dir.iterdir():
            if not campaign_path.is_dir():
                continue

            campaign_name = campaign_path.name
            logger.debug(f"Processing campaign: {campaign_name}")

            if campaign_name == "telescope_data":
                logger.debug("Skip Telescope Data")
                continue

            for date_path in campaign_path.iterdir():
                if not date_path.is_dir():
                    continue

                for flight_path in date_path.iterdir():
                    flight_name = flight_path.name
                    if flight_name in ["base", "calibration"]:
                        continue
                    if not flight_path.is_dir():
                        continue

                    flight_dict = self._build_flight_dict_from_filesystem(
                        campaign_name, date_path.name, flight_name, flight_path
                    )
                    if flight_dict:
                        flights.append(flight_dict)

        logger.info(f"Loaded {len(flights)} flights from filesystem")
        return flights

    def load_all_campaign_flights(
        self, campaign_name: str | None = None, campaign_id=None
    ) -> dict[str, Any] | None:
        """
        Load data for a single flight.

        Parameters
        ----------
        campaign_name : Optional[str]
            Flight name to load (alternative to flight_id)

        Returns
        -------
        Optional[Dict[str, Any]]
            Flight dictionary with metadata and paths, or None if not found.
        """
        if campaign_name:
            raise ValueError("Either flight_id or flight_name must be provided")

        logger.info(f"Loading all campaign flighs: campaign_name={campaign_name}")

        all_flights = self.load_all_flights()

        for flight in all_flights:
            if flight.get("campaign_name") == campaign_name:
                return flight

        return None

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
        Optional[Dict[str, Any]]
            Flight dictionary with metadata and paths, or None if not found.
        """
        if not flight_id and not flight_name:
            raise ValueError("Either flight_id or flight_name must be provided")

        logger.info(
            f"Loading single flight: flight_id={flight_id}, flight_name={flight_name}"
        )

        all_flights = self.load_all_flights()

        for flight in all_flights:
            if flight_id and flight.get("flight_id") == flight_id:
                return flight
            if flight_name and flight.get("flight_name") == flight_name:
                return flight

        return None

    def _build_flight_dict_from_filesystem(
        self, campaign_name: str, date_folder: str, flight_name: str, flight_path: Path
    ) -> dict[str, Any] | None:
        """Build flight dictionary from filesystem structure."""
        try:
            # Extract date from folder name (YYYYMMDD format)
            takeoff_date = datetime.strptime(flight_name[7:], "%Y%m%d_%H%M").replace(
                tzinfo=UTC
            )

            flight_dict = {
                "campaign_name": campaign_name,
                "flight_name": flight_name,
                "flight_date": date_folder,
                "takeoff_datetime": takeoff_date.isoformat(),
                "landing_datetime": takeoff_date.isoformat(),  # Not available from filesystem
                "drone_data_folder_path": str(flight_path / "drone"),
                "aux_data_folder_path": str(flight_path / "aux"),
                "processed_data_folder_path": str(flight_path / "proc"),
            }
            return flight_dict
        except Exception as e:
            logger.warning(
                f"Could not build flight dict for {flight_name} {flight_path}: {e}"
            )
            return None
