"""Find flights (their metadata and raw data folders) in the STOUT database.

Examples
--------
>>> from pils.loader import StoutLoader
>>> loader = StoutLoader()
>>> flight = loader.load_single_flight(flight_name="flight_20251208_1506")
>>> flights = loader.load_flights_by_date("2025-01-01", "2025-01-15")
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pils.integrations.stout as stout_db

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StoutLoader:
    """
    Find flights in the STOUT database.

    Attributes
    ----------
    use_stout : bool
        Whether STOUT is installed (else every query raises RuntimeError)
    base_data_path : Optional[Path]
        Base path where all campaign data is stored
    """

    def __init__(self):
        self.base_data_path = None
        self.use_stout = stout_db.is_available()

        if self.use_stout:
            self.base_data_path = stout_db.data_path()
            logger.info(
                f"Initialized with stout database, base path: {self.base_data_path}"
            )
        else:
            logger.warning("STOUT is not installed: database queries will fail.")

    def _require_stout(self) -> None:
        if not self.use_stout:
            raise RuntimeError("Campaign service not initialized")

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

        self._require_stout()
        try:
            flights = stout_db.all_flights()
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
        list[dict[str, Any]]
            The campaign's flight dictionaries.
        """
        if not campaign_id and not campaign_name:
            raise ValueError("Either flight_id or flight_name must be provided")

        logger.info(
            f"Loading single flight: flight_id={campaign_id}, flight_name={campaign_name}"
        )

        self._require_stout()
        try:
            flights = stout_db.campaign_flights(campaign_id or campaign_name)
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

        self._require_stout()
        try:
            flight = stout_db.flight(flight_id=flight_id, flight_name=flight_name)
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
            End date in format 'YYYY-MM-DD' (included)
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

        self._require_stout()
        try:
            filtered_flights = []
            for flight in stout_db.all_flights():
                takeoff = flight.get("takeoff_datetime")
                if takeoff is None:
                    continue
                if isinstance(takeoff, str):
                    takeoff = datetime.fromisoformat(takeoff.replace("Z", "+00:00"))
                if takeoff.tzinfo is None:
                    takeoff = takeoff.replace(tzinfo=UTC)

                if start_dt <= takeoff < end_dt and (
                    campaign_id is None or flight.get("campaign_id") == campaign_id
                ):
                    filtered_flights.append(flight)

            logger.info(f"Loaded {len(filtered_flights)} flights in date range")
            return filtered_flights
        except Exception as e:
            logger.error(f"Error loading flights by date from database: {e}")
            raise
