"""
FlightDataHandler - High-level unified interface for loading and accessing flight data.

This is the main user-facing API that combines:
- StoutDataLoader for database queries
- DataDecoder for automatic data parsing
- Convenient data access patterns

Recommended Usage:
    from pils import FlightDataHandler

    # Initialize with STOUT database
    handler = FlightDataHandler(use_stout=True)

    # Load and decode a flight
    flight = handler.load_flight(flight_id='flight-123')

    # Access decoded data
    gps_df = flight.payload.gps.data
    drone_df = flight.drone.data
    imu_data = flight.payload.imu

    # Load multiple flights by date
    flights = handler.load_campaign_flights(
        campaign_id='camp-123',
        date_range=('2025-01-01', '2025-01-31')
    )
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from .loader import StoutDataLoader
from .decoder import DataDecoder

logger = logging.getLogger(__name__)


class Flight:
    """
    Represents a single flight with all its decoded data.

    Attributes:
        metadata: Flight metadata (ID, name, timestamps, paths)
        drone: Decoded drone flight log data
        payload: Decoded payload sensor data
        litchi: Litchi flight plan/telemetry if available
        raw_files: Catalog of raw data files
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        drone: Optional[Any] = None,
        payload: Optional[Any] = None,
        litchi: Optional[Any] = None,
        raw_files: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize Flight object."""
        self.metadata = metadata
        self.drone = drone
        self.payload = payload
        self.litchi = litchi
        self.raw_files = raw_files or {}

    @property
    def flight_id(self) -> str:
        """Get flight ID."""
        return self.metadata.get("flight_id", "unknown")

    @property
    def flight_name(self) -> str:
        """Get flight name."""
        return self.metadata.get("flight_name", "unknown")

    @property
    def flight_path(self) -> str:
        """Get flight directory path."""
        # Construct from standard structure
        paths = self.metadata
        drone_path = paths.get("drone_data_folder_path", "")
        if drone_path:
            return drone_path.rsplit("/drone", 1)[0]
        return ""

    @property
    def campaign_id(self) -> str:
        """Get campaign ID."""
        return self.metadata.get("campaign_id", "unknown")

    @property
    def takeoff_datetime(self) -> Optional[datetime]:
        """Get takeoff datetime."""
        dt_str = self.metadata.get("takeoff_datetime")
        if isinstance(dt_str, str):
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt_str

    def __repr__(self) -> str:
        """String representation."""
        return f"Flight({self.flight_name}, id={self.flight_id})"


class FlightDataHandler:
    """
    High-level interface for loading and accessing flight data.

    Combines StoutDataLoader (database queries) and DataDecoder
    (automatic data parsing) for seamless data access.

    This is the recommended entry point for users.
    """

    def __init__(
        self,
        use_stout: bool = True,
        base_data_path: Optional[str] = None,
        auto_decode: bool = True,
        adc_gain_config: Optional[int] = None,
    ):
        """
        Initialize FlightDataHandler.

        Args:
            use_stout: Use STOUT database for queries
            base_data_path: Base path for data (required if use_stout=False)
            auto_decode: Automatically decode data when loading flights
            adc_gain_config: ADC gain configuration (1, 2, 4, 8, or 16).
                             If None, auto-detects from config.yml in each flight's aux folder.
        """
        self.loader = StoutDataLoader(
            use_stout=use_stout, base_data_path=base_data_path
        )
        self.auto_decode = auto_decode
        self.adc_gain_config = adc_gain_config

        logger.info(f"Initialized FlightDataHandler (auto_decode={auto_decode})")

    def load_flight(
        self,
        flight_id: Optional[str] = None,
        flight_name: Optional[str] = None,
        decode: Optional[bool] = None,
        drone_model: Optional[str] = None,
    ) -> Optional[Flight]:
        """
        Load a single flight with optional data decoding.

        Args:
            flight_id: Flight ID
            flight_name: Flight name
            decode: Override auto_decode setting
            drone_model: Drone model hint for decoder

        Returns:
            Flight object or None if not found
        """
        # Query metadata from database
        flight_meta = self.loader.load_single_flight(
            flight_id=flight_id, flight_name=flight_name
        )

        if not flight_meta:
            logger.warning(f"Flight not found: id={flight_id}, name={flight_name}")
            return None

        # Create Flight object
        flight = Flight(metadata=flight_meta)

        # Optionally decode data
        should_decode = decode if decode is not None else self.auto_decode
        if should_decode:
            self._decode_flight(flight, drone_model)

        logger.info(f"Loaded flight: {flight}")
        return flight

    def load_campaign_flights(
        self,
        campaign_id: str,
        date_range: Optional[Tuple[str, str]] = None,
        decode: Optional[bool] = None,
        drone_model: Optional[str] = None,
    ) -> List[Flight]:
        """
        Load all flights from a campaign.

        Args:
            campaign_id: Campaign ID
            date_range: Optional (start_date, end_date) tuple
            decode: Override auto_decode setting
            drone_model: Drone model hint for decoder

        Returns:
            List of Flight objects
        """
        logger.info(f"Loading campaign {campaign_id}")

        # Get all flights
        if date_range:
            start_date, end_date = date_range
            flight_metas = self.loader.load_flights_by_date(
                start_date=start_date, end_date=end_date, campaign_id=campaign_id
            )
        else:
            # Get all flights and filter by campaign
            all_flights = self.loader.load_all_campaign_flights()
            flight_metas = [
                f for f in all_flights if f.get("campaign_id") == campaign_id
            ]

        # Create Flight objects
        flights = []
        should_decode = decode if decode is not None else self.auto_decode

        for meta in flight_metas:
            flight = Flight(metadata=meta)
            if should_decode:
                try:
                    self._decode_flight(flight, drone_model)
                except Exception as e:
                    logger.warning(f"Could not decode {flight.flight_name}: {e}")
            flights.append(flight)

        logger.info(f"Loaded {len(flights)} flights from campaign")
        return flights

    def load_flights_by_date(
        self,
        start_date: str,
        end_date: str,
        campaign_id: Optional[str] = None,
        decode: Optional[bool] = None,
        drone_model: Optional[str] = None,
    ) -> List[Flight]:
        """
        Load flights within a date range.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            campaign_id: Optional campaign filter
            decode: Override auto_decode setting
            drone_model: Drone model hint for decoder

        Returns:
            List of Flight objects
        """
        logger.info(f"Loading flights from {start_date} to {end_date}")

        flight_metas = self.loader.load_flights_by_date(
            start_date=start_date, end_date=end_date, campaign_id=campaign_id
        )

        # Create Flight objects
        flights = []
        should_decode = decode if decode is not None else self.auto_decode

        for meta in flight_metas:
            flight = Flight(metadata=meta)
            if should_decode:
                try:
                    self._decode_flight(flight, drone_model)
                except Exception as e:
                    logger.warning(f"Could not decode {flight.flight_name}: {e}")
            flights.append(flight)

        logger.info(f"Loaded {len(flights)} flights in date range")
        return flights

    def get_campaigns(self) -> List[Dict[str, Any]]:
        """
        Get list of all campaigns.

        Returns:
            List of campaign dictionaries
        """
        return self.loader.get_campaign_list()

    # ==================== Private Methods ====================

    def _decode_flight(self, flight: Flight, drone_model: Optional[str] = None) -> None:
        """
        Decode flight data and attach to Flight object.

        Args:
            flight: Flight object to decode
            drone_model: Drone model hint
        """
        flight_path = flight.flight_path

        if not flight_path:
            logger.warning(f"Cannot decode {flight.flight_name}: no path available")
            return

        try:
            decoder = DataDecoder(
                flight_path,
                drone_model=drone_model,
                adc_gain_config=self.adc_gain_config,
            )

            # Load drone data
            try:
                decoder.load_drone_data()
                flight.drone = decoder.drone
            except Exception as e:
                logger.debug(f"Could not load drone data: {e}")

            # Load payload data
            try:
                decoder.load_payload_data()
                flight.payload = decoder.payload
            except Exception as e:
                logger.debug(f"Could not load payload data: {e}")

            # Load litchi data
            try:
                decoder.load_litchi_data()
                flight.litchi = decoder.litchi
            except Exception as e:
                logger.debug(f"No Litchi data: {e}")

            # Catalog raw files
            flight.raw_files = decoder._catalog_raw_files()

            logger.info(f"Decoded flight {flight.flight_name}")

        except Exception as e:
            logger.error(f"Error decoding flight {flight.flight_name}: {e}")
            raise
